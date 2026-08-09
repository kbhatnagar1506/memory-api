"""Multi-benchmark runner.

    python -m bench.run --benchmark longmemeval --corpora 30
    python -m bench.run --benchmark locomo --corpora 10
    python -m bench.run --benchmark all

Reports per capability. There is deliberately no blended score: a system can
be excellent at recall and dangerous at abstention, and one number hides that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mapi.config import EmbeddingBackend, RerankBackend, Settings, StoreBackend
from mapi.core.logging import configure_logging
from mapi.domain.embeddings import build_embedder
from mapi.domain.retrieval.rerank import HeuristicReranker, NoopReranker

from .cache import DiskVectorCache
from .datasets.base import Dataset
from .datasets.locomo import LoCoMo
from .datasets.longmemeval import LongMemEval
from .harness import evaluate_end_to_end, evaluate_retrieval, ingest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"

DATASETS: dict[str, tuple[type[Dataset], str]] = {
    "locomo": (LoCoMo, "bench/data/locomo10.json"),
    "longmemeval": (LongMemEval, "bench/data/longmemeval_s.json"),
}

#: Retrieval configurations. Entity bridging is included so its contribution is
#: measured per benchmark rather than assumed — it is inert on LoCoMo (0.10
#: entities per turn, 91% of turns have none) and may not be on LongMemEval
#: (50 per session).
CONFIGS: dict[str, dict[str, Any]] = {
    "vector_only": {
        "vector_weight": 1.0,
        "lexical_weight": 0.0,
        "use_rerank": False,
        "use_mmr": False,
        "use_decay": False,
    },
    "lexical_only": {
        "vector_weight": 0.0,
        "lexical_weight": 1.0,
        "use_rerank": False,
        "use_mmr": False,
        "use_decay": False,
    },
    "hybrid_rrf": {"use_rerank": False, "use_mmr": False, "use_decay": False},
    "hybrid_rerank": {"use_rerank": True, "use_mmr": False, "use_decay": False},
    "hybrid_rerank_entity": {
        "use_rerank": True,
        "use_mmr": False,
        "use_decay": False,
        "use_entity_expansion": True,
    },
}


async def run_one(name: str, args: argparse.Namespace, out: Path) -> dict[str, Any] | None:
    cls, default_path = DATASETS[name]
    path = REPO / default_path
    if not path.exists():
        print(f"  {name}: data not found at {path} -- skipping")
        return None

    dataset = cls(path)
    print(f"\n=== {name} ({dataset.evidence_granularity}-level evidence) ===")
    corpora = dataset.load(limit_corpora=args.corpora)
    documents = sum(len(c.documents) for c in corpora)
    questions = sum(len(c.questions) for c in corpora)
    print(f"  {len(corpora)} corpora, {documents} documents, {questions} questions")

    settings = Settings(
        store_backend=StoreBackend.MEMORY,
        embedding_backend=EmbeddingBackend(args.embeddings),
        embedding_dimensions=args.dimensions,
        embedding_model="text-embedding-004",
        embedding_batch_size=args.batch_size,
        embedding_cache_size=20000,
        rerank_backend=RerankBackend.HEURISTIC,
    )
    embedder = build_embedder(settings)
    print(f"  embedder: {embedder.name}/{embedder.model}  ingesting ...", flush=True)

    started = time.perf_counter()
    cache = DiskVectorCache(REPO / "bench" / "data" / "cache" / "vectors.sqlite")
    ingested = await ingest(
        corpora,
        embedder,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        cache=cache,
    )
    print(f"  embedding cache: {cache.stats()}", flush=True)
    print(
        f"  ingested {ingested.documents} docs / {ingested.chunks} chunks in "
        f"{time.perf_counter() - started:.0f}s ({ingested.embed_seconds:.0f}s embedding)",
        flush=True,
    )

    # Warm the query cache before timing so latency compares configs, not order.
    embed_query = getattr(embedder, "embed_query", None)
    all_questions = [q for c in corpora for q in c.questions]
    warm = asyncio.Semaphore(args.concurrency)

    async def warm_one(text: str) -> None:
        async with warm:
            try:
                if embed_query is not None:
                    await embed_query(text)
                else:
                    await embedder.embed_one(text)
            except Exception:
                pass

    await asyncio.gather(*(warm_one(q.text) for q in all_questions))

    results = []
    for config_name, config in CONFIGS.items():
        reranker = HeuristicReranker() if config.get("use_rerank") else NoopReranker()
        started = time.perf_counter()
        summary = await evaluate_retrieval(
            corpora,
            ingested,
            embedder,
            reranker,
            k=args.k,
            config=config,
            concurrency=args.concurrency,
        )
        summary["name"] = config_name
        results.append(summary)
        print(
            f"    {config_name:22s} FULL={summary['full_recall@k']:.3f} "
            f"hit={summary['hit@k']:.3f} mrr={summary['mrr']:.3f} "
            f"{summary['latency_ms_mean']:.0f}ms ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    end_to_end = None
    if args.end_to_end:
        best = max(results, key=lambda r: r["full_recall@k"])
        winner = CONFIGS[best["name"]]
        print(f"  end-to-end with {best['name']} (best full recall) ...", flush=True)
        started = time.perf_counter()
        end_to_end = await evaluate_end_to_end(
            corpora,
            ingested,
            embedder,
            HeuristicReranker() if winner.get("use_rerank") else NoopReranker(),
            k=args.answer_k,
            config=winner,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            official_judge=not args.strict_judge,
            use_derive=args.derive,
            thinking_budget=args.thinking_budget,
            dynamic_k=args.dynamic_k,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            concurrency=args.concurrency,
        )
        end_to_end["name"] = best["name"]
        lo, hi = end_to_end["accuracy_ci"]
        print(
            f"    accuracy={end_to_end['accuracy']:.3f} [{lo:.3f},{hi:.3f}] "
            f"abstention={end_to_end['abstention_rate']} "
            f"errors={end_to_end['n_errors']} ({time.perf_counter() - started:.0f}s)"
        )

    payload = {
        "benchmark": name,
        "end_to_end": end_to_end,
        "evidence_granularity": dataset.evidence_granularity,
        "corpora": len(corpora),
        "documents": documents,
        "questions": questions,
        "k": args.k,
        "embedder": {"name": embedder.name, "model": embedder.model},
        "configs": results,
    }
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    return payload


def write_report(out: Path, payloads: list[dict[str, Any]]) -> None:
    lines = [
        "# Multi-benchmark retrieval report",
        "",
        "Retrieval scored against each benchmark's labelled evidence — no LLM, "
        "so these numbers carry no judge variance and are exactly reproducible.",
        "",
        "**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item "
        "is retrieved. `hit@k` (any evidence found) is shown beside it because "
        "it is what most published numbers report, and the gap between the two "
        "is the point — on multi-evidence questions hit@k overstates readiness "
        "badly. Answering a three-hop question with one of three facts is a "
        "wrong answer.",
        "",
        "Evidence granularity differs by benchmark and the numbers are **not "
        "comparable across benchmarks**: LoCoMo labels individual dialogue "
        "turns, LongMemEval labels whole sessions, and finding the right "
        "session is a far easier target than finding the right turn.",
        "",
    ]
    for payload in payloads:
        lines += [
            f"## {payload['benchmark']}",
            "",
            f"{payload['corpora']} corpora · {payload['documents']} documents · "
            f"{payload['questions']} questions · k={payload['k']} · "
            f"evidence at **{payload['evidence_granularity']}** level · "
            f"embedder `{payload['embedder']['model']}`",
            "",
            "| config | full_recall@k | hit@k | MRR | nDCG | latency ms |",
            "|---|---|---|---|---|---|",
        ]
        for cfg in payload["configs"]:
            lines.append(
                f"| `{cfg['name']}` | **{cfg['full_recall@k']:.3f}** | "
                f"{cfg['hit@k']:.3f} | {cfg['mrr']:.3f} | {cfg['ndcg@k']:.3f} | "
                f"{cfg['latency_ms_mean']:.0f} |"
            )
        best = max(payload["configs"], key=lambda c: c["full_recall@k"])
        lines += ["", f"### Per capability — `{best['name']}` (best full recall)", ""]
        lines += [
            "| capability | n | avg evidence | full_recall@k | 95% CI | hit@k |",
            "|---|---|---|---|---|---|",
        ]
        for name, entry in best["by_category"].items():
            lo, hi = entry["full_recall_ci"]
            lines.append(
                f"| {name} | {entry['n']} | {entry['avg_evidence']} | "
                f"**{entry['full_recall@k']:.3f}** | {lo:.2f}-{hi:.2f} | "
                f"{entry['hit@k']:.3f} |"
            )
        e2e = payload.get("end_to_end")
        if e2e:
            lo, hi = e2e["accuracy_ci"]
            lines += [
                "",
                f"### End-to-end (LongMemEval's published metric) — `{e2e['name']}`",
                "",
                f"Answering model `{e2e['answer_model']}`, judge "
                f"`{e2e['judge_model']}`, judge protocol "
                f"`{e2e.get('judge_protocol', 'unknown')}`.",
                "",
                f"**QA accuracy: {e2e['accuracy']:.1%}** "
                f"(95% CI {lo:.1%}-{hi:.1%}, n={e2e['n_answerable']})  ",
                f"Abstention: {e2e['abstention_rate']} (n={e2e['n_abstention']}) · "
                f"context {e2e['context_tokens_mean']:.0f} tok · "
                f"search {e2e['latency_ms_mean']:.0f} ms · errors {e2e['n_errors']}",
                "",
                "| capability | n | accuracy | 95% CI |",
                "|---|---|---|---|",
            ]
            for cname, entry in e2e["by_category"].items():
                clo, chi = entry["ci"]
                lines.append(
                    f"| {cname} | {entry['n']} | {entry['accuracy']:.1%} | "
                    f"{clo:.2f}-{chi:.2f} |"
                )
        errors = sum(c["n_failed"] for c in payload["configs"])
        bias = (e2e or {}).get("judge_bias")
        if bias:
            direction = (
                "more generous to itself"
                if bias["delta_self_minus_independent"] > 0
                else "stricter on itself"
            )
            lines += [
                "",
                "### Judge independence",
                "",
                "The answering model does not grade itself. Both arms below "
                "grade the **same** predictions with the **same** prompt; the "
                "only variable is whether the grader produced them.",
                "",
                "| arm | grader | accuracy |",
                "|---|---|---|",
                f"| independent (headline) | `{bias['independent_judge']}` | "
                f"{bias['independent_accuracy']:.1%} |",
                f"| self-graded (control) | `{bias['self_judge']}` | "
                f"{bias['self_accuracy']:.1%} |",
                "",
                f"Self-preference: **{bias['delta_self_minus_independent']:+.1%}** "
                f"({direction}), judge agreement {bias['agreement']:.1%} on "
                f"n={bias['n_paired']}. The headline uses the independent "
                "grader; this delta is what a self-graded number would have "
                "silently added.",
            ]
        lines += [
            "",
            "### Methodology caveats",
            "",
            "* The advice/preference router that selects the personalisation "
            "prompt was **tuned against this benchmark's own "
            "`single-session-preference` questions**. Its specificity is "
            "independently validated (0 false positives on 1,986 LoCoMo "
            "questions), but its sensitivity is fitted, so the "
            "`single-session-preference` figure above is optimistically "
            "biased. Every other capability is untouched by it.",
            "* `question_date` is a benchmark-provided input, not a label: it "
            'is what "how many weeks ago" is measured from, and every '
            "system evaluated here receives it.",
        ]
        lines += ["", "### Completeness", ""]
        lines.append(
            f"Failed searches: **{errors}** (excluded from retrieval metrics, "
            "never substituted)."
        )
        if e2e:
            attempted = e2e["n_attempted"]
            scored = e2e["n_scored"]
            lines.append(
                f"Failed answer/judge calls: **{e2e['n_errors']}** of {attempted} "
                f"attempted; {scored} scored. An API failure is missing data, not a "
                "wrong answer, so it is excluded from accuracy — but the gap is "
                "reported here because an unreported gap silently inflates the "
                "headline."
            )
        lines.append("")

    (out / "report.md").write_text("\n".join(lines) + "\n")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="all", choices=[*DATASETS, "all"])
    parser.add_argument("--corpora", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embeddings", default="gemini")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="also run answer+judge (LongMemEval's published metric)",
    )
    parser.add_argument(
        "--answer-k",
        type=int,
        default=10,
        help=(
            "sessions retrieved AND passed to the answerer. This is the "
            "delivery budget, and it was the bottleneck: at 4, complete "
            "evidence reached the answerer for only 87.0% of questions while "
            "the ablation reported full_recall@10 = 0.968 -- two different "
            "arms. Accuracy is 87.5% when evidence is complete and 26.2% when "
            "it is not. The old default of 4 was justified by 'MRR 0.939 means "
            "the right session is ranked first', which is a hit@k argument "
            "applied to a CONJUNCTIVE metric: 324 of 500 questions need two or "
            "more sessions, and 17 need five or more (arithmetically "
            "impossible to satisfy at k=4)."
        ),
    )
    parser.add_argument(
        "--dynamic-k",
        action="store_true",
        help=(
            "size the evidence budget per question from its SHAPE instead of "
            "using one k for everything. Measured: gold-session need varies 6x "
            "by shape (advice max 1, multi-session up to 5), so a fixed k "
            "always starves someone or drowns someone. 43% of questions are "
            "single-evidence shapes currently receiving ten sessions."
        ),
    )
    parser.add_argument("--answer-model", default="gemini-2.5-flash")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=128,
        help=(
            "reasoning tokens the ANSWERER may spend before its first output "
            "token. Applies to Gemini answerers only. 128 is the historical "
            "default and has never been varied; 19 of the 40 failures that "
            "received complete evidence are multi-step chains, so reasoning "
            "depth is a candidate bottleneck. NOTE: a gain here is an answerer "
            "property, not a memory property, and should be reported as such."
        ),
    )
    parser.add_argument(
        "--judge-model",
        # Deliberately NOT the answering model. Grading your own output is a
        # known self-preference bias, and a self-graded headline is the first
        # thing a reader should distrust -- the largest published gap on this
        # benchmark (94.4% self-reported vs 49.0% independently measured) is
        # attributed to judge configuration, not to the memory engine. When
        # this differs from --answer-model the harness also runs the answerer
        # as a second grader and reports the measured gap.
        default="claude-sonnet-4-5@20250929",
        help="grader; keep it different from --answer-model. Default is "
        "CROSS-FAMILY (Anthropic on Vertex vs a Gemini answerer): a "
        "different vendor shares no training lineage, so it does not "
        "inherit the answerer's blind spots the way a same-family judge "
        "does. Also enables self-preference measurement.",
    )
    parser.add_argument(
        "--derive",
        action="store_true",
        help="route aggregate questions (count/order/date-arith) through the "
        "map->ground->reduce derive path and escalate declines through it; "
        "code does the arithmetic, the model only finds instances",
    )
    parser.add_argument(
        "--strict-judge",
        action="store_true",
        help="grade with our custom stricter prompt instead of LongMemEval's "
        "official per-question-type prompts. Not comparable to published "
        "numbers; kept for ablation.",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    configure_logging("WARNING", False)
    run_id = args.run_id or f"multi-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}")

    names = list(DATASETS) if args.benchmark == "all" else [args.benchmark]
    payloads = [p for n in names if (p := await run_one(n, args, out)) is not None]
    if payloads:
        write_report(out, payloads)
        print(f"\nwrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
