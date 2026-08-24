"""Does `use_expansion` earn its model call? Measured on LoCoMo, not LongMemEval.

`use_expansion` has shipped defaulting False since it was written, with no
evidence either way. That is the profile the reranker had, and the reranker
turned out to be the worst of five configs on every retrieval metric at once,
so an untested flag with a real implementation behind it is not a neutral thing
to leave lying around.

WHY LoCoMo AND NOT LME. Expansion can only help where retrieval currently
fails, and LongMemEval barely fails: full_recall@10 is 0.968, leaving 3.2% of
headroom for a mechanism costing a model call per query. LoCoMo labels
individual dialogue TURNS rather than whole sessions and sits at recall@10 =
0.644, so it has room to move either way. Measuring a lever against a saturated
metric concludes "no effect" about the corpus rather than about the lever.

INGEST GOES THROUGH `bench.harness.ingest`, with the same disk vector cache
every other number in bench/ was produced with. The first version of this file
reimplemented ingest as a sequential uncached loop and died on an embedding
timeout ~800 documents in; through the cache the same corpus loads in 2s.

THE RESULT WAS A FALSE NULL, and the reason is recorded here because the shape
recurs. 600 questions, both arms, recall identical to three decimals in every
category, 0 of 600 questions moved -- and the expanded arm ran FASTER than the
plain one, which is impossible if it were making an extra model call. HyDE was
returning nothing: the probe gave the completer a 256-token budget that
gemini-2.5-flash's reasoning consumed before emitting any text, `expand()`
returned [] on every query, and `_embed_query` fell back to the plain vector.
`CompletionExpander` exists because `use_expansion=True` was once a silent
no-op everywhere; it had become one again by a different route.

So the arms below carry an explicit assertion that expansion actually fired.
A measurement that cannot tell "no effect" from "never ran" is not a
measurement.

    python -m bench.lab.expansion_probe [n_corpora] [per_corpus]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.domain.retrieval.expansion import CompletionExpander
from mapi.domain.retrieval.pipeline import RetrievalPipeline, SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker

from .stats import fmt_p, sign_test

K = 10
REPO = Path(__file__).resolve().parents[2]
#: Generous on purpose. 256 was the false null: reasoning tokens are drawn from
#: the same allowance as the answer on gemini-2.5-flash, so a tight budget
#: yields an empty completion rather than a short one.
EXPANDER_MAX_TOKENS = 1024


async def main(n_corpora: int, per_corpus: int) -> None:
    from bench.cache import DiskVectorCache
    from bench.datasets.locomo import LoCoMo
    from bench.harness import build_model_client, ingest
    from bench.metrics import full_recall_at_k, recall_at_k

    corpora = LoCoMo(REPO / "bench" / "data" / "locomo10.json").load(limit_corpora=n_corpora)
    for c in corpora:
        c.questions = [q for q in c.questions if q.evidence_ids][:per_corpus]
    total = sum(len(c.questions) for c in corpora)
    print(f"LoCoMo: {len(corpora)} corpora, {total} questions with labelled evidence")

    embedder = GeminiEmbedder(dimensions=768)
    cache = DiskVectorCache(REPO / "bench" / "data" / "cache" / "vectors.sqlite")
    print("ingesting (disk-cached embeddings) ...")
    started = time.perf_counter()
    ingested = await ingest(corpora, embedder, cache=cache)
    print(
        f"  {ingested.documents} documents, {ingested.chunks} chunks "
        f"in {time.perf_counter() - started:.0f}s"
    )

    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    gap = asyncio.Semaphore(8)
    fired = {"ok": 0, "empty": 0}

    async def complete(prompt: str) -> str:
        # NO SEMAPHORE HERE. `ask` already holds a slot while it awaits
        # `pipeline.search`, and the expander runs INSIDE that search -- taking
        # the same semaphore again is a re-entrant wait that can never be
        # satisfied. Measured: all 8 slots held by searches blocked on
        # expanders queued behind those very searches, 597 of 600 queries
        # hitting HyDE's 12s timeout, median query latency 12,240ms, and one
        # single successful expansion in the whole run. Concurrency is already
        # bounded by the caller.
        raw, _tokens = await client.complete(prompt, max_tokens=EXPANDER_MAX_TOKENS)
        if raw and raw.strip():
            fired["ok"] += 1
        else:
            fired["empty"] += 1
        return raw

    # THE PRECONDITION. Before spending the run, prove the expander produces a
    # passage at all -- this is the check whose absence produced the false null.
    probe = await CompletionExpander(complete).expand("What did she say about her new job?")
    if not probe:
        print("\nABORT: the expander returned no passage. Expansion cannot be")
        print("measured because it is not running. Raise EXPANDER_MAX_TOKENS or")
        print("check the completer before reading anything below as a result.")
        return
    print(f"expander precondition OK -> {probe[0][:90]!r}\n")

    # Two pipelines over ONE store: identical corpus, identical index, the
    # expander is the only difference between the arms.
    plain = RetrievalPipeline(ingested.store, embedder, HeuristicReranker())
    expanded = RetrievalPipeline(
        ingested.store,
        embedder,
        HeuristicReranker(),
        expander=CompletionExpander(complete),
    )

    async def ask(pipeline: RetrievalPipeline, question, expand: bool) -> dict | None:
        space_id = ingested.spaces.get(question.corpus_id)
        if space_id is None:
            return None
        async with gap:
            t0 = time.perf_counter()
            try:
                r = await pipeline.search(
                    SearchRequest(
                        query=question.text,
                        org_id=ingested.org_id,
                        space_id=space_id,
                        limit=K,
                        known_speakers=ingested.speakers,
                        asked_at=question.asked_at,
                        use_expansion=expand,
                    )
                )
            except Exception as exc:
                return {"qid": question.qid, "error": f"{type(exc).__name__}: {exc}"[:120]}
            ms = (time.perf_counter() - t0) * 1000
        got = [ingested.doc_by_memory.get(h.memory.id, "") for h in r.results]
        want = set(question.evidence_ids)
        return {
            "qid": question.qid,
            "category": question.category,
            "expand": expand,
            "recall": recall_at_k(got, want, K),
            "full": full_recall_at_k(got, want, K),
            "ms": ms,
        }

    questions = [q for c in corpora for q in c.questions]
    print(f"querying {len(questions)} x 2 arms ...")
    rows: list[dict] = []
    for name, pipe, flag in (("base", plain, False), ("expand", expanded, True)):
        before = dict(fired)
        out = await asyncio.gather(*(ask(pipe, q, flag) for q in questions))
        good = [r for r in out if r and "error" not in r]
        errs = [r for r in out if r and "error" in r]
        calls = (fired["ok"] - before["ok"], fired["empty"] - before["empty"])
        note = f"  e.g. {errs[0]['error']}" if errs else ""
        print(
            f"  {name}: {len(good)} scored, {len(errs)} errors, "
            f"expander calls ok/empty = {calls[0]}/{calls[1]}{note}"
        )
        rows += good

    base = {r["qid"]: r for r in rows if not r["expand"]}
    exp = {r["qid"]: r for r in rows if r["expand"]}
    common = sorted(set(base) & set(exp))
    n = len(common)
    if not n:
        print("no paired questions -- nothing to compare")
        return

    print(f"\nscored {n} paired questions, k={K}\n")
    print(f"{'arm':18}{'recall@10':>11}{'full_recall':>13}{'median ms':>11}")
    for label, d in (("no expansion", base), ("expansion (HyDE)", exp)):
        ms = sorted(d[q]["ms"] for q in common)
        print(
            f"{label:18}{sum(d[q]['recall'] for q in common) / n:>11.3f}"
            f"{sum(d[q]['full'] for q in common) / n:>13.3f}{ms[len(ms) // 2]:>11.0f}"
        )

    win = sum(1 for q in common if exp[q]["recall"] > base[q]["recall"])
    lose = sum(1 for q in common if exp[q]["recall"] < base[q]["recall"])
    print(
        f"\nPAIRED  expansion better on {win}, worse on {lose}, "
        f"tied on {n - win - lose}   {fmt_p(sign_test(win, lose))}"
    )
    if win == 0 and lose == 0:
        print("  ALL TIED -- verify the expander fired before reading this as a null.")

    cats: dict[str, list[str]] = {}
    for q in common:
        cats.setdefault(str(base[q]["category"]), []).append(q)
    print(f"\n{'category':24}{'base':>8}{'expand':>9}{'n':>6}")
    for cat, qs in sorted(cats.items()):
        b = sum(base[q]["recall"] for q in qs) / len(qs)
        e = sum(exp[q]["recall"] for q in qs) / len(qs)
        print(f"{cat:24}{b:>8.3f}{e:>9.3f}{len(qs):>6}")

    out_path = Path(__file__).with_name("expansion_probe_results.json")
    out_path.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out_path.name}")


if __name__ == "__main__":
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    _per = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    raise SystemExit(asyncio.run(main(_n, _per)))
