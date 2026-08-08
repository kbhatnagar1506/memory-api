"""LoCoMo benchmark harness.

Implements the MemoryBench pipeline shape — ingest -> index -> search -> answer
-> evaluate -> report — against this service, and reports MemScore's triple
(`accuracy% / latencyMs / contextTokens`).

Two honesty constraints govern the whole file:

  * Every number written to a results file comes from a real API response. There
    is no placeholder path and no default that silently substitutes a plausible
    value. A failed call is recorded as a failure and shows up as missing data.
  * Retrieval metrics are scored against LoCoMo's labelled evidence turns, not
    by an LLM, so the ablation carries no judge variance. Only end-to-end answer
    accuracy uses a judge, and that is reported separately.

This is NOT a run of the `supermemoryai/memorybench` repository: that requires
bun and hosted provider keys. It is a faithful re-implementation of its pipeline
against a local service, with a different judge model, so the numbers are
comparable in kind to their leaderboard but not head-to-head with it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench import locomo
from bench.context import merge_intervals, render_context
from supermemory.config import (
    EmbeddingBackend,
    RerankBackend,
    Settings,
    StoreBackend,
)
from supermemory.core.logging import configure_logging
from supermemory.domain.embeddings import build_embedder
from supermemory.domain.models import Chunk, Memory, Organization, Space
from supermemory.domain.retrieval.expansion import HydeExpander
from supermemory.domain.retrieval.pipeline import (
    RetrievalPipeline,
    SearchRequest,
)
from supermemory.domain.retrieval.rerank import (
    HeuristicReranker,
    LLMReranker,
    NoopReranker,
)
from supermemory.domain.text import analyze
from supermemory.store.memory import InMemoryStore

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"


# -- ablation configurations ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    vector_weight: float = 1.0
    lexical_weight: float = 1.0
    use_rerank: bool = False
    use_mmr: bool = False
    use_decay: bool = False
    reranker: str = "none"
    mmr_lambda: float = 0.7
    use_expansion: bool = False

    def describe(self) -> str:
        parts = []
        if self.vector_weight and self.lexical_weight:
            parts.append("hybrid")
        elif self.vector_weight:
            parts.append("vector")
        else:
            parts.append("lexical")
        if self.use_rerank:
            parts.append(f"rerank:{self.reranker}")
        if self.use_mmr:
            parts.append(f"mmr:{self.mmr_lambda}")
        if self.use_decay:
            parts.append("decay")
        if self.use_expansion:
            parts.append("hyde")
        return " + ".join(parts)


ABLATION = [
    Config("vector_only", vector_weight=1.0, lexical_weight=0.0),
    Config("lexical_only", vector_weight=0.0, lexical_weight=1.0),
    Config("hybrid_rrf"),
    Config("hybrid_rerank_heuristic", use_rerank=True, reranker="heuristic"),
    Config("hybrid_mmr", use_mmr=True),
    Config("hybrid_decay", use_decay=True),
    Config(
        "full_no_llm",
        use_rerank=True,
        reranker="heuristic",
        use_mmr=True,
        use_decay=True,
    ),
]
LLM_CONFIG = Config("hybrid_rerank_llm", use_rerank=True, reranker="llm")
#: Query expansion costs an LLM call per query, so it is opt-in behind a flag
#: rather than part of the default sweep.
EXPANSION_CONFIGS = [
    Config("hybrid_hyde", use_expansion=True),
    Config(
        "hybrid_hyde_rerank",
        use_expansion=True,
        use_rerank=True,
        reranker="heuristic",
    ),
]


# -- ingestion -----------------------------------------------------------------


@dataclass
class Corpus:
    store: InMemoryStore
    org_id: str
    #: sample_id -> space_id
    spaces: dict[str, str] = field(default_factory=dict)
    #: memory_id -> dia_id, so retrieved memories map back to evidence labels
    dia_by_memory: dict[str, str] = field(default_factory=dict)
    #: sample_id -> ordered turn texts, and dia_id -> its index. Used to widen
    #: a retrieved turn with the ones around it at answer time.
    ordered_turns: dict[str, list[str]] = field(default_factory=dict)
    #: sample_id -> ordered (date, content) pairs — the structured form the v2
    #: context assembler needs so date headers can be deduplicated.
    turn_records: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    turn_index: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    embed_seconds: float = 0.0

    def window(self, sample_id: str, dia_id: str, radius: int) -> str:
        """A retrieved turn plus its neighbours, in conversation order.

        Retrieval is precise (one turn per memory, so evidence labels map
        exactly) but a single line of dialogue is rarely self-contained: "Yeah,
        psychology" means nothing without the question before it. Widening at
        READ time keeps the retrieval metric honest while giving the answering
        model something it can actually reason over.
        """
        turns = self.ordered_turns.get(sample_id, [])
        centre = self.turn_index.get(f"{sample_id}:{dia_id}")
        if centre is None or not turns:
            return ""
        lo = max(0, centre - radius)
        hi = min(len(turns), centre + radius + 1)
        return "\n".join(turns[lo:hi])


async def ingest(
    conversations: list[locomo.Conversation], embedder, batch_size: int = 64
) -> Corpus:
    """Bulk-load turns as one memory each.

    Deliberately bypasses MemoryService.ingest: that path embeds per memory,
    which would be 5,882 single-text API calls. Batching is the only difference;
    the stored records are identical, and every turn keeps its own embedding.
    Dedup is off because two speakers genuinely repeat themselves in dialogue
    and collapsing that would destroy evidence labels.
    """
    store = InMemoryStore()
    org = await store.create_organization(Organization(name="locomo"))
    corpus = Corpus(store=store, org_id=org.id)
    semaphore = asyncio.Semaphore(6)

    async def embed_batch(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return (await embedder.embed(batch)).vectors

    for convo in conversations:
        space = await store.create_space(
            Space(
                org_id=org.id,
                slug=convo.sample_id.lower().replace("_", "-")[:60],
                name=convo.sample_id,
            )
        )
        corpus.spaces[convo.sample_id] = space.id
        corpus.ordered_turns[convo.sample_id] = [
            f"[{t.session_date[:10]}] {t.content}" for t in convo.turns
        ]
        corpus.turn_records[convo.sample_id] = [
            (t.session_date[:10], t.content) for t in convo.turns
        ]
        for position, turn in enumerate(convo.turns):
            corpus.turn_index[f"{convo.sample_id}:{turn.dia_id}"] = position

        texts = [t.content for t in convo.turns]
        started = time.perf_counter()
        batches = [
            texts[start : start + batch_size] for start in range(0, len(texts), batch_size)
        ]
        results = await asyncio.gather(*(embed_batch(b) for b in batches))
        vectors: list[list[float]] = [v for batch in results for v in batch]
        corpus.embed_seconds += time.perf_counter() - started

        for turn, vector in zip(convo.turns, vectors, strict=True):
            memory = Memory(
                org_id=org.id,
                space_id=space.id,
                content=turn.content,
                source=turn.speaker,
                tags=[turn.session],
                metadata={"dia_id": turn.dia_id, "session": turn.session},
                occurred_at=datetime.fromisoformat(turn.session_date),
            )
            memory = memory.model_copy(
                update={
                    "chunks": [
                        Chunk(
                            memory_id=memory.id,
                            ordinal=0,
                            text=turn.content,
                            embedding=vector,
                        )
                    ]
                }
            )
            await store.upsert_memory(memory)
            corpus.dia_by_memory[memory.id] = turn.dia_id
            corpus.turns += 1

    return corpus


# -- retrieval evaluation ------------------------------------------------------


async def evaluate_retrieval(
    corpus: Corpus,
    questions: list[locomo.Question],
    config: Config,
    embedder,
    *,
    k: int,
) -> dict[str, Any]:
    reranker = {
        "none": NoopReranker(),
        "heuristic": HeuristicReranker(),
    }.get(config.reranker)
    if config.reranker == "llm":
        reranker = LLMReranker(model="gemini-2.5-flash", timeout_s=25.0, project=_project())
    expander = (
        HydeExpander(model="gemini-2.5-flash", timeout_s=20.0, project=_project())
        if config.use_expansion
        else None
    )
    pipeline = RetrievalPipeline(corpus.store, embedder, reranker or NoopReranker(), expander)

    per_question: list[dict[str, Any]] = []
    latencies: list[float] = []
    failures = 0

    for question in questions:
        space_id = corpus.spaces.get(question.sample_id)
        if space_id is None:
            continue
        started = time.perf_counter()
        try:
            response = await pipeline.search(
                SearchRequest(
                    query=question.question,
                    org_id=corpus.org_id,
                    space_id=space_id,
                    limit=k,
                    use_rerank=config.use_rerank,
                    use_mmr=config.use_mmr,
                    use_decay=config.use_decay,
                    mmr_lambda=config.mmr_lambda,
                    vector_weight=config.vector_weight,
                    lexical_weight=config.lexical_weight,
                    use_expansion=config.use_expansion,
                    candidate_multiplier=6,
                    rerank_candidates=32,
                )
            )
        except Exception as exc:
            failures += 1
            per_question.append(
                {"qid": question.qid, "error": f"{type(exc).__name__}: {exc}"[:200]}
            )
            continue
        latencies.append((time.perf_counter() - started) * 1000)

        retrieved = [corpus.dia_by_memory.get(hit.memory.id, "") for hit in response.results]
        relevant = set(question.evidence)
        per_question.append(
            {
                "qid": question.qid,
                "category": question.category,
                "category_name": question.category_name,
                "n_evidence": len(relevant),
                "retrieved": retrieved,
                "context_chars": sum(len(hit.memory.content) for hit in response.results),
                "recall@k": locomo.recall_at_k(retrieved, relevant, k),
                "hit@1": locomo.hit_at_k(retrieved, relevant, 1),
                "hit@5": locomo.hit_at_k(retrieved, relevant, 5),
                "hit@k": locomo.hit_at_k(retrieved, relevant, k),
                "mrr": locomo.mrr(retrieved, relevant),
                "ndcg@k": locomo.ndcg_at_k(retrieved, relevant, k),
                "strategies": response.strategies,
            }
        )

    # Adversarial questions have no retrievable evidence by construction, so
    # including them would drag every retrieval metric toward zero for a reason
    # that has nothing to do with retrieval quality.
    scored = [row for row in per_question if "error" not in row and row["n_evidence"] > 0]

    def mean(key: str) -> float:
        values = [row[key] for row in scored]
        return round(statistics.fmean(values), 4) if values else 0.0

    by_category: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        grouped[row["category_name"]].append(row)
    for name, rows in sorted(grouped.items()):
        by_category[name] = {
            "n": len(rows),
            "recall@k": round(statistics.fmean(r["recall@k"] for r in rows), 4),
            "hit@k": round(statistics.fmean(r["hit@k"] for r in rows), 4),
            "mrr": round(statistics.fmean(r["mrr"] for r in rows), 4),
        }

    return {
        "config": config.name,
        "description": config.describe(),
        "k": k,
        "n_scored": len(scored),
        "n_failed": failures,
        "recall@k": mean("recall@k"),
        "hit@1": mean("hit@1"),
        "hit@5": mean("hit@5"),
        "hit@k": mean("hit@k"),
        "mrr": mean("mrr"),
        "ndcg@k": mean("ndcg@k"),
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latency_ms_p95": (
            round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
            if len(latencies) >= 20
            else None
        ),
        "context_chars_mean": (
            round(statistics.fmean(r["context_chars"] for r in scored), 1) if scored else 0.0
        ),
        "by_category": by_category,
        "per_question": per_question,
    }


# -- answer + judge ------------------------------------------------------------


def _project() -> str | None:
    import os

    return os.getenv("GOOGLE_CLOUD_PROJECT")


ANSWER_PROMPT = """\
You are answering a question about a long-running conversation between two \
people, using excerpts retrieved from it. Each excerpt is a short stretch of \
dialogue, prefixed with the date it happened.

Excerpts:
{context}

Question: {question}

Work through it briefly, then give a final answer.

- Combine information across excerpts when the answer needs more than one.
- Dates in brackets are when the line was said; use them for questions about \
when something happened or what came first.
- If the excerpts genuinely do not contain the answer, the final answer is \
NO_ANSWER. Do not guess at facts that are absent. But if the excerpts DO \
support an answer, give it, even if you must infer a little.

Reply in exactly this form:
REASONING: <one or two sentences>
ANSWER: <the answer in as few words as possible, or NO_ANSWER>"""

JUDGE_PROMPT = """\
You are grading a question-answering system against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {prediction}

Mark CORRECT if the system answer conveys the same fact as the reference, \
allowing for differences in wording, formatting, or extra detail. Dates may be \
written differently but must refer to the same date. Mark INCORRECT otherwise.

Reply with exactly one word: CORRECT or INCORRECT"""


class Gemini:
    """Thin Vertex client for the answer and judge phases."""

    def __init__(self, model: str, project: str | None) -> None:
        from google import genai

        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for the answer/judge phases")
        self.client = genai.Client(vertexai=True, project=project, location="global")
        self.model = model

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> tuple[str, int]:
        from google.genai import types

        def call() -> tuple[str, int]:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # Thinking models draw reasoning tokens from this same
                    # budget; without headroom the answer comes back truncated.
                    max_output_tokens=max_tokens + 512,
                    thinking_config=types.ThinkingConfig(thinking_budget=128),
                ),
            )
            usage = response.usage_metadata
            return (response.text or ""), int(getattr(usage, "prompt_token_count", 0) or 0)

        return await asyncio.to_thread(call)


def parse_answer(raw: str) -> str:
    """Pull the final answer out of a REASONING/ANSWER response.

    Falls back to the last non-empty line: a model that ignores the format has
    still usually put its answer last, and discarding that would score a
    formatting slip as a wrong answer.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("ANSWER:"):
            return stripped[len("ANSWER:") :].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[-1].upper().startswith("REASONING:"):
        return ""
    return lines[-1] if lines else ""


async def evaluate_end_to_end(
    corpus: Corpus,
    questions: list[locomo.Question],
    config: Config,
    embedder,
    *,
    k: int,
    answer_model: str,
    judge_model: str,
    concurrency: int,
    context_radius: int = 0,
    compress: bool = False,
) -> dict[str, Any]:
    reranker = HeuristicReranker() if config.use_rerank else NoopReranker()
    expander = (
        HydeExpander(model="gemini-2.5-flash", timeout_s=20.0, project=_project())
        if config.use_expansion
        else None
    )
    pipeline = RetrievalPipeline(corpus.store, embedder, reranker, expander)
    answerer = Gemini(answer_model, _project())
    judge = Gemini(judge_model, _project())
    semaphore = asyncio.Semaphore(concurrency)

    async def one(question: locomo.Question) -> dict[str, Any]:
        space_id = corpus.spaces.get(question.sample_id)
        if space_id is None:
            return {"qid": question.qid, "error": "no space"}
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await pipeline.search(
                    SearchRequest(
                        query=question.question,
                        org_id=corpus.org_id,
                        space_id=space_id,
                        limit=k,
                        use_rerank=config.use_rerank,
                        use_mmr=config.use_mmr,
                        use_decay=config.use_decay,
                        vector_weight=config.vector_weight,
                        lexical_weight=config.lexical_weight,
                        use_expansion=config.use_expansion,
                    )
                )
            except Exception as exc:
                return {"qid": question.qid, "error": f"search: {exc}"[:200]}
            search_ms = (time.perf_counter() - started) * 1000

            # Every turn carries an event date; the model cannot resolve
            # "tomorrow" or answer "when did X happen" without it. Omitting
            # this was a harness bug: it drove the entire `temporal` category
            # to 0/40 correct even when retrieval found the right turn
            # (hit@k=1.0), because the model had a fact but no date to report.
            if context_radius > 0:
                # v2 assembly: widen each hit with neighbours, then MERGE the
                # overlapping windows so every turn is emitted exactly once,
                # with date headers only on change. v1 concatenated per-hit
                # windows and paid for every overlap twice. Retrieval itself is
                # untouched — evidence labels still map turn-for-turn.
                turns = corpus.turn_records.get(question.sample_id, [])
                centers: set[int] = set()
                fallbacks: list[str] = []
                for hit in response.results:
                    dia_id = corpus.dia_by_memory.get(hit.memory.id, "")
                    center = corpus.turn_index.get(f"{question.sample_id}:{dia_id}")
                    if center is None or not turns:
                        fallbacks.append(
                            f"[{hit.memory.occurred_at.date().isoformat()}] "
                            f"{hit.memory.content}"
                        )
                    else:
                        centers.add(center)
                merged = merge_intervals(
                    [
                        (max(0, c - context_radius), min(len(turns) - 1, c + context_radius))
                        for c in sorted(centers)
                    ]
                )
                keep_terms: set[str] | None = None
                if compress and centers:
                    # Prune neighbour turns sharing no analyzed term with the
                    # question or any hit turn; +-1 of a hit always survives.
                    keep_terms = set(analyze(question.question))
                    for c in centers:
                        keep_terms.update(analyze(turns[c][1]))
                rendered = render_context(
                    turns,
                    merged,
                    centers,
                    keep_terms=keep_terms,
                    analyze=analyze if keep_terms is not None else None,
                )
                context = "\n---\n".join(part for part in [rendered, *fallbacks] if part)
            else:
                context = "\n".join(
                    f"- [{hit.memory.occurred_at.date().isoformat()}] {hit.memory.content}"
                    for hit in response.results
                )
            try:
                raw, context_tokens = await answerer.complete(
                    ANSWER_PROMPT.format(context=context, question=question.question),
                    # The prompt asks for a reasoning line before the answer, so
                    # 64 tokens truncates the response before it ever reaches
                    # "ANSWER:" and the parse falls back to reasoning prose.
                    max_tokens=200,
                )
            except Exception as exc:
                return {"qid": question.qid, "error": f"answer: {exc}"[:200]}
            prediction = parse_answer(raw)

            declined = prediction.upper().startswith("NO_ANSWER")
            if question.is_adversarial:
                # The correct behaviour on an unanswerable question is to
                # decline. No judge needed, and no judge variance introduced.
                correct = declined
                verdict = "DECLINED" if declined else "ANSWERED"
            elif declined:
                correct, verdict = False, "DECLINED"
            else:
                try:
                    grade, _ = await judge.complete(
                        JUDGE_PROMPT.format(
                            question=question.question,
                            reference=question.answer,
                            prediction=prediction,
                        ),
                        max_tokens=8,
                    )
                except Exception as exc:
                    return {"qid": question.qid, "error": f"judge: {exc}"[:200]}
                verdict = (
                    "CORRECT"
                    if "CORRECT" in grade.upper() and "INCORRECT" not in grade.upper()
                    else "INCORRECT"
                )
                correct = verdict == "CORRECT"

            retrieved = [
                corpus.dia_by_memory.get(hit.memory.id, "") for hit in response.results
            ]
            return {
                "qid": question.qid,
                "category": question.category,
                "category_name": question.category_name,
                "question": question.question,
                "reference": question.answer,
                "prediction": prediction[:300],
                "verdict": verdict,
                "correct": correct,
                "search_ms": round(search_ms, 2),
                "context_tokens": context_tokens,
                "hit@k": locomo.hit_at_k(retrieved, set(question.evidence), k),
            }

    rows = await asyncio.gather(*(one(q) for q in questions))
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]

    answerable = [r for r in ok if r["category"] != locomo.ADVERSARIAL]
    adversarial = [r for r in ok if r["category"] == locomo.ADVERSARIAL]

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ok:
        grouped[row["category_name"]].append(row)
    for name, group in sorted(grouped.items()):
        by_category[name] = {
            "n": len(group),
            "accuracy": round(sum(1 for r in group if r["correct"]) / len(group), 4),
        }

    accuracy = (
        sum(1 for r in answerable if r["correct"]) / len(answerable) if answerable else 0.0
    )
    latencies = [r["search_ms"] for r in ok]
    tokens = [r["context_tokens"] for r in ok if r["context_tokens"]]

    return {
        "config": config.name,
        "description": config.describe(),
        "context_radius": context_radius,
        "context_compress": compress,
        "context_strategy": "merged_v2" if context_radius > 0 else "flat",
        "answer_model": answer_model,
        "judge_model": judge_model,
        "k": k,
        "n_attempted": len(rows),
        "n_scored": len(ok),
        "n_errors": len(errors),
        "accuracy_answerable": round(accuracy, 4),
        "n_answerable": len(answerable),
        "adversarial_decline_rate": (
            round(sum(1 for r in adversarial if r["correct"]) / len(adversarial), 4)
            if adversarial
            else None
        ),
        "n_adversarial": len(adversarial),
        "retrieval_hit@k": round(statistics.fmean([r["hit@k"] for r in answerable]), 4)
        if answerable
        else 0.0,
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "context_tokens_mean": round(statistics.fmean(tokens), 1) if tokens else 0.0,
        "memscore": (
            f"{accuracy * 100:.0f}% / "
            f"{statistics.fmean(latencies):.0f}ms / "
            f"{statistics.fmean(tokens):.0f}tok"
            if latencies and tokens
            else "n/a"
        ),
        "by_category": by_category,
        "errors": errors[:20],
        "per_question": ok,
    }


# -- driver --------------------------------------------------------------------


def sample_questions(
    conversations: list[locomo.Conversation], n: int, seed: int
) -> list[locomo.Question]:
    """Stratified sample across categories, so no category is crowded out."""
    everything = [q for c in conversations for q in c.questions]
    rng = random.Random(seed)
    by_category: dict[int, list[locomo.Question]] = defaultdict(list)
    for question in everything:
        by_category[question.category].append(question)

    picked: list[locomo.Question] = []
    categories = sorted(by_category)
    per_category = max(1, n // max(len(categories), 1))
    for category in categories:
        pool = by_category[category]
        rng.shuffle(pool)
        picked.extend(pool[:per_category])
    rng.shuffle(picked)
    return picked[:n]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(REPO / "bench" / "data" / "locomo10.json"))
    parser.add_argument("--conversations", type=int, default=10)
    parser.add_argument("--questions", type=int, default=150)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--embeddings", default="gemini")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--answer-model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--include-llm-rerank", action="store_true")
    parser.add_argument(
        "--include-expansion",
        action="store_true",
        help="add HyDE query-expansion arms (one LLM call per query)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="extractively prune neighbour turns with no term overlap (measured separately)",
    )
    parser.add_argument(
        "--context-radius",
        type=int,
        default=0,
        help="turns of surrounding dialogue to include around each retrieved turn",
    )
    parser.add_argument(
        "--e2e-config",
        default=None,
        help=(
            "config name to run end-to-end. Defaults to best MRR; pin it to "
            "attribute an answer-side change at fixed retrieval."
        ),
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    configure_logging("WARNING", False)
    run_id = args.run_id or f"locomo-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)

    print(f"run {run_id}")
    conversations = locomo.load(Path(args.data), limit_conversations=args.conversations)
    turns = sum(len(c.turns) for c in conversations)
    all_questions = sum(len(c.questions) for c in conversations)
    print(
        f"  loaded {len(conversations)} conversations, {turns} turns, {all_questions} questions"
    )

    settings = Settings(
        store_backend=StoreBackend.MEMORY,
        embedding_backend=EmbeddingBackend(args.embeddings),
        embedding_dimensions=args.dimensions,
        embedding_model="text-embedding-004",
        embedding_batch_size=64,
        rerank_backend=RerankBackend.NONE,
        google_cloud_project=_project(),
    )
    embedder = build_embedder(settings)
    print(f"  embedder: {embedder.name} / {embedder.model} / {embedder.dimensions}d")

    print(f"  ingesting {turns} turns ...", flush=True)
    started = time.perf_counter()
    corpus = await ingest(conversations, embedder)
    print(
        f"  ingested {corpus.turns} memories in {time.perf_counter() - started:.1f}s "
        f"({corpus.embed_seconds:.1f}s embedding)"
    )

    questions = sample_questions(conversations, args.questions, args.seed)
    print(f"  sampled {len(questions)} questions")

    manifest = {
        "run_id": run_id,
        "dataset": "locomo10",
        "conversations": len(conversations),
        "turns": corpus.turns,
        "questions_total": all_questions,
        "questions_sampled": len(questions),
        "k": args.k,
        "seed": args.seed,
        "embedder": {
            "name": embedder.name,
            "model": embedder.model,
            "dimensions": embedder.dimensions,
        },
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "context_radius": args.context_radius,
        "started_at": datetime.now().isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # -- retrieval ablation (no LLM, exactly reproducible) --------------------
    configs = list(ABLATION)
    if args.include_expansion:
        configs.extend(EXPANSION_CONFIGS)
    if args.include_llm_rerank:
        configs.append(LLM_CONFIG)

    # Warm the query-embedding cache BEFORE timing anything. Otherwise the
    # first config pays for every query embedding and the rest ride its cache,
    # which makes the latency column a measure of ordering, not of retrieval.
    print("\n  warming query-embedding cache ...", flush=True)
    warm_started = time.perf_counter()
    embed_query = getattr(embedder, "embed_query", None)
    warm_semaphore = asyncio.Semaphore(8)

    async def warm(text: str) -> None:
        async with warm_semaphore:
            if embed_query is not None:
                await embed_query(text)
            else:
                await embedder.embed_one(text)

    await asyncio.gather(*(warm(q.question) for q in questions))
    warm_elapsed = time.perf_counter() - warm_started
    manifest["query_embed_seconds"] = round(warm_elapsed, 2)
    manifest["query_embed_ms_per_query"] = round(
        warm_elapsed * 1000 / max(len(questions), 1), 2
    )
    print(
        f"    {len(questions)} query embeddings in {warm_elapsed:.1f}s "
        f"({warm_elapsed * 1000 / max(len(questions), 1):.0f}ms each, uncached)"
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n  retrieval ablation")
    ablation: list[dict[str, Any]] = []
    for config in configs:
        started = time.perf_counter()
        result = await evaluate_retrieval(corpus, questions, config, embedder, k=args.k)
        elapsed = time.perf_counter() - started
        ablation.append(result)
        print(
            f"    {config.name:26s} recall@{args.k}={result['recall@k']:.3f} "
            f"hit@{args.k}={result['hit@k']:.3f} mrr={result['mrr']:.3f} "
            f"ndcg={result['ndcg@k']:.3f} {result['latency_ms_mean']:.1f}ms "
            f"({elapsed:.0f}s)"
        )
    (out / "retrieval_ablation.json").write_text(json.dumps(ablation, indent=2))

    # -- end-to-end -----------------------------------------------------------
    end_to_end: dict[str, Any] | None = None
    if not args.skip_end_to_end:
        if args.e2e_config:
            winner = next((c for c in configs if c.name == args.e2e_config), None)
            if winner is None:
                print(
                    f"    unknown --e2e-config {args.e2e_config!r}; known: "
                    f"{[c.name for c in configs]}"
                )
                return 2
            reason = "pinned"
        else:
            best = max(ablation, key=lambda r: r["mrr"])
            winner = next(c for c in configs if c.name == best["config"])
            reason = "best MRR"
        print(f"\n  end-to-end with {winner.name} ({reason}) ...", flush=True)
        started = time.perf_counter()
        end_to_end = await evaluate_end_to_end(
            corpus,
            questions,
            winner,
            embedder,
            k=args.k,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            context_radius=args.context_radius,
            compress=args.compress,
        )
        print(
            f"    accuracy={end_to_end['accuracy_answerable']:.3f} "
            f"adversarial_decline={end_to_end['adversarial_decline_rate']} "
            f"errors={end_to_end['n_errors']} "
            f"({time.perf_counter() - started:.0f}s)"
        )
        print(f"    MemScore: {end_to_end['memscore']}")
        (out / "end_to_end.json").write_text(json.dumps(end_to_end, indent=2))

    write_report(out, manifest, ablation, end_to_end)
    print(f"\nwrote {out / 'report.md'}")
    return 0


def write_report(
    out: Path,
    manifest: dict[str, Any],
    ablation: list[dict[str, Any]],
    end_to_end: dict[str, Any] | None,
) -> None:
    k = manifest["k"]
    lines = [
        f"# LoCoMo benchmark — {manifest['run_id']}",
        "",
        f"Corpus: **{manifest['turns']} dialogue turns** across "
        f"{manifest['conversations']} conversations, one memory per turn, one "
        f"space per conversation.  ",
        f"Questions: **{manifest['questions_sampled']} sampled** "
        f"(stratified by category) from {manifest['questions_total']}.  ",
        f"Embeddings: `{manifest['embedder']['model']}` "
        f"({manifest['embedder']['dimensions']}d, {manifest['embedder']['name']}).  ",
        f"Retrieval depth k = {k}. Seed {manifest['seed']}.",
        "",
        "Latency below is **retrieval only**: the query-embedding cache is warmed "
        "before timing, so every config is measured on equal footing. Query "
        f"embedding costs a further "
        f"{manifest.get('query_embed_ms_per_query', 0):.0f} ms per query "
        "uncached, and is reported separately because it is identical across "
        "configs and would otherwise swamp the comparison.",
        "",
        "Retrieval metrics are scored against LoCoMo's labelled evidence turns "
        "— no LLM, so they carry no judge variance and are exactly reproducible. "
        "Adversarial (unanswerable) questions are excluded from retrieval "
        "metrics, since they have no evidence to retrieve by construction.",
        "",
        "## Retrieval ablation",
        "",
        f"| config | recall@{k} | hit@1 | hit@{k} | MRR | nDCG@{k} | latency ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ablation:
        lines.append(
            f"| `{row['config']}` | {row['recall@k']:.3f} | {row['hit@1']:.3f} | "
            f"{row['hit@k']:.3f} | {row['mrr']:.3f} | {row['ndcg@k']:.3f} | "
            f"{row['latency_ms_mean']:.1f} |"
        )

    baseline = next((r for r in ablation if r["config"] == "hybrid_rrf"), None)
    if baseline:
        lines += [
            "",
            "### Against the hybrid baseline",
            "",
            "| config | Δ MRR | Δ recall | verdict |",
            "|---|---|---|---|",
        ]
        for row in ablation:
            if row["config"] == "hybrid_rrf":
                continue
            d_mrr = row["mrr"] - baseline["mrr"]
            d_recall = row["recall@k"] - baseline["recall@k"]
            verdict = "helps" if d_mrr > 0.01 else "hurts" if d_mrr < -0.01 else "no effect"
            lines.append(f"| `{row['config']}` | {d_mrr:+.3f} | {d_recall:+.3f} | {verdict} |")

    lines += ["", "### By question category", ""]
    categories = sorted({c for r in ablation for c in r["by_category"]})
    lines.append("| config | " + " | ".join(categories) + " |")
    lines.append("|---" * (len(categories) + 1) + "|")
    for row in ablation:
        cells = []
        for name in categories:
            entry = row["by_category"].get(name)
            cells.append(f"{entry['mrr']:.3f}" if entry else "-")
        lines.append(f"| `{row['config']}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Cells are MRR. Category counts: "
        + ", ".join(
            f"{name}={ablation[0]['by_category'][name]['n']}"
            for name in categories
            if name in ablation[0]["by_category"]
        )
    )

    if end_to_end:
        lines += [
            "",
            "## End-to-end",
            "",
            f"Config `{end_to_end['config']}` (best MRR). "
            f"Answering model `{end_to_end['answer_model']}`, "
            f"judge `{end_to_end['judge_model']}`.",
            "",
            f"**MemScore: {end_to_end['memscore']}**",
            "",
            "| metric | value |",
            "|---|---|",
            f"| accuracy (answerable) | {end_to_end['accuracy_answerable']:.1%} "
            f"(n={end_to_end['n_answerable']}) |",
            f"| adversarial decline rate | {end_to_end['adversarial_decline_rate']} "
            f"(n={end_to_end['n_adversarial']}) |",
            f"| retrieval hit@{k} | {end_to_end['retrieval_hit@k']:.3f} |",
            f"| mean search latency | {end_to_end['latency_ms_mean']:.1f} ms |",
            f"| mean context tokens | {end_to_end['context_tokens_mean']:.0f} |",
            f"| errors | {end_to_end['n_errors']} |",
            "",
            "Accuracy by category:",
            "",
            "| category | n | accuracy |",
            "|---|---|---|",
        ]
        for name, entry in end_to_end["by_category"].items():
            lines.append(f"| {name} | {entry['n']} | {entry['accuracy']:.1%} |")
        lines += [
            "",
            "The adversarial rate is a *decline* rate: the correct behaviour on "
            "an unanswerable question is to answer `NO_ANSWER`. It is scored "
            "without a judge and kept out of the headline accuracy, because a "
            "system that confidently answers the unanswerable should not be "
            "rewarded for it.",
        ]

    lines += [
        "",
        "## What this is and is not",
        "",
        "This is **not** a run of the `supermemoryai/memorybench` repository. "
        "That harness requires `bun` and hosted provider API keys "
        "(`SUPERMEMORY_API_KEY`, `MEM0_API_KEY`, `ZEP_API_KEY`), none of which "
        "were available. This re-implements its pipeline shape — ingest, index, "
        "search, answer, evaluate, report — against this service, with a "
        "different judge model.",
        "",
        "So these numbers are comparable **in kind** to a MemoryBench "
        "leaderboard, not head-to-head with it. A different judge, a different "
        "answering model, a question subset and a re-implemented prompt all move "
        "absolute accuracy.",
        "",
        "Every number here came from a real API response. Failed calls are "
        "counted in `n_errors` and excluded rather than substituted.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
