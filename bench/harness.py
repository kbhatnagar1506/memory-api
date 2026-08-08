"""Benchmark-agnostic pipeline: ingest -> search -> answer -> evaluate.

One harness, any `Dataset`. LoCoMo and LongMemEval differ in every practical
way — turn-level vs session-level evidence, one shared haystack vs one per
question, 400-character turns vs 12,000-character sessions — and the only thing
that varies here is which loader is passed in.

Reporting is per capability, never blended. A system can be excellent at recall
and dangerous at abstention; one number hides exactly that, and blended scores
are what discredited vendor benchmark marketing in this category.

Honesty constraints, same as the LoCoMo harness they were learned on:
  * Every number comes from a real API response. Failures are counted, never
    substituted.
  * Retrieval metrics are scored against labelled evidence, so they carry no
    judge variance. Only answer accuracy uses a judge.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from supermemory.domain.chunking import chunk_text, estimate_tokens
from supermemory.domain.embeddings.base import EmbeddingProvider
from supermemory.domain.models import Chunk, Memory, Organization, Space
from supermemory.domain.retrieval.pipeline import RetrievalPipeline, SearchRequest
from supermemory.domain.retrieval.rerank import Reranker
from supermemory.store.memory import InMemoryStore

from .cache import DiskVectorCache, cache_key
from .datasets.base import Corpus, Question
from .metrics import full_recall_at_k, hit_at_k, mrr, ndcg_at_k, recall_at_k, wilson


@dataclass
class Ingested:
    store: InMemoryStore
    org_id: str
    #: corpus_id -> space_id
    spaces: dict[str, str] = field(default_factory=dict)
    #: memory_id -> the dataset's document id, so hits map back to evidence
    doc_by_memory: dict[str, str] = field(default_factory=dict)
    #: corpus_id -> ordered (date, text) documents, for context assembly
    docs_by_corpus: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    #: corpus_id -> {doc_id: position}
    index_by_corpus: dict[str, dict[str, int]] = field(default_factory=dict)
    speakers: tuple[str, ...] = ()
    documents: int = 0
    chunks: int = 0
    embed_seconds: float = 0.0


def _token_batches(texts: list[str], *, max_items: int, max_tokens: int) -> list[list[int]]:
    """Group indices into requests bounded by BOTH count and token budget.

    Embedding APIs cap tokens per request, not just items. LongMemEval sessions
    run to ~3,000 tokens each, so a 32-item batch is ~96,000 tokens against a
    20,000 limit — the run died on exactly that. Batching by count alone is
    only safe when documents are uniformly small, which is a property of the
    corpus, not of the code.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    budget = 0
    for index, text in enumerate(texts):
        cost = estimate_tokens(text)
        if current and (len(current) >= max_items or budget + cost > max_tokens):
            batches.append(current)
            current, budget = [], 0
        current.append(index)
        budget += cost
    if current:
        batches.append(current)
    return batches


async def ingest(
    corpora: list[Corpus],
    embedder: EmbeddingProvider,
    *,
    batch_size: int = 32,
    concurrency: int = 6,
    chunk_target_tokens: int = 320,
    chunk_overlap_tokens: int = 48,
    max_request_tokens: int = 15_000,
    progress_every: int = 25,
    cache: DiskVectorCache | None = None,
) -> Ingested:
    """Bulk-load documents, chunked and embedded in token-bounded batches.

    Documents are chunked with the service's own chunker rather than embedded
    whole: a LongMemEval session is ~12,000 characters, far past any embedding
    model's per-input limit, and truncating it would silently drop the evidence
    we are about to score against. One memory per document, N chunks inside it
    — so evidence still maps at document granularity while retrieval matches
    the passage that actually contains the answer.

    Deliberately bypasses MemoryService.ingest, which embeds per memory: at
    this scale that is tens of thousands of single-text calls. The stored
    records are identical; only the batching differs. Dedup is off because
    collapsing near-identical documents would destroy evidence labels.
    """
    store = InMemoryStore()
    org = await store.create_organization(Organization(name="bench"))
    out = Ingested(store=store, org_id=org.id)
    semaphore = asyncio.Semaphore(concurrency)
    speakers: set[str] = set()

    async def embed_batch(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return (await embedder.embed(batch)).vectors

    started_all = time.perf_counter()
    for position, corpus in enumerate(corpora, start=1):
        if not corpus.documents:
            continue
        # Progress is not cosmetic. A 500-corpus run once hung silently for two
        # hours because ingestion printed nothing until it finished; a stalled
        # run and a slow one looked identical.
        if progress_every and (position == 1 or position % progress_every == 0):
            elapsed = time.perf_counter() - started_all
            rate = position / elapsed if elapsed > 0 else 0
            eta = (len(corpora) - position) / rate if rate > 0 else 0
            print(
                f"    [{position}/{len(corpora)}] {out.chunks} chunks, "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                flush=True,
            )
        space = await store.create_space(
            Space(
                org_id=org.id,
                slug=_slug(corpus.corpus_id, len(out.spaces)),
                name=corpus.corpus_id[:200],
            )
        )
        out.spaces[corpus.corpus_id] = space.id
        out.docs_by_corpus[corpus.corpus_id] = [
            (d.occurred_at.date().isoformat(), d.text) for d in corpus.documents
        ]
        out.index_by_corpus[corpus.corpus_id] = {
            d.id: i for i, d in enumerate(corpus.documents)
        }
        speakers.update(d.speaker for d in corpus.documents if d.speaker)

        # Chunk first, then batch by token budget across the flattened chunks.
        pieces: list[str] = []
        owner: list[int] = []
        for position, document in enumerate(corpus.documents):
            parts = (
                chunk_text(
                    document.text,
                    target_tokens=chunk_target_tokens,
                    overlap_tokens=chunk_overlap_tokens,
                )
                or []
            )
            texts_for_doc = [p.text for p in parts] or [document.text[:2000]]
            for text in texts_for_doc:
                pieces.append(text)
                owner.append(position)

        started = time.perf_counter()
        vectors: list[list[float]] = [None] * len(pieces)  # type: ignore[list-item]

        # Content-addressed cache first: re-running the same corpus with a
        # changed ANSWER path should cost nothing on the embedding side.
        pending = list(range(len(pieces)))
        if cache is not None:
            keys = [cache_key(embedder.model, embedder.dimensions, t) for t in pieces]
            found = cache.get_many(keys)
            pending = []
            for index, key in enumerate(keys):
                hit = found.get(key)
                if hit is not None:
                    vectors[index] = hit
                else:
                    pending.append(index)

        if pending:
            batches = _token_batches(
                [pieces[i] for i in pending],
                max_items=batch_size,
                max_tokens=max_request_tokens,
            )
            results = await asyncio.gather(
                *(embed_batch([pieces[pending[i]] for i in batch]) for batch in batches)
            )
            fresh: list[tuple[str, list[float]]] = []
            for batch, batch_vectors in zip(batches, results, strict=True):
                for local_index, vector in zip(batch, batch_vectors, strict=True):
                    index = pending[local_index]
                    vectors[index] = vector
                    if cache is not None:
                        fresh.append(
                            (
                                cache_key(embedder.model, embedder.dimensions, pieces[index]),
                                vector,
                            )
                        )
            if cache is not None and fresh:
                cache.put_many(fresh)
        out.embed_seconds += time.perf_counter() - started

        chunks_by_doc: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
        for piece, position, vector in zip(pieces, owner, vectors, strict=True):
            chunks_by_doc[position].append((piece, vector))

        for position, document in enumerate(corpus.documents):
            parts = chunks_by_doc.get(position, [])
            if not parts:
                continue
            memory = Memory(
                org_id=org.id,
                space_id=space.id,
                content=document.text,
                source=document.speaker,
                occurred_at=document.occurred_at,
                metadata={"doc_id": document.id, **document.metadata},
            )
            memory = memory.model_copy(
                update={
                    "chunks": [
                        Chunk(
                            memory_id=memory.id,
                            ordinal=ordinal,
                            text=text,
                            embedding=vector,
                        )
                        for ordinal, (text, vector) in enumerate(parts)
                    ]
                }
            )
            await store.upsert_memory(memory)
            out.doc_by_memory[memory.id] = document.id
            out.documents += 1
            out.chunks += len(parts)

    out.speakers = tuple(sorted(speakers))
    return out


def _slug(corpus_id: str, index: int) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in corpus_id.lower()).strip("-")
    return (cleaned or f"c{index}")[:60] or f"c{index}"


# -- retrieval -----------------------------------------------------------------


async def evaluate_retrieval(
    corpora: list[Corpus],
    ingested: Ingested,
    embedder: EmbeddingProvider,
    reranker: Reranker,
    *,
    k: int,
    config: dict[str, Any],
    concurrency: int = 8,
) -> dict[str, Any]:
    """Score retrieval against labelled evidence. No LLM, so no judge variance."""
    pipeline = RetrievalPipeline(ingested.store, embedder, reranker)
    questions = [q for c in corpora for q in c.questions]
    semaphore = asyncio.Semaphore(concurrency)

    async def one(question: Question) -> dict[str, Any] | None:
        space_id = ingested.spaces.get(question.corpus_id)
        if space_id is None or not question.evidence_ids:
            return None
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await pipeline.search(
                    SearchRequest(
                        query=question.text,
                        org_id=ingested.org_id,
                        space_id=space_id,
                        limit=k,
                        known_speakers=ingested.speakers,
                        **config,
                    )
                )
            except Exception as exc:
                return {"qid": question.qid, "error": f"{type(exc).__name__}: {exc}"[:200]}
            elapsed = (time.perf_counter() - started) * 1000

        retrieved = [ingested.doc_by_memory.get(h.memory.id, "") for h in response.results]
        relevant = set(question.evidence_ids)
        return {
            "qid": question.qid,
            "category": question.category,
            "n_evidence": len(relevant),
            "full_recall@k": full_recall_at_k(retrieved, relevant, k),
            "recall@k": recall_at_k(retrieved, relevant, k),
            "hit@k": hit_at_k(retrieved, relevant, k),
            "mrr": mrr(retrieved, relevant),
            "ndcg@k": ndcg_at_k(retrieved, relevant, k),
            "latency_ms": elapsed,
            "strategies": response.strategies,
            "entities_used": response.entities_used[:6],
        }

    rows = [r for r in await asyncio.gather(*(one(q) for q in questions)) if r]
    scored = [r for r in rows if "error" not in r]
    failures = [r for r in rows if "error" in r]
    return _summarize_retrieval(scored, failures, k, config)


def _summarize_retrieval(
    scored: list[dict], failures: list[dict], k: int, config: dict
) -> dict[str, Any]:
    def mean(key: str, rows: list[dict]) -> float:
        values = [r[key] for r in rows]
        return round(statistics.fmean(values), 4) if values else 0.0

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        grouped[row["category"]].append(row)
    for name, rows in sorted(grouped.items()):
        successes = round(mean("full_recall@k", rows) * len(rows))
        lo, hi = wilson(successes, len(rows))
        by_category[name] = {
            "n": len(rows),
            "avg_evidence": round(mean("n_evidence", rows), 2),
            "full_recall@k": mean("full_recall@k", rows),
            "full_recall_ci": [round(lo, 4), round(hi, 4)],
            "hit@k": mean("hit@k", rows),
            "mrr": mean("mrr", rows),
        }

    return {
        "config": config,
        "k": k,
        "n_scored": len(scored),
        "n_failed": len(failures),
        # The headline. hit@k is reported alongside because it is what most
        # published numbers use, and the gap between them is the story.
        "full_recall@k": mean("full_recall@k", scored),
        "recall@k": mean("recall@k", scored),
        "hit@k": mean("hit@k", scored),
        "mrr": mean("mrr", scored),
        "ndcg@k": mean("ndcg@k", scored),
        "latency_ms_mean": mean("latency_ms", scored),
        "by_category": by_category,
        "errors": failures[:10],
    }


__all__ = ["Ingested", "evaluate_retrieval", "ingest"]


# -- answer + judge ------------------------------------------------------------
#
# LongMemEval's published metric is LLM-judged answer accuracy, not retrieval
# recall, so this is the path that produces numbers comparable in kind to
# vendor leaderboards. Retrieval metrics above remain the trustworthy ones:
# they carry no judge variance.

ANSWER_PROMPT = """\
You are answering a question using excerpts retrieved from a long history of \
conversations between a user and an assistant. Each excerpt is prefixed with \
the date it happened.

Excerpts:
{context}

Question: {question}

Read the excerpts carefully and answer.

- The answer is usually present somewhere in these excerpts, often in an \
aside rather than the obvious place. Search all of them before concluding \
it is absent.
- Combine facts across excerpts when the answer needs more than one.
- Excerpts are ordered oldest to newest. If a fact CHANGED, the answer is the \
value from the most recent excerpt that mentions it.
- Answer with the specific value asked for (a name, place, date, number), not \
a description of where it came from.
- Reserve NO_ANSWER for when the information is genuinely absent. Do not use \
it because the answer is stated indirectly or requires a small inference \
across excerpts.

Reply in exactly this form:
FACTS: <the relevant facts, or "none">
ANSWER: <the answer in as few words as possible, or NO_ANSWER>"""

JUDGE_PROMPT = """\
You are grading a question-answering system against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {prediction}

Mark CORRECT if the system answer conveys the same fact as the reference, \
allowing differences in wording, formatting or extra detail. Dates may be \
written differently but must refer to the same date. Numbers must match. \
Mark INCORRECT otherwise.

Reply with exactly one word: CORRECT or INCORRECT"""


def parse_answer(raw: str) -> str:
    """Pull the final answer out of a FACTS/ANSWER response.

    Falls back to the last non-empty line: a model that ignores the format has
    usually still put its answer last, and scoring a formatting slip as a wrong
    answer would understate accuracy. A response that is only FACTS with no
    ANSWER line yields "" rather than the fact list, which would otherwise be
    graded as if it were an answer.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("ANSWER:"):
            return stripped[len("ANSWER:") :].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[-1].upper().startswith("FACTS:"):
        return ""
    return lines[-1] if lines else ""


class Gemini:
    """Thin Vertex client for the answer and judge phases."""

    def __init__(self, model: str, project: str | None) -> None:
        from google import genai

        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for answer/judge")
        from google.genai import types as genai_types

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location="global",
            http_options=genai_types.HttpOptions(timeout=120_000),
        )
        self.model = model

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> tuple[str, int]:
        """Native async. The thread-pool version deadlocked a 500-corpus run:
        `asyncio.to_thread` cannot be cancelled, so each timeout burned an
        executor slot until none were left."""
        from google.genai import types

        async def call() -> tuple[str, int]:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # Reasoning tokens share the answer budget on thinking
                    # models; without headroom the response is truncated before
                    # it reaches "ANSWER:".
                    max_output_tokens=max_tokens + 640,
                    thinking_config=types.ThinkingConfig(thinking_budget=128),
                ),
            )
            usage = response.usage_metadata
            return (response.text or ""), int(getattr(usage, "prompt_token_count", 0) or 0)

        return await call()


async def evaluate_end_to_end(
    corpora: list[Corpus],
    ingested: Ingested,
    embedder: EmbeddingProvider,
    reranker: Reranker,
    *,
    k: int,
    config: dict[str, Any],
    answer_model: str,
    judge_model: str,
    project: str | None,
    concurrency: int = 8,
    max_session_chars: int = 12_000,
) -> dict[str, Any]:
    pipeline = RetrievalPipeline(ingested.store, embedder, reranker)
    answerer = Gemini(answer_model, project)
    judge = Gemini(judge_model, project)
    semaphore = asyncio.Semaphore(concurrency)
    questions = [q for c in corpora for q in c.questions]

    async def one(question: Question) -> dict[str, Any]:
        space_id = ingested.spaces.get(question.corpus_id)
        if space_id is None:
            return {"qid": question.qid, "error": "no space"}
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await pipeline.search(
                    SearchRequest(
                        query=question.text,
                        org_id=ingested.org_id,
                        space_id=space_id,
                        limit=k,
                        known_speakers=ingested.speakers,
                        **config,
                    )
                )
            except Exception as exc:
                return {"qid": question.qid, "error": f"search: {exc}"[:200]}
            search_ms = (time.perf_counter() - started) * 1000

            # Pass the WHOLE retrieved session, not the chunk that matched.
            # Chunks exist to fit the embedding model's input limit; they are an
            # indexing device. Reading only the matched chunk handed the model
            # ~9% of each session and caused 186 of 216 failures: the answer was
            # in a session we had already retrieved, just in a different chunk.
            #
            # Ordered oldest-first by event date so "which value is current" is
            # answerable positionally. That matters most for knowledge-update
            # and temporal-reasoning, where retrieval is already at 1.000/0.910
            # and the only remaining question is which of several values wins.
            ordered = sorted(response.results, key=lambda h: h.memory.occurred_at)
            context = "\n\n---\n\n".join(
                f"[{hit.memory.occurred_at.date().isoformat()}]\n"
                f"{hit.memory.content[:max_session_chars]}"
                for hit in ordered
            )
            try:
                raw, context_tokens = await answerer.complete(
                    ANSWER_PROMPT.format(context=context, question=question.text),
                    max_tokens=256,
                )
            except Exception as exc:
                return {"qid": question.qid, "error": f"answer: {exc}"[:200]}
            prediction = parse_answer(raw)
            declined = prediction.upper().startswith("NO_ANSWER")

            if question.is_abstention:
                # Correct behaviour on an unanswerable question is to decline.
                # Scored without a judge, so no judge variance enters here.
                correct, verdict = declined, "DECLINED" if declined else "ANSWERED"
            elif declined:
                correct, verdict = False, "DECLINED"
            else:
                try:
                    grade, _ = await judge.complete(
                        JUDGE_PROMPT.format(
                            question=question.text,
                            reference=question.answer,
                            prediction=prediction,
                        ),
                        max_tokens=8,
                    )
                except Exception as exc:
                    return {"qid": question.qid, "error": f"judge: {exc}"[:200]}
                upper = grade.upper()
                verdict = (
                    "CORRECT"
                    if "CORRECT" in upper and "INCORRECT" not in upper
                    else "INCORRECT"
                )
                correct = verdict == "CORRECT"

            retrieved = [ingested.doc_by_memory.get(h.memory.id, "") for h in response.results]
            return {
                "qid": question.qid,
                "category": question.category,
                "is_abstention": question.is_abstention,
                "question": question.text[:300],
                "reference": question.answer[:300],
                "prediction": prediction[:300],
                "verdict": verdict,
                "correct": correct,
                "search_ms": round(search_ms, 2),
                "context_tokens": context_tokens,
                "full_recall@k": full_recall_at_k(retrieved, set(question.evidence_ids), k)
                if question.evidence_ids
                else None,
            }

    rows = await asyncio.gather(*(one(q) for q in questions))
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    answerable = [r for r in ok if not r["is_abstention"]]
    abstention = [r for r in ok if r["is_abstention"]]

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ok:
        grouped[row["category"]].append(row)
    for name, group in sorted(grouped.items()):
        hits = sum(1 for r in group if r["correct"])
        lo, hi = wilson(hits, len(group))
        by_category[name] = {
            "n": len(group),
            "accuracy": round(hits / len(group), 4),
            "ci": [round(lo, 4), round(hi, 4)],
        }

    correct = sum(1 for r in answerable if r["correct"])
    total = len(answerable)
    lo, hi = wilson(correct, total) if total else (0.0, 0.0)
    latencies = [r["search_ms"] for r in ok]
    tokens = [r["context_tokens"] for r in ok if r["context_tokens"]]
    return {
        "config": config,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "n_attempted": len(rows),
        "n_scored": len(ok),
        "n_errors": len(errors),
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_ci": [round(lo, 4), round(hi, 4)],
        "n_answerable": total,
        "abstention_rate": round(
            sum(1 for r in abstention if r["correct"]) / len(abstention), 4
        )
        if abstention
        else None,
        "n_abstention": len(abstention),
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "context_tokens_mean": round(statistics.fmean(tokens), 1) if tokens else 0.0,
        "by_category": by_category,
        "errors": errors[:10],
        "per_question": ok,
    }


__all__ += ["evaluate_end_to_end", "parse_answer"]
