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
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mapi.domain.chunking import chunk_text, estimate_tokens
from mapi.domain.embeddings.base import EmbeddingProvider
from mapi.domain.models import Chunk, Memory, Organization, Space
from mapi.domain.retrieval.pipeline import RetrievalPipeline, SearchRequest
from mapi.domain.retrieval.rerank import Reranker
from mapi.domain.synthesis import QuestionKind, classify
from mapi.domain.synthesis.derive import SourceDoc, derive_answer
from mapi.domain.synthesis.extract import DEFAULT_SUBJECT, extract_claims
from mapi.store.memory import InMemoryStore

from .cache import DiskClaimCache, DiskVectorCache, cache_key
from .datasets.base import Corpus, Document, Question
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
    #: Stored units that are extracted claims rather than source documents.
    claims: int = 0
    embed_seconds: float = 0.0
    extract_seconds: float = 0.0


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


#: Documents per durable checkpoint. Large enough that the cache write is
#: negligible against the model calls it protects, small enough that a crash
#: costs at most this many round-trips.
_EXTRACT_CHECKPOINT = 500

#: LongMemEval sessions are rendered "role: text" by the loader.
_TURN_RE = re.compile(r"(?:^|\s)(user|assistant):\s")


def split_turns(text: str) -> list[tuple[str, str]]:
    """A session transcript as (speaker, text) turns, or one unlabelled turn.

    Extraction is run per TURN, not per session, and the reason is measured.
    On a 20,107-character session the model returned 9 claims, four of them
    restatements of one topic near the end -- classic long-context recency,
    and far short of what the transcript states. The same session split into
    its 6 user turns returned 6 claims with no redundancy.

    The count going DOWN is the point. Three of the nine were grounded in the
    ASSISTANT's text, which passes the quote check (the assistant's words are
    in the passage) while asserting the assistant's suggestions as the user's
    facts. Splitting by speaker makes that structurally impossible: text the
    user never said is never in the prompt that produces a fact about them.

    Both roles are extracted, with the speaker as the subject. Dropping the
    assistant's turns would be cheaper and would silently gut LongMemEval's
    `single-session-assistant` capability, which asks what the assistant said.
    """
    parts = _TURN_RE.split(text)
    if len(parts) < 3:
        return [("", text)]
    roles, bodies = parts[1::2], parts[2::2]
    return [
        (role, body.strip()) for role, body in zip(roles, bodies, strict=False) if body.strip()
    ]


async def extract_corpora(
    corpora: list[Corpus],
    extractor: Any,
    *,
    concurrency: int = 24,
    cache: DiskClaimCache | None = None,
    model: str = "",
    progress_every: int = 200,
) -> tuple[dict[str, list[tuple[str, str]]], float]:
    """Decompose every document into atomic claims, before any ingestion.

    A separate PASS rather than a step inside `ingest`, for one reason:
    extraction depends on nothing but the document it reads, so all of it can
    be in flight at once, while ingestion cannot be reordered freely. Run
    serially inside ingest, a 25,000-session corpus is a day of round-trips;
    run here it is bounded by `concurrency`.

    Deduplicated by document TEXT, not by id. LongMemEval reuses the same
    filler sessions across many corpora, so the same passage appears in
    dozens of haystacks and extracting it once is worth more than any
    batching.
    """
    # Keyed by (text, date), not text alone. LongMemEval reuses the same
    # filler session across many haystacks, so deduping by text is worth a lot
    # -- but the SAME text dated differently must extract separately, because
    # the date is what "last Thursday" resolves against.
    unique: dict[tuple[str, datetime], None] = {}
    for corpus in corpora:
        for document in corpus.documents:
            unique.setdefault((document.text, document.occurred_at), None)
    items = list(unique)

    out: dict[str, list[tuple[str, str]]] = {}
    pending = items
    if cache is not None:
        keys = [DiskClaimCache.key(model, f"{when.isoformat()}|{t}") for t, when in items]
        found = cache.get_many(keys)
        pending = []
        for (text, when), key in zip(items, keys, strict=True):
            hit = found.get(key)
            if hit is None:
                pending.append((text, when))
            else:
                out[text] = hit
        print(
            f"    claim cache: {len(out)} hits, {len(pending)} to extract",
            flush=True,
        )

    started = time.perf_counter()
    if pending:
        gate = asyncio.Semaphore(concurrency)
        done = 0
        failures = 0

        async def one_turn(speaker: str, body: str, when: datetime) -> list[tuple[str, str]]:
            nonlocal failures
            if speaker == "assistant":
                # Kept VERBATIM, not decomposed. Measured: asking for "durable
                # facts stated by the assistant" over a recommendation list
                # returns nothing at all -- the frame does not fit, because an
                # assistant turn is content, not claims about a person. Storing
                # it whole is also what LongMemEval's single-session-assistant
                # capability needs: those questions ask what the assistant
                # SAID, so a lossy restatement is the wrong unit even when
                # extraction does produce one.
                return [(body, body)]
            subject = {"user": "the user"}.get(speaker, DEFAULT_SUBJECT)
            async with gate:
                try:
                    # `as_of` is the DOCUMENT's date, not today's. Without it
                    # the prompt resolves "last Thursday" against the wall
                    # clock, which on a 2023 corpus writes 2026 dates into
                    # memory -- a fabricated fact, not a missing one.
                    claims = await extract_claims(body, extractor, as_of=when, subject=subject)
                except Exception:
                    # `extract_claims` already fails open, so reaching here
                    # means something outside it broke. Counted, never
                    # silently folded into "this document had no facts".
                    failures += 1
                    claims = []
            return [(c.fact, c.quote) for c in claims]

        async def one(text: str, when: datetime) -> tuple[str, list[tuple[str, str]]]:
            nonlocal done
            groups = await asyncio.gather(
                *(one_turn(role, body, when) for role, body in split_turns(text))
            )
            claims = [c for group in groups for c in group]
            done += 1
            if progress_every and done % progress_every == 0:
                rate = done / max(time.perf_counter() - started, 1e-9)
                left = (len(pending) - done) / max(rate, 1e-9) / 60
                print(
                    f"    extracted {done}/{len(pending)} docs "
                    f"({rate:.1f}/s, ~{left:.0f}m left)",
                    flush=True,
                )
            return text, claims

        # Checkpointed, not one gather over everything. A 20,000-document
        # extraction is ~40 minutes of model calls, and writing the cache only
        # at the end means any failure in that window throws all of it away.
        # Each slice is durable as soon as it lands, so a re-run resumes.
        for start in range(0, len(pending), _EXTRACT_CHECKPOINT):
            window = pending[start : start + _EXTRACT_CHECKPOINT]
            results = await asyncio.gather(*(one(t, w) for t, w in window))
            out.update(dict(results))
            if cache is not None:
                keyed = zip(window, results, strict=True)
                cache.put_many(
                    (
                        DiskClaimCache.key(model, f"{when.isoformat()}|{text}"),
                        claims,
                    )
                    for (text, when), (_t, claims) in keyed
                )
        if failures:
            print(f"    WARNING: {failures} extraction calls failed outright", flush=True)

    total = sum(len(v) for v in out.values())
    empty = sum(1 for v in out.values() if not v)
    print(
        f"    {total} claims from {len(items)} documents "
        f"({empty} yielded none) in {time.perf_counter() - started:.0f}s",
        flush=True,
    )
    return out, time.perf_counter() - started


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
    claims_by_doc: dict[str, list[tuple[str, str]]] | None = None,
    extract_mode: str = "off",
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

        # What actually gets stored, in order. Normally one unit per document;
        # with extraction on, also one per claim, each still owned by the
        # document it came from so evidence keeps mapping at the granularity
        # the benchmark labels.
        units: list[tuple[str, Document]] = []
        for document in corpus.documents:
            found = (claims_by_doc or {}).get(document.text) or []
            if extract_mode != "only" or not found:
                # "only" still stores the original when extraction returned
                # nothing: dropping the document would delete evidence the
                # benchmark is about to score against.
                units.append((document.text, document))
            for fact, _quote in found:
                units.append((fact, document))

        # Chunk first, then batch by token budget across the flattened chunks.
        pieces: list[str] = []
        owner: list[int] = []
        for position, (text_to_store, _document) in enumerate(units):
            parts = (
                chunk_text(
                    text_to_store,
                    target_tokens=chunk_target_tokens,
                    overlap_tokens=chunk_overlap_tokens,
                )
                or []
            )
            texts_for_doc = [p.text for p in parts] or [text_to_store[:2000]]
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

        for position, (text_to_store, document) in enumerate(units):
            parts = chunks_by_doc.get(position, [])
            if not parts:
                continue
            memory = Memory(
                org_id=org.id,
                space_id=space.id,
                content=text_to_store,
                source=document.speaker,
                occurred_at=document.occurred_at,
                # `doc_id` is the SOURCE document even for a claim. A claim is
                # a statement that document made, so retrieving it is
                # retrieving that session -- which is the granularity
                # LongMemEval labels evidence at. Anything else would score a
                # correct hit as a miss.
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
            if text_to_store != document.text:
                out.claims += 1

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

Today's date is {asked_at}.

Excerpts:
{context}

Question: {question}

Read the excerpts carefully and answer.

- "How long ago", "how many days/weeks/months ago" and "since" are measured \
from TODAY'S DATE above, not from the excerpt's own date. Compute the \
difference explicitly.

- The answer is usually present somewhere in these excerpts, often in an \
aside rather than the obvious place. Search all of them before concluding \
it is absent.
- Combine facts across excerpts when the answer needs more than one.
- Excerpts are ordered oldest to newest. If a fact CHANGED, the answer is the \
value from the most recent excerpt that mentions it.
- Answer with the specific value asked for (a name, place, date, number), not \
a description of where it came from.
- Two different reasons to withhold, and only one of them is right. If the \
answer is stated indirectly, or needs a small inference across excerpts, \
ANSWER it. If the excerpts genuinely do not contain what was asked, reply \
NO_ANSWER.

Reply in exactly this form:
FACTS: <the relevant facts, or "none">
ANSWER: <the answer in as few words as possible, or NO_ANSWER>"""

#: Advice questions are a different task wearing the same clothes. "Can you
#: suggest a hotel for my Miami trip?" has no stored answer to look up — the
#: job is to USE what the history says the user likes. Under the fact-lookup
#: prompt above the model searched for a hotel it had been told about, found
#: none, and returned NO_ANSWER on a question whose whole point is
#: personalisation. Measured: single-session-preference sat at 26.7% while its
#: retrieval was a perfect 1.000, and its siblings scored 95-98%.
#:
#: This is not benchmark shaping. A memory system that answers "suggest a
#: hotel" with NO_ANSWER while holding the user's stated love of rooftop pools
#: is failing at the only thing memory is for.
ADVICE_PROMPT = """\
You are advising a user, drawing on excerpts from your history of \
conversations with them. Each excerpt is prefixed with the date it happened.

Today's date is {asked_at}.

Excerpts:
{context}

Request: {question}

The excerpts will NOT contain a ready-made answer — they contain what this \
user likes, owns, does and cares about. Your job is to make a recommendation \
that visibly reflects those preferences.

- Mine the excerpts for the user's tastes, constraints, brands, skills and \
past choices relevant to this request.
- Give concrete suggestions, and make the connection to their preferences \
explicit ("Sony-compatible, since you shoot Sony").
- Never reply NO_ANSWER. A recommendation is always possible from stated \
preferences; refusing is the failure mode here.
- Two to four sentences.

Reply in exactly this form:
FACTS: <the user preferences you are drawing on>
ANSWER: <your recommendation>"""

#: The official judge asks for "yes or no only", and the reflex is to cap output
#: at ~8 tokens to match. That silently breaks thinking models: measured here,
#: gemini-2.5-pro spent 19 tokens reasoning before its first output token and
#: returned a truncated 'N' at max_tokens=24 -- which parses as "no" and would
#: have scored every answer wrong while looking like a working grader. The
#: verdict is still one word; the budget is headroom for the thinking that
#: precedes it.
_JUDGE_MAX_TOKENS = 800

#: Our own judge. Kept ONLY as the strict comparison arm — it is not the
#: benchmark's metric, and using it as the headline made our numbers
#: incomparable to every published result. See OFFICIAL_JUDGE below.
STRICT_JUDGE_PROMPT = """\
You are grading a question-answering system against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {prediction}

Mark CORRECT if the system answer conveys the same fact as the reference, \
allowing differences in wording, formatting or extra detail. Dates may be \
written differently but must refer to the same date. Numbers must match. \
Mark INCORRECT otherwise.

Reply with exactly one word: CORRECT or INCORRECT"""

#: LongMemEval's official judge prompts, verbatim from the benchmark's own
#: `src/evaluation/evaluate_qa.py` (`get_anscheck_prompt`). Reproduced exactly,
#: including the per-question-type wording, because paraphrasing a judge is
#: paraphrasing the metric.
#:
#: We ran a homegrown judge first and it was systematically STRICTER than the
#: benchmark on three of our four weakest categories:
#:
#:   * single-session-preference -- the `answer` field is a 391-char RUBRIC, not
#:     an answer (every other type averages 20-44 chars). Grading it with
#:     "conveys the same fact as the reference" is a category error: a short
#:     factual reply cannot convey the same fact as a preference rubric. This
#:     alone accounted for our worst score (16.7%) with no memory failure.
#:   * temporal-reasoning -- gold answers literally read "7 days. 8 days
#:     (including the last day) is also acceptable", and the official prompt
#:     says not to penalise off-by-one day counts. Ours said "Numbers must
#:     match", overriding the dataset's own stated tolerance on 133 questions.
#:   * knowledge-update -- restating the superseded value alongside the updated
#:     one is explicitly correct here; our prompt gave no such allowance.
_OFFICIAL_FACTUAL = (
    "I will give you a question, a correct answer, and a response from a "
    "model. Please answer yes if the response contains the correct answer. "
    "Otherwise, answer no. If the response is equivalent to the correct answer "
    "or contains all the intermediate steps to get the correct answer, you "
    "should also answer yes. If the response only contains a subset of the "
    "information required by the answer, answer no. "
)
_OFFICIAL_TAIL = (
    "\n\nQuestion: {question}\n\nCorrect Answer: {reference}\n\n"
    "Model Response: {prediction}\n\nIs the model response correct? "
    "Answer yes or no only."
)

OFFICIAL_JUDGE_PROMPTS: dict[str, str] = {
    "single-session-user": _OFFICIAL_FACTUAL + _OFFICIAL_TAIL,
    "single-session-assistant": _OFFICIAL_FACTUAL + _OFFICIAL_TAIL,
    "multi-session": _OFFICIAL_FACTUAL + _OFFICIAL_TAIL,
    "temporal-reasoning": _OFFICIAL_FACTUAL
    + "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the "
    "model makes off-by-one errors (e.g., predicting 19 days when the answer "
    "is 18), the model's response is still correct. " + _OFFICIAL_TAIL,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be "
        "considered as correct as long as the updated answer is the required "
        "answer." + _OFFICIAL_TAIL
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized "
        "response, and a response from a model. Please answer yes if the "
        "response satisfies the desired response. Otherwise, answer no. The "
        "model does not need to reflect all the points in the rubric. The "
        "response is correct as long as it recalls and utilizes the user's "
        "personal information correctly."
        "\n\nQuestion: {question}\n\nRubric: {reference}\n\n"
        "Model Response: {prediction}\n\nIs the model response correct? "
        "Answer yes or no only."
    ),
}

#: Abstention gets its own prompt: the reference field holds an EXPLANATION of
#: why the question is unanswerable, not an answer. We score abstention
#: judge-free (did the model decline?) and use this only as a fallback for
#: models that decline in prose without emitting the sentinel.
OFFICIAL_ABSTENTION_PROMPT = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {question}\n\nExplanation: {reference}\n\n"
    "Model Response: {prediction}\n\nDoes the model correctly identify the "
    "question as unanswerable? Answer yes or no only."
)


def judge_prompt(
    category: str, question: str, reference: str, prediction: str, *, official: bool
) -> str:
    """Build the grading prompt for one question.

    Unknown categories fall back to the generic factual template rather than
    raising: a new question type should score, not crash a four-hour run.
    """
    if not official:
        return STRICT_JUDGE_PROMPT.format(
            question=question, reference=reference, prediction=prediction
        )
    template = OFFICIAL_JUDGE_PROMPTS.get(category, _OFFICIAL_FACTUAL + _OFFICIAL_TAIL)
    return template.format(question=question, reference=reference, prediction=prediction)


def parse_grade(raw: str, *, official: bool) -> bool:
    """Read a verdict. Official grammar is yes/no, ours is CORRECT/INCORRECT.

    `INCORRECT` contains `CORRECT` as a substring, so the strict arm must
    exclude it explicitly — checking for the positive word alone would grade
    every wrong answer as right.
    """
    text = raw.strip().lower()
    if official:
        return text.startswith("yes") or "yes" in text[:20]
    upper = raw.upper()
    return "CORRECT" in upper and "INCORRECT" not in upper


def _facts_block(raw: str) -> str:
    """The model's stated reasoning, for post-mortems.

    Recorded because every failure diagnosis in this project so far has been
    inferred from the final answer string alone -- which is how a context
    truncation bug looked like a reasoning failure for weeks. Truncated: this
    is a debugging aid, not a transcript.
    """
    for line in raw.splitlines():
        if line.strip().upper().startswith("FACTS:"):
            return line.split(":", 1)[1].strip()[:400]
    return ""


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

    def __init__(self, model: str, project: str | None, *, thinking_budget: int = 128) -> None:
        from google import genai

        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for answer/judge")
        #: Reasoning tokens the model may spend before its first output token.
        #: 128 is barely any: 19 of the 40 delivery-clean failures are
        #: multi-step chains, so this is a candidate binding constraint that
        #: has never been varied in this project.
        self.thinking_budget = thinking_budget
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
                    thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                ),
            )
            usage = response.usage_metadata
            return (response.text or ""), int(getattr(usage, "prompt_token_count", 0) or 0)

        return await call()


class AnthropicVertexJudge:
    """Claude on Vertex, for CROSS-FAMILY grading.

    A different capability tier of the same family (2.5-pro judging 2.5-flash)
    breaks self-recognition, but both models share a training lineage, so a
    shared blind spot -- a misread date format, a temporal expression parsed the
    same wrong way -- is still graded as correct by a grader that makes the
    identical mistake. A different vendor does not share that lineage, which is
    what makes it the strongest independence available to us.

    Reached through the same ADC as the Gemini path; needs the model enabled in
    Model Garden and pinned to a region that serves it.
    """

    #: Anthropic-on-Vertex model ids carry an @version suffix, and the model is
    #: regional. Getting either wrong returns an HTML 404 rather than a JSON API
    #: error, which reads like "not provisioned" even when it is.
    REGION = "us-east5"

    def __init__(self, model: str, project: str | None) -> None:
        from anthropic import AsyncAnthropicVertex

        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for answer/judge")
        # Async client, never the sync one under `to_thread`: that combination
        # deadlocked a 500-corpus run, because a thread blocked on a socket read
        # cannot be cancelled and each timeout permanently burned an executor
        # slot. Same failure mode would apply here verbatim.
        self.client = AsyncAnthropicVertex(
            project_id=project, region=self.REGION, timeout=120.0
        )
        self.model = model

    async def complete(self, prompt: str, *, max_tokens: int = 256) -> tuple[str, int]:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return text, int(getattr(message.usage, "input_tokens", 0) or 0)


#: Routed on the model id: anything Claude goes to Vertex's Anthropic publisher,
#: everything else to the Gemini client. Keeping this in one place means
#: `--judge-model` and `--answer-model` accept either family without the caller
#: knowing which SDK backs it.
def build_model_client(
    model: str, project: str | None, *, thinking_budget: int = 128
) -> Gemini | AnthropicVertexJudge:
    if model.startswith("claude"):
        return AnthropicVertexJudge(model, project)
    return Gemini(model, project, thinking_budget=thinking_budget)


#: Evidence budget by question SHAPE, not one number for everything.
#:
#: Measured on 500 questions, the number of gold sessions a question needs
#: varies 6x by shape and our label-free classifier predicts it:
#:
#:     advice      median 1, max 1      <- never needs more than one
#:     direct      median 1, p90 2, max 5
#:     count       median 2, p90 4, max 6
#:     order       median 2, p90 3, max 6
#:     date_arith  median 2, p90 2, max 6
#:
#: A single fixed k is therefore always the wrong number for someone. At k=10
#: the 213 single-evidence questions get nine sessions of pure distraction --
#: measured cost: single-session-user -2.8 going 4->10, and preference -16.7
#: going 10->20. At the same time multi-session gained +10.5 from the increase
#: and needs up to five. One budget cannot serve both, and every k sweep so
#: far has been finding the least-bad compromise between them.
#:
#: Deterministic and label-free: routed on the same classifier the derive path
#: uses, so the product behaves identically. Budgets sit above each shape's
#: observed maximum, not at its median, because retrieval is imperfect and a
#: budget that only fits the typical case starves the tail.
_EVIDENCE_BUDGET: dict[QuestionKind, int] = {
    QuestionKind.ADVICE: 5,
    QuestionKind.DIRECT: 6,
}
#: Everything else is a multi-evidence shape (count, order, span, compare,
#: list) and keeps the measured-best budget.
_DEFAULT_BUDGET = 12


def evidence_budget(question: str, *, cap: int) -> int:
    """RETIRED -- measured harmful. Kept only so the flag still resolves.

    Budgets denominated in DOCUMENTS cannot transfer: the constants were
    fitted where a document is a 12,000-char session, and on a corpus
    where a document is a 400-char turn the same numbers starved
    retrieval -- LoCoMo delivery 0.833 -> 0.586, accuracy 72.5% -> 50.1%.
    The successor is domain/retrieval/budget.py, which stops on the score
    cliff and so reads concentration off the result set instead of
    assuming a document size.
    """
    return min(_EVIDENCE_BUDGET.get(classify(question), _DEFAULT_BUDGET), cap)


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
    # 20k, not 12k: sweeping the cap against every gold turn's character
    # offset, 12,000 cuts the answer sentence out of 32 questions (11 of which
    # had ALL their gold sessions delivered and still failed), 16,000 cuts 8,
    # and 20,000 cuts none. Cost is +15% context (mean session 8,894 -> 10,225
    # chars). Truncating the sentence that holds the answer is the cheapest
    # possible way to lose a question.
    max_session_chars: int = 20_000,
    official_judge: bool = True,
    measure_judge_bias: bool = True,
    use_derive: bool = False,
    verbose: bool = True,
    thinking_budget: int = 128,
    dynamic_k: bool = False,
) -> dict[str, Any]:
    pipeline = RetrievalPipeline(ingested.store, embedder, reranker)
    answerer = build_model_client(answer_model, project, thinking_budget=thinking_budget)
    judge = build_model_client(judge_model, project)
    # Only a control arm when the judge is genuinely a different model. Pointing
    # both at the same model would "measure" a bias of exactly zero by
    # construction and read as evidence of impartiality.
    bias_judge = (
        build_model_client(answer_model, project)
        if measure_judge_bias and judge_model != answer_model
        else None
    )
    semaphore = asyncio.Semaphore(concurrency)
    questions = [q for c in corpora for q in c.questions]

    async def one(question: Question) -> dict[str, Any]:
        space_id = ingested.spaces.get(question.corpus_id)
        if space_id is None:
            return {"qid": question.qid, "error": "no space"}
        async with semaphore:
            started = time.perf_counter()
            # Per-question evidence budget, not one k for everyone. `k` is
            # the ceiling; the question's shape decides how much to use.
            per_question_k = evidence_budget(question.text, cap=k) if dynamic_k else k
            try:
                response = await pipeline.search(
                    SearchRequest(
                        query=question.text,
                        org_id=ingested.org_id,
                        space_id=space_id,
                        limit=per_question_k,
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
                # Full timestamp, not just the date: three questions have two
                # gold sessions on the SAME day, and a date-only header leaves
                # them positionally unorderable — the model cannot answer
                # "which happened first" from evidence it cannot sequence.
                f"[{hit.memory.occurred_at.isoformat(timespec='minutes')}]\n"
                f"{hit.memory.content[:max_session_chars]}"
                for hit in ordered
            )
            # Advice requests get the personalisation prompt; everything else
            # the fact-lookup prompt. Routed on the question's SHAPE via the
            # same classifier the derive path uses -- no dataset label is
            # consulted, so this behaves identically in the product.
            asked_at = question.asked_at.date().isoformat() if question.asked_at else "unknown"
            is_advice = classify(question.text) is QuestionKind.ADVICE
            template = ADVICE_PROMPT if is_advice else ANSWER_PROMPT
            try:
                raw, context_tokens = await answerer.complete(
                    template.format(context=context, question=question.text, asked_at=asked_at),
                    max_tokens=512 if is_advice else 256,
                )
            except Exception as exc:
                return {"qid": question.qid, "error": f"answer: {exc}"[:200]}
            prediction = parse_answer(raw)
            if not prediction.strip():
                # An empty completion is a transport failure wearing the mask
                # of a wrong answer: the model returned nothing, and scoring
                # that as "incorrect" attributes an API hiccup to the memory
                # system. One retry, then it counts honestly as whatever it is.
                try:
                    raw, context_tokens = await answerer.complete(
                        template.format(
                            context=context, question=question.text, asked_at=asked_at
                        ),
                        max_tokens=512 if is_advice else 256,
                    )
                    prediction = parse_answer(raw)
                except Exception:
                    pass
            declined = prediction.upper().startswith("NO_ANSWER")

            # -- derive path (opt-in, ablation-flagged) -----------------------
            # Two triggers, both measured on lme-full:
            #   * aggregate kinds (COUNT/ORDER/...): 45% of all failures with
            #     complete evidence were counts — "three bikes" answered
            #     "Multiple". Code-computed answers are PREFERRED over a
            #     non-declined direct answer for these kinds, because the
            #     direct model demonstrably miscounts with the evidence in
            #     hand. The flag exists so this policy is attributable.
            #   * declines: 126 questions declined while holding complete
            #     evidence. Escalate any decline through extract-then-compose.
            # Truly unanswerable questions stay safe through mechanism, not
            # special-casing: extraction finds no relevant rows, the table is
            # empty, derivation returns None, the decline stands.
            derive_kind = QuestionKind.DIRECT if is_advice else classify(question.text)
            derived_used = False
            if use_derive and (declined or derive_kind is not QuestionKind.DIRECT):
                docs = [
                    SourceDoc(
                        id=hit.memory.id,
                        text=hit.memory.content[:max_session_chars],
                        occurred_at=hit.memory.occurred_at,
                    )
                    for hit in ordered
                ]

                async def complete_text(prompt: str) -> str:
                    text, _ = await answerer.complete(prompt, max_tokens=1024)
                    return text

                derived = await derive_answer(
                    question.text,
                    derive_kind,
                    docs,
                    complete_text,
                    asked_at=question.asked_at.date() if question.asked_at else None,
                )
                # Derivation FILLS declines; it never overrides an answer.
                # The previous policy preferred any code-computed value over a
                # non-declined direct answer, reasoning that the direct model
                # demonstrably miscounts with evidence in hand. Measured on 500
                # questions that policy flipped 67 right answers wrong to gain
                # 30 -- net -37 -- and dragged the headline from ~66% to 45%.
                # A model that answers is evidence it found something;
                # overriding it requires a derivation whose accuracy has been
                # measured, and ours has not been (roadmap Phase 7).
                # A decline is the model's judgement that the evidence is
                # absent, made with the FULL sessions in view. Overriding it
                # requires evidence stronger than the model already had -- and
                # the compose path is strictly weaker: the same model, looking
                # at a table instead of the sessions.
                #
                # Measured on 500 questions: decline-filling fired 16 times,
                # gained 1 answerable question (and that one by luck -- it
                # answered "crystal chandelier" to a question about jewellery
                # and the judge accepted it on "aunt"), and lost 6 correct
                # abstentions. "How many years of formal education" became
                # "4979 days"; "how many fish in my 30-gallon tank" became
                # "the user's tank is a 20-gallon tank". Net -5.
                #
                # So the override is narrowed to the one reduce whose semantics
                # are unambiguous: a COUNT computed in code over at least two
                # independently grounded rows. Prose composed from a thin table
                # produced every one of the bad answers above, and a date span
                # between unrelated events produced the rest.
                safe_to_override = (
                    derived is not None
                    and derived.computed
                    and derived.kind is QuestionKind.COUNT
                    and len(derived.table) >= 2
                )
                if safe_to_override and derived is not None and declined and derived.answer:
                    prediction = derived.answer
                    declined = False
                    derived_used = True

            self_correct: bool | None = None
            if question.is_abstention:
                # Correct behaviour on an unanswerable question is to decline.
                # The sentinel path is judge-free, so no judge variance enters.
                # A model that declines in prose ("I don't have that") without
                # emitting the sentinel is still right, and the official
                # abstention prompt is the only way to catch that -- scoring it
                # ANSWERED would penalise phrasing, not memory.
                if declined:
                    correct, verdict = True, "DECLINED"
                else:
                    try:
                        grade, _ = await judge.complete(
                            OFFICIAL_ABSTENTION_PROMPT.format(
                                question=question.text,
                                reference=question.answer,
                                prediction=prediction,
                            ),
                            max_tokens=_JUDGE_MAX_TOKENS,
                        )
                    except Exception as exc:
                        return {"qid": question.qid, "error": f"judge: {exc}"[:200]}
                    correct = parse_grade(grade, official=True)
                    verdict = "DECLINED_PROSE" if correct else "ANSWERED"
            elif declined:
                correct, verdict = False, "DECLINED"
            else:
                prompt = judge_prompt(
                    question.category,
                    question.text,
                    question.answer,
                    prediction,
                    official=official_judge,
                )
                try:
                    grade, _ = await judge.complete(prompt, max_tokens=_JUDGE_MAX_TOKENS)
                except Exception as exc:
                    return {"qid": question.qid, "error": f"judge: {exc}"[:200]}
                correct = parse_grade(grade, official=official_judge)
                verdict = "CORRECT" if correct else "INCORRECT"

                # Self-grading control. The primary judge is a DIFFERENT model
                # from the answerer; this re-grades the same prediction with the
                # answerer itself. The gap between the two is a direct estimate
                # of self-preference bias on our own data, which turns "we grade
                # ourselves" from an unquantified caveat into a number.
                if bias_judge is not None:
                    try:
                        self_grade, _ = await bias_judge.complete(
                            prompt, max_tokens=_JUDGE_MAX_TOKENS
                        )
                        self_correct = parse_grade(self_grade, official=official_judge)
                    except Exception:
                        self_correct = None  # never fail a run over the control arm

            retrieved = [ingested.doc_by_memory.get(h.memory.id, "") for h in response.results]
            return {
                "qid": question.qid,
                "category": question.category,
                "is_abstention": question.is_abstention,
                # Stored untruncated so a run can be RE-GRADED offline without
                # repeating it. Retrieval and answering cost hours; judging is
                # seconds, so the judge must never be the reason to re-run.
                # These were capped at 300 chars, which silently cut
                # single-session-preference rubrics (mean 391) -- re-grading
                # would have used a truncated rubric on the most fragile
                # capability and looked like a memory failure.
                "question": question.text,
                "reference": question.answer,
                "prediction": prediction,
                # The FACTS block the model wrote before answering. No run in
                # this project had ever recorded WHY an answer was wrong --
                # every diagnosis was inferred from the final string, which is
                # how a truncation bug read as a reasoning failure for weeks.
                "facts": _facts_block(raw),
                "verdict": verdict,
                "correct": correct,
                "self_correct": self_correct,
                "derive_kind": str(derive_kind),
                "derived": derived_used,
                "search_ms": round(search_ms, 2),
                "context_tokens": context_tokens,
                # THE field whose absence let a delivery defect survive six
                # runs. `full_recall@k` here is measured at the ANSWER window
                # (k = answer_k), while the ablation table reports it at the
                # retrieval k. Those were two different numbers under one name:
                # 0.870 delivered vs 0.968 retrieved. Both are recorded now,
                # explicitly named, so the gap is impossible to overlook again.
                "answer_k": per_question_k,
                "full_recall@answer_k": full_recall_at_k(
                    retrieved, set(question.evidence_ids), per_question_k
                )
                if question.evidence_ids
                else None,
                "full_recall@k": full_recall_at_k(retrieved, set(question.evidence_ids), k)
                if question.evidence_ids
                else None,
            }

    # Live per-question output. The QA phase was a silent black box for its
    # whole duration -- the same shape as the ingestion stall that once looked
    # identical to a hung run for two hours. Streaming the verdict as each
    # question resolves makes a slow run distinguishable from a dead one, and
    # makes systematic wrongness visible while it is happening rather than in
    # a post-mortem. `flush=True` because stdout is block-buffered when
    # redirected to a file, which is exactly how the earlier silence happened.
    done = 0
    total = len(questions)

    async def one_verbose(question: Question) -> dict[str, Any]:
        nonlocal done
        row = await one(question)
        done += 1
        if verbose:
            if "error" in row:
                mark, detail = "ERROR  ", str(row["error"])[:60]
            else:
                mark = {
                    "CORRECT": "ok     ",
                    "DECLINED": "decline",
                    "DECLINED_PROSE": "decline",
                }.get(str(row["verdict"]), "WRONG  ")
                detail = (
                    f"{str(row['prediction'])[:44]!r} (gold {str(row['reference'])[:30]!r})"
                )
            print(
                f"    [{done:3d}/{total}] {mark} {row.get('category', '?')!s:24s}"
                f" {str(row.get('question', ''))[:52]:52s} -> {detail}",
                flush=True,
            )
        return row

    rows = await asyncio.gather(*(one_verbose(q) for q in questions))
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

    # Self-grading bias, measured rather than disclaimed. Both arms grade the
    # SAME predictions with the SAME prompt; the only difference is whether the
    # grader is the model that produced them. `delta` > 0 means the answerer was
    # more generous to itself than an independent grader was, in accuracy points.
    judge_bias: dict[str, Any] | None = None
    paired = [r for r in answerable if r.get("self_correct") is not None]
    if paired:
        indep = sum(1 for r in paired if r["correct"])
        selfg = sum(1 for r in paired if r["self_correct"])
        agree = sum(1 for r in paired if r["correct"] == r["self_correct"])
        judge_bias = {
            "n_paired": len(paired),
            "independent_judge": judge_model,
            "self_judge": answer_model,
            "independent_accuracy": round(indep / len(paired), 4),
            "self_accuracy": round(selfg / len(paired), 4),
            "delta_self_minus_independent": round((selfg - indep) / len(paired), 4),
            "agreement": round(agree / len(paired), 4),
        }

    return {
        "config": config,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "thinking_budget": thinking_budget,
        "dynamic_k": dynamic_k,
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
        "judge_protocol": "longmemeval-official" if official_judge else "strict-custom",
        "judge_bias": judge_bias,
        "errors": errors[:10],
        "per_question": ok,
    }


__all__ += ["evaluate_end_to_end", "parse_answer"]
