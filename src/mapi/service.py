"""Application service: the transactional boundary between API and domain.

Route handlers do HTTP; the domain does algorithms; this does the workflow that
joins them. Keeping it here rather than in a route means the same ingest path is
reachable from a worker, a CLI or a test without spinning up FastAPI.

Ingestion is the interesting one:

    validate -> normalize -> exact-dup check -> chunk -> embed
             -> near-dup check -> optional supersession -> persist

Deduplication runs *before* embedding where it can (the exact-hash path),
because embedding is the expensive step and re-ingesting an unchanged document
is the single most common write a memory system sees.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .bulkwrite import (
    BulkStats,
    BulkView,
    NeedsSequential,
    TimedSession,
    WriteStore,
    timed_session,
)
from .config import Settings
from .core.errors import (
    MapiError,
    NotFoundError,
    PayloadTooLargeError,
    ProviderError,
    ValidationError,
)
from .core.logging import get_logger
from .core.metrics import EMBEDDINGS, INGESTED, SEARCH_LATENCY, SEARCH_STAGE_LATENCY
from .core.quota import Quota, check_bytes, check_memories, check_writes
from .domain.association import entity_memo, propose_associations
from .domain.chunking import TextChunk, chunk_text, normalize
from .domain.consolidation import (
    CONTRADICT_LOW,
    ContradictionProposal,
    DuplicateKind,
    apply_supersession,
    dedupe_candidates,
    detect_near_duplicate,
    distinctive_memo,
    keyed_unchanged,
    merge_duplicate,
    propose_contradictions,
    propose_supersessions,
    unexplained_pairs,
)
from .domain.embeddings.base import EmbeddingProvider, similarity_memo
from .domain.embeddings.context import build_header, for_embedding, wants_header
from .domain.models import (
    Chunk,
    Memory,
    MemoryKind,
    MemoryStatus,
    MemoryVersion,
    Organization,
    RelationEdge,
    RelationType,
    ReplaceMode,
    Space,
    utcnow,
)
from .domain.retrieval.confidence import ConfidenceBands
from .domain.retrieval.expansion import CompletionExpander
from .domain.retrieval.pipeline import RetrievalPipeline, SearchRequest, SearchResponse
from .domain.retrieval.rerank import Reranker
from .domain.retrieval.similar import SimilarRequest, SimilarResponse, find_similar
from .domain.synthesis import classify
from .domain.synthesis.adjudicate import (
    adjudicate_contradictions,
    adjudicate_supersessions,
)
from .domain.synthesis.chat import ChatAnswer
from .domain.synthesis.chat import Turn as ChatTurn
from .domain.synthesis.chat import answer as chat_answer
from .domain.synthesis.derive import CompleteFn, DerivedAnswer, SourceDoc, derive_answer
from .domain.synthesis.extract import DEFAULT_SUBJECT, Claim, extract_claims
from .domain.synthesis.hydrate import Neighbourhood, assemble
from .domain.synthesis.understand import QueryUnderstanding
from .store.base import BulkEraseReport, MemoryFilter, MemoryStore, Page, PurgeReport

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """A memory resolved together with its whole relation neighborhood.

    `current_head` is non-empty exactly when `is_current` is false. Neighbors
    that no longer resolve (erased) are omitted, not stubbed.
    """

    memory: Memory
    is_current: bool
    current_head: list[Memory]
    replaced: list[Memory]
    derived_from: list[Memory]
    derivatives: list[Memory]
    references: list[Memory]
    contradicts: list[Memory]


@dataclass(frozen=True, slots=True)
class Graph:
    """A space as memories plus the typed relations between them."""

    space_id: str
    memories: list[Memory]
    edges: list[RelationEdge]
    degree: dict[str, int]
    counts: dict[str, Any]


@dataclass(slots=True)
class IngestResult:
    memory: Memory
    created: bool
    duplicate_of: str | None = None
    duplicate_kind: DuplicateKind = DuplicateKind.NONE
    similarity: float = 0.0
    superseded: list[str] = field(default_factory=list)
    #: Memories this write is judged to CONTRADICT. Recorded as edges and
    #: reported; never suppressed, because either side may be the true one.
    contradicts: list[str] = field(default_factory=list)
    #: Ids of the atomic claims extracted from this write. The parent is
    #: untouched and still retrievable; these are additions, not a rewrite.
    extracted: list[str] = field(default_factory=list)
    #: Supersessions proposed but NOT applied, because confidence fell below
    #: `supersede_min_confidence`. Reported so the decision is auditable
    #: rather than a silent drop -- these memories are still active.
    supersede_declined: list[str] = field(default_factory=list)
    chunk_count: int = 0
    #: Memories a keyed write with replace=erase destroyed. (Superseded ones
    #: are in `superseded`, as for any other supersession.)
    erased: list[str] = field(default_factory=list)


# -- write-path inputs and outcomes ---------------------------------------------


@dataclass(frozen=True, slots=True)
class IngestItem:
    """One write in a bulk request. Mirrors `ingest`'s keyword arguments."""

    content: str
    summary: str = ""
    metadata: Mapping[str, object] | None = None
    tags: Sequence[str] = ()
    source: str = ""
    occurred_at: datetime | None = None
    key: str | None = None
    replace: ReplaceMode = ReplaceMode.SUPERSEDE


@dataclass(slots=True)
class BulkOutcome:
    """One bulk item's fate: its result, or the error that stopped it alone."""

    index: int
    result: IngestResult | None = None
    error: MapiError | None = None


@dataclass(slots=True)
class _Prepared:
    """A write validated, normalized and chunked, waiting only for vectors.

    `memory` already has its final id, content, hash and key; `texts` are
    the strings to embed (chunks framed by the context header), which are
    never what gets stored.
    """

    memory: Memory
    pieces: list[TextChunk]
    texts: list[str]
    size: int
    replace: ReplaceMode = ReplaceMode.SUPERSEDE

    def with_vectors(self, vectors: Sequence[list[float]]) -> Memory:
        chunks = [
            Chunk(
                memory_id=self.memory.id,
                ordinal=piece.ordinal,
                text=piece.text,
                token_estimate=piece.token_estimate,
                embedding=list(vector),
            )
            for piece, vector in zip(self.pieces, vectors, strict=True)
        ]
        return self.memory.model_copy(update={"chunks": chunks})


@dataclass(slots=True)
class _Stage3:
    """What bulk ingest's write stage works from, shared by both its paths."""

    prepared: dict[int, _Prepared]
    vectors: dict[int, list[list[float]]]
    #: Stage 1's snapshots: ACTIVE row per key, exact-duplicate holder per digest.
    active: dict[str, Memory]
    by_hash: dict[str, Memory]
    repeated: set[int]
    outcomes: list[BulkOutcome]
    dedupe: bool
    auto_supersede: bool
    detect_conflicts: bool
    extract: bool
    stats: BulkStats


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        settings: Settings,
        completer: CompleteFn | None = None,
        extractor: CompleteFn | None = None,
        understander: CompleteFn | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        #: Async prompt -> text, or None when synthesis is off. Injected so
        #: tests drive derivation deterministically and the service never
        #: imports a vendor SDK.
        self.completer = completer
        #: Write-path completion, separate from `completer` because
        #: extraction runs per document and derivation runs per question --
        #: different models, different budgets, tuned independently.
        self.extractor = extractor
        #: Search-path completion: one sentence in, one label out. Shared
        #: across requests so its cache and circuit breaker mean anything --
        #: a per-request instance would call the vendor for every search and
        #: would never trip a breaker.
        self.understanding = QueryUnderstanding(
            understander, timeout_s=settings.understanding_timeout_s
        )
        #: HyDE, when there is a model to generate the hypothetical answer with.
        #:
        #: The pipeline defaulted to `NoopExpander` and nothing ever passed
        #: anything else, so `HydeExpander` was never constructed anywhere in the
        #: tree -- `use_expansion=True` was silently a no-op even for a caller who
        #: found the flag. Now the expander exists whenever a completer does, and
        #: the flag decides per request whether to spend the call.
        expander = CompletionExpander(completer) if completer is not None else None
        self.pipeline = RetrievalPipeline(
            store,
            embedder,
            reranker,
            understanding=self.understanding,
            expander=expander,
            confidence_bands=ConfidenceBands(
                strong=settings.confidence_strong, weak=settings.confidence_weak
            ),
        )
        #: (org_id, space_id) -> (monotonic time cached, space). Hits only;
        #: see `get_space_or_raise`.
        self._space_cache: dict[tuple[str, str], tuple[float, Space]] = {}

    # -- spaces ------------------------------------------------------------

    async def ensure_organization(self, name: str) -> Organization:
        org = Organization(name=name)
        return await self.store.create_organization(org)

    async def create_space(
        self,
        org_id: str,
        *,
        slug: str,
        name: str,
        description: str = "",
        metadata: dict[str, object] | None = None,
    ) -> Space:
        space = Space(
            org_id=org_id,
            slug=slug,
            name=name,
            description=description,
            metadata=metadata or {},
        )
        return await self.store.create_space(space)

    #: Bound on cached spaces, so a caller cycling through ids cannot grow the
    #: process. Far above one event's per-person spaces.
    _SPACE_CACHE_MAX = 50_000

    async def get_space_or_raise(self, org_id: str, space_id: str) -> Space:
        """The space, or 404 -- from a short-lived cache when it has been seen.

        Every read and write starts here, so an uncached lookup was one DB
        session on every request for a row that essentially never changes.
        Keyed by (org, space), so the cache cannot cross a tenant boundary any
        more than the query could. Only hits are cached: a space created a
        moment ago -- possibly by another process -- is never reported
        missing. A space deleted through this process is evicted at once
        (`forget_space`); one deleted elsewhere can still be found here for up
        to `space_cache_ttl_s`, and then reads find it empty, not someone
        else's.
        """
        ttl = self.settings.space_cache_ttl_s
        key = (org_id, space_id)
        if ttl > 0:
            hit = self._space_cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < ttl:
                return hit[1]
        space = await self.store.get_space(org_id, space_id)
        if space is None:
            self._space_cache.pop(key, None)
            # A space in another org must be indistinguishable from one that
            # does not exist, or the 404/403 difference leaks its existence.
            raise NotFoundError(f"space {space_id} not found", field="space_id")
        if ttl > 0:
            self._space_cache.pop(key, None)
            while len(self._space_cache) >= self._SPACE_CACHE_MAX:
                # Oldest first: dicts iterate in insertion order, and a
                # refreshed entry was re-inserted at the end just above.
                del self._space_cache[next(iter(self._space_cache))]
            self._space_cache[key] = (time.monotonic(), space)
        return space

    def forget_space(self, org_id: str, space_id: str) -> None:
        """Evict one space from the lookup cache. Call on delete and purge."""
        self._space_cache.pop((org_id, space_id), None)

    # -- ingestion ---------------------------------------------------------

    def _quota(self) -> Quota:
        return Quota(
            max_memories_per_org=self.settings.max_memories_per_org,
            max_bytes_per_org=self.settings.max_bytes_per_org,
            max_writes_per_day=self.settings.max_writes_per_day,
        )

    def _similarity_ceiling(self) -> float:
        """Upper edge of the supersession and contradiction bands.

        It was the constant SUPERSEDE_HIGH (0.97), which only coincided with
        `dedupe_threshold` at its default. The band exists to hand everything
        below the duplicate line to belief revision and everything above it to
        dedupe; raise the dedupe threshold (0.995 on gemini-embedding-001,
        where value changes like "moved to Austin" vs "moved to Boston" land
        at 0.97-0.99) and a fixed 0.97 left a gap in which a changed fact was
        neither merged nor ever considered for supersession.

        Floored at CONTRADICT_LOW so a very low dedupe threshold cannot invert
        a band -- pairs above the dedupe line never reach the proposers anyway,
        because dedupe returns first.
        """
        return max(self.settings.dedupe_threshold, CONTRADICT_LOW)

    def _chat_search_request(
        self, org_id: str, space_id: str, message: str, k: int
    ) -> SearchRequest:
        """The search behind /chat, configured like the search endpoint's.

        Chat built a bare `SearchRequest`, so it ran on the dataclass defaults
        instead of the deployment's settings -- a tuned `max_per_source`,
        `coverage_limit` or `route_by_kind` changed /search and silently not
        the answers built on it -- and with no `asked_at`, which leaves the
        temporal stage dead: "what did I do last week" could not prefer last
        week. Mirrors api/v1/search.py; the question is asked now.
        """
        s = self.settings
        return SearchRequest(
            query=message,
            org_id=org_id,
            space_id=space_id,
            limit=k,
            candidate_multiplier=s.candidate_multiplier,
            rerank_candidates=s.rerank_candidates,
            rrf_k=s.rrf_k,
            coverage_limit=s.coverage_limit,
            max_per_source=s.max_per_source,
            route_by_kind=s.route_by_kind,
            asked_at=utcnow(),
        )

    async def _check_quota(self, org_id: str, incoming_bytes: int) -> None:
        """Refuse a write that would take a tenant past its limits.

        One usage query for three limits, and only when at least one is
        configured -- the shipped default is unlimited, and a deployment that
        has not opted in must not pay for a count on every write.
        """
        quota = self._quota()
        if not quota.enforced:
            return
        # Calendar day in UTC. A rolling 24-hour window would be fairer and
        # would need a per-write timestamp index; a day boundary is what
        # people mean by "per day" and costs one comparison.
        midnight = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        usage = await self.store.tenant_usage(org_id, since=midnight)
        check_memories(quota, usage.memories)
        check_bytes(quota, usage.bytes_stored, incoming_bytes)
        check_writes(quota, usage.writes_today)

    async def _hydrate(
        self,
        org_id: str,
        space_id: str,
        hits: Sequence[Any],
    ) -> list[SourceDoc]:
        """Retrieved memories plus the turns around them, budgeted.

        Only what is ASSEMBLED changes; retrieval is untouched, so
        `full_recall@k` and MRR are computed on the same set as before and
        any movement in accuracy is attributable.
        """
        anchors = [
            SourceDoc(
                id=hit.memory.id,
                text=hit.memory.content,
                occurred_at=hit.memory.occurred_at,
            )
            for hit in hits
        ]
        width = self.settings.answer_neighbours
        if not anchors or width <= 0:
            return anchors

        hoods = []
        for hit, anchor in zip(hits, anchors, strict=True):
            before, after = await self._surrounding(org_id, space_id, hit.memory, width)
            hoods.append(Neighbourhood(anchor=anchor, before=before, after=after))
        return assemble(hoods, budget_chars=self.settings.answer_budget_chars)

    @staticmethod
    def _grouping(memory: Memory) -> tuple[str, str] | None:
        """What "the same conversation" means for this memory.

        Ordered by how specific each key is, and the order is load-bearing.
        `source` is LAST and is a fallback, because callers use it for
        coarse labels: the benchmark harness sets `source=document.speaker`,
        so grouping on it would treat every turn the user ever spoke as one
        conversation and hydrate an anchor with unrelated turns from months
        away. A neighbourhood has to be the unit the caller actually wrote.

        None means no grouping is known, and the memory travels alone --
        which is the previous behaviour, and the right answer when we cannot
        tell what it belongs with.
        """
        meta = memory.metadata or {}
        for key in ("session_id", "conversation_id", "doc_id", "extracted_from"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return (key, value)
        if memory.source:
            return ("source", memory.source)
        return None

    async def _surrounding(
        self,
        org_id: str,
        space_id: str,
        memory: Memory,
        width: int,
    ) -> tuple[tuple[SourceDoc, ...], tuple[SourceDoc, ...]]:
        """The turns immediately before and after `memory` in its own unit."""
        group = self._grouping(memory)
        if group is None:
            return (), ()
        key, value = group
        filters = (
            MemoryFilter(source=value)
            if key == "source"
            else MemoryFilter(metadata=((key, value),))
        )
        page = await self.store.list_memories(
            org_id,
            space_id,
            filters=filters,
            # Enough either side to find the anchor's position without
            # paging a long conversation into memory.
            limit=max(width * 8, 32),
        )
        ordered = sorted(page.items, key=lambda m: (m.occurred_at, m.id))
        try:
            at = next(i for i, m in enumerate(ordered) if m.id == memory.id)
        except StopIteration:
            return (), ()

        def to_docs(items: Sequence[Memory]) -> tuple[SourceDoc, ...]:
            return tuple(
                SourceDoc(id=m.id, text=m.content, occurred_at=m.occurred_at)
                for m in items
                if m.id != memory.id and m.status is MemoryStatus.ACTIVE
            )

        return (
            to_docs(ordered[max(0, at - width) : at]),
            to_docs(ordered[at + 1 : at + 1 + width]),
        )

    async def _confirm_supersessions(
        self,
        memory: Memory,
        shortlist: Sequence[Any],
        *,
        store: WriteStore | None = None,
    ) -> list[Any]:
        """Keep only the shortlisted supersessions the model agrees with."""
        if not shortlist:
            return []
        if self.extractor is None:
            # No adjudicator, so nothing is confirmed. Cosine alone is not
            # trusted to hide a memory -- that is the whole finding.
            log.info("supersessions_unconfirmed", count=len(shortlist))
            return []
        by_id = {p.old_id: p for p in shortlist}
        existing = await (store or self.store).get_memories(
            memory.org_id, memory.space_id, list(by_id)
        )
        pairs = [(mid, m.content) for mid, m in existing.items()]
        verdicts = await adjudicate_supersessions(memory.content, pairs, self.extractor)
        # Carry the ADJUDICATOR's reason onto the edge, not the cosine
        # proposer's. The proposer explains why a pair was nominated ("same
        # subject (similarity 0.83); shared tags"), which is the wrong
        # question: what a reader needs months later is why the model APPROVED
        # hiding a memory, and the verdict says that in the form
        # "attribute: old -> new". Returning the original proposal threw the
        # only auditable justification away.
        return [
            replace(
                by_id[v.memory_id],
                reason=v.reason,
                confidence=v.confidence,
            )
            for v in verdicts
            if v.memory_id in by_id
        ]

    def _unexplained_conflicts(
        self,
        memory: Memory,
        embedding: list[float],
        pairs: Sequence[tuple[Memory, list[float]]],
    ) -> list[ContradictionProposal]:
        """Close pairs with no lexical signal, as bare shortlist entries."""
        return [
            ContradictionProposal(
                left_id=memory.id,
                right_id=other.id,
                similarity=score,
                reason="",
                confidence=0.0,
            )
            for other, score in unexplained_pairs(
                memory, embedding, pairs, high=self._similarity_ceiling()
            )
        ]

    async def _confirm_conflicts(
        self,
        memory: Memory,
        shortlist: Sequence[ContradictionProposal],
        *,
        store: WriteStore | None = None,
    ) -> list[ContradictionProposal]:
        """Keep only the shortlisted conflicts the model agrees with.

        Fails closed. Without an adjudicator nothing is recorded: a false
        CONTRADICTS edge between two facts that merely differ erodes trust
        faster than a missed one, and the lexical signals produce those in
        volume on any corpus with dates or quantities in it.
        """
        if not shortlist:
            return []
        if self.extractor is None:
            log.info("conflicts_unconfirmed", count=len(shortlist))
            return []
        by_id = {p.right_id: p for p in shortlist}
        existing = await (store or self.store).get_memories(
            memory.org_id, memory.space_id, list(by_id)
        )
        verdicts = await adjudicate_contradictions(
            memory.content,
            [(mid, m.content) for mid, m in existing.items()],
            self.extractor,
        )
        return [
            ContradictionProposal(
                left_id=memory.id,
                right_id=v.memory_id,
                similarity=by_id[v.memory_id].similarity,
                reason=v.reason,
                confidence=v.confidence,
            )
            for v in verdicts
            if v.memory_id in by_id
        ]

    async def _unused_adjudicate_remaining(
        self,
        memory: Memory,
        embedding: list[float],
        pairs: Sequence[tuple[Memory, list[float]]],
    ) -> list[ContradictionProposal]:
        """Conflicts the lexical signals could not see, judged by a model.

        Runs only over pairs that already cleared the similarity gate and
        that the negation/antonym/figure checks left unexplained -- a handful
        per write, not the corpus. Off entirely without an extraction
        backend, which is the configuration this feature shipped with, so
        turning a backend on can only ADD conflicts and never change the ones
        already being found.

        Reuses the extraction completer rather than adding a fourth: this is
        a write-path judgement over a short passage, which is the job that
        completer is already configured and budgeted for.
        """
        if self.extractor is None:
            return []
        remaining = unexplained_pairs(memory, embedding, pairs, high=self._similarity_ceiling())
        if not remaining:
            return []
        verdicts = await adjudicate_contradictions(
            memory.content,
            [(m.id, m.content) for m, _ in remaining],
            self.extractor,
        )
        similarity_by_id = {m.id: score for m, score in remaining}
        return [
            ContradictionProposal(
                left_id=memory.id,
                right_id=v.memory_id,
                similarity=similarity_by_id.get(v.memory_id, 0.0),
                reason=v.reason,
                confidence=v.confidence,
            )
            for v in verdicts
        ]

    async def ingest(
        self,
        *,
        org_id: str,
        space_id: str,
        content: str,
        summary: str = "",
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] = (),
        source: str = "",
        occurred_at: datetime | None = None,
        # Every consolidation step defaults ON. The API does not expose
        # switches for these at all; the keyword arguments survive only so
        # the benchmark harness can isolate one behaviour at a time, which is
        # what an A/B arm is for.
        dedupe: bool = True,
        auto_supersede: bool = True,
        detect_conflicts: bool = True,
        extract: bool = True,
        claims: Sequence[Claim] | None = None,
        kind: MemoryKind = MemoryKind.EPISODIC,
        key: str | None = None,
        replace: ReplaceMode | str = ReplaceMode.SUPERSEDE,
    ) -> IngestResult:
        """Write one memory. With a `key`, replace whatever holds that key.

        Unkeyed writes run the full pipeline in the module docstring. Keyed
        writes skip it: the caller has already said which fact this is, so
        near-duplicate detection, supersession proposals, contradiction
        adjudication and claim extraction have nothing to decide -- and each
        was a way for a changed value to be merged into its old text or
        hidden behind it.
        """
        await self.get_space_or_raise(org_id, space_id)
        prepared = self._prepare(
            org_id,
            space_id,
            content=content,
            summary=summary,
            metadata=metadata,
            tags=tags,
            source=source,
            occurred_at=occurred_at,
            kind=kind,
            key=key,
            replace=ReplaceMode(replace),
        )

        # Quotas before embedding, because embedding is where the money is.
        # Checking after would let a tenant over their limit still spend the
        # vendor call that the limit exists to prevent.
        await self._check_quota(org_id, prepared.size)

        if prepared.memory.key is not None:
            return await self._write_keyed(prepared)

        # -- exact duplicate: cheapest check, before any embedding ----------
        if dedupe:
            duplicate = await self._fold_exact_duplicate(prepared)
            if duplicate is not None:
                return duplicate

        vectors = await self._embed_prepared(prepared)
        return await self._write_unkeyed(
            prepared,
            vectors,
            dedupe=dedupe,
            auto_supersede=auto_supersede,
            detect_conflicts=detect_conflicts,
            extract=extract,
            claims=claims,
        )

    # -- write-path stages -------------------------------------------------

    def _prepare(
        self,
        org_id: str,
        space_id: str,
        *,
        content: str,
        summary: str = "",
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] = (),
        source: str = "",
        occurred_at: datetime | None = None,
        kind: MemoryKind = MemoryKind.EPISODIC,
        key: str | None = None,
        replace: ReplaceMode = ReplaceMode.SUPERSEDE,
    ) -> _Prepared:
        """Validate, normalize and chunk one write. No I/O.

        Split out so bulk ingest can prepare every item before embedding any
        of them -- which is what lets a hundred items share a handful of
        embedding calls instead of making one each.
        """
        from pydantic import ValidationError as ModelValidationError

        cleaned = normalize(content)
        if not cleaned:
            raise ValidationError("content is empty after normalization", field="content")
        size = len(cleaned.encode("utf-8"))
        if size > self.settings.max_content_bytes:
            raise PayloadTooLargeError(
                f"content is {size} bytes, limit is {self.settings.max_content_bytes}",
                field="content",
            )
        try:
            memory = Memory(
                org_id=org_id,
                space_id=space_id,
                content=cleaned,
                summary=summary,
                metadata=dict(metadata or {}),
                tags=list(tags),
                source=source,
                kind=kind,
                occurred_at=occurred_at or utcnow(),
                key=key,
            )
        except ModelValidationError as exc:
            # A 422 naming the field, not a 500: the model's own validators
            # (tag length, key shape) are input validation too.
            errors = exc.errors()
            message = errors[0]["msg"] if errors else "invalid memory"
            location = ".".join(str(p) for p in errors[0]["loc"]) if errors else ""
            raise ValidationError(message, field=location or None) from exc

        pieces = chunk_text(
            cleaned,
            target_tokens=self.settings.chunk_target_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        if not pieces:
            raise ValidationError("content produced no chunks", field="content")

        # Embedded WITH a context header, stored WITHOUT one. The header
        # gives an isolated chunk the date, subject and speaker its own text
        # never states; the persisted chunk stays byte-identical to what the
        # caller wrote, so search results, quotes and grounding are unchanged.
        header = (
            build_header(
                occurred_at=memory.occurred_at,
                source=source,
                tags=tuple(tags),
                metadata=metadata or {},
            )
            if self.settings.contextual_embedding
            # A short memory is embedded bare; see `wants_header`.
            and wants_header(cleaned, min_chars=self.settings.contextual_min_chars)
            else ""
        )
        return _Prepared(
            memory=memory,
            pieces=pieces,
            texts=[for_embedding(p.text, header) for p in pieces],
            size=size,
            replace=replace,
        )

    async def _embed_prepared(self, prepared: _Prepared) -> list[list[float]]:
        try:
            result = await self.embedder.embed(prepared.texts)
            EMBEDDINGS.labels(provider=self.embedder.name, outcome="ok").inc(
                len(prepared.texts)
            )
        except Exception:
            EMBEDDINGS.labels(provider=self.embedder.name, outcome="error").inc()
            INGESTED.labels(outcome="embed_error").inc()
            raise
        return result.vectors

    async def _fold_exact_duplicate(
        self, prepared: _Prepared, *, store: WriteStore | None = None
    ) -> IngestResult | None:
        """Merge an exact restatement into the memory already holding it.

        Looks the holder up at write time, always. Only ACTIVE unkeyed
        memories qualify -- see `consolidation.dedupe_eligible`.

        `store` (here and on the other write stages) is the real store, or
        a bulk request's `BulkView` over it; see `bulkwrite`.
        """
        store = store or self.store
        memory = prepared.memory
        existing = await store.find_by_content_hash(
            memory.org_id, memory.space_id, memory.content_sha256
        )
        if existing is None:
            return None
        merged = merge_duplicate(existing, memory)
        await store.upsert_memory(merged)
        INGESTED.labels(outcome="duplicate_exact").inc()
        return IngestResult(
            memory=merged,
            created=False,
            duplicate_of=existing.id,
            duplicate_kind=DuplicateKind.EXACT,
            similarity=1.0,
            chunk_count=len(merged.chunks),
        )

    @staticmethod
    def _unchanged(existing: Memory) -> IngestResult:
        """The report for a keyed write that found its payload already there."""
        INGESTED.labels(outcome="unchanged_key").inc()
        return IngestResult(
            memory=existing,
            created=False,
            duplicate_of=existing.id,
            duplicate_kind=DuplicateKind.EXACT,
            similarity=1.0,
            chunk_count=len(existing.chunks),
        )

    async def _write_keyed(
        self,
        prepared: _Prepared,
        vectors: Sequence[list[float]] | None = None,
        *,
        known: Memory | None = None,
        looked_up: bool = False,
        store: WriteStore | None = None,
    ) -> IngestResult:
        """Replace whatever holds this key, atomically; or do nothing at all.

        The unchanged check runs twice on purpose. Here, before embedding, so
        a re-sync of an unchanged snapshot costs no vendor call; and again
        inside `store.write_keyed`, under the key's lock, because another
        writer may have moved the key in between.
        """
        store = store or self.store
        memory = prepared.memory
        org_id, space_id, key = memory.org_id, memory.space_id, memory.key
        if key is None:  # pragma: no cover - callers route on the key
            raise ValidationError("a keyed write needs a key", field="key")
        active = known
        if not looked_up:
            found = await store.get_active_by_keys(org_id, space_id, [key])
            active = found.get(key)
        if active is not None and keyed_unchanged(active, memory):
            return self._unchanged(active)

        if vectors is None:
            vectors = await self._embed_prepared(prepared)
        outcome = await store.write_keyed(prepared.with_vectors(vectors), mode=prepared.replace)
        if outcome.unchanged:
            return self._unchanged(outcome.memory)

        if prepared.replace is ReplaceMode.SUPERSEDE:
            # The replaced row still exists, so its derivation edges do too.
            for old_id in outcome.replaced:
                await self._mark_derivations_stale(
                    org_id,
                    space_id,
                    await self._direct_derivatives(org_id, space_id, old_id, store=store),
                    store=store,
                )
        else:
            await self._mark_derivations_stale(
                org_id, space_id, outcome.derivatives, store=store
            )
        INGESTED.labels(outcome="created").inc()
        return IngestResult(
            memory=outcome.memory,
            created=True,
            superseded=outcome.replaced if prepared.replace is ReplaceMode.SUPERSEDE else [],
            erased=outcome.replaced if prepared.replace is ReplaceMode.ERASE else [],
            chunk_count=len(outcome.memory.chunks),
        )

    async def _write_unkeyed(
        self,
        prepared: _Prepared,
        vectors: Sequence[list[float]],
        *,
        dedupe: bool = True,
        auto_supersede: bool = True,
        detect_conflicts: bool = True,
        extract: bool = True,
        claims: Sequence[Claim] | None = None,
        store: WriteStore | None = None,
    ) -> IngestResult:
        """The consolidation pipeline for a write with its vectors in hand."""
        store = store or self.store
        memory = prepared.with_vectors(vectors)
        org_id, space_id = memory.org_id, memory.space_id
        cleaned = memory.content
        tags = memory.tags
        chunks = memory.chunks

        # -- neighbours: one lookup, four consumers ----------------------------
        # Deduplication, supersession, contradiction and association all ask
        # the same question. This used to be TWO scans -- a 512-row
        # `sample_embeddings` for dedupe and a separate one for the rest --
        # and the dedupe scan carried no text, which is why it could only
        # compare vectors.
        pairs: list[tuple[Memory, list[float]]] = []
        if chunks:
            pairs = await store.neighbours(
                org_id,
                space_id,
                chunks[0].embedding or [],
                limit=self.settings.consolidation_candidates,
                exclude_id=memory.id,
            )

        # -- near duplicate ----------------------------------------------------
        if dedupe and chunks:
            # Only ACTIVE, unkeyed neighbours may absorb this write: a
            # superseded row would swallow a restated value and stay hidden,
            # and a keyed row belongs to its key.
            mergeable = dedupe_candidates(pairs)
            verdict = detect_near_duplicate(
                chunks[0].embedding or [],
                [(m.id, v) for m, v in mergeable],
                threshold=self.settings.dedupe_threshold,
                text=cleaned,
                texts={m.id: m.content for m, _ in mergeable},
            )
            if verdict.is_duplicate and verdict.existing_id:
                existing = await store.get_memory(org_id, space_id, verdict.existing_id)
                if existing is not None:
                    merged = merge_duplicate(existing, memory)
                    await store.upsert_memory(merged)
                    INGESTED.labels(outcome="duplicate_near").inc()
                    return IngestResult(
                        memory=merged,
                        created=False,
                        duplicate_of=existing.id,
                        duplicate_kind=DuplicateKind.NEAR,
                        similarity=verdict.similarity,
                        chunk_count=len(merged.chunks),
                    )

        # -- belief revision ---------------------------------------------------
        # Two different operations that both need the same candidate scan:
        # supersession (revision -- newer replaces older) and contradiction
        # (disagreement -- two ACTIVE memories that cannot both be true).
        #
        # Both are gated, but for different reasons. Supersession is gated on
        # SAFETY: it hides a memory, and hiding a user's data because a
        # similarity score crossed a threshold has an invisible failure mode.
        # Contradiction hides nothing, so it is gated purely on COST -- it
        # needs a candidate scan, and a cheap write path is the architectural
        # bet of this whole system. Enable it per request or per space.
        superseded: list[str] = []
        conflicts: list[str] = []
        declined: list[str] = []
        if chunks and (auto_supersede or detect_conflicts):
            # NEAREST, not newest. This used to page the 256 most recent
            # memories and pull every embedding across the wire -- roughly
            # 1.5MB per write at 768 dimensions -- and then discard almost
            # all of them, because both proposers immediately filter on
            # cosine similarity anyway. Worse, it was a silent scale ceiling:
            # in a space with more than 256 memories, anything older simply
            # stopped being a supersession or contradiction candidate, so a
            # fact stated last year could never be revised.
            # A keyed memory is replaced through its key and nothing else, so
            # it is never a supersession target for an unkeyed write.
            revisable = [(m, v) for m, v in pairs if m.key is None]
            all_proposals = (
                propose_supersessions(
                    memory,
                    chunks[0].embedding or [],
                    revisable,
                    high=self._similarity_ceiling(),
                )
                if auto_supersede
                else []
            )
            # `propose_supersessions` ranks candidates and documents that the
            # caller decides what to do with them. Applying every one is the
            # wrong decision: at the proposer's floor, "same subject" means
            # cosine 0.72, and two chat turns about the same hobby clear that
            # easily without either replacing the other. Applying it flips the
            # older memory to SUPERSEDED, which default retrieval hides, so a
            # topical coincidence silently deletes a true memory from every
            # answer. Declining costs far less -- both stay visible.
            # The cosine proposer SHORTLISTS; the model decides.
            #
            # On its own it hid four true memories out of five on a real
            # corpus -- "runs on Heroku with two dynos" replaced by "the
            # Python runtime is 3.13", both true, similarity 0.87, same tag.
            # Embedding distance can say two statements are about the same
            # area; it cannot say one replaced the other, because that is a
            # question about what changed and the geometry encodes no change.
            #
            # Supersession is the only operation here that HIDES a memory, so
            # it is the one that gets a second opinion. Without an extraction
            # backend nothing is confirmed and nothing is hidden, which is
            # the safe direction.
            floor = self.settings.supersede_min_confidence
            shortlist = [p for p in all_proposals if p.confidence >= floor]
            declined = [p.old_id for p in all_proposals if p.confidence < floor]
            proposals = await self._confirm_supersessions(memory, shortlist, store=store)
            # Compare by TARGET ID, not by proposal value.
            #
            # This was `[p.old_id for p in shortlist if p not in proposals]`,
            # and `SupersessionProposal` is a frozen dataclass, so `in` compares
            # all five fields -- including `reason` and `confidence`, which
            # `_confirm_supersessions` deliberately REPLACES with the
            # adjudicator's verdict. So no confirmed proposal ever equalled its
            # own shortlist entry, and every applied supersession was also
            # reported as declined: the same memory id appeared in both
            # `superseded` and `supersede_declined`, and the
            # `supersession_declined` log line counted confirmations.
            #
            # Which contradicts what the field promises -- "proposed but NOT
            # applied ... these memories are still active" -- for a memory that
            # is, at that moment, SUPERSEDED.
            applied = {p.old_id for p in proposals}
            declined += [p.old_id for p in shortlist if p.old_id not in applied]
            if declined:
                # Reported, never silent: a caller that wanted those merges
                # needs to see that the system saw them and held back.
                log.info(
                    "supersession_declined",
                    memory_id=memory.id,
                    count=len(declined),
                    floor=floor,
                )
            # Contradiction now works exactly like supersession: the cheap
            # signals SHORTLIST, and the model decides.
            #
            # They used to be a verdict, and on real data they were wrong
            # almost every time. `_figure_conflict` fires whenever two
            # topically-close memories carry different numbers, so a phone
            # number and a relocation year read as "same subject, different
            # figures" -- a contradiction between a contact detail and a move
            # date. Any corpus containing dates or quantities is mostly those
            # pairs.
            conflict_shortlist = (
                propose_contradictions(
                    memory,
                    chunks[0].embedding or [],
                    pairs,
                    high=self._similarity_ceiling(),
                )
                if detect_conflicts
                else []
            )
            # Reads the SAME neighbour set again, so the ordinary "these are
            # about the same thing" relation costs no extra query. It is the
            # only edge here that is not a claim about truth, and the only
            # one most corpora produce in quantity -- without it a space of
            # true, non-conflicting facts is a field of isolated dots.
            association_proposals = propose_associations(
                memory, chunks[0].embedding or [], pairs
            )
            conflict_proposals: list[ContradictionProposal] = []
            if detect_conflicts:
                # Pairs the lexical signals could not explain join the same
                # shortlist rather than bypassing the judge.
                conflict_shortlist += self._unexplained_conflicts(
                    memory, chunks[0].embedding or [], pairs
                )
                conflict_proposals = await self._confirm_conflicts(
                    memory, conflict_shortlist, store=store
                )
                # A pair cannot be BOTH revised and disputed. Supersession
                # says time orders them; contradiction says nothing does, and
                # they mean opposite things to a reader -- one hides the old
                # memory, the other insists both stay visible.
                #
                # Observed live: "I live in Berlin" came back as superseded by
                # AND contradicting "I moved to Madrid", so the graph asserted
                # a replacement and a standoff about the same two rows.
                # Supersession wins: it is the more specific claim, and it is
                # the one that already acted on the data.
                revised = {p.old_id for p in proposals}
                conflict_proposals = [
                    c for c in conflict_proposals if c.right_id not in revised
                ]
            # The memory must exist before an edge can point at it: the edge
            # has a foreign key to both endpoints.
            memory = await store.upsert_memory(memory)

            for conflict in conflict_proposals:
                # `contradicts` is symmetric, so both directions are written --
                # otherwise the answer to "does this conflict with anything"
                # would depend on which memory you happened to look up first.
                # Neither side is superseded or hidden: either may be true.
                for source_id, target_id in (
                    (memory.id, conflict.right_id),
                    (conflict.right_id, memory.id),
                ):
                    await store.create_relation(
                        RelationEdge(
                            org_id=org_id,
                            space_id=space_id,
                            source_id=source_id,
                            target_id=target_id,
                            type=RelationType.CONTRADICTS,
                            reason=conflict.reason,
                            confidence=conflict.confidence,
                        )
                    )
                conflicts.append(conflict.right_id)

            for association in association_proposals:
                # One direction only. `references` is read through
                # `list_relations` in both directions anyway, and writing
                # both would double an already-dense edge type.
                await store.create_relation(
                    RelationEdge(
                        org_id=org_id,
                        space_id=space_id,
                        source_id=memory.id,
                        target_id=association.right_id,
                        type=RelationType.REFERENCES,
                        reason=association.reason,
                        confidence=association.confidence,
                    )
                )

            for proposal in proposals:
                old = await store.get_memory(org_id, space_id, proposal.old_id)
                if old is None:
                    continue
                edge, updated_old = apply_supersession(memory, old, proposal)
                await store.create_relation(edge)
                await store.upsert_memory(updated_old)
                superseded.append(proposal.old_id)
                # A superseded source invalidates what was computed from it:
                # the derivation may now describe a replaced state of the world.
                await self._mark_derivations_stale(
                    org_id,
                    space_id,
                    await self._direct_derivatives(
                        org_id, space_id, proposal.old_id, store=store
                    ),
                    store=store,
                )

        stored = await store.upsert_memory(memory)
        INGESTED.labels(outcome="created").inc()

        written_claims: list[str] = []
        if claims is not None or extract:
            written_claims = await self._write_claims(
                stored,
                tags=tags,
                auto_supersede=auto_supersede,
                detect_conflicts=detect_conflicts,
                claims=claims,
            )

        return IngestResult(
            memory=stored,
            created=True,
            superseded=superseded,
            contradicts=conflicts,
            supersede_declined=declined,
            extracted=written_claims,
            chunk_count=len(chunks),
        )

    # -- bulk ingest -------------------------------------------------------

    async def ingest_many(
        self,
        *,
        org_id: str,
        space_id: str,
        items: Sequence[IngestItem],
        dedupe: bool = True,
        auto_supersede: bool = True,
        detect_conflicts: bool = True,
        extract: bool = True,
        stats: BulkStats | None = None,
    ) -> list[BulkOutcome]:
        """Write many memories with one embedding pass. Per-item outcomes.

        Three stages, because the expensive one parallelises and the other
        two must not:

          1. PREPARE every item and settle, with two batched lookups, the ones
             that need no vectors at all -- an unchanged keyed item, an exact
             restatement of an existing memory.
          2. EMBED every remaining chunk in one pass: batches of the
             provider's batch size, `embedding_concurrency` at a time. A
             100-item bulk used to make ~100 sequential embedding calls; this
             makes ceil(chunks / batch_size).
          3. WRITE in request order, with the vectors in hand. In order on
             purpose: dedupe, keys and supersession all compare a write
             against what is already stored, so item 7 must see item 6. Each
             write re-checks what stage 1 assumed, which makes stage 1 an
             optimisation and never a correctness input.

        Stage 3 decides each item with the same methods a single write uses.
        With `bulk_write_batching` (the default) they run against a
        `BulkView` inside ONE store transaction, which answers their reads
        from a single prefetch plus the earlier items' pending writes and
        applies every write at the end in a few multi-row statements -- about
        a dozen round trips for the request instead of ~35 per item. See
        `bulkwrite`. The sequential path remains for write-time extraction
        (a model call per item dominates anyway) and for the rare item the
        view hands back (`NeedsSequential`).

        One bad item no longer loses the whole response: validation, a
        failed embedding batch or a write error is reported against the
        items it touched, and every other item proceeds.

        `stats`, when given, is filled with where the time went.
        """
        stats = stats if stats is not None else BulkStats()
        started = time.perf_counter()
        stats.items = len(items)
        with similarity_memo(), entity_memo(), distinctive_memo():
            outcomes = await self._ingest_many(
                org_id=org_id,
                space_id=space_id,
                items=items,
                dedupe=dedupe,
                auto_supersede=auto_supersede,
                detect_conflicts=detect_conflicts,
                extract=extract,
                stats=stats,
            )
        stats.total_ms = (time.perf_counter() - started) * 1000
        log.info(
            "bulk_ingest",
            items=stats.items,
            created=sum(1 for o in outcomes if o.result is not None and o.result.created),
            failed=sum(1 for o in outcomes if o.error is not None),
            segments=stats.segments,
            sequential_items=stats.sequential_items,
            total_ms=round(stats.total_ms, 1),
            db_ms=round(stats.db_ms, 1),
            embed_ms=round(stats.embed_ms, 1),
            cpu_ms=round(stats.cpu_ms, 1),
        )
        return outcomes

    async def _ingest_many(
        self,
        *,
        org_id: str,
        space_id: str,
        items: Sequence[IngestItem],
        dedupe: bool,
        auto_supersede: bool,
        detect_conflicts: bool,
        extract: bool,
        stats: BulkStats,
    ) -> list[BulkOutcome]:
        clock = time.perf_counter()
        await self.get_space_or_raise(org_id, space_id)

        outcomes = [BulkOutcome(index=i) for i in range(len(items))]
        prepared: dict[int, _Prepared] = {}
        for i, item in enumerate(items):
            try:
                prepared[i] = self._prepare(
                    org_id,
                    space_id,
                    content=item.content,
                    summary=item.summary,
                    metadata=item.metadata,
                    tags=item.tags,
                    source=item.source,
                    occurred_at=item.occurred_at,
                    key=item.key,
                    replace=ReplaceMode(item.replace),
                )
            except MapiError as exc:
                outcomes[i].error = exc

        # One quota check for the request. Quota is account state, not a
        # property of an item, and checking per item would cost a usage
        # query for each one. The overshoot is bounded by one request, the
        # same trade quota.py already makes for concurrency.
        await self._check_quota(org_id, sum(p.size for p in prepared.values()))

        # -- stage 1: what needs no vectors ---------------------------------
        # Vectorless reads: this only asks WHETHER a key or digest is held.
        keys = [p.memory.key for p in prepared.values() if p.memory.key is not None]
        digests = [p.memory.content_sha256 for p in prepared.values() if p.memory.key is None]
        active_lookup = (
            self.store.get_active_by_keys(org_id, space_id, keys, with_embeddings=False)
            if keys
            else None
        )
        hash_lookup = (
            self.store.find_by_content_hashes(org_id, space_id, digests, with_embeddings=False)
            if dedupe and digests
            else None
        )
        active: dict[str, Memory] = {}
        by_hash: dict[str, Memory] = {}
        if active_lookup is not None and hash_lookup is not None:
            active, by_hash = await asyncio.gather(active_lookup, hash_lookup)
        elif active_lookup is not None:
            active = await active_lookup
        elif hash_lookup is not None:
            by_hash = await hash_lookup
        stats.db_ms += (time.perf_counter() - clock) * 1000

        needs_vectors: list[int] = []
        #: Items whose key or content an EARLIER item in this request also
        #: carries. Stage 1's lookups predate that earlier write, so these
        #: look again at write time -- and embed then, if they must.
        repeated: set[int] = set()
        seen_keys: set[str] = set()
        seen_hashes: set[str] = set()
        for i, p in prepared.items():
            key, digest = p.memory.key, p.memory.content_sha256
            if key is not None:
                current = active.get(key)
                if key in seen_keys:
                    repeated.add(i)
                elif current is None or not keyed_unchanged(current, p.memory):
                    needs_vectors.append(i)
                seen_keys.add(key)
            else:
                if dedupe and digest in seen_hashes:
                    repeated.add(i)
                elif not (dedupe and digest in by_hash):
                    needs_vectors.append(i)
                seen_hashes.add(digest)

        # -- stage 2: one embedding pass ------------------------------------
        clock = time.perf_counter()
        vectors, failures = await self._embed_many({i: prepared[i] for i in needs_vectors})
        stats.embed_ms += (time.perf_counter() - clock) * 1000

        # -- stage 3: writes, in request order ------------------------------
        for i in failures:
            outcomes[i].error = failures[i]
        order = [i for i in prepared if i not in failures]
        stage = _Stage3(
            prepared=prepared,
            vectors=vectors,
            active=active,
            by_hash=by_hash,
            repeated=repeated,
            outcomes=outcomes,
            dedupe=dedupe,
            auto_supersede=auto_supersede,
            detect_conflicts=detect_conflicts,
            extract=extract,
            stats=stats,
        )
        if self.settings.bulk_write_batching and self.extractor is None:
            await self._write_batched(org_id, space_id, order, stage)
        else:
            await self._write_sequential(order, stage)
        return outcomes

    async def _write_item(
        self, i: int, stage: _Stage3, *, store: WriteStore | None = None
    ) -> None:
        """Stage 3 for one item: exactly what a single write would do.

        Against the real store, stage 1's snapshot says whether a key was
        already looked up; against a bulk view the key is read again, which
        costs nothing there and is fresher under the view's key locks.
        """
        p = stage.prepared[i]
        in_view = store is not None
        try:
            if p.memory.key is not None:
                stage.outcomes[i].result = await self._write_keyed(
                    p,
                    stage.vectors.get(i),
                    known=stage.active.get(p.memory.key),
                    looked_up=not in_view and i not in stage.repeated,
                    store=store,
                )
                return
            digest = p.memory.content_sha256
            if stage.dedupe and (i in stage.repeated or digest in stage.by_hash):
                # Re-read, never fold into stage 1's snapshot: an earlier
                # item may have superseded that row since, and merging
                # into the stale copy would write it back ACTIVE. Stage 1
                # only decided that this item could skip the embed pass;
                # if the row is no longer eligible, it embeds below.
                duplicate = await self._fold_exact_duplicate(p, store=store)
                if duplicate is not None:
                    stage.outcomes[i].result = duplicate
                    return
            item_vectors = stage.vectors.get(i)
            if item_vectors is None:
                clock = time.perf_counter()
                try:
                    item_vectors = await self._embed_prepared(p)
                finally:
                    stage.stats.embed_ms += (time.perf_counter() - clock) * 1000
                # Kept, so a batched segment that falls back to the
                # sequential path does not embed this item twice.
                stage.vectors[i] = item_vectors
            stage.outcomes[i].result = await self._write_unkeyed(
                p,
                item_vectors,
                dedupe=stage.dedupe,
                auto_supersede=stage.auto_supersede,
                detect_conflicts=stage.detect_conflicts,
                extract=stage.extract,
                store=store,
            )
        except MapiError as exc:
            stage.outcomes[i].error = exc

    async def _write_sequential(self, order: Sequence[int], stage: _Stage3) -> None:
        """Stage 3 one store call at a time: the path before batching."""
        clock = time.perf_counter()
        embedded = stage.stats.embed_ms
        for i in order:
            await self._write_item(i, stage)
        stage.stats.sequential_items += len(order)
        elapsed = (time.perf_counter() - clock) * 1000
        stage.stats.db_ms += max(0.0, elapsed - (stage.stats.embed_ms - embedded))

    async def _write_batched(
        self, org_id: str, space_id: str, order: list[int], stage: _Stage3
    ) -> None:
        """Stage 3 in as few transactions as the items allow -- usually one.

        Each segment is one `BulkSession`: lock the keys it writes, prefetch
        what its items will read, decide every item against a `BulkView`,
        apply. An item the view cannot take (`must_go_sequential`) ends the
        segment: what came before is applied and committed, that item is
        written sequentially, and a new segment picks up after it -- so it
        sees exactly the state the sequential path would have shown it.

        If anything goes wrong below the view -- `NeedsSequential` from a
        read it could not answer, or a failed `apply` -- the transaction
        rolls back, nothing of the segment has been written, and the segment
        is written sequentially instead. Vectors already computed are kept.
        """
        pos = 0
        limit = self.settings.consolidation_candidates

        def erases(i: int) -> bool:
            p = stage.prepared[i]
            return p.memory.key is not None and p.replace is ReplaceMode.ERASE

        while pos < len(order):
            # A keyed ERASE is always sequential (`BulkView.must_go_sequential`);
            # a run of them goes straight there, without a segment to end.
            run = pos
            while run < len(order) and erases(order[run]):
                run += 1
            if run > pos:
                await self._write_sequential(order[pos:run], stage)
                pos = run
                continue
            segment = order[pos:]
            stop: int | None = None
            stage.stats.segments += 1
            try:
                async with timed_session(
                    self.store.bulk_session(org_id, space_id, neighbour_limit=limit),
                    stage.stats,
                ) as raw:
                    session = TimedSession(raw, stage.stats)
                    view = BulkView(session, org_id, space_id, limit=limit)
                    keys = [
                        key
                        for i in segment
                        if (key := stage.prepared[i].memory.key) is not None
                    ]
                    unkeyed = [i for i in segment if stage.prepared[i].memory.key is None]
                    await session.lock_keys(keys)
                    await view.load(
                        keys=keys,
                        digests=(
                            [stage.prepared[i].memory.content_sha256 for i in unkeyed]
                            if stage.dedupe
                            else []
                        ),
                        queries=[stage.vectors[i][0] for i in unkeyed if i in stage.vectors],
                    )
                    for n, i in enumerate(segment):
                        p = stage.prepared[i]
                        if await view.must_go_sequential(p.memory, p.replace):
                            stop = pos + n
                            break
                        await self._write_item(i, stage, store=view)
                    try:
                        await session.apply(view.ops)
                    except Exception as exc:
                        raise NeedsSequential(
                            f"bulk apply failed: {type(exc).__name__}"
                        ) from exc
            except NeedsSequential as exc:
                log.warning(
                    "bulk_segment_sequential",
                    reason=str(exc)[:200],
                    items=len(segment),
                    cause=type(exc.__cause__).__name__ if exc.__cause__ else None,
                )
                for i in segment:
                    stage.outcomes[i].result = None
                    stage.outcomes[i].error = None
                await self._write_sequential(segment, stage)
                return
            if stop is None:
                return
            await self._write_sequential([order[stop]], stage)
            pos = stop + 1

    async def _embed_many(
        self, prepared: Mapping[int, _Prepared]
    ) -> tuple[dict[int, list[list[float]]], dict[int, MapiError]]:
        """Every chunk of every item, packed into provider-sized batches.

        Chunks are packed across item boundaries, so the call count is
        ceil(total chunks / batch size) however the chunks fall into items.
        A failed batch fails exactly the items with a chunk in it; the other
        batches' vectors are still used. Batches run `embedding_concurrency`
        at a time -- bounded, because unbounded fan-out is how one bulk
        request turns into a burst of 429s.
        """
        flat: list[tuple[int, int, str]] = [
            (i, ordinal, text)
            for i, p in prepared.items()
            for ordinal, text in enumerate(p.texts)
        ]
        size = max(1, self.embedder.batch_size)
        batches = [flat[start : start + size] for start in range(0, len(flat), size)]
        gate = asyncio.Semaphore(max(1, self.settings.embedding_concurrency))
        pieces: dict[int, dict[int, list[float]]] = {i: {} for i in prepared}
        failures: dict[int, MapiError] = {}

        async def run(batch: list[tuple[int, int, str]]) -> None:
            async with gate:
                try:
                    result = await self.embedder.embed([text for _, _, text in batch])
                except MapiError as exc:
                    error: MapiError = exc
                except Exception as exc:  # a provider bug, not a caller's
                    error = ProviderError(f"{self.embedder.name}: {type(exc).__name__}")
                else:
                    EMBEDDINGS.labels(provider=self.embedder.name, outcome="ok").inc(len(batch))
                    for (i, ordinal, _), vector in zip(batch, result.vectors, strict=True):
                        pieces[i][ordinal] = vector
                    return
            EMBEDDINGS.labels(provider=self.embedder.name, outcome="error").inc()
            INGESTED.labels(outcome="embed_error").inc()
            for i, _, _ in batch:
                failures.setdefault(i, error)

        await asyncio.gather(*(run(batch) for batch in batches))
        vectors = {
            i: [by_ordinal[n] for n in range(len(prepared[i].texts))]
            for i, by_ordinal in pieces.items()
            if i not in failures
        }
        return vectors, failures

    async def _write_claims(
        self,
        parent: Memory,
        *,
        tags: Sequence[str],
        auto_supersede: bool,
        detect_conflicts: bool,
        claims: Sequence[Claim] | None = None,
    ) -> list[str]:
        """Store the atomic claims a document contains, pointing back at it.

        ADDITIVE, never destructive: the parent stays exactly as written and
        stays retrievable. If extraction misses the one attribute a question
        needed, the original text still holds it. That is the difference
        between an upgrade and a lossy rewrite of the user's data.

        Each claim gets a `derived_from` edge to its parent, which is not
        bookkeeping -- it is what makes deletion propagate. Erase the source
        turn and every claim drawn from it goes stale by the machinery that
        already exists, instead of surviving as an orphaned assertion nobody
        can trace.

        Consolidation runs on the CLAIMS, which is the point of the whole
        exercise: cosine between two atomic facts measures whether they are
        the same claim, where cosine between two multi-topic blobs only
        measures whether they share a subject.
        """
        if claims is not None:
            # Pre-computed upstream. Extraction is one model round-trip per
            # document and depends on nothing but that document, so it
            # parallelises freely -- while the WRITES cannot, because
            # supersession compares each write against what already exists.
            # Splitting the two is what lets a bulk ingest saturate the API
            # without making its edges depend on scheduling order.
            found = list(claims)
        elif self.extractor is not None:
            # The parent's own event time is the anchor that turns "last
            # Thursday" into a date; without it the model has no reference.
            found = await extract_claims(
                parent.content,
                self.extractor,
                as_of=parent.occurred_at,
                subject=str(parent.metadata.get("subject") or DEFAULT_SUBJECT),
            )
        else:
            return []
        if not found:
            return []

        written: list[str] = []
        for claim in found:
            try:
                result = await self.ingest(
                    org_id=parent.org_id,
                    space_id=parent.space_id,
                    content=claim.fact,
                    occurred_at=parent.occurred_at,
                    source=parent.source,
                    tags=tags,
                    metadata={
                        **parent.metadata,
                        "extracted_from": parent.id,
                        "quote": claim.quote,
                    },
                    kind=MemoryKind.DERIVED,
                    auto_supersede=auto_supersede,
                    detect_conflicts=detect_conflicts,
                    # Not recursive: a claim is already atomic, and asking a
                    # model to decompose one sentence burns a call to return
                    # the sentence.
                    extract=False,
                )
            except MapiError:
                # One bad claim must not lose the parent write, which has
                # already succeeded and is the thing the caller asked for.
                continue
            if result.memory.id == parent.id:
                # The claim restated the parent almost verbatim, so dedupe
                # resolved it back to the parent itself. Real: a one-sentence
                # turn is already atomic and has nothing to decompose into.
                # There is no second memory here and nothing to link -- a
                # self-edge would claim a document was derived from itself.
                continue
            await self.store.create_relation(
                RelationEdge(
                    org_id=parent.org_id,
                    space_id=parent.space_id,
                    source_id=result.memory.id,
                    target_id=parent.id,
                    type=RelationType.DERIVED_FROM,
                    reason="extracted at write time",
                    confidence=1.0,
                )
            )
            written.append(result.memory.id)
        return written

    # -- reads -------------------------------------------------------------

    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory:
        memory = await self.store.get_memory(org_id, space_id, memory_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")
        return memory

    async def _mark_derivations_stale(
        self,
        org_id: str,
        space_id: str,
        source_ids: Sequence[str],
        *,
        store: WriteStore | None = None,
    ) -> list[str]:
        """Transitively mark every derivation of `source_ids` STALE.

        The invariant this enforces: a derivation must never outlive the
        evidence it was computed from. Erase a source episode and "you own 3
        bikes" derived from it is no longer known — serving it anyway would be
        a lie with provenance attached. STALE (not deletion) is the right
        response because the derivation is *recomputable* from the surviving
        sources, and its `derived_from` edges say exactly how.

        Transitive because derivations stack (a profile fact derived from a
        timeline derived from episodes): BFS over incoming DERIVED_FROM edges
        with a visited set, so diamond-shaped provenance terminates. Edges to
        the *removed* memory die with it in both backends, so callers collect
        the first hop BEFORE removal and pass it in; hops beyond the first are
        walked here, where the edges still exist.

        Only ACTIVE rows transition — a SUPERSEDED derivation stays superseded
        (its replacement is the current truth; do not resurrect it as merely
        stale). Returns the ids actually transitioned, for attestations.
        """
        store = store or self.store
        staled: list[str] = []
        queue = list(dict.fromkeys(source_ids))
        visited: set[str] = set()
        while queue:
            derived_id = queue.pop(0)
            if derived_id in visited:
                continue
            visited.add(derived_id)
            derived = await store.get_memory(org_id, space_id, derived_id)
            if derived is None:
                continue
            if derived.status is MemoryStatus.ACTIVE:
                await store.upsert_memory(
                    derived.model_copy(
                        update={
                            "status": MemoryStatus.STALE,
                            "version": derived.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
                staled.append(derived_id)
            # Filtered by the store, not after: the answer is the same, and a
            # bulk request's view can answer exactly this question.
            incoming = await store.list_relations(
                org_id, space_id, derived_id, direction="in", type=RelationType.DERIVED_FROM
            )
            queue.extend(e.source_id for e in incoming if e.type is RelationType.DERIVED_FROM)
        return staled

    async def _direct_derivatives(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        store: WriteStore | None = None,
    ) -> list[str]:
        """First-hop derivations of a memory, collected while its edges exist."""
        incoming = await (store or self.store).list_relations(
            org_id, space_id, memory_id, direction="in", type=RelationType.DERIVED_FROM
        )
        return sorted({e.source_id for e in incoming if e.type is RelationType.DERIVED_FROM})

    async def delete_memory(self, org_id: str, space_id: str, memory_id: str) -> None:
        # Collect BEFORE the delete: the edges naming the derivations die with
        # the memory (FK cascade on postgres, edge sweep in-memory).
        derivatives = await self._direct_derivatives(org_id, space_id, memory_id)
        if not await self.store.delete_memory(org_id, space_id, memory_id):
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")
        await self._mark_derivations_stale(org_id, space_id, derivatives)

    async def list_memories(
        self,
        org_id: str,
        space_id: str,
        *,
        filters: MemoryFilter,
        limit: int,
        cursor: str | None,
    ) -> Page:
        await self.get_space_or_raise(org_id, space_id)
        return await self.store.list_memories(
            org_id, space_id, filters=filters, limit=limit, cursor=cursor
        )

    async def link(
        self,
        org_id: str,
        space_id: str,
        *,
        source_id: str,
        target_id: str,
        relation: RelationType,
        reason: str = "",
    ) -> RelationEdge:
        """Create a typed relation edge, applying its side effects.

        Returns the edge, not the source memory: the source memory is
        unchanged by this operation. Writing an edge is a statement about the
        graph, and bumping the source's version for it would produce a version
        history full of entries whose content is identical.
        """
        if source_id == target_id:
            raise ValidationError("a memory cannot relate to itself", field="target_id")
        # Both endpoints must exist and be visible to this tenant. Checked
        # before any write so a relation can never dangle.
        await self.get_memory(org_id, space_id, source_id)
        target = await self.get_memory(org_id, space_id, target_id)

        edge = await self.store.create_relation(
            RelationEdge(
                org_id=org_id,
                space_id=space_id,
                source_id=source_id,
                target_id=target_id,
                type=relation,
                reason=reason,
            )
        )

        if relation is RelationType.SUPERSEDES:
            if target.status is not MemoryStatus.SUPERSEDED:
                await self.store.upsert_memory(
                    target.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "version": target.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
                await self._mark_derivations_stale(
                    org_id,
                    space_id,
                    await self._direct_derivatives(org_id, space_id, target_id),
                )
        elif relation.symmetric:
            # `contradicts` is mutual; recording one direction only would make
            # the answer depend on which memory you happened to look at.
            # create_relation is idempotent, so the reverse edge is safe to
            # write unconditionally.
            await self.store.create_relation(
                RelationEdge(
                    org_id=org_id,
                    space_id=space_id,
                    source_id=target_id,
                    target_id=source_id,
                    type=relation,
                    reason=reason,
                )
            )
        return edge

    async def list_relations(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        direction: str = "out",
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        await self.get_memory(org_id, space_id, memory_id)
        return await self.store.list_relations(
            org_id, space_id, memory_id, direction=direction, type=type
        )

    async def get_lineage(
        self, org_id: str, space_id: str, memory_id: str
    ) -> dict[str, object]:
        """The supersession lineage around a memory, in both directions.

        `ancestors` is what this memory replaced, oldest last. `successors` is
        what replaced it — non-empty means this memory is stale, and the last
        entry is the current head of truth.
        """
        await self.get_memory(org_id, space_id, memory_id)
        backward = await self.store.walk_supersession_chain(
            org_id, space_id, memory_id, direction="backward"
        )
        forward = await self.store.walk_supersession_chain(
            org_id, space_id, memory_id, direction="forward"
        )
        return {
            "memory_id": memory_id,
            "ancestors": [e.target_id for e in backward],
            "successors": [e.source_id for e in forward],
            "is_current": not forward,
            "head": forward[-1].source_id if forward else memory_id,
        }

    async def get_graph(self, org_id: str, space_id: str, *, limit: int = 300) -> Graph:
        """Every memory in a space plus the typed edges between them.

        The edges are the point. A graph built from write-time extraction is a
        forest: each memory has exactly one parent, the document it was pulled
        from, so the picture is disconnected stars and the only question it can
        answer is "where did this come from". Ours are relations BETWEEN
        memories -- what replaced what, what disagrees with what, what was
        computed from what -- so the picture answers "what does this system
        believe, and why".

        Nodes carry status, so a viewer can show a superseded fact greyed
        behind its replacement and a STALE derivation as a fact whose evidence
        moved. Both are states no extraction-time graph represents.
        """
        await self.get_space_or_raise(org_id, space_id)
        page = await self.store.list_memories(
            org_id,
            space_id,
            filters=MemoryFilter(statuses=frozenset(MemoryStatus)),
            limit=limit,
            cursor=None,
        )
        memories = page.items
        ids = [m.id for m in memories]
        edges = await self.store.get_relations_between(org_id, space_id, ids) if ids else []

        present = set(ids)
        # `contradicts` is stored as a symmetric pair; render one line, not two.
        seen: set[tuple[str, str, str]] = set()
        out_edges = []
        for e in edges:
            if e.source_id not in present or e.target_id not in present:
                continue
            key = (
                (min(e.source_id, e.target_id), max(e.source_id, e.target_id), e.type)
                if e.type is RelationType.CONTRADICTS
                else (e.source_id, e.target_id, e.type)
            )
            if key in seen:
                continue
            seen.add(key)
            out_edges.append(e)

        degree: dict[str, int] = {}
        for e in out_edges:
            degree[e.source_id] = degree.get(e.source_id, 0) + 1
            degree[e.target_id] = degree.get(e.target_id, 0) + 1

        return Graph(
            space_id=space_id,
            memories=memories,
            edges=out_edges,
            degree=degree,
            counts={
                "memories": len(memories),
                "edges": len(out_edges),
                "isolated": sum(1 for m in memories if degree.get(m.id, 0) == 0),
                "by_status": {
                    s.value: sum(1 for m in memories if m.status is s) for s in MemoryStatus
                },
                "by_kind": {
                    k.value: sum(1 for m in memories if m.kind is k) for k in MemoryKind
                },
                "by_edge": {
                    t.value: sum(1 for e in out_edges if e.type is t) for t in RelationType
                },
            },
        )

    async def get_memory_context(
        self, org_id: str, space_id: str, memory_id: str, *, limit_per_relation: int = 20
    ) -> MemoryContext:
        """The full relation neighborhood of one memory, resolved to content.

        This is the "entire context" read: one call answers what an agent
        otherwise assembles from four — the memory itself, whether it is still
        current (and what replaced it), what it replaced, what it was derived
        from, what was derived from it, and what it references or contradicts.

        Neighbors whose memory no longer resolves (erased, or outside this
        tenant's view) are silently omitted rather than surfaced as broken
        stubs: an erased source must not leak even its existence through the
        context of a surviving derivative.
        """
        memory = await self.get_memory(org_id, space_id, memory_id)
        lineage = await self.get_lineage(org_id, space_id, memory_id)

        async def resolve(ids: list[str]) -> list[Memory]:
            out: list[Memory] = []
            for candidate in ids[:limit_per_relation]:
                found = await self.store.get_memory(org_id, space_id, candidate)
                if found is not None:
                    out.append(found)
            return out

        outgoing = await self.store.list_relations(org_id, space_id, memory_id, direction="out")
        incoming = await self.store.list_relations(org_id, space_id, memory_id, direction="in")

        def ids_of(edges: list[RelationEdge], kind: RelationType, *, source: bool) -> list[str]:
            return [(e.source_id if source else e.target_id) for e in edges if e.type is kind]

        is_current = bool(lineage["is_current"])
        return MemoryContext(
            memory=memory,
            is_current=is_current,
            current_head=[] if is_current else await resolve([str(lineage["head"])]),
            replaced=await resolve(ids_of(outgoing, RelationType.SUPERSEDES, source=False)),
            derived_from=await resolve(
                ids_of(outgoing, RelationType.DERIVED_FROM, source=False)
            ),
            derivatives=await resolve(ids_of(incoming, RelationType.DERIVED_FROM, source=True)),
            references=await resolve(ids_of(outgoing, RelationType.REFERENCES, source=False)),
            contradicts=await resolve(
                ids_of(outgoing, RelationType.CONTRADICTS, source=False)
                + ids_of(incoming, RelationType.CONTRADICTS, source=True)
            ),
        )

    async def get_memory_as_of(
        self, org_id: str, space_id: str, memory_id: str, as_of: datetime
    ) -> Memory:
        memory = await self.store.get_memory_as_of(org_id, space_id, memory_id, as_of)
        if memory is None:
            raise NotFoundError(
                f"memory {memory_id} did not exist at {as_of.isoformat()}",
                field="as_of",
            )
        return memory

    async def list_memory_versions(
        self, org_id: str, space_id: str, memory_id: str
    ) -> list[MemoryVersion]:
        versions = await self.store.list_memory_versions(org_id, space_id, memory_id)
        if not versions:
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")
        return versions

    async def erase_memory(
        self, org_id: str, space_id: str, memory_id: str
    ) -> dict[str, object]:
        """Right-to-erasure purge with an attestation of what was destroyed.

        Unlike delete (which preserves the audit trail), erase removes the
        content everywhere it can be reached: live row, chunks and embeddings,
        every edge touching it, and the full version history — so point-in-time
        reads cannot resurrect it. The attestation records the content hash
        (proof of WHICH content was destroyed, without retaining the content),
        counts of everything purged, and the ids of memories that had claimed
        derivation from the erased one, so a compliance process can review the
        blast radius.

        Works on live AND already-deleted memories: delete leaves history
        behind by design, and an erasure request must be able to purge that
        residue too.

        Idempotent, as DECISIONS.md has always said it is: erasing what is
        not there reports `already_erased` instead of a 404. A retried
        compliance request -- the response to the first attempt lost in
        transit -- must not look like a failure, and a memory in another
        tenant reads exactly the same way, so nothing leaks.
        """
        memory = await self.store.get_memory(org_id, space_id, memory_id)
        versions = await self.store.list_memory_versions(org_id, space_id, memory_id)
        if memory is None and not versions:
            return {
                "memory_id": memory_id,
                "space_id": space_id,
                "content_sha256": "",
                "chunks_removed": 0,
                "edges_removed": 0,
                "edges_bridged": 0,
                "versions_purged": 0,
                "derived_memories_affected": [],
                "erased_at": utcnow().isoformat(),
                "already_erased": True,
            }

        # The hash must be captured BEFORE the purge; afterwards there is
        # nothing left to hash. Prefer the live row, fall back to the last
        # historical snapshot for the deleted-but-remembered case.
        if memory is not None:
            digest = memory.content_sha256
        else:
            from .domain.models import content_hash

            digest = content_hash(versions[-1].content)

        derivatives: list[str] = []
        if memory is not None:
            derivatives = await self._direct_derivatives(org_id, space_id, memory_id)

        report = await self.store.erase_memory(org_id, space_id, memory_id)
        # The derivations named in the attestation are not merely "affected":
        # they are marked STALE, transitively, so nothing computed from the
        # erased content is ever served as current truth again.
        await self._mark_derivations_stale(org_id, space_id, derivatives)
        INGESTED.labels(outcome="erased").inc()
        log.info(
            "memory_erased",
            memory_id=memory_id,
            versions_purged=report.versions_purged,
            edges_removed=report.edges_removed,
        )
        return {
            "memory_id": memory_id,
            "space_id": space_id,
            "content_sha256": digest,
            "chunks_removed": report.chunks_removed,
            "edges_removed": report.edges_removed,
            "edges_bridged": report.edges_bridged,
            "versions_purged": report.versions_purged,
            "derived_memories_affected": derivatives,
            "erased_at": utcnow().isoformat(),
            "already_erased": False,
        }

    # -- bulk erasure, retirement and purge --------------------------------

    async def _attest_bulk(
        self, org_id: str, space_id: str, report: BulkEraseReport, **identity: str
    ) -> dict[str, object]:
        """Stale the survivors' derivations, then describe what was destroyed."""
        await self._mark_derivations_stale(org_id, space_id, report.derivatives)
        if report.memory_ids:
            INGESTED.labels(outcome="erased").inc(len(report.memory_ids))
        log.info(
            "memories_erased",
            space_id=space_id,
            count=len(report.memory_ids),
            versions_purged=report.versions_purged,
        )
        return {
            **identity,
            "space_id": space_id,
            "memory_ids": report.memory_ids,
            "memories_erased": len(report.memory_ids),
            "chunks_removed": report.chunks_removed,
            "edges_removed": report.edges_removed,
            "edges_bridged": report.edges_bridged,
            "versions_purged": report.versions_purged,
            "derived_memories_affected": report.derivatives,
            "erased_at": utcnow().isoformat(),
        }

    async def erase_by_tag(self, org_id: str, space_id: str, tag: str) -> dict[str, object]:
        """Erase everything a tag has ever been on, in one transaction.

        The one-call answer to "forget everything about this person" in a
        shared space: a caller that tags each writer's memories (facemash
        tags directory cards `person:u-<id>`) can withdraw all of them
        without listing, paging and erasing one id at a time -- N+2 calls,
        any of which could fail half way.
        """
        normalized = tag.strip().casefold()
        if not normalized:
            raise ValidationError("tag must not be empty", field="tag")
        report = await self.store.erase_by_tag(org_id, space_id, normalized)
        return await self._attest_bulk(org_id, space_id, report, tag=normalized)

    async def erase_by_key(self, org_id: str, space_id: str, key: str) -> dict[str, object]:
        """Erase every memory that holds or ever held `key`, history included."""
        report = await self.store.erase_by_key(org_id, space_id, key)
        return await self._attest_bulk(org_id, space_id, report, key=key)

    async def retire_keys(
        self,
        org_id: str,
        space_id: str,
        keys: Sequence[str],
        *,
        status: MemoryStatus = MemoryStatus.ARCHIVED,
    ) -> tuple[list[Memory], list[str]]:
        """Archive the facts under `keys`: hidden from search, kept as history.

        For a sync client whose source dropped a section: the key's memory
        should stop answering questions, but nothing was asked to be
        forgotten, so this is not an erase. Returns (retired, keys with no
        active memory). The second list is not an error -- a retried retire
        finds nothing left to do.
        """
        if status is not MemoryStatus.ARCHIVED:
            raise ValidationError("keys can only be retired to 'archived'", field="status")
        await self.get_space_or_raise(org_id, space_id)
        retired = await self.store.retire_keys(org_id, space_id, keys, status=status)
        for memory in retired:
            # Archival hides a source exactly as deletion does, so what was
            # derived from it stops being served as current.
            await self._mark_derivations_stale(
                org_id, space_id, await self._direct_derivatives(org_id, space_id, memory.id)
            )
        found = {m.key for m in retired}
        missing = [k for k in dict.fromkeys(keys) if k not in found]
        return retired, missing

    async def purge_space(self, org_id: str, space_id: str) -> PurgeReport:
        """Destroy a space and all of its history. See `MemoryStore.purge_space`.

        No existence check first, on purpose: purging a space that is
        already gone still has work to do (the version history DELETE left
        behind), and a retry must succeed with zeros rather than 404.
        """
        report = await self.store.purge_space(org_id, space_id)
        # The space lookup cache must never outlive the row it caches.
        self.forget_space(org_id, space_id)
        log.info(
            "space_purged",
            space_id=space_id,
            memories=report.memories,
            memory_versions=report.memory_versions,
        )
        return report

    async def chat(
        self,
        org_id: str,
        space_id: str,
        *,
        message: str,
        history: Sequence[ChatTurn] = (),
        k: int = 8,
        remember: bool = False,
    ) -> tuple[ChatAnswer, SearchResponse]:
        """Answer from this space's memories, with citations.

        Retrieval first, then one grounded completion. The search is the
        ordinary pipeline, not a special path -- so chat inherits question
        classification, coverage for comprehensive questions, supersession
        suppression and conflict surfacing, and improving any of those
        improves this for free.

        `remember` writes the user's message back as a memory. Off by default:
        a chat that silently records every question turns a question into a
        fact, and "is Krishna on an F-1 visa?" stored as a memory is a claim
        nobody made.
        """
        if self.completer is None:
            raise ProviderError("no synthesis backend configured; set synthesis_backend=gemini")
        # No space check of its own: `search` below makes it, and making it
        # twice was a second DB session per chat for the same answer.
        response = await self.search(self._chat_search_request(org_id, space_id, message, k))
        answer = await chat_answer(message, response.results, self.completer, history=history)
        if remember:
            await self.ingest(org_id=org_id, space_id=space_id, content=message)
        return answer, response

    # -- derivation and profiles -------------------------------------------

    async def derive(
        self,
        org_id: str,
        space_id: str,
        *,
        question: str,
        k: int = 10,
        materialize: bool = False,
        bucket: str | None = None,
    ) -> tuple[DerivedAnswer, Memory | None]:
        """Compute an answer from episodes; optionally store it as a memory.

        The read half runs map -> ground -> reduce over the top-k retrieved
        episodes (code does the arithmetic; every table row carries a verbatim
        quote checked against its source). The write half — `materialize` —
        stores the answer as a `kind=derived` memory with `derived_from` edges
        to every source, which buys three things nothing else in the market
        has together:

          * provenance: the answer names the episodes that made it true;
          * a demand-driven index: the next similar question is a lookup, and
            a wrong guess cost nothing because the episodes are still there;
          * invalidation: erase/supersede a source and the stored fact goes
            STALE via the existing lifecycle — it can never outlive its
            evidence.

        `bucket` tags the fact into a profile ("preferences", "dietary", ...).
        Re-deriving the same bucket+question SUPERSEDES the previous fact, so
        profiles update through the same revision machinery as everything
        else — history, `?as_of=`, and lineage included.
        """
        if self.completer is None:
            raise ProviderError(
                "no synthesis backend configured; set synthesis_backend=gemini "
                "or inject a completer"
            )
        response = await self.search(
            SearchRequest(query=question, org_id=org_id, space_id=space_id, limit=k)
        )
        docs = await self._hydrate(org_id, space_id, response.results)
        # The search already decided what this question is asking for -- reuse
        # it rather than classifying twice. They would usually agree, and
        # "usually" is exactly the kind of divergence nobody finds later.
        kind = response.intent.kind if response.intent else classify(question)
        derived = await derive_answer(question, kind, docs, self.completer)
        if derived is None:
            derived = DerivedAnswer(
                answer="", kind=kind, table=(), source_ids=(), computed=False
            )
        if not materialize or not derived.answer:
            return derived, None

        tags = [f"profile:{bucket}"] if bucket else []
        result = await self.ingest(
            org_id=org_id,
            space_id=space_id,
            content=derived.answer,
            summary=question,
            metadata={
                "question": question,
                "derive_kind": str(derived.kind),
                "computed": derived.computed,
                "table": [
                    {
                        "date": row.date.isoformat() if row.date else None,
                        "fact": row.fact,
                        "source_id": row.source_id,
                    }
                    for row in derived.table
                ],
            },
            tags=tags,
            source="derive",
            kind=MemoryKind.DERIVED,
            # A derived fact is not a duplicate of the episodes it summarises,
            # and near-dup collapsing against them would eat the derivation.
            dedupe=False,
        )
        stored = result.memory
        for source_id in derived.source_ids:
            await self.store.create_relation(
                RelationEdge(
                    org_id=org_id,
                    space_id=space_id,
                    source_id=stored.id,
                    target_id=source_id,
                    type=RelationType.DERIVED_FROM,
                    reason="derive: grounded extraction",
                )
            )
        # Same bucket + same question -> this fact replaces the previous one,
        # through the normal supersession machinery (status flip + edge), so
        # profile history is ordinary memory history.
        if bucket:
            previous = await self.list_memories(
                org_id,
                space_id,
                filters=MemoryFilter(tags=(f"profile:{bucket}".casefold(),)),
                limit=50,
                cursor=None,
            )
            for old in previous.items:
                if old.id == stored.id or old.kind is not MemoryKind.DERIVED:
                    continue
                if old.metadata.get("question") != question:
                    continue
                await self.store.create_relation(
                    RelationEdge(
                        org_id=org_id,
                        space_id=space_id,
                        source_id=stored.id,
                        target_id=old.id,
                        type=RelationType.SUPERSEDES,
                        reason="re-derived profile fact",
                    )
                )
                await self.store.upsert_memory(
                    old.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "version": old.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
        return derived, stored

    async def refresh_stale_derivations(
        self, org_id: str, space_id: str, *, budget: int = 20
    ) -> dict[str, object]:
        """Re-derive facts whose sources changed. Consolidation, done safely.

        Phase 2 marks a derivation STALE when any source is erased, deleted or
        superseded, but nothing re-computed it -- an open loop that left
        profiles permanently empty after the first invalidation. This closes
        it: find STALE derived memories, re-run the question that produced
        them against the surviving evidence, and supersede the stale fact with
        the fresh one.

        This is the same machinery competitors run as a "dream cycle", with the
        destructive half removed. Theirs merges facts into cleaner
        abstractions, resolves contradictions by newest timestamp, and deletes
        on `forgetAfter`. Ours only ADDS a recomputed fact and supersedes the
        one it replaces -- episodes are never touched, so a bad consolidation
        costs a re-derivation rather than the evidence.

        `budget` bounds LLM spend per call. Background consolidation is exactly
        how competitors' token bills became marketing liabilities, so the cost
        is capped and attributable rather than ambient.

        Safe to call repeatedly: a derivation that cannot be re-derived (its
        sources are gone entirely) stays STALE rather than being deleted or
        silently resurrected.
        """
        if self.completer is None:
            raise ProviderError("no synthesis backend configured")
        await self.get_space_or_raise(org_id, space_id)

        page = await self.store.list_memories(
            org_id,
            space_id,
            filters=MemoryFilter(statuses=frozenset({MemoryStatus.STALE})),
            limit=max(budget * 4, budget),
            cursor=None,
        )
        stale = [m for m in page.items if m.kind is MemoryKind.DERIVED][:budget]

        refreshed: list[str] = []
        abandoned: list[str] = []
        for memory in stale:
            question = str(memory.metadata.get("question") or memory.summary or "")
            if not question:
                # Nothing records what this fact answered, so it cannot be
                # recomputed. Leave it stale rather than guess.
                abandoned.append(memory.id)
                continue
            bucket = next(
                (t.split(":", 1)[1] for t in memory.tags if t.startswith("profile:")), None
            )
            try:
                _, stored = await self.derive(
                    org_id, space_id, question=question, materialize=True, bucket=bucket
                )
            except Exception:
                abandoned.append(memory.id)
                continue
            if stored is None:
                # The surviving evidence no longer supports an answer. Correct
                # outcome: the fact stays stale and the profile stays silent.
                abandoned.append(memory.id)
                continue
            if bucket is None:
                # Bucketed facts are superseded inside `derive`; unbucketed
                # ones need the link written here so lineage stays walkable.
                await self.store.create_relation(
                    RelationEdge(
                        org_id=org_id,
                        space_id=space_id,
                        source_id=stored.id,
                        target_id=memory.id,
                        type=RelationType.SUPERSEDES,
                        reason="re-derived after its sources changed",
                    )
                )
            current = await self.store.get_memory(org_id, space_id, memory.id)
            if current is not None and current.status is MemoryStatus.STALE:
                await self.store.upsert_memory(
                    current.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "version": current.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
            refreshed.append(stored.id)

        return {
            "examined": len(stale),
            "refreshed": refreshed,
            "abandoned": abandoned,
            "budget": budget,
        }

    async def get_profile(
        self, org_id: str, space_id: str, bucket: str, *, limit: int = 100
    ) -> list[Memory]:
        """Current facts in one profile bucket.

        ACTIVE only, on purpose: superseded facts are history (visible via
        versions/lineage), stale facts are pending re-derivation, and neither
        is something an agent should act on as current truth.
        """
        await self.get_space_or_raise(org_id, space_id)
        page = await self.list_memories(
            org_id,
            space_id,
            filters=MemoryFilter(tags=(f"profile:{bucket}".casefold(),)),
            limit=limit,
            cursor=None,
        )
        return [m for m in page.items if m.kind is MemoryKind.DERIVED]

    # -- search ------------------------------------------------------------

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        await self.get_space_or_raise(request.org_id, request.space_id)
        space_ms = (time.perf_counter() - started) * 1000
        with SEARCH_LATENCY.time():
            response = await self.pipeline.search(request)
        response.timings_ms["space_ms"] = round(space_ms, 2)
        for stage, ms in response.timings_ms.items():
            SEARCH_STAGE_LATENCY.labels(stage=stage.removesuffix("_ms")).observe(ms / 1000.0)
        return response

    # -- read path: one question, several spaces (B9) ---------------------------

    async def multi_search(
        self, query: str, requests: Sequence[SearchRequest]
    ) -> tuple[list[SearchResponse], bool, dict[str, float]]:
        """Run one question against several spaces, embedding it ONCE.

        Returns (per-request responses in order, whether the embedding
        failed, timings). Every space is checked against its request's org
        BEFORE anything is spent, so one foreign or missing id fails the whole
        call with the same 404 a single search gives -- partial answers would
        let a caller probe which ids exist elsewhere, target by target.

        The targets then run concurrently. Each is an ordinary `search` with
        the vector handed in, so everything a search does -- fusion, decay,
        supersession, caps, the score floor -- applies per target unchanged.
        """
        loop = asyncio.get_running_loop()
        timings: dict[str, float] = {}
        started = loop.time()
        for request in requests:
            await self.get_space_or_raise(request.org_id, request.space_id)
        timings["space_ms"] = round((loop.time() - started) * 1000, 2)

        t0 = loop.time()
        vector = await self.pipeline.embed_query(query)
        timings["embed_ms"] = round((loop.time() - t0) * 1000, 2)
        # Empty, not None, on failure: "tried and failed, go lexical" rather
        # than "not embedded yet", which would retry the provider per target.
        for request in requests:
            request.query_vector = vector if vector is not None else []

        responses = await asyncio.gather(*(self.search(r) for r in requests))
        timings["total_ms"] = round((loop.time() - started) * 1000, 2)
        return list(responses), vector is None, timings

    async def similar(
        self,
        org_id: str,
        space_id: str,
        *,
        source_space_id: str,
        source_memory_id: str | None = None,
        source_key: str | None = None,
        limit: int = 10,
        filters: MemoryFilter | None = None,
        max_per_source: int = 0,
        min_score: float = 0.0,
    ) -> tuple[str, SimilarResponse]:
        """Neighbours in `space_id` of a memory stored in `source_space_id`.

        Returns (the source memory's id, the response). BOTH spaces are checked
        against `org_id`, and the source is looked up only inside its own
        space, so neither side can reach across a tenant: a source in another
        org is a 404, indistinguishable from one that does not exist.

        No embedding call: the source's first chunk already has a vector.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        await self.get_space_or_raise(org_id, space_id)
        await self.get_space_or_raise(org_id, source_space_id)

        if source_key is not None:
            # Keyed memories are the write-path workstream's (migration 0006,
            # `MemoryStore.get_active_by_keys`). Looked up by name so this
            # route works the moment a backend has them, and says so plainly
            # until then instead of guessing at a key from metadata.
            lookup = getattr(self.store, "get_active_by_keys", None)
            if lookup is None:
                raise ValidationError(
                    "source_key needs keyed memories, which this server does not "
                    "have yet; pass source_memory_id",
                    field="source_key",
                )
            found: dict[str, Memory] = await lookup(org_id, source_space_id, [source_key])
            source = found.get(source_key)
            if source is None:
                raise NotFoundError(
                    f"no active memory with key {source_key!r} in space {source_space_id}",
                    field="source_key",
                )
            memory_id = source.id
        elif source_memory_id is not None:
            memory_id = source_memory_id
        else:  # pragma: no cover - the request schema requires exactly one
            raise ValidationError(
                "pass source_memory_id or source_key", field="source_memory_id"
            )

        vector = await self.store.chunk_embedding(org_id, source_space_id, memory_id)
        if vector is None:
            # Missing memory, or one stored without a vector: either way
            # there is nothing to look with, and the caller named it.
            raise NotFoundError(
                f"memory {memory_id} not found in space {source_space_id}",
                field="source_memory_id",
            )
        source_ms = (loop.time() - started) * 1000

        response = await find_similar(
            self.store,
            SimilarRequest(
                org_id=org_id,
                space_id=space_id,
                vector=vector,
                # Ids are global, so excluding it is a no-op unless the two
                # spaces coincide -- where it is its own nearest neighbour.
                source_memory_id=memory_id,
                limit=limit,
                filters=filters or MemoryFilter(),
                max_per_source=max_per_source,
                min_score=min_score,
                candidate_multiplier=self.settings.candidate_multiplier,
            ),
        )
        response.timings_ms["source_ms"] = source_ms
        response.timings_ms["total_ms"] = (loop.time() - started) * 1000
        response.timings_ms = {k: round(v, 2) for k, v in response.timings_ms.items()}
        return memory_id, response


__all__ = ["IngestResult", "MemoryService"]
