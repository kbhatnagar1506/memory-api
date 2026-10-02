"""Wire schemas.

Separate from domain models on purpose. The domain is free to grow fields
(embeddings, internal versions) without leaking them into a public contract, and
the API can rename or omit without touching the domain. `extra="forbid"` on
every request body means a typo in a field name is a 422 rather than a silently
ignored option — the kind of failure that otherwise costs an afternoon.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from pydantic_core.core_schema import ValidationInfo

from ..domain.models import (
    Memory,
    MemoryKind,
    MemoryStatus,
    MemoryVersion,
    RelationEdge,
    RelationType,
    Scope,
    ScoredMemory,
)

Slug = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$")
]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Response(BaseModel):
    model_config = ConfigDict(extra="ignore")


# -- spaces -------------------------------------------------------------------


class CreateSpaceRequest(Request):
    slug: Slug
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    description: Annotated[str, StringConstraints(max_length=2000)] = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpaceResponse(Response):
    id: str
    slug: str
    name: str
    description: str
    metadata: dict[str, Any]
    created_at: datetime
    memory_count: int | None = None


class SpaceListResponse(Response):
    items: list[SpaceResponse]


# -- memories -----------------------------------------------------------------


class CreateMemoryRequest(Request):
    content: Annotated[str, StringConstraints(min_length=1)]
    summary: Annotated[str, StringConstraints(max_length=2000)] = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[Annotated[str, StringConstraints(max_length=64)]] = Field(
        default_factory=list, max_length=64
    )
    source: Annotated[str, StringConstraints(max_length=500)] = ""
    occurred_at: datetime | None = None

    # There are no consolidation flags on this body, and that is the design.
    #
    # Deduplication, supersession, contradiction detection and claim
    # extraction all run on every write. A memory API whose graph is empty
    # unless the caller knew to ask for it is a memory API that does nothing
    # by default -- and nobody reads the docs to discover that the feature
    # they are paying for has an off switch that is on.
    #
    # These were opt-in for one reason: each needed a candidate scan, and a
    # cheap write path was the architectural bet. That cost is gone. The scan
    # was a page of the 256 newest memories with every embedding pulled
    # across the wire; it is now a bounded nearest-neighbour lookup over the
    # ANN index, and the same lookup serves all three checks at once.
    #
    # Cost control belongs at the account level, metered on what a tenant
    # actually consumes -- not as a per-request flag that makes the product
    # worse for everyone who leaves it alone.

    @field_validator("metadata")
    @classmethod
    def _metadata_is_shallow(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(v) > 64:
            raise ValueError("at most 64 metadata keys")
        for key, value in v.items():
            if len(key) > 128:
                raise ValueError(f"metadata key too long: {key[:32]}...")
            if isinstance(value, (dict, list)):
                raise ValueError(
                    f"metadata[{key!r}] must be a scalar; nested structures cannot "
                    "be indexed for filtering"
                )
        return v


class BulkCreateMemoryRequest(Request):
    items: list[CreateMemoryRequest] = Field(min_length=1, max_length=100)


class RelationResponse(Response):
    id: str
    source_id: str
    target_id: str
    type: RelationType
    reason: str
    confidence: float
    created_at: datetime

    @classmethod
    def from_domain(cls, edge: RelationEdge) -> RelationResponse:
        return cls(
            id=edge.id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            type=edge.type,
            reason=edge.reason,
            confidence=edge.confidence,
            created_at=edge.created_at,
        )


class RelationListResponse(Response):
    items: list[RelationResponse]


class LineageResponse(Response):
    """Where a memory sits in its supersession chain.

    `is_current` is the field most callers want: false means this memory has
    been replaced and `head` is the id of the fact that replaced it.
    """

    memory_id: str
    #: What this memory replaced, transitively. Most recent first.
    ancestors: list[str]
    #: What replaced this memory, transitively. Most recent last.
    successors: list[str]
    is_current: bool
    head: str


class MemoryVersionResponse(Response):
    """One historical state. `valid_to = null` marks the current version."""

    version: int
    content: str
    summary: str
    metadata: dict[str, Any]
    tags: list[str]
    source: str
    status: MemoryStatus
    #: Event time: when the thing happened.
    occurred_at: datetime
    #: System time: when our database believed this.
    valid_from: datetime
    valid_to: datetime | None

    @classmethod
    def from_domain(cls, version: MemoryVersion) -> MemoryVersionResponse:
        return cls(
            version=version.version,
            content=version.content,
            summary=version.summary,
            metadata=version.metadata,
            tags=version.tags,
            source=version.source,
            status=version.status,
            occurred_at=version.occurred_at,
            valid_from=version.valid_from,
            valid_to=version.valid_to,
        )


class DeriveRequest(Request):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)
    #: Store the answer as a `derived` memory with provenance edges.
    materialize: bool = False
    #: Profile bucket to tag the materialized fact into (e.g. "preferences").
    bucket: str | None = Field(default=None, min_length=1, max_length=64)


class DerivedRowResponse(Response):
    date: str | None
    fact: str
    quote: str
    source_id: str


class DeriveResponse(Response):
    """A computed answer plus the grounded table that proves it.

    `computed` is true when code (len, max, timedelta) produced the value —
    the model only found the rows. Empty `answer` means derivation could not
    help; callers fall back to plain search.
    """

    answer: str
    kind: str
    computed: bool
    #: True only when code did the arithmetic over >= 2 independently grounded
    #: rows. Provenance is persuasive and that is a hazard -- a wrong count
    #: wrapped in a table with source ids reads as more trustworthy, not less.
    #: Measured extraction is exactly right on 60% of cases, so an unverified
    #: answer is advisory even though it ships with a table.
    verified: bool = False
    #: Rows dropped for having no date while the question bounded itself in
    #: time. Non-zero means the aggregate may be undercounting because the
    #: SOURCE DATA lacks dates, not because the window excluded things.
    undated_dropped: int = 0
    table: list[DerivedRowResponse]
    source_ids: list[str]
    memory: MemoryResponse | None


class GraphNode(Response):
    """One memory as a graph node. `content` is a preview, not the full text."""

    id: str
    content: str
    kind: MemoryKind
    status: MemoryStatus
    occurred_at: datetime
    tags: list[str]
    #: How many typed relations touch this memory. Zero means an island --
    #: which in an extraction-built graph is most of them.
    degree: int


class GraphEdge(Response):
    source: str
    target: str
    type: RelationType
    reason: str
    confidence: float


class GraphResponse(Response):
    """A space as a graph of memories and the typed relations between them.

    Not a document->chunk tree. Every edge here is a claim about how two
    MEMORIES relate: what replaced what, what disagrees with what, what was
    computed from what.
    """

    space_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    counts: dict[str, Any]
    #: Set when the space has more memories than `limit`: pass it as `cursor` for the
    #: next page. Each page holds the edges leaving its memories.
    next_cursor: str | None = None


class ConsolidateResponse(Response):
    """What a consolidation pass did.

    `abandoned` is not a failure list: a derivation whose surviving evidence
    no longer supports an answer SHOULD stay stale, and the profile should
    stay silent about it.
    """

    examined: int
    refreshed: list[str]
    abandoned: list[str]
    budget: int


class ProfileResponse(Response):
    bucket: str
    facts: list[MemoryResponse]


class MemoryContextResponse(Response):
    """A memory resolved together with its whole relation neighborhood.

    `current_head` is non-empty exactly when `is_current` is false: the memory
    has been superseded and the head is what an agent should trust instead.
    Neighbors that no longer resolve (erased) are omitted, not stubbed — an
    erased source must not leak through the context of its derivative.
    """

    memory: MemoryResponse
    is_current: bool
    current_head: list[MemoryResponse]
    replaced: list[MemoryResponse]
    derived_from: list[MemoryResponse]
    derivatives: list[MemoryResponse]
    references: list[MemoryResponse]
    contradicts: list[MemoryResponse]
    depends_on: list[MemoryResponse] = Field(default_factory=list)
    dependents: list[MemoryResponse] = Field(default_factory=list)


class MemoryVersionListResponse(Response):
    memory_id: str
    items: list[MemoryVersionResponse]


class EraseAttestation(Response):
    """Proof of destruction for a right-to-erasure request.

    `content_sha256` identifies WHICH content was destroyed without retaining
    it. `versions_purged` covers the point-in-time history: after this
    response, `?as_of=` reads cannot resurrect the content at any timestamp.
    `derived_memories_affected` lists memories that had recorded a
    `derived_from` edge to the erased one — the review surface for whether
    derived content also needs action.
    """

    memory_id: str
    space_id: str
    content_sha256: str
    chunks_removed: int
    edges_removed: int
    #: SUPERSEDES edges re-created around the erased memory so revision
    #: chains survive it; carries surviving ids only, never erased content.
    edges_bridged: int
    versions_purged: int
    derived_memories_affected: list[str]
    erased_at: str


class MemoryResponse(Response):
    id: str
    space_id: str
    content: str
    summary: str
    metadata: dict[str, Any]
    tags: list[str]
    source: str
    kind: MemoryKind
    status: MemoryStatus
    occurred_at: datetime
    created_at: datetime
    updated_at: datetime
    version: int
    chunk_count: int

    @classmethod
    def from_domain(cls, memory: Memory) -> MemoryResponse:
        return cls(
            id=memory.id,
            space_id=memory.space_id,
            content=memory.content,
            summary=memory.summary,
            metadata=memory.metadata,
            tags=memory.tags,
            source=memory.source,
            kind=memory.kind,
            status=memory.status,
            occurred_at=memory.occurred_at,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            version=memory.version,
            chunk_count=len(memory.chunks),
        )


class CreateMemoryResponse(Response):
    memory: MemoryResponse
    created: bool
    duplicate_of: str | None = None
    duplicate_kind: str = "none"
    similarity: float = 0.0
    superseded: list[str] = Field(default_factory=list)
    #: Memories this write CONTRADICTS. Recorded as symmetric edges and
    #: reported; neither side is hidden, because either may be the true one.
    contradicts: list[str] = Field(default_factory=list)
    chunk_count: int = 0


class BulkCreateMemoryResponse(Response):
    items: list[CreateMemoryResponse]
    created: int
    duplicates: int


class MemoryListResponse(Response):
    items: list[MemoryResponse]
    next_cursor: str | None = None
    total: int | None = None


class LinkRequest(Request):
    target_id: str
    relation: RelationType
    reason: Annotated[str, StringConstraints(max_length=1000)] = ""


# -- search -------------------------------------------------------------------


class SearchRequestBody(Request):
    query: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    limit: int = Field(default=10, ge=1, le=100)
    tags: list[str] = Field(default_factory=list, max_length=32)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None
    statuses: list[MemoryStatus] = Field(default_factory=lambda: [MemoryStatus.ACTIVE])
    #: Restrict to episodic or derived memories. Empty means both.
    #:
    #: The distinction is worth exposing because it is the one that
    #: measurably changes answers: EPISODIC memories are what happened,
    #: DERIVED memories are what is true. A question about a preference wants
    #: claims; a question about an event wants the episode it came from, and
    #: a single ranked list mixing them serves whichever is lexically luckier.
    kinds: list[MemoryKind] = Field(default_factory=list, max_length=4)
    include_superseded: bool = False
    #: 1.0 = pure relevance, 0.0 = maximum diversity.
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    #: OFF, matching `SearchRequest.use_mmr` -- which it did not.
    #:
    #: This defaulted True while the field it feeds defaults False, and the
    #: reason recorded on that field is a measurement: MMR "changed no retrieval
    #: metric while costing 2.5x latency (61ms -> 110ms on LoCoMo)". So every
    #: HTTP search paid for a stage that had been measured not to help, and the
    #: documented decision was true of the library and false of the product.
    #:
    #: Still exposed, because MMR is the right tool for a corpus that genuinely
    #: accumulates restatements. Opt in per request.
    use_mmr: bool = False
    #: Bring each hit's dependencies with it: the pieces of its own memory the matched
    #: piece uses (code, JSON), and the memories it rests on (`depends_on`,
    #: `derived_from`), up to `dependency_depth` hops. Off by default: it costs a walk
    #: per hit, and a question about a fact needs none.
    with_dependencies: bool = False
    dependency_depth: int = Field(default=2, ge=1, le=5)
    #: OFF, for the same reason and on stronger evidence than MMR's.
    #:
    #: The reranker was the shipped default and is the WORST of the five
    #: configs in this repo's own ablation, measured over 500 corpora and 500
    #: questions (bench/results/lme-v24-base2):
    #:
    #:     config           full_recall@k   MRR     hit@k
    #:     vector_only          0.968       0.938   0.994
    #:     hybrid_rrf           0.952       0.945   0.996   <- this, with it off
    #:     hybrid_rerank        0.890       0.864   0.976   <- this, with it on
    #:
    #: It is not a trade: turning it off improves full recall by 6.2 points,
    #: MRR by 8.1 and hit@k by 2.0 at the same time. On 500 questions that is
    #: roughly 31 answers whose evidence was complete and got reordered out of
    #: the window. `full_recall@k` is the metric that matters here because it
    #: is CONJUNCTIVE -- 1.0 only when every evidence item survives -- and a
    #: reranker that promotes one good hit while dropping its partner scores
    #: well on hit@k and answers the question wrong.
    #:
    #: Still exposed, because a lexical reranker is the right tool when a query
    #: carries rare exact tokens. Opt in per request.
    use_rerank: bool = False
    use_decay: bool = True
    half_life_days: float = Field(default=180.0, gt=0, le=36500)
    #: Minimum COSINE similarity, not minimum fused score.
    #:
    #: See `retrieval/confidence.calibrated_score`: this is compared against a
    #: hit's `vector_score` when the vector arm ran. It used to be compared
    #: against the post-fusion `score`, whose scale depends on `use_rerank` --
    #: so `min_score=0.3` with reranking off silently discarded every result of
    #: every query.
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vector_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    lexical_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    #: Return the complete set rather than the best few.
    #:
    #: Leave unset and the question decides for itself: "what is our entire
    #: infrastructure" is asking what the territory contains, and a ranked
    #: top-10 answers a different question. Set it explicitly when you know
    #: better than the question does -- true widens the window to
    #: `coverage_limit`, false pins it to `limit`.
    coverage: bool | None = None
    #: Include per-stage score provenance on every hit.
    explain: bool = False

    # -- fields that existed on SearchRequest and could not be reached --------
    #
    # These eight were settable nowhere in `src/`, so they sat at their dataclass
    # defaults forever. Two consequences, both invisible:
    #
    #   * pipeline stage 4b (temporal scope) is gated on `asked_at is not None`,
    #     and nothing ever set it -- so the whole stage was dead code in the
    #     product while being measured and documented in the benchmark harness.
    #   * `use_expansion` and `use_entity_expansion` could not be turned on at
    #     all, so two features with measured write-ups were unreachable.
    #
    # Wiring them is not enabling them. Every default below is the previous
    # effective behaviour, EXCEPT `asked_at`, which the route now fills with the
    # current time -- see the route for why that is the safe direction.

    #: When the question was asked. Resolves "last March" and "three weeks ago"
    #: against the asker's clock rather than the server's wall time, and is what
    #: brings the temporal stage to life. Defaults to now at the route.
    asked_at: datetime | None = None
    #: Bias candidates whose event time falls inside a window the question names.
    #: Inert without `asked_at`; a bias, never a filter.
    use_temporal_scope: bool = True
    #: Let question shape pick the fusion weights (`retrieval/tuning.py`).
    tune_by_intent: bool = True
    #: HyDE: generate a hypothetical answer, embed it, average with the query.
    #:
    #: Costs an extra LLM call per search, which is why it stays off by default
    #: even now that it can be switched on. Requires a synthesis backend; with
    #: none configured the expander is a no-op and this flag does nothing.
    use_expansion: bool = False
    #: Bridge to memories sharing a salient entity with the query. Costs one
    #: extra store lookup per entity, so also opt-in.
    use_entity_expansion: bool = False
    entity_budget: int = Field(default=6, ge=0, le=32)
    entity_weight: float = Field(default=0.6, ge=0.0, le=10.0)
    #: Speaker names to exclude from entity extraction -- in a dialogue corpus
    #: the participants are the most frequent proper nouns and the least
    #: discriminating.
    known_speakers: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("occurred_before")
    @classmethod
    def _range_is_ordered(cls, v: datetime | None, info: ValidationInfo) -> datetime | None:
        after = info.data.get("occurred_after")
        if v is not None and after is not None and v < after:
            raise ValueError("occurred_before must not precede occurred_after")
        return v


class IntentBlock(Response):
    """What the question was read as asking for, and who read it that way."""

    kind: str
    #: "llm", "cache", "rules" or "explicit". A comprehensive question comes
    #: back a different shape, and that decision should be visible rather
    #: than inferred from the result count.
    source: str
    #: True when the window was widened to return a complete set.
    comprehensive: bool


class ChunkDependency(Response):
    """A piece of the hit's own memory that the matched piece uses."""

    ordinal: int
    text: str


class DependencyRef(Response):
    """A memory the hit depends on, `depth` hops away along dependency edges."""

    id: str
    summary: str
    content: str
    depth: int


class SearchHit(Response):
    memory: MemoryResponse
    score: float
    matched_text: str = ""
    #: With `with_dependencies`: what the matched piece and its memory depend on,
    #: dependencies first.
    chunk_dependencies: list[ChunkDependency] | None = None
    dependencies: list[DependencyRef] | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    recency_factor: float | None = None
    explain: list[str] | None = None

    @classmethod
    def from_domain(cls, scored: ScoredMemory, *, explain: bool) -> SearchHit:
        return cls(
            memory=MemoryResponse.from_domain(scored.memory),
            score=round(scored.score, 6),
            matched_text=scored.matched_text,
            vector_score=scored.vector_score,
            lexical_score=scored.lexical_score,
            fusion_score=scored.fusion_score,
            rerank_score=scored.rerank_score,
            recency_factor=scored.recency_factor,
            explain=scored.explain if explain else None,
        )


class ConfidenceBlock(Response):
    level: str
    top_score: float
    margin: float
    n_results: int
    has_conflicts: bool
    reason: str
    refusal_reason: str | None = None


class ChunkNode(Response):
    ordinal: int
    depends_on: list[int]
    token_estimate: int


class DependencyNode(Response):
    memory: MemoryResponse
    depth: int
    #: The memory's pieces and which use which: its own internal dependency graph.
    chunks: list[ChunkNode]


class DependenciesResponse(Response):
    """A memory's dependency closure: `nodes` in `order`, dependencies first and the
    memory itself last, with the edges between them."""

    memory_id: str
    order: list[str]
    nodes: list[DependencyNode]
    edges: list[RelationResponse]
    truncated: bool


class SearchResponseBody(Response):
    query: str
    results: list[SearchHit]
    count: int
    total_candidates: int
    strategies: list[str]
    #: True when reranking was requested but fell back to first-stage order.
    rerank_degraded: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    #: Pairs of returned memory ids that CONTRADICT each other. Surfaced,
    #: never silently resolved: an agent told two facts disagree can ask the
    #: user, an agent handed the newest one cannot.
    conflicts: list[list[str]] = Field(default_factory=list)
    #: How far this evidence supports asserting an answer. Computed from the
    #: score distribution, not asked of a model -- a model's certainty is a
    #: property of its tone. `refusal_reason` distinguishes correct silence
    #: ("no_relevant_memory") from a real gap ("weak_evidence").
    confidence: ConfidenceBlock | None = None
    #: How the question was read. Present whenever a search ran.
    intent: IntentBlock | None = None


# -- chat ---------------------------------------------------------------------


class ChatTurn(Request):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=8000)]


class ChatRequest(Request):
    message: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    #: Prior turns, oldest first. Bounded server-side -- the memories are the
    #: point, and an unbounded transcript crowds out the retrieval it exists
    #: to support.
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    #: Memories retrieved as context for the answer.
    k: int = Field(default=8, ge=1, le=50)
    #: Store the message as a memory too. Off by default: a chat that records
    #: every question turns questions into facts, and "is he on an F-1 visa?"
    #: stored as a memory is a claim nobody made.
    remember: bool = False


class ChatCitation(Response):
    id: str
    content: str
    score: float
    occurred_at: datetime
    tags: list[str] = Field(default_factory=list)
    #: True when the answer actually cited this memory, false when it was
    #: retrieved and passed over. Both are worth showing.
    cited: bool = False
    #: From the memory's own metadata. "unverified" means the answer built on
    #: it should be read with the same caveat.
    confidence: str = ""


class ChatResponse(Response):
    reply: str
    #: Cited memories first, in citation order, then the rest.
    citations: list[ChatCitation] = Field(default_factory=list)
    #: True when the answer rests on a memory marked unverified.
    used_unverified: bool = False
    #: How the question was read: direct, list_all, count, order, ...
    intent: str | None = None
    conflicts: list[list[str]] = Field(default_factory=list)


# -- keys ---------------------------------------------------------------------


class CreateApiKeyRequest(Request):
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    scopes: list[Scope] = Field(default_factory=list)
    expires_at: datetime | None = None
    #: Limit the key to these spaces (one tenant's own). Omitted: every space of the
    #: organization. An admin key can't be limited.
    space_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)


class ApiKeyResponse(Response):
    id: str
    name: str
    prefix: str
    scopes: list[Scope]
    space_ids: list[str] | None = None
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None


class CreateApiKeyResponse(Response):
    key: ApiKeyResponse
    #: Shown exactly once. It is not recoverable afterwards.
    plaintext: str


class ApiKeyListResponse(Response):
    items: list[ApiKeyResponse]


# -- health -------------------------------------------------------------------


class HealthResponse(Response):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    checks: dict[str, bool]


__all__ = [
    "ApiKeyListResponse",
    "ApiKeyResponse",
    "BulkCreateMemoryRequest",
    "BulkCreateMemoryResponse",
    "CreateApiKeyRequest",
    "CreateApiKeyResponse",
    "CreateMemoryRequest",
    "CreateMemoryResponse",
    "CreateSpaceRequest",
    "EraseAttestation",
    "HealthResponse",
    "LineageResponse",
    "LinkRequest",
    "MemoryListResponse",
    "MemoryResponse",
    "MemoryVersionListResponse",
    "MemoryVersionResponse",
    "RelationListResponse",
    "RelationResponse",
    "SearchHit",
    "SearchRequestBody",
    "SearchResponseBody",
    "SpaceListResponse",
    "SpaceResponse",
]
