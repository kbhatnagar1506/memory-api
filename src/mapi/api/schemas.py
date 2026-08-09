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
    #: Collapse exact and near duplicates into the existing memory.
    dedupe: bool = True
    #: Let a newer conflicting memory mark older ones superseded. Off by
    #: default: hiding a user's data on a heuristic is not a safe default.
    auto_supersede: bool = False
    #: Detect memories this write CONTRADICTS. Costs a candidate scan on the
    #: write path, so it is opt-in; it never hides anything.
    detect_conflicts: bool = False

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
    include_superseded: bool = False
    #: 1.0 = pure relevance, 0.0 = maximum diversity.
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    use_mmr: bool = True
    use_rerank: bool = True
    use_decay: bool = True
    half_life_days: float = Field(default=180.0, gt=0, le=36500)
    min_score: float = Field(default=0.0, ge=0.0)
    vector_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    lexical_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    #: Include per-stage score provenance on every hit.
    explain: bool = False

    @field_validator("occurred_before")
    @classmethod
    def _range_is_ordered(cls, v: datetime | None, info: ValidationInfo) -> datetime | None:
        after = info.data.get("occurred_after")
        if v is not None and after is not None and v < after:
            raise ValueError("occurred_before must not precede occurred_after")
        return v


class SearchHit(Response):
    memory: MemoryResponse
    score: float
    matched_text: str = ""
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


# -- keys ---------------------------------------------------------------------


class CreateApiKeyRequest(Request):
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    scopes: list[Scope] = Field(default_factory=list)
    expires_at: datetime | None = None


class ApiKeyResponse(Response):
    id: str
    name: str
    prefix: str
    scopes: list[Scope]
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
