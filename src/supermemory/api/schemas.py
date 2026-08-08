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

from ..domain.models import Memory, MemoryStatus, RelationType, Scope, ScoredMemory

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
    type: RelationType
    target_id: str
    reason: str
    confidence: float


class MemoryResponse(Response):
    id: str
    space_id: str
    content: str
    summary: str
    metadata: dict[str, Any]
    tags: list[str]
    source: str
    status: MemoryStatus
    occurred_at: datetime
    created_at: datetime
    updated_at: datetime
    version: int
    chunk_count: int
    relations: list[RelationResponse]

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
            status=memory.status,
            occurred_at=memory.occurred_at,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            version=memory.version,
            chunk_count=len(memory.chunks),
            relations=[
                RelationResponse(
                    type=r.type,
                    target_id=r.target_id,
                    reason=r.reason,
                    confidence=r.confidence,
                )
                for r in memory.relations
            ],
        )


class CreateMemoryResponse(Response):
    memory: MemoryResponse
    created: bool
    duplicate_of: str | None = None
    duplicate_kind: str = "none"
    similarity: float = 0.0
    superseded: list[str] = Field(default_factory=list)
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


class SearchResponseBody(Response):
    query: str
    results: list[SearchHit]
    count: int
    total_candidates: int
    strategies: list[str]
    #: True when reranking was requested but fell back to first-stage order.
    rerank_degraded: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


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
    "HealthResponse",
    "LinkRequest",
    "MemoryListResponse",
    "MemoryResponse",
    "RelationResponse",
    "SearchHit",
    "SearchRequestBody",
    "SearchResponseBody",
    "SpaceListResponse",
    "SpaceResponse",
]
