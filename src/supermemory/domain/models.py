"""Domain entities.

These are storage-agnostic and transport-agnostic: no SQLAlchemy, no FastAPI.
The Postgres models and the API schemas both convert to and from these, which is
what lets the same conformance suite validate two storage backends.

The relation model is the part worth reading. A memory system whose facts only
accumulate is wrong within a week of real use: people change their minds, plans
get replaced, and two sources disagree. So memories form a small typed graph —
`supersedes`, `contradicts`, `derived_from`, `references` — and retrieval uses
it to suppress facts that have been overtaken. See domain/consolidation.py.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..core.ids import is_valid, new_id

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


def utcnow() -> datetime:
    return datetime.now(UTC)


def content_hash(text: str) -> str:
    """Stable hash of normalized content, used for exact-duplicate detection.

    Normalization collapses whitespace and case so that two writes differing
    only in formatting are recognised as the same fact.
    """
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    #: A newer memory supersedes this one. Retrievable only on request.
    SUPERSEDED = "superseded"
    #: Soft-deleted. Never returned by search.
    ARCHIVED = "archived"


class RelationType(StrEnum):
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    REFERENCES = "references"

    @property
    def symmetric(self) -> bool:
        return self is RelationType.CONTRADICTS


class Scope(StrEnum):
    MEMORIES_READ = "memories:read"
    MEMORIES_WRITE = "memories:write"
    SPACES_READ = "spaces:read"
    SPACES_WRITE = "spaces:write"
    SEARCH = "search"
    ADMIN = "admin"

    @classmethod
    def all(cls) -> frozenset[Scope]:
        return frozenset(cls)


class Base(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
    )


class Organization(Base):
    id: str = Field(default_factory=lambda: new_id("org"))
    name: NonEmptyStr
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not is_valid(v, "org"):
            raise ValueError(f"invalid organization id: {v!r}")
        return v


class Space(Base):
    """A tenant-scoped namespace. Retrieval never crosses a space boundary."""

    id: str = Field(default_factory=lambda: new_id("space"))
    org_id: str
    slug: Annotated[
        str,
        StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$"),
    ]
    name: NonEmptyStr
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not is_valid(v, "space"):
            raise ValueError(f"invalid space id: {v!r}")
        return v


class Relation(Base):
    type: RelationType
    target_id: str
    #: Free-text justification, e.g. the LLM's reason for judging a conflict.
    reason: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)


class Chunk(Base):
    """An embedded slice of a memory. Retrieval matches chunks, returns memories."""

    id: str = Field(default_factory=lambda: new_id("chunk"))
    memory_id: str
    ordinal: int = Field(ge=0)
    text: NonEmptyStr
    token_estimate: int = Field(default=0, ge=0)
    embedding: list[float] | None = None

    @field_validator("embedding")
    @classmethod
    def _finite(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if not v:
            raise ValueError("embedding must not be empty")
        for x in v:
            if x != x or x in (float("inf"), float("-inf")):
                raise ValueError("embedding contains NaN or infinity")
        return v


class Memory(Base):
    id: str = Field(default_factory=lambda: new_id("memory"))
    org_id: str
    space_id: str
    content: NonEmptyStr
    #: Optional short summary used for reranking and display.
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    source: str = ""
    status: MemoryStatus = MemoryStatus.ACTIVE
    #: Caller-supplied event time. Defaults to ingestion time. Recency decay
    #: uses this, not created_at, so backfilled history ages correctly.
    occurred_at: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    content_sha256: str = ""
    relations: list[Relation] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    #: Monotonic version, incremented on every mutation. Enables optimistic
    #: concurrency control via If-Match.
    version: int = Field(default=1, ge=1)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not is_valid(v, "memory"):
            raise ValueError(f"invalid memory id: {v!r}")
        return v

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for tag in v:
            t = tag.strip().casefold()
            if not t:
                continue
            if len(t) > 64:
                raise ValueError(f"tag too long (max 64): {tag!r}")
            seen.setdefault(t, None)
        if len(seen) > 64:
            raise ValueError("at most 64 tags per memory")
        return list(seen)

    @model_validator(mode="after")
    def _fill_hash(self) -> Self:
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", content_hash(self.content))
        return self

    @property
    def is_retrievable(self) -> bool:
        return self.status is MemoryStatus.ACTIVE

    def superseded_by(self) -> list[str]:
        """Ids of memories that supersede this one."""
        return [r.target_id for r in self.relations if r.type is RelationType.SUPERSEDES]


class ApiKey(Base):
    id: str = Field(default_factory=lambda: new_id("key"))
    org_id: str
    name: NonEmptyStr
    #: Only the hash is ever stored. The plaintext is shown once, at creation.
    key_hash: str
    #: First characters of the plaintext, for identification in a UI.
    prefix: str
    scopes: frozenset[Scope]
    created_at: datetime = Field(default_factory=utcnow)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    def is_active(self, *, now: datetime | None = None) -> bool:
        current = now or utcnow()
        if self.revoked_at is not None:
            return False
        return not (self.expires_at is not None and self.expires_at <= current)

    def allows(self, scope: Scope) -> bool:
        return Scope.ADMIN in self.scopes or scope in self.scopes


class ScoredMemory(Base):
    """A retrieval result with its full score provenance.

    Every component is retained rather than collapsed into one number, because
    "why did this rank here" is the first question anyone asks of a retrieval
    system, and reconstructing it after the fact is impossible.
    """

    memory: Memory
    score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    recency_factor: float | None = None
    matched_chunk_id: str | None = None
    matched_text: str = ""
    #: Human-readable trace, e.g. ["vector rank 3", "lexical rank 1", "rrf 0.031"].
    explain: list[str] = Field(default_factory=list)


__all__ = [
    "ApiKey",
    "Base",
    "Chunk",
    "Memory",
    "MemoryStatus",
    "NonEmptyStr",
    "Organization",
    "Relation",
    "RelationType",
    "Scope",
    "ScoredMemory",
    "Space",
    "content_hash",
    "utcnow",
]
