"""Domain entities.

These are storage-agnostic and transport-agnostic: no SQLAlchemy, no FastAPI.
The Postgres models and the API schemas both convert to and from these, which is
what lets the same conformance suite validate two storage backends.

The relation model is the part worth reading. A memory system whose facts only
accumulate is wrong within a week of real use: people change their minds, plans
get replaced, and two sources disagree. So memories form a small typed graph —
`supersedes`, `contradicts`, `derived_from`, `references` — and retrieval uses
it to suppress facts that have been overtaken. See domain/consolidation.py.

`RelationEdge` is a first-class, independently identified entity rather than a
list embedded on `Memory`. Two reasons: an embedded list has no reverse index,
so "what supersedes X" is an O(n) scan of every memory's blob; and a typed graph
that lives in one place has one history, not two copies that can drift.
`MemoryStore.get_relations_between` and `walk_supersession_chain` are what a
graph structure buys over a blob — see store/base.py.

`MemoryVersion` makes the system bitemporal. Every mutation closes the
previous version's `valid_to` and opens a new one, so "what did our database
say as of time T" (system time) is answerable without disturbing "what
happened as of time T" (`Memory.occurred_at`, event time) — they answer
different questions and must not be conflated.
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
    #: A derived memory whose sources changed after it was computed. A stale
    #: derivation served as truth is a lie with provenance, so staleness is a
    #: hard status (excluded from default search like every non-ACTIVE state),
    #: not a score penalty. Distinct from SUPERSEDED because nothing replaced
    #: it — re-deriving from the surviving sources is the recovery, and the
    #: `derived_from` edges say exactly what to re-derive from.
    STALE = "stale"


class MemoryKind(StrEnum):
    #: Ground truth: something that happened, stored losslessly. Episodes are
    #: never invalidated by other memories — only corrected (versioning) or
    #: replaced (supersession) by the caller.
    EPISODIC = "episodic"
    #: Computed from episodes at read time (a count, a timeline, a profile
    #: fact). Carries `derived_from` edges to every source; goes STALE when
    #: any source is erased, deleted, or superseded, because a derivation
    #: must never outlive the evidence it was computed from.
    DERIVED = "derived"


class RelationType(StrEnum):
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    #: Mapi's own associations ("about the same thing") use this too, so it never means
    #: "needs": a dependency is DEPENDS_ON.
    REFERENCES = "references"
    #: The source uses the target: a function uses the functions whose output it needs, a
    #: recipe uses its functions. Written by callers, never inferred; the dependency walk
    #: follows it.
    DEPENDS_ON = "depends_on"

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


class User(Base):
    """A person, identified by the email Google asserts.

    THE EMAIL IS THE IDENTITY. An API key authorises a request against an
    organization; it says nothing about who is behind it, and two people on
    one team legitimately share one. Conflating the two -- treating a key as
    a user -- is how audit trails become useless and how revoking a person's
    access turns into revoking a service's.

    `google_sub` is Google's stable subject id and is what we actually match
    on: an email can be reassigned within a workspace, the subject cannot.
    The email is stored because it is what a human recognises.
    """

    id: str = Field(default_factory=lambda: new_id("user"))
    email: NonEmptyStr
    google_sub: NonEmptyStr
    name: str = ""
    picture: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    last_seen_at: datetime = Field(default_factory=utcnow)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        # Case-folded so one person cannot become two accounts by capitalising
        # their own address.
        cleaned = v.strip().casefold()
        if "@" not in cleaned:
            raise ValueError(f"not an email address: {v!r}")
        return cleaned

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not is_valid(v, "user"):
            raise ValueError(f"invalid user id: {v!r}")
        return v


class MemberRole(StrEnum):
    """What a member may do. Owner is the only role that can delete the org."""

    OWNER = "owner"
    MEMBER = "member"


class Membership(Base):
    """A user's place in an organization.

    Separate from both sides because the relationship carries its own facts --
    when they joined, in what role -- and because a user belongs to several
    organizations while an organization has several users.
    """

    id: str = Field(default_factory=lambda: new_id("membership"))
    user_id: str
    org_id: str
    role: MemberRole = MemberRole.MEMBER
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not is_valid(v, "membership"):
            raise ValueError(f"invalid membership id: {v!r}")
        return v


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


class RelationEdge(Base):
    """A typed, directed edge between two memories. Immutable once written.

    Edges are never updated and never hard-deleted in normal operation, which
    is what makes `created_at` a free, correct history of the graph: "what
    relations existed as of T" is just `WHERE created_at <= T`, no separate
    versioning needed for this entity.

    `SUPERSEDES` direction: `source_id` is the NEWER memory, `target_id` is the
    one it replaces — i.e. "source supersedes target". `CONTRADICTS` is
    symmetric and is written as a pair of edges, one in each direction, so the
    answer does not depend on which memory you happened to look up first.
    """

    id: str = Field(default_factory=lambda: new_id("edge"))
    org_id: str
    space_id: str
    source_id: str
    target_id: str
    type: RelationType
    #: Free-text justification, e.g. the LLM's reason for judging a conflict.
    reason: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not is_valid(v, "edge"):
            raise ValueError(f"invalid relation edge id: {v!r}")
        return v

    @model_validator(mode="after")
    def _no_self_loop(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError("a memory cannot relate to itself")
        return self


class Chunk(Base):
    """An embedded slice of a memory. Retrieval matches chunks, returns memories."""

    id: str = Field(default_factory=lambda: new_id("chunk"))
    memory_id: str
    ordinal: int = Field(ge=0)
    text: NonEmptyStr
    token_estimate: int = Field(default=0, ge=0)
    embedding: list[float] | None = None
    #: Ordinals of this memory's chunks that this one depends on (it uses a name they
    #: define): the dependency graph between the pieces of one function or file.
    depends_on: list[int] = Field(default_factory=list)

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
    kind: MemoryKind = MemoryKind.EPISODIC
    status: MemoryStatus = MemoryStatus.ACTIVE
    #: Caller-supplied event time. Defaults to ingestion time. Recency decay
    #: uses this, not created_at, so backfilled history ages correctly.
    occurred_at: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    content_sha256: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    #: Monotonic version, incremented on every mutation to this memory's own
    #: fields. Enables optimistic concurrency control via If-Match, and is the
    #: version number `MemoryVersion` snapshots correspond to. Attaching a
    #: relation to a memory does NOT bump this: an edge is a fact about the
    #: graph, not a change to the memory's own content.
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
        """`content_sha256` is DERIVED from content, always and only.

        This used to be guarded by `if not self.content_sha256`, and the guard
        was the bug. `Base` sets `validate_assignment=True`, so this validator
        does re-run when a field is assigned -- but with a hash already present
        the guard made it a no-op, and `memory.content = "something new"` kept
        the old digest.

        Nothing in `src/` sets this field to anything but the derived value, and
        nothing could reasonably want to: every reader either compares it
        (`find_by_content_hash`, the exact-duplicate gate) or publishes it (the
        erase attestation, which certifies WHICH content was destroyed). A hash
        that disagrees with its content is wrong in both roles -- it makes a
        revised memory dedupe against its own former text, and it makes the
        attestation describe something other than what was erased.

        So it is recomputed unconditionally. Note the remaining hole, which is
        pydantic's and not ours: `model_copy(update={"content": ...})` skips
        validation entirely, so it does NOT come through here. Use `revise()`
        below for that, and see `test_model_invariants.py` for the pin.
        """
        object.__setattr__(self, "content_sha256", content_hash(self.content))
        return self

    def revise(self, **changes: object) -> Self:
        """`model_copy` with derived fields kept honest.

        `model_copy` bypasses validators by design, so a copy that changes
        `content` carries the previous `content_sha256` -- which the conformance
        suite was doing, storing content "revised" under the hash of "original".
        This clears the digest before the copy so the constructor re-derives it.
        """
        if "content" in changes:
            changes.setdefault("content_sha256", "")
            copied = self.model_copy(update=changes)
            object.__setattr__(copied, "content_sha256", content_hash(str(changes["content"])))
            return copied
        return self.model_copy(update=changes)

    @property
    def is_retrievable(self) -> bool:
        return self.status is MemoryStatus.ACTIVE


class MemoryVersion(Base):
    """An immutable snapshot of a memory's own fields, one row per change.

    `valid_from`/`valid_to` are SYSTEM time — when this snapshot was the
    current row in our database — which is a different axis from `occurred_at`
    (event time, carried inside the snapshot). `valid_to is None` means this
    snapshot is the current one.

    Chunks and embeddings are deliberately NOT versioned here: duplicating
    vector blobs on every edit is expensive, and a historical snapshot is for
    audit/point-in-time reads, not for making old text searchable again. If
    that need arises later it is a clean, separate extension.
    """

    id: str = Field(default_factory=lambda: new_id("version"))
    memory_id: str
    org_id: str
    space_id: str
    version: int = Field(ge=1)
    content: str
    summary: str
    metadata: dict[str, Any]
    tags: list[str]
    source: str
    kind: MemoryKind = MemoryKind.EPISODIC
    status: MemoryStatus
    occurred_at: datetime
    valid_from: datetime
    valid_to: datetime | None = None

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        if not is_valid(v, "version"):
            raise ValueError(f"invalid memory version id: {v!r}")
        return v

    @classmethod
    def snapshot(cls, memory: Memory, *, valid_from: datetime) -> MemoryVersion:
        """The version row that captures `memory`'s current field values."""
        return cls(
            memory_id=memory.id,
            org_id=memory.org_id,
            space_id=memory.space_id,
            version=memory.version,
            content=memory.content,
            summary=memory.summary,
            metadata=memory.metadata,
            tags=memory.tags,
            source=memory.source,
            kind=memory.kind,
            status=memory.status,
            occurred_at=memory.occurred_at,
            valid_from=valid_from,
        )


class ApiKey(Base):
    id: str = Field(default_factory=lambda: new_id("key"))
    org_id: str
    name: NonEmptyStr
    #: Only the hash is ever stored. The plaintext is shown once, at creation.
    key_hash: str
    #: First characters of the plaintext, for identification in a UI.
    prefix: str
    scopes: frozenset[Scope]
    #: The spaces this key may touch; None is every space of its organization. A key handed
    #: to one tenant of a multi-tenant caller (AgentCompile: one space per customer) is
    #: scoped to that tenant's space, so it cannot read or write another's.
    space_ids: frozenset[str] | None = None
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
    "MemoryKind",
    "MemoryStatus",
    "MemoryVersion",
    "NonEmptyStr",
    "Organization",
    "RelationEdge",
    "RelationType",
    "Scope",
    "ScoredMemory",
    "Space",
    "content_hash",
    "utcnow",
]
