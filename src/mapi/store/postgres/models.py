"""SQLAlchemy 2.0 schema.

Indexing decisions, which are most of what makes this backend work at scale:

  * **HNSW over IVFFlat** for the vector index. HNSW needs no training step, so
    it is correct on an empty table and stays correct as the corpus grows;
    IVFFlat built on a small table degrades badly once the data outgrows its
    lists and has to be rebuilt.
  * `vector_cosine_ops` because embeddings are L2-normalized on write, which
    makes cosine distance and inner product equivalent and lets the index answer
    the query the retrieval layer actually asks.
  * A **generated** tsvector column with a GIN index, rather than computing
    to_tsvector at query time. The generated column is maintained by Postgres on
    write and is the only form the planner can index.
  * Every tenant-scoped index leads with `(org_id, space_id)`. Retrieval is
    always inside one space, so a composite index prefixed that way is the
    difference between an index scan and a sequential scan of every tenant.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class OrganizationRow(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class UserRow(Base):
    """A person. Not scoped to any organization -- they exist before they
    belong to one, and belong to several."""

    __tablename__ = "users"
    __table_args__ = (
        # Google's subject id is the identity we match on: an email can be
        # reassigned inside a workspace, the subject cannot.
        UniqueConstraint("google_sub", name="uq_users_google_sub"),
        Index("ix_users_email", "email"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    google_sub: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False, server_default="")
    picture: Mapped[str] = mapped_column(String(500), nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MembershipRow(Base):
    __tablename__ = "memberships"
    __table_args__ = (
        # One row per person per org. The cap on how many orgs a user may
        # join is a policy decision and lives in the service; this constraint
        # is the invariant that must hold regardless of policy.
        UniqueConstraint("user_id", "org_id", name="uq_memberships_user_org"),
        Index("ix_memberships_user", "user_id"),
        Index("ix_memberships_org", "org_id"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    org_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False, server_default="member")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SpaceRow(Base):
    __tablename__ = "spaces"
    __table_args__ = (
        UniqueConstraint("org_id", "slug", name="uq_spaces_org_slug"),
        Index("ix_spaces_org", "org_id"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    org_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON, "sqlite"), default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_org", "org_id"),
        # Lookup is by hash on every authenticated request, so it must be unique
        # and indexed; anything else turns auth into a table scan.
        UniqueConstraint("key_hash", name="uq_api_keys_hash"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    org_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String).with_variant(JSON, "sqlite"), nullable=False
    )
    # NULL: every space of the org. Otherwise the only spaces the key may touch.
    space_ids: Mapped[list[str] | None] = mapped_column(
        ARRAY(String).with_variant(JSON, "sqlite"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryRow(Base):
    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_tenant", "org_id", "space_id"),
        # Supports the default listing (active memories, newest first) and the
        # cursor pagination that walks it.
        Index("ix_memories_tenant_status_id", "org_id", "space_id", "status", "id"),
        Index("ix_memories_tenant_occurred", "org_id", "space_id", "occurred_at"),
        # Exact-duplicate detection is a hot path on every write.
        Index("ix_memories_content_hash", "org_id", "space_id", "content_sha256"),
        Index(
            "ix_memories_tags",
            "tags",
            postgresql_using="gin",
        ),
        Index("ix_memories_metadata", "meta", postgresql_using="gin"),
        CheckConstraint("version >= 1", name="ck_memories_version_positive"),
        CheckConstraint(
            "status in ('active','superseded','archived','stale')",
            name="ck_memories_status_valid",
        ),
        CheckConstraint(
            "kind in ('episodic','derived')",
            name="ck_memories_kind_valid",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(40), nullable=False)
    space_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    kind: Mapped[str] = mapped_column(String(20), default="episodic", nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON, "sqlite"), default=dict, nullable=False
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String).with_variant(JSON, "sqlite"), default=list, nullable=False
    )
    source: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="memory",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="ChunkRow.ordinal",
    )


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_memory", "memory_id"),
        Index("ix_chunks_tenant", "org_id", "space_id"),
        Index(
            "ix_chunks_fts",
            "search_vector",
            postgresql_using="gin",
        ),
        UniqueConstraint("memory_id", "ordinal", name="uq_chunks_memory_ordinal"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    org_id: Mapped[str] = mapped_column(String(40), nullable=False)
    space_id: Mapped[str] = mapped_column(String(40), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Dimension is fixed at migration time. Changing embedding model dimensions
    # requires a migration and a re-index; there is no safe in-place change.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
    )

    memory: Mapped[MemoryRow] = relationship(back_populates="chunks")


class RelationEdgeRow(Base):
    """Typed graph edges between memories.

    Indexed in BOTH directions. The reverse index is the whole point: "what
    supersedes this memory" is the question retrieval asks on every result,
    and answering it from a JSONB blob on each memory meant scanning the
    table. Graph walks are per-hop indexed lookups rather than scans.

    The unique constraint makes `create_relation` idempotent at the database
    level, so a retried request cannot produce two parallel edges even if two
    workers race.
    """

    __tablename__ = "relation_edges"
    __table_args__ = (
        Index("ix_edges_source", "org_id", "space_id", "source_id", "type"),
        Index("ix_edges_target", "org_id", "space_id", "target_id", "type"),
        UniqueConstraint("space_id", "source_id", "target_id", "type", name="uq_edges_triple"),
        CheckConstraint("source_id <> target_id", name="ck_edges_no_self_loop"),
        CheckConstraint(
            "type in ('supersedes','contradicts','derived_from','references')",
            name="ck_edges_type_valid",
        ),
        CheckConstraint(
            "confidence >= 0 and confidence <= 1", name="ck_edges_confidence_range"
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(40), nullable=False)
    space_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MemoryVersionRow(Base):
    """Append-only bitemporal snapshots of a memory's own fields.

    `valid_from`/`valid_to` are SYSTEM time (when our database believed this),
    distinct from `occurred_at` inside the snapshot, which is EVENT time (when
    the thing happened). Keeping both is what lets the API answer "what did we
    know on the 3rd about what happened in January".

    No foreign key to `memories`: the audit trail must outlive the row it
    describes, so deleting a memory does not erase the history of what it
    said. That is deliberate, and the reason this table is not cascaded.
    """

    __tablename__ = "memory_versions"
    __table_args__ = (
        Index("ix_versions_lookup", "org_id", "space_id", "memory_id", "valid_from"),
        # Partial index over the single current row per memory: the common
        # "as of now" read never scans history.
        Index(
            "ix_versions_current",
            "memory_id",
            postgresql_where=text("valid_to IS NULL"),
        ),
        UniqueConstraint("memory_id", "version", name="uq_versions_memory_version"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    memory_id: Mapped[str] = mapped_column(String(40), nullable=False)
    org_id: Mapped[str] = mapped_column(String(40), nullable=False)
    space_id: Mapped[str] = mapped_column(String(40), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    kind: Mapped[str] = mapped_column(String(20), default="episodic", nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON, "sqlite"), default=dict, nullable=False
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String).with_variant(JSON, "sqlite"), default=list, nullable=False
    )
    source: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


#: Created out-of-band in the migration: SQLAlchemy cannot express HNSW options.
HNSW_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
"""

__all__ = [
    "HNSW_INDEX_DDL",
    "ApiKeyRow",
    "Base",
    "ChunkRow",
    "MemoryRow",
    "MemoryVersionRow",
    "OrganizationRow",
    "RelationEdgeRow",
    "SpaceRow",
]
