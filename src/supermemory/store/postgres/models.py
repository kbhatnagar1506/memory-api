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
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
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
            "status in ('active','superseded','archived')",
            name="ck_memories_status_valid",
        ),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(40), nullable=False)
    space_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
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
    relations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB().with_variant(JSON, "sqlite"), default=list, nullable=False
    )

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
    "OrganizationRow",
    "SpaceRow",
]
