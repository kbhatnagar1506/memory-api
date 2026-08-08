"""Initial schema: organizations, spaces, api keys, memories, chunks.

Creates the pgvector extension, the generated tsvector column for full-text
search, and an HNSW index for approximate nearest neighbours.

HNSW rather than IVFFlat: IVFFlat must be built against representative data and
degrades as the corpus grows past its list count, which means an index built on
an empty table is wrong from the first insert. HNSW has no training step.
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIMENSIONS = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "organizations",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "spaces",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "org_id",
            sa.String(40),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("meta", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("org_id", "slug", name="uq_spaces_org_slug"),
    )
    op.create_index("ix_spaces_org", "spaces", ["org_id"])

    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "org_id",
            sa.String(40),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False),
        sa.Column("scopes", sa.dialects.postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_hash"),
    )
    op.create_index("ix_api_keys_org", "api_keys", ["org_id"])

    op.create_table(
        "memories",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("org_id", sa.String(40), nullable=False),
        sa.Column(
            "space_id",
            sa.String(40),
            sa.ForeignKey("spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("meta", sa.dialects.postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "tags",
            sa.dialects.postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("source", sa.String(500), nullable=False, server_default=""),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "relations", sa.dialects.postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.CheckConstraint("version >= 1", name="ck_memories_version_positive"),
        sa.CheckConstraint(
            "status in ('active','superseded','archived')", name="ck_memories_status_valid"
        ),
    )
    op.create_index("ix_memories_tenant", "memories", ["org_id", "space_id"])
    op.create_index(
        "ix_memories_tenant_status_id", "memories", ["org_id", "space_id", "status", "id"]
    )
    op.create_index(
        "ix_memories_tenant_occurred", "memories", ["org_id", "space_id", "occurred_at"]
    )
    op.create_index(
        "ix_memories_content_hash", "memories", ["org_id", "space_id", "content_sha256"]
    )
    op.create_index("ix_memories_tags", "memories", ["tags"], postgresql_using="gin")
    op.create_index("ix_memories_metadata", "memories", ["meta"], postgresql_using="gin")

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "memory_id",
            sa.String(40),
            sa.ForeignKey("memories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("org_id", sa.String(40), nullable=False),
        sa.Column("space_id", sa.String(40), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSIONS)),
        sa.Column(
            "search_vector",
            sa.dialects.postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
        ),
        sa.UniqueConstraint("memory_id", "ordinal", name="uq_chunks_memory_ordinal"),
    )
    op.create_index("ix_chunks_memory", "chunks", ["memory_id"])
    op.create_index("ix_chunks_tenant", "chunks", ["org_id", "space_id"])
    op.create_index("ix_chunks_fts", "chunks", ["search_vector"], postgresql_using="gin")
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("memories")
    op.drop_table("api_keys")
    op.drop_table("spaces")
    op.drop_table("organizations")
