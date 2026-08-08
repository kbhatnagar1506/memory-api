"""Promote relations to an indexed edge table; add bitemporal version history.

Two changes, both about making temporal questions answerable:

1. `memories.relations` (JSONB) becomes the `relation_edges` table, indexed in
   both directions. The blob had no reverse index, so "what supersedes this
   memory" — asked on every search result — was a scan of every row's JSON.

2. `memory_versions` records an append-only snapshot per version, with system
   time (`valid_from`/`valid_to`) kept separate from event time
   (`occurred_at`). That makes "what did we believe as of T" answerable, which
   an in-place-overwrite table cannot answer at all.

The upgrade MIGRATES existing relation data rather than dropping it: the JSONB
column is read, expanded into edge rows, and only then dropped. A migration
that silently discards user data because it was inconvenient to carry is not
an acceptable upgrade path.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "relation_edges",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("org_id", sa.String(40), nullable=False),
        sa.Column(
            "space_id",
            sa.String(40),
            sa.ForeignKey("spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            sa.String(40),
            sa.ForeignKey("memories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_id",
            sa.String(40),
            sa.ForeignKey("memories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "space_id", "source_id", "target_id", "type", name="uq_edges_triple"
        ),
        sa.CheckConstraint("source_id <> target_id", name="ck_edges_no_self_loop"),
        sa.CheckConstraint(
            "type in ('supersedes','contradicts','derived_from','references')",
            name="ck_edges_type_valid",
        ),
        sa.CheckConstraint(
            "confidence >= 0 and confidence <= 1", name="ck_edges_confidence_range"
        ),
    )
    op.create_index(
        "ix_edges_source", "relation_edges", ["org_id", "space_id", "source_id", "type"]
    )
    op.create_index(
        "ix_edges_target", "relation_edges", ["org_id", "space_id", "target_id", "type"]
    )

    op.create_table(
        "memory_versions",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("memory_id", sa.String(40), nullable=False),
        sa.Column("org_id", sa.String(40), nullable=False),
        sa.Column("space_id", sa.String(40), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
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
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("memory_id", "version", name="uq_versions_memory_version"),
    )
    op.create_index(
        "ix_versions_lookup",
        "memory_versions",
        ["org_id", "space_id", "memory_id", "valid_from"],
    )
    op.create_index(
        "ix_versions_current",
        "memory_versions",
        ["memory_id"],
        postgresql_where=sa.text("valid_to IS NULL"),
    )

    # Carry existing data across before dropping anything.
    #
    # Edges: expand each memory's JSONB array into rows. `gen_random_uuid()`
    # would not match our id format, so ids are built with the 'edg_' prefix
    # and a random suffix; they only need to be unique, not sortable, since
    # these are backfilled rows with no meaningful creation order.
    op.execute(
        """
        INSERT INTO relation_edges
            (id, org_id, space_id, source_id, target_id, type, reason, confidence, created_at)
        SELECT
            'edg_' || substr(md5(random()::text || m.id || rel->>'target_id'), 1, 26),
            m.org_id,
            m.space_id,
            m.id,
            rel->>'target_id',
            rel->>'type',
            coalesce(rel->>'reason', ''),
            coalesce((rel->>'confidence')::float, 1.0),
            coalesce((rel->>'created_at')::timestamptz, now())
        FROM memories m
        CROSS JOIN LATERAL jsonb_array_elements(m.relations) AS rel
        WHERE jsonb_typeof(m.relations) = 'array'
          AND rel->>'target_id' IS NOT NULL
          -- A relation pointing at a deleted memory cannot become an edge:
          -- the foreign key would reject it, and a dangling edge is worse
          -- than a dropped one.
          AND EXISTS (SELECT 1 FROM memories t WHERE t.id = rel->>'target_id')
          AND rel->>'target_id' <> m.id
        ON CONFLICT ON CONSTRAINT uq_edges_triple DO NOTHING
        """
    )

    # Versions: seed one open snapshot per existing memory, so history starts
    # from the current state rather than from nothing.
    op.execute(
        """
        INSERT INTO memory_versions
            (id, memory_id, org_id, space_id, version, content, summary, meta,
             tags, source, status, occurred_at, valid_from, valid_to)
        SELECT
            'ver_' || substr(md5(random()::text || m.id), 1, 26),
            m.id, m.org_id, m.space_id, m.version, m.content, m.summary,
            m.meta, m.tags, m.source, m.status, m.occurred_at,
            coalesce(m.updated_at, m.created_at, now()),
            NULL
        FROM memories m
        ON CONFLICT ON CONSTRAINT uq_versions_memory_version DO NOTHING
        """
    )

    op.drop_column("memories", "relations")


def downgrade() -> None:
    op.add_column(
        "memories",
        sa.Column(
            "relations",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="[]",
        ),
    )
    # Fold edges back into the blob so a downgrade is not lossy either.
    op.execute(
        """
        UPDATE memories m
        SET relations = coalesce(sub.payload, '[]'::jsonb)
        FROM (
            SELECT source_id,
                   jsonb_agg(jsonb_build_object(
                       'type', type,
                       'target_id', target_id,
                       'reason', reason,
                       'confidence', confidence,
                       'created_at', to_char(created_at, 'YYYY-MM-DD"T"HH24:MI:SS.USOF')
                   )) AS payload
            FROM relation_edges
            GROUP BY source_id
        ) AS sub
        WHERE m.id = sub.source_id
        """
    )
    op.drop_index("ix_versions_current", table_name="memory_versions")
    op.drop_index("ix_versions_lookup", table_name="memory_versions")
    op.drop_table("memory_versions")
    op.drop_index("ix_edges_target", table_name="relation_edges")
    op.drop_index("ix_edges_source", table_name="relation_edges")
    op.drop_table("relation_edges")
