"""Keyed memories: `memory_key` on memories and their history.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-26

A key is a caller's name for ONE fact -- "card:stuck_on", a section of an
agent's notes -- and a write under an existing key replaces the memory that
holds it. Without this, a changed value was a new write judged by cosine
alone, and it lost two ways that were measured live: value changes merged
into the OLD text as near-duplicates (cosine 0.972-0.989 at the 0.97
threshold), and A -> B -> A resolved to the superseded A, hiding the value
the user had just restated.

Three objects:

  * `memories.memory_key`, nullable -- ordinary memories have no key and
    keep deduplicating by content.
  * `uq_memories_space_key_active`: at most one ACTIVE row per key per
    space. Partial, so superseded history under a key accumulates freely;
    the database holds the invariant rather than every writer agreeing.
  * `memory_versions.memory_key` plus an index on both tables, so erasing a
    key also reaches history whose live row was already deleted (delete
    keeps versions by design, and an erasure that leaves them is not one).

Additive: every existing row gets NULL, which is exactly "unkeyed", so no
backfill. Postgres adds a nullable column without rewriting the table.
Row-level security needs nothing new -- the policies are per table, not per
column.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("memory_key", sa.Text(), nullable=True))
    op.add_column("memory_versions", sa.Column("memory_key", sa.Text(), nullable=True))
    op.create_index(
        "uq_memories_space_key_active",
        "memories",
        ["space_id", "memory_key"],
        unique=True,
        postgresql_where=sa.text("status = 'active' AND memory_key IS NOT NULL"),
    )
    op.create_index(
        "ix_memories_key",
        "memories",
        ["org_id", "space_id", "memory_key"],
        postgresql_where=sa.text("memory_key IS NOT NULL"),
    )
    op.create_index(
        "ix_versions_key",
        "memory_versions",
        ["org_id", "space_id", "memory_key"],
        postgresql_where=sa.text("memory_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_versions_key", table_name="memory_versions")
    op.drop_index("ix_memories_key", table_name="memories")
    op.drop_index("uq_memories_space_key_active", table_name="memories")
    op.drop_column("memory_versions", "memory_key")
    op.drop_column("memories", "memory_key")
