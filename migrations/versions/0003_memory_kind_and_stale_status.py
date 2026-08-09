"""Memory kind (episodic|derived) and the STALE status.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-09

Two additions that make derived memories first-class:

  * `kind` distinguishes ground-truth episodes from facts computed out of
    them. Every existing row is an episode, so the server default backfills
    correctly.
  * `stale` joins the status check constraint. A derived memory goes stale
    when any of its sources is erased, deleted, or superseded — a derivation
    must never outlive the evidence it was computed from, and staleness is a
    hard status (excluded from search like every non-active state), not a
    score penalty.

Both changes are additive and reversible without data loss: downgrading
drops the column and re-narrows the constraint after re-labelling any stale
rows as archived (the closest surviving semantics — hidden from search,
recoverable by a later migration, never silently re-activated).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("kind", sa.String(20), nullable=False, server_default="episodic"),
    )
    op.add_column(
        "memory_versions",
        sa.Column("kind", sa.String(20), nullable=False, server_default="episodic"),
    )
    op.create_check_constraint(
        "ck_memories_kind_valid", "memories", "kind in ('episodic','derived')"
    )
    op.drop_constraint("ck_memories_status_valid", "memories", type_="check")
    op.create_check_constraint(
        "ck_memories_status_valid",
        "memories",
        "status in ('active','superseded','archived','stale')",
    )


def downgrade() -> None:
    # Stale rows must not become active on downgrade: archived is the closest
    # surviving semantics (hidden from search, recoverable, never silently
    # re-served as truth).
    op.execute("UPDATE memories SET status = 'archived' WHERE status = 'stale'")
    op.drop_constraint("ck_memories_status_valid", "memories", type_="check")
    op.create_check_constraint(
        "ck_memories_status_valid",
        "memories",
        "status in ('active','superseded','archived')",
    )
    op.drop_constraint("ck_memories_kind_valid", "memories", type_="check")
    op.drop_column("memory_versions", "kind")
    op.drop_column("memories", "kind")
