"""Chunk dependencies: the pieces of one function or file as a dependency graph.

A chunk of structured content (code, JSON) that uses a name another chunk of the same memory
defines -- a step id it reads from, a function it calls -- records that chunk's ordinal, so a
retrieved piece can be followed to everything it needs. NULL for prose and for chunks written
before this.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks", sa.Column("depends_on", postgresql.ARRAY(sa.Integer()), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("chunks", "depends_on")
