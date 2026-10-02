"""A `depends_on` relation: the source uses the target.

`references` also carries Mapi's own associations ("about the same thing"), so it cannot
mean "needs". A function using the functions whose output it needs, and a recipe using its
functions, are `depends_on`, and the dependency walk follows only that and `derived_from`.

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

_OLD = "type in ('supersedes','contradicts','derived_from','references')"
_NEW = "type in ('supersedes','contradicts','derived_from','references','depends_on')"


def upgrade() -> None:
    op.drop_constraint("ck_edges_type_valid", "relation_edges", type_="check")
    op.create_check_constraint("ck_edges_type_valid", "relation_edges", _NEW)


def downgrade() -> None:
    op.execute("DELETE FROM relation_edges WHERE type = 'depends_on'")
    op.drop_constraint("ck_edges_type_valid", "relation_edges", type_="check")
    op.create_check_constraint("ck_edges_type_valid", "relation_edges", _OLD)
