"""Users and memberships: identity separate from authorisation.

An API key authorises a request against an organization. It says nothing
about WHO made it -- two people on one team legitimately share one key -- so
a key can never be an identity. The email Google asserts is the identity, and
these two tables are where it lives.

Users are deliberately not scoped by org_id, unlike everything else in this
schema. A person exists before they belong to anything and belongs to
several organizations; the boundary they respect is their own id.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        # Google's stable subject id, and what we actually match on: an email
        # can be reassigned within a workspace, the subject cannot.
        sa.Column("google_sub", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False, server_default=""),
        sa.Column("picture", sa.String(500), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("google_sub", name="uq_users_google_sub"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "memberships",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(40),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "org_id",
            sa.String(40),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # One row per person per org. How MANY orgs a user may join is policy
        # and lives in the service; this is the invariant that holds whatever
        # the policy is.
        sa.UniqueConstraint("user_id", "org_id", name="uq_memberships_user_org"),
    )
    op.create_index("ix_memberships_user", "memberships", ["user_id"])
    op.create_index("ix_memberships_org", "memberships", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_memberships_org", table_name="memberships")
    op.drop_index("ix_memberships_user", table_name="memberships")
    op.drop_table("memberships")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
