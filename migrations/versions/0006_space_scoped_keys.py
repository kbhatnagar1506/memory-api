"""Space-scoped API keys: a key may be limited to some spaces of its organization.

A caller that is itself multi-tenant (AgentCompile keeps one space per customer) hands each of
its tenants a key for that tenant's space only. NULL keeps today's meaning: every space of the
key's organization. Enforcement is in the API (`mapi.api.deps.space_access`), on every route
that names a space.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "api_keys", sa.Column("space_ids", postgresql.ARRAY(sa.String()), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("api_keys", "space_ids")
