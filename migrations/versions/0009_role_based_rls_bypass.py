"""Row-level security has no setting-based bypass any more.

Migration 0005 let a session read every tenant by setting `app.bypass_rls = 'on'`, and any
session can set any custom setting: a single injected statement would have read all tenants.
Work that legitimately spans tenants (migrations, backups, operators) uses a database role
with BYPASSRLS instead, which the application's own role never has.

Revision ID: 0009
Revises: 0008
"""

from __future__ import annotations

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

TENANT_TABLES = ("spaces", "memories", "chunks", "relation_edges", "memory_versions")


def upgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_admin_bypass ON {table}")


def downgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(
            f"""
            CREATE POLICY {table}_admin_bypass ON {table}
            USING (current_setting('app.bypass_rls', true) = 'on')
            WITH CHECK (current_setting('app.bypass_rls', true) = 'on')
            """
        )
