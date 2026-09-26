"""Drop the row-level-security bypass policies.

0005 put two policies on every tenant table: `*_tenant_isolation`, which
admits rows whose `org_id` matches `app.org_id`, and `*_admin_bypass`, which
admits EVERY row when `app.bypass_rls = 'on'`. Postgres ORs permissive
policies together, so the second one is not an escape hatch for operators --
it is a switch any session can flip. A custom GUC needs no privilege to set:

    SET app.bypass_rls = 'on';
    SELECT content FROM memories;   -- every tenant's memories

That turns the backstop 0005 exists to provide into a one-statement bypass for
exactly the code it was meant to contain: a query that forgot its WHERE clause
is harmless, but an injected or careless `set_config('app.bypass_rls', ...)`
reads everything. The application role owns the tables and FORCE binds it, so
the policy was the only thing standing between it and the whole database.

Nothing uses it. No code under src/ or tests/ sets `app.bypass_rls`, and the
operations it was reserved for (migrations, backups, operator queries) belong
to a role granted BYPASSRLS explicitly, which is visible in `pg_roles` and
cannot be granted by the application to itself.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

from alembic import op

revision = "0007"
#: 0006 (keyed memories) was built on a parallel branch; with both merged the
#: history is one line, and `tests/unit/test_migrations.py` fails if it forks.
down_revision = "0006"
branch_labels = None
depends_on = None

#: The tables 0005 put policies on. Kept in step with it by the RLS lane, which
#: asserts the bypass is refused on every one of them.
TENANT_TABLES = ("spaces", "memories", "chunks", "relation_edges", "memory_versions")


def upgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_admin_bypass ON {table}")


def downgrade() -> None:
    # Restores 0005's policy verbatim. Downgrading re-opens the bypass; that
    # is what downgrade means, and it is why nothing should.
    for table in TENANT_TABLES:
        op.execute(
            f"""
            CREATE POLICY {table}_admin_bypass ON {table}
            USING (current_setting('app.bypass_rls', true) = 'on')
            WITH CHECK (current_setting('app.bypass_rls', true) = 'on')
            """
        )
