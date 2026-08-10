"""Row-level security: the database gets an opinion about tenants.

Until now isolation was entirely application code. Every query carries
`org_id` by hand, every one of them correctly -- and that is the problem.
A property enforced by 24 store methods agreeing with each other is a
property that holds until someone adds the 25th. The failure mode is a
cross-tenant read, which is the worst outcome this system has, and it is
invisible in review: the missing clause looks exactly like the clauses
around it.

So the same predicate is stated once more, in the one place that cannot be
forgotten. The application still filters by `org_id` -- the policies are a
backstop, not a replacement, and a query that forgets its WHERE clause now
returns nothing instead of returning someone else's memories.

WHICH TABLES. The five that hold tenant data and carry `org_id`:
`spaces`, `memories`, `chunks`, `relation_edges`, `memory_versions`.

Deliberately NOT covered:

  * `api_keys` -- authentication looks a key up BY HASH in order to discover
    which org it belongs to. There is no org context yet at that moment, so
    an org-scoped policy would make login impossible. Its isolation is the
    hash: an unguessable secret is the credential, and the row it finds is
    what establishes the tenant for everything after.
  * `organizations`, `users`, `memberships` -- identity, which is keyed on a
    Google subject id and spans orgs by design. A user belongs to several
    organizations; that is the product.

FORCE, not merely ENABLE. Postgres exempts a table's OWNER from its own
policies unless forced, and the application connects as the owner. Without
FORCE this migration would apply to nobody and read as protection.

The GUC is `app.org_id`, set per transaction by the store. Unset means
`current_setting(..., true)` returns NULL, every comparison is NULL, and
every row is filtered out -- fail-closed. A code path that forgets to scope
its session sees an empty database, which is loud, local, and safe.
"""

from __future__ import annotations

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

#: Tenant tables, each carrying an `org_id` column.
TENANT_TABLES = ("spaces", "memories", "chunks", "relation_edges", "memory_versions")


def upgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        # One policy for every command. Splitting SELECT from INSERT/UPDATE
        # would let a write land in a tenant a read cannot see, which is a
        # more confusing failure than either alone.
        #
        # WITH CHECK matters as much as USING: USING governs which rows are
        # visible, WITH CHECK governs which rows may be written. Without it,
        # a compromised or buggy path could INSERT a row into another org
        # even while being unable to read one back.
        op.execute(
            f"""
            CREATE POLICY {table}_tenant_isolation ON {table}
            USING (org_id = current_setting('app.org_id', true))
            WITH CHECK (org_id = current_setting('app.org_id', true))
            """
        )
        # An escape hatch for migrations, backups and operator queries, which
        # legitimately span tenants and run as a different role. Nothing the
        # application does uses this -- the app role is not BYPASSRLS.
        op.execute(
            f"""
            CREATE POLICY {table}_admin_bypass ON {table}
            USING (current_setting('app.bypass_rls', true) = 'on')
            WITH CHECK (current_setting('app.bypass_rls', true) = 'on')
            """
        )


def downgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_admin_bypass ON {table}")
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
