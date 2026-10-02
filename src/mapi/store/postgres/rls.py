"""Row-level security on the tenant tables, the same policies migration 0005 creates.

`PostgresStore.initialize()` applies them to a schema it built itself (development, tests),
so a database made without the migrations is isolated the same way production is. Each step
runs only when it is missing: a store starting against a migrated database changes nothing
and takes no table lock.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

TENANT_TABLES = ("spaces", "memories", "chunks", "relation_edges", "memory_versions")


async def ensure_row_level_security(conn: Any) -> None:
    state = {
        name: (enabled, forced)
        for name, enabled, forced in (
            await conn.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE relname = ANY(:names) AND relkind = 'r'"
                ),
                {"names": list(TENANT_TABLES)},
            )
        ).all()
    }
    policies = {
        row[0]
        for row in (
            await conn.execute(
                text("SELECT policyname FROM pg_policies WHERE tablename = ANY(:names)"),
                {"names": list(TENANT_TABLES)},
            )
        ).all()
    }
    for table in TENANT_TABLES:
        if table not in state:
            continue
        enabled, forced = state[table]
        if not enabled:
            await conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        if not forced:
            await conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
        if f"{table}_tenant_isolation" not in policies:
            await conn.execute(
                text(
                    f"CREATE POLICY {table}_tenant_isolation ON {table} "
                    "USING (org_id = current_setting('app.org_id', true)) "
                    "WITH CHECK (org_id = current_setting('app.org_id', true))"
                )
            )
        if f"{table}_admin_bypass" not in policies:
            await conn.execute(
                text(
                    f"CREATE POLICY {table}_admin_bypass ON {table} "
                    "USING (current_setting('app.bypass_rls', true) = 'on') "
                    "WITH CHECK (current_setting('app.bypass_rls', true) = 'on')"
                )
            )


async def role_bypasses_row_level_security(conn: Any) -> bool:
    """Whether the connected role ignores the policies: a superuser or a BYPASSRLS role does,
    even on FORCEd tables. Then the application's own filters are the only isolation."""
    row = (
        await conn.execute(
            text("SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user")
        )
    ).first()
    return bool(row and row[0])


__all__ = ["TENANT_TABLES", "ensure_row_level_security", "role_bypasses_row_level_security"]
