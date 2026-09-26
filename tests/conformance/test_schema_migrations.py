"""Migrations and the boot-time schema guard, against a real Postgres.

Each test builds its own throwaway schema (tests/support/postgres.py
`scratch_schema`) so it can start from nothing -- which is the state that
matters for the guard, and the one no shared test database is ever in.

What is proven here and nowhere else:

  * The migration chain builds the whole schema from an empty database, as
    the lanes' role (owner, not superuser), with pgvector from `public`.
  * 0007 removes the bypass policies and its downgrade restores them.
  * A store that requires migrations refuses an empty database and runs no
    DDL doing so -- the "production never creates a schema without RLS"
    promise, checked on the catalogue rather than on a mock.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from tests.support import postgres as pg_support

from mapi.core.errors import ConfigurationError
from mapi.store.postgres.store import SCHEMA_REVISION

pytestmark = pytest.mark.postgres

TENANT_TABLES = ["spaces", "memories", "chunks", "relation_edges", "memory_versions"]


async def _tables(store, schema: str) -> set[str]:
    async with store._engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = :s"),
            {"s": schema},
        )
        return {r[0] for r in rows}


async def _policies(store, schema: str) -> set[str]:
    async with store._engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT policyname FROM pg_policies WHERE schemaname = :s"), {"s": schema}
        )
        return {r[0] for r in rows}


async def test_the_chain_builds_everything_and_0007_round_trips() -> None:
    async with pg_support.scratch_schema() as (schema, store):
        await pg_support.migrate(store)
        assert await store.schema_revision() == SCHEMA_REVISION
        assert set(TENANT_TABLES) <= await _tables(store, schema)

        policies = await _policies(store, schema)
        assert {f"{t}_tenant_isolation" for t in TENANT_TABLES} <= policies
        assert not {p for p in policies if p.endswith("_admin_bypass")}

        async with store._engine.connect() as conn:
            forced = await conn.execute(
                text(
                    "SELECT relname FROM pg_class WHERE relrowsecurity AND relforcerowsecurity "
                    "AND relnamespace = to_regnamespace(:s)"
                ),
                {"s": schema},
            )
            assert set(TENANT_TABLES) <= {r[0] for r in forced}

        # One step down restores 0005's bypass policy on every table...
        await pg_support.migrate(store, "-1", down=True)
        assert await store.schema_revision() != SCHEMA_REVISION
        assert {f"{t}_admin_bypass" for t in TENANT_TABLES} <= await _policies(store, schema)

        # ...and back up removes it again.
        await pg_support.migrate(store)
        assert not {p for p in await _policies(store, schema) if p.endswith("_admin_bypass")}


async def test_an_empty_database_is_refused_and_left_empty() -> None:
    """The production boot path, against the state it exists to refuse."""
    async with pg_support.scratch_schema(require_migrated=True) as (schema, store):
        with pytest.raises(ConfigurationError, match="alembic upgrade head"):
            await store.initialize()
        assert await _tables(store, schema) == set(), "a refused boot still ran DDL"


async def test_a_database_behind_the_build_is_refused() -> None:
    async with pg_support.scratch_schema(require_migrated=True) as (_schema, store):
        await pg_support.migrate(store, "0005")
        with pytest.raises(ConfigurationError, match="0005"):
            await store.initialize()


async def test_a_migrated_database_boots_with_no_ddl_at_all(monkeypatch) -> None:
    """At head, `initialize` must not reach the create-on-boot path.

    That path needs table ownership (the HNSW `CREATE INDEX IF NOT EXISTS`
    does, even when the index exists), which is why the app role had to own
    its tables just to start.
    """
    async with pg_support.scratch_schema(require_migrated=True) as (_schema, store):
        await pg_support.migrate(store)

        async def _no_ddl() -> None:
            raise AssertionError("initialize ran boot DDL against a migrated schema")

        monkeypatch.setattr(store, "_create_schema", _no_ddl)
        await store.initialize()


async def test_local_development_still_creates_the_schema_on_boot() -> None:
    """Without `require_migrated` the old convenience survives, for laptops."""
    async with pg_support.scratch_schema() as (schema, store):
        await store.initialize()
        assert set(TENANT_TABLES) <= await _tables(store, schema)
        assert await store.schema_revision() is None


# -- the whole app, not just the store ----------------------------------------------
#
# The store-level tests above prove the decision. These prove the wiring: that
# `MAPI_ENVIRONMENT=production` really reaches `require_migrated` through
# `build_store`, and that the lifespan lets the refusal stop the process
# rather than logging it and serving. Each app gets the real `build_store`,
# pointed at a scratch schema -- the one thing a test must change.


def _production_settings():
    from mapi.config import Settings

    return Settings(
        environment="production",
        store_backend="postgres",
        database_url=pg_support.database_url(),
        embedding_dimensions=pg_support.dimensions(),
        # Everything validate_production asks for; none of it is contacted
        # before the schema check, and the limiter fails open without Redis.
        embedding_backend="gemini",
        embedding_model="gemini-embedding-001",
        gemini_api_key="not-a-real-key",
        redis_url="redis://127.0.0.1:1/0",
        api_key_pepper="a-production-shaped-pepper-for-this-test",
    )


def _app_on(schema: str, monkeypatch):
    import mapi.main as main_module
    from mapi.store import build_store

    def _build(settings):
        store = build_store(settings)
        pg_support.pin_search_path(store, schema)
        return store

    monkeypatch.setattr(main_module, "build_store", _build)
    return main_module.create_app(_production_settings())


async def test_production_refuses_to_start_on_an_empty_database(monkeypatch) -> None:
    async with pg_support.scratch_schema() as (schema, store):
        app = _app_on(schema, monkeypatch)
        with pytest.raises(ConfigurationError, match="alembic upgrade head"):
            async with app.router.lifespan_context(app):
                pytest.fail("production served on a schema alembic never built")
        assert await _tables(store, schema) == set(), "a refused boot still ran DDL"


async def test_production_starts_once_alembic_has_run(monkeypatch) -> None:
    import httpx

    from mapi.store.postgres.store import PostgresStore

    async with pg_support.scratch_schema() as (schema, store):
        await pg_support.migrate(store)
        app = _app_on(schema, monkeypatch)

        async def _no_ddl(self) -> None:
            raise AssertionError("production ran boot DDL against a migrated schema")

        monkeypatch.setattr(PostgresStore, "_create_schema", _no_ddl)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as client,
        ):
            ready = await client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["checks"] == {"store": True}
