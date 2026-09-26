"""Postgres plumbing shared by every lane that talks to a real database.

Three settings, all from the environment, all read here so the conformance,
row-level-security and migration lanes cannot disagree about them:

  * `MAPI_TEST_DATABASE_URL` -- the database. Unset means the lanes skip.
  * `MAPI_TEST_DIMENSIONS` -- the width of `chunks.embedding` in that
    database. The column is `Vector(768)` in the models and in migration 0001,
    so any database the app or alembic created is 768 wide, and the old
    default of 128 failed 67 tests in CI with "expected 768 dimensions, not
    128". The default stays 128 only because it is what the in-memory suites
    were written at; every real database lane sets 768.
  * `MAPI_TEST_SCHEMA` -- optional. Pins every connection's search_path to
    this schema (then `public`, where the vector extension lives), so several
    people can run the lanes against one shared database without their rows,
    migrations or `alembic_version` colliding. The schema must exist and be
    migrated first -- `scripts/pg_lanes.py` does both and then runs the lanes.

`scratch_schema` is the other half: a throwaway schema for the tests that
need an EMPTY or specially-shaped database (the migration round trip, the
refusal to boot unmigrated), created and dropped around one test.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

#: The repository root, for alembic.ini and migrations/.
ROOT = Path(__file__).resolve().parents[2]

#: A bare identifier only. It is interpolated into SET search_path, which
#: cannot take a bind parameter.
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def database_url() -> str | None:
    return os.getenv("MAPI_TEST_DATABASE_URL") or None


def dimensions() -> int:
    return int(os.getenv("MAPI_TEST_DIMENSIONS", "128"))


def schema() -> str | None:
    value = os.getenv("MAPI_TEST_SCHEMA") or None
    if value is not None and not _IDENTIFIER.match(value):
        raise ValueError(f"MAPI_TEST_SCHEMA must be a lowercase identifier, got {value!r}")
    return value


def pin_search_path(store: Any, schema_name: str) -> None:
    """Make every connection `store` opens resolve names in `schema_name` first.

    A pool `connect` hook rather than a store option: the production store has
    no reason to know about schemas, and a session-level SET on a pooled
    connection is exactly the scope wanted here -- every checkout, forever.
    """
    if not _IDENTIFIER.match(schema_name):
        raise ValueError(f"not a plain identifier: {schema_name!r}")
    from sqlalchemy import event

    statement = f'SET search_path TO "{schema_name}", public'

    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.run_async(lambda conn: conn.execute(statement))

    event.listen(store._engine.sync_engine, "connect", _on_connect)


def make_store(*, in_schema: str | None = None, **kwargs: Any) -> Any:
    """A PostgresStore on the test database, at the test width and schema.

    `in_schema` overrides `MAPI_TEST_SCHEMA` for this one store. Skips the
    calling test when no database is configured, which is the normal case on
    a laptop without one.
    """
    url = database_url()
    if not url:
        pytest.skip("MAPI_TEST_DATABASE_URL is not set")
    from mapi.store.postgres.store import PostgresStore

    kwargs.setdefault("dimensions", dimensions())
    store = PostgresStore(url, **kwargs)
    pinned = in_schema or schema()
    if pinned is not None:
        pin_search_path(store, pinned)
    return store


def alembic(connection: Any, command_name: str, revision: str) -> None:
    """Run an alembic command on an existing SYNC connection.

    Goes through migrations/env.py's `attributes["connection"]` path, so the
    migrations run exactly as `alembic upgrade` would run them, but on this
    connection and whatever search_path it carries.
    """
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["connection"] = connection
    getattr(command, command_name)(config, revision)


async def migrate(store: Any, revision: str = "head", *, down: bool = False) -> None:
    """Upgrade (or downgrade) the database `store` points at, and commit."""
    async with store._engine.connect() as conn:
        await conn.run_sync(alembic, "downgrade" if down else "upgrade", revision)
        await conn.commit()


@asynccontextmanager
async def scratch_schema(**store_kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
    """A new empty schema and a store pinned to it; both gone afterwards.

    Needs CREATE on the database, which the lanes' role has as its owner. The
    name is random so parallel runs against a shared database never meet.
    """
    name = f"scratch_{uuid.uuid4().hex[:12]}"
    admin = make_store()
    try:
        async with admin._engine.begin() as conn:
            from sqlalchemy import text

            await conn.execute(text(f'CREATE SCHEMA "{name}"'))
        store = make_store(in_schema=name, **store_kwargs)
        try:
            yield name, store
        finally:
            await store.aclose()
    finally:
        async with admin._engine.begin() as conn:
            from sqlalchemy import text

            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))
        await admin.aclose()


__all__ = [
    "ROOT",
    "alembic",
    "database_url",
    "dimensions",
    "make_store",
    "migrate",
    "pin_search_path",
    "schema",
    "scratch_schema",
]
