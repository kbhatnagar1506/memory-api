"""One shared Postgres store for the read-path lanes.

The same reasoning as `tests/conformance/test_store_contract.py`: building a
store runs `create_all` plus the HNSW DDL, which against a managed instance is
tens of seconds, so it is built once per process and reused. Tests never share
tenants -- each creates its own organization -- so sharing the pool is safe.

Skips (not fails) without `MAPI_TEST_DATABASE_URL`, so the in-memory half of a
parametrized test still runs on a laptop with no database.

`MAPI_TEST_SCHEMA`, when set, pins every connection's search_path to that
schema and then `public` (where the vector extension lives), so several people
can run the lanes against one shared database without their tables meeting.
The schema must already exist and be migrated.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest

#: Must match the vector column of the database under test.
DIMENSIONS = int(os.getenv("MAPI_TEST_DIMENSIONS", "128"))

#: A bare identifier only: it is interpolated into a search_path.
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_STORE: Any = None


def database_url() -> str:
    """The test database, or skip the calling test."""
    url = os.getenv("MAPI_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MAPI_TEST_DATABASE_URL is not set")
    return url


def server_settings() -> dict[str, str]:
    """Connection settings every lane store is opened with."""
    schema = os.getenv("MAPI_TEST_SCHEMA") or None
    if schema is None:
        return {}
    if not _IDENTIFIER.match(schema):
        raise ValueError(f"MAPI_TEST_SCHEMA must be a lowercase identifier, got {schema!r}")
    return {"search_path": f'"{schema}", public'}


def make_store(**kwargs: Any) -> Any:
    """A fresh PostgresStore on the test database, at the test width and schema."""
    from mapi.store.postgres.store import PostgresStore

    kwargs.setdefault("dimensions", DIMENSIONS)
    kwargs.setdefault("server_settings", server_settings())
    return PostgresStore(database_url(), **kwargs)


async def shared_postgres_store() -> Any:
    global _STORE
    if _STORE is None:
        _STORE = make_store()
        await _STORE.initialize()
    return _STORE


__all__ = [
    "DIMENSIONS",
    "database_url",
    "make_store",
    "server_settings",
    "shared_postgres_store",
]
