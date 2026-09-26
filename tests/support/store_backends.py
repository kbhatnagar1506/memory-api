"""Both storage backends, for tests that must hold on each.

The conformance suite's `_make_store` pattern, shared: the in-memory store is
fresh per test, the Postgres store is built once per process (initialize()
over a WAN link costs seconds) and skipped cleanly without
`MAPI_TEST_DATABASE_URL`. Tenants never collide because every test creates
its own organization.

`MAPI_TEST_DIMENSIONS` must match the database's vector column (768 for a
database migrated by this repo); the in-memory backend accepts anything.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from mapi.store.base import MemoryStore
from mapi.store.memory import InMemoryStore

DIMENSIONS = int(os.getenv("MAPI_TEST_DIMENSIONS", "128"))

_POSTGRES: Any = None

#: Parametrize a `backend` fixture with this. The marker rides on the
#: Postgres param so `-m "not postgres"` deselects it.
BACKENDS = [
    pytest.param("memory", id="memory"),
    pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
]


async def make_store(kind: str) -> MemoryStore:
    global _POSTGRES
    if kind == "memory":
        return InMemoryStore()
    url = os.getenv("MAPI_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MAPI_TEST_DATABASE_URL is not set")
    if _POSTGRES is None:
        from mapi.store.postgres.store import PostgresStore

        _POSTGRES = PostgresStore(url, dimensions=DIMENSIONS)
        await _POSTGRES.initialize()
    store: MemoryStore = _POSTGRES
    return store


__all__ = ["BACKENDS", "DIMENSIONS", "make_store"]
