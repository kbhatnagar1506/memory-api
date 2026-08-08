"""Shared fixtures.

Everything defaults to the in-memory backend and the deterministic embedder, so
the entire suite runs with no Docker, no database and no credentials. Tests that
genuinely need Postgres are marked `postgres` and skip cleanly when
`SUPERMEMORY_TEST_DATABASE_URL` is unset.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest

from supermemory.config import Settings
from supermemory.core.security import build_api_key
from supermemory.domain.embeddings import DeterministicEmbedder
from supermemory.domain.models import Organization, Scope, Space
from supermemory.domain.retrieval.rerank import HeuristicReranker
from supermemory.main import create_app
from supermemory.service import MemoryService
from supermemory.store.memory import InMemoryStore

TEST_DIMENSIONS = 128
TEST_PEPPER = "test-pepper"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=TEST_DIMENSIONS,
        rerank_backend="heuristic",
        api_key_pepper=TEST_PEPPER,
        chunk_target_tokens=64,
        chunk_overlap_tokens=8,
        rate_limit_per_minute=100_000,
        rate_limit_burst=100_000,
    )


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder(dimensions=TEST_DIMENSIONS)


@pytest.fixture
async def store() -> AsyncIterator[InMemoryStore]:
    s = InMemoryStore()
    yield s
    await s.reset()


@pytest.fixture
async def service(
    store: InMemoryStore, embedder: DeterministicEmbedder, settings: Settings
) -> MemoryService:
    return MemoryService(store, embedder, HeuristicReranker(), settings)


@pytest.fixture
async def org(store: InMemoryStore) -> Organization:
    return await store.create_organization(Organization(name="Test Org"))


@pytest.fixture
async def space(store: InMemoryStore, org: Organization) -> Space:
    return await store.create_space(Space(org_id=org.id, slug="default", name="Default"))


# -- API fixtures --------------------------------------------------------------


@pytest.fixture
async def app_context(settings: Settings):
    """A fully wired app with lifespan run, plus a seeded org/space/admin key."""
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        store = app.state.store
        organization = await store.create_organization(Organization(name="Acme"))
        sp = await store.create_space(
            Space(org_id=organization.id, slug="default", name="Default")
        )
        record, plaintext = build_api_key(
            org_id=organization.id,
            name="test",
            pepper=TEST_PEPPER,
            scopes=frozenset(Scope.all()),
        )
        await store.create_api_key(record)
        yield {
            "app": app,
            "org": organization,
            "space": sp,
            "key": plaintext,
            "store": store,
        }


@pytest.fixture
async def client(app_context) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {app_context['key']}"},
    ) as c:
        yield c


@pytest.fixture
def space_id(app_context) -> str:
    return app_context["space"].id


@pytest.fixture
def anon_client(app_context) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app_context["app"])
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# -- Postgres ------------------------------------------------------------------


def postgres_url() -> str | None:
    return os.getenv("SUPERMEMORY_TEST_DATABASE_URL")


@pytest.fixture
async def postgres_store():
    url = postgres_url()
    if not url:
        pytest.skip("SUPERMEMORY_TEST_DATABASE_URL is not set")
    from supermemory.store.postgres.store import PostgresStore

    s = PostgresStore(url, dimensions=TEST_DIMENSIONS)
    await s.initialize()
    yield s
    await s.aclose()
