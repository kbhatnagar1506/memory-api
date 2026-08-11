"""Shared fixtures.

Everything defaults to the in-memory backend and the deterministic embedder, so
the entire suite runs with no Docker, no database and no credentials. Tests that
genuinely need Postgres are marked `postgres` and skip cleanly when
`MAPI_TEST_DATABASE_URL` is unset.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest

from mapi.config import Settings
from mapi.core.security import build_api_key
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Organization, Scope, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.main import create_app
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

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


async def _confirm_everything(prompt: str) -> str:
    """A stand-in adjudicator: confirms conflicts, refuses supersessions.

    Contradiction and supersession both FAIL CLOSED -- without a model to
    confirm, nothing is recorded, because the cheap signals produce false
    positives in volume on any corpus with dates or quantities in it. The
    e2e app has no vendor, so tests asserting a conflict was recorded have
    to supply the judgement the design requires.

    Supersession is deliberately refused. It is the only operation that
    HIDES a memory, and a fixture that hid memories behind every test's back
    would make unrelated assertions fail in ways nobody would think to
    attribute to a stub. Tests that want supersession inject their own.
    """
    if "REPLACES" in prompt:
        return "[]"
    return '[{"n": 1, "reason": "adjudicated", "confidence": 0.95}]'


@pytest.fixture
async def app_context(settings: Settings):
    """A fully wired app with lifespan run, plus a seeded org/space/admin key."""
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.service.extractor = _confirm_everything
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
    return os.getenv("MAPI_TEST_DATABASE_URL")


@pytest.fixture
async def postgres_store():
    url = postgres_url()
    if not url:
        pytest.skip("MAPI_TEST_DATABASE_URL is not set")
    from mapi.store.postgres.store import PostgresStore

    s = PostgresStore(url, dimensions=TEST_DIMENSIONS)
    await s.initialize()
    yield s
    await s.aclose()
