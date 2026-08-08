"""Storage conformance.

One suite, parametrized over every backend. The in-memory backend always runs;
Postgres runs when SUPERMEMORY_TEST_DATABASE_URL is set (CI sets it).

This is the file that makes two implementations safe to have. Without it, the
backends drift — a filter that means one thing in Python and another in SQL,
a cursor that is stable in one and not the other — and the difference surfaces
as a production bug that no test reproduces.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from supermemory.core.errors import BadRequestError, ConflictError
from supermemory.domain.embeddings import DeterministicEmbedder
from supermemory.domain.models import (
    ApiKey,
    Chunk,
    Memory,
    MemoryStatus,
    Organization,
    Scope,
    Space,
)
from supermemory.store.base import MemoryFilter
from supermemory.store.memory import InMemoryStore

DIMENSIONS = 128
NOW = datetime.now(UTC)


async def _make_store(kind: str):
    if kind == "memory":
        return InMemoryStore(), None
    url = os.getenv("SUPERMEMORY_TEST_DATABASE_URL")
    if not url:
        pytest.skip("SUPERMEMORY_TEST_DATABASE_URL is not set")
    from supermemory.store.postgres.store import PostgresStore

    store = PostgresStore(url, dimensions=DIMENSIONS)
    await store.initialize()
    return store, store.aclose


@pytest.fixture(params=["memory", "postgres"])
async def backend(request):
    store, closer = await _make_store(request.param)
    yield store
    if closer is not None:
        await closer()


@pytest.fixture
async def tenant(backend):
    org = await backend.create_organization(Organization(name="Conformance"))
    space = await backend.create_space(
        Space(org_id=org.id, slug=f"s{org.id[-8:]}", name="Space")
    )
    return backend, org, space


async def _add(store, org, space, text: str, *, embedder=None, **kw) -> Memory:
    embedder = embedder or DeterministicEmbedder(dimensions=DIMENSIONS)
    vector = await embedder.embed_one(text)
    memory = Memory(org_id=org.id, space_id=space.id, content=text, **kw)
    memory = memory.model_copy(
        update={"chunks": [Chunk(memory_id=memory.id, ordinal=0, text=text, embedding=vector)]}
    )
    return await store.upsert_memory(memory)


# -- spaces --------------------------------------------------------------------


async def test_space_round_trip(tenant) -> None:
    store, org, space = tenant
    fetched = await store.get_space(org.id, space.id)
    assert fetched is not None
    assert fetched.slug == space.slug


async def test_space_slug_is_unique_per_org(tenant) -> None:
    store, org, space = tenant
    with pytest.raises(ConflictError):
        await store.create_space(Space(org_id=org.id, slug=space.slug, name="Duplicate"))


async def test_space_is_invisible_to_another_org(tenant) -> None:
    store, _, space = tenant
    other = await store.create_organization(Organization(name="Other"))
    assert await store.get_space(other.id, space.id) is None


async def test_delete_space_removes_its_memories(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "some content")
    assert await store.delete_space(org.id, space.id) is True
    assert await store.get_memory(org.id, space.id, memory.id) is None
    assert await store.delete_space(org.id, space.id) is False


# -- memories ------------------------------------------------------------------


async def test_memory_round_trip_preserves_fields(tenant) -> None:
    store, org, space = tenant
    memory = await _add(
        store,
        org,
        space,
        "hello world",
        tags=["a", "b"],
        metadata={"k": "v"},
        source="unit-test",
    )
    fetched = await store.get_memory(org.id, space.id, memory.id)
    assert fetched is not None
    assert fetched.content == "hello world"
    assert set(fetched.tags) == {"a", "b"}
    assert fetched.metadata == {"k": "v"}
    assert fetched.source == "unit-test"
    assert len(fetched.chunks) == 1
    assert fetched.chunks[0].embedding is not None


async def test_memory_is_invisible_across_tenants(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "tenant secret")
    other = await store.create_organization(Organization(name="Intruder"))
    assert await store.get_memory(other.id, space.id, memory.id) is None
    assert await store.get_memories(other.id, space.id, [memory.id]) == {}


async def test_upsert_replaces_chunks_wholesale(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "original")
    updated = memory.model_copy(
        update={
            "content": "revised",
            "chunks": [
                Chunk(
                    memory_id=memory.id, ordinal=0, text="revised", embedding=[0.1] * DIMENSIONS
                )
            ],
            "version": 2,
        }
    )
    await store.upsert_memory(updated)
    fetched = await store.get_memory(org.id, space.id, memory.id)
    assert fetched is not None
    assert fetched.content == "revised"
    assert len(fetched.chunks) == 1
    assert fetched.chunks[0].text == "revised"


async def test_delete_is_idempotent_and_reports_absence(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "delete me")
    assert await store.delete_memory(org.id, space.id, memory.id) is True
    assert await store.delete_memory(org.id, space.id, memory.id) is False


async def test_get_memories_omits_missing_ids(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "present")
    found = await store.get_memories(
        org.id, space.id, [memory.id, "mem_00000000000000000000000000"]
    )
    assert set(found) == {memory.id}


async def test_content_hash_lookup(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "Hash Me   Please")
    found = await store.find_by_content_hash(org.id, space.id, memory.content_sha256)
    assert found is not None
    assert found.id == memory.id


# -- filters -------------------------------------------------------------------


async def test_status_filter(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "active one")
    await _add(store, org, space, "archived one", status=MemoryStatus.ARCHIVED)
    active = await store.count_memories(org.id, space.id, filters=MemoryFilter())
    assert active == 1
    everything = await store.count_memories(
        org.id,
        space.id,
        filters=MemoryFilter(statuses=frozenset(MemoryStatus)),
    )
    assert everything == 2


async def test_tag_filter_requires_all_tags(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "both tags", tags=["x", "y"])
    await _add(store, org, space, "one tag", tags=["x"])
    both = await store.count_memories(org.id, space.id, filters=MemoryFilter(tags=("x", "y")))
    assert both == 1


async def test_source_filter(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "from slack", source="slack")
    await _add(store, org, space, "from email", source="email")
    assert (
        await store.count_memories(org.id, space.id, filters=MemoryFilter(source="slack")) == 1
    )


async def test_occurred_range_filter(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "old", occurred_at=NOW - timedelta(days=100))
    await _add(store, org, space, "new", occurred_at=NOW)
    recent = await store.count_memories(
        org.id,
        space.id,
        filters=MemoryFilter(occurred_after=NOW - timedelta(days=10)),
    )
    assert recent == 1


# -- pagination ----------------------------------------------------------------


async def test_pagination_covers_everything_exactly_once(tenant) -> None:
    store, org, space = tenant
    for i in range(10):
        await _add(store, org, space, f"memory number {i}")

    seen: list[str] = []
    cursor = None
    for _ in range(20):  # bounded to catch a non-terminating cursor
        page = await store.list_memories(
            org.id, space.id, filters=MemoryFilter(), limit=3, cursor=cursor
        )
        seen.extend(m.id for m in page.items)
        if not page.next_cursor:
            break
        cursor = page.next_cursor
    assert len(seen) == 10
    assert len(set(seen)) == 10


async def test_pagination_reports_total(tenant) -> None:
    store, org, space = tenant
    for i in range(5):
        await _add(store, org, space, f"item {i}")
    page = await store.list_memories(org.id, space.id, filters=MemoryFilter(), limit=2)
    assert page.total == 5
    assert len(page.items) == 2


async def test_last_page_has_no_cursor(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "only one")
    page = await store.list_memories(org.id, space.id, filters=MemoryFilter(), limit=10)
    assert page.next_cursor is None


async def test_malformed_cursor_is_a_client_error(tenant) -> None:
    store, org, space = tenant
    with pytest.raises(BadRequestError):
        await store.list_memories(
            org.id, space.id, filters=MemoryFilter(), limit=2, cursor="!!!not-base64"
        )


async def test_zero_limit_returns_nothing(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "content")
    page = await store.list_memories(org.id, space.id, filters=MemoryFilter(), limit=0)
    assert page.items == []


# -- retrieval -----------------------------------------------------------------


async def test_vector_search_orders_by_similarity(tenant) -> None:
    store, org, space = tenant
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
    await _add(store, org, space, "kafka event streaming platform", embedder=embedder)
    await _add(store, org, space, "the cat sat on the mat", embedder=embedder)
    query = await embedder.embed_one("kafka streaming")
    hits = await store.vector_search(org.id, space.id, query, limit=5, filters=MemoryFilter())
    assert hits
    assert "kafka" in hits[0].text
    assert hits == sorted(hits, key=lambda h: -h.score)
    # Similarity, not distance: a good match must score high.
    assert hits[0].score > 0.3


async def test_vector_search_respects_tenancy(tenant) -> None:
    store, org, space = tenant
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
    await _add(store, org, space, "tenant one secret", embedder=embedder)
    other = await store.create_organization(Organization(name="Other"))
    other_space = await store.create_space(
        Space(org_id=other.id, slug=f"o{other.id[-8:]}", name="Other")
    )
    await _add(store, other, other_space, "tenant two secret", embedder=embedder)

    query = await embedder.embed_one("secret")
    hits = await store.vector_search(org.id, space.id, query, limit=10, filters=MemoryFilter())
    assert all("two" not in h.text for h in hits)


async def test_vector_search_applies_filters(tenant) -> None:
    store, org, space = tenant
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
    await _add(store, org, space, "kafka tagged", tags=["keep"], embedder=embedder)
    await _add(store, org, space, "kafka untagged", embedder=embedder)
    query = await embedder.embed_one("kafka")
    hits = await store.vector_search(
        org.id, space.id, query, limit=10, filters=MemoryFilter(tags=("keep",))
    )
    assert len(hits) == 1
    assert "tagged" in hits[0].text


async def test_lexical_search_finds_exact_terms(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "the postgres migration was approved")
    await _add(store, org, space, "completely unrelated content here")
    hits = await store.lexical_search(
        org.id, space.id, "postgres", limit=5, filters=MemoryFilter()
    )
    assert len(hits) == 1
    assert "postgres" in hits[0].text


async def test_lexical_search_tolerates_punctuation_and_operators(tenant) -> None:
    """User queries contain characters that break naive tsquery construction."""
    store, org, space = tenant
    await _add(store, org, space, "error handling in production")
    for query in ["error!", "what's the error?", "error & handling", "a | b", "'quoted'"]:
        hits = await store.lexical_search(
            org.id, space.id, query, limit=5, filters=MemoryFilter()
        )
        assert isinstance(hits, list)


async def test_searches_on_empty_corpus_return_empty(tenant) -> None:
    store, org, space = tenant
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
    query = await embedder.embed_one("anything")
    assert (
        await store.vector_search(org.id, space.id, query, limit=5, filters=MemoryFilter())
        == []
    )
    assert (
        await store.lexical_search(
            org.id, space.id, "anything", limit=5, filters=MemoryFilter()
        )
        == []
    )


async def test_blank_lexical_query_returns_empty(tenant) -> None:
    store, org, space = tenant
    await _add(store, org, space, "content")
    assert (
        await store.lexical_search(org.id, space.id, "   ", limit=5, filters=MemoryFilter())
        == []
    )


async def test_sample_embeddings_returns_one_vector_per_memory(tenant) -> None:
    store, org, space = tenant
    for i in range(3):
        await _add(store, org, space, f"memory {i}")
    samples = await store.sample_embeddings(org.id, space.id, limit=10)
    assert len(samples) == 3
    assert all(len(vector) == DIMENSIONS for _, vector in samples)


# -- api keys ------------------------------------------------------------------


async def test_api_key_lookup_by_hash(tenant) -> None:
    store, org, _ = tenant
    key = await store.create_api_key(
        ApiKey(
            org_id=org.id,
            name="k",
            key_hash="a" * 64,
            prefix="sm_aaaa",
            scopes=frozenset({Scope.SEARCH}),
        )
    )
    found = await store.get_api_key_by_hash("a" * 64)
    assert found is not None
    assert found.id == key.id
    assert Scope.SEARCH in found.scopes


async def test_revoked_key_reports_inactive(tenant) -> None:
    store, org, _ = tenant
    key = await store.create_api_key(
        ApiKey(
            org_id=org.id,
            name="k",
            key_hash="b" * 64,
            prefix="sm_bbbb",
            scopes=frozenset({Scope.SEARCH}),
        )
    )
    assert await store.revoke_api_key(org.id, key.id) is True
    found = await store.get_api_key_by_hash("b" * 64)
    assert found is not None
    assert not found.is_active()
    assert await store.revoke_api_key(org.id, key.id) is False


async def test_ping(tenant) -> None:
    store, _, _ = tenant
    assert await store.ping() is True
