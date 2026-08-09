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
    MemoryKind,
    MemoryStatus,
    Organization,
    RelationEdge,
    RelationType,
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


# -- relations: the typed graph ------------------------------------------------


async def _edge(store, org, space, source, target, type_=RelationType.SUPERSEDES):
    return await store.create_relation(
        RelationEdge(
            org_id=org.id,
            space_id=space.id,
            source_id=source.id,
            target_id=target.id,
            type=type_,
        )
    )


async def test_relation_round_trip(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "newer")
    b = await _add(store, org, space, "older")
    edge = await _edge(store, org, space, a, b)

    out = await store.list_relations(org.id, space.id, a.id, direction="out")
    assert [e.id for e in out] == [edge.id]
    assert out[0].source_id == a.id and out[0].target_id == b.id


async def test_relation_reverse_lookup(tenant) -> None:
    """The index that makes "what supersedes this" cheap."""
    store, org, space = tenant
    a = await _add(store, org, space, "newer")
    b = await _add(store, org, space, "older")
    await _edge(store, org, space, a, b)

    incoming = await store.list_relations(org.id, space.id, b.id, direction="in")
    assert [e.source_id for e in incoming] == [a.id]
    assert await store.list_relations(org.id, space.id, b.id, direction="out") == []


async def test_relation_creation_is_idempotent(tenant) -> None:
    """A retried request must not create a parallel edge."""
    store, org, space = tenant
    a = await _add(store, org, space, "newer")
    b = await _add(store, org, space, "older")
    first = await _edge(store, org, space, a, b)
    second = await _edge(store, org, space, a, b)
    assert first.id == second.id
    assert len(await store.list_relations(org.id, space.id, a.id)) == 1


async def test_relations_filter_by_type(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "one")
    b = await _add(store, org, space, "two")
    await _edge(store, org, space, a, b, RelationType.SUPERSEDES)
    await _edge(store, org, space, a, b, RelationType.CONTRADICTS)

    assert len(await store.list_relations(org.id, space.id, a.id)) == 2
    only = await store.list_relations(org.id, space.id, a.id, type=RelationType.CONTRADICTS)
    assert [e.type for e in only] == [RelationType.CONTRADICTS]


async def test_relations_do_not_cross_tenants(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "newer")
    b = await _add(store, org, space, "older")
    await _edge(store, org, space, a, b)
    other = await store.create_organization(Organization(name="Intruder"))
    assert await store.list_relations(other.id, space.id, a.id) == []


async def test_get_relations_between_is_the_induced_subgraph(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "a")
    b = await _add(store, org, space, "b")
    c = await _add(store, org, space, "c")
    await _edge(store, org, space, a, b)
    await _edge(store, org, space, b, c)

    both = await store.get_relations_between(org.id, space.id, [a.id, b.id, c.id])
    assert len(both) == 2
    # Only A and C: neither edge has both endpoints in the set.
    assert await store.get_relations_between(org.id, space.id, [a.id, c.id]) == []
    assert await store.get_relations_between(org.id, space.id, []) == []


async def test_reachable_superseders_is_transitive(tenant) -> None:
    """The property one-hop suppression got wrong: A->B->C must reach A from C."""
    store, org, space = tenant
    a = await _add(store, org, space, "newest")
    b = await _add(store, org, space, "middle")
    c = await _add(store, org, space, "oldest")
    await _edge(store, org, space, b, c)
    await _edge(store, org, space, a, b)

    assert await store.reachable_superseders(org.id, space.id, c.id) == {a.id, b.id}
    assert await store.reachable_superseders(org.id, space.id, b.id) == {a.id}
    assert await store.reachable_superseders(org.id, space.id, a.id) == set()


async def test_reachable_superseders_handles_fan_in(tenant) -> None:
    """Two people independently correcting the same fact both make it stale."""
    store, org, space = tenant
    x = await _add(store, org, space, "original")
    y1 = await _add(store, org, space, "correction one")
    y2 = await _add(store, org, space, "correction two")
    await _edge(store, org, space, y1, x)
    await _edge(store, org, space, y2, x)
    assert await store.reachable_superseders(org.id, space.id, x.id) == {y1.id, y2.id}


async def test_graph_walks_terminate_on_a_cycle(tenant) -> None:
    """Malformed data must not hang a request."""
    store, org, space = tenant
    a = await _add(store, org, space, "a")
    b = await _add(store, org, space, "b")
    await _edge(store, org, space, a, b)
    await _edge(store, org, space, b, a)

    assert await store.reachable_superseders(org.id, space.id, a.id) == {b.id}
    chain = await store.walk_supersession_chain(org.id, space.id, a.id)
    assert len(chain) <= 2


async def test_deleting_a_memory_removes_its_edges(tenant) -> None:
    """A dangling edge is worse than no edge: it points at nothing."""
    store, org, space = tenant
    a = await _add(store, org, space, "newer")
    b = await _add(store, org, space, "older")
    await _edge(store, org, space, a, b)
    await store.delete_memory(org.id, space.id, b.id)
    assert await store.list_relations(org.id, space.id, a.id, direction="out") == []


async def test_walk_supersession_chain_both_directions(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "newest")
    b = await _add(store, org, space, "middle")
    c = await _add(store, org, space, "oldest")
    await _edge(store, org, space, b, c)
    await _edge(store, org, space, a, b)

    backward = await store.walk_supersession_chain(org.id, space.id, a.id)
    assert [e.target_id for e in backward] == [b.id, c.id]
    forward = await store.walk_supersession_chain(org.id, space.id, c.id, direction="forward")
    assert [e.source_id for e in forward] == [b.id, a.id]


async def test_walk_rejects_a_zero_depth(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "a")
    with pytest.raises(ValueError, match="max_depth"):
        await store.walk_supersession_chain(org.id, space.id, a.id, max_depth=0)


async def test_invalid_direction_rejected(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "a")
    with pytest.raises(ValueError, match="direction"):
        await store.list_relations(org.id, space.id, a.id, direction="sideways")


# -- bitemporal history --------------------------------------------------------


async def test_every_write_records_a_version(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "v1")
    versions = await store.list_memory_versions(org.id, space.id, memory.id)
    assert len(versions) == 1
    assert versions[0].content == "v1"
    assert versions[0].valid_to is None


async def test_version_bump_opens_a_new_snapshot_and_closes_the_old(tenant) -> None:
    store, org, space = tenant
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    memory = Memory(org_id=org.id, space_id=space.id, content="v1")
    await store.upsert_memory(memory, now=t0)
    await store.upsert_memory(memory.model_copy(update={"content": "v2", "version": 2}), now=t1)

    versions = await store.list_memory_versions(org.id, space.id, memory.id)
    assert [v.content for v in versions] == ["v1", "v2"]
    assert versions[0].valid_to == t1
    assert versions[1].valid_to is None


async def test_same_version_write_is_an_in_place_correction(tenant) -> None:
    """Re-saving without bumping must not accumulate duplicate snapshots.

    Postgres would also reject it outright on uq_versions_memory_version, so
    the two backends have to agree on this or they diverge on a normal write.
    """
    store, org, space = tenant
    memory = Memory(org_id=org.id, space_id=space.id, content="v1")
    await store.upsert_memory(memory)
    await store.upsert_memory(memory.model_copy(update={"content": "corrected"}))

    versions = await store.list_memory_versions(org.id, space.id, memory.id)
    assert len(versions) == 1
    assert versions[0].content == "corrected"


async def test_point_in_time_read(tenant) -> None:
    store, org, space = tenant
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(hours=2)
    memory = Memory(org_id=org.id, space_id=space.id, content="v1")
    await store.upsert_memory(memory, now=t0)
    await store.upsert_memory(memory.model_copy(update={"content": "v2", "version": 2}), now=t1)

    mid = await store.get_memory_as_of(org.id, space.id, memory.id, t0 + timedelta(hours=1))
    assert mid is not None and mid.content == "v1"
    later = await store.get_memory_as_of(org.id, space.id, memory.id, t1 + timedelta(hours=1))
    assert later is not None and later.content == "v2"


async def test_point_in_time_boundary_is_half_open(tenant) -> None:
    """At exactly valid_to the old snapshot is already gone."""
    store, org, space = tenant
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    memory = Memory(org_id=org.id, space_id=space.id, content="v1")
    await store.upsert_memory(memory, now=t0)
    await store.upsert_memory(memory.model_copy(update={"content": "v2", "version": 2}), now=t1)
    at_boundary = await store.get_memory_as_of(org.id, space.id, memory.id, t1)
    assert at_boundary is not None and at_boundary.content == "v2"


async def test_point_in_time_before_creation_is_none(tenant) -> None:
    store, org, space = tenant
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    memory = Memory(org_id=org.id, space_id=space.id, content="v1")
    await store.upsert_memory(memory, now=t0)
    assert (
        await store.get_memory_as_of(org.id, space.id, memory.id, t0 - timedelta(days=1))
        is None
    )


async def test_history_does_not_cross_tenants(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "secret")
    other = await store.create_organization(Organization(name="Intruder"))
    assert await store.list_memory_versions(other.id, space.id, memory.id) == []
    assert await store.get_memory_as_of(other.id, space.id, memory.id, NOW) is None


async def test_status_change_is_captured_in_history(tenant) -> None:
    """Supersession is a status transition, and history must show it."""
    store, org, space = tenant
    memory = await _add(store, org, space, "fact")
    await store.upsert_memory(
        memory.model_copy(update={"status": MemoryStatus.SUPERSEDED, "version": 2})
    )
    versions = await store.list_memory_versions(org.id, space.id, memory.id)
    assert [v.status for v in versions] == [
        MemoryStatus.ACTIVE,
        MemoryStatus.SUPERSEDED,
    ]


# -- erasure: the compliance path ----------------------------------------------


async def test_erase_purges_the_reconstructed_past(tenant) -> None:
    """The property that separates erase from delete.

    After delete_memory, as_of still reconstructs history (audit default).
    After erase_memory, it must not: an erasure that survives point-in-time
    reads is not an erasure.
    """
    store, org, space = tenant
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    memory = Memory(org_id=org.id, space_id=space.id, content="right to be forgotten")
    await store.upsert_memory(memory, now=t0)

    report = await store.erase_memory(org.id, space.id, memory.id)
    assert report.existed is True
    assert report.versions_purged == 1
    assert await store.get_memory(org.id, space.id, memory.id) is None
    assert (
        await store.get_memory_as_of(org.id, space.id, memory.id, t0 + timedelta(hours=1))
        is None
    )
    assert await store.list_memory_versions(org.id, space.id, memory.id) == []


async def test_erase_purges_history_left_by_delete(tenant) -> None:
    """delete then erase: the residual audit trail must also be destroyable."""
    store, org, space = tenant
    memory = await _add(store, org, space, "soft-deleted but remembered")
    assert await store.delete_memory(org.id, space.id, memory.id) is True
    assert len(await store.list_memory_versions(org.id, space.id, memory.id)) == 1

    report = await store.erase_memory(org.id, space.id, memory.id)
    assert report.existed is True
    assert report.versions_purged == 1
    assert await store.list_memory_versions(org.id, space.id, memory.id) == []


async def test_erase_removes_edges_and_counts_everything(tenant) -> None:
    store, org, space = tenant
    a = await _add(store, org, space, "newer fact")
    b = await _add(store, org, space, "the fact being erased")
    c = await _add(store, org, space, "derived note")
    await _edge(store, org, space, a, b)
    await _edge(store, org, space, c, b, RelationType.DERIVED_FROM)

    report = await store.erase_memory(org.id, space.id, b.id)
    assert report.existed is True
    assert report.edges_removed == 2
    assert report.chunks_removed == 1
    assert await store.list_relations(org.id, space.id, a.id, direction="out") == []
    assert await store.list_relations(org.id, space.id, c.id, direction="out") == []


async def test_erase_is_idempotent(tenant) -> None:
    """A retried compliance request must not fail on the second attempt."""
    store, org, space = tenant
    memory = await _add(store, org, space, "erase twice")
    first = await store.erase_memory(org.id, space.id, memory.id)
    second = await store.erase_memory(org.id, space.id, memory.id)
    assert first.existed is True
    assert second.existed is False
    assert second.versions_purged == 0


async def test_erase_is_tenant_scoped(tenant) -> None:
    store, org, space = tenant
    memory = await _add(store, org, space, "not yours to erase")
    other = await store.create_organization(Organization(name="Intruder"))
    report = await store.erase_memory(other.id, space.id, memory.id)
    assert report.existed is False
    assert await store.get_memory(org.id, space.id, memory.id) is not None
    assert len(await store.list_memory_versions(org.id, space.id, memory.id)) == 1


async def test_erase_leaves_search_clean(tenant) -> None:
    store, org, space = tenant
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
    keep = await _add(store, org, space, "kafka handles the event bus", embedder=embedder)
    gone = await _add(store, org, space, "kafka secrets to erase", embedder=embedder)
    await store.erase_memory(org.id, space.id, gone.id)

    query = await embedder.embed_one("kafka")
    vector_hits = await store.vector_search(
        org.id, space.id, query, limit=10, filters=MemoryFilter()
    )
    lexical_hits = await store.lexical_search(
        org.id, space.id, "kafka", limit=10, filters=MemoryFilter()
    )
    found = {h.memory_id for h in vector_hits} | {h.memory_id for h in lexical_hits}
    assert found == {keep.id}


# -- supersession bridging across removal --------------------------------------
#
# Found live: erasing the middle of a "Portland" -> "Austin" -> "Seattle"
# revision chain brought "lives in Portland" back as a current fact, because
# the only SUPERSEDES path from the newest to the oldest ran through the
# removed middle. Every removal path must bridge around the gap.


async def _chain(store, org, space, *texts: str) -> list[Memory]:
    """Oldest first; each later memory supersedes the one before it."""
    memories = []
    for offset, text in enumerate(texts):
        memories.append(
            await _add(store, org, space, text, occurred_at=NOW + timedelta(days=offset))
        )
    for newer, older in zip(memories[1:], memories, strict=False):
        await store.create_relation(
            RelationEdge(
                org_id=org.id,
                space_id=space.id,
                source_id=newer.id,
                target_id=older.id,
                type=RelationType.SUPERSEDES,
                confidence=0.9,
            )
        )
    return memories


async def test_erasing_a_middle_revision_does_not_resurrect_the_oldest(tenant) -> None:
    store, org, space = tenant
    oldest, middle, newest = await _chain(
        store, org, space, "lives in Portland", "moved to Austin", "moved to Seattle"
    )
    report = await store.erase_memory(org.id, space.id, middle.id)
    assert report.edges_bridged == 1
    superseders = await store.reachable_superseders(org.id, space.id, oldest.id)
    assert newest.id in superseders, "the replaced fact resurfaced as current"


async def test_deleting_a_middle_revision_does_not_resurrect_the_oldest(tenant) -> None:
    """Plain delete severed chains the same way erase did — postgres via FK
    cascade, in-memory via the explicit edge sweep."""
    store, org, space = tenant
    oldest, middle, newest = await _chain(store, org, space, "v1", "v2", "v3")
    assert await store.delete_memory(org.id, space.id, middle.id)
    superseders = await store.reachable_superseders(org.id, space.id, oldest.id)
    assert newest.id in superseders


async def test_bridge_survives_removing_two_consecutive_middles(tenant) -> None:
    """A -> B -> C -> D, remove C then B: each removal re-bridges over the
    hole the previous one left, so D still transitively supersedes A."""
    store, org, space = tenant
    a, b, c, d = await _chain(store, org, space, "v1", "v2", "v3", "v4")
    await store.erase_memory(org.id, space.id, c.id)
    await store.erase_memory(org.id, space.id, b.id)
    superseders = await store.reachable_superseders(org.id, space.id, a.id)
    assert d.id in superseders


async def test_bridge_confidence_is_the_weakest_hop(tenant) -> None:
    store, org, space = tenant
    oldest, middle, newest = await _chain(store, org, space, "v1", "v2", "v3")
    await store.erase_memory(org.id, space.id, middle.id)
    edges = await store.list_relations(
        org.id, space.id, newest.id, direction="out", type=RelationType.SUPERSEDES
    )
    bridges = [e for e in edges if e.target_id == oldest.id]
    assert len(bridges) == 1
    assert bridges[0].confidence == pytest.approx(0.9)


async def test_bridge_reason_leaks_nothing_of_the_erased_memory(tenant) -> None:
    """The bridge is created BY an erasure; it must not defeat the erasure.
    Neither the removed memory's id nor any of its content may survive on it."""
    store, org, space = tenant
    oldest, middle, newest = await _chain(store, org, space, "v1", "SECRET-PAYLOAD-42", "v3")
    await store.erase_memory(org.id, space.id, middle.id)
    edges = await store.list_relations(
        org.id, space.id, newest.id, direction="out", type=RelationType.SUPERSEDES
    )
    bridge = next(e for e in edges if e.target_id == oldest.id)
    assert "SECRET-PAYLOAD-42" not in bridge.reason
    assert middle.id not in bridge.reason


async def test_non_supersession_edges_are_not_bridged(tenant) -> None:
    """derived_from does not compose: if B derived from C and A derived from
    B, nothing says A derived from C. A broken derivation must stay visibly
    broken rather than acquire an invented provenance."""
    store, org, space = tenant
    a, b, c = await _chain(store, org, space, "v1", "v2", "v3")
    await store.create_relation(
        RelationEdge(
            org_id=org.id,
            space_id=space.id,
            source_id=c.id,
            target_id=b.id,
            type=RelationType.DERIVED_FROM,
        )
    )
    await store.create_relation(
        RelationEdge(
            org_id=org.id,
            space_id=space.id,
            source_id=b.id,
            target_id=a.id,
            type=RelationType.DERIVED_FROM,
        )
    )
    await store.erase_memory(org.id, space.id, b.id)
    derived = await store.list_relations(
        org.id, space.id, c.id, direction="out", type=RelationType.DERIVED_FROM
    )
    assert all(e.target_id != a.id for e in derived)


async def test_erasing_an_endpoint_bridges_nothing(tenant) -> None:
    """Only middles need bridging. Erasing the newest (no incoming SUPERSEDES)
    or the oldest (no outgoing) must not invent edges."""
    store, org, space = tenant
    oldest, _middle, newest = await _chain(store, org, space, "v1", "v2", "v3")
    report = await store.erase_memory(org.id, space.id, newest.id)
    assert report.edges_bridged == 0
    report = await store.erase_memory(org.id, space.id, oldest.id)
    assert report.edges_bridged == 0


# -- memory kind and the derived lifecycle -------------------------------------


async def test_kind_round_trips_and_defaults_to_episodic(tenant) -> None:
    store, org, space = tenant
    episode = await _add(store, org, space, "plain episode")
    assert episode.kind is MemoryKind.EPISODIC
    derived = await _add(store, org, space, "you own 3 bikes", kind=MemoryKind.DERIVED)
    fetched = await store.get_memory(org.id, space.id, derived.id)
    assert fetched is not None and fetched.kind is MemoryKind.DERIVED


async def test_stale_memories_are_excluded_from_default_search(tenant) -> None:
    """STALE is a hard status: a derivation whose sources changed must vanish
    from results, not rank lower. Serving it is a lie with provenance."""
    store, org, space = tenant
    embedder = DeterministicEmbedder(dimensions=DIMENSIONS)
    fresh = await _add(store, org, space, "kafka current fact", embedder=embedder)
    stale = await _add(
        store, org, space, "kafka derived count", embedder=embedder, kind=MemoryKind.DERIVED
    )
    await store.upsert_memory(
        stale.model_copy(update={"status": MemoryStatus.STALE, "version": 2})
    )
    query = await embedder.embed_one("kafka")
    vector_hits = await store.vector_search(
        org.id, space.id, query, limit=10, filters=MemoryFilter()
    )
    lexical_hits = await store.lexical_search(
        org.id, space.id, "kafka", limit=10, filters=MemoryFilter()
    )
    found = {h.memory_id for h in vector_hits} | {h.memory_id for h in lexical_hits}
    assert fresh.id in found
    assert stale.id not in found


async def test_kind_survives_point_in_time_reads(tenant) -> None:
    """`?as_of=` reconstruction must not silently relabel a derived fact as an
    episode — provenance class is part of what the past looked like."""
    store, org, space = tenant
    derived = await _add(store, org, space, "derived timeline", kind=MemoryKind.DERIVED)
    versions = await store.list_memory_versions(org.id, space.id, derived.id)
    assert versions and versions[-1].kind is MemoryKind.DERIVED
    historical = await store.get_memory_as_of(
        org.id, space.id, derived.id, versions[-1].valid_from
    )
    assert historical is not None and historical.kind is MemoryKind.DERIVED
