"""Keyed memories: a key names one fact, and a write under it replaces it.

The losses these pin were measured live, not imagined:

  * A -> B -> A: the restated A deduplicated into the SUPERSEDED original and
    stayed hidden, so B -- the value the user had just corrected -- stayed
    current.
  * A changed value without a digit in it ("Mocha" -> "Latte") scored cosine
    0.972-0.989 against the old text and was merged into it as a near-duplicate:
    200 created=false, new value gone.
  * An unchanged re-sync bumped versions and rewrote chunks every time.

Store-level tests run on both backends; the service and API tests run on the
reference backend, whose semantics the Postgres one is held to above.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.support.store_backends import BACKENDS, DIMENSIONS, make_store

from mapi.config import Settings
from mapi.core.errors import ConflictError, ValidationError
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import (
    Chunk,
    Memory,
    MemoryStatus,
    Organization,
    RelationType,
    ReplaceMode,
    Space,
)
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.base import MemoryFilter, MemoryStore

ALL = MemoryFilter(statuses=frozenset(MemoryStatus))


class CountingEmbedder(DeterministicEmbedder):
    """Counts provider calls, so "no embedding" is an assertion, not a hope."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.calls = 0

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return await super()._embed_batch(texts)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "store_backend": "memory",
        "embedding_backend": "deterministic",
        "embedding_dimensions": DIMENSIONS,
        "rerank_backend": "none",
        "api_key_pepper": "test-pepper",
        "chunk_target_tokens": 64,
        "chunk_overlap_tokens": 8,
        "embedding_cache_size": 0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(params=BACKENDS)
async def backend(request) -> MemoryStore:
    return await make_store(request.param)


@pytest.fixture
async def tenant(backend: MemoryStore):
    org = await backend.create_organization(Organization(name="Keyed"))
    space = await backend.create_space(
        Space(org_id=org.id, slug=f"k{org.id[-8:]}", name="Keyed")
    )
    return backend, org, space


@pytest.fixture
async def keyed(tenant):
    store, org, space = tenant
    embedder = CountingEmbedder(dimensions=DIMENSIONS)
    service = MemoryService(store, embedder, HeuristicReranker(), _settings())
    return service, embedder, org, space


async def _memory(store: MemoryStore, org, space, text: str, *, key: str | None) -> Memory:
    vector = await DeterministicEmbedder(dimensions=DIMENSIONS).embed_one(text)
    memory = Memory(org_id=org.id, space_id=space.id, content=text, key=key)
    return memory.model_copy(
        update={"chunks": [Chunk(memory_id=memory.id, ordinal=0, text=text, embedding=vector)]}
    )


async def _active_under(store: MemoryStore, org, space, key: str) -> list[Memory]:
    page = await store.list_memories(org.id, space.id, filters=MemoryFilter(), limit=200)
    return [m for m in page.items if m.key == key]


# -- store level: both backends --------------------------------------------------


async def test_a_write_under_a_key_supersedes_the_row_holding_it(tenant) -> None:
    store, org, space = tenant
    first = await store.write_keyed(
        await _memory(store, org, space, "favourite drink: mocha", key="bean:drink"),
        mode=ReplaceMode.SUPERSEDE,
    )
    second = await store.write_keyed(
        await _memory(store, org, space, "favourite drink: latte", key="bean:drink"),
        mode=ReplaceMode.SUPERSEDE,
    )
    assert not first.unchanged and not second.unchanged
    assert second.replaced == [first.memory.id]

    old = await store.get_memory(org.id, space.id, first.memory.id)
    assert old is not None and old.status is MemoryStatus.SUPERSEDED
    assert old.version == 2, "the demotion is a new version, not an in-place edit"
    edges = await store.list_relations(
        org.id, space.id, second.memory.id, type=RelationType.SUPERSEDES
    )
    assert [e.target_id for e in edges] == [first.memory.id]
    active = await _active_under(store, org, space, "bean:drink")
    assert [m.content for m in active] == ["favourite drink: latte"]


async def test_the_same_payload_under_its_key_writes_nothing(tenant) -> None:
    store, org, space = tenant
    first = await store.write_keyed(
        await _memory(store, org, space, "room: Klaus 1116", key="fm:room"),
        mode=ReplaceMode.SUPERSEDE,
    )
    again = await store.write_keyed(
        await _memory(store, org, space, "room:   KLAUS 1116", key="fm:room"),
        mode=ReplaceMode.SUPERSEDE,
    )
    assert again.unchanged
    assert again.memory.id == first.memory.id
    versions = await store.list_memory_versions(org.id, space.id, first.memory.id)
    assert len(versions) == 1, "an unchanged re-sync must not open a version"


async def test_replace_erase_leaves_one_row_and_no_history(tenant) -> None:
    store, org, space = tenant
    first = await store.write_keyed(
        await _memory(store, org, space, "section v1", key="muse:user:projects"),
        mode=ReplaceMode.SUPERSEDE,
    )
    second = await store.write_keyed(
        await _memory(store, org, space, "section v2", key="muse:user:projects"),
        mode=ReplaceMode.SUPERSEDE,
    )
    third = await store.write_keyed(
        await _memory(store, org, space, "section v3", key="muse:user:projects"),
        mode=ReplaceMode.ERASE,
    )
    # ERASE takes the whole history of the key, not just the active row.
    assert sorted(third.replaced) == sorted([first.memory.id, second.memory.id])
    for gone in (first.memory.id, second.memory.id):
        assert await store.get_memory(org.id, space.id, gone) is None
        assert await store.list_memory_versions(org.id, space.id, gone) == []
    page = await store.list_memories(org.id, space.id, filters=ALL, limit=50)
    assert [m.content for m in page.items if m.key == "muse:user:projects"] == ["section v3"]


async def test_two_people_writing_the_same_sentence_keep_two_facts(tenant) -> None:
    store, org, space = tenant
    for person in ("u-1", "u-2"):
        await store.write_keyed(
            await _memory(store, org, space, "Can help with: Rust", key=f"card:{person}:help"),
            mode=ReplaceMode.SUPERSEDE,
        )
    page = await store.list_memories(org.id, space.id, filters=MemoryFilter(), limit=10)
    assert sorted(m.key or "" for m in page.items) == ["card:u-1:help", "card:u-2:help"]
    # Nor may the exact-duplicate gate hand one person's row to an unkeyed
    # writer: a keyed memory changes only through its key.
    from mapi.domain.models import content_hash

    assert (
        await store.find_by_content_hash(org.id, space.id, content_hash("Can help with: Rust"))
        is None
    )


async def test_concurrent_writes_to_one_key_leave_one_active_row(tenant) -> None:
    store, org, space = tenant
    writes = [
        await _memory(store, org, space, f"status update number {n}", key="fm:status")
        for n in range(6)
    ]
    results = await asyncio.gather(
        *(store.write_keyed(m, mode=ReplaceMode.SUPERSEDE) for m in writes)
    )
    active = await _active_under(store, org, space, "fm:status")
    assert len(active) == 1
    assert all(not r.unchanged for r in results)
    page = await store.list_memories(org.id, space.id, filters=ALL, limit=50)
    under_key = [m for m in page.items if m.key == "fm:status"]
    assert len(under_key) == 6
    assert sum(1 for m in under_key if m.status is MemoryStatus.SUPERSEDED) == 5


async def test_a_second_active_row_under_a_key_is_refused(tenant) -> None:
    """The partial unique index, held identically by both backends."""
    store, org, space = tenant
    await store.upsert_memory(await _memory(store, org, space, "one", key="dup"))
    with pytest.raises(ConflictError):
        await store.upsert_memory(await _memory(store, org, space, "two", key="dup"))


async def test_retire_archives_and_reports_what_it_did_not_find(tenant) -> None:
    store, org, space = tenant
    written = await store.write_keyed(
        await _memory(store, org, space, "bank: likes chess", key="muse:bank:1"),
        mode=ReplaceMode.SUPERSEDE,
    )
    retired = await store.retire_keys(org.id, space.id, ["muse:bank:1", "never-written"])
    assert [(m.id, m.status) for m in retired] == [(written.memory.id, MemoryStatus.ARCHIVED)]
    assert retired[0].version == 2
    # A retried retire finds nothing left to do, and says so by returning [].
    assert await store.retire_keys(org.id, space.id, ["muse:bank:1"]) == []
    # The key is free again: a new write creates a new ACTIVE row.
    fresh = await store.write_keyed(
        await _memory(store, org, space, "bank: likes chess", key="muse:bank:1"),
        mode=ReplaceMode.SUPERSEDE,
    )
    assert not fresh.unchanged and fresh.memory.id != written.memory.id


async def test_erase_by_key_reaches_history_whose_row_is_gone(tenant) -> None:
    store, org, space = tenant
    first = await store.write_keyed(
        await _memory(store, org, space, "old card text", key="card:bio"),
        mode=ReplaceMode.SUPERSEDE,
    )
    second = await store.write_keyed(
        await _memory(store, org, space, "new card text", key="card:bio"),
        mode=ReplaceMode.SUPERSEDE,
    )
    # Plain delete keeps the version history by design: that residue is
    # exactly what erasing a key must still find.
    assert await store.delete_memory(org.id, space.id, first.memory.id)
    report = await store.erase_by_key(org.id, space.id, "card:bio")
    assert sorted(report.memory_ids) == sorted([first.memory.id, second.memory.id])
    assert report.versions_purged >= 3
    for gone in (first.memory.id, second.memory.id):
        assert await store.list_memory_versions(org.id, space.id, gone) == []
    again = await store.erase_by_key(org.id, space.id, "card:bio")
    assert again.memory_ids == [] and again.versions_purged == 0


async def test_as_of_reads_carry_the_key(tenant) -> None:
    from datetime import UTC, datetime

    store, org, space = tenant
    written = await store.write_keyed(
        await _memory(store, org, space, "keyed history", key="k:hist"),
        mode=ReplaceMode.SUPERSEDE,
    )
    past = await store.get_memory_as_of(org.id, space.id, written.memory.id, datetime.now(UTC))
    assert past is not None and past.key == "k:hist"


async def test_write_keyed_requires_a_key(tenant) -> None:
    store, org, space = tenant
    with pytest.raises(ValidationError):
        await store.write_keyed(
            await _memory(store, org, space, "no key", key=None), mode=ReplaceMode.SUPERSEDE
        )


async def test_a_keyed_write_must_bring_a_new_id(tenant) -> None:
    """The Postgres backend skips the reads that reconcile an existing row
    for a keyed write's new row. These are the guards that make that safe,
    held identically by both backends -- and nothing moved when they fire."""
    store, org, space = tenant
    first = await store.write_keyed(
        await _memory(store, org, space, "value one", key="g:k"), mode=ReplaceMode.SUPERSEDE
    )
    second = await store.write_keyed(
        await _memory(store, org, space, "value two", key="g:k"), mode=ReplaceMode.SUPERSEDE
    )
    unrelated = await store.upsert_memory(await _memory(store, org, space, "loose", key=None))

    # The very row being replaced.
    # (Built fresh, not model_copy'd: a copy keeps the old content hash.)
    same_row = Memory(
        id=second.memory.id, org_id=org.id, space_id=space.id, content="value 3", key="g:k"
    )
    with pytest.raises(ValidationError):
        await store.write_keyed(same_row, mode=ReplaceMode.SUPERSEDE)
    # A superseded row's id, and an unrelated memory's id.
    for reused in (first.memory.id, unrelated.id):
        clash = Memory(id=reused, org_id=org.id, space_id=space.id, content="v4", key="g:k")
        with pytest.raises(ConflictError):
            await store.write_keyed(clash, mode=ReplaceMode.SUPERSEDE)

    active = await _active_under(store, org, space, "g:k")
    assert [m.id for m in active] == [second.memory.id]
    kept = await store.get_memory(org.id, space.id, unrelated.id)
    assert kept is not None and kept.content == "loose"
    versions = await store.list_memory_versions(org.id, space.id, second.memory.id)
    assert len(versions) == 1, "a refused write left no trace"


# -- service level ------------------------------------------------------------------


async def test_a_to_b_to_a_leaves_a_current(keyed) -> None:
    service, _embedder, org, space = keyed
    a1 = await service.ingest(
        org_id=org.id, space_id=space.id, content="I live in Austin", key="fact:home"
    )
    b = await service.ingest(
        org_id=org.id, space_id=space.id, content="I live in Seattle", key="fact:home"
    )
    a2 = await service.ingest(
        org_id=org.id, space_id=space.id, content="I live in Austin", key="fact:home"
    )
    assert a2.created and a2.memory.id not in (a1.memory.id, b.memory.id)
    assert a2.superseded == [b.memory.id]
    active = await service.store.get_active_by_keys(org.id, space.id, ["fact:home"])
    assert active["fact:home"].content == "I live in Austin"

    response = await service.search(
        SearchRequest(query="where do I live", org_id=org.id, space_id=space.id, limit=5)
    )
    ids = [r.memory.id for r in response.results]
    assert a2.memory.id in ids
    assert a1.memory.id not in ids and b.memory.id not in ids


async def test_a_to_b_to_a_without_keys_no_longer_resurrects_a_hidden_row(keyed) -> None:
    """The unkeyed form of the same bug: the restated A used to dedupe into
    the superseded original and come back `created=False`, still hidden."""
    service, _embedder, org, space = keyed
    a1 = await service.ingest(org_id=org.id, space_id=space.id, content="I live in Austin")
    b = await service.ingest(org_id=org.id, space_id=space.id, content="I live in Seattle")
    await service.link(
        org.id,
        space.id,
        source_id=b.memory.id,
        target_id=a1.memory.id,
        relation=RelationType.SUPERSEDES,
    )
    a2 = await service.ingest(org_id=org.id, space_id=space.id, content="I live in Austin")
    assert a2.created, "an exact restatement of a SUPERSEDED memory is a new fact"
    assert a2.memory.status is MemoryStatus.ACTIVE


async def test_near_duplicates_never_merge_into_superseded_rows(keyed) -> None:
    service, _embedder, org, space = keyed
    old = await service.ingest(
        org_id=org.id, space_id=space.id, content="Alice works in building 7 on robotics"
    )
    newer = await service.ingest(
        org_id=org.id, space_id=space.id, content="Alice moved to the chemistry lab"
    )
    await service.link(
        org.id,
        space.id,
        source_id=newer.memory.id,
        target_id=old.memory.id,
        relation=RelationType.SUPERSEDES,
    )
    # Same tokens, different hash: near-identical to the superseded row.
    restated = await service.ingest(
        org_id=org.id, space_id=space.id, content="Alice works in building 7 on robotics."
    )
    assert restated.created
    assert restated.duplicate_of is None


async def test_an_unchanged_keyed_resend_costs_no_embedding(keyed) -> None:
    service, embedder, org, space = keyed
    first = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="stuck on: CUDA drivers",
        key="card:stuck_on",
        metadata={"person": "u-1"},
        tags=["card"],
    )
    calls = embedder.calls
    again = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="stuck on: CUDA drivers",
        key="card:stuck_on",
        metadata={"person": "u-1"},
        tags=["card"],
    )
    assert embedder.calls == calls
    assert not again.created
    assert again.duplicate_of == first.memory.id
    assert again.memory.version == first.memory.version


async def test_a_metadata_change_under_a_key_is_a_change(keyed) -> None:
    service, _embedder, org, space = keyed
    first = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="stuck on: CUDA",
        key="card:stuck_on",
        metadata={"visibility": "private"},
    )
    second = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="stuck on: CUDA",
        key="card:stuck_on",
        metadata={"visibility": "directory"},
    )
    assert second.created and second.superseded == [first.memory.id]
    assert second.memory.metadata == {"visibility": "directory"}


async def test_keyed_writes_skip_near_duplicate_merging(keyed) -> None:
    """A digit-free value change scores ~0.98 against its old text; unkeyed,
    it merged into the OLD content. Keyed, it replaces it."""
    service, _embedder, org, space = keyed
    first = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="Krishna's bean in GT Campus Quest: round, color Mocha",
        key="fm:bean",
    )
    second = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="Krishna's bean in GT Campus Quest: round, color Latte",
        key="fm:bean",
    )
    assert second.created and second.duplicate_of is None
    assert second.superseded == [first.memory.id]


async def test_replace_erase_through_the_service_reports_erased(keyed) -> None:
    service, _embedder, org, space = keyed
    first = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="notes v1",
        key="muse:mem:notes",
        replace=ReplaceMode.ERASE,
    )
    second = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="notes v2",
        key="muse:mem:notes",
        replace="erase",
    )
    assert second.erased == [first.memory.id] and second.superseded == []
    assert await service.store.list_memory_versions(org.id, space.id, first.memory.id) == []


async def test_an_unkeyed_restatement_does_not_merge_into_a_keyed_row(keyed) -> None:
    service, _embedder, org, space = keyed
    keyed_row = await service.ingest(
        org_id=org.id, space_id=space.id, content="Looking for: a co-founder", key="card:look"
    )
    loose = await service.ingest(
        org_id=org.id, space_id=space.id, content="Looking for: a co-founder"
    )
    assert loose.created and loose.memory.id != keyed_row.memory.id
    stored = await service.store.get_memory(org.id, space.id, keyed_row.memory.id)
    assert stored is not None and stored.version == 1, "the keyed row was left alone"


async def test_superseding_a_keyed_row_stales_what_was_derived_from_it(keyed) -> None:
    service, _embedder, org, space = keyed
    source = await service.ingest(
        org_id=org.id, space_id=space.id, content="team size is four", key="fact:team"
    )
    derived = await service.ingest(org_id=org.id, space_id=space.id, content="a small team")
    await service.link(
        org.id,
        space.id,
        source_id=derived.memory.id,
        target_id=source.memory.id,
        relation=RelationType.DERIVED_FROM,
    )
    await service.ingest(
        org_id=org.id, space_id=space.id, content="team size is nine", key="fact:team"
    )
    stale = await service.store.get_memory(org.id, space.id, derived.memory.id)
    assert stale is not None and stale.status is MemoryStatus.STALE


async def test_retire_through_the_service(keyed) -> None:
    service, _embedder, org, space = keyed
    await service.ingest(org_id=org.id, space_id=space.id, content="bank item", key="b:1")
    retired, missing = await service.retire_keys(org.id, space.id, ["b:1", "b:2"])
    assert [m.key for m in retired] == ["b:1"] and missing == ["b:2"]
    with pytest.raises(ValidationError):
        await service.retire_keys(org.id, space.id, ["b:1"], status=MemoryStatus.STALE)


async def test_bad_keys_are_validation_errors_not_crashes(keyed) -> None:
    service, _embedder, org, space = keyed
    for bad in ("x" * 201, "tab\there"):
        with pytest.raises(ValidationError):
            await service.ingest(org_id=org.id, space_id=space.id, content="c", key=bad)


# -- HTTP ------------------------------------------------------------------------------


async def test_http_keyed_write_round_trip(client, space_id) -> None:
    url = f"/v1/spaces/{space_id}/memories"
    first = await client.post(url, json={"content": "building: a robot arm", "key": "card:b"})
    assert first.status_code == 201
    assert first.json()["memory"]["key"] == "card:b"

    same = await client.post(url, json={"content": "building: a robot arm", "key": "card:b"})
    assert same.status_code == 200
    assert same.json()["created"] is False
    assert same.json()["duplicate_of"] == first.json()["memory"]["id"]

    changed = await client.post(
        url, json={"content": "building: a drone", "key": "card:b", "replace": "erase"}
    )
    assert changed.status_code == 201
    assert changed.json()["erased"] == [first.json()["memory"]["id"]]


async def test_http_replace_without_a_key_is_422(client, space_id) -> None:
    response = await client.post(
        f"/v1/spaces/{space_id}/memories", json={"content": "x", "replace": "erase"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("key", ["", "a" * 201, "line\nbreak"])
async def test_http_malformed_keys_are_422(client, space_id, key: str) -> None:
    response = await client.post(
        f"/v1/spaces/{space_id}/memories", json={"content": "x", "key": key}
    )
    assert response.status_code == 422


async def test_http_retire_and_erase_by_key(client, space_id) -> None:
    base = f"/v1/spaces/{space_id}/memories"
    written = (
        await client.post(base, json={"content": "section text", "key": "muse:user:a/b"})
    ).json()["memory"]
    retired = await client.post(f"{base}/retire", json={"keys": ["muse:user:a/b", "gone"]})
    assert retired.status_code == 200
    assert retired.json()["retired"] == [
        {"key": "muse:user:a/b", "memory_id": written["id"], "version": 2}
    ]
    assert retired.json()["missing"] == ["gone"]

    # A key with '/' and '#' in it, percent-encoded as a client must.
    erased = await client.delete(f"{base}/by-key/muse%3Auser%3Aa/b")
    assert erased.status_code == 200
    assert erased.json()["memory_ids"] == [written["id"]]
    assert erased.json()["versions_purged"] == 2
    again = await client.delete(f"{base}/by-key/muse%3Auser%3Aa/b")
    assert again.status_code == 200 and again.json()["memories_erased"] == 0

    bad_status = await client.post(f"{base}/retire", json={"keys": ["k"], "status": "stale"})
    assert bad_status.status_code == 422
