"""Right-to-delete primitives: purge a space, erase by tag, idempotent erase.

What these close, each probed live before the fix:

  * DELETE /v1/spaces/{id} removes the live rows and -- by the audit-trail
    design -- keeps `memory_versions`, which has no foreign key to cascade
    through. The deleted space's full content stayed readable through
    `/versions`. Purge takes versions, edges, chunks, memories and the space
    in ONE transaction and reports every count.
  * Forgetting one person in a shared space took list + page + N erases, any
    of which could fail half way. Erase-by-tag is one call, one transaction.
  * Erasing an id with nothing left was a 404 -- contradicting DECISIONS.md,
    and turning every retried compliance request into a failure.

Store-level tests run on both backends. Row counts are read straight from
the tables (under the tenant scope row-level security demands), so "0
memory_versions afterwards" is a statement about the database, not about
what an API chose to show.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text
from tests.support.store_backends import BACKENDS, DIMENSIONS, make_store

from mapi.config import Settings
from mapi.core.security import build_api_key
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import (
    MemoryKind,
    MemoryStatus,
    Organization,
    RelationEdge,
    RelationType,
    ReplaceMode,
    Scope,
    Space,
)
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.base import MemoryFilter, MemoryStore
from mapi.store.memory import InMemoryStore

TABLES = ("spaces", "memories", "chunks", "relation_edges", "memory_versions")
EVERY_STATUS = MemoryFilter(statuses=frozenset(MemoryStatus))


async def rows(store: MemoryStore, org_id: str, space_id: str) -> dict[str, int]:
    """Rows per table that belong to `space_id`, read from storage itself."""
    if isinstance(store, InMemoryStore):
        live = [m for m in store._memories.values() if m.space_id == space_id]
        return {
            "spaces": int(space_id in store._spaces),
            "memories": len(live),
            "chunks": sum(len(m.chunks) for m in live),
            "relation_edges": sum(1 for e in store._edges.values() if e.space_id == space_id),
            "memory_versions": sum(
                1 for vs in store._versions.values() for v in vs if v.space_id == space_id
            ),
        }
    pg: Any = store
    counts: dict[str, int] = {}
    async with pg._session() as session, session.begin():
        await pg._scope(session, org_id)
        for table in TABLES:
            column = "id" if table == "spaces" else "space_id"
            counts[table] = int(
                await session.scalar(
                    text(f"SELECT count(*) FROM {table} WHERE {column} = :sid"),
                    {"sid": space_id},
                )
                or 0
            )
    return counts


def _settings() -> Settings:
    return Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=DIMENSIONS,
        rerank_backend="none",
        api_key_pepper="test-pepper",
        chunk_target_tokens=64,
        chunk_overlap_tokens=8,
    )


@pytest.fixture(params=BACKENDS)
async def backend(request) -> MemoryStore:
    return await make_store(request.param)


@pytest.fixture
async def world(backend: MemoryStore):
    """Two orgs; the first has two spaces. Only `space` gets purged."""
    store = backend
    org = await store.create_organization(Organization(name="Purge"))
    space = await store.create_space(Space(org_id=org.id, slug=f"p{org.id[-8:]}", name="P"))
    sibling = await store.create_space(
        Space(org_id=org.id, slug=f"s{org.id[-8:]}", name="Sibling")
    )
    other_org = await store.create_organization(Organization(name="Other"))
    other = await store.create_space(
        Space(org_id=other_org.id, slug=f"o{other_org.id[-8:]}", name="Other")
    )
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=DIMENSIONS), HeuristicReranker(), _settings()
    )
    return service, org, space, sibling, other_org, other


async def _populate(service: MemoryService, org_id: str, space_id: str) -> None:
    """Every kind of row a space can hold, history and residue included."""
    a = await service.ingest(
        org_id=org_id, space_id=space_id, content="card bio: builds robot arms", key="card:bio"
    )
    await service.ingest(
        org_id=org_id, space_id=space_id, content="card bio: builds drones", key="card:bio"
    )
    plain = await service.ingest(
        org_id=org_id, space_id=space_id, content="met Priya at the robotics table"
    )
    doomed = await service.ingest(
        org_id=org_id, space_id=space_id, content="a memory that gets deleted, not erased"
    )
    await service.link(
        org_id,
        space_id,
        source_id=plain.memory.id,
        target_id=a.memory.id,
        relation=RelationType.REFERENCES,
    )
    # Plain delete: the live row goes, the history stays. Purge must find it.
    assert await service.store.delete_memory(org_id, space_id, doomed.memory.id)


# -- purge: store level, both backends -------------------------------------------------


async def test_purge_leaves_nothing_behind_not_even_history(world) -> None:
    service, org, space, _sibling, _other_org, _other = world
    await _populate(service, org.id, space.id)
    before = await rows(service.store, org.id, space.id)
    assert before["memory_versions"] >= 5 and before["relation_edges"] >= 2

    report = await service.purge_space(org.id, space.id)

    assert (
        report.spaces,
        report.memories,
        report.chunks,
        report.relation_edges,
        report.memory_versions,
    ) == (
        before["spaces"],
        before["memories"],
        before["chunks"],
        before["relation_edges"],
        before["memory_versions"],
    ), "the report is the attestation: it must match what was really there"
    after = await rows(service.store, org.id, space.id)
    assert after == dict.fromkeys(TABLES, 0)
    assert after["memory_versions"] == 0


async def test_a_second_purge_reports_zeros(world) -> None:
    service, org, space, *_ = world
    await _populate(service, org.id, space.id)
    await service.purge_space(org.id, space.id)
    again = await service.purge_space(org.id, space.id)
    assert (
        again.spaces,
        again.memories,
        again.chunks,
        again.relation_edges,
        again.memory_versions,
    ) == (0, 0, 0, 0, 0)


async def test_purge_after_delete_removes_the_history_delete_left(world) -> None:
    service, org, space, *_ = world
    await _populate(service, org.id, space.id)
    assert await service.store.delete_space(org.id, space.id)
    residue = await rows(service.store, org.id, space.id)
    assert residue["spaces"] == 0 and residue["memory_versions"] > 0, (
        "the premise: DELETE keeps history, which is why purge exists"
    )
    report = await service.purge_space(org.id, space.id)
    assert report.spaces == 0 and report.memory_versions == residue["memory_versions"]
    assert await rows(service.store, org.id, space.id) == dict.fromkeys(TABLES, 0)


async def test_a_purged_space_is_not_served_from_the_lookup_cache(world) -> None:
    """Purge evicts the space cache, as DELETE does: no writes into a ghost."""
    from mapi.core.errors import NotFoundError

    service, org, space, *_ = world
    service.settings = service.settings.model_copy(update={"space_cache_ttl_s": 3600})
    assert (await service.get_space_or_raise(org.id, space.id)).id == space.id
    await service.purge_space(org.id, space.id)
    with pytest.raises(NotFoundError):
        await service.get_space_or_raise(org.id, space.id)


async def test_purge_touches_only_its_own_space(world) -> None:
    service, org, space, sibling, other_org, other = world
    for org_id, space_id in (
        (org.id, space.id),
        (org.id, sibling.id),
        (other_org.id, other.id),
    ):
        await _populate(service, org_id, space_id)
    sibling_before = await rows(service.store, org.id, sibling.id)
    other_before = await rows(service.store, other_org.id, other.id)

    await service.purge_space(org.id, space.id)

    assert await rows(service.store, org.id, sibling.id) == sibling_before
    assert await rows(service.store, other_org.id, other.id) == other_before


async def test_purging_another_orgs_space_does_nothing(world) -> None:
    service, org, _space, _sibling, other_org, other = world
    await _populate(service, other_org.id, other.id)
    before = await rows(service.store, other_org.id, other.id)
    report = await service.purge_space(org.id, other.id)
    assert (report.spaces, report.memories, report.memory_versions) == (0, 0, 0)
    assert await rows(service.store, other_org.id, other.id) == before


# -- erase by tag -----------------------------------------------------------------------


async def test_erase_by_tag_takes_every_status_kind_and_history(world) -> None:
    service, org, space, *_ = world
    tag = "person:u-42"
    active = await service.ingest(
        org_id=org.id, space_id=space.id, content="u-42 likes pottery", tags=[tag]
    )
    keyed_old = await service.ingest(
        org_id=org.id, space_id=space.id, content="u-42 stuck on CUDA", key="c:42", tags=[tag]
    )
    keyed_new = await service.ingest(
        org_id=org.id, space_id=space.id, content="u-42 stuck on ROS", key="c:42", tags=[tag]
    )
    archived = await service.ingest(
        org_id=org.id, space_id=space.id, content="u-42 archived note", key="c:43", tags=[tag]
    )
    await service.retire_keys(org.id, space.id, ["c:43"])
    derived = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="u-42 is into hardware",
        tags=[tag],
        kind=MemoryKind.DERIVED,
    )
    # Retagged since: its CURRENT tags lack the tag, its history has it.
    retagged = await service.ingest(
        org_id=org.id, space_id=space.id, content="u-42 early draft", tags=[tag]
    )
    current = await service.store.get_memory(org.id, space.id, retagged.memory.id)
    assert current is not None
    await service.store.upsert_memory(
        current.model_copy(update={"tags": ["unrelated"], "version": current.version + 1})
    )
    bystander = await service.ingest(
        org_id=org.id, space_id=space.id, content="u-7 likes chess", tags=["person:u-7"]
    )

    report = await service.erase_by_tag(org.id, space.id, tag)

    doomed = {
        active.memory.id,
        keyed_old.memory.id,
        keyed_new.memory.id,
        archived.memory.id,
        derived.memory.id,
        retagged.memory.id,
    }
    assert set(report["memory_ids"]) == doomed  # type: ignore[arg-type]
    assert report["memories_erased"] == len(doomed)
    for memory_id in doomed:
        assert await service.store.get_memory(org.id, space.id, memory_id) is None
        assert await service.store.list_memory_versions(org.id, space.id, memory_id) == []
    assert await service.store.get_memory(org.id, space.id, bystander.memory.id) is not None

    again = await service.erase_by_tag(org.id, space.id, tag)
    assert again["memories_erased"] == 0 and again["versions_purged"] == 0


async def test_erase_by_tag_matches_the_stored_form_of_the_tag(world) -> None:
    service, org, space, *_ = world
    await service.ingest(
        org_id=org.id, space_id=space.id, content="tagged", tags=["Person:U-9"]
    )
    report = await service.erase_by_tag(org.id, space.id, "  PERSON:u-9 ")
    assert report["memories_erased"] == 1 and report["tag"] == "person:u-9"


async def test_erase_by_tag_stales_what_survivors_derived(world) -> None:
    service, org, space, *_ = world
    source = await service.ingest(
        org_id=org.id, space_id=space.id, content="u-5 works at the robotics lab", tags=["p:5"]
    )
    summary = await service.ingest(
        org_id=org.id, space_id=space.id, content="several people do robotics"
    )
    await service.store.create_relation(
        RelationEdge(
            org_id=org.id,
            space_id=space.id,
            source_id=summary.memory.id,
            target_id=source.memory.id,
            type=RelationType.DERIVED_FROM,
            reason="summarised",
        )
    )
    report = await service.erase_by_tag(org.id, space.id, "p:5")
    assert report["derived_memories_affected"] == [summary.memory.id]
    stale = await service.store.get_memory(org.id, space.id, summary.memory.id)
    assert stale is not None and stale.status is MemoryStatus.STALE


async def test_erase_by_key_and_keyed_erase_mode_leave_no_history(world) -> None:
    service, org, space, *_ = world
    for text_ in ("v1", "v2", "v3"):
        await service.ingest(
            org_id=org.id, space_id=space.id, content=f"section {text_}", key="muse:s"
        )
    report = await service.erase_by_key(org.id, space.id, "muse:s")
    assert report["memories_erased"] == 3
    assert await rows(service.store, org.id, space.id) == {
        "spaces": 1,
        "memories": 0,
        "chunks": 0,
        "relation_edges": 0,
        "memory_versions": 0,
    }
    written = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="fresh",
        key="muse:s",
        replace=ReplaceMode.ERASE,
    )
    assert written.created and written.erased == []


# -- idempotent erase ---------------------------------------------------------------------


async def test_erasing_twice_is_two_successes(world) -> None:
    service, org, space, *_ = world
    written = await service.ingest(org_id=org.id, space_id=space.id, content="erase me")
    first = await service.erase_memory(org.id, space.id, written.memory.id)
    second = await service.erase_memory(org.id, space.id, written.memory.id)
    assert first["already_erased"] is False and first["versions_purged"] == 1
    assert second["already_erased"] is True
    assert second["versions_purged"] == 0 and second["chunks_removed"] == 0


# -- HTTP ---------------------------------------------------------------------------------


async def test_http_purge(client, space_id) -> None:
    base = f"/v1/spaces/{space_id}"
    written = await client.post(f"{base}/memories", json={"content": "to be purged"})
    memory_id = written.json()["memory"]["id"]
    await client.delete(f"{base}/memories/{memory_id}")

    response = await client.post(f"{base}/purge")
    assert response.status_code == 200
    body = response.json()
    assert body["space_id"] == space_id
    assert body["spaces"] == 1 and body["memories"] == 0 and body["memory_versions"] == 1
    assert body["purged_at"]

    assert (await client.get(base)).status_code == 404
    versions = await client.get(f"{base}/memories/{memory_id}/versions")
    assert versions.status_code == 404 or versions.json().get("items") == []

    again = await client.post(f"{base}/purge")
    assert again.status_code == 200
    assert {k: again.json()[k] for k in TABLES} == dict.fromkeys(TABLES, 0)


async def test_http_purge_needs_spaces_write(app_context, anon_client, space_id) -> None:
    record, plaintext = build_api_key(
        org_id=app_context["org"].id,
        name="reader",
        pepper="test-pepper",
        scopes=frozenset({Scope.MEMORIES_READ, Scope.MEMORIES_WRITE}),
    )
    await app_context["store"].create_api_key(record)
    response = await anon_client.post(
        f"/v1/spaces/{space_id}/purge", headers={"Authorization": f"Bearer {plaintext}"}
    )
    assert response.status_code == 403
    assert await app_context["store"].get_space(app_context["org"].id, space_id) is not None


async def test_http_purge_rejects_a_malformed_id(client) -> None:
    response = await client.post("/v1/spaces/not-a-space/purge")
    assert response.status_code == 422


async def test_http_erase_by_tag(client, space_id) -> None:
    base = f"/v1/spaces/{space_id}/memories"
    for n in range(3):
        await client.post(base, json={"content": f"u-1 fact {n}", "tags": ["person:u-1"]})
    await client.post(base, json={"content": "u-2 fact", "tags": ["person:u-2"]})

    response = await client.post(f"{base}/erase-by-tag", json={"tag": "person:u-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["tag"] == "person:u-1" and body["memories_erased"] == 3
    assert body["versions_purged"] == 3

    listing = (await client.get(base)).json()
    assert [m["content"] for m in listing["items"]] == ["u-2 fact"]
    again = await client.post(f"{base}/erase-by-tag", json={"tag": "person:u-1"})
    assert again.status_code == 200 and again.json()["memories_erased"] == 0


async def test_http_erase_by_tag_needs_a_tag(client, space_id) -> None:
    response = await client.post(f"/v1/spaces/{space_id}/memories/erase-by-tag", json={})
    assert response.status_code == 422
