"""Batched bulk writes decide and store exactly what the sequential path did.

`ingest_many` used to write a bulk request one store call at a time, ~30
round trips per item. The batched path (`bulkwrite`, `store/postgres/bulk.py`)
runs the same write-path methods against an overlay inside one transaction.
It is only an optimisation if nothing observable changes, so every test here
runs a request through BOTH paths, in two spaces seeded identically, and
compares everything a caller or a later read can see:

  * each item's outcome -- created or not, duplicate kind and target,
    similarity, superseded / erased / contradicted ids, chunk count, the
    memory as returned, or the error and its status;
  * every row left in the space, in every status, with its version history;
  * every edge.

Ids differ between the two spaces, so they are compared through labels: a
seed's name, or the index of the bulk item that created the row.

Both backends: the reference store runs the batched path by replaying its
operations, Postgres by compiling them into multi-row statements.
"""

from __future__ import annotations

from typing import Any

import pytest
from bench.bulk_write import attendee_sections, count_round_trips
from tests.support.store_backends import BACKENDS, DIMENSIONS, make_store

from mapi.bulkwrite import BulkStats
from mapi.config import Settings
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import (
    MemoryKind,
    MemoryStatus,
    Organization,
    RelationEdge,
    RelationType,
    ReplaceMode,
    Space,
)
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import BulkOutcome, IngestItem, MemoryService
from mapi.store.base import MemoryFilter, MemoryStore
from mapi.store.memory import InMemoryStore
from mapi.store.postgres.bulk import compile_ops

ALL = MemoryFilter(statuses=frozenset(MemoryStatus))


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "store_backend": "memory",
        "embedding_backend": "deterministic",
        "embedding_dimensions": DIMENSIONS,
        "rerank_backend": "none",
        "api_key_pepper": "test-pepper",
        "embedding_cache_size": 0,
        "max_content_bytes": 20_000,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(params=BACKENDS)
async def backend(request) -> MemoryStore:
    return await make_store(request.param)


def _service(store: MemoryStore, *, batched: bool, **settings: object) -> MemoryService:
    return MemoryService(
        store,
        DeterministicEmbedder(dimensions=DIMENSIONS, batch_size=64),
        HeuristicReranker(),
        _settings(bulk_write_batching=batched, **settings),
    )


async def _space(store: MemoryStore, org: Organization, name: str) -> Space:
    return await store.create_space(
        Space(org_id=org.id, slug=f"{name}-{org.id[-8:]}", name=name)
    )


# -- observation --------------------------------------------------------------------------


class Labels:
    """Space-independent names for memory ids."""

    def __init__(self, seeds: dict[str, str]) -> None:
        self.by_id = {memory_id: name for name, memory_id in seeds.items()}

    def learn(self, outcomes: list[BulkOutcome], prefix: str = "item") -> None:
        for o in outcomes:
            if o.result is not None and o.result.created:
                self.by_id.setdefault(o.result.memory.id, f"{prefix}{o.index}")

    def __call__(self, memory_id: str | None) -> str | None:
        if memory_id is None:
            return None
        return self.by_id.get(memory_id, "?")


def _outcomes(outcomes: list[BulkOutcome], label: Labels) -> list[Any]:
    out: list[Any] = []
    for o in outcomes:
        if o.error is not None or o.result is None:
            out.append(
                (o.index, "error", type(o.error).__name__, getattr(o.error, "status_code", 0))
            )
            continue
        r = o.result
        m = r.memory
        out.append(
            (
                o.index,
                r.created,
                str(r.duplicate_kind),
                label(r.duplicate_of),
                round(r.similarity, 9),
                sorted(label(i) or "" for i in r.superseded),
                sorted(label(i) or "" for i in r.erased),
                sorted(label(i) or "" for i in r.contradicts),
                sorted(label(i) or "" for i in r.supersede_declined),
                r.chunk_count,
                label(m.id),
                m.version,
                m.status.value,
                m.content,
                sorted(m.tags),
                m.metadata,
                m.key,
                m.source,
                len(m.chunks),
            )
        )
    return out


async def _state(store: MemoryStore, org: Organization, space: Space, label: Labels) -> Any:
    page = await store.list_memories(org.id, space.id, filters=ALL, limit=200)
    rows = []
    edges = set()
    for m in page.items:
        versions = await store.list_memory_versions(org.id, space.id, m.id)
        rows.append(
            (
                label(m.id),
                m.status.value,
                m.version,
                m.kind.value,
                m.content,
                sorted(m.tags),
                m.metadata,
                m.key,
                m.source,
                len(m.chunks),
                [
                    (v.version, v.status.value, v.content, v.key, v.valid_to is None)
                    for v in sorted(versions, key=lambda v: v.version)
                ],
            )
        )
        for e in await store.list_relations(org.id, space.id, m.id, direction="out"):
            edges.add(
                (label(e.source_id), label(e.target_id), e.type.value, e.reason, e.confidence)
            )
    return sorted(rows, key=lambda r: str(r[0])), sorted(edges)


# -- the fixture every behaviour lives in ------------------------------------------------


async def _seed(service: MemoryService, org: Organization, space: Space) -> dict[str, str]:
    async def put(text: str, **kw: Any) -> str:
        result = await service.ingest(org_id=org.id, space_id=space.id, content=text, **kw)
        return result.memory.id

    seeds = {
        "austin": await put("Lives in Austin, near the lake"),
        "room": await put("room: Klaus 1116", key="fm:room"),
        "secret": await put("pin: 1234", key="fm:secret"),
        "job": await put("job: engineer at Initech", key="fm:job"),
        "priya": await put("Priya Raman leads the robotics lab at Globex", tags=["people"]),
    }
    # Something DERIVED_FROM the keyed job row, so replacing it must mark the
    # derivation stale -- a walk only the sequential path does.
    derived = await service.ingest(
        org_id=org.id,
        space_id=space.id,
        content="Works in software at a company called Initech",
        kind=MemoryKind.DERIVED,
    )
    seeds["derived"] = derived.memory.id
    await service.store.create_relation(
        RelationEdge(
            org_id=org.id,
            space_id=space.id,
            source_id=derived.memory.id,
            target_id=seeds["job"],
            type=RelationType.DERIVED_FROM,
            reason="seeded",
        )
    )
    return seeds


def _items() -> list[IngestItem]:
    maya = "Maya climbs at Yosemite every spring"
    return [
        # 0: exact restatement of a stored memory -> folds into it, tags merge
        IngestItem(content="lives in  AUSTIN, near the lake", tags=["home"]),
        # 1: new
        IngestItem(content=maya, source="muse:1"),
        # 2: exact restatement of item 1, inside the request
        IngestItem(content=maya.lower(), tags=["again"]),
        # 3: near duplicate of item 1 (same words, new punctuation)
        IngestItem(content=maya + "!", source="muse:1"),
        # 4: keyed replace of a stored key
        IngestItem(content="room: CULC 144", key="fm:room"),
        # 5-7: A -> B -> A under one key, all inside the request
        IngestItem(content="mood: happy", key="fm:mood"),
        IngestItem(content="mood: sad", key="fm:mood"),
        IngestItem(content="mood: happy", key="fm:mood"),
        # 8: unchanged keyed payload -> nothing written
        IngestItem(content="room: CULC 144", key="fm:room"),
        # 9: shares referents with a stored memory -> an association edge
        IngestItem(
            content="Priya Raman mentors the robotics students at Globex", tags=["people"]
        ),
        # 10: invalid alone
        IngestItem(content="   "),
        # 11: ERASE with history under the key -> the sequential path, mid-request
        IngestItem(content="pin: 9999", key="fm:secret", replace=ReplaceMode.ERASE),
        # 12: exact restatement of item 1, across that segment boundary
        IngestItem(content=maya),
        # 13: replaces a row something is derived from -> sequential again
        IngestItem(content="job: staff engineer at Initech", key="fm:job"),
        # 14: exact restatement of item 9
        IngestItem(content="Priya Raman mentors the robotics students at Globex"),
        # 15: too large alone
        IngestItem(content="x" * 25_000),
        # 16-18: A -> B -> A without a key: the third folds into the first
        IngestItem(content="status: green across the board"),
        IngestItem(content="status: amber, two alerts open"),
        IngestItem(content="status: green across the board", tags=["later"]),
        # 19: a keyed first write with ERASE and no history
        IngestItem(content="badge: 42", key="fm:badge", replace=ReplaceMode.ERASE),
        # 20: keyed first write, SUPERSEDE
        IngestItem(content="team: blue", key="fm:team"),
    ]


async def _run(store: MemoryStore, org: Organization, *, batched: bool, name: str) -> Any:
    service = _service(store, batched=batched)
    space = await _space(store, org, name)
    seeds = await _seed(service, org, space)
    label = Labels(seeds)
    stats = BulkStats()
    outcomes = await service.ingest_many(
        org_id=org.id, space_id=space.id, items=_items(), stats=stats
    )
    label.learn(outcomes)
    return _outcomes(outcomes, label), await _state(store, org, space, label), stats


async def test_batched_bulk_matches_sequential_bulk(backend: MemoryStore) -> None:
    org = await backend.create_organization(Organization(name="Equivalence"))
    sequential = await _run(backend, org, batched=False, name="seq")
    batched = await _run(backend, org, batched=True, name="bat")

    assert batched[0] == sequential[0], "per-item outcomes differ"
    assert batched[1][0] == sequential[1][0], "stored rows or their versions differ"
    assert batched[1][1] == sequential[1][1], "edges differ"

    stats: BulkStats = batched[2]
    # The ERASE (11) and the derived replace (13) went sequential, splitting
    # the batch around them; everything else was batched.
    assert stats.sequential_items == 3  # 11, 13 and the ERASE first write (19)
    assert stats.segments == 4
    assert sequential[2].segments == 0

    outcomes = {row[0]: row for row in batched[0]}
    # Spot checks, so "equal" cannot mean "equally broken".
    assert outcomes[0][1:4] == (False, "exact", "austin")
    assert outcomes[2][1:4] == (False, "exact", "item1")
    assert outcomes[3][1:4] == (False, "near", "item1")
    assert outcomes[4][1] and outcomes[4][5] == ["room"]
    assert outcomes[6][5] == ["item5"] and outcomes[7][5] == ["item6"]
    assert outcomes[8][1:4] == (False, "exact", "item4")
    assert outcomes[10][1:3] == ("error", "ValidationError")
    assert outcomes[11][6] == ["secret"]
    assert outcomes[12][1:4] == (False, "exact", "item1")
    assert outcomes[13][5] == ["job"]
    assert outcomes[14][1:4] == (False, "exact", "item9")
    assert outcomes[15][1:4] == ("error", "PayloadTooLargeError", 413)
    assert outcomes[18][1:4] == (False, "exact", "item16")

    rows = {row[0]: row for row in batched[1][0]}
    assert rows["derived"][1] == "stale"
    assert rows["room"][1] == "superseded"
    assert "secret" not in rows
    edges = batched[1][1]
    assert ("item9", "priya", "references") in {e[:3] for e in edges}
    assert ("item4", "room", "supersedes") in {e[:3] for e in edges}


async def test_attendee_sync_matches_sequential(backend: MemoryStore) -> None:
    """The workload the batching is for: sections, a re-sync, an edit."""
    n = 91 if isinstance(backend, InMemoryStore) else 14
    first = attendee_sections(n)
    edited = [
        {**s, "content": s["content"] + " Edited in 2026."} if i % 4 == 0 else s
        for i, s in enumerate(first)
    ]
    org = await backend.create_organization(Organization(name="Attendee"))
    results = []
    for batched in (False, True):
        service = _service(backend, batched=batched)
        space = await _space(backend, org, "bat" if batched else "seq")
        label = Labels({})
        seen = []
        for round_, sections in enumerate((first, first, edited)):
            outcomes = await service.ingest_many(
                org_id=org.id,
                space_id=space.id,
                items=[IngestItem(**s) for s in sections],
            )
            label.learn(outcomes, prefix=f"r{round_}.")
            seen.append(_outcomes(outcomes, label))
        results.append((seen, await _state(backend, org, space, label)))
    sequential, batched_ = results
    assert batched_[0] == sequential[0]
    assert batched_[1] == sequential[1]
    created = [sum(1 for row in rnd if row[1] is True) for rnd in batched_[0]]
    assert created[0] == n and created[1] == 0


async def test_a_plain_bulk_is_one_segment_and_nothing_sequential() -> None:
    store = InMemoryStore()
    org = await store.create_organization(Organization(name="Plain"))
    space = await _space(store, org, "plain")
    stats = BulkStats()
    outcomes = await _service(store, batched=True).ingest_many(
        org_id=org.id,
        space_id=space.id,
        items=[IngestItem(**s) for s in attendee_sections(20)],
        stats=stats,
    )
    assert all(o.result and o.result.created for o in outcomes)
    assert (stats.segments, stats.sequential_items, stats.items) == (1, 0, 20)


async def test_a_failed_apply_rolls_back_and_writes_the_segment_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing of a segment is written until `apply`, so when it fails the
    segment can simply be written the old way -- same outcomes, same state."""
    from mapi.store.bulk import ReplayBulkSession

    async def broken(self: ReplayBulkSession, ops: object) -> None:
        raise RuntimeError("connection reset during apply")

    store = InMemoryStore()
    org = await store.create_organization(Organization(name="Broken"))
    expected = await _run(store, org, batched=False, name="seq")
    monkeypatch.setattr(ReplayBulkSession, "apply", broken)
    got = await _run(store, org, batched=True, name="bat")
    assert got[0] == expected[0]
    assert got[1] == expected[1]
    assert got[2].sequential_items == 19  # every valid item, one way or the other


# -- compiling operations: pure --------------------------------------------------------------


def test_compile_ops_writes_each_row_once_with_its_whole_history() -> None:
    from datetime import UTC, datetime, timedelta

    from mapi.domain.models import Chunk, Memory
    from mapi.store.bulk import EdgeOp, KeyedOp, UpsertOp

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    new = Memory(org_id="org_x", space_id="spc_x", content="alpha fact")
    new = new.model_copy(
        update={
            "chunks": [Chunk(memory_id=new.id, ordinal=0, text="alpha fact", embedding=[1.0])]
        }
    )
    stored = Memory(org_id="org_x", space_id="spc_x", content="stored fact", version=3)
    keyed_old = Memory(org_id="org_x", space_id="spc_x", content="room: 1", key="k", version=2)
    keyed_new = Memory(org_id="org_x", space_id="spc_x", content="room: 2", key="k")
    edge = RelationEdge(
        org_id="org_x",
        space_id="spc_x",
        source_id=new.id,
        target_id=stored.id,
        type=RelationType.REFERENCES,
    )
    ops = [
        UpsertOp(memory=new, when=t0, fresh=True),
        EdgeOp(edge=edge),
        # the second upsert of a created memory, same version: in place
        UpsertOp(
            memory=new.model_copy(update={"summary": "s"}), when=t0 + timedelta(1), fresh=False
        ),
        # a fold into a stored row, twice
        UpsertOp(
            memory=stored.model_copy(update={"version": 4}), when=t0 + timedelta(2), fresh=False
        ),
        UpsertOp(
            memory=stored.model_copy(update={"version": 5}), when=t0 + timedelta(3), fresh=False
        ),
        # a fold into the created row
        UpsertOp(
            memory=new.model_copy(update={"version": 2}), when=t0 + timedelta(4), fresh=False
        ),
        KeyedOp(
            memory=keyed_new,
            mode=ReplaceMode.SUPERSEDE,
            reason="replaced under the same key",
            when=t0 + timedelta(5),
            replaced=keyed_old.model_copy(
                update={"status": MemoryStatus.SUPERSEDED, "version": 3}
            ),
        ),
    ]
    plan = compile_ops(ops, dimensions=1)
    assert [r["id"] for r in plan.inserts] == [new.id, keyed_new.id]
    assert plan.inserts[0]["version"] == 2
    # The same-version upsert overwrote version 1's snapshot in place.
    first = next(v for v in plan.versions if v["memory_id"] == new.id and v["version"] == 1)
    assert first["summary"] == "s" and first["valid_from"] == t0
    assert {r["id"]: r["version"] for r in plan.updates} == {stored.id: 5, keyed_old.id: 3}
    # Stored rows: their open version closes at the FIRST change.
    assert dict(zip(plan.close_ids, plan.close_at, strict=True)) == {
        stored.id: t0 + timedelta(2),
        keyed_old.id: t0 + timedelta(5),
    }
    chains: dict[str, list[tuple[int, Any, Any]]] = {}
    for v in plan.versions:
        chains.setdefault(v["memory_id"], []).append(
            (v["version"], v["valid_from"], v["valid_to"])
        )
    assert chains[new.id] == [(1, t0, t0 + timedelta(4)), (2, t0 + timedelta(4), None)]
    assert chains[stored.id] == [
        (4, t0 + timedelta(2), t0 + timedelta(3)),
        (5, t0 + timedelta(3), None),
    ]
    assert chains[keyed_old.id] == [(3, t0 + timedelta(5), None)]
    assert [c["memory_id"] for c in plan.chunks] == [new.id]
    assert [(e["type"], e["target_id"]) for e in plan.edges] == [
        ("references", stored.id),
        ("supersedes", keyed_old.id),
    ]


# -- the point of it: round trips ---------------------------------------------------------


@pytest.mark.postgres
async def test_a_91_item_bulk_is_a_fixed_handful_of_round_trips() -> None:
    """Round trips are the portable cost: production is ~2.5 ms from its
    database, a laptop through the proxy ~35 ms, and the count is the same.
    The sequential path paid ~28 per ITEM; the batched one pays about a
    dozen per REQUEST (plus statement preparation the first time a
    connection sees each statement)."""
    store = await make_store("postgres")
    org = await store.create_organization(Organization(name="RoundTrips"))
    items = [IngestItem(**s) for s in attendee_sections(91)]
    service = _service(store, batched=True)
    space = await _space(store, org, "rt")
    await service.get_space_or_raise(org.id, space.id)
    with count_round_trips() as trips:
        outcomes = await service.ingest_many(org_id=org.id, space_id=space.id, items=items)
    assert sum(1 for o in outcomes if o.result and o.result.created) == 91
    assert trips.total <= 40, trips
    # A re-sync: every item an exact duplicate.
    with count_round_trips() as again:
        outcomes = await service.ingest_many(org_id=org.id, space_id=space.id, items=items)
    assert all(o.result and not o.result.created for o in outcomes)
    assert again.total <= 30, again

    sequential = _service(store, batched=False)
    space = await _space(store, org, "rt-seq")
    await sequential.get_space_or_raise(org.id, space.id)
    with count_round_trips() as slow:
        await sequential.ingest_many(org_id=org.id, space_id=space.id, items=items[:5])
    assert slow.total / 5 > 10 * trips.total / 91
