"""Batched supersession lookup: one call per search, same answer as the walk (B7).

Supersession suppression asked "what transitively supersedes this?" once per
pooled candidate, and on Postgres each ask was a breadth-first walk issuing a
session per hop. With a 32-memory pool that was the largest DB cost in a search
and the one stage `timings_ms` did not show.

It is now one question for the whole pool -- `reachable_superseders_many`, a
recursive CTE on Postgres. A rewrite of graph code is only as good as its
agreement with what it replaced, so the core of this file is a property test:
random graphs, including cyclic ones and depth-bounded walks, answered by the
original breadth-first walk (kept here verbatim as the oracle) and by every
backend, compared node for node.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from typing import Any

import pytest
from sqlalchemy import insert
from tests.support.pg import shared_postgres_store

from mapi.domain.models import (
    Memory,
    MemoryStatus,
    Organization,
    RelationEdge,
    RelationType,
    Space,
)
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.store.memory import InMemoryStore

Graph = list[tuple[int, int]]  # (newer, older): newer SUPERSEDES older


def oracle(edges: Graph, node: int, max_depth: int) -> set[int]:
    """The breadth-first walk `reachable_superseders` shipped with, verbatim in shape."""
    incoming: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        incoming[target].append(source)
    seen: set[int] = set()
    frontier = [node]
    for _ in range(max_depth):
        if not frontier:
            break
        nxt: list[int] = []
        for current in frontier:
            for source in incoming[current]:
                if source not in seen and source != node:
                    seen.add(source)
                    nxt.append(source)
        frontier = nxt
    return seen


def random_dag(rng: random.Random, n: int, density: float) -> Graph:
    """Edges only from higher to lower index: acyclic by construction."""
    return [(j, i) for j in range(n) for i in range(j) if rng.random() < density]


def random_graph_with_cycles(rng: random.Random, n: int, density: float) -> Graph:
    """Any direction, no self-loops (the table forbids those)."""
    return [(a, b) for a in range(n) for b in range(n) if a != b and rng.random() < density]


def chain(n: int) -> Graph:
    """0 <- 1 <- 2 <- ... : a long revision history, for the depth bound."""
    return [(i + 1, i) for i in range(n - 1)]


CASES: list[tuple[str, Graph, int, int]] = []
_rng = random.Random(20260926)
for _i in range(12):
    _n = _rng.randint(2, 12)
    CASES.append((f"dag{_i}", random_dag(_rng, _n, _rng.uniform(0.1, 0.6)), _n, 20))
for _i in range(4):
    _n = _rng.randint(3, 8)
    CASES.append(
        (f"cyclic{_i}", random_graph_with_cycles(_rng, _n, _rng.uniform(0.15, 0.4)), _n, 20)
    )
CASES.append(("chain25-depth20", chain(25), 25, 20))
CASES.append(("chain8-depth3", chain(8), 8, 3))
CASES.append(("diamond-depth1", [(1, 0), (2, 0), (3, 1), (3, 2)], 4, 1))
CASES.append(("empty", [], 5, 20))


async def _load(store: Any, edges: Graph, n: int) -> tuple[str, str, list[str]]:
    """Materialize a graph as memories and SUPERSEDES edges in a fresh tenant."""
    org = await store.create_organization(Organization(name="CTE"))
    space = await store.create_space(Space(org_id=org.id, slug=f"c{org.id[-10:]}", name="G"))
    memories = [Memory(org_id=org.id, space_id=space.id, content=f"node {i}") for i in range(n)]
    edge_models = [
        RelationEdge(
            org_id=org.id,
            space_id=space.id,
            source_id=memories[s].id,
            target_id=memories[t].id,
            type=RelationType.SUPERSEDES,
        )
        for s, t in edges
    ]
    if isinstance(store, InMemoryStore):
        for memory in memories:
            await store.upsert_memory(memory)
        for edge in edge_models:
            await store.create_relation(edge)
    else:
        # Two statements rather than a session per row: the lane runs over a
        # WAN link, and the property is about the read, not about the write.
        from mapi.store.postgres.models import MemoryRow, RelationEdgeRow

        async with store._session() as session, session.begin():
            await store._scope(session, org.id)
            await session.execute(
                insert(MemoryRow),
                [
                    {
                        "id": m.id,
                        "org_id": m.org_id,
                        "space_id": m.space_id,
                        "content": m.content,
                        "summary": "",
                        "kind": m.kind.value,
                        "meta": {},
                        "tags": [],
                        "source": "",
                        "status": m.status.value,
                        "occurred_at": m.occurred_at,
                        "created_at": m.created_at,
                        "updated_at": m.updated_at,
                        "content_sha256": m.content_sha256,
                        "version": 1,
                    }
                    for m in memories
                ],
            )
            if edge_models:
                await session.execute(
                    insert(RelationEdgeRow),
                    [
                        {
                            "id": e.id,
                            "org_id": e.org_id,
                            "space_id": e.space_id,
                            "source_id": e.source_id,
                            "target_id": e.target_id,
                            "type": e.type.value,
                            "reason": "",
                            "confidence": 1.0,
                            "created_at": e.created_at,
                        }
                        for e in edge_models
                    ],
                )
    return org.id, space.id, [m.id for m in memories]


_BACKENDS = [
    pytest.param("memory", id="memory"),
    pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
]


@pytest.fixture(params=_BACKENDS)
async def backend(request: pytest.FixtureRequest) -> Any:
    if request.param == "memory":
        return InMemoryStore()
    return await shared_postgres_store()


@pytest.mark.parametrize(("label", "edges", "n", "depth"), CASES, ids=[c[0] for c in CASES])
async def test_the_batched_lookup_agrees_with_the_walk(
    backend: Any, label: str, edges: Graph, n: int, depth: int
) -> None:
    org_id, space_id, ids = await _load(backend, edges, n)
    got = await backend.reachable_superseders_many(org_id, space_id, ids, max_depth=depth)
    assert set(got) == set(ids), "every requested id must be a key"
    for i, memory_id in enumerate(ids):
        expected = {ids[j] for j in oracle(edges, i, depth)}
        assert got[memory_id] == expected, f"{label}: node {i}"


async def test_the_single_lookup_still_agrees(backend: Any) -> None:
    """`reachable_superseders` now delegates to the batch on Postgres."""
    edges = [(1, 0), (2, 1), (3, 0)]
    org_id, space_id, ids = await _load(backend, edges, 4)
    assert await backend.reachable_superseders(org_id, space_id, ids[0]) == {
        ids[1],
        ids[2],
        ids[3],
    }


async def test_unknown_and_duplicate_ids_are_harmless(backend: Any) -> None:
    org_id, space_id, ids = await _load(backend, [(1, 0)], 2)
    ghost = Memory(org_id=org_id, space_id=space_id, content="never stored").id
    got = await backend.reachable_superseders_many(
        org_id, space_id, [ids[0], ids[0], ghost], max_depth=20
    )
    assert got == {ids[0]: {ids[1]}, ghost: set()}
    assert await backend.reachable_superseders_many(org_id, space_id, []) == {}


async def test_another_orgs_edges_are_invisible(backend: Any) -> None:
    _, space_id, ids = await _load(backend, [(1, 0)], 2)
    stranger = await backend.create_organization(Organization(name="Stranger"))
    got = await backend.reachable_superseders_many(stranger.id, space_id, ids)
    assert got == {ids[0]: set(), ids[1]: set()}


async def test_depth_must_be_positive(backend: Any) -> None:
    org_id, space_id, ids = await _load(backend, [], 1)
    with pytest.raises(ValueError):
        await backend.reachable_superseders_many(org_id, space_id, ids, max_depth=0)


# -- the pipeline asks once ------------------------------------------------------


async def test_a_search_asks_about_the_whole_pool_in_one_call(service, org, space) -> None:
    store = service.store
    for i in range(12):
        await service.ingest(org_id=org.id, space_id=space.id, content=f"kubernetes note {i}")

    calls: list[int] = []
    direct: list[str] = []
    inside = False
    many = store.reachable_superseders_many
    single = store.reachable_superseders

    async def counting_many(*args: Any, **kwargs: Any) -> Any:
        nonlocal inside
        calls.append(len(args[2]))
        inside = True
        try:
            return await many(*args, **kwargs)
        finally:
            inside = False

    async def watched_single(*args: Any, **kwargs: Any) -> Any:
        # The in-memory default answers the batch by walking each id, which
        # is fine; what must not happen is the PIPELINE walking per candidate.
        if not inside:
            direct.append(args[2])
        return await single(*args, **kwargs)

    store.reachable_superseders_many = counting_many
    store.reachable_superseders = watched_single
    response = await service.search(
        SearchRequest(query="kubernetes", org_id=org.id, space_id=space.id, limit=10)
    )
    assert len(calls) == 1
    assert direct == []
    assert calls[0] >= len(response.results) > 0
    assert "superseders_ms" in response.timings_ms
    assert "total_ms" in response.timings_ms


async def test_suppression_still_hides_a_transitively_replaced_fact(
    service, org, space
) -> None:
    """A->B->C with only A and C matching: C must still be hidden."""
    store = service.store
    a = await service.ingest(org_id=org.id, space_id=space.id, content="office city is seattle")
    b = await service.ingest(org_id=org.id, space_id=space.id, content="relocated, see notes")
    c = await service.ingest(
        org_id=org.id, space_id=space.id, content="office city is portland"
    )
    for newer, older in ((a.memory, b.memory), (b.memory, c.memory)):
        await store.create_relation(
            RelationEdge(
                org_id=org.id,
                space_id=space.id,
                source_id=newer.id,
                target_id=older.id,
                type=RelationType.SUPERSEDES,
            )
        )
    response = await service.search(
        SearchRequest(query="office city", org_id=org.id, space_id=space.id, limit=10)
    )
    ids = [hit.memory.id for hit in response.results]
    assert a.memory.id in ids
    assert c.memory.id not in ids


async def test_superseded_status_is_dropped_without_asking(service, org, space) -> None:
    store = service.store
    result = await service.ingest(org_id=org.id, space_id=space.id, content="old fact")
    await store.upsert_memory(
        result.memory.model_copy(update={"status": MemoryStatus.SUPERSEDED, "version": 2})
    )
    response = await service.search(
        SearchRequest(
            query="old fact",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            include_superseded=False,
        )
    )
    assert response.results == []


# -- the lookups ride alongside the hydrate ----------------------------------------


class _InFlight:
    """Wraps store methods with a delay, recording how many ran at once."""

    def __init__(self, store: Any, names: tuple[str, ...], delay: float = 0.05) -> None:
        self.now = 0
        self.peak = 0
        self.calls: list[str] = []
        for name in names:
            setattr(store, name, self._wrap(name, getattr(store, name), delay))

    def _wrap(self, name: str, original: Any, delay: float) -> Any:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            self.now += 1
            self.peak = max(self.peak, self.now)
            try:
                await asyncio.sleep(delay)
                return await original(*args, **kwargs)
            finally:
                self.now -= 1

        return wrapped


_WAVE = ("get_memories", "reachable_superseders_many", "get_relations_between")


async def test_the_relation_lookups_overlap_the_hydrate(service, org, space) -> None:
    """Three sessions in one wave, not three waves: the critical path is one."""
    for i in range(6):
        await service.ingest(org_id=org.id, space_id=space.id, content=f"graphql note {i}")
    watch = _InFlight(service.store, _WAVE)
    response = await service.search(
        SearchRequest(query="graphql note", org_id=org.id, space_id=space.id, limit=5)
    )
    assert response.results
    assert sorted(watch.calls) == sorted(_WAVE)
    assert watch.peak == 3
    timings = response.timings_ms
    for stage in ("hydrate_ms", "superseders_ms", "conflicts_ms"):
        assert timings[stage] >= 45, (stage, timings)
    # Overlapped: the search took about one delay, not the sum of three.
    assert timings["total_ms"] < 140, timings


async def test_include_superseded_skips_the_supersession_lookup(service, org, space) -> None:
    await service.ingest(org_id=org.id, space_id=space.id, content="graphql one")
    await service.ingest(org_id=org.id, space_id=space.id, content="graphql two")
    watch = _InFlight(service.store, _WAVE, delay=0)
    response = await service.search(
        SearchRequest(
            query="graphql",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            include_superseded=True,
        )
    )
    assert response.results
    assert "reachable_superseders_many" not in watch.calls
    assert "superseders_ms" not in response.timings_ms


async def test_a_failed_hydrate_raises_its_own_error_and_stops_the_lookups(
    service, org, space
) -> None:
    """The store's exception, unwrapped, and no lookup left holding a connection."""
    from mapi.core.errors import StoreError

    for i in range(3):
        await service.ingest(org_id=org.id, space_id=space.id, content=f"graphql fact {i}")
    store = service.store
    cancelled: list[str] = []

    async def broken(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.01)
        raise StoreError("hydrate failed")

    def slow(name: str) -> Any:
        async def wait(*args: Any, **kwargs: Any) -> Any:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        return wait

    store.get_memories = broken
    store.reachable_superseders_many = slow("superseders")
    store.get_relations_between = slow("contradictions")
    with pytest.raises(StoreError, match="hydrate failed"):
        await service.search(
            SearchRequest(query="graphql fact", org_id=org.id, space_id=space.id, limit=5)
        )
    assert sorted(cancelled) == ["contradictions", "superseders"]


async def test_conflicts_are_only_reported_between_results(service, org, space) -> None:
    """Read over the pool, reported over the results: a pool-only side hides the pair."""
    store = service.store
    a = await service.ingest(org_id=org.id, space_id=space.id, content="standup is at nine")
    b = await service.ingest(org_id=org.id, space_id=space.id, content="standup is at ten")
    for source, target in ((a.memory.id, b.memory.id), (b.memory.id, a.memory.id)):
        await store.create_relation(
            RelationEdge(
                org_id=org.id,
                space_id=space.id,
                source_id=source,
                target_id=target,
                type=RelationType.CONTRADICTS,
            )
        )
    both = await service.search(
        SearchRequest(query="standup", org_id=org.id, space_id=space.id, limit=5)
    )
    assert {h.memory.id for h in both.results} == {a.memory.id, b.memory.id}
    assert both.conflicts == [tuple(sorted((a.memory.id, b.memory.id)))]

    one = await service.search(
        SearchRequest(query="standup", org_id=org.id, space_id=space.id, limit=1)
    )
    assert len(one.results) == 1
    assert one.conflicts == []
