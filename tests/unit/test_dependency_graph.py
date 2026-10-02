"""What coexists is connected: the pieces of one function by the names they use, and
functions by the functions they use, both walkable as a dependency graph."""

from __future__ import annotations

import json

import httpx
import pytest

from mapi.config import Settings
from mapi.core.errors import NotFoundError
from mapi.domain.chunking import chunk_closure, chunk_content
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Organization, RelationType, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

DOES = "Cancels a pending order after the customer confirms which one."


def _recipe(reads: int) -> str:
    recipe = {
        "does": DOES,
        "reads": [
            {
                "id": f"r{i}",
                "tool": f"get_thing_{i}",
                "args": {"order": {"dep": f"r{i - 1}.orders[*].id"}},
                "note": "word " * 60,
            }
            for i in range(reads)
        ],
        "write": {"tool": "cancel_order", "args": {"order_id": {"step": f"r{reads - 1}"}}},
    }
    return f"{DOES}\n\n{json.dumps(recipe, sort_keys=True)}"


# -- the pieces of one function ------------------------------------------------


def test_a_piece_depends_on_the_pieces_defining_what_it_uses() -> None:
    chunks = chunk_content(_recipe(40))
    assert len(chunks) == 3
    graph = {c.ordinal: c.depends_on for c in chunks}
    # Each read uses the read before it, so each piece uses the piece before it.
    assert graph == {0: (), 1: (0,), 2: (1,)}
    # The write's piece, walked: everything it needs, dependencies first.
    assert chunk_closure(graph, 2) == [0, 1, 2]


def test_code_pieces_depend_on_the_functions_they_call() -> None:
    defs = [
        f"def helper_{i}(x):\n    return helper_{i - 1}(x) + 1  " + "# pad " * 80
        for i in range(1, 30)
    ]
    content = "Cart helpers.\n\n" + "\n\n".join(defs) + "\n\ndef helper_0(x):\n    return x"
    chunks = chunk_content(content)
    graph = {c.ordinal: c.depends_on for c in chunks}
    last = chunks[-1]
    assert "def helper_0(" in last.text
    assert last.ordinal in graph[0]  # helper_1 calls helper_0, defined at the end
    # A cycle (the first piece uses the last, which uses the one before) ends the walk.
    walked = chunk_closure(graph, 0)
    assert sorted(walked) == sorted(set(walked)) and walked[-1] == 0


def test_prose_pieces_depend_on_nothing() -> None:
    chunks = chunk_content("I moved to Madrid in May. The flat is near the park. " * 200)
    assert len(chunks) > 1 and all(c.depends_on == () for c in chunks)


def test_the_walk_is_bounded() -> None:
    chain = {i: (i - 1,) for i in range(1, 100)}
    assert len(chunk_closure(chain, 99, limit=5)) == 5


# -- functions that use functions ---------------------------------------------


async def _service() -> tuple[MemoryService, str, str]:
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=64,
        rerank_backend="heuristic",
        api_key_pepper="x" * 32,
    )
    store = InMemoryStore()
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=64), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    return service, org.id, space.id


async def _function(service: MemoryService, org: str, space: str, name: str) -> str:
    result = await service.ingest(
        org_id=org,
        space_id=space,
        content=f"{name} does its part.\n\n"
        + json.dumps({"output": name, "impl": {"tool": name}}),
        extract=False,
        dedupe=False,
    )
    return result.memory.id


async def test_a_memory_s_dependencies_are_walked_dependencies_first() -> None:
    service, org, space = await _service()
    lookup = await _function(service, org, space, "read:get_order")
    check = await _function(service, org, space, "check:get_order.status")
    pick = await _function(service, org, space, "value:cancel_order.order_id")
    recipe = await _function(service, org, space, "recipe:cancel_order")
    unrelated = await _function(service, org, space, "read:get_weather")
    for source, target in ((check, lookup), (pick, lookup), (recipe, check), (recipe, pick)):
        await service.link(
            org, space, source_id=source, target_id=target, relation=RelationType.DEPENDS_ON
        )
    # "About the same thing" is not "needs": an association is never walked.
    await service.link(
        org, space, source_id=recipe, target_id=unrelated, relation=RelationType.REFERENCES
    )

    found = await service.get_dependencies(org, space, recipe)
    assert set(found.order) == {lookup, check, pick, recipe} and unrelated not in found.order
    position = {m: i for i, m in enumerate(found.order)}
    assert position[lookup] < position[check] < position[recipe]
    assert position[lookup] < position[pick] < position[recipe]
    assert found.order[-1] == recipe
    assert found.depth == {recipe: 0, check: 1, pick: 1, lookup: 2}
    assert not found.truncated

    shallow = await service.get_dependencies(org, space, recipe, depth=1)
    assert lookup not in shallow.order
    capped = await service.get_dependencies(org, space, recipe, limit=1)
    assert capped.truncated and len(capped.order) == 2

    # Nothing depends on the lookup: its closure is itself.
    assert (await service.get_dependencies(org, space, lookup)).order == [lookup]
    with pytest.raises(NotFoundError):
        await service.get_dependencies(org, space, "mem_00000000000000000000000000")


async def test_a_dependency_cycle_ends_the_walk() -> None:
    service, org, space = await _service()
    a = await _function(service, org, space, "read:a")
    b = await _function(service, org, space, "read:b")
    await service.link(org, space, source_id=a, target_id=b, relation=RelationType.DEPENDS_ON)
    await service.link(org, space, source_id=b, target_id=a, relation=RelationType.DERIVED_FROM)
    found = await service.get_dependencies(org, space, a)
    assert found.order == [b, a]


# -- through the API ------------------------------------------------------------


async def test_the_api_walks_dependencies_and_search_brings_them(
    client: httpx.AsyncClient, space_id: str
) -> None:
    async def write(content: str) -> str:
        made = await client.post(f"/v1/spaces/{space_id}/memories", json={"content": content})
        assert made.status_code in (200, 201), made.text
        return str(made.json()["memory"]["id"])

    lookup = await write("Looks up the order.\n\n" + json.dumps({"output": "read:get_order"}))
    recipe = await write(_recipe(40))
    linked = await client.post(
        f"/v1/spaces/{space_id}/memories/{recipe}/relations",
        json={"target_id": lookup, "relation": "depends_on"},
    )
    assert linked.status_code == 201, linked.text

    walked = await client.get(f"/v1/spaces/{space_id}/memories/{recipe}/dependencies")
    assert walked.status_code == 200, walked.text
    body = walked.json()
    assert body["order"] == [lookup, recipe]
    pieces = next(n for n in body["nodes"] if n["memory"]["id"] == recipe)["chunks"]
    assert [p["depends_on"] for p in pieces] == [[], [0], [1]]

    found = await client.post(
        f"/v1/spaces/{space_id}/search",
        json={"query": "cancel_order write order_id", "with_dependencies": True, "limit": 5},
    )
    assert found.status_code == 200, found.text
    hit = next(h for h in found.json()["results"] if h["memory"]["id"] == recipe)
    assert [d["id"] for d in hit["dependencies"]] == [lookup]
    assert hit["chunk_dependencies"] is not None
    plain = await client.post(
        f"/v1/spaces/{space_id}/search", json={"query": "cancel_order write order_id"}
    )
    assert all(h["dependencies"] is None for h in plain.json()["results"])
