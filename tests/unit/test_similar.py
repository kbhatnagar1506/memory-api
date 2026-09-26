"""POST /v1/spaces/{target}/similar: neighbours of a stored memory (B9).

The matcher's inner loop is "which directory cards sit closest to this card".
The card is already embedded -- it is a stored memory -- so searching with its
stored vector costs no provider call, which is the reason this endpoint exists.

Two spaces are involved, so there are two tenant checks, and the one that is
easy to forget is the SOURCE: a caller naming a memory in another org's space
must get the same 404 as for a memory that does not exist, never a list of
neighbours computed from someone else's vector.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from mapi.domain.models import Memory, Organization, Space


async def _add(client: httpx.AsyncClient, space_id: str, content: str, **metadata: Any) -> str:
    response = await client.post(
        f"/v1/spaces/{space_id}/memories", json={"content": content, "metadata": metadata}
    )
    assert response.status_code in (200, 201), response.text
    return str(response.json()["memory"]["id"])


@pytest.fixture
async def world(client: httpx.AsyncClient, space_id: str) -> dict[str, str]:
    directory = (
        await client.post("/v1/spaces", json={"slug": "dir", "name": "Directory"})
    ).json()["id"]
    ids = {
        "own": space_id,
        "directory": directory,
        "mine": await _add(
            client, space_id, "stuck on rust lifetimes in an async web server", user_id="me"
        ),
        "me_dir": await _add(
            client,
            directory,
            "stuck on rust lifetimes in an async web server",
            user_id="me",
            doc_id="me",
        ),
        "ana1": await _add(
            client,
            directory,
            "rust async web server lifetimes expert",
            user_id="ana",
            doc_id="ana",
        ),
        "ana2": await _add(
            client,
            directory,
            "rust lifetimes and async borrow help",
            user_id="ana",
            doc_id="ana",
        ),
        "bo": await _add(
            client, directory, "figma prototypes and design critique", doc_id="bo"
        ),
    }
    return ids


class _NoEmbedding:
    """Fails the test if the provider is asked for anything."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        embedder._embed_batch = self

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError(f"/similar must not embed; was asked for {texts!r}")


async def test_neighbours_come_from_the_stored_vector(app_context, client, world) -> None:
    _NoEmbedding(app_context["app"].state.service.pipeline.embedder)
    response = await client.post(
        f"/v1/spaces/{world['directory']}/similar",
        json={"source_space_id": world["own"], "source_memory_id": world["mine"], "limit": 5},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["source_memory_id"] == world["mine"]
    ids = [hit["memory"]["id"] for hit in data["results"]]
    assert ids[0] == world["me_dir"]  # identical text, cosine 1
    assert world["bo"] == ids[-1]
    scores = [hit["score"] for hit in data["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all(hit["vector_score"] == pytest.approx(hit["score"]) for hit in data["results"])
    assert {"source_ms", "candidates_ms", "total_ms", "auth_ms"} <= set(data["timings_ms"])


async def test_the_caller_can_exclude_themselves_and_dedupe_people(client, world) -> None:
    response = await client.post(
        f"/v1/spaces/{world['directory']}/similar",
        json={
            "source_space_id": world["own"],
            "source_memory_id": world["mine"],
            "exclude_metadata": {"user_id": "me"},
            "max_per_source": 1,
        },
    )
    people = [hit["memory"]["metadata"]["doc_id"] for hit in response.json()["results"]]
    assert "me" not in people
    assert people == ["ana", "bo"]


async def test_a_source_in_its_own_space_is_not_its_own_neighbour(client, world) -> None:
    response = await client.post(
        f"/v1/spaces/{world['directory']}/similar",
        json={"source_space_id": world["directory"], "source_memory_id": world["ana1"]},
    )
    ids = [hit["memory"]["id"] for hit in response.json()["results"]]
    assert world["ana1"] not in ids
    assert ids  # the rest are still there


async def test_min_score_is_a_cosine_floor(client, world) -> None:
    everything = (
        await client.post(
            f"/v1/spaces/{world['directory']}/similar",
            json={"source_space_id": world["own"], "source_memory_id": world["mine"]},
        )
    ).json()["results"]
    floor = everything[1]["score"]
    kept = (
        await client.post(
            f"/v1/spaces/{world['directory']}/similar",
            json={
                "source_space_id": world["own"],
                "source_memory_id": world["mine"],
                "min_score": floor,
            },
        )
    ).json()["results"]
    assert kept and all(hit["score"] >= floor for hit in kept)
    assert len(kept) < len(everything)


async def test_filters_and_limit_apply(client, world) -> None:
    response = await client.post(
        f"/v1/spaces/{world['directory']}/similar",
        json={
            "source_space_id": world["own"],
            "source_memory_id": world["mine"],
            "metadata": {"doc_id": "ana"},
            "limit": 1,
            "include_content": False,
            "snippet_chars": 8,
        },
    )
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["memory"]["metadata"]["doc_id"] == "ana"
    assert results[0]["memory"]["content"] == ""
    assert len(results[0]["matched_text"]) <= 8


# -- tenancy ------------------------------------------------------------------------


@pytest.fixture
async def foreign(app_context) -> dict[str, str]:
    """Another org, with a space and a memory in it."""
    store = app_context["store"]
    embedder = app_context["app"].state.service.pipeline.embedder
    other = await store.create_organization(Organization(name="Elsewhere"))
    space = await store.create_space(Space(org_id=other.id, slug="theirs", name="Theirs"))
    from tests.support.factories import memory as build_memory

    stored = build_memory(
        org_id=other.id,
        space_id=space.id,
        content="their private card about rust lifetimes",
        vector=await embedder.embed_one("their private card about rust lifetimes"),
    )
    await store.upsert_memory(stored)
    return {"org": other.id, "space": space.id, "memory": stored.id}


async def test_a_source_in_another_org_is_a_404(client, world, foreign) -> None:
    response = await client.post(
        f"/v1/spaces/{world['directory']}/similar",
        json={"source_space_id": foreign["space"], "source_memory_id": foreign["memory"]},
    )
    assert response.status_code == 404


async def test_a_target_in_another_org_is_a_404(client, world, foreign) -> None:
    response = await client.post(
        f"/v1/spaces/{foreign['space']}/similar",
        json={"source_space_id": world["own"], "source_memory_id": world["mine"]},
    )
    assert response.status_code == 404


async def test_a_foreign_memory_named_under_an_own_space_is_a_404(
    client, world, foreign
) -> None:
    """The memory id is only looked up inside the named space."""
    response = await client.post(
        f"/v1/spaces/{world['directory']}/similar",
        json={"source_space_id": world["own"], "source_memory_id": foreign["memory"]},
    )
    assert response.status_code == 404


async def test_a_missing_memory_is_a_404(client, world) -> None:
    ghost = Memory(org_id="org_x", space_id=world["own"], content="never stored").id
    response = await client.post(
        f"/v1/spaces/{world['directory']}/similar",
        json={"source_space_id": world["own"], "source_memory_id": ghost},
    )
    assert response.status_code == 404


# -- the request shape ----------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [{}, {"source_memory_id": "mem_x", "source_key": "card:stuck_on"}],
    ids=["neither", "both"],
)
async def test_exactly_one_source(client, world, source: dict[str, str]) -> None:
    body = {"source_space_id": world["own"], **source}
    response = await client.post(f"/v1/spaces/{world['directory']}/similar", json=body)
    assert response.status_code == 422


async def test_a_malformed_memory_id_is_a_422(client, world) -> None:
    body = {"source_space_id": world["own"], "source_memory_id": "not-an-id"}
    response = await client.post(f"/v1/spaces/{world['directory']}/similar", json=body)
    assert response.status_code == 422


async def test_source_key_says_what_is_missing(app_context, client, world) -> None:
    """Until keyed memories land, a key is a clear 422 naming the alternative."""
    if hasattr(app_context["store"], "get_active_by_keys"):
        pytest.skip("this store has keyed memories; the 422 path is unreachable")
    body = {"source_space_id": world["own"], "source_key": "card:stuck_on"}
    response = await client.post(f"/v1/spaces/{world['directory']}/similar", json=body)
    assert response.status_code == 422
    assert "source_memory_id" in response.text


async def test_source_key_resolves_through_the_store_hook(app_context, client, world) -> None:
    """With a backend that has keys, the key finds the memory and the rest is the same.

    `get_active_by_keys` is the write-path workstream's lookup (migration
    0006); this stands one in so the route is proven before that lands.
    """
    store = app_context["store"]

    async def get_active_by_keys(
        org_id: str, space_id: str, keys: list[str]
    ) -> dict[str, Memory]:
        assert keys == ["card:stuck_on"]
        assert space_id == world["own"]
        memory = await store.get_memory(org_id, space_id, world["mine"])
        return {keys[0]: memory} if memory is not None else {}

    store.get_active_by_keys = get_active_by_keys
    body = {"source_space_id": world["own"], "source_key": "card:stuck_on"}
    response = await client.post(f"/v1/spaces/{world['directory']}/similar", json=body)
    assert response.status_code == 200
    assert response.json()["source_memory_id"] == world["mine"]

    async def nothing(org_id: str, space_id: str, keys: list[str]) -> dict[str, Memory]:
        return {}

    store.get_active_by_keys = nothing
    response = await client.post(f"/v1/spaces/{world['directory']}/similar", json=body)
    assert response.status_code == 404
