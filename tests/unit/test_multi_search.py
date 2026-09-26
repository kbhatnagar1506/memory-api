"""POST /v1/multi-search: one question, several spaces, one embedding (B9).

The facemash question path asks a person's own memory and, when the question is
about people, the event directory -- and "who should I meet about X" asks two
facets of that directory. As separate searches that is two or three embeddings
of the same sentence. Multi-search embeds it once and hands the vector to every
target, which is the whole point, so it is the first thing tested.

Isolation is the second: every space must be checked against the caller's org
before anything is spent, and one foreign id must fail the whole call exactly
as a single search would -- a per-target error would let a caller probe which
ids exist in other tenants.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from mapi.core.errors import ProviderError
from mapi.core.security import build_api_key
from mapi.domain.models import Organization, Scope, Space


class _EmbedCounter:
    """Counts query-side embedding attempts at the pipeline boundary."""

    def __init__(self, pipeline: Any) -> None:
        self.calls = 0
        self._original = pipeline._embed_query
        pipeline._embed_query = self

    async def __call__(self, query: str, *, expand: bool = False) -> Any:
        self.calls += 1
        return await self._original(query, expand=expand)


async def _seed(client: httpx.AsyncClient, space_id: str, *items: tuple[str, dict[str, Any]]):
    for content, metadata in items:
        response = await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": content, "metadata": metadata, "tags": metadata.get("tags", [])},
        )
        assert response.status_code in (200, 201), response.text


@pytest.fixture
async def spaces(client: httpx.AsyncClient, space_id: str) -> dict[str, str]:
    directory = (
        await client.post("/v1/spaces", json={"slug": "directory", "name": "Directory"})
    ).json()["id"]
    await _seed(
        client,
        space_id,
        ("I am stuck on rust borrow checker lifetimes", {}),
        ("dinner plans with the team on friday", {}),
    )
    await _seed(
        client,
        directory,
        ("can help with rust lifetimes and async", {"user_id": "u_me", "doc_id": "u_me"}),
        ("can help with rust embedded and probes", {"user_id": "u_ana", "doc_id": "u_ana"}),
        ("rust macros, also rust tooling talks", {"user_id": "u_ana", "doc_id": "u_ana"}),
        ("can help with figma and design reviews", {"user_id": "u_bo", "doc_id": "u_bo"}),
    )
    return {"own": space_id, "directory": directory}


async def test_three_targets_cost_one_embedding(app_context, client, spaces) -> None:
    counter = _EmbedCounter(app_context["app"].state.service.pipeline)
    body = {
        "query": "rust lifetimes",
        "targets": [
            {"space_id": spaces["own"], "limit": 5},
            {"space_id": spaces["directory"], "exclude_metadata": {"user_id": "u_me"}},
            {"space_id": spaces["directory"], "max_per_source": 1},
        ],
    }
    response = await client.post("/v1/multi-search", json=body)
    assert response.status_code == 200, response.text
    assert counter.calls == 1
    data = response.json()
    assert [t["space_id"] for t in data["targets"]] == [
        spaces["own"],
        spaces["directory"],
        spaces["directory"],
    ]
    assert data["degraded"] is False
    assert all("vector" in t["strategies"] for t in data["targets"])
    assert {"embed_ms", "total_ms", "space_ms", "auth_ms"} <= set(data["timings_ms"])


async def test_each_target_matches_its_own_single_search(client, spaces) -> None:
    """Sharing the vector must not change any target's answer."""
    target = {
        "space_id": spaces["directory"],
        "limit": 3,
        "exclude_metadata": {"user_id": "u_me"},
    }
    multi = await client.post(
        "/v1/multi-search", json={"query": "rust lifetimes", "targets": [target]}
    )
    single = await client.post(
        f"/v1/spaces/{spaces['directory']}/search",
        json={"query": "rust lifetimes", "limit": 3, "exclude_metadata": {"user_id": "u_me"}},
    )
    multi_ids = [h["memory"]["id"] for h in multi.json()["targets"][0]["results"]]
    single_ids = [h["memory"]["id"] for h in single.json()["results"]]
    assert multi_ids == single_ids
    assert multi_ids  # and the comparison is not vacuous


async def test_per_target_options_apply_per_target(client, spaces) -> None:
    body = {
        "query": "rust",
        "targets": [
            {"space_id": spaces["directory"], "exclude_metadata": {"user_id": "u_me"}},
            {"space_id": spaces["directory"], "max_per_source": 1},
            {"space_id": spaces["own"], "include_content": False, "snippet_chars": 10},
        ],
    }
    data = (await client.post("/v1/multi-search", json=body)).json()
    excluded, capped, own = data["targets"]
    assert all(h["memory"]["metadata"].get("user_id") != "u_me" for h in excluded["results"])
    people = [h["memory"]["metadata"]["doc_id"] for h in capped["results"]]
    assert len(people) == len(set(people))
    assert own["results"]
    assert all(h["memory"]["content"] == "" for h in own["results"])
    assert all(len(h["matched_text"]) <= 10 for h in own["results"])


async def test_a_foreign_space_fails_the_whole_call_before_embedding(
    app_context, client, spaces
) -> None:
    store = app_context["store"]
    other = await store.create_organization(Organization(name="Other tenant"))
    foreign = await store.create_space(Space(org_id=other.id, slug="theirs", name="Theirs"))
    counter = _EmbedCounter(app_context["app"].state.service.pipeline)
    body = {
        "query": "rust",
        "targets": [{"space_id": spaces["own"]}, {"space_id": foreign.id}],
    }
    response = await client.post("/v1/multi-search", json=body)
    assert response.status_code == 404
    assert counter.calls == 0


async def test_a_missing_space_is_the_same_404(client, spaces) -> None:
    ghost = Space(org_id="org_01h0000000000000000000000", slug="x", name="x").id
    response = await client.post(
        "/v1/multi-search",
        json={"query": "rust", "targets": [{"space_id": spaces["own"]}, {"space_id": ghost}]},
    )
    assert response.status_code == 404


async def test_at_least_one_target(client) -> None:
    response = await client.post("/v1/multi-search", json={"query": "q", "targets": []})
    assert response.status_code == 422


async def test_at_most_four_targets(client, spaces) -> None:
    body = {"query": "q", "targets": [{"space_id": spaces["own"]}] * 5}
    assert (await client.post("/v1/multi-search", json=body)).status_code == 422


async def test_a_malformed_space_id_names_the_target(client) -> None:
    response = await client.post(
        "/v1/multi-search", json={"query": "q", "targets": [{"space_id": "nope"}]}
    )
    assert response.status_code == 422
    assert "targets[0].space_id" in response.text


async def test_the_expensive_stages_are_not_on_offer(client, spaces) -> None:
    """HyDE would re-embed per target; the body refuses it rather than ignoring it."""
    for field in ("use_expansion", "use_rerank", "use_mmr", "use_entity_expansion"):
        body = {"query": "q", "targets": [{"space_id": spaces["own"], field: True}]}
        assert (await client.post("/v1/multi-search", json=body)).status_code == 422, field


async def test_a_failed_embedding_degrades_every_target_once(
    app_context, client, spaces
) -> None:
    """Lexical-only everywhere, flagged, and the provider tried ONCE -- not per target."""
    pipeline = app_context["app"].state.service.pipeline
    embedder = pipeline.embedder
    attempts = 0

    async def failing(texts: list[str]) -> list[list[float]]:
        nonlocal attempts
        attempts += 1
        raise ProviderError("quota exhausted")

    embedder._embed_batch = failing
    embedder.max_attempts = 1
    embedder._cache.clear()
    body = {
        "query": "rust lifetimes borrow",
        "targets": [{"space_id": spaces["own"]}, {"space_id": spaces["directory"]}],
    }
    response = await client.post("/v1/multi-search", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["degraded"] is True
    assert attempts == 1
    assert all("vector" not in t["strategies"] for t in data["targets"])


async def test_the_search_scope_is_required(app_context, spaces) -> None:
    record, plaintext = build_api_key(
        org_id=app_context["org"].id,
        name="writer",
        pepper="test-pepper",
        scopes=frozenset({Scope.MEMORIES_WRITE}),
    )
    await app_context["store"].create_api_key(record)
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as other:
        response = await other.post(
            "/v1/multi-search",
            json={"query": "q", "targets": [{"space_id": spaces["own"]}]},
            headers={"Authorization": f"Bearer {plaintext}"},
        )
    assert response.status_code == 403


async def test_query_vector_cannot_be_sent(client, spaces) -> None:
    body = {
        "query": "q",
        "query_vector": [0.1] * 128,
        "targets": [{"space_id": spaces["own"]}],
    }
    assert (await client.post("/v1/multi-search", json=body)).status_code == 422
