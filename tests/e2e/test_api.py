"""End-to-end API behaviour through the real ASGI app.

These exercise the whole stack — middleware, auth, validation, service, store —
with no mocks anywhere. The only substitutions are the storage and embedding
backends, and both of those are real implementations.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from supermemory.core.security import build_api_key
from supermemory.domain.models import Organization, Scope, Space

# -- operations ----------------------------------------------------------------


async def test_health_is_public(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readiness_reports_store_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["store"] is True


async def test_metrics_are_exposed(client: httpx.AsyncClient) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "supermemory_http_requests_total" in response.text


async def test_openapi_document_is_valid(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    assert schema["info"]["title"] == "Supermemory"
    assert "/v1/spaces/{space_id}/search" in schema["paths"]


async def test_every_response_carries_a_request_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.headers.get("x-request-id")


# -- authentication ------------------------------------------------------------


async def test_missing_credentials_are_rejected(app_context) -> None:
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        response = await anon.get("/v1/spaces")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate")


@pytest.mark.parametrize(
    "header",
    ["Bearer sm_totallywrongkeyvaluehere", "Bearer garbage", "Bearer ", "notascheme"],
)
async def test_invalid_credentials_are_rejected(app_context, header: str) -> None:
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        response = await anon.get("/v1/spaces", headers={"Authorization": header})
    assert response.status_code == 401


async def test_x_api_key_header_is_accepted(app_context) -> None:
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        response = await anon.get("/v1/spaces", headers={"X-API-Key": app_context["key"]})
    assert response.status_code == 200


async def test_revoked_key_stops_working(app_context, client) -> None:
    keys = (await client.get("/v1/keys")).json()["items"]
    assert keys
    # Mint a second key, revoke it, confirm it is refused.
    created = (
        await client.post("/v1/keys", json={"name": "temp", "scopes": ["search"]})
    ).json()
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as other:
        headers = {"Authorization": f"Bearer {created['plaintext']}"}
        assert (await other.get("/v1/spaces", headers=headers)).status_code == 403
        await client.delete(f"/v1/keys/{created['key']['id']}")
        assert (await other.get("/v1/spaces", headers=headers)).status_code == 401


async def test_scope_enforcement(app_context) -> None:
    """A search-only key must not be able to write."""
    store = app_context["store"]
    record, plaintext = build_api_key(
        org_id=app_context["org"].id,
        name="readonly",
        pepper="test-pepper",
        scopes=frozenset({Scope.SEARCH}),
    )
    await store.create_api_key(record)
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://t",
        headers={"Authorization": f"Bearer {plaintext}"},
    ) as limited:
        space_id = app_context["space"].id
        write = await limited.post(f"/v1/spaces/{space_id}/memories", json={"content": "nope"})
        assert write.status_code == 403
        assert write.json()["required_scope"] == "memories:write"


# -- tenancy -------------------------------------------------------------------


async def test_one_org_cannot_read_anothers_memories(app_context, client, space_id) -> None:
    await client.post(f"/v1/spaces/{space_id}/memories", json={"content": "tenant one"})

    store = app_context["store"]
    intruder = await store.create_organization(Organization(name="Intruder"))
    await store.create_space(Space(org_id=intruder.id, slug="x", name="X"))
    record, plaintext = build_api_key(
        org_id=intruder.id,
        name="k",
        pepper="test-pepper",
        scopes=frozenset(Scope.all()),
    )
    await store.create_api_key(record)

    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://t",
        headers={"Authorization": f"Bearer {plaintext}"},
    ) as other:
        # The victim's space must be indistinguishable from one that never existed.
        assert (await other.get(f"/v1/spaces/{space_id}")).status_code == 404
        listing = await other.get(f"/v1/spaces/{space_id}/memories")
        assert listing.status_code == 404
        search = await other.post(f"/v1/spaces/{space_id}/search", json={"query": "tenant"})
        assert search.status_code == 404


# -- spaces --------------------------------------------------------------------


async def test_space_lifecycle(client: httpx.AsyncClient) -> None:
    created = await client.post("/v1/spaces", json={"slug": "notes", "name": "Notes"})
    assert created.status_code == 201
    space_id = created.json()["id"]

    assert (await client.get(f"/v1/spaces/{space_id}")).status_code == 200
    listing = await client.get("/v1/spaces")
    assert any(s["id"] == space_id for s in listing.json()["items"])
    assert (await client.delete(f"/v1/spaces/{space_id}")).status_code == 204
    assert (await client.get(f"/v1/spaces/{space_id}")).status_code == 404


async def test_duplicate_slug_conflicts(client: httpx.AsyncClient) -> None:
    await client.post("/v1/spaces", json={"slug": "dup", "name": "One"})
    second = await client.post("/v1/spaces", json={"slug": "dup", "name": "Two"})
    assert second.status_code == 409


@pytest.mark.parametrize("slug", ["Bad Slug", "UPPER", "-leading", "", "a" * 100])
async def test_invalid_slugs_rejected(client: httpx.AsyncClient, slug: str) -> None:
    response = await client.post("/v1/spaces", json={"slug": slug, "name": "X"})
    assert response.status_code == 422


# -- memories ------------------------------------------------------------------


async def test_create_and_fetch_memory(client, space_id) -> None:
    created = await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={"content": "Kafka was chosen for the event bus", "tags": ["eng"]},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["created"] is True
    assert body["chunk_count"] >= 1
    memory_id = body["memory"]["id"]

    fetched = await client.get(f"/v1/spaces/{space_id}/memories/{memory_id}")
    assert fetched.status_code == 200
    assert fetched.json()["tags"] == ["eng"]
    assert fetched.headers.get("etag")


async def test_exact_duplicate_is_merged_not_duplicated(client, space_id) -> None:
    payload = {"content": "The deploy pipeline uses GitHub Actions"}
    first = await client.post(f"/v1/spaces/{space_id}/memories", json=payload)
    second = await client.post(f"/v1/spaces/{space_id}/memories", json=payload)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["duplicate_kind"] == "exact"
    assert second.json()["duplicate_of"] == first.json()["memory"]["id"]

    listing = await client.get(f"/v1/spaces/{space_id}/memories")
    assert listing.json()["total"] == 1


async def test_dedupe_can_be_disabled(client, space_id) -> None:
    payload = {"content": "Repeated on purpose", "dedupe": False}
    await client.post(f"/v1/spaces/{space_id}/memories", json=payload)
    second = await client.post(f"/v1/spaces/{space_id}/memories", json=payload)
    assert second.status_code == 201
    assert second.json()["created"] is True


async def test_bulk_ingest(client, space_id) -> None:
    items = [{"content": f"bulk memory number {i}"} for i in range(5)]
    response = await client.post(f"/v1/spaces/{space_id}/memories/bulk", json={"items": items})
    assert response.status_code == 201
    assert response.json()["created"] == 5


async def test_bulk_rejects_oversized_batches(client, space_id) -> None:
    items = [{"content": f"item {i}"} for i in range(101)]
    response = await client.post(f"/v1/spaces/{space_id}/memories/bulk", json={"items": items})
    assert response.status_code == 422


async def test_delete_memory(client, space_id) -> None:
    created = await client.post(
        f"/v1/spaces/{space_id}/memories", json={"content": "temporary"}
    )
    memory_id = created.json()["memory"]["id"]
    assert (
        await client.delete(f"/v1/spaces/{space_id}/memories/{memory_id}")
    ).status_code == 204
    assert (await client.get(f"/v1/spaces/{space_id}/memories/{memory_id}")).status_code == 404


async def test_pagination_walks_the_whole_collection(client, space_id) -> None:
    for i in range(7):
        await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": f"paged item {i}"}
        )
    seen: set[str] = set()
    cursor = None
    for _ in range(10):
        params = {"limit": 3}
        if cursor:
            params["cursor"] = cursor
        page = (await client.get(f"/v1/spaces/{space_id}/memories", params=params)).json()
        seen.update(item["id"] for item in page["items"])
        cursor = page.get("next_cursor")
        if not cursor:
            break
    assert len(seen) == 7


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, 422),  # missing content
        ({"content": ""}, 422),  # empty content
        ({"content": "x", "unknown": 1}, 422),  # unknown field
        ({"content": "x", "tags": ["y" * 100]}, 422),
        ({"content": "x", "metadata": {"k": {"nested": 1}}}, 422),
    ],
)
async def test_memory_validation(client, space_id, payload, expected) -> None:
    assert (
        await client.post(f"/v1/spaces/{space_id}/memories", json=payload)
    ).status_code == expected


async def test_malformed_ids_are_rejected_before_lookup(client, space_id) -> None:
    assert (await client.get(f"/v1/spaces/{space_id}/memories/nope")).status_code == 422
    assert (await client.get("/v1/spaces/notaspace/memories")).status_code == 422


async def test_unknown_memory_is_404(client, space_id) -> None:
    missing = "mem_00000000000000000000000000"
    assert (await client.get(f"/v1/spaces/{space_id}/memories/{missing}")).status_code == 404


# -- relations and belief revision --------------------------------------------


async def test_supersede_relation_hides_the_older_memory(client, space_id) -> None:
    old = (
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": "Deploys go through Jenkins"},
        )
    ).json()["memory"]
    new = (
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": "Deploys now go through GitHub Actions"},
        )
    ).json()["memory"]

    linked = await client.post(
        f"/v1/spaces/{space_id}/memories/{new['id']}/relations",
        json={"target_id": old["id"], "relation": "supersedes", "reason": "migrated"},
    )
    assert linked.status_code == 201
    assert linked.json()["source_id"] == new["id"]
    assert linked.json()["target_id"] == old["id"]

    results = (
        await client.post(
            f"/v1/spaces/{space_id}/search", json={"query": "how do deploys work"}
        )
    ).json()["results"]
    contents = [r["memory"]["content"] for r in results]
    assert any("GitHub Actions" in c for c in contents)
    assert not any("Jenkins" in c for c in contents)

    with_old = (
        await client.post(
            f"/v1/spaces/{space_id}/search",
            json={"query": "how do deploys work", "include_superseded": True},
        )
    ).json()["results"]
    assert any("Jenkins" in r["memory"]["content"] for r in with_old)


async def test_contradiction_is_recorded_on_both_memories(client, space_id) -> None:
    a = (
        await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": "The API is stable"}
        )
    ).json()["memory"]
    b = (
        await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": "The API is unstable"}
        )
    ).json()["memory"]
    await client.post(
        f"/v1/spaces/{space_id}/memories/{a['id']}/relations",
        json={"target_id": b["id"], "relation": "contradicts"},
    )
    # `contradicts` is symmetric, so B must know about A without being asked
    # about A first -- otherwise the answer depends on lookup order.
    other = (await client.get(f"/v1/spaces/{space_id}/memories/{b['id']}/relations")).json()
    assert any(r["target_id"] == a["id"] for r in other["items"])


async def test_self_relation_is_rejected(client, space_id) -> None:
    memory = (
        await client.post(f"/v1/spaces/{space_id}/memories", json={"content": "alone"})
    ).json()["memory"]
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/{memory['id']}/relations",
        json={"target_id": memory["id"], "relation": "supersedes"},
    )
    assert response.status_code == 422


# -- search --------------------------------------------------------------------


@pytest.fixture
async def seeded(client, space_id):
    corpus = [
        "We chose Kafka for the event bus",
        "Kafka cluster upgraded to version 3.7",
        "The team offsite moved to Lisbon",
        "Postgres migration approved for Q3",
        "Latency budget breached on the checkout path",
    ]
    for content in corpus:
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": content, "tags": ["seed"]},
        )
    return space_id


async def test_hybrid_search_uses_both_strategies(client, seeded) -> None:
    response = await client.post(
        f"/v1/spaces/{seeded}/search", json={"query": "kafka event bus", "limit": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert set(body["strategies"]) == {"vector", "lexical"}
    assert "Kafka" in body["results"][0]["memory"]["content"]


async def test_explain_exposes_score_provenance(client, seeded) -> None:
    body = (
        await client.post(
            f"/v1/spaces/{seeded}/search", json={"query": "kafka", "explain": True}
        )
    ).json()
    hit = body["results"][0]
    assert hit["explain"]
    assert hit["fusion_score"] is not None
    assert hit["recency_factor"] is not None
    assert any("rank" in part for part in hit["explain"])


async def test_explain_is_omitted_by_default(client, seeded) -> None:
    body = (await client.post(f"/v1/spaces/{seeded}/search", json={"query": "kafka"})).json()
    assert body["results"][0]["explain"] is None


async def test_min_score_filters_irrelevant_matches(client, seeded) -> None:
    loose = (
        await client.post(f"/v1/spaces/{seeded}/search", json={"query": "zzzz nonsense qqqq"})
    ).json()
    strict = (
        await client.post(
            f"/v1/spaces/{seeded}/search",
            json={"query": "zzzz nonsense qqqq", "min_score": 0.5},
        )
    ).json()
    assert strict["count"] <= loose["count"]
    assert strict["count"] == 0


async def test_tag_filter_narrows_search(client, seeded, space_id) -> None:
    await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={"content": "Kafka note with another tag", "tags": ["other"]},
    )
    body = (
        await client.post(
            f"/v1/spaces/{seeded}/search", json={"query": "kafka", "tags": ["other"]}
        )
    ).json()
    assert body["count"] == 1
    assert body["results"][0]["memory"]["tags"] == ["other"]


async def test_search_timings_are_reported(client, seeded) -> None:
    body = (await client.post(f"/v1/spaces/{seeded}/search", json={"query": "kafka"})).json()
    assert "candidates_ms" in body["timings_ms"]
    assert all(v >= 0 for v in body["timings_ms"].values())


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": 10_000},
        {"query": "x", "mmr_lambda": 2.0},
        {"query": "x", "half_life_days": 0},
        {"query": "x", "unknown_field": True},
        {},
    ],
)
async def test_search_validation(client, seeded, payload) -> None:
    assert (await client.post(f"/v1/spaces/{seeded}/search", json=payload)).status_code == 422


async def test_search_date_range_must_be_ordered(client, seeded) -> None:
    response = await client.post(
        f"/v1/spaces/{seeded}/search",
        json={
            "query": "kafka",
            "occurred_after": "2026-01-01T00:00:00Z",
            "occurred_before": "2025-01-01T00:00:00Z",
        },
    )
    assert response.status_code == 422


async def test_search_on_an_empty_space_returns_no_results(client) -> None:
    empty = (await client.post("/v1/spaces", json={"slug": "empty", "name": "Empty"})).json()[
        "id"
    ]
    body = (await client.post(f"/v1/spaces/{empty}/search", json={"query": "anything"})).json()
    assert body["count"] == 0
    assert body["results"] == []


async def test_mmr_reduces_near_duplicate_results(client, space_id) -> None:
    variants = [
        "The quarterly planning meeting is on Monday",
        "Quarterly planning meeting happens Monday",
        "The planning meeting for the quarter is Monday",
        "Kafka handles the event bus",
    ]
    for content in variants:
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": content, "dedupe": False},
        )
    diverse = (
        await client.post(
            f"/v1/spaces/{space_id}/search",
            json={"query": "planning meeting", "limit": 2, "mmr_lambda": 0.3},
        )
    ).json()
    assert diverse["count"] == 2


# -- error contract ------------------------------------------------------------


async def test_errors_are_problem_json(client, space_id) -> None:
    response = await client.get(f"/v1/spaces/{space_id}/memories/bad-id")
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    for key in ("type", "code", "title", "status", "detail", "request_id"):
        assert key in body


async def test_validation_errors_name_the_offending_field(client, space_id) -> None:
    body = (await client.post(f"/v1/spaces/{space_id}/memories", json={"content": ""})).json()
    assert body["code"] == "validation_error"
    assert any("content" in e["field"] for e in body["errors"])


async def test_unknown_route_is_problem_json(client) -> None:
    response = await client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_oversized_body_is_rejected(app_context) -> None:
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        response = await anon.post(
            "/v1/spaces",
            headers={
                "Authorization": f"Bearer {app_context['key']}",
                "content-length": "99999999",
            },
            content=b"{}",
        )
    assert response.status_code == 413


# -- concurrency ---------------------------------------------------------------


async def test_concurrent_writes_all_persist(client, space_id) -> None:
    async def write(i: int):
        return await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": f"concurrent memory {i}"},
        )

    responses = await asyncio.gather(*(write(i) for i in range(20)))
    assert all(r.status_code == 201 for r in responses)
    listing = (
        await client.get(f"/v1/spaces/{space_id}/memories", params={"limit": 100})
    ).json()
    assert listing["total"] == 20


async def test_concurrent_searches_are_consistent(client, seeded) -> None:
    async def query():
        return await client.post(
            f"/v1/spaces/{seeded}/search", json={"query": "kafka", "limit": 3}
        )

    responses = await asyncio.gather(*(query() for _ in range(10)))
    assert all(r.status_code == 200 for r in responses)
    orderings = {tuple(h["memory"]["id"] for h in r.json()["results"]) for r in responses}
    # Deterministic pipeline: identical requests must produce identical order.
    assert len(orderings) == 1


# -- temporal graph: multi-hop suppression, lineage, history -------------------


async def _chain_of_three(client, space_id):
    """Jenkins <- CircleCI <- GitHub Actions, oldest first."""
    ids = []
    for content in (
        "Deploys go through Jenkins",
        "Deploys go through CircleCI",
        "Deploys go through GitHub Actions",
    ):
        created = await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": content}
        )
        ids.append(created.json()["memory"]["id"])
    oldest, middle, newest = ids
    for source, target in ((middle, oldest), (newest, middle)):
        await client.post(
            f"/v1/spaces/{space_id}/memories/{source}/relations",
            json={"target_id": target, "relation": "supersedes"},
        )
    return oldest, middle, newest


async def test_multi_hop_supersession_hides_every_stale_revision(client, space_id) -> None:
    """The bug this graph exists to fix.

    With a relations list on each memory, suppression could only see one hop:
    the two-hops-stale fact survived as though it were current.
    """
    await _chain_of_three(client, space_id)
    results = (
        await client.post(
            f"/v1/spaces/{space_id}/search",
            json={"query": "how do deploys work", "limit": 10},
        )
    ).json()["results"]
    contents = [r["memory"]["content"] for r in results]

    assert any("GitHub Actions" in c for c in contents)
    assert not any("CircleCI" in c for c in contents), "one hop stale"
    assert not any("Jenkins" in c for c in contents), "two hops stale"


async def test_lineage_resolves_the_head_of_truth(client, space_id) -> None:
    oldest, middle, newest = await _chain_of_three(client, space_id)

    stale = (await client.get(f"/v1/spaces/{space_id}/memories/{oldest}/lineage")).json()
    assert stale["is_current"] is False
    assert stale["successors"] == [middle, newest]
    assert stale["head"] == newest

    current = (await client.get(f"/v1/spaces/{space_id}/memories/{newest}/lineage")).json()
    assert current["is_current"] is True
    assert current["ancestors"] == [middle, oldest]
    assert current["head"] == newest


async def test_relations_endpoint_both_directions(client, space_id) -> None:
    oldest, middle, newest = await _chain_of_three(client, space_id)

    outgoing = (
        await client.get(
            f"/v1/spaces/{space_id}/memories/{newest}/relations",
            params={"direction": "out"},
        )
    ).json()["items"]
    assert [e["target_id"] for e in outgoing] == [middle]

    incoming = (
        await client.get(
            f"/v1/spaces/{space_id}/memories/{oldest}/relations",
            params={"direction": "in"},
        )
    ).json()["items"]
    assert [e["source_id"] for e in incoming] == [middle]


async def test_relations_endpoint_rejects_a_bad_direction(client, space_id) -> None:
    created = await client.post(f"/v1/spaces/{space_id}/memories", json={"content": "anything"})
    memory_id = created.json()["memory"]["id"]
    response = await client.get(
        f"/v1/spaces/{space_id}/memories/{memory_id}/relations",
        params={"direction": "sideways"},
    )
    assert response.status_code == 422


async def test_version_history_records_the_supersession(client, space_id) -> None:
    oldest, _, _ = await _chain_of_three(client, space_id)
    versions = (await client.get(f"/v1/spaces/{space_id}/memories/{oldest}/versions")).json()[
        "items"
    ]

    assert [v["version"] for v in versions] == [1, 2]
    assert [v["status"] for v in versions] == ["active", "superseded"]
    # Exactly one open version, and it is the last.
    assert [v["valid_to"] is None for v in versions] == [False, True]


async def test_point_in_time_read_returns_the_older_state(client, space_id) -> None:
    oldest, _, _ = await _chain_of_three(client, space_id)
    versions = (await client.get(f"/v1/spaces/{space_id}/memories/{oldest}/versions")).json()[
        "items"
    ]

    historical = (
        await client.get(
            f"/v1/spaces/{space_id}/memories/{oldest}",
            params={"as_of": versions[0]["valid_from"]},
        )
    ).json()
    assert historical["status"] == "active"
    assert historical["version"] == 1

    current = (await client.get(f"/v1/spaces/{space_id}/memories/{oldest}")).json()
    assert current["status"] == "superseded"
    assert current["version"] == 2


async def test_point_in_time_before_creation_is_404(client, space_id) -> None:
    created = await client.post(f"/v1/spaces/{space_id}/memories", json={"content": "recent"})
    memory_id = created.json()["memory"]["id"]
    response = await client.get(
        f"/v1/spaces/{space_id}/memories/{memory_id}",
        params={"as_of": "2001-01-01T00:00:00Z"},
    )
    assert response.status_code == 404


async def test_relation_creation_is_idempotent_over_http(client, space_id) -> None:
    oldest, middle, _ = await _chain_of_three(client, space_id)
    repeat = await client.post(
        f"/v1/spaces/{space_id}/memories/{middle}/relations",
        json={"target_id": oldest, "relation": "supersedes"},
    )
    assert repeat.status_code == 201
    edges = (
        await client.get(
            f"/v1/spaces/{space_id}/memories/{middle}/relations",
            params={"direction": "out"},
        )
    ).json()["items"]
    assert len(edges) == 1


async def test_relation_to_a_missing_memory_is_404(client, space_id) -> None:
    created = await client.post(f"/v1/spaces/{space_id}/memories", json={"content": "solo"})
    memory_id = created.json()["memory"]["id"]
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/{memory_id}/relations",
        json={
            "target_id": "mem_00000000000000000000000000",
            "relation": "supersedes",
        },
    )
    assert response.status_code == 404


async def test_lineage_is_tenant_scoped(app_context, client, space_id) -> None:
    oldest, _, _ = await _chain_of_three(client, space_id)

    store = app_context["store"]
    intruder = await store.create_organization(Organization(name="Intruder"))
    record, plaintext = build_api_key(
        org_id=intruder.id,
        name="k",
        pepper="test-pepper",
        scopes=frozenset(Scope.all()),
    )
    await store.create_api_key(record)
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://t",
        headers={"Authorization": f"Bearer {plaintext}"},
    ) as other:
        assert (
            await other.get(f"/v1/spaces/{space_id}/memories/{oldest}/lineage")
        ).status_code == 404


# -- erasure -------------------------------------------------------------------


async def test_erase_returns_an_attestation_and_scrubs_history(client, space_id) -> None:
    created = await client.post(
        f"/v1/spaces/{space_id}/memories", json={"content": "delete my data please"}
    )
    memory_id = created.json()["memory"]["id"]
    versions = (
        await client.get(f"/v1/spaces/{space_id}/memories/{memory_id}/versions")
    ).json()["items"]
    first_valid_from = versions[0]["valid_from"]

    response = await client.post(f"/v1/spaces/{space_id}/memories/{memory_id}/erase")
    assert response.status_code == 200
    attestation = response.json()
    assert attestation["memory_id"] == memory_id
    assert len(attestation["content_sha256"]) == 64
    assert attestation["versions_purged"] >= 1
    assert attestation["chunks_removed"] >= 1

    # Gone from the present...
    assert (await client.get(f"/v1/spaces/{space_id}/memories/{memory_id}")).status_code == 404
    # ...and from the reconstructed past. This is the line delete does not cross.
    as_of = await client.get(
        f"/v1/spaces/{space_id}/memories/{memory_id}",
        params={"as_of": first_valid_from},
    )
    assert as_of.status_code == 404
    assert (
        await client.get(f"/v1/spaces/{space_id}/memories/{memory_id}/versions")
    ).status_code == 404


async def test_erase_reports_derived_memories(client, space_id) -> None:
    source = (
        await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": "original document"}
        )
    ).json()["memory"]
    derived = (
        await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": "summary of the original"}
        )
    ).json()["memory"]
    await client.post(
        f"/v1/spaces/{space_id}/memories/{derived['id']}/relations",
        json={"target_id": source["id"], "relation": "derived_from"},
    )

    attestation = (
        await client.post(f"/v1/spaces/{space_id}/memories/{source['id']}/erase")
    ).json()
    assert attestation["derived_memories_affected"] == [derived["id"]]
    assert attestation["edges_removed"] == 1


async def test_erase_missing_memory_is_404(client, space_id) -> None:
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/mem_00000000000000000000000000/erase"
    )
    assert response.status_code == 404


async def test_erase_requires_write_scope(app_context, space_id) -> None:
    store = app_context["store"]
    record, plaintext = build_api_key(
        org_id=app_context["org"].id,
        name="ro",
        pepper="test-pepper",
        scopes=frozenset({Scope.SEARCH, Scope.MEMORIES_READ}),
    )
    await store.create_api_key(record)
    transport = httpx.ASGITransport(app=app_context["app"])
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://t",
        headers={"Authorization": f"Bearer {plaintext}"},
    ) as limited:
        response = await limited.post(
            f"/v1/spaces/{space_id}/memories/mem_00000000000000000000000000/erase"
        )
        assert response.status_code == 403
