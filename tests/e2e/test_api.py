"""End-to-end API behaviour through the real ASGI app.

These exercise the whole stack — middleware, auth, validation, service, store —
with no mocks anywhere. The only substitutions are the storage and embedding
backends, and both of those are real implementations.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from mapi.core.security import build_api_key
from mapi.domain.models import Organization, Scope, Space

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
    assert "mapi_http_requests_total" in response.text


async def test_openapi_document_is_valid(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    assert schema["info"]["title"] == "Mapi"
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


async def test_consolidation_cannot_be_switched_off(client, space_id) -> None:
    """The write body carries no behaviour flags, and unknown fields are rejected.

    Deduplication, supersession, contradiction detection and extraction all
    run on every write. They used to be opt-in, which meant the graph was
    empty for anyone who did not know to ask -- a memory API whose headline
    features have an off switch that defaults to off is a memory API that
    does nothing.
    """
    for removed in ("dedupe", "extract", "auto_supersede", "detect_conflicts"):
        response = await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": "a memory", removed: False},
        )
        assert response.status_code == 422, f"{removed} is still accepted"


async def test_an_exact_duplicate_is_always_collapsed(client, space_id) -> None:
    payload = {"content": "Repeated on purpose"}
    first = await client.post(f"/v1/spaces/{space_id}/memories", json=payload)
    second = await client.post(f"/v1/spaces/{space_id}/memories", json=payload)
    assert first.status_code == 201
    # 200, not 201: the second write created nothing, and the status says so.
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["memory"]["id"] == first.json()["memory"]["id"]


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
            json={"content": content},
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


async def test_context_assembles_the_entire_neighborhood(client, space_id) -> None:
    """One read returns the memory plus everything an agent needs to trust it:
    currency, what replaced it, provenance, derivatives, references. The
    cluster is the relation graph made readable."""

    async def add(content: str) -> dict:
        response = await client.post(
            f"/v1/spaces/{space_id}/memories", json={"content": content}
        )
        return response.json()["memory"]

    async def link(source: dict, target: dict, relation: str) -> None:
        response = await client.post(
            f"/v1/spaces/{space_id}/memories/{source['id']}/relations",
            json={"target_id": target["id"], "relation": relation},
        )
        assert response.status_code == 201

    old = await add("We deploy with Jenkins")
    new = await add("We deploy with GitHub Actions")
    source_a = await add("Standup note: gym on Monday")
    source_b = await add("Retro note: gym again on Thursday")
    derived = await add("Derived: went to the gym twice this week")
    reference = await add("See the fitness challenge thread")

    await link(new, old, "supersedes")
    await link(derived, source_a, "derived_from")
    await link(derived, source_b, "derived_from")
    await link(derived, reference, "references")

    # The derived fact: provenance cluster plus references, and it is current.
    context = (
        await client.get(f"/v1/spaces/{space_id}/memories/{derived['id']}/context")
    ).json()
    assert context["is_current"] is True
    assert context["current_head"] == []
    assert {m["id"] for m in context["derived_from"]} == {source_a["id"], source_b["id"]}
    assert [m["id"] for m in context["references"]] == [reference["id"]]

    # A source sees the derivative pointing back at it.
    context = (
        await client.get(f"/v1/spaces/{space_id}/memories/{source_a['id']}/context")
    ).json()
    assert [m["id"] for m in context["derivatives"]] == [derived["id"]]

    # The superseded memory knows it is stale and names its replacement.
    context = (await client.get(f"/v1/spaces/{space_id}/memories/{old['id']}/context")).json()
    assert context["is_current"] is False
    assert [m["id"] for m in context["current_head"]] == [new["id"]]

    # An erased source disappears from the derivative's context — omitted,
    # not stubbed: erasure must not leak through surviving neighbors.
    erased = await client.post(f"/v1/spaces/{space_id}/memories/{source_b['id']}/erase")
    assert erased.status_code == 200
    context = (
        await client.get(f"/v1/spaces/{space_id}/memories/{derived['id']}/context")
    ).json()
    assert {m["id"] for m in context["derived_from"]} == {source_a["id"]}


# -- derived-memory lifecycle --------------------------------------------------


async def _mk(client, space_id: str, content: str) -> dict:
    response = await client.post(f"/v1/spaces/{space_id}/memories", json={"content": content})
    return response.json()["memory"]


async def _rel(client, space_id: str, source: dict, target: dict, relation: str) -> None:
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/{source['id']}/relations",
        json={"target_id": target["id"], "relation": relation},
    )
    assert response.status_code == 201


async def _status_of(client, space_id: str, memory: dict) -> str:
    return (await client.get(f"/v1/spaces/{space_id}/memories/{memory['id']}")).json()["status"]


async def test_erasing_a_source_marks_its_derivation_stale(client, space_id) -> None:
    """The invariant: a derivation must never outlive its evidence. Erase a
    source episode and the fact computed from it stops being served as truth —
    stale, not deleted, because it is recomputable from the survivors."""
    source_a = await _mk(client, space_id, "Bought a road bike in March")
    source_b = await _mk(client, space_id, "Bought an e-bike in May")
    derived = await _mk(client, space_id, "Owns two bikes")
    await _rel(client, space_id, derived, source_a, "derived_from")
    await _rel(client, space_id, derived, source_b, "derived_from")

    erased = await client.post(f"/v1/spaces/{space_id}/memories/{source_a['id']}/erase")
    assert erased.status_code == 200
    assert derived["id"] in erased.json()["derived_memories_affected"]
    assert await _status_of(client, space_id, derived) == "stale"


async def test_superseding_a_source_marks_its_derivation_stale(client, space_id) -> None:
    """Supersession invalidates too: the derivation may describe a replaced
    state of the world."""
    old = await _mk(client, space_id, "Lives in Portland")
    derived = await _mk(client, space_id, "Commute is 20 minutes, from Portland")
    await _rel(client, space_id, derived, old, "derived_from")
    new = await _mk(client, space_id, "Moved to Seattle")
    await _rel(client, space_id, new, old, "supersedes")
    assert await _status_of(client, space_id, derived) == "stale"


async def test_staleness_propagates_through_derivation_chains(client, space_id) -> None:
    """Profile fact derived from a timeline derived from an episode: erasing
    the episode must reach the profile fact, two hops away."""
    episode = await _mk(client, space_id, "Gym session on Tuesday")
    timeline = await _mk(client, space_id, "Timeline: one gym visit this week")
    profile = await _mk(client, space_id, "User exercises regularly")
    await _rel(client, space_id, timeline, episode, "derived_from")
    await _rel(client, space_id, profile, timeline, "derived_from")

    await client.post(f"/v1/spaces/{space_id}/memories/{episode['id']}/erase")
    assert await _status_of(client, space_id, timeline) == "stale"
    assert await _status_of(client, space_id, profile) == "stale"


async def test_stale_derivations_leave_search_results(client, space_id) -> None:
    source = await _mk(client, space_id, "Cycling log: bought a gravel bike")
    derived = await _mk(client, space_id, "Owns one gravel bike total")
    await _rel(client, space_id, derived, source, "derived_from")

    results = (
        await client.post(f"/v1/spaces/{space_id}/search", json={"query": "gravel bike"})
    ).json()["results"]
    assert any(r["memory"]["id"] == derived["id"] for r in results)

    await client.post(f"/v1/spaces/{space_id}/memories/{source['id']}/erase")
    results = (
        await client.post(f"/v1/spaces/{space_id}/search", json={"query": "gravel bike"})
    ).json()["results"]
    assert not any(r["memory"]["id"] == derived["id"] for r in results)


async def test_deleting_a_source_also_invalidates(client, space_id) -> None:
    """Plain delete severs edges (FK cascade / edge sweep), so the derivative
    set is collected before the delete — the invalidation must still land."""
    source = await _mk(client, space_id, "Subscription: Netflix at 15 dollars")
    derived = await _mk(client, space_id, "Total subscriptions: one")
    await _rel(client, space_id, derived, source, "derived_from")
    response = await client.delete(f"/v1/spaces/{space_id}/memories/{source['id']}")
    assert response.status_code == 204
    assert await _status_of(client, space_id, derived) == "stale"


# -- derive endpoint and profiles ----------------------------------------------


def _install_fake_completer(app_context, mapping: dict[str, str]) -> None:
    """Deterministic completer: first key found in the prompt wins."""

    async def fake(prompt: str) -> str:
        for needle, reply in mapping.items():
            if needle in prompt:
                return reply
        return "[]"

    app_context["app"].state.service.completer = fake


async def test_derive_without_backend_is_503_not_silent(client, space_id) -> None:
    response = await client.post(
        f"/v1/spaces/{space_id}/derive", json={"question": "How many bikes do I own?"}
    )
    assert response.status_code == 502 or response.status_code == 503


async def test_derive_counts_in_code_and_materializes_with_provenance(
    client, app_context, space_id
) -> None:
    """The full loop: episodes in, computed answer out, provenance stored,
    profile readable, invalidation live."""
    a = await _mk(client, space_id, "Picked up a road bike today, love it")
    b = await _mk(client, space_id, "My e-bike arrived, second bike in the garage")
    _install_fake_completer(
        app_context,
        {
            "road bike": '[{"date": "2026-03-01", "fact": "owns a road bike",'
            ' "quote": "road bike"}]',
            "e-bike": '[{"date": "2026-03-08", "fact": "owns an e-bike", "quote": "e-bike"}]',
        },
    )

    response = await client.post(
        f"/v1/spaces/{space_id}/derive",
        json={
            "question": "How many bikes do I own?",
            "materialize": True,
            "bucket": "possessions",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "2"
    assert payload["computed"] is True
    assert set(payload["source_ids"]) == {a["id"], b["id"]}
    stored = payload["memory"]
    assert stored is not None and stored["kind"] == "derived"

    # Provenance visible through the context read.
    context = (
        await client.get(f"/v1/spaces/{space_id}/memories/{stored['id']}/context")
    ).json()
    assert {m["id"] for m in context["derived_from"]} == {a["id"], b["id"]}

    # Profile lists the fact as current truth.
    profile = (await client.get(f"/v1/spaces/{space_id}/profiles/possessions")).json()
    assert [f["id"] for f in profile["facts"]] == [stored["id"]]

    # Invalidation: erase a source, the profile fact stops being truth.
    await client.post(f"/v1/spaces/{space_id}/memories/{a['id']}/erase")
    assert await _status_of(client, space_id, stored) == "stale"
    profile = (await client.get(f"/v1/spaces/{space_id}/profiles/possessions")).json()
    assert profile["facts"] == []


async def test_rederiving_a_bucket_supersedes_the_previous_fact(
    client, app_context, space_id
) -> None:
    """Profiles update through ordinary revision machinery: the new fact
    supersedes the old, history stays walkable."""
    await _mk(client, space_id, "Bought a gravel bike, my first bicycle ever")
    _install_fake_completer(
        app_context,
        {
            "gravel": '[{"date": "2026-04-01", "fact": "owns a gravel bike",'
            ' "quote": "gravel bike"}]'
        },
    )
    body = {"question": "How many bikes do I own?", "materialize": True, "bucket": "gear"}
    first = (await client.post(f"/v1/spaces/{space_id}/derive", json=body)).json()["memory"]
    second = (await client.post(f"/v1/spaces/{space_id}/derive", json=body)).json()["memory"]
    assert first["id"] != second["id"]

    assert await _status_of(client, space_id, first) == "superseded"
    profile = (await client.get(f"/v1/spaces/{space_id}/profiles/gear")).json()
    assert [f["id"] for f in profile["facts"]] == [second["id"]]

    lineage = (await client.get(f"/v1/spaces/{space_id}/memories/{first['id']}/lineage")).json()
    assert lineage["is_current"] is False
    assert lineage["head"] == second["id"]


async def test_derive_fails_open_to_empty_answer_on_garbage(
    client, app_context, space_id
) -> None:
    await _mk(client, space_id, "Some cycling note")
    _install_fake_completer(app_context, {"cycling": "sorry, no JSON from me"})
    response = await client.post(
        f"/v1/spaces/{space_id}/derive",
        json={"question": "How many bikes do I own?", "materialize": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == ""
    assert payload["memory"] is None  # nothing materialized from nothing


# -- contradiction surfacing ---------------------------------------------------


async def test_conflicting_writes_are_detected_and_surfaced_in_search(client, space_id) -> None:
    """Disagreement, not revision: neither memory is hidden, and search REPORTS
    the conflict. Every competitor resolves this invisibly by picking the
    newest timestamp, which is indistinguishable from there being no conflict
    at all. An agent told two facts disagree can ask the user."""
    first = (
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": "The quarterly planning meeting is on Tuesday at 3pm"},
        )
    ).json()["memory"]
    second = await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={
            "content": "The quarterly planning meeting is on Thursday at 3pm",
        },
    )
    assert second.status_code == 201
    payload = second.json()
    assert first["id"] in payload["contradicts"]

    # Both remain active and retrievable — either may be the true one.
    body = (
        await client.post(
            f"/v1/spaces/{space_id}/search", json={"query": "quarterly planning meeting"}
        )
    ).json()
    ids = {r["memory"]["id"] for r in body["results"]}
    assert {first["id"], payload["memory"]["id"]} <= ids

    # And the disagreement is reported rather than silently resolved.
    pairs = {tuple(sorted(pair)) for pair in body["conflicts"]}
    assert tuple(sorted((first["id"], payload["memory"]["id"]))) in pairs


async def test_contradiction_is_visible_from_both_sides(client, space_id) -> None:
    """`contradicts` is symmetric: the answer must not depend on which memory
    you happen to look up first."""
    a = (
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": "The API rate limit is 100 requests per minute"},
        )
    ).json()["memory"]
    b = (
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={
                "content": "The API rate limit is 500 requests per minute",
            },
        )
    ).json()["memory"]

    for left, right in ((a, b), (b, a)):
        ctx = (await client.get(f"/v1/spaces/{space_id}/memories/{left['id']}/context")).json()
        assert right["id"] in {m["id"] for m in ctx["contradicts"]}


async def test_conflict_detection_needs_no_asking(client, space_id) -> None:
    """It used to be opt-in, on the theory that the candidate scan was the
    cost that mattered. It was not: the scan is now a bounded
    nearest-neighbour lookup, and a memory system whose conflict detection
    is off unless you knew to ask for it does not detect conflicts."""
    await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={"content": "The office moves to the 4th floor in June"},
    )
    second = (
        await client.post(
            f"/v1/spaces/{space_id}/memories",
            json={"content": "The office moves to the 9th floor in June"},
        )
    ).json()
    assert second["contradicts"], "no flag was passed, and none should be needed"


# -- calibration ---------------------------------------------------------------


async def test_search_reports_confidence_and_names_correct_silence(client, space_id) -> None:
    """A query matching nothing must be distinguishable from one matching
    something weakly. Only the second is a memory gap; the first is the system
    behaving correctly, and an agent needs to tell them apart."""
    await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={"content": "The deployment pipeline runs on GitHub Actions"},
    )
    hit = (
        await client.post(
            f"/v1/spaces/{space_id}/search", json={"query": "deployment pipeline"}
        )
    ).json()
    assert hit["confidence"]["level"] in {"high", "medium", "low"}
    assert hit["confidence"]["n_results"] >= 1

    miss = (
        await client.post(
            f"/v1/spaces/{space_id}/search",
            json={"query": "deployment pipeline", "min_score": 0.99},
        )
    ).json()
    assert miss["results"] == []
    assert miss["confidence"]["level"] == "none"
    assert miss["confidence"]["refusal_reason"] == "no_relevant_memory"


async def test_conflicting_results_lower_confidence(client, space_id) -> None:
    """Contradiction feeds calibration: a perfect match to two memories that
    disagree is the worst case, not the best."""
    await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={"content": "The retention window is 30 days for all customers"},
    )
    await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={
            "content": "The retention window is 90 days for all customers",
        },
    )
    body = (
        await client.post(f"/v1/spaces/{space_id}/search", json={"query": "retention window"})
    ).json()
    assert body["conflicts"]
    assert body["confidence"]["level"] == "low"
    assert body["confidence"]["refusal_reason"] == "conflicting_evidence"
    assert body["confidence"]["has_conflicts"] is True


# -- consolidation -------------------------------------------------------------


async def test_consolidation_rederives_a_stale_profile_fact(
    client, app_context, space_id
) -> None:
    """Closes the loop Phase 2 left open: invalidation marked facts STALE and
    nothing recomputed them, so a profile stayed permanently empty after its
    first source was erased."""
    a = await _mk(client, space_id, "Picked up a road bike today, love it")
    b = await _mk(client, space_id, "My e-bike arrived, second bike in the garage")
    _install_fake_completer(
        app_context,
        {
            "road bike": '[{"date": "2026-03-01", "fact": "owns a road bike",'
            ' "quote": "road bike"}]',
            "e-bike": '[{"date": "2026-03-08", "fact": "owns an e-bike", "quote": "e-bike"}]',
        },
    )
    body = {"question": "How many bikes do I own?", "materialize": True, "bucket": "gear"}
    first = (await client.post(f"/v1/spaces/{space_id}/derive", json=body)).json()["memory"]
    assert first["kind"] == "derived"
    assert (await client.get(f"/v1/spaces/{space_id}/profiles/gear")).json()["facts"]

    # Erase a source: the fact goes stale and the profile falls silent.
    await client.post(f"/v1/spaces/{space_id}/memories/{a['id']}/erase")
    assert await _status_of(client, space_id, first) == "stale"
    assert (await client.get(f"/v1/spaces/{space_id}/profiles/gear")).json()["facts"] == []

    # Consolidation recomputes it from what survives.
    report = (await client.post(f"/v1/spaces/{space_id}/consolidate")).json()
    assert report["examined"] >= 1
    assert report["refreshed"]

    facts = (await client.get(f"/v1/spaces/{space_id}/profiles/gear")).json()["facts"]
    assert len(facts) == 1
    assert facts[0]["id"] != first["id"]  # a NEW fact, not the stale one revived
    assert await _status_of(client, space_id, first) == "superseded"
    # And it was recomputed from the surviving source only.
    ctx = (await client.get(f"/v1/spaces/{space_id}/memories/{facts[0]['id']}/context")).json()
    assert {m["id"] for m in ctx["derived_from"]} == {b["id"]}


async def test_consolidation_leaves_unrecoverable_facts_stale(
    client, app_context, space_id
) -> None:
    """When the surviving evidence no longer supports an answer, staying stale
    is the CORRECT outcome — not deletion, and not silent resurrection."""
    source = await _mk(client, space_id, "Bought a single gravel bike this spring")
    _install_fake_completer(
        app_context,
        {
            "gravel": '[{"date": "2026-04-01", "fact": "owns a gravel bike",'
            ' "quote": "gravel bike"}]'
        },
    )
    fact = (
        await client.post(
            f"/v1/spaces/{space_id}/derive",
            json={"question": "How many bikes do I own?", "materialize": True},
        )
    ).json()["memory"]
    await client.post(f"/v1/spaces/{space_id}/memories/{source['id']}/erase")
    assert await _status_of(client, space_id, fact) == "stale"

    report = (await client.post(f"/v1/spaces/{space_id}/consolidate")).json()
    assert fact["id"] in report["abandoned"]
    assert await _status_of(client, space_id, fact) == "stale"


async def test_consolidation_budget_is_bounded(client, app_context, space_id) -> None:
    """Background LLM spend is how competitors' token bills became marketing
    liabilities. The cost is capped and reported, never ambient."""
    _install_fake_completer(app_context, {})
    report = (await client.post(f"/v1/spaces/{space_id}/consolidate?budget=5")).json()
    assert report["budget"] == 5
    assert report["examined"] == 0  # nothing stale in a fresh space


async def test_consolidation_without_a_backend_fails_loudly(client, space_id) -> None:
    """Re-derivation needs a synthesis backend. Silence would look like
    "nothing to do" while stale facts accumulated forever."""
    response = await client.post(f"/v1/spaces/{space_id}/consolidate")
    assert response.status_code in (502, 503)


# -- memory graph --------------------------------------------------------------


async def test_graph_returns_typed_edges_between_memories(client, space_id) -> None:
    """The distinction that matters: these are relations between MEMORIES, not
    between a document and the chunks pulled out of it. An extraction-built
    graph gives every node exactly one parent and answers only "where did this
    text come from"; these edges answer "what does the system believe, and
    why"."""
    a = await _mk(client, space_id, "We deploy with Jenkins on Fridays")
    b = await _mk(client, space_id, "We deploy with GitHub Actions now")
    src = await _mk(client, space_id, "Standup: shipped the auth fix")
    fact = await _mk(client, space_id, "Derived: one release this week")
    other = await _mk(client, space_id, "Unrelated note about lunch")

    await _rel(client, space_id, b, a, "supersedes")
    await _rel(client, space_id, fact, src, "derived_from")

    g = (await client.get(f"/v1/spaces/{space_id}/graph")).json()
    ids = {n["id"] for n in g["nodes"]}
    assert {a["id"], b["id"], src["id"], fact["id"], other["id"]} <= ids

    kinds = {(e["source"], e["target"]): e["type"] for e in g["edges"]}
    assert kinds[(b["id"], a["id"])] == "supersedes"
    assert kinds[(fact["id"], src["id"])] == "derived_from"

    # Superseded memories stay IN the graph -- they are history, not deletions.
    assert next(n for n in g["nodes"] if n["id"] == a["id"])["status"] == "superseded"
    # Degree drives node size, so an island is visibly an island.
    assert next(n for n in g["nodes"] if n["id"] == other["id"])["degree"] == 0
    assert g["counts"]["isolated"] >= 1


async def test_graph_reports_contradictions_once_not_twice(client, space_id) -> None:
    """`contradicts` is stored symmetrically so lookups work from either side.
    The graph must render one line between the pair, not two overlapping."""
    await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={"content": "The retention window is 30 days for all customers"},
    )
    await client.post(
        f"/v1/spaces/{space_id}/memories",
        json={
            "content": "The retention window is 90 days for all customers",
        },
    )
    g = (await client.get(f"/v1/spaces/{space_id}/graph")).json()
    conflicts = [e for e in g["edges"] if e["type"] == "contradicts"]
    assert len(conflicts) == 1
    assert g["counts"]["by_edge"]["contradicts"] == 1


async def test_graph_ui_is_served(client) -> None:
    """The viewer ships with the app: a debugging surface needing its own
    build step is one nobody runs."""
    r = await client.get("/graph")
    assert r.status_code == 200
    assert "memory graph" in r.text.lower()
    assert "/context" in r.text  # clicking a node resolves the neighbourhood
