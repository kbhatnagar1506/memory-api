"""Client behaviour against a mocked service."""

from __future__ import annotations

import httpx
import pytest
import respx

from mapi_sdk import (
    AuthenticationError,
    Mapi,
    MapiError,
    NotFoundError,
    RateLimitError,
)

BASE = "https://api.test"
SPACES = {"items": [{"id": "spc_ada", "slug": "ada", "name": "Ada", "memory_count": 2}]}


@pytest.fixture
def client() -> Mapi:
    return Mapi(api_key="sm_test", base_url=BASE, max_retries=2)


def test_an_api_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAPI_API_KEY", raising=False)
    with pytest.raises(MapiError, match="no API key"):
        Mapi(base_url=BASE)


def test_the_key_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAPI_API_KEY", "sm_from_env")
    assert Mapi(base_url=BASE).api_key == "sm_from_env"


@respx.mock
def test_a_slug_is_resolved_to_an_id_once(client: Mapi) -> None:
    spaces = respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    create = respx.post(f"{BASE}/v1/spaces/spc_ada/memories").mock(
        httpx.Response(201, json={"memory": {"id": "mem_1", "content": "hi"}})
    )
    client.memories.add("hi", space="ada")
    client.memories.add("again", space="ada")
    assert spaces.call_count == 1, "the slug lookup must be cached"
    assert create.call_count == 2


@respx.mock
def test_an_id_skips_the_lookup_entirely(client: Mapi) -> None:
    spaces = respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    respx.post(f"{BASE}/v1/spaces/spc_direct/memories").mock(
        httpx.Response(201, json={"memory": {"id": "mem_1", "content": "hi"}})
    )
    client.memories.add("hi", space="spc_direct")
    assert spaces.call_count == 0


@respx.mock
def test_an_unknown_space_is_created_rather_than_refused(client: Mapi) -> None:
    """Writing to a space that does not exist yet is the first write.

    This used to raise "no space with slug X, create it first" -- ceremony
    that existed only because the server keeps spaces in a table.
    """
    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json={"items": []}))
    created = respx.post(f"{BASE}/v1/spaces").mock(
        httpx.Response(201, json={"id": "spc_new", "slug": "fresh", "name": "fresh"})
    )
    write = respx.post(f"{BASE}/v1/spaces/spc_new/memories").mock(
        httpx.Response(201, json={"memory": {"id": "mem_1", "content": "hi"}})
    )

    client.memories.add("hi", space="fresh")

    assert created.call_count == 1
    assert write.call_count == 1


@respx.mock
def test_a_created_space_is_cached_like_any_other(client: Mapi) -> None:
    """One creation per process, not one per write."""
    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json={"items": []}))
    created = respx.post(f"{BASE}/v1/spaces").mock(
        httpx.Response(201, json={"id": "spc_new", "slug": "fresh", "name": "fresh"})
    )
    respx.post(f"{BASE}/v1/spaces/spc_new/memories").mock(
        httpx.Response(201, json={"memory": {"id": "mem_1", "content": "hi"}})
    )

    client.memories.add("one", space="fresh")
    client.memories.add("two", space="fresh")

    assert created.call_count == 1


@respx.mock
def test_search_returns_iterable_hits(client: Mapi) -> None:
    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    respx.post(f"{BASE}/v1/spaces/spc_ada/search").mock(
        httpx.Response(
            200,
            json={
                "query": "seat",
                "results": [
                    {"memory": {"id": "mem_1", "content": "window seats"}, "score": 0.9},
                    {"memory": {"id": "mem_2", "content": "aisle once"}, "score": 0.4},
                ],
            },
        )
    )
    hits = client.search.execute("seat", space="ada")
    assert len(hits) == 2
    assert [h.id for h in hits] == ["mem_1", "mem_2"]
    assert hits[0].content == "window seats"


@respx.mock
def test_errors_carry_the_request_id(client: Mapi) -> None:
    """The single most useful thing when reporting a production problem."""
    respx.get(f"{BASE}/v1/spaces").mock(
        httpx.Response(
            404,
            json={"code": "not_found", "detail": "no such space", "request_id": "req_9"},
        )
    )
    with pytest.raises(NotFoundError, match="req_9"):
        client.spaces.list()


@respx.mock
def test_a_bad_key_raises_authentication_error(client: Mapi) -> None:
    respx.get(f"{BASE}/v1/spaces").mock(
        httpx.Response(401, json={"code": "unauthorized", "detail": "bad key"})
    )
    with pytest.raises(AuthenticationError):
        client.spaces.list()


@respx.mock
def test_rate_limits_are_retried_then_raised(client: Mapi) -> None:
    route = respx.get(f"{BASE}/v1/spaces").mock(
        httpx.Response(429, headers={"retry-after": "0"}, json={"code": "rate_limited"})
    )
    with pytest.raises(RateLimitError):
        client.spaces.list()
    assert route.call_count == 3, "two retries then give up"


@respx.mock
def test_a_transient_gateway_error_recovers(client: Mapi) -> None:
    respx.get(f"{BASE}/v1/spaces").mock(
        side_effect=[
            httpx.Response(503, json={}),
            httpx.Response(200, json=SPACES),
        ]
    )
    assert [s.slug for s in client.spaces.list()] == ["ada"]


@respx.mock
def test_a_500_is_not_retried(client: Mapi) -> None:
    """A request that made the server throw will throw again; retrying hides it."""
    route = respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(500, json={}))
    with pytest.raises(MapiError):
        client.spaces.list()
    assert route.call_count == 1


@respx.mock
def test_a_non_json_error_body_does_not_mask_the_status(client: Mapi) -> None:
    """Proxies return HTML 502s. That must not surface as a decode error."""
    respx.get(f"{BASE}/v1/spaces").mock(
        side_effect=[httpx.Response(502, text="<html>bad gateway</html>")] * 3
    )
    with pytest.raises(MapiError) as caught:
        client.spaces.list()
    assert caught.value.status == 502


@respx.mock
def test_event_time_is_sent_when_given(client: Mapi) -> None:
    from datetime import UTC, datetime

    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    route = respx.post(f"{BASE}/v1/spaces/spc_ada/memories").mock(
        httpx.Response(201, json={"memory": {"id": "m", "content": "c"}})
    )
    client.memories.add(
        "ran a 5K",
        space="ada",
        occurred_at=datetime(2023, 5, 20, tzinfo=UTC),
    )
    body = route.calls[0].request.content.decode()
    assert "2023-05-20" in body
    # No behaviour flags travel: consolidation is automatic, and the server
    # forbids unknown fields, so sending one would be a 422 rather than a
    # no-op.
    for gone in ("extract", "dedupe", "auto_supersede", "detect_conflicts"):
        assert gone not in body


@respx.mock
def test_context_parses_the_whole_neighbourhood(client: Mapi) -> None:
    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    respx.get(f"{BASE}/v1/spaces/spc_ada/memories/mem_1/context").mock(
        httpx.Response(
            200,
            json={
                "memory": {"id": "mem_1", "content": "PB is 25:50"},
                "is_current": False,
                "current_head": [{"id": "mem_2", "content": "PB is 24:10"}],
                "replaced": [{"id": "mem_0", "content": "PB is 27:12"}],
            },
        )
    )
    ctx = client.memories.context("mem_1", space="ada")
    assert ctx.is_current is False
    assert ctx.current_head[0].content == "PB is 24:10"
    assert ctx.replaced[0].content == "PB is 27:12"


@respx.mock
def test_the_key_is_sent_as_a_bearer_token(client: Mapi) -> None:
    route = respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    client.spaces.list()
    assert route.calls[0].request.headers["authorization"] == "Bearer sm_test"


# -- async: same surface, awaited ---------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_client_mirrors_the_sync_one() -> None:
    from mapi_sdk import AsyncMapi

    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    respx.post(f"{BASE}/v1/spaces/spc_ada/memories").mock(
        httpx.Response(201, json={"memory": {"id": "mem_a", "content": "hi"}})
    )
    respx.post(f"{BASE}/v1/spaces/spc_ada/search").mock(
        httpx.Response(
            200,
            json={"results": [{"memory": {"id": "mem_a", "content": "hi"}, "score": 0.7}]},
        )
    )
    async with AsyncMapi(api_key="sm_test", base_url=BASE) as client:
        memory = await client.memories.add("hi", space="ada")
        assert memory.id == "mem_a"
        hits = await client.search.execute("hi", space="ada")
        assert hits[0].score == 0.7


@pytest.mark.asyncio
@respx.mock
async def test_async_errors_are_the_same_types() -> None:
    from mapi_sdk import AsyncMapi

    respx.get(f"{BASE}/v1/spaces").mock(
        httpx.Response(401, json={"code": "unauthorized", "detail": "bad key"})
    )
    async with AsyncMapi(api_key="sm_test", base_url=BASE) as client:
        with pytest.raises(AuthenticationError):
            await client.spaces.list()


def test_sync_and_async_expose_exactly_the_same_methods() -> None:
    """A method on one and not the other is drift, and this is where it shows.

    No allow-list: an exemption here is how the async client quietly falls a
    release behind, which is the failure mode this whole split invites.
    """
    from mapi_sdk import _resources

    for name in ("Memories", "Search", "Spaces", "Graph"):
        sync = {n for n in dir(getattr(_resources, name)) if not n.startswith("_")}
        asynchronous = {
            n for n in dir(getattr(_resources, "Async" + name)) if not n.startswith("_")
        }
        assert sync == asynchronous, (
            f"{name}: sync-only {sorted(sync - asynchronous)}, "
            f"async-only {sorted(asynchronous - sync)}"
        )


@respx.mock
def test_a_write_reports_what_it_did(client: Mapi) -> None:
    """A bare Memory threw the consolidation outcome away.

    A write that quietly retired last month's answer is the single most
    important thing a memory API can tell a caller, and it was not reaching
    them: `add` returned only the stored row.
    """
    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    respx.post(f"{BASE}/v1/spaces/spc_ada/memories").mock(
        httpx.Response(
            201,
            json={
                "memory": {"id": "mem_new", "content": "I live in Madrid now."},
                "created": True,
                "superseded": ["mem_old"],
                "contradicts": [],
                "supersede_declined": ["mem_maybe"],
                "chunk_count": 1,
            },
        )
    )

    result = client.memories.add("I live in Madrid now.", space="ada")

    assert result.id == "mem_new"
    assert result.content == "I live in Madrid now."
    assert result.superseded == ["mem_old"]
    assert result.supersede_declined == ["mem_maybe"]
    assert result.created


@respx.mock
def test_a_collapsed_duplicate_says_so(client: Mapi) -> None:
    respx.get(f"{BASE}/v1/spaces").mock(httpx.Response(200, json=SPACES))
    respx.post(f"{BASE}/v1/spaces/spc_ada/memories").mock(
        httpx.Response(
            200,
            json={
                "memory": {"id": "mem_existing", "content": "same thing"},
                "created": False,
                "duplicate_of": "mem_existing",
                "duplicate_kind": "near",
                "similarity": 0.981,
            },
        )
    )

    result = client.memories.add("same thing", space="ada")

    assert not result.created
    assert result.duplicate_of == "mem_existing"
    assert result.duplicate_kind == "near"
    assert result.similarity == pytest.approx(0.981)
