"""Spaces by slug, and a listing whose cost does not grow with the org.

`PUT /v1/spaces/by-slug/{slug}` is how a client that names its spaces itself
(one per attendee, one `directory` per event) gets "the space called X"
without listing everything and racing other workers to create it.
"""

from __future__ import annotations

import asyncio

import httpx

from mapi.core.security import build_api_key
from mapi.domain.models import Organization, Scope
from mapi.store.memory import InMemoryStore


async def _client_for(
    app_context, *, org_id: str, scopes: frozenset[Scope]
) -> httpx.AsyncClient:
    record, plaintext = build_api_key(
        org_id=org_id, name="scoped", pepper="test-pepper", scopes=scopes
    )
    await app_context["store"].create_api_key(record)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_context["app"]),
        base_url="http://t",
        headers={"Authorization": f"Bearer {plaintext}"},
    )


async def test_put_by_slug_creates_then_returns_the_same_space(client) -> None:
    first = await client.put(
        "/v1/spaces/by-slug/attendee-42", json={"name": "Attendee 42", "metadata": {"u": 42}}
    )
    assert first.status_code == 201
    created = first.json()
    assert created["slug"] == "attendee-42"
    assert created["name"] == "Attendee 42"
    assert created["metadata"] == {"u": 42}
    assert created["memory_count"] == 0

    again = await client.put("/v1/spaces/by-slug/attendee-42", json={"name": "Renamed?"})
    assert again.status_code == 200
    assert again.json()["id"] == created["id"]
    # Create-or-GET: an existing space is returned as it is, never updated.
    assert again.json()["name"] == "Attendee 42"


async def test_put_without_a_body_names_the_space_after_its_slug(client) -> None:
    response = await client.put("/v1/spaces/by-slug/directory")
    assert response.status_code == 201
    assert response.json()["name"] == "directory"


async def test_existing_space_reports_its_memory_count(client) -> None:
    space = (await client.put("/v1/spaces/by-slug/counted")).json()
    await client.post(f"/v1/spaces/{space['id']}/memories", json={"content": "one fact"})
    again = await client.put("/v1/spaces/by-slug/counted")
    assert again.json()["memory_count"] == 1


async def test_concurrent_puts_end_with_one_space(client) -> None:
    responses = await asyncio.gather(
        *(client.put("/v1/spaces/by-slug/racing") for _ in range(12))
    )
    assert {r.status_code for r in responses} <= {200, 201}
    assert [r.status_code for r in responses].count(201) == 1
    assert len({r.json()["id"] for r in responses}) == 1


async def test_losing_the_create_race_reads_the_winner(
    app_context, client, monkeypatch
) -> None:
    """The branch `asyncio.gather` cannot reach on a store with one lock.

    The loser's lookup saw nothing, then its insert hit the unique slug. It
    must answer with the space that won, not a 409.
    """
    store: InMemoryStore = app_context["store"]
    winner = (await client.put("/v1/spaces/by-slug/contested")).json()

    original = store.get_space_by_slug
    calls = 0

    async def _stale_first(org_id: str, slug: str):
        nonlocal calls
        calls += 1
        return None if calls == 1 else await original(org_id, slug)

    monkeypatch.setattr(store, "get_space_by_slug", _stale_first)
    response = await client.put("/v1/spaces/by-slug/contested")
    assert response.status_code == 200
    assert response.json()["id"] == winner["id"]


async def test_a_malformed_slug_is_refused(client) -> None:
    for bad in ("Upper", "-leading", "has space", "x" * 65):
        response = await client.put(f"/v1/spaces/by-slug/{bad}")
        assert response.status_code == 422, bad


async def test_put_needs_spaces_write_and_get_needs_spaces_read(app_context, client) -> None:
    await client.put("/v1/spaces/by-slug/shared")
    reader = await _client_for(
        app_context, org_id=app_context["org"].id, scopes=frozenset({Scope.SPACES_READ})
    )
    async with reader:
        denied = await reader.put("/v1/spaces/by-slug/shared")
        assert denied.status_code == 403
        assert denied.json()["required_scope"] == "spaces:write"
        assert (await reader.get("/v1/spaces/by-slug/shared")).status_code == 200


async def test_slugs_are_per_organization(app_context, client) -> None:
    ours = (await client.put("/v1/spaces/by-slug/directory")).json()
    other_org = await app_context["store"].create_organization(Organization(name="Other"))
    other = await _client_for(app_context, org_id=other_org.id, scopes=frozenset(Scope.all()))
    async with other:
        # The same slug in another org is a different space, not ours...
        theirs = await other.put("/v1/spaces/by-slug/directory")
        assert theirs.status_code == 201
        assert theirs.json()["id"] != ours["id"]
        # ...and a slug that exists only in our org is a plain 404 to them.
        await client.put("/v1/spaces/by-slug/only-ours")
        assert (await other.get("/v1/spaces/by-slug/only-ours")).status_code == 404


async def test_get_by_slug(client) -> None:
    assert (await client.get("/v1/spaces/by-slug/nothing-here")).status_code == 404
    created = (await client.put("/v1/spaces/by-slug/found")).json()
    fetched = await client.get("/v1/spaces/by-slug/found")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == created["id"]


async def test_listing_counts_with_one_grouped_query(app_context, client, monkeypatch) -> None:
    """The listing used to call `count_memories` once per space."""
    store: InMemoryStore = app_context["store"]
    a = (await client.put("/v1/spaces/by-slug/list-a")).json()
    b = (await client.put("/v1/spaces/by-slug/list-b")).json()
    for text in ("alpha one", "alpha two"):
        await client.post(f"/v1/spaces/{a['id']}/memories", json={"content": text})
    await client.post(f"/v1/spaces/{b['id']}/memories", json={"content": "beta one"})

    per_space = 0
    grouped = 0
    original_count = store.count_memories
    original_grouped = store.count_memories_by_space

    async def _count(*args, **kwargs):
        nonlocal per_space
        per_space += 1
        return await original_count(*args, **kwargs)

    async def _grouped(org_id: str):
        nonlocal grouped
        grouped += 1
        return await original_grouped(org_id)

    monkeypatch.setattr(store, "count_memories", _count)
    monkeypatch.setattr(store, "count_memories_by_space", _grouped)
    listing = (await client.get("/v1/spaces")).json()["items"]

    counts = {item["slug"]: item["memory_count"] for item in listing}
    assert counts["list-a"] == 2
    assert counts["list-b"] == 1
    assert counts["default"] == 0
    assert grouped == 1
    assert per_space == 0
