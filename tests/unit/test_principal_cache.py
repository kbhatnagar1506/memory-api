"""Auth and space-lookup caching on the request path (B6).

Every request used to authenticate by asking the store for the key by hash and
then UPDATE-ing the key's `last_used_at`, and every read and write then asked
the store for the space again. With one client holding one key -- the facemash
deployment -- that is two extra sessions and a hot-row write on every request,
for answers that change about once an event.

The caches are only worth having if they are invisible in the ways that matter,
so most of this file is about what they must NOT change: a revoked key stops
working at once in the process that revoked it, an expired key is refused even
while cached, a deleted space is gone, and nothing is shared between apps.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import httpx
import pytest

from mapi.api.deps import AuthCache, Principal, auth_cache
from mapi.config import Settings
from mapi.core.security import build_api_key
from mapi.domain.models import Organization, Scope, Space, utcnow
from mapi.main import create_app

PEPPER = "test-pepper"


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "test",
        "store_backend": "memory",
        "embedding_backend": "deterministic",
        "embedding_dimensions": 128,
        "api_key_pepper": PEPPER,
        "rate_limit_per_minute": 100_000,
        "rate_limit_burst": 100_000,
    }
    base.update(overrides)
    return Settings(**base)


class _Counter:
    """Wraps one store method and counts its calls."""

    def __init__(self, store: Any, name: str) -> None:
        self.calls = 0
        self._original = getattr(store, name)
        setattr(store, name, self)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return await self._original(*args, **kwargs)


async def _wired(settings: Settings) -> AsyncIterator[dict[str, Any]]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        store = app.state.store
        org = await store.create_organization(Organization(name="Cache Org"))
        space = await store.create_space(Space(org_id=org.id, slug="default", name="Default"))
        record, plaintext = build_api_key(
            org_id=org.id, name="test", pepper=PEPPER, scopes=frozenset(Scope.all())
        )
        await store.create_api_key(record)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": f"Bearer {plaintext}"},
        ) as client:
            yield {
                "app": app,
                "store": store,
                "org": org,
                "space": space,
                "key": record,
                "client": client,
            }


@pytest.fixture
async def wired() -> AsyncIterator[dict[str, Any]]:
    async for ctx in _wired(_settings()):
        yield ctx


@pytest.fixture
async def uncached() -> AsyncIterator[dict[str, Any]]:
    async for ctx in _wired(
        _settings(auth_cache_ttl_s=0, touch_api_key_interval_s=0, space_cache_ttl_s=0)
    ):
        yield ctx


# -- the principal cache ---------------------------------------------------------


async def test_a_second_request_does_not_look_the_key_up_again(wired) -> None:
    lookups = _Counter(wired["store"], "get_api_key_by_hash")
    for _ in range(5):
        assert (await wired["client"].get("/v1/spaces")).status_code == 200
    assert lookups.calls == 1


async def test_ttl_zero_restores_a_lookup_per_request(uncached) -> None:
    lookups = _Counter(uncached["store"], "get_api_key_by_hash")
    for _ in range(3):
        assert (await uncached["client"].get("/v1/spaces")).status_code == 200
    assert lookups.calls == 3


async def test_revoking_through_the_api_takes_effect_immediately(wired) -> None:
    """The case the cache must never delay: revocation in this process."""
    client = wired["client"]
    created = (
        await client.post("/v1/keys", json={"name": "doomed", "scopes": ["spaces:read"]})
    ).json()
    headers = {"Authorization": f"Bearer {created['plaintext']}"}
    # Warm the cache for the doomed key.
    assert (await client.get("/v1/spaces", headers=headers)).status_code == 200
    assert (await client.get("/v1/spaces", headers=headers)).status_code == 200

    assert (await client.delete(f"/v1/keys/{created['key']['id']}")).status_code == 204
    assert (await client.get("/v1/spaces", headers=headers)).status_code == 401


async def test_revoking_elsewhere_is_bounded_by_the_ttl(wired, monkeypatch) -> None:
    """Another process revoking is seen once the entry ages out, and not before."""
    client, store, key = wired["client"], wired["store"], wired["key"]
    assert (await client.get("/v1/spaces")).status_code == 200
    # Revoke behind this process's back, as the CLI or another replica would.
    assert await store.revoke_api_key(key.org_id, key.id)
    assert (await client.get("/v1/spaces")).status_code == 200

    cache = auth_cache(wired["app"])
    real = __import__("time").monotonic
    monkeypatch.setattr("mapi.api.deps.time.monotonic", lambda: real() + cache.ttl_s + 1)
    assert (await client.get("/v1/spaces")).status_code == 401


async def test_an_expired_key_is_refused_even_while_cached(wired, monkeypatch) -> None:
    """The cache TTL has not passed; the key's own lifetime has."""
    store, org = wired["store"], wired["org"]
    record, plaintext = build_api_key(
        org_id=org.id,
        name="short-lived",
        pepper=PEPPER,
        scopes=frozenset({Scope.SPACES_READ}),
        expires_at=utcnow() + timedelta(seconds=30),
    )
    await store.create_api_key(record)
    headers = {"Authorization": f"Bearer {plaintext}"}
    assert (await wired["client"].get("/v1/spaces", headers=headers)).status_code == 200

    later = utcnow() + timedelta(seconds=60)
    monkeypatch.setattr("mapi.api.deps.utcnow", lambda: later)
    monkeypatch.setattr("mapi.domain.models.utcnow", lambda: later)
    assert (await wired["client"].get("/v1/spaces", headers=headers)).status_code == 401


def test_a_cached_entry_past_its_keys_expiry_is_a_miss() -> None:
    cache = AuthCache(ttl_s=30, touch_interval_s=60)
    principal = Principal(org_id="o", key_id="k", scopes=frozenset())
    cache.put("h", principal, utcnow() - timedelta(seconds=1))
    assert cache.get("h") is None
    cache.put("h", principal, utcnow() + timedelta(seconds=60))
    assert cache.get("h") == principal


async def test_unknown_keys_are_never_cached(wired) -> None:
    lookups = _Counter(wired["store"], "get_api_key_by_hash")
    record, plaintext = build_api_key(
        org_id=wired["org"].id, name="late", pepper=PEPPER, scopes=frozenset(Scope.all())
    )
    headers = {"Authorization": f"Bearer {plaintext}"}
    assert (await wired["client"].get("/v1/spaces", headers=headers)).status_code == 401
    # Minted after the miss: it must work on its very next request.
    await wired["store"].create_api_key(record)
    assert (await wired["client"].get("/v1/spaces", headers=headers)).status_code == 200
    assert lookups.calls == 2


async def test_caches_are_per_app() -> None:
    """A key hash cached for one app must not authenticate against another.

    Tests build many apps over many stores, and two of them can hold the same
    literal key -- a module-level cache would hand one app's org to the other.
    """
    first = _settings()
    second = _settings()
    apps = [create_app(first), create_app(second)]
    assert auth_cache(apps[0]) is not auth_cache(apps[1])
    assert auth_cache(apps[0]) is auth_cache(apps[0])


# -- last_used_at, throttled and off the request -----------------------------------


async def test_last_used_is_written_at_most_once_per_interval(wired) -> None:
    touches = _Counter(wired["store"], "touch_api_key")
    for _ in range(10):
        assert (await wired["client"].get("/v1/spaces")).status_code == 200
    await auth_cache(wired["app"]).drain()
    assert touches.calls == 1


async def test_last_used_is_still_recorded(wired) -> None:
    assert (await wired["client"].get("/v1/spaces")).status_code == 200
    await auth_cache(wired["app"]).drain()
    keys = (await wired["client"].get("/v1/keys")).json()["items"]
    assert any(k["last_used_at"] is not None for k in keys)


async def test_interval_zero_touches_every_request(uncached) -> None:
    touches = _Counter(uncached["store"], "touch_api_key")
    for _ in range(4):
        assert (await uncached["client"].get("/v1/spaces")).status_code == 200
    await auth_cache(uncached["app"]).drain()
    assert touches.calls == 4


async def test_a_failing_touch_never_fails_the_request(wired) -> None:
    async def broken(*_: Any, **__: Any) -> None:
        raise RuntimeError("database on fire")

    wired["store"].touch_api_key = broken
    assert (await wired["client"].get("/v1/spaces")).status_code == 200
    await auth_cache(wired["app"]).drain()


def test_the_cache_is_bounded() -> None:
    cache = AuthCache(ttl_s=30, touch_interval_s=60)
    cache.MAX_ENTRIES = 3  # type: ignore[misc]
    for i in range(5):
        cache.put(f"h{i}", Principal(org_id="o", key_id=f"k{i}", scopes=frozenset()), None)
    assert list(cache._entries) == ["h2", "h3", "h4"]


def test_invalidate_drops_every_entry_for_the_key() -> None:
    cache = AuthCache(ttl_s=30, touch_interval_s=60)
    principal = Principal(org_id="o", key_id="k1", scopes=frozenset())
    cache.put("a", principal, None)
    cache.put("b", Principal(org_id="o", key_id="k2", scopes=frozenset()), None)
    cache.invalidate_key("k1")
    assert cache.get("a") is None
    assert cache.get("b") is not None


# -- auth time is visible ------------------------------------------------------------


async def test_search_timings_report_auth_and_total(wired) -> None:
    space_id = wired["space"].id
    await wired["client"].post(
        f"/v1/spaces/{space_id}/memories", json={"content": "the build uses bazel"}
    )
    body = (
        await wired["client"].post(f"/v1/spaces/{space_id}/search", json={"query": "bazel"})
    ).json()
    timings = body["timings_ms"]
    for stage in ("auth_ms", "space_ms", "total_ms", "superseders_ms"):
        assert stage in timings, f"missing {stage}: {sorted(timings)}"
        assert timings[stage] >= 0


# -- the space lookup cache ------------------------------------------------------------


async def test_repeated_searches_look_the_space_up_once(wired) -> None:
    lookups = _Counter(wired["store"], "get_space")
    space_id = wired["space"].id
    for _ in range(4):
        response = await wired["client"].post(
            f"/v1/spaces/{space_id}/search", json={"query": "anything"}
        )
        assert response.status_code == 200
    assert lookups.calls == 1


async def test_space_ttl_zero_looks_up_every_time(uncached) -> None:
    lookups = _Counter(uncached["store"], "get_space")
    space_id = uncached["space"].id
    for _ in range(3):
        await uncached["client"].post(f"/v1/spaces/{space_id}/search", json={"query": "x"})
    assert lookups.calls == 3


async def test_deleting_a_space_evicts_it(wired) -> None:
    client = wired["client"]
    space_id = (await client.post("/v1/spaces", json={"slug": "gone", "name": "Gone"})).json()[
        "id"
    ]
    assert (
        await client.post(f"/v1/spaces/{space_id}/search", json={"query": "x"})
    ).status_code == 200
    assert (await client.delete(f"/v1/spaces/{space_id}")).status_code == 204
    after = await client.post(f"/v1/spaces/{space_id}/search", json={"query": "x"})
    assert after.status_code == 404


async def test_a_missing_space_is_not_cached(wired) -> None:
    """A space reported missing and then created must be found at once."""
    service, store, org = wired["app"].state.service, wired["store"], wired["org"]
    space = Space(org_id=org.id, slug="later", name="Later")
    from mapi.core.errors import NotFoundError

    with pytest.raises(NotFoundError):
        await service.get_space_or_raise(org.id, space.id)
    await store.create_space(space)
    assert (await service.get_space_or_raise(org.id, space.id)).id == space.id


async def test_the_space_cache_cannot_cross_orgs(wired) -> None:
    service, store = wired["app"].state.service, wired["store"]
    other = await store.create_organization(Organization(name="Other"))
    space_id = wired["space"].id
    await service.get_space_or_raise(wired["org"].id, space_id)  # cached for its owner
    from mapi.core.errors import NotFoundError

    with pytest.raises(NotFoundError):
        await service.get_space_or_raise(other.id, space_id)


async def test_chat_checks_the_space_once(wired) -> None:
    """The duplicate lookup chat used to make before its own search."""
    service = wired["app"].state.service

    async def complete(prompt: str) -> str:
        return "nothing to say [1]"

    service.completer = complete
    service.settings = service.settings.model_copy(update={"space_cache_ttl_s": 0})
    lookups = _Counter(wired["store"], "get_space")
    await service.chat(wired["org"].id, wired["space"].id, message="hello")
    assert lookups.calls == 1


async def test_touches_do_not_outlive_their_loop(wired) -> None:
    """Fire-and-forget still finishes: nothing pending after a drain."""
    assert (await wired["client"].get("/v1/spaces")).status_code == 200
    cache = auth_cache(wired["app"])
    await cache.drain()
    assert not cache._pending
    await asyncio.sleep(0)
