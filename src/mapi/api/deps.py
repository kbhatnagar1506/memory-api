"""Request-scoped dependencies: authentication, scope checks, rate limiting.

Authentication resolves an API key to a principal once per request and caches it
on `request.state`, so a handler needing both the org id and a scope check does
not hash the key twice.

Across requests, `AuthCache` keeps the resolved principal for a short TTL and
writes `last_used_at` at most once a minute per key. Both used to happen on
every request: one lookup session plus one UPDATE, and with a single client
holding a single key that UPDATE is a hot row every request queues on.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from ..config import Settings
from ..core.errors import ForbiddenError, RateLimitedError, UnauthorizedError
from ..core.logging import org_id_var
from ..core.metrics import RATE_LIMITED
from ..core.security import hash_key, looks_like_key, parse_authorization
from ..domain.models import ApiKey, Scope, utcnow
from ..service import MemoryService
from ..store.base import MemoryStore


@dataclass(frozen=True, slots=True)
class Principal:
    org_id: str
    key_id: str
    scopes: frozenset[Scope]

    def require(self, scope: Scope) -> None:
        if Scope.ADMIN not in self.scopes and scope not in self.scopes:
            raise ForbiddenError(
                f"this key lacks the {scope.value!r} scope",
                extra={"required_scope": scope.value},
            )


@dataclass(frozen=True, slots=True)
class _CachedPrincipal:
    principal: Principal
    #: The key's own expiry, re-checked on every hit: a cached entry must not
    #: outlive the key it stands for by up to a TTL.
    expires_at: datetime | None
    cached_at: float


class AuthCache:
    """Resolved principals and last-touch times, for ONE app in ONE process.

    Per app rather than per module: tests build many apps over many stores,
    and a key hash cached for one would authenticate against another.

    Revocation through this process evicts at once (`invalidate_key`).
    Another process -- a separate read and write deployment -- keeps serving
    a revoked key for at most `ttl_s`; that bound is the price of the cache,
    and it is configurable down to zero.
    """

    #: Plenty for any real key population; a bound so that a flood of valid
    #: keys cannot grow the process without limit.
    MAX_ENTRIES = 10_000

    def __init__(self, *, ttl_s: float, touch_interval_s: float) -> None:
        self.ttl_s = ttl_s
        self.touch_interval_s = touch_interval_s
        self._entries: OrderedDict[str, _CachedPrincipal] = OrderedDict()
        self._last_touch: dict[str, float] = {}
        #: Strong references to in-flight touches. The event loop keeps only
        #: weak ones, and a fire-and-forget task nobody holds can be collected
        #: mid-flight.
        self._pending: set[asyncio.Task[None]] = set()

    def get(self, key_hash: str) -> Principal | None:
        if self.ttl_s <= 0:
            return None
        entry = self._entries.get(key_hash)
        if entry is None:
            return None
        if time.monotonic() - entry.cached_at >= self.ttl_s or (
            entry.expires_at is not None and entry.expires_at <= utcnow()
        ):
            del self._entries[key_hash]
            return None
        return entry.principal

    def put(self, key_hash: str, principal: Principal, expires_at: datetime | None) -> None:
        if self.ttl_s <= 0:
            return
        self._entries[key_hash] = _CachedPrincipal(principal, expires_at, time.monotonic())
        self._entries.move_to_end(key_hash)
        while len(self._entries) > self.MAX_ENTRIES:
            self._entries.popitem(last=False)

    def invalidate_key(self, key_id: str) -> None:
        """Forget every cached principal for `key_id`. Called on revoke."""
        for key_hash in [h for h, e in self._entries.items() if e.principal.key_id == key_id]:
            del self._entries[key_hash]
        self._last_touch.pop(key_id, None)

    def touch(self, store: MemoryStore, key_id: str) -> None:
        """Record last use, at most once per `touch_interval_s`, off the request.

        Fire-and-forget: the request never waits on the write, and a failed
        write is dropped -- last-used tracking must never fail a request, which
        the store contract already says and this keeps true.
        """
        now = time.monotonic()
        last = self._last_touch.get(key_id)
        if last is not None and now - last < self.touch_interval_s:
            return
        self._last_touch[key_id] = now
        task = asyncio.get_running_loop().create_task(_touch(store, key_id))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Wait for in-flight touches. For tests and orderly shutdown."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)


async def _touch(store: MemoryStore, key_id: str) -> None:
    with contextlib.suppress(Exception):
        await store.touch_api_key(key_id, utcnow())


def auth_cache(app: FastAPI) -> AuthCache:
    """This app's cache, created on first use from its settings."""
    cache: AuthCache | None = getattr(app.state, "auth_cache", None)
    if cache is None:
        settings: Settings = app.state.settings
        cache = AuthCache(
            ttl_s=settings.auth_cache_ttl_s,
            touch_interval_s=settings.touch_api_key_interval_s,
        )
        app.state.auth_cache = cache
    return cache


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_store(request: Request) -> MemoryStore:
    return request.app.state.store  # type: ignore[no-any-return]


def get_service(request: Request) -> MemoryService:
    return request.app.state.service  # type: ignore[no-any-return]


async def authenticate(request: Request) -> Principal:
    cached = getattr(request.state, "principal", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    started = time.perf_counter()
    settings: Settings = request.app.state.settings
    store: MemoryStore = request.app.state.store
    cache = auth_cache(request.app)

    raw = parse_authorization(
        request.headers.get("authorization") or request.headers.get("x-api-key")
    )
    if not raw:
        raise UnauthorizedError(
            "provide an API key via 'Authorization: Bearer <key>' or 'X-API-Key'"
        )
    # Reject obviously malformed keys before touching the database.
    if not looks_like_key(raw):
        raise UnauthorizedError("malformed API key")

    key_hash = hash_key(raw, settings.api_key_pepper)
    principal = cache.get(key_hash)
    if principal is None:
        record: ApiKey | None = await store.get_api_key_by_hash(key_hash)
        # Identical message and status for unknown, revoked and expired keys: the
        # difference between them is information an attacker can use.
        if record is None or not record.is_active():
            raise UnauthorizedError("invalid or expired API key")
        principal = Principal(
            org_id=record.org_id, key_id=record.id, scopes=frozenset(record.scopes)
        )
        # Only successes are cached. A miss is always re-asked, so a key minted
        # a moment ago works on its first request, and an attacker guessing
        # keys gets no cheaper answer than before.
        cache.put(key_hash, principal, record.expires_at)

    request.state.principal = principal
    org_id_var.set(principal.org_id)
    cache.touch(store, principal.key_id)
    # Reported in search timings: the part of a request's latency spent
    # before any handler code ran.
    request.state.auth_ms = (time.perf_counter() - started) * 1000
    return principal


async def enforce_rate_limit(
    request: Request, principal: Annotated[Principal, Depends(authenticate)]
) -> Principal:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return principal
    decision = await limiter.check(principal.org_id)
    request.state.rate_limit = decision
    if not decision.allowed:
        RATE_LIMITED.inc()
        raise RateLimitedError(
            f"rate limit of {limiter.per_minute}/min exceeded for this organization",
            retry_after=decision.retry_after,
        )
    return principal


def require_scope(scope: Scope) -> Callable[..., Awaitable[Principal]]:
    """Dependency factory: authenticate, rate limit, then check one scope."""

    async def _dep(
        principal: Annotated[Principal, Depends(enforce_rate_limit)],
    ) -> Principal:
        principal.require(scope)
        return principal

    return _dep


CurrentPrincipal = Annotated[Principal, Depends(enforce_rate_limit)]
ServiceDep = Annotated[MemoryService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
StoreDep = Annotated[MemoryStore, Depends(get_store)]

__all__ = [
    "AuthCache",
    "CurrentPrincipal",
    "Principal",
    "ServiceDep",
    "SettingsDep",
    "StoreDep",
    "auth_cache",
    "authenticate",
    "enforce_rate_limit",
    "get_service",
    "get_settings_dep",
    "get_store",
    "require_scope",
]
