"""Boot warm-up and keep-warm (`mapi.warm`).

What these pin: every configured dependency is warmed through the client the
request path uses, a dependency that raises or hangs never takes boot or the
loop down with it, the pool warm-up really opens N DISTINCT connections, the
keep-warm loop never takes a connection a request is waiting for, and the
embedding warm-up never calls an embedding endpoint (no tokens spent).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from mapi.config import Settings
from mapi.core.ratelimit import InMemoryRateLimiter, RedisRateLimiter
from mapi.domain.embeddings import DeterministicEmbedder, build_embedder
from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.main import create_app
from mapi.store.memory import InMemoryStore
from mapi.warm import Warmer

DIMS = 8


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "test",
        "store_backend": "memory",
        "embedding_backend": "deterministic",
        "embedding_dimensions": DIMS,
        "warm_timeout_s": 0.2,
    }
    base.update(overrides)
    return Settings(**base)


# -- fakes -----------------------------------------------------------------------


class _FakeConn:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine

    async def execute(self, _statement: Any) -> None:
        self.engine.selects += 1


class _FakePool:
    def __init__(self, checked_out: int = 0, checked_in: int = 0) -> None:
        self._out, self._in = checked_out, checked_in

    def checkedout(self) -> int:
        return self._out

    def checkedin(self) -> int:
        return self._in


class _FakeEngine:
    """Counts connections held at the same time, like a pool would open them."""

    def __init__(self, pool: _FakePool | None = None, fail: bool = False) -> None:
        self.pool = pool or _FakePool()
        self.fail = fail
        self.open = 0
        self.peak = 0
        self.selects = 0

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[_FakeConn]:
        if self.fail:
            raise ConnectionRefusedError("no database")
        self.open += 1
        self.peak = max(self.peak, self.open)
        try:
            await asyncio.sleep(0)
            yield _FakeConn(self)
        finally:
            self.open -= 1


class _EngineStore(InMemoryStore):
    """An in-memory store that exposes a pool engine the way PostgresStore does."""

    def __init__(self, engine: _FakeEngine) -> None:
        super().__init__()
        self._engine = engine


class _RaisingStore(InMemoryStore):
    async def ping(self) -> bool:
        raise RuntimeError("database is on fire")


class _HangingStore(InMemoryStore):
    async def ping(self) -> bool:
        await asyncio.sleep(30)
        return True


class _FakeRedis:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.pings = 0

    async def ping(self) -> bool:
        self.pings += 1
        if self.error is not None:
            raise self.error
        return True


def _warmer(
    *,
    store: Any = None,
    rate_limiter: Any = None,
    embedder: Any = None,
    **settings: Any,
) -> Warmer:
    return Warmer(
        store=store if store is not None else InMemoryStore(),
        rate_limiter=rate_limiter or InMemoryRateLimiter(per_minute=60, burst=10),
        embedder=embedder or DeterministicEmbedder(dimensions=DIMS),
        settings=_settings(**settings),
    )


# -- boot warm-up: success -------------------------------------------------------


async def test_in_memory_backends_warm_the_store_and_nothing_else() -> None:
    warmer = _warmer()
    assert set(warmer.steps()) == {"db"}  # no Redis, deterministic embedder
    results = await warmer.warm_once()
    assert results["db"].ok is True
    assert results["db"].latency_ms >= 0


async def test_boot_opens_n_distinct_pool_connections_at_once() -> None:
    engine = _FakeEngine()
    warmer = _warmer(store=_EngineStore(engine), warm_db_connections=4)
    results = await warmer.warm_once()
    assert results["db"].ok is True
    assert results["db"].detail == "4/4 connections"
    # Held together, so the pool opened four, not one reused four times.
    assert engine.peak == 4
    assert engine.selects == 4
    assert engine.open == 0  # all released back to the pool


async def test_warm_connections_are_clipped_to_the_pool_size() -> None:
    engine = _FakeEngine()
    warmer = _warmer(store=_EngineStore(engine), warm_db_connections=50, db_pool_size=3)
    await warmer.warm_once()
    assert engine.peak == 3


async def test_zero_connections_skips_the_database() -> None:
    assert "db" not in _warmer(warm_db_connections=0).steps()


async def test_redis_is_pinged_through_the_rate_limiters_own_client() -> None:
    redis = _FakeRedis()
    limiter = RedisRateLimiter(redis, per_minute=60, burst=10)
    warmer = _warmer(rate_limiter=limiter)
    results = await warmer.warm_once()
    assert results["redis"].ok is True
    assert redis.pings == 1


async def test_start_runs_concurrently_and_stop_cancels_the_loop() -> None:
    redis = _FakeRedis()
    warmer = _warmer(rate_limiter=RedisRateLimiter(redis, per_minute=60, burst=10))
    await warmer.start()
    assert set(warmer.results) == {"db", "redis"}
    assert warmer._task is not None and not warmer._task.done()
    task = warmer._task
    await warmer.stop()
    assert task.done()
    await warmer.stop()  # idempotent


async def test_disabled_does_nothing() -> None:
    warmer = _warmer(warm_enabled=False)
    await warmer.start()
    assert warmer.results == {}
    assert warmer._task is None


# -- boot warm-up: failure tolerance ---------------------------------------------


async def test_a_dependency_that_raises_does_not_crash_boot() -> None:
    redis = _FakeRedis(error=ConnectionError("redis://user:secret@host"))
    warmer = _warmer(
        store=_RaisingStore(), rate_limiter=RedisRateLimiter(redis, per_minute=60, burst=10)
    )
    await warmer.start()  # must not raise
    try:
        assert warmer.results["db"].ok is False
        assert warmer.results["db"].detail == "RuntimeError"
        assert warmer.results["redis"].ok is False
        # The type name only: an exception message can carry a URL with a secret.
        assert warmer.results["redis"].detail == "ConnectionError"
        assert "secret" not in str(warmer.snapshot())
    finally:
        await warmer.stop()


async def test_a_dependency_that_hangs_is_bounded_by_the_timeout() -> None:
    warmer = _warmer(store=_HangingStore(), warm_timeout_s=0.05)
    started = time.perf_counter()
    results = await warmer.warm_once()
    assert time.perf_counter() - started < 2
    assert results["db"].ok is False
    assert results["db"].detail == "timeout"


async def test_a_pool_that_cannot_connect_reports_rather_than_raises() -> None:
    warmer = _warmer(store=_EngineStore(_FakeEngine(fail=True)))
    results = await warmer.warm_once()
    assert results["db"].ok is False
    assert results["db"].detail == "ConnectionRefusedError"


# -- keep-warm -----------------------------------------------------------------------


async def test_keep_warm_touches_only_idle_connections_under_traffic() -> None:
    engine = _FakeEngine(pool=_FakePool(checked_out=3, checked_in=1))
    warmer = _warmer(store=_EngineStore(engine), warm_db_connections=4)
    results = await warmer.warm_once(boot=False)
    assert results["db"].detail == "1/1 connections"
    assert engine.peak == 1


async def test_keep_warm_leaves_a_fully_busy_pool_alone() -> None:
    engine = _FakeEngine(pool=_FakePool(checked_out=10, checked_in=0))
    warmer = _warmer(store=_EngineStore(engine))
    results = await warmer.warm_once(boot=False)
    assert results["db"].ok is True
    assert results["db"].detail == "pool busy"
    assert engine.peak == 0


async def test_keep_warm_refills_an_idle_pool() -> None:
    engine = _FakeEngine(pool=_FakePool(checked_out=0, checked_in=1))
    warmer = _warmer(store=_EngineStore(engine), warm_db_connections=4)
    await warmer.warm_once(boot=False)
    assert engine.peak == 4


async def test_the_loop_survives_failing_rounds_and_keeps_going() -> None:
    redis = _FakeRedis(error=ConnectionError("down"))
    warmer = _warmer(
        rate_limiter=RedisRateLimiter(redis, per_minute=60, burst=10),
        keep_warm_interval_s=0.01,
        keep_warm_jitter_s=0,
    )
    await warmer.start()
    try:
        for _ in range(200):
            if redis.pings >= 4:
                break
            await asyncio.sleep(0.01)
        assert redis.pings >= 4  # boot + several failing rounds, loop still alive
        assert warmer._task is not None and not warmer._task.done()
        redis.error = None
        for _ in range(200):
            if warmer.results["redis"].ok:
                break
            await asyncio.sleep(0.01)
        assert warmer.results["redis"].ok is True
    finally:
        await warmer.stop()


def test_the_default_period_is_45_to_60_seconds() -> None:
    warmer = _warmer()
    delays = [warmer.next_delay() for _ in range(200)]
    assert all(45 <= d <= 60 for d in delays)


# -- embedding host: no tokens ------------------------------------------------------


def _gemini(handler: Any) -> tuple[GeminiEmbedder, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    async def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return await handler(request)

    embedder = GeminiEmbedder(
        api_key="test-key",
        dimensions=DIMS,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(record)),
    )
    return embedder, seen


async def _model_metadata(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"name": "models/gemini-embedding-001"})


async def test_embed_warm_is_a_metadata_get_through_the_embedders_client() -> None:
    embedder, seen = _gemini(_model_metadata)
    warmer = _warmer(embedder=embedder)
    results = await warmer.warm_once()
    assert results["embed"].ok is True
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert seen[0].url.path.endswith("/models/gemini-embedding-001")
    # Never an embedding call: nothing that bills tokens.
    assert ":" not in seen[0].url.path  # no :embedContent / :batchEmbedContents verb
    assert not seen[0].content  # and no body to bill


async def test_an_http_error_answer_still_counts_as_warm() -> None:
    async def forbidden(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"code": 403, "message": "no"}})

    embedder, _ = _gemini(forbidden)
    results = await _warmer(embedder=embedder).warm_once()
    assert results["embed"].ok is True
    assert results["embed"].detail == "http 403"


async def test_an_unreachable_provider_is_reported_not_raised() -> None:
    async def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    embedder, _ = _gemini(refused)
    results = await _warmer(embedder=embedder).warm_once()
    assert results["embed"].ok is False
    assert results["embed"].detail == "ConnectError"


def test_gemini_keeps_idle_provider_connections_past_the_keep_warm_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = _settings(embedding_backend="gemini", gemini_api_key="test-key")
    embedder = build_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)
    limits = embedder._client._api_client._async_httpx_client_args["limits"]
    assert limits.keepalive_expiry == settings.embedding_keepalive_s == 120.0
    assert settings.embedding_keepalive_s > settings.keep_warm_interval_s + (
        settings.keep_warm_jitter_s
    )


# -- the app ---------------------------------------------------------------------------


async def test_the_app_warms_before_ready_and_reports_it() -> None:
    app = create_app(_settings())
    async with app.router.lifespan_context(app):
        warmer = app.state.warmer
        assert warmer.results["db"].ok is True
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            ready = (await c.get("/ready")).json()
            health = (await c.get("/health")).json()
        assert ready["checks"] == {"store": True}
        assert ready["warm"]["db"]["ok"] is True
        assert set(ready["warm"]["db"]) >= {"ok", "latency_ms", "age_s"}
        assert "warm" not in health  # /health is unchanged
        task = warmer._task
    assert task is not None and task.done()  # cancelled on shutdown


async def test_ready_is_still_503_when_the_store_is_down() -> None:
    app = create_app(_settings())
    async with app.router.lifespan_context(app):

        async def down() -> bool:
            return False

        app.state.store.ping = down
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"


# -- Postgres: the real pool -----------------------------------------------------------


@pytest.mark.postgres
async def test_the_real_pool_holds_the_warmed_connections(postgres_store: Any) -> None:
    warmer = _warmer(store=postgres_store, warm_db_connections=4, warm_timeout_s=10)
    results = await warmer.warm_once()
    assert results["db"].ok is True, results["db"].detail
    assert postgres_store._engine.pool.checkedin() >= 4
