"""Boot warm-up and keep-warm: no request pays a cold cost.

At boot, before the app reports ready, every outbound dependency is opened
through the SAME client or pool the request path uses -- a separate client
would warm nothing a request ever touches:

  * db     -- `warm_db_connections` pool connections opened at once, each
              running SELECT 1, then returned to the pool idle.
  * redis  -- PING on the rate limiter's client, when Redis is configured.
  * embed  -- the embedding provider's `warm_connection`: a model-metadata
              GET through the embedder's own HTTP client. TCP, TLS, DNS and
              (on Vertex) the ADC token fetch, and never a token spent.

The warmers run concurrently, each bounded by `warm_timeout_s`, and each is
failure-tolerant: a dependency that is down or slow is logged and reported,
and boot carries on. Readiness is still decided by /ready's store check.

Then a background task repeats the same cheap calls every
`keep_warm_interval_s` + jitter, so idle connections and TLS sessions are not
reaped between bursts of traffic. It never blocks a request: it only touches
IDLE pool connections, and it is cancelled cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from .config import Settings
from .core.logging import get_logger
from .core.ratelimit import RateLimiter, RedisRateLimiter
from .domain.embeddings import EmbeddingProvider
from .store.base import MemoryStore

log = get_logger("mapi.warm")

_SELECT_1 = text("SELECT 1")

#: One warmer: does its cheap call and returns an optional detail string
#: ("4/4 connections"), or raises.
WarmStep = Callable[[], Awaitable[str | None]]


@dataclass(frozen=True, slots=True)
class WarmResult:
    ok: bool
    latency_ms: float
    #: Never an exception message: those can carry URLs, hosts or keys. Only
    #: a type name, "timeout", an HTTP status or a count.
    detail: str | None
    #: `time.monotonic()` when the check finished.
    at: float


class Warmer:
    """Warms the app's dependencies once at boot, then keeps them warm."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        rate_limiter: RateLimiter | None,
        embedder: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._store = store
        self._rate_limiter = rate_limiter
        self._embedder = embedder
        self._settings = settings
        self._timeout_s = settings.warm_timeout_s
        self._db_connections = min(settings.warm_db_connections, settings.db_pool_size)
        self.results: dict[str, WarmResult] = {}
        self._task: asyncio.Task[None] | None = None
        #: Which dependencies failed on the previous round, so the loop logs a
        #: change of state rather than the same warning every minute.
        self._failing: set[str] = set()

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Boot warm-up (bounded by `warm_timeout_s`), then start the loop."""
        if not self._settings.warm_enabled:
            return
        started = time.perf_counter()
        results = await self.warm_once()
        total_ms = (time.perf_counter() - started) * 1000
        summary = ", ".join(
            f"{name} {r.latency_ms:.0f}ms" + ("" if r.ok else f" FAILED ({r.detail})")
            for name, r in results.items()
        )
        log.info(
            "warm",
            summary=f"{summary or 'nothing to warm'} in {total_ms:.0f} ms",
            total_ms=round(total_ms, 1),
            **{f"{name}_ok": r.ok for name, r in results.items()},
        )
        if self._settings.keep_warm_interval_s > 0 and results:
            self._task = asyncio.create_task(self._keep_warm(), name="mapi-keep-warm")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # -- one round -------------------------------------------------------------

    def steps(self, *, boot: bool = True) -> dict[str, WarmStep]:
        """The warmers that apply to this configuration, by dependency name."""
        steps: dict[str, WarmStep] = {}
        if self._db_connections > 0:
            steps["db"] = lambda: self._warm_db(boot=boot)
        if isinstance(self._rate_limiter, RedisRateLimiter):
            steps["redis"] = self._warm_redis
        if callable(getattr(self._embedder, "warm_connection", None)):
            steps["embed"] = self._warm_embed
        return steps

    async def warm_once(self, *, boot: bool = True) -> dict[str, WarmResult]:
        """Run every warmer concurrently. Never raises."""
        steps = self.steps(boot=boot)
        names = list(steps)
        outcomes = await asyncio.gather(*(self._timed(steps[n]) for n in names))
        results = dict(zip(names, outcomes, strict=True))
        self.results.update(results)
        return results

    async def _timed(self, step: WarmStep) -> WarmResult:
        started = time.perf_counter()
        ok, detail = True, None
        try:
            detail = await asyncio.wait_for(step(), timeout=self._timeout_s)
        except TimeoutError:
            ok, detail = False, "timeout"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ok, detail = False, type(exc).__name__
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return WarmResult(ok=ok, latency_ms=latency_ms, detail=detail, at=time.monotonic())

    # -- warmers ---------------------------------------------------------------

    async def _warm_db(self, *, boot: bool) -> str | None:
        # The engine is private to PostgresStore; reading it here keeps the
        # store untouched. Any other store (in-memory) gets its own ping.
        engine = getattr(self._store, "_engine", None)
        if engine is None:
            if not await self._store.ping():
                raise ConnectionError("store ping failed")
            return None
        want = self._db_connections
        if not boot:
            pool = engine.pool
            checked_out = _pool_count(pool, "checkedout")
            idle = _pool_count(pool, "checkedin")
            if checked_out is not None and idle is not None and checked_out > 0:
                # Traffic is using the pool: busy connections are warm by
                # definition, and taking one a request is waiting for would
                # be the warmer blocking a request. Touch only idle ones.
                want = min(want, idle)
                if want == 0:
                    return "pool busy"
        opened = await _hold_open(engine, want)
        return f"{opened}/{want} connections"

    async def _warm_redis(self) -> str | None:
        assert isinstance(self._rate_limiter, RedisRateLimiter)
        client: Any = self._rate_limiter.client
        await client.ping()
        return None

    async def _warm_embed(self) -> str | None:
        warm: Callable[[float], Awaitable[None]] = self._embedder.warm_connection  # type: ignore[attr-defined]
        try:
            await warm(self._timeout_s)
        except Exception as exc:
            status = _http_status(exc)
            if status is None:
                raise
            # The provider ANSWERED: the connection is open, which is all a
            # warm-up is for. A 403 on model metadata is not a cold start.
            return f"http {status}"
        return None

    # -- keep-warm loop ----------------------------------------------------------

    def next_delay(self) -> float:
        return self._settings.keep_warm_interval_s + random.uniform(
            0, self._settings.keep_warm_jitter_s
        )

    async def _keep_warm(self) -> None:
        while True:
            await asyncio.sleep(self.next_delay())
            try:
                results = await self.warm_once(boot=False)
            except Exception as exc:  # pragma: no cover - warm_once does not raise
                log.warning("keep_warm_error", error=type(exc).__name__)
                continue
            self._log_changes(results)

    def _log_changes(self, results: dict[str, WarmResult]) -> None:
        for name, r in results.items():
            if not r.ok and name not in self._failing:
                self._failing.add(name)
                log.warning("keep_warm_failed", dependency=name, detail=r.detail)
            elif r.ok and name in self._failing:
                self._failing.discard(name)
                log.info("keep_warm_recovered", dependency=name, latency_ms=r.latency_ms)

    # -- reporting -------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Per-dependency state for /ready. No hosts, no URLs, no messages."""
        now = time.monotonic()
        return {
            name: {
                "ok": r.ok,
                "latency_ms": r.latency_ms,
                "detail": r.detail,
                "age_s": round(now - r.at, 1),
            }
            for name, r in self.results.items()
        }


async def _hold_open(engine: Any, n: int) -> int:
    """Open `n` pool connections AT ONCE, SELECT 1 on each, then release them.

    Held together until all have settled: released one at a time, the pool
    would hand the same connection back to the next task and `n` checkouts
    would open one connection. Returns how many succeeded; raises the first
    error only if none did.
    """
    settled = 0
    all_settled = asyncio.Event()

    def settle() -> None:
        nonlocal settled
        settled += 1
        if settled >= n:
            all_settled.set()

    async def one() -> None:
        counted = False
        try:
            async with engine.connect() as conn:
                await conn.execute(_SELECT_1)
                counted = True
                settle()
                await all_settled.wait()
        finally:
            if not counted:
                settle()

    outcomes = await asyncio.gather(*(one() for _ in range(n)), return_exceptions=True)
    errors = [o for o in outcomes if isinstance(o, BaseException)]
    for err in errors:
        if isinstance(err, asyncio.CancelledError):
            raise err
    if errors and len(errors) == n:
        raise errors[0]
    return n - len(errors)


def _pool_count(pool: Any, method: str) -> int | None:
    fn = getattr(pool, method, None)
    if not callable(fn):
        return None
    try:
        value = fn()
    except Exception:
        return None
    return value if isinstance(value, int) else None


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status a provider SDK error carries, if it carries one."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None


__all__ = ["WarmResult", "Warmer"]
