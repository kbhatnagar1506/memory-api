"""Token-bucket rate limiting.

A token bucket rather than a fixed window because fixed windows allow a client
to send 2x the limit across a boundary — the last instant of one window plus the
first of the next — which is exactly when a retry storm arrives.

Two backends behind one interface. The in-memory one is per-process, which is
honest but wrong the moment there is more than one replica; production
validation warns when `redis_url` is unset for that reason. The Redis one is
atomic via a Lua script, so concurrent replicas share one budget.
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass

from .logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    limit: int
    reset_after: float
    retry_after: float = 0.0


class RateLimiter(abc.ABC):
    def __init__(self, *, per_minute: int, burst: int) -> None:
        if per_minute <= 0 or burst <= 0:
            raise ValueError("rate limit and burst must be positive")
        self.per_minute = per_minute
        self.burst = burst
        self.refill_per_second = per_minute / 60.0

    @abc.abstractmethod
    async def check(self, key: str, cost: int = 1) -> RateLimitDecision: ...

    async def aclose(self) -> None:
        return None


class InMemoryRateLimiter(RateLimiter):
    """Per-process token bucket. Correct for one replica, not for many."""

    def __init__(self, *, per_minute: int, burst: int) -> None:
        super().__init__(per_minute=per_minute, burst=burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitDecision:
        now = time.monotonic()
        async with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.refill_per_second)
            if tokens >= cost:
                tokens -= cost
                self._buckets[key] = (tokens, now)
                return RateLimitDecision(
                    True, int(tokens), self.burst,
                    (self.burst - tokens) / self.refill_per_second,
                )
            self._buckets[key] = (tokens, now)
            deficit = cost - tokens
            retry = deficit / self.refill_per_second
            return RateLimitDecision(False, 0, self.burst, retry, retry)

    async def reset(self) -> None:
        async with self._lock:
            self._buckets.clear()


_LUA = """
local key = KEYS[1]
local burst = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then tokens = burst; ts = now end
tokens = math.min(burst, tokens + (now - ts) * refill)
local allowed = 0
if tokens >= cost then tokens = tokens - cost; allowed = 1 end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(burst / refill) + 60)
return {allowed, tostring(tokens)}
"""


class RedisRateLimiter(RateLimiter):
    """Shared token bucket. The Lua script makes read-modify-write atomic."""

    def __init__(self, redis_client: object, *, per_minute: int, burst: int,
                 namespace: str = "sm:rl") -> None:
        super().__init__(per_minute=per_minute, burst=burst)
        self._redis = redis_client
        self._namespace = namespace
        self._script = None

    async def check(self, key: str, cost: int = 1) -> RateLimitDecision:
        full_key = f"{self._namespace}:{key}"
        try:
            if self._script is None:
                self._script = self._redis.register_script(_LUA)  # type: ignore[attr-defined]
            allowed, tokens_raw = await self._script(  # type: ignore[misc]
                keys=[full_key],
                args=[self.burst, self.refill_per_second, time.time(), cost],
            )
            tokens = float(tokens_raw)
        except Exception as exc:  # noqa: BLE001
            # Fail OPEN. A rate limiter outage must not become a service outage;
            # the alternative is that one Redis blip rejects all traffic.
            log.warning("ratelimit_backend_error", error=str(exc)[:200])
            return RateLimitDecision(True, self.burst, self.burst, 0.0)
        if int(allowed) == 1:
            return RateLimitDecision(
                True, int(tokens), self.burst,
                (self.burst - tokens) / self.refill_per_second,
            )
        retry = max(cost - tokens, 0) / self.refill_per_second
        return RateLimitDecision(False, 0, self.burst, retry, retry)


def build_rate_limiter(*, per_minute: int, burst: int, redis_url: str | None) -> RateLimiter:
    if redis_url:
        try:
            import redis.asyncio as aioredis  # noqa: PLC0415

            client = aioredis.from_url(redis_url, decode_responses=True)
            return RedisRateLimiter(client, per_minute=per_minute, burst=burst)
        except Exception as exc:  # noqa: BLE001
            log.warning("ratelimit_redis_unavailable", error=str(exc)[:200])
    return InMemoryRateLimiter(per_minute=per_minute, burst=burst)


__all__ = [
    "InMemoryRateLimiter", "RateLimitDecision", "RateLimiter", "RedisRateLimiter",
    "build_rate_limiter",
]
