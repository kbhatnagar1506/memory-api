"""Embedding provider interface, batching and vector utilities.

Providers implement `_embed_batch`. Everything else — batching, retries,
dimension validation, degenerate-vector rejection, caching — lives here so a new
provider is roughly thirty lines and cannot skip the safety checks.

The validation is not ceremonial. A provider returning a zero vector (which some
do for whitespace-only input) poisons cosine similarity with a division by zero,
and a silent dimension change between model versions corrupts an entire index
with no error until recall quietly collapses.
"""

from __future__ import annotations

import abc
import asyncio
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ...core.errors import ConfigurationError, ProviderError, ProviderTimeoutError
from ...core.logging import get_logger

log = get_logger(__name__)

Vector = list[float]


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """How the retry loops treat one failure.

    A provider-neutral verdict, so the loops below never import a vendor SDK
    to read an exception. The distinction that matters is not "error or not"
    but which errors a retry can fix: a 429 clears if you wait, a 400 or 404
    never does -- a retired model used to burn 1.1 s of backoff before
    failing with the error it had from the first attempt.
    """

    retryable: bool
    #: Seconds the provider asked us to wait (Retry-After or google.rpc
    #: RetryInfo), when it said. Honoured as a floor under our own backoff.
    retry_after: float | None = None
    #: The provider refused on quota (HTTP 429). Pauses query hedging.
    rate_limited: bool = False
    #: Upstream HTTP status, when there was one. For logs and error extras.
    status: int | None = None


#: How long query hedging stays off after a 429. Hedging duplicates requests;
#: doing that into a quota wall doubles the load that built the wall.
HEDGE_PAUSE_AFTER_429_S = 60.0


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: list[Vector]
    model: str
    dimensions: int
    #: Tokens billed, when the provider reports them.
    total_tokens: int | None = None
    cache_hits: int = 0


def l2_normalize(vec: Sequence[float]) -> Vector:
    """Scale to unit length so cosine similarity reduces to a dot product."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("cannot normalize a zero or non-finite vector")
    return [x / norm for x in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [-1, 1] against float drift."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cannot compare empty vectors")
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(na * nb)))


class EmbeddingProvider(abc.ABC):
    """Base class. Subclasses implement `_embed_batch` and `name`."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        batch_size: int = 32,
        timeout_s: float = 20.0,
        max_attempts: int = 3,
        cache_size: int = 0,
    ) -> None:
        if dimensions <= 0:
            raise ConfigurationError("embedding dimensions must be positive")
        if batch_size <= 0:
            raise ConfigurationError("embedding batch_size must be positive")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self._cache: OrderedDict[str, Vector] = OrderedDict()
        self._cache_size = max(0, cache_size)
        # -- runtime limits, set by `configure()` ---------------------------
        # The defaults reproduce what every provider did before these knobs
        # existed, so a provider built directly (bench, tests) is unchanged.
        #: Document batches in flight at once within one `embed()` call.
        self.concurrency = 1
        #: Per-attempt deadline and attempt count for query embeddings.
        self.query_timeout_s = timeout_s
        self.query_attempts = max_attempts
        #: Hedge delay for query embeddings, in ms. 0 is off.
        self.query_hedge_ms = 0
        #: Longest a Retry-After is honoured before giving up instead.
        self.max_retry_delay_s = 30.0
        self._hedge_paused_until = 0.0
        #: Injectable so retry tests assert on delays without sleeping them.
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def configure(
        self,
        *,
        concurrency: int | None = None,
        query_timeout_s: float | None = None,
        query_attempts: int | None = None,
        query_hedge_ms: int | None = None,
        max_retry_delay_s: float | None = None,
    ) -> None:
        """Apply the runtime limits from settings.

        A method rather than more constructor arguments: every provider
        forwards its constructor to this base by hand, and a knob added there
        is a knob some provider forgets to pass through. Set once, at build
        time, by `build_embedder`.
        """
        if concurrency is not None:
            if concurrency < 1:
                raise ConfigurationError("embedding concurrency must be at least 1")
            self.concurrency = concurrency
        if query_timeout_s is not None:
            if query_timeout_s <= 0:
                raise ConfigurationError("query embedding timeout must be positive")
            self.query_timeout_s = query_timeout_s
        if query_attempts is not None:
            if query_attempts < 1:
                raise ConfigurationError("query embedding attempts must be at least 1")
            self.query_attempts = query_attempts
        if query_hedge_ms is not None:
            self.query_hedge_ms = max(0, query_hedge_ms)
        if max_retry_delay_s is not None:
            self.max_retry_delay_s = max(0.0, max_retry_delay_s)

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        """Embed one batch. Raise ProviderError on failure."""

    # -- caching -----------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return f"{self.name}:{self.model}:{self.dimensions}:{text}"

    def _cache_get(self, text: str) -> Vector | None:
        if self._cache_size == 0:
            return None
        key = self._cache_key(text)
        vec = self._cache.get(key)
        if vec is not None:
            self._cache.move_to_end(key)
        return vec

    def _cache_put(self, text: str, vec: Vector) -> None:
        if self._cache_size == 0:
            return
        key = self._cache_key(text)
        self._cache[key] = vec
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # -- validation --------------------------------------------------------

    def _validate(self, vectors: list[Vector], expected: int) -> list[Vector]:
        if len(vectors) != expected:
            raise ProviderError(
                f"{self.name} returned {len(vectors)} vectors for {expected} inputs"
            )
        out: list[Vector] = []
        for i, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ProviderError(
                    f"{self.name} returned {len(vec)} dimensions, expected "
                    f"{self.dimensions}. A model or config change of this kind "
                    "silently corrupts an existing index."
                )
            if any(not math.isfinite(x) for x in vec):
                raise ProviderError(f"{self.name} returned a non-finite value at {i}")
            norm = math.sqrt(sum(x * x for x in vec))
            if norm == 0.0:
                raise ProviderError(
                    f"{self.name} returned a zero vector at index {i}; cosine "
                    "similarity is undefined for it"
                )
            out.append([x / norm for x in vec])
        return out

    # -- public API --------------------------------------------------------

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed texts, preserving order. Empty input yields an empty result.

        Blank strings are rejected rather than silently embedded: most providers
        return a zero or arbitrary vector for them, and that becomes a retrieval
        result that matches everything equally.
        """
        if not texts:
            return EmbeddingResult([], self.model, self.dimensions, 0, 0)

        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"texts[{i}] is {type(t).__name__}, expected str")
            if not t.strip():
                raise ValueError(f"texts[{i}] is empty or whitespace-only")

        resolved: list[Vector | None] = [self._cache_get(t) for t in texts]
        cache_hits = sum(1 for v in resolved if v is not None)
        pending = [
            (i, t) for i, (t, v) in enumerate(zip(texts, resolved, strict=True)) if v is None
        ]

        batches = [
            pending[start : start + self.batch_size]
            for start in range(0, len(pending), self.batch_size)
        ]
        # Batches run up to `concurrency` at a time. Sequential batches made a
        # large write one round trip per batch end to end; unbounded fan-out
        # is how one big document becomes a burst of 429s. A TaskGroup, so one
        # failed batch cancels its siblings instead of leaving them billing.
        gate = asyncio.Semaphore(self.concurrency)

        async def run(batch: list[tuple[int, str]]) -> None:
            async with gate:
                vectors = await self._with_retries([t for _, t in batch])
            for (idx, text), vec in zip(batch, vectors, strict=True):
                resolved[idx] = vec
                self._cache_put(text, vec)

        if len(batches) == 1 or self.concurrency == 1:
            for batch in batches:
                await run(batch)
        else:
            try:
                async with asyncio.TaskGroup() as group:
                    for batch in batches:
                        group.create_task(run(batch))
            except BaseExceptionGroup as grouped:
                # Callers catch ProviderError, not an exception group: surface
                # the first real failure exactly as the sequential path would.
                raise grouped.exceptions[0] from None

        final = [v for v in resolved if v is not None]
        if len(final) != len(texts):  # pragma: no cover - defensive
            raise ProviderError("embedding pipeline lost vectors")
        return EmbeddingResult(
            vectors=final,
            model=self.model,
            dimensions=self.dimensions,
            cache_hits=cache_hits,
        )

    async def embed_one(self, text: str) -> Vector:
        return (await self.embed([text])).vectors[0]

    # -- failure handling --------------------------------------------------

    def classify_error(self, exc: BaseException) -> RetryDecision:
        """Whether a failed call is worth repeating. Providers refine this.

        The default keeps the behaviour every provider had: timeouts and
        unknown failures are retried, a dimension mismatch is not (it will
        not fix itself, and retrying it only delays a loud config error).
        """
        if isinstance(exc, TimeoutError):
            return RetryDecision(retryable=True)
        if isinstance(exc, ProviderError) and "dimensions" in str(exc):
            return RetryDecision(retryable=False)
        return RetryDecision(retryable=True)

    def _as_provider_error(
        self, exc: BaseException, decision: RetryDecision, *, timeout_s: float
    ) -> ProviderError:
        """The error a caller sees: always a ProviderError, with the hints kept.

        `retry_after` rides along in the problem document, so a client that
        gets a 502 after a 429 upstream knows how long to back off rather
        than guessing and hitting the same wall.
        """
        extra: dict[str, object] = {}
        if decision.status is not None:
            extra["upstream_status"] = decision.status
        if decision.retry_after is not None:
            extra["retry_after"] = round(decision.retry_after, 3)
        if isinstance(exc, TimeoutError):
            return ProviderTimeoutError(
                f"{self.name} timed out after {timeout_s}s", extra=extra or None
            )
        if isinstance(exc, ProviderError):
            if extra:
                exc.extra = {**exc.extra, **extra}
            return exc
        return ProviderError(f"{self.name}: {type(exc).__name__}: {exc}", extra=extra or None)

    def _note_failure(self, decision: RetryDecision) -> None:
        if decision.rate_limited:
            self._hedge_paused_until = time.monotonic() + HEDGE_PAUSE_AFTER_429_S

    async def _with_retries(self, texts: list[str]) -> list[Vector]:
        """One document batch, retried only where a retry can help.

        Retry-After is a FLOOR under the exponential backoff, not a
        suggestion: retrying a 429 before the quota window reopens spends an
        attempt to learn nothing, and three attempts inside 0.75 s -- the old
        schedule -- was exactly that. A hint longer than `max_retry_delay_s`
        fails now with the hint attached, because the request would outlive
        its caller's deadline waiting for it.
        """
        delay = 0.25
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = await asyncio.wait_for(self._embed_batch(texts), timeout=self.timeout_s)
                return self._validate(raw, len(texts))
            except Exception as exc:
                decision = self.classify_error(exc)
                error = self._as_provider_error(exc, decision, timeout_s=self.timeout_s)
                self._note_failure(decision)
                log.warning(
                    "embedding_failed",
                    provider=self.name,
                    attempt=attempt,
                    retryable=decision.retryable,
                    status=decision.status,
                    retry_after=decision.retry_after,
                    error=str(error)[:200],
                )
                if not decision.retryable or attempt >= self.max_attempts:
                    raise error from exc
                wait = delay
                if decision.retry_after is not None:
                    if decision.retry_after > self.max_retry_delay_s:
                        raise error from exc
                    wait = max(wait, decision.retry_after)
                await self._sleep(wait)
                delay *= 2
        raise ProviderError(f"{self.name} failed with no diagnostic")  # pragma: no cover

    # -- query path ----------------------------------------------------------

    @property
    def hedging_active(self) -> bool:
        """Hedging is configured and not paused by a recent 429."""
        return self.query_hedge_ms > 0 and time.monotonic() >= self._hedge_paused_until

    async def _query_with_deadline(self, call: Callable[[], Awaitable[list[Vector]]]) -> Vector:
        """One query embedding under its own deadline, retried at most once.

        The document path's budget -- `timeout_s` per attempt, three attempts
        with backoff -- is right for a background write and wrong for a
        search, where a caller is waiting and has its own fallback. Here each
        attempt gets `query_timeout_s`, a 429 is never retried (the caller's
        fallback beats a second trip into the quota wall), and nothing sleeps
        between attempts.
        """
        last: ProviderError | None = None
        for attempt in range(1, self.query_attempts + 1):
            try:
                raw = await self._hedged(call)
                return self._validate(raw, 1)[0]
            except Exception as exc:
                decision = self.classify_error(exc)
                error = self._as_provider_error(exc, decision, timeout_s=self.query_timeout_s)
                self._note_failure(decision)
                log.warning(
                    "query_embedding_failed",
                    provider=self.name,
                    attempt=attempt,
                    retryable=decision.retryable,
                    status=decision.status,
                    error=str(error)[:200],
                )
                last = error
                if not decision.retryable or decision.rate_limited:
                    raise error from exc
        raise last or ProviderError(f"{self.name} query failed with no diagnostic")

    async def _hedged(self, call: Callable[[], Awaitable[list[Vector]]]) -> list[Vector]:
        """`call()` under the query deadline, with an optional hedge.

        With hedging on, a second identical request starts if the first has
        not answered after `query_hedge_ms`; the first SUCCESS wins and the
        loser is cancelled. A fast failure does not win -- the other request
        may still succeed -- but if both fail, the first failure is raised.
        """
        deadline = self.query_timeout_s
        if not self.hedging_active:
            return await asyncio.wait_for(call(), timeout=deadline)

        loop = asyncio.get_running_loop()
        started = loop.time()
        first = asyncio.ensure_future(call())
        tasks = [first]
        try:
            hedge_after = min(self.query_hedge_ms / 1000, deadline)
            done, _ = await asyncio.wait(tasks, timeout=hedge_after)
            # Re-checked after the wait: a 429 elsewhere during it pauses
            # hedging, and this request should respect that too.
            if not done and self.hedging_active and loop.time() - started < deadline:
                tasks.append(asyncio.ensure_future(call()))
            failure: BaseException | None = None
            pending = set(tasks)
            while pending:
                remaining = deadline - (loop.time() - started)
                if remaining <= 0:
                    break
                done, pending = await asyncio.wait(
                    pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    exc = task.exception()
                    if exc is None:
                        return task.result()
                    failure = failure or exc
            if failure is not None:
                raise failure
            raise TimeoutError
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def health(self) -> bool:
        try:
            await self.embed_one("health check")
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        """Release provider resources. Overridden where a client needs closing."""
        return None


__all__ = [
    "HEDGE_PAUSE_AFTER_429_S",
    "EmbeddingProvider",
    "EmbeddingResult",
    "RetryDecision",
    "Vector",
    "cosine_similarity",
    "l2_normalize",
]
