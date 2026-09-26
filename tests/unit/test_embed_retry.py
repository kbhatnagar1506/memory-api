"""Embedding retries: who retries what, for how long, and what a caller sees.

The failures these pin were live ones:

  * a retired model (404) burned 1.1 s of backoff before failing with the
    error it had on the first attempt;
  * 429s were retried on a fixed 0.25/0.5 s schedule, three attempts inside
    a quota window that reopens in tens of seconds -- every retry a wasted
    request into the same wall;
  * a search's query embedding inherited the document budget, 3 x 20 s, so
    one stuck call could hold a request for a minute;
  * a batch of 101 texts is a 400 on the Developer API, deterministically,
    and nothing refused that configuration at boot.

The Gemini tests drive the REAL google-genai SDK through an httpx mock
transport, so the error objects, headers and the SDK's own retry loop are the
production ones. Waits in MAPI's loop go through the provider's injectable
`_sleep` and are recorded rather than slept.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from mapi.config import Settings
from mapi.core.errors import ConfigurationError, ProviderError, ProviderTimeoutError
from mapi.domain.embeddings import DeterministicEmbedder, build_embedder
from mapi.domain.embeddings import gemini as gemini_module
from mapi.domain.embeddings.base import HEDGE_PAUSE_AFTER_429_S, EmbeddingProvider
from mapi.domain.embeddings.gemini import (
    DEFAULT_MODEL,
    DEVELOPER_API_MAX_BATCH,
    GeminiEmbedder,
    retry_after_seconds,
)

DIMS = 8

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _ok(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    count = len(body.get("requests", [])) or 1
    return httpx.Response(
        200, json={"embeddings": [{"values": [0.5] * DIMS} for _ in range(count)]}
    )


def _error(status: int, *, headers: dict[str, str] | None = None, retry_delay: str = "") -> Any:
    def respond(_request: httpx.Request) -> httpx.Response:
        details = (
            [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
            if retry_delay
            else []
        )
        return httpx.Response(
            status,
            headers=headers or {},
            json={"error": {"code": status, "message": "scripted", "details": details}},
        )

    return respond


class Script:
    """Answers each request with the next scripted response; counts them."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.requests = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        step = self.steps[min(self.requests, len(self.steps) - 1)]
        self.requests += 1
        result = step(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result  # type: ignore[no-any-return]


def _hang(seconds: float) -> Any:
    async def respond(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(seconds)
        return _ok(request)

    return respond


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK's own 5xx backoff, shrunk: its retry loop still runs for real."""
    monkeypatch.setattr(gemini_module, "_SDK_RETRY_INITIAL_DELAY_S", 0.001)
    monkeypatch.setattr(gemini_module, "_SDK_RETRY_JITTER_S", 0.001)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _embedder(script: Script, **kw: Any) -> tuple[GeminiEmbedder, list[float]]:
    embedder = GeminiEmbedder(
        api_key="test-key",
        dimensions=DIMS,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(script)),
        **kw,
    )
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    embedder._sleep = record
    return embedder, slept


# -- document path: MAPI's loop ------------------------------------------------


async def test_a_429_waits_at_least_as_long_as_retry_after_says() -> None:
    script = Script(_error(429, headers={"retry-after": "2"}), _ok)
    embedder, slept = _embedder(script)
    result = await embedder.embed(["hello"])
    assert len(result.vectors) == 1
    assert script.requests == 2
    assert slept == [2.0], "Retry-After is a floor under the 0.25 s backoff"


async def test_a_429_without_a_header_reads_retry_info_from_the_body() -> None:
    script = Script(_error(429, retry_delay="7s"), _ok)
    embedder, slept = _embedder(script)
    await embedder.embed(["hello"])
    assert slept == [7.0]


async def test_a_429_with_no_hint_falls_back_to_exponential_backoff() -> None:
    script = Script(_error(429), _error(429), _ok)
    embedder, slept = _embedder(script)
    await embedder.embed(["hello"])
    assert slept == [0.25, 0.5]


async def test_a_retry_after_longer_than_the_cap_fails_now_with_the_hint() -> None:
    script = Script(_error(429, headers={"retry-after": "120"}), _ok)
    embedder, slept = _embedder(script)
    embedder.configure(max_retry_delay_s=30)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(["hello"])
    assert script.requests == 1 and slept == []
    assert caught.value.extra["retry_after"] == 120.0
    assert caught.value.extra["upstream_status"] == 429
    # The hint reaches the problem document a client reads.
    problem = caught.value.to_problem()
    assert problem["retry_after"] == 120.0 and problem["status"] == 502


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_errors_are_never_retried(status: int) -> None:
    """A retired model, a bad key or an oversized batch answers the same way
    every time; retrying only delays the error."""
    script = Script(_error(status), _ok)
    embedder, slept = _embedder(script)
    started = time.perf_counter()
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(["hello"])
    assert script.requests == 1, "the SDK must not retry it either"
    assert slept == []
    assert caught.value.extra["upstream_status"] == status
    assert time.perf_counter() - started < 0.5


async def test_a_503_is_retried_by_the_sdk_not_by_mapi() -> None:
    script = Script(_error(503), _ok)
    embedder, slept = _embedder(script)
    await embedder.embed(["hello"])
    assert script.requests == 2
    assert slept == [], "the SDK's own attempt absorbed it; MAPI never backed off"


async def test_a_persistent_5xx_is_bounded() -> None:
    """SDK attempts (2) times MAPI attempts (3): a bounded outage, not a loop."""
    script = Script(_error(503))
    embedder, slept = _embedder(script)
    with pytest.raises(ProviderError) as caught:
        await embedder.embed(["hello"])
    assert script.requests == 2 * embedder.max_attempts
    assert slept == [0.25, 0.5]
    assert caught.value.extra["upstream_status"] == 503


async def test_a_dimension_mismatch_is_still_not_retried() -> None:
    def short(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{"values": [0.5] * 3}]})

    script = Script(short)
    embedder, slept = _embedder(script)
    with pytest.raises(ProviderError, match="dimensions"):
        await embedder.embed(["hello"])
    assert script.requests == 1 and slept == []


# -- query path: its own deadline, at most one retry ---------------------------------


async def test_a_query_gets_its_own_deadline_and_one_retry() -> None:
    script = Script(_hang(5.0), _ok)
    embedder, _ = _embedder(script)
    embedder.configure(query_timeout_s=0.2, query_attempts=2)
    started = time.perf_counter()
    vector = await embedder.embed_query("where is the robotics lab")
    elapsed = time.perf_counter() - started
    assert len(vector) == DIMS
    assert script.requests == 2
    assert elapsed < 1.0, f"the query waited {elapsed:.2f}s; the document budget leaked in"


async def test_a_query_that_keeps_hanging_fails_after_two_attempts() -> None:
    script = Script(_hang(5.0))
    embedder, _ = _embedder(script)
    embedder.configure(query_timeout_s=0.15, query_attempts=2)
    started = time.perf_counter()
    with pytest.raises(ProviderTimeoutError):
        await embedder.embed_query("anyone into rust")
    assert script.requests == 2
    assert time.perf_counter() - started < 1.0


async def test_a_query_429_is_not_retried_and_pauses_hedging() -> None:
    script = Script(_error(429, headers={"retry-after": "1"}), _ok)
    embedder, slept = _embedder(script)
    embedder.configure(query_attempts=2, query_hedge_ms=50)
    assert embedder.hedging_active
    with pytest.raises(ProviderError):
        await embedder.embed_query("who should I meet")
    assert script.requests == 1, "the caller's fallback beats a second trip into the wall"
    assert slept == []
    assert not embedder.hedging_active
    remaining = embedder._hedge_paused_until - time.monotonic()
    assert HEDGE_PAUSE_AFTER_429_S - 1 < remaining <= HEDGE_PAUSE_AFTER_429_S


async def test_a_query_400_is_not_retried() -> None:
    script = Script(_error(400), _ok)
    embedder, _ = _embedder(script)
    embedder.configure(query_attempts=2)
    with pytest.raises(ProviderError):
        await embedder.embed_query("hello")
    assert script.requests == 1


async def test_the_query_path_turns_off_the_sdk_retry() -> None:
    """A 503 on a query is one request per MAPI attempt: the SDK's one-second
    backoff cannot fit in a sub-second budget."""
    script = Script(_error(503), _ok)
    embedder, _ = _embedder(script)
    embedder.configure(query_attempts=1)
    with pytest.raises(ProviderError):
        await embedder.embed_query("hello")
    assert script.requests == 1


# -- hedging --------------------------------------------------------------------------


async def test_hedging_is_off_by_default() -> None:
    script = Script(_hang(0.2), _ok)
    embedder, _ = _embedder(script)
    embedder.configure(query_timeout_s=2.0)
    await embedder.embed_query("hello")
    assert script.requests == 1


async def test_a_slow_query_is_hedged_and_the_first_success_wins() -> None:
    script = Script(_hang(3.0), _ok)
    embedder, _ = _embedder(script)
    embedder.configure(query_timeout_s=2.0, query_hedge_ms=50)
    started = time.perf_counter()
    await embedder.embed_query("hello")
    assert script.requests == 2
    assert time.perf_counter() - started < 1.0, "the hedge's answer should have won"


async def test_no_hedge_fires_while_paused_after_a_429() -> None:
    script = Script(_hang(0.3), _ok)
    embedder, _ = _embedder(script)
    embedder.configure(query_timeout_s=2.0, query_hedge_ms=20)
    embedder._hedge_paused_until = time.monotonic() + HEDGE_PAUSE_AFTER_429_S
    await embedder.embed_query("hello")
    assert script.requests == 1


async def test_a_fast_failure_does_not_beat_a_hedge_that_succeeds() -> None:
    class Flaky(EmbeddingProvider):
        """The first request fails AFTER the hedge has started but BEFORE the
        hedge answers. The failure lands first; the success must still win."""

        def __init__(self) -> None:
            super().__init__(model="m", dimensions=DIMS, max_attempts=1)
            self.calls = 0

        @property
        def name(self) -> str:
            return "flaky"

        async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.05)
                raise RuntimeError("connection reset")
            await asyncio.sleep(0.1)
            return [[0.5] * DIMS for _ in texts]

    provider = Flaky()
    provider.configure(query_timeout_s=1.0, query_attempts=1, query_hedge_ms=10)
    vector = await provider._query_with_deadline(lambda: provider._embed_batch(["q"]))
    assert len(vector) == DIMS and provider.calls == 2


# -- configuration ---------------------------------------------------------------------


def test_gemini_embedding_001_is_the_default_everywhere() -> None:
    assert Settings().embedding_model == "gemini-embedding-001"
    assert DEFAULT_MODEL == "gemini-embedding-001"
    assert GeminiEmbedder(api_key="k", dimensions=DIMS).model == "gemini-embedding-001"


def test_a_batch_over_100_is_refused_at_boot_with_a_key() -> None:
    base: dict[str, Any] = {"embedding_backend": "gemini", "gemini_api_key": "k"}
    Settings(**base, embedding_batch_size=DEVELOPER_API_MAX_BATCH)
    with pytest.raises(ValueError, match="exceeds the Gemini Developer API"):
        Settings(**base, embedding_batch_size=DEVELOPER_API_MAX_BATCH + 1)
    with pytest.raises(ConfigurationError):
        GeminiEmbedder(api_key="k", dimensions=DIMS, batch_size=DEVELOPER_API_MAX_BATCH + 1)


def test_the_key_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(ValueError, match="exceeds"):
        Settings(embedding_backend="gemini", embedding_batch_size=101)


def test_vertex_takes_larger_batches() -> None:
    settings = Settings(
        embedding_backend="gemini", google_cloud_project="p", embedding_batch_size=250
    )
    assert settings.embedding_batch_size == 250


def test_build_embedder_applies_the_runtime_limits() -> None:
    settings = Settings(
        embedding_concurrency=3,
        query_embedding_timeout_s=0.8,
        query_embedding_attempts=1,
        query_embedding_hedge_ms=150,
        embedding_max_retry_delay_s=12,
    )
    provider = build_embedder(settings)
    assert provider.concurrency == 3
    assert provider.query_timeout_s == 0.8
    assert provider.query_attempts == 1
    assert provider.query_hedge_ms == 150
    assert provider.max_retry_delay_s == 12


def test_configure_refuses_nonsense() -> None:
    provider = DeterministicEmbedder(dimensions=DIMS)
    with pytest.raises(ConfigurationError):
        provider.configure(concurrency=0)
    with pytest.raises(ConfigurationError):
        provider.configure(query_timeout_s=0)
    with pytest.raises(ConfigurationError):
        provider.configure(query_attempts=0)


# -- the Retry-After parser -------------------------------------------------------------


class _Exc:
    def __init__(self, headers: dict[str, str] | None = None, details: Any = None) -> None:
        self.response = httpx.Response(429, headers=headers or {})
        self.details = details


def test_retry_after_accepts_seconds_dates_and_retry_info() -> None:
    assert retry_after_seconds(_Exc({"retry-after": "3"})) == 3.0
    assert retry_after_seconds(_Exc({"retry-after": "1.5"})) == 1.5
    later = datetime.now(UTC) + timedelta(seconds=30)
    parsed = retry_after_seconds(_Exc({"retry-after": format_datetime(later, usegmt=True)}))
    assert parsed is not None and 25 <= parsed <= 31
    body = {
        "error": {
            "details": [
                {"@type": "type.googleapis.com/google.rpc.ErrorInfo"},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
            ]
        }
    }
    assert retry_after_seconds(_Exc(details=body)) == 34.0
    # The header wins when both are present: it is the transport's own word.
    assert retry_after_seconds(_Exc({"retry-after": "2"}, details=body)) == 2.0


def test_retry_after_ignores_what_it_cannot_parse() -> None:
    assert retry_after_seconds(_Exc({"retry-after": "soon"})) is None
    assert retry_after_seconds(_Exc(details={"error": {"details": "nope"}})) is None
    assert retry_after_seconds(_Exc(details=[1, 2])) is None
    assert retry_after_seconds(object()) is None


# -- bounded concurrency inside one embed() ------------------------------------------------


class _Gauge(EmbeddingProvider):
    def __init__(self, *, fail_on: int | None = None) -> None:
        super().__init__(model="m", dimensions=DIMS, batch_size=2, max_attempts=1)
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.fail_on = fail_on

    @property
    def name(self) -> str:
        return "gauge"

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        call = self.calls
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.02)
            if call == self.fail_on:
                raise RuntimeError("batch exploded")
            return [[float(len(t))] + [0.5] * (DIMS - 1) for t in texts]
        finally:
            self.in_flight -= 1


async def test_batches_run_at_most_concurrency_at_a_time_and_keep_order() -> None:
    provider = _Gauge()
    provider.configure(concurrency=2)
    texts = [f"t{'x' * n}" for n in range(9)]  # 5 batches of 2
    result = await provider.embed(texts)
    assert provider.calls == 5
    assert provider.peak == 2
    firsts = [v[0] for v in result.vectors]
    assert firsts == sorted(firsts), "vectors came back out of order"


async def test_a_failed_concurrent_batch_surfaces_as_a_provider_error() -> None:
    provider = _Gauge(fail_on=2)
    provider.configure(concurrency=3)
    with pytest.raises(ProviderError, match="batch exploded"):
        await provider.embed([f"text {n}" for n in range(6)])
