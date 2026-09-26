"""Google embeddings via Vertex AI (ADC) or the Gemini Developer API (key).

Vertex + Application Default Credentials is the default because it is what
production on GCP actually uses: no key material in the environment, rotation
handled by the platform. A `GEMINI_API_KEY` switches to the Developer API for
local work.

Two provider-specific details that matter:

  * Retrieval embeddings are asymmetric. Documents are embedded with
    `RETRIEVAL_DOCUMENT` and queries with `RETRIEVAL_QUERY`; using one task type
    for both measurably degrades recall. `embed_query` exists for that reason.
  * Calls use the SDK's NATIVE ASYNC client, not the blocking client in a
    worker thread. The thread-based version deadlocked under load: see
    `_call` for the full post-mortem.

Retries are split by who can do them properly -- see `classify_error`.
"""

from __future__ import annotations

import email.utils
import os
import re
import time
from typing import Any

from ...core.errors import ConfigurationError, ProviderError
from ...core.logging import get_logger
from .base import EmbeddingProvider, RetryDecision, Vector, normalize_query

log = get_logger(__name__)

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

#: Texts per request the Developer API accepts. 101 is a 400 (verified live),
#: and a 400 is never retried, so an oversized batch fails every large write.
DEVELOPER_API_MAX_BATCH = 100

#: The default model. text-embedding-004 is retired on the Developer API.
DEFAULT_MODEL = "gemini-embedding-001"

#: Statuses the SDK's own retry loop handles: server-side and transient. 429
#: is deliberately NOT here -- the SDK waits on a jittered exponential and
#: ignores Retry-After/RetryInfo, which for a per-minute quota means retrying
#: into the same closed window. MAPI's loop reads the hint instead.
_SDK_RETRY_STATUSES = [500, 502, 503, 504]
#: The SDK's backoff for those: first delay and jitter, in seconds (its own
#: defaults). Module constants so tests driving the real SDK through a mock
#: transport can shrink them instead of sleeping through them.
_SDK_RETRY_INITIAL_DELAY_S = 1.0
_SDK_RETRY_JITTER_S = 1.0

#: Client errors a retry cannot fix: bad request, auth, missing model.
_NEVER_RETRY = frozenset({400, 401, 403, 404, 409, 413, 422})

_RETRY_DELAY = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


class GeminiEmbedder(EmbeddingProvider):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        dimensions: int = 768,
        batch_size: int = 32,
        timeout_s: float = 20.0,
        cache_size: int = 0,
        query_cache_size: int = 0,
        project: str | None = None,
        location: str = "global",
        api_key: str | None = None,
        http_client: Any = None,
        keepalive_s: float | None = None,
        **_: object,
    ) -> None:
        """`http_client`: an `httpx.AsyncClient` for the SDK to send through.

        None (the default) lets the SDK build its own. Passing one shares a
        connection pool across embedders -- and it is how the retry tests put
        a mock transport under the REAL SDK, so what they pin is the SDK's
        actual error objects and retry loop rather than a stand-in for them.

        `keepalive_s`: how long an idle connection to the provider stays
        pooled when the SDK builds its own client. httpx's default is 5 s, so
        a search arriving after any longer pause paid a fresh TCP + TLS
        handshake to Google, and `mapi.warm`'s keep-warm ping could not hold a
        connection open. None keeps the SDK's default.
        """
        super().__init__(
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
            timeout_s=timeout_s,
            max_attempts=3,
            cache_size=cache_size,
            query_cache_size=query_cache_size,
        )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "google-genai is not installed. Install the 'gemini' extra."
            ) from exc

        from google.genai import types as genai_types

        # A real HTTP-level deadline, in milliseconds. This is what actually
        # aborts a stuck request. Set slightly above our own timeout so the
        # transport gives up first and raises, rather than us abandoning an
        # await while the request lives on.
        #
        # The SDK retries server errors and dropped connections itself, with
        # jittered backoff: that is what it does well, and it never saw a
        # retry option before this (retry_options=None means one attempt).
        # Two attempts, not five, because MAPI's own loop sits outside it and
        # the two multiply.
        http_options = genai_types.HttpOptions(
            timeout=int(timeout_s * 1000) + 5_000,
            retry_options=genai_types.HttpRetryOptions(
                attempts=2,
                initial_delay=_SDK_RETRY_INITIAL_DELAY_S,
                jitter=_SDK_RETRY_JITTER_S,
                max_delay=8.0,
                http_status_codes=_SDK_RETRY_STATUSES,
            ),
            httpx_async_client=http_client,
            async_client_args=(
                _pool_args(keepalive_s)
                if http_client is None and keepalive_s is not None
                else None
            ),
        )

        key = api_key or os.getenv("GEMINI_API_KEY")
        if key and batch_size > DEVELOPER_API_MAX_BATCH:
            raise ConfigurationError(
                f"batch_size={batch_size} exceeds the Gemini Developer API's limit of "
                f"{DEVELOPER_API_MAX_BATCH} texts per request"
            )
        if key:
            self._client = genai.Client(api_key=key, http_options=http_options)
            self.backend = "developer-api"
        else:
            proj = project or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not proj:
                raise ConfigurationError(
                    "Gemini embeddings need GEMINI_API_KEY, or "
                    "GOOGLE_CLOUD_PROJECT for Vertex AI with ADC "
                    "(gcloud auth application-default login)."
                )
            self._client = genai.Client(
                vertexai=True,
                project=proj,
                location=location,
                http_options=http_options,
            )
            self.backend = "vertex-adc"
        self._task_type = TASK_DOCUMENT

    @property
    def name(self) -> str:
        return "gemini"

    async def warm_connection(self, timeout_s: float) -> None:
        """Open (or reuse) the provider connection WITHOUT spending tokens.

        A GET of the model's metadata, through the same client, transport,
        auth and pool the embedding calls use: the TCP + TLS handshake, the
        DNS lookup and -- on Vertex -- the ADC access-token fetch all happen
        here instead of on the first real request. One attempt, no SDK retry.

        An HTTP error answer (403, 404) still proves the connection is open;
        it is re-raised for `mapi.warm` to report, which counts it as warm.
        """
        from google.genai import types

        await self._client.aio.models.get(
            model=self.model,
            config=types.GetModelConfig(
                http_options=types.HttpOptions(
                    timeout=int(timeout_s * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                )
            ),
        )

    async def _call(
        self, texts: list[str], task_type: str, *, sdk_retry: bool = True
    ) -> list[Vector]:
        """Native async SDK call — no worker thread involved.

        This used to run the blocking client through `asyncio.to_thread` under
        an `asyncio.wait_for` deadline. That combination deadlocks: `wait_for`
        cancels the *await*, but a thread running a blocking socket read cannot
        be cancelled, so every timeout permanently consumed one slot of the
        default executor (12 on an 8-core machine). A 500-corpus run
        accumulated 17 timeouts, exhausted the pool, and hung at 0% CPU for two
        hours holding 12 dead sockets — silently, because nothing raised.

        The async client has no thread pool to exhaust, and cancellation
        propagates to the transport, so a timeout genuinely releases the
        connection.
        """
        from google.genai import types

        config: dict[str, Any] = {
            "task_type": task_type,
            "output_dimensionality": self.dimensions,
        }
        if not sdk_retry:
            # The query path owns its whole budget: one retry at most, inside
            # a deadline measured in hundreds of milliseconds. An SDK retry
            # with a one-second initial delay cannot fit in it.
            config["http_options"] = types.HttpOptions(
                timeout=int(self.query_timeout_s * 1000) + 1_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            )
        response = await self._client.aio.models.embed_content(
            model=self.model,
            contents=list(texts),  # type: ignore[arg-type]  # SDK stub is narrower than runtime
            config=types.EmbedContentConfig(**config),
        )
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise ProviderError("gemini returned no embeddings")
        out: list[Vector] = []
        for item in embeddings:
            values = getattr(item, "values", None)
            if values is None:
                raise ProviderError("gemini embedding entry has no values")
            out.append([float(x) for x in values])
        return out

    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        return await self._call(texts, self._task_type)

    async def embed_query(self, text: str) -> Vector:
        """Embed a search query with the query-side task type.

        Retrieval embeddings are asymmetric: matching a RETRIEVAL_QUERY vector
        against RETRIEVAL_DOCUMENT vectors is what the model was trained for.
        """
        # Normalized once, and the NORMALIZED text is what gets embedded: the
        # cache key and the vector then always describe the same string, so a
        # hit can never serve one spelling's vector for another's question.
        query = normalize_query(text)
        if not query:
            raise ValueError("query is empty or whitespace-only")
        cached = self._query_cache_get(query)
        if cached is not None:
            return cached
        # `_query_with_deadline` validates the vector count and dimensions.
        vec = await self._query_with_deadline(
            lambda: self._call([query], TASK_QUERY, sdk_retry=False)
        )
        self._query_cache_put(query, vec)
        return vec

    # -- failure classification ------------------------------------------

    def classify_error(self, exc: BaseException) -> RetryDecision:
        """Sort a google-genai failure by who, if anyone, should retry it.

        * 429: MAPI retries, waiting at least as long as the server asked
          (Retry-After header, else the RetryInfo detail Gemini attaches).
        * 5xx and dropped connections: the SDK has already retried with
          backoff; MAPI may retry once more on top, as for any unknown
          failure, which bounds a persistent outage at a few attempts.
        * 400/401/403/404 and friends: nobody. A retired model, a bad key or
          an oversized batch answers the same way every time.
        """
        try:
            from google.genai import errors as genai_errors
        except ImportError:  # pragma: no cover - the extra is installed if we exist
            return super().classify_error(exc)
        if not isinstance(exc, genai_errors.APIError):
            return super().classify_error(exc)
        code = int(exc.code or 0)
        if code == 429:
            return RetryDecision(
                retryable=True,
                retry_after=retry_after_seconds(exc),
                rate_limited=True,
                status=code,
            )
        if code in _NEVER_RETRY or (400 <= code < 500 and code != 408):
            return RetryDecision(retryable=False, status=code)
        return RetryDecision(retryable=True, status=code or None)


def _pool_args(keepalive_s: float) -> dict[str, Any]:
    """httpx client args for the SDK: its own pool sizes, a longer idle life."""
    import httpx

    return {
        "limits": httpx.Limits(
            max_connections=100, max_keepalive_connections=20, keepalive_expiry=keepalive_s
        )
    }


def retry_after_seconds(exc: Any) -> float | None:
    """The wait a 429 asked for, from the header or the error body.

    Gemini states it in two places and not always both: an HTTP
    `Retry-After` (seconds or an HTTP date) and a `google.rpc.RetryInfo`
    entry in the error details (`"retryDelay": "34s"`). None when neither is
    present or parseable -- the caller then falls back to its own backoff.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    if raw:
        value = str(raw).strip()
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError):
                when = None
            if when is not None:
                return max(0.0, when.timestamp() - time.time())

    details = getattr(exc, "details", None)
    body = details.get("error", details) if isinstance(details, dict) else None
    entries = body.get("details") if isinstance(body, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("@type", "")).endswith("google.rpc.RetryInfo"):
            continue
        match = _RETRY_DELAY.match(str(entry.get("retryDelay", "")))
        if match:
            return float(match.group(1))
    return None


__all__ = [
    "DEFAULT_MODEL",
    "DEVELOPER_API_MAX_BATCH",
    "TASK_DOCUMENT",
    "TASK_QUERY",
    "GeminiEmbedder",
    "retry_after_seconds",
]
