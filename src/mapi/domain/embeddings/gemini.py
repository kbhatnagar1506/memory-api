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
"""

from __future__ import annotations

import os
from typing import Any

from ...core.errors import ConfigurationError, ProviderError
from ...core.logging import get_logger
from .base import EmbeddingProvider, Vector

log = get_logger(__name__)

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class GeminiEmbedder(EmbeddingProvider):
    def __init__(
        self,
        *,
        model: str = "text-embedding-004",
        dimensions: int = 768,
        batch_size: int = 32,
        timeout_s: float = 20.0,
        cache_size: int = 0,
        project: str | None = None,
        location: str = "global",
        api_key: str | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
            timeout_s=timeout_s,
            max_attempts=3,
            cache_size=cache_size,
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
        http_options = genai_types.HttpOptions(timeout=int(timeout_s * 1000) + 5_000)

        key = api_key or os.getenv("GEMINI_API_KEY")
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

    async def _call(self, texts: list[str], task_type: str) -> list[Vector]:
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
        response = await self._client.aio.models.embed_content(
            model=self.model,
            contents=list(texts),
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
        if not text.strip():
            raise ValueError("query is empty or whitespace-only")
        cached = self._cache_get(f"__query__{text}")
        if cached is not None:
            return cached
        raw = await self._call([text], TASK_QUERY)
        vec = self._validate(raw, 1)[0]
        self._cache_put(f"__query__{text}", vec)
        return vec


__all__ = ["TASK_DOCUMENT", "TASK_QUERY", "GeminiEmbedder"]
