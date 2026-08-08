"""Google embeddings via Vertex AI (ADC) or the Gemini Developer API (key).

Vertex + Application Default Credentials is the default because it is what
production on GCP actually uses: no key material in the environment, rotation
handled by the platform. A `GEMINI_API_KEY` switches to the Developer API for
local work.

Two provider-specific details that matter:

  * Retrieval embeddings are asymmetric. Documents are embedded with
    `RETRIEVAL_DOCUMENT` and queries with `RETRIEVAL_QUERY`; using one task type
    for both measurably degrades recall. `embed_query` exists for that reason.
  * The SDK is synchronous, so calls run in a worker thread. Doing them inline
    would block the event loop for the entire batch.
"""

from __future__ import annotations

import asyncio
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

        key = api_key or os.getenv("GEMINI_API_KEY")
        if key:
            self._client = genai.Client(api_key=key)
            self.backend = "developer-api"
        else:
            proj = project or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not proj:
                raise ConfigurationError(
                    "Gemini embeddings need GEMINI_API_KEY, or "
                    "GOOGLE_CLOUD_PROJECT for Vertex AI with ADC "
                    "(gcloud auth application-default login)."
                )
            self._client = genai.Client(vertexai=True, project=proj, location=location)
            self.backend = "vertex-adc"
        self._task_type = TASK_DOCUMENT

    @property
    def name(self) -> str:
        return "gemini"

    def _call(self, texts: list[str], task_type: str) -> list[Vector]:
        from google.genai import types

        config: dict[str, Any] = {
            "task_type": task_type,
            "output_dimensionality": self.dimensions,
        }
        response = self._client.models.embed_content(
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
        # The SDK is blocking; keep it off the event loop.
        return await asyncio.to_thread(self._call, texts, self._task_type)

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
        raw = await asyncio.to_thread(self._call, [text], TASK_QUERY)
        vec = self._validate(raw, 1)[0]
        self._cache_put(f"__query__{text}", vec)
        return vec


__all__ = ["TASK_DOCUMENT", "TASK_QUERY", "GeminiEmbedder"]
