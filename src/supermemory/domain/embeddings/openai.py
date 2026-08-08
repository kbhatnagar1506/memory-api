"""OpenAI embeddings. Requires OPENAI_API_KEY; there is no ADC path."""

from __future__ import annotations

import asyncio
import os

from ...core.errors import ConfigurationError, ProviderError
from .base import EmbeddingProvider, Vector


class OpenAIEmbedder(EmbeddingProvider):
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 768,
        batch_size: int = 64,
        timeout_s: float = 20.0,
        cache_size: int = 0,
        api_key: str | None = None,
        **_: object,
    ) -> None:
        super().__init__(
            model=model, dimensions=dimensions, batch_size=batch_size,
            timeout_s=timeout_s, max_attempts=3, cache_size=cache_size,
        )
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "openai is not installed. Install the 'openai' extra."
            ) from exc
        if not (api_key or os.getenv("OPENAI_API_KEY")):
            raise ConfigurationError("OPENAI_API_KEY is required for the openai backend")
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @property
    def name(self) -> str:
        return "openai"

    def _call(self, texts: list[str]) -> list[Vector]:
        resp = self._client.embeddings.create(
            model=self.model, input=list(texts), dimensions=self.dimensions,
        )
        if not resp.data:
            raise ProviderError("openai returned no embeddings")
        # The API does not guarantee ordering; it returns an index per item.
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [[float(x) for x in d.embedding] for d in ordered]

    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        return await asyncio.to_thread(self._call, texts)


__all__ = ["OpenAIEmbedder"]
