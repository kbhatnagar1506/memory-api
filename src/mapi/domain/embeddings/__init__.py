"""Embedding providers and the factory that selects one from settings."""

from __future__ import annotations

from ...config import EmbeddingBackend, Settings
from ...core.errors import ConfigurationError
from .base import (
    EmbeddingProvider,
    EmbeddingResult,
    Vector,
    cosine_similarity,
    l2_normalize,
    normalize_query,
)
from .deterministic import DeterministicEmbedder


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Construct the configured provider. Import errors surface as config errors."""
    common = {
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "batch_size": settings.embedding_batch_size,
        "timeout_s": settings.embedding_timeout_s,
        "cache_size": settings.embedding_cache_size,
    }
    backend = settings.embedding_backend
    if backend is EmbeddingBackend.DETERMINISTIC:
        return DeterministicEmbedder(**common)  # type: ignore[arg-type]
    if backend is EmbeddingBackend.GEMINI:
        from .gemini import GeminiEmbedder

        return GeminiEmbedder(
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            api_key=settings.gemini_api_key,
            query_cache_size=settings.query_embedding_cache_size,
            **common,  # type: ignore[arg-type]
        )
    if backend is EmbeddingBackend.OPENAI:
        from .openai import OpenAIEmbedder

        return OpenAIEmbedder(api_key=settings.openai_api_key, **common)  # type: ignore[arg-type]
    raise ConfigurationError(f"unknown embedding backend: {backend}")


__all__ = [
    "DeterministicEmbedder",
    "EmbeddingProvider",
    "EmbeddingResult",
    "Vector",
    "build_embedder",
    "cosine_similarity",
    "l2_normalize",
    "normalize_query",
]
