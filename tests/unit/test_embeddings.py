"""Embedding provider contract and vector maths."""

from __future__ import annotations

import math

import pytest

from supermemory.core.errors import ProviderError
from supermemory.domain.embeddings import DeterministicEmbedder, cosine_similarity
from supermemory.domain.embeddings.base import EmbeddingProvider, l2_normalize


async def test_vectors_are_unit_length(embedder: DeterministicEmbedder) -> None:
    result = await embedder.embed(["hello world"])
    norm = math.sqrt(sum(x * x for x in result.vectors[0]))
    assert norm == pytest.approx(1.0, abs=1e-6)


async def test_similar_texts_score_higher_than_unrelated(embedder) -> None:
    vectors = (
        await embedder.embed(
            ["the cat sat on the mat", "a cat sat upon the mat", "quantum chromodynamics"]
        )
    ).vectors
    assert cosine_similarity(vectors[0], vectors[1]) > cosine_similarity(vectors[0], vectors[2])


async def test_deterministic_across_calls(embedder) -> None:
    a = (await embedder.embed(["stable input"])).vectors[0]
    b = (await embedder.embed(["stable input"])).vectors[0]
    assert a == b


async def test_order_is_preserved(embedder) -> None:
    texts = [f"document number {i}" for i in range(10)]
    vectors = (await embedder.embed(texts)).vectors
    for i, text in enumerate(texts):
        assert vectors[i] == (await embedder.embed([text])).vectors[0]


async def test_empty_input_returns_empty(embedder) -> None:
    result = await embedder.embed([])
    assert result.vectors == []


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
async def test_blank_text_is_rejected(embedder, bad: str) -> None:
    with pytest.raises(ValueError, match="empty or whitespace"):
        await embedder.embed([bad])


async def test_non_string_input_is_rejected(embedder) -> None:
    with pytest.raises(ValueError, match="expected str"):
        await embedder.embed([123])  # type: ignore[list-item]


async def test_symbol_only_input_still_yields_a_usable_vector(embedder) -> None:
    """Zero vectors make cosine similarity undefined, so they must not happen."""
    vector = (await embedder.embed(["###"])).vectors[0]
    assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0, abs=1e-6)


async def test_batching_matches_single_calls() -> None:
    small = DeterministicEmbedder(dimensions=64, batch_size=2)
    texts = [f"text {i}" for i in range(7)]
    batched = (await small.embed(texts)).vectors
    for i, text in enumerate(texts):
        assert batched[i] == (await small.embed([text])).vectors[0]


async def test_cache_returns_identical_vectors_and_counts_hits() -> None:
    cached = DeterministicEmbedder(dimensions=64, cache_size=16)
    first = await cached.embed(["repeated"])
    second = await cached.embed(["repeated"])
    assert first.vectors == second.vectors
    assert second.cache_hits == 1


def test_cosine_similarity_bounds() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_similarity_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        cosine_similarity([], [])


def test_cosine_similarity_of_zero_vector_is_zero_not_nan() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero or non-finite"):
        l2_normalize([0.0, 0.0])


class _BadDimensions(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "bad"

    async def _embed_batch(self, texts):
        return [[0.1] * 5 for _ in texts]


class _ZeroVector(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "zero"

    async def _embed_batch(self, texts):
        return [[0.0] * self.dimensions for _ in texts]


class _WrongCount(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "count"

    async def _embed_batch(self, texts):
        return [[0.1] * self.dimensions]


async def test_dimension_mismatch_is_reported_and_not_retried() -> None:
    """A model change that alters dimensions corrupts an index silently."""
    provider = _BadDimensions(model="m", dimensions=64, max_attempts=3)
    with pytest.raises(ProviderError, match="dimensions"):
        await provider.embed(["x"])


async def test_zero_vector_from_provider_is_rejected() -> None:
    provider = _ZeroVector(model="m", dimensions=8, max_attempts=1)
    with pytest.raises(ProviderError, match="zero vector"):
        await provider.embed(["x"])


async def test_vector_count_mismatch_is_rejected() -> None:
    provider = _WrongCount(model="m", dimensions=8, max_attempts=1)
    with pytest.raises(ProviderError, match="returned 1 vectors"):
        await provider.embed(["a", "b"])
