"""Deterministic offline embeddings.

This is not a mock. It is a real feature-hashing embedder: character n-grams and
word unigrams are hashed into a fixed-dimensional space with signed buckets,
then L2-normalized. Texts that share vocabulary land near each other, so the
whole retrieval pipeline — vector search, fusion, MMR, dedup — can be tested
end to end with genuine similarity structure and no network, no credentials and
no flakiness.

What it is not: semantic. "car" and "automobile" are unrelated here. That is
exactly why `Settings.validate_production()` refuses to boot production with it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict

from .base import EmbeddingProvider, Vector

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NGRAM_SIZES = (3, 4)


def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    """Map a token to a bucket and a sign, both from the same stable digest."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimensions, 1.0 if (value >> 63) & 1 else -1.0


class DeterministicEmbedder(EmbeddingProvider):
    """Feature-hashing embedder. Stable across processes, platforms and runs."""

    def __init__(
        self,
        *,
        dimensions: int = 768,
        model: str = "deterministic-hash-v1",
        batch_size: int = 256,
        cache_size: int = 0,
        **_: object,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
            timeout_s=5.0,
            max_attempts=1,
            cache_size=cache_size,
        )

    @property
    def name(self) -> str:
        return "deterministic"

    def _vector(self, text: str) -> Vector:
        normalized = text.casefold()
        weights: dict[int, float] = defaultdict(float)

        words = _WORD_RE.findall(normalized)
        for word in words:
            idx, sign = _bucket(f"w:{word}", self.dimensions)
            # Sub-linear term weighting, as in TF-IDF: the tenth occurrence of a
            # word says much less than the first.
            weights[idx] += sign

        padded = f" {' '.join(words)} " if words else f" {normalized} "
        for size in _NGRAM_SIZES:
            if len(padded) < size:
                continue
            for i in range(len(padded) - size + 1):
                idx, sign = _bucket(f"c{size}:{padded[i : i + size]}", self.dimensions)
                weights[idx] += sign * 0.5

        vec = [0.0] * self.dimensions
        for idx, raw in weights.items():
            vec[idx] = math.copysign(math.log1p(abs(raw)), raw)

        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            # Input with no word characters and no n-grams (a lone symbol).
            # Fall back to a stable non-zero direction so cosine stays defined.
            idx, sign = _bucket(f"fallback:{normalized}", self.dimensions)
            vec[idx] = sign
            norm = 1.0
        return [x / norm for x in vec]

    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        return [self._vector(t) for t in texts]

    async def health(self) -> bool:
        return True


__all__ = ["DeterministicEmbedder"]
