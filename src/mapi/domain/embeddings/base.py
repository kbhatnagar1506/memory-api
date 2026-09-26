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
import unicodedata
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from ...core.errors import ConfigurationError, ProviderError, ProviderTimeoutError
from ...core.logging import get_logger

log = get_logger(__name__)

Vector = list[float]


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


def normalize_query(text: str) -> str:
    """The form of a query used both as its cache key and as what gets embedded.

    NFKC folds compatibility characters (full-width letters, ligatures, the
    non-breaking space a phone keyboard inserts) into their plain forms, and
    whitespace is collapsed, so "who knows  Rust?" and "who knows Rust? " are
    one question. Case is kept: the embedding model distinguishes "Go" from
    "go", and a cache that merged them would answer one with the other's vector.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


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
        query_cache_size: int = 0,
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
        #: Query-side LRU, separate from `_cache`; see `_query_cache_get`.
        self._query_cache: OrderedDict[str, array[float]] = OrderedDict()
        self._query_cache_size = max(0, query_cache_size)

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

    # -- query cache -------------------------------------------------------
    #
    # A second LRU for QUERY vectors, for two reasons the document cache
    # cannot serve:
    #
    #   * Eviction pressure. Documents arrive in bulk -- one Muse upload is
    #     hundreds of chunks -- and share `_cache` with nothing that repeats.
    #     The questions a matcher asks every few minutes are exactly the
    #     entries worth keeping, and one ingest used to push them all out.
    #   * Size. A `list[float]` of 768 Python floats measured ~31 KB; the same
    #     vector as `array('f')` is ~3 KB. float32 is what pgvector stores
    #     anyway, so the precision given up here was never used downstream.
    #
    # Keys are normalized (`normalize_query`), and the provider embeds the
    # normalized text, so two spellings that share a key also share the
    # vector they would have produced.

    def _query_cache_key(self, normalized: str) -> str:
        return f"{self.name}:{self.model}:{self.dimensions}:{normalized}"

    def _query_cache_get(self, normalized: str) -> Vector | None:
        if self._query_cache_size == 0:
            return None
        key = self._query_cache_key(normalized)
        packed = self._query_cache.get(key)
        if packed is None:
            return None
        self._query_cache.move_to_end(key)
        # A fresh list per hit: callers are free to mutate what they get, and
        # a shared list would let one request corrupt the next one's query.
        return packed.tolist()

    def _query_cache_put(self, normalized: str, vec: Vector) -> None:
        if self._query_cache_size == 0:
            return
        key = self._query_cache_key(normalized)
        self._query_cache[key] = array("f", vec)
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)

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

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = await self._with_retries([t for _, t in batch])
            for (idx, text), vec in zip(batch, vectors, strict=True):
                resolved[idx] = vec
                self._cache_put(text, vec)

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

    async def _with_retries(self, texts: list[str]) -> list[Vector]:
        delay = 0.25
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = await asyncio.wait_for(self._embed_batch(texts), timeout=self.timeout_s)
                return self._validate(raw, len(texts))
            except TimeoutError as exc:
                last = ProviderTimeoutError(f"{self.name} timed out after {self.timeout_s}s")
                log.warning("embedding_timeout", provider=self.name, attempt=attempt)
                _ = exc
            except ProviderError as exc:
                # A dimension mismatch will not fix itself on retry.
                if "dimensions" in str(exc):
                    raise
                last = exc
                log.warning(
                    "embedding_failed",
                    provider=self.name,
                    attempt=attempt,
                    error=str(exc)[:200],
                )
            except Exception as exc:
                last = ProviderError(f"{self.name}: {type(exc).__name__}: {exc}")
                log.warning(
                    "embedding_error",
                    provider=self.name,
                    attempt=attempt,
                    error=str(exc)[:200],
                )
            if attempt < self.max_attempts:
                await asyncio.sleep(delay)
                delay *= 2
        raise last or ProviderError(f"{self.name} failed with no diagnostic")

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
    "EmbeddingProvider",
    "EmbeddingResult",
    "Vector",
    "cosine_similarity",
    "l2_normalize",
    "normalize_query",
]
