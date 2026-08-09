"""A real embedder that answers from disk after the first run.

The demo graph is only worth looking at if its edges are ones the system
actually derived. The deterministic embedder is a hash: it scores lexically
close strings highly and everything else near zero, so on real chat turns it
finds no supersessions and no contradictions, and the picture is a few hundred
disconnected dots. That would be an honest rendering of a useless run.

So the seeder uses the same provider the benchmark uses, wrapped in the
benchmark's content-addressed cache. The first boot embeds (needs credentials
and about a minute); every boot after that is a cache hit and works offline.

On a miss with no credentials the wrapper does NOT silently fall back to the
hash -- mixing two embedding spaces in one index produces plausible, wrong
neighbours and no error. It raises, and the seeder says what to do.
"""

from __future__ import annotations

from pathlib import Path

from bench.cache import DiskVectorCache, cache_key

from mapi.domain.embeddings.base import EmbeddingProvider, Vector


class DiskCachedEmbedder(EmbeddingProvider):
    """Wraps a provider; reads and writes `bench/data/cache/vectors.sqlite`."""

    def __init__(self, inner: EmbeddingProvider, cache_path: str | Path) -> None:
        super().__init__(
            model=inner.model,
            dimensions=inner.dimensions,
            batch_size=inner.batch_size,
            timeout_s=inner.timeout_s,
        )
        self._inner = inner
        self._cache = DiskVectorCache(cache_path)

    @property
    def name(self) -> str:
        return f"cached:{self._inner.name}"

    @property
    def stats(self) -> dict[str, int]:
        return self._cache.stats()

    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        keys = [cache_key(self.model, self.dimensions, t) for t in texts]
        found = self._cache.get_many(keys)

        pairs = enumerate(zip(texts, keys, strict=True))
        misses = [(i, t) for i, (t, k) in pairs if k not in found]
        fresh: dict[int, Vector] = {}
        if misses:
            # `embed` (not `_embed_batch`) so the miss path keeps the base
            # class's retries and dimension checks.
            result = await self._inner.embed([t for _, t in misses])
            fresh = {i: v for (i, _), v in zip(misses, result.vectors, strict=True)}
            self._cache.put_many((keys[i], vector) for i, vector in fresh.items())

        return [fresh[i] if i in fresh else found[k] for i, k in enumerate(keys)]

    async def aclose(self) -> None:
        self._cache.close()
        await self._inner.aclose()


__all__ = ["DiskCachedEmbedder"]
