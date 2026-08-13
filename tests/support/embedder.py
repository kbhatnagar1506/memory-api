"""An embedder that returns the vectors you told it to return.

Pairs with `vectors.py`: that module builds a geometry, this one attaches it to
text so the geometry survives a trip through `service.ingest` and
`RetrievalPipeline`.

TWO DESIGN POINTS THAT MATTER MORE THAN THE CLASS ITSELF.

**1. Markers, not exact strings.** `settings.contextual_embedding` defaults to
True, so `service.ingest` does not embed the text you wrote -- it embeds
`for_embedding(chunk.text, header)`, wrapping it in a `<context>` block
carrying the date, an optional title, the source and the first two tags
(`domain/embeddings/context.py`). Chunking can also split a long memory before
the embedder ever sees it. An exact-string lookup table therefore misses on
almost every realistic write. So the primary lookup is: find the first
registered MARKER that appears as a substring. Put a marker in the content
(`"g0007"`) and it survives headers, chunking and clipping.

**2. A miss must never raise.** This is the subtle one and it is what makes the
class safe. `EmbeddingProvider._with_retries` wraps any exception in
`ProviderError`, and `RetrievalPipeline._embed_query` catches `ProviderError`
and returns `None` -- silently degrading the search to lexical-only. So an
embedder that raised on an unregistered query would produce a *green* test that
never ran vector search at all, which is worse than a red one. Instead a miss
returns a reserved direction orthogonal to everything the corpus uses, and
records the text in `misses`. An autouse fixture asserts `misses == []` and
names the unregistered string.

There is deliberately **no `embed_query`**. `RetrievalPipeline` prefers it when
the provider defines it, and an asymmetric query-side transform would quietly
invalidate every cosine this harness claims to control.
"""

from __future__ import annotations

from mapi.domain.embeddings.base import EmbeddingProvider, Vector

from .vectors import DIMS, axis

#: Where a miss lands. The last axis, so it is orthogonal to the anchor and to
#: every off-direction a corpus is expected to use -- a miss can therefore never
#: accidentally rank first and make a broken test look like a passing one.
MISS_AXIS = DIMS - 1


class ScriptedEmbedder(EmbeddingProvider):
    """Maps registered text (or markers within it) to hand-specified vectors."""

    def __init__(self, *, dimensions: int = DIMS, cache_size: int = 0) -> None:
        super().__init__(
            model="scripted-v1",
            dimensions=dimensions,
            batch_size=256,
            timeout_s=1.0,
            max_attempts=1,
            cache_size=cache_size,
        )
        self._exact: dict[str, Vector] = {}
        # Insertion-ordered, and longest-marker-first at lookup time, so
        # registering both "g1" and "g10" resolves the way a reader expects.
        self._markers: dict[str, Vector] = {}
        self._misses: list[str] = []
        self._miss_vector = axis(MISS_AXIS, dimensions)

    @property
    def name(self) -> str:
        return "scripted"

    # -- registration ------------------------------------------------------

    def register(self, text: str, vector: Vector) -> None:
        """Bind an exact string. Tried before markers."""
        self._check(vector)
        self._exact[text] = list(vector)

    def register_marker(self, marker: str, vector: Vector) -> None:
        """Bind any text CONTAINING `marker`. The mode that survives ingest."""
        if not marker:
            raise ValueError("marker must not be empty")
        self._check(vector)
        self._markers[marker] = list(vector)

    def _check(self, vector: Vector) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"vector has {len(vector)} dimensions, embedder expects {self.dimensions}"
            )

    # -- misses ------------------------------------------------------------

    @property
    def misses(self) -> list[str]:
        """Texts that matched nothing. MUST be asserted empty by the caller.

        A non-empty `misses` means some memory or query was embedded into the
        reserved miss direction, so any ranking assertion in that test was
        measuring the sentinel rather than the corpus.
        """
        return list(self._misses)

    def clear_misses(self) -> None:
        self._misses.clear()

    # -- the provider contract --------------------------------------------

    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        return [self._lookup(text) for text in texts]

    def _lookup(self, text: str) -> Vector:
        exact = self._exact.get(text)
        if exact is not None:
            return list(exact)
        for marker in sorted(self._markers, key=len, reverse=True):
            if marker in text:
                return list(self._markers[marker])
        self._misses.append(text)
        return list(self._miss_vector)


__all__ = ["MISS_AXIS", "ScriptedEmbedder"]
