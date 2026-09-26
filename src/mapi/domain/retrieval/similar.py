"""Nearest neighbours of a stored memory, in another space.

"Who should I meet" is, underneath, one question asked many times: which
directory entries sit closest to what this person wrote about themselves. The
person's own text is already embedded -- it is a stored memory -- so searching
with it needs no provider call at all, only the vector sitting in the table.
That is the whole reason this exists next to `search`: the matcher's inner loop
becomes one indexed query instead of an embedding round trip per facet.

Deliberately NOT the search pipeline. There is no query text, so there is
nothing to classify, nothing for a lexical arm to match and nothing to fuse;
what remains is one ANN scan, a hydrate, and the per-source cap and score
floor a caller needs to turn neighbours into distinct people.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ...store.base import MemoryFilter, MemoryStore
from ..embeddings.base import Vector
from ..models import ScoredMemory
from .pipeline import _cap_per_source

#: Hard ceiling on candidates fetched for a capped request, matching the
#: search pipeline's: past this the scan is an export.
_MAX_FETCH = 600


@dataclass(slots=True)
class SimilarRequest:
    org_id: str
    #: Where to look for neighbours.
    space_id: str
    #: The vector to look with -- the source memory's first chunk.
    vector: Vector
    #: Never returned as its own neighbour, when source and target coincide.
    source_memory_id: str = ""
    limit: int = 10
    filters: MemoryFilter = field(default_factory=MemoryFilter)
    #: Most results one source document may contribute; 0 is off. With the
    #: directory's entries carrying `doc_id=<person>`, 1 means one row per person.
    max_per_source: int = 0
    #: Floor on cosine similarity.
    min_score: float = 0.0
    #: Candidates fetched per requested result when capping, so the cap has
    #: rows to choose among rather than emptying the window.
    candidate_multiplier: int = 6


@dataclass(slots=True)
class SimilarResponse:
    results: list[ScoredMemory]
    timings_ms: dict[str, float] = field(default_factory=dict)


async def find_similar(store: MemoryStore, request: SimilarRequest) -> SimilarResponse:
    """Memories in `request.space_id` nearest `request.vector`, best first.

    Scores are raw cosine similarity -- `score` and `vector_score` agree -- so a
    `min_score` fitted on search's `vector_score` means the same thing here.
    """
    loop = asyncio.get_running_loop()
    timings: dict[str, float] = {}
    if request.limit <= 0 or not request.vector:
        return SimilarResponse([], timings)

    # One spare row for the source itself, which is its own nearest neighbour
    # whenever it lives in the searched space.
    wanted = request.limit + 1
    if request.max_per_source > 0:
        wanted = request.limit * max(request.candidate_multiplier, 1) + 1
    wanted = min(wanted, _MAX_FETCH)

    t0 = loop.time()
    hits = await store.vector_search(
        request.org_id,
        request.space_id,
        request.vector,
        limit=wanted,
        filters=request.filters,
    )
    timings["candidates_ms"] = (loop.time() - t0) * 1000
    hits = [h for h in hits if h.memory_id != request.source_memory_id]
    if request.min_score > 0.0:
        # Before hydrating: a floor is the cheapest filter there is, and
        # everything under it would be fetched only to be dropped.
        hits = [h for h in hits if h.score >= request.min_score]
    if not hits:
        return SimilarResponse([], timings)

    t0 = loop.time()
    memories = await store.get_memories(
        request.org_id,
        request.space_id,
        [h.memory_id for h in hits],
        with_embeddings=False,
    )
    timings["hydrate_ms"] = (loop.time() - t0) * 1000

    scored = [
        ScoredMemory(
            memory=memories[h.memory_id],
            score=h.score,
            vector_score=h.score,
            matched_chunk_id=h.chunk_id,
            matched_text=h.text,
            explain=[f"cosine {h.score:.4f} to the source"],
        )
        for h in hits
        if h.memory_id in memories
    ]
    if request.max_per_source > 0:
        scored = _cap_per_source(scored, request.max_per_source)
    return SimilarResponse(scored[: request.limit], timings)


__all__ = ["SimilarRequest", "SimilarResponse", "find_similar"]
