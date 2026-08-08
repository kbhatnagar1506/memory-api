"""The retrieval pipeline.

    query
      -> embed (query-side task type)
      -> vector search  ┐
      -> lexical search ┘  (concurrent)
      -> reciprocal rank fusion
      -> hydrate memories
      -> rerank (heuristic or LLM)
      -> recency decay
      -> supersession suppression
      -> MMR diversification
      -> top-k

Ordering rationale, since it is the part that is easy to get subtly wrong:

  * Fusion comes before reranking so the reranker sees candidates that *either*
    strategy liked, not just the vector winner.
  * Reranking comes before decay because a reranker judges topical relevance; it
    has no idea how old anything is, and feeding it decayed scores would let
    age leak into a judgement that should be about meaning.
  * Decay comes before MMR so diversification trades off the final relevance,
    not a pre-decay one.
  * Supersession suppression runs last among the filters: it needs the full
    candidate set to know whether a superseding memory is *also* in the results,
    which is the only case where hiding the old one is safe.

Every stage records into `explain`, so a result can always answer "why is this
here, and why here rather than three places up".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from ...core.logging import get_logger
from ...store.base import MemoryFilter, MemoryStore
from ..embeddings.base import EmbeddingProvider, Vector
from ..models import Memory, MemoryStatus, RelationType, ScoredMemory
from .decay import apply_decay
from .fusion import RankedList, reciprocal_rank_fusion
from .mmr import MMRCandidate, maximal_marginal_relevance
from .rerank import RerankCandidate, Reranker

log = get_logger(__name__)


@dataclass(slots=True)
class SearchRequest:
    query: str
    org_id: str
    space_id: str
    limit: int = 10
    filters: MemoryFilter = field(default_factory=MemoryFilter)
    #: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_lambda: float = 0.7
    half_life_days: float = 180.0
    use_decay: bool = True
    use_rerank: bool = True
    use_mmr: bool = True
    #: Return memories that a newer memory has superseded.
    include_superseded: bool = False
    candidate_multiplier: int = 6
    rerank_candidates: int = 32
    rrf_k: int = 60
    #: Weight of each first-stage strategy in fusion.
    vector_weight: float = 1.0
    lexical_weight: float = 1.0
    #: Drop results scoring below this after all stages. Vector search is a
    #: nearest-neighbour operation, not a threshold: without a floor, a query
    #: matching nothing still returns the k least-unrelated memories, and an
    #: agent has no way to tell that from a real answer.
    min_score: float = 0.0


@dataclass(slots=True)
class SearchResponse:
    results: list[ScoredMemory]
    query: str
    total_candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    #: True when a reranker was asked but degraded to first-stage order.
    rerank_degraded: bool = False
    strategies: list[str] = field(default_factory=list)


class RetrievalPipeline:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    async def _embed_query(self, query: str) -> Vector | None:
        """Query-side embedding. A provider failure degrades to lexical-only.

        Losing vector search is a quality regression; failing the request is an
        outage. The response records which strategies actually ran.
        """
        try:
            embed_query = getattr(self.embedder, "embed_query", None)
            if embed_query is not None:
                return await embed_query(query)  # type: ignore[no-any-return]
            return await self.embedder.embed_one(query)
        except Exception as exc:
            log.warning("query_embedding_failed", error=str(exc)[:200])
            return None

    async def search(self, request: SearchRequest) -> SearchResponse:
        timings: dict[str, float] = {}
        loop = asyncio.get_running_loop()

        query = request.query.strip()
        if not query:
            return SearchResponse([], request.query, 0, timings, False, [])
        if request.limit <= 0:
            return SearchResponse([], request.query, 0, timings, False, [])

        fetch = max(request.limit * max(request.candidate_multiplier, 1), request.limit)

        # -- stage 1: candidate generation, concurrent -----------------------
        t0 = loop.time()
        embedding = await self._embed_query(query)
        timings["embed_ms"] = (loop.time() - t0) * 1000

        t0 = loop.time()
        vector_task = (
            self.store.vector_search(
                request.org_id,
                request.space_id,
                embedding,
                limit=fetch,
                filters=request.filters,
            )
            if embedding is not None
            else None
        )
        lexical_task = self.store.lexical_search(
            request.org_id,
            request.space_id,
            query,
            limit=fetch,
            filters=request.filters,
        )
        if vector_task is not None:
            vector_hits, lexical_hits = await asyncio.gather(vector_task, lexical_task)
        else:
            vector_hits, lexical_hits = [], await lexical_task
        timings["candidates_ms"] = (loop.time() - t0) * 1000

        strategies: list[str] = []
        if vector_hits:
            strategies.append("vector")
        if lexical_hits:
            strategies.append("lexical")
        if not vector_hits and not lexical_hits:
            return SearchResponse([], request.query, 0, timings, False, strategies)

        # -- stage 2: fusion --------------------------------------------------
        t0 = loop.time()
        ranked = []
        if vector_hits:
            ranked.append(
                RankedList(
                    "vector",
                    [h.memory_id for h in vector_hits],
                    {h.memory_id: h.score for h in vector_hits},
                    weight=request.vector_weight,
                )
            )
        if lexical_hits:
            ranked.append(
                RankedList(
                    "lexical",
                    [h.memory_id for h in lexical_hits],
                    {h.memory_id: h.score for h in lexical_hits},
                    weight=request.lexical_weight,
                )
            )
        fused = reciprocal_rank_fusion(ranked, k=request.rrf_k)
        timings["fusion_ms"] = (loop.time() - t0) * 1000

        # -- stage 3: hydrate --------------------------------------------------
        t0 = loop.time()
        pool = fused[: max(request.rerank_candidates, request.limit)]
        memories = await self.store.get_memories(
            request.org_id, request.space_id, [f.id for f in pool]
        )
        timings["hydrate_ms"] = (loop.time() - t0) * 1000

        best_chunk = {h.memory_id: (h.chunk_id, h.text) for h in lexical_hits}
        best_chunk.update({h.memory_id: (h.chunk_id, h.text) for h in vector_hits})
        vector_scores = {h.memory_id: h.score for h in vector_hits}
        lexical_scores = {h.memory_id: h.score for h in lexical_hits}

        scored: list[ScoredMemory] = []
        for item in pool:
            memory = memories.get(item.id)
            if memory is None:
                # Deleted between the search and the hydrate. Skip rather than
                # emit a dangling id.
                continue
            chunk_id, chunk_text = best_chunk.get(item.id, (None, ""))
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=item.score,
                    fusion_score=item.score,
                    vector_score=vector_scores.get(item.id),
                    lexical_score=lexical_scores.get(item.id),
                    matched_chunk_id=chunk_id,
                    matched_text=chunk_text,
                    explain=item.explain(),
                )
            )

        if not scored:
            return SearchResponse([], request.query, len(fused), timings, False, strategies)

        # -- stage 4: rerank ---------------------------------------------------
        degraded = False
        if request.use_rerank and len(scored) > 1:
            t0 = loop.time()
            rerank_candidates = [
                RerankCandidate(
                    id=s.memory.id,
                    text=s.matched_text or s.memory.summary or s.memory.content,
                    prior_score=s.score,
                )
                for s in scored
            ]
            reranked = await self.reranker.rerank(
                query, rerank_candidates, limit=len(rerank_candidates)
            )
            timings["rerank_ms"] = (loop.time() - t0) * 1000
            degraded = bool(reranked) and not reranked[0].reranked

            positions = {r.id: r for r in reranked}
            for s in scored:
                r = positions.get(s.memory.id)
                if r is None:
                    continue
                s.rerank_score = r.score
                s.score = r.score
                s.explain.append(
                    f"{self.reranker.name} rerank rank {r.rank}"
                    + ("" if r.reranked else " (fallback)")
                )
            scored.sort(key=lambda s: (-s.score, s.memory.id))

        # -- stage 5: recency decay ---------------------------------------------
        if request.use_decay:
            now = datetime.now(tz=None).astimezone()
            for s in scored:
                decayed, factor = apply_decay(
                    s.score,
                    s.memory.occurred_at,
                    now=now,
                    half_life_days=request.half_life_days,
                )
                s.recency_factor = factor
                s.score = decayed
                s.explain.append(f"recency x{factor:.3f}")
            scored.sort(key=lambda s: (-s.score, s.memory.id))

        # -- stage 6: supersession suppression -----------------------------------
        if not request.include_superseded:
            scored = self._suppress_superseded(scored)

        # -- stage 7: MMR diversification ----------------------------------------
        t0 = loop.time()
        if request.use_mmr and len(scored) > 1:
            by_id = {s.memory.id: s for s in scored}
            mmr_candidates = [
                MMRCandidate(
                    id=s.memory.id,
                    relevance=s.score,
                    embedding=self._representative_embedding(s.memory),
                )
                for s in scored
            ]
            selection = maximal_marginal_relevance(
                mmr_candidates, limit=request.limit, lambda_=request.mmr_lambda
            )
            ordered = []
            for sel in selection:
                s = by_id[sel.id]
                if sel.redundancy > 0:
                    s.explain.append(f"mmr redundancy {sel.redundancy:.3f}")
                ordered.append(s)
            scored = ordered
        else:
            scored = scored[: request.limit]
        timings["mmr_ms"] = (loop.time() - t0) * 1000

        if request.min_score > 0.0:
            kept = [s for s in scored if s.score >= request.min_score]
            if len(kept) != len(scored):
                log.debug(
                    "min_score_filtered",
                    dropped=len(scored) - len(kept),
                    threshold=request.min_score,
                )
            scored = kept

        return SearchResponse(
            results=scored[: request.limit],
            query=request.query,
            total_candidates=len(fused),
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            rerank_degraded=degraded,
            strategies=strategies,
        )

    @staticmethod
    def _representative_embedding(memory: Memory) -> Vector | None:
        for chunk in memory.chunks:
            if chunk.embedding is not None:
                return chunk.embedding
        return None

    @staticmethod
    def _suppress_superseded(scored: list[ScoredMemory]) -> list[ScoredMemory]:
        """Drop memories that a *present* result supersedes.

        Only suppress when the replacement is in the same result set. If the
        newer memory did not match the query, hiding the older one would answer
        the question with silence, which is worse than answering with a fact
        that is merely stale.
        """
        present = {s.memory.id for s in scored}
        superseded_here: set[str] = set()
        for s in scored:
            for relation in s.memory.relations:
                if relation.type is RelationType.SUPERSEDES and relation.target_id in present:
                    superseded_here.add(relation.target_id)

        out: list[ScoredMemory] = []
        for s in scored:
            if s.memory.id in superseded_here:
                continue
            if s.memory.status is MemoryStatus.SUPERSEDED:
                continue
            out.append(s)
        return out


__all__ = ["RetrievalPipeline", "SearchRequest", "SearchResponse"]
