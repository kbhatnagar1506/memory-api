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
import math
from dataclasses import dataclass, field
from datetime import datetime

from ...core.logging import get_logger
from ...store.base import LexicalHit, MemoryFilter, MemoryStore, VectorHit
from ..embeddings.base import EmbeddingProvider, Vector
from ..models import Memory, MemoryStatus, RelationType, ScoredMemory
from .confidence import RetrievalConfidence, assess
from .decay import apply_decay
from .entities import salient_entities
from .expansion import NoopExpander, QueryExpander
from .fusion import FusedItem, RankedList, reciprocal_rank_fusion
from .mmr import MMRCandidate, maximal_marginal_relevance
from .rerank import RerankCandidate, Reranker

log = get_logger(__name__)


def _mean_unit_vector(vectors: list[Vector]) -> Vector:
    """Centroid of unit vectors, renormalized.

    Vectors arriving here are already L2-normalized by the embedding provider,
    so a plain mean is a fair blend; renormalizing keeps the result on the unit
    sphere where cosine similarity reduces to a dot product.
    """
    usable = [v for v in vectors if v]
    if not usable:
        raise ValueError("no vectors to average")
    if len(usable) == 1:
        return usable[0]
    width = len(usable[0])
    if any(len(v) != width for v in usable):
        raise ValueError("cannot average vectors of differing dimensions")
    summed = [sum(v[i] for v in usable) / len(usable) for i in range(width)]
    norm = math.sqrt(sum(x * x for x in summed))
    if norm == 0.0:
        # Diametrically opposed vectors cancel. Keep the query, which is the
        # component we trust.
        return usable[0]
    return [x / norm for x in summed]


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
    #: Default OFF on measured evidence. MMR suppresses near-duplicates, but
    #: measured twice on conversational corpora it changed no retrieval metric
    #: while costing 2.5x latency (61ms -> 110ms on LoCoMo). It is still the
    #: right tool for corpora that genuinely accumulate restatements; it is not
    #: free enough to be a default.
    use_mmr: bool = False
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
    #: Embed a hypothetical answer alongside the query (HyDE). Helps when the
    #: question's vocabulary does not overlap the corpus's — the case where
    #: pure dense retrieval is weakest. Costs one LLM call per query.
    use_expansion: bool = False
    #: Bridge to evidence that shares an ENTITY with a seed result but no
    #: vocabulary with the query. Measured need: 97% of multi-hop evidence is
    #: in a different session, median 204 turns away, unreachable by widening
    #: k or the context window. No LLM — one extra lexical lookup per entity.
    use_entity_expansion: bool = False
    #: How many entities to expand on. Each costs one indexed lexical lookup.
    entity_budget: int = 6
    #: Weight of the entity arm in fusion. Below 1.0 by default: an entity
    #: match is weaker evidence of relevance than a direct query match, and
    #: should break ties rather than dominate them.
    entity_weight: float = 0.6
    #: Names to never expand on. In a two-person dialogue the speakers appear
    #: in nearly every turn, so bridging on them returns the whole corpus.
    known_speakers: tuple[str, ...] = ()


@dataclass(slots=True)
class SearchResponse:
    results: list[ScoredMemory]
    query: str
    total_candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    #: True when a reranker was asked but degraded to first-stage order.
    rerank_degraded: bool = False
    strategies: list[str] = field(default_factory=list)
    #: Entities the bridging stage expanded on. Part of the explain surface:
    #: "why is this result here" must be answerable for bridged hits too.
    entities_used: list[str] = field(default_factory=list)
    #: Pairs of returned memory ids joined by a CONTRADICTS edge. Surfaced,
    #: not resolved: every competitor picks a winner invisibly (newest
    #: timestamp), which is indistinguishable from there being no conflict at
    #: all. An agent told "these two disagree" can ask the user; an agent
    #: handed the winner cannot.
    conflicts: list[tuple[str, str]] = field(default_factory=list)
    #: How much this result set supports asserting an answer, computed from
    #: the score distribution rather than asked of a model. Measured need: on
    #: 500 questions the system failed in BOTH calibration directions at once
    #: -- 24 declines while holding the evidence, 8 answers to unanswerable
    #: questions -- and a model's own certainty is a property of its tone,
    #: not of the data.
    confidence: RetrievalConfidence | None = None


class RetrievalPipeline:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        expander: QueryExpander | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.expander = expander or NoopExpander()

    async def _embed_query(self, query: str, *, expand: bool = False) -> Vector | None:
        """Query-side embedding. A provider failure degrades to lexical-only.

        Losing vector search is a quality regression; failing the request is an
        outage. The response records which strategies actually ran.

        With `expand`, a hypothetical answer is embedded and averaged in — see
        expansion.py for why that helps and why it is averaged rather than
        substituted.
        """
        try:
            embed_query = getattr(self.embedder, "embed_query", None)
            if embed_query is not None:
                base = await embed_query(query)
            else:
                base = await self.embedder.embed_one(query)
        except Exception as exc:
            log.warning("query_embedding_failed", error=str(exc)[:200])
            return None

        if not expand:
            return base  # type: ignore[no-any-return]

        try:
            passages = await self.expander.expand(query)
            if not passages:
                return base  # type: ignore[no-any-return]
            # Hypothetical passages are documents, so they use the document-side
            # embedding, not the query-side one.
            extra = (await self.embedder.embed(passages)).vectors
            return _mean_unit_vector([base, *extra])
        except Exception as exc:
            # Expansion is an enhancement; never let it fail the search.
            log.warning("query_expansion_failed", error=str(exc)[:200])
            return base  # type: ignore[no-any-return]

    async def _expand_by_entity(
        self,
        request: SearchRequest,
        fused: list[FusedItem],
        vector_hits: list[VectorHit],
        lexical_hits: list[LexicalHit],
        fetch: int,
    ) -> tuple[list[str], list[LexicalHit]]:
        """Find evidence that shares an entity with a seed but not the query.

        Seeds are the top fused results. Entities are mined from their text,
        filtered against the query (already searched) and against ubiquity
        (a name in every turn matches everything). Each surviving entity gets
        one indexed lexical lookup, run concurrently.
        """
        text_by_id: dict[str, str] = {h.memory_id: h.text for h in lexical_hits}
        for vector_hit in vector_hits:
            text_by_id.setdefault(vector_hit.memory_id, vector_hit.text)

        seed_texts = [text_by_id[item.id] for item in fused[:8] if item.id in text_by_id]
        entities = salient_entities(
            seed_texts,
            query=request.query,
            max_entities=request.entity_budget,
            speakers=request.known_speakers,
        )
        if not entities:
            return [], []

        async def lookup(entity: str) -> list[LexicalHit]:
            try:
                return await self.store.lexical_search(
                    request.org_id,
                    request.space_id,
                    entity,
                    limit=max(fetch // 2, request.limit),
                    filters=request.filters,
                )
            except Exception as exc:
                # Expansion is an enhancement; one bad lookup must not fail
                # the search.
                log.warning("entity_lookup_failed", entity=entity, error=str(exc)[:160])
                return []

        batches = await asyncio.gather(*(lookup(e) for e in entities))

        # Keep the best score per memory across entity lookups, and drop
        # anything the first stage already found — re-ranking a seed through
        # the entity arm would double-count it in fusion.
        already = {item.id for item in fused}
        best: dict[str, LexicalHit] = {}
        for batch in batches:
            for entity_hit in batch:
                if entity_hit.memory_id in already:
                    continue
                current = best.get(entity_hit.memory_id)
                if current is None or entity_hit.score > current.score:
                    best[entity_hit.memory_id] = entity_hit
        ordered = sorted(best.values(), key=lambda h: (-h.score, h.memory_id))
        return entities, ordered[:fetch]

    async def search(self, request: SearchRequest) -> SearchResponse:
        timings: dict[str, float] = {}
        loop = asyncio.get_running_loop()

        query = request.query.strip()
        if not query:
            return SearchResponse(
                [], request.query, 0, timings, False, [], confidence=assess([])
            )
        if request.limit <= 0:
            return SearchResponse(
                [], request.query, 0, timings, False, [], confidence=assess([])
            )

        fetch = max(request.limit * max(request.candidate_multiplier, 1), request.limit)

        # -- stage 1: candidate generation, concurrent -----------------------
        t0 = loop.time()
        embedding = await self._embed_query(query, expand=request.use_expansion)
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
            return SearchResponse(
                [], request.query, 0, timings, False, strategies, confidence=assess([])
            )

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

        # -- stage 2b: entity bridging ----------------------------------------
        # Runs AFTER first-stage fusion because it needs seeds to mine entities
        # from, and BEFORE hydration so bridged candidates compete on equal
        # terms with the originals rather than being appended as an afterthought.
        entities: list[str] = []
        if request.use_entity_expansion and fused:
            t0 = loop.time()
            entities, entity_hits = await self._expand_by_entity(
                request, fused, vector_hits, lexical_hits, fetch
            )
            if entity_hits:
                ranked.append(
                    RankedList(
                        "entity",
                        [h.memory_id for h in entity_hits],
                        {h.memory_id: h.score for h in entity_hits},
                        weight=request.entity_weight,
                    )
                )
                fused = reciprocal_rank_fusion(ranked, k=request.rrf_k)
                strategies.append("entity")
            timings["entity_ms"] = (loop.time() - t0) * 1000

        # -- stage 3: hydrate --------------------------------------------------
        t0 = loop.time()
        pool = fused[: max(request.rerank_candidates, request.limit)]
        memories = await self.store.get_memories(
            request.org_id, request.space_id, [f.id for f in pool]
        )
        timings["hydrate_ms"] = (loop.time() - t0) * 1000

        bridged = {item.id for item in fused if "entity" in item.ranks}
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
                    explain=(
                        [*item.explain(), "bridged by entity"]
                        if item.id in bridged
                        and "vector" not in item.ranks
                        and "lexical" not in item.ranks
                        else item.explain()
                    ),
                )
            )

        if not scored:
            return SearchResponse(
                [], request.query, len(fused), timings, False, strategies, entities
            )

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
            scored = await self._suppress_superseded(scored, request)

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

        final = scored[: request.limit]
        conflicts = await self._find_conflicts(final, request)
        confidence = assess([s.score for s in final], has_conflicts=bool(conflicts))
        return SearchResponse(
            results=final,
            query=request.query,
            total_candidates=len(fused),
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            rerank_degraded=degraded,
            strategies=strategies,
            entities_used=entities,
            conflicts=conflicts,
            confidence=confidence,
        )

    async def _find_conflicts(
        self, scored: list[ScoredMemory], request: SearchRequest
    ) -> list[tuple[str, str]]:
        """CONTRADICTS edges joining two memories in this result set.

        Surfaced, never resolved. A contradicted memory stays in the results
        because it may be the true one — suppressing either would answer with
        silence, and picking the newer one (the market default) hides the
        disagreement entirely. Reporting it lets the agent ask.

        Only pairs where BOTH sides are visible are reported: a conflict with
        something the caller cannot see is not actionable.
        """
        if len(scored) < 2:
            return []
        ids = [s.memory.id for s in scored]
        edges = await self.store.get_relations_between(
            request.org_id, request.space_id, ids, type=RelationType.CONTRADICTS
        )
        present = set(ids)
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for edge in edges:
            if edge.source_id not in present or edge.target_id not in present:
                continue
            # `contradicts` is symmetric and stored as two edges; report once.
            pair = (
                (edge.source_id, edge.target_id)
                if edge.source_id < edge.target_id
                else (edge.target_id, edge.source_id)
            )
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
        return out

    @staticmethod
    def _representative_embedding(memory: Memory) -> Vector | None:
        for chunk in memory.chunks:
            if chunk.embedding is not None:
                return chunk.embedding
        return None

    async def _suppress_superseded(
        self, scored: list[ScoredMemory], request: SearchRequest
    ) -> list[ScoredMemory]:
        """Drop memories that a *present* result transitively supersedes.

        Only suppress when the replacement is in the same result set. If the
        newer memory did not match the query, hiding the older one would answer
        the question with silence, which is worse than answering with a fact
        that is merely stale.

        "Transitively" is the part that needed a graph. The previous version
        read a relations list off each memory and could only see one hop: given
        A supersedes B supersedes C, with A and C both retrieved but B not, C
        survived as a current fact even though A had long replaced it. Walking
        incoming SUPERSEDES edges answers the real question — "is anything that
        replaced this also in front of the user" — regardless of how many
        intermediate revisions happened, and regardless of whether those
        intermediates matched the query.

        One store round-trip per candidate, each an indexed lookup, only for
        results that survive to this stage.
        """
        present = {s.memory.id for s in scored}
        out: list[ScoredMemory] = []
        for s in scored:
            if s.memory.status is MemoryStatus.SUPERSEDED:
                continue
            superseders = await self.store.reachable_superseders(
                request.org_id, request.space_id, s.memory.id
            )
            if superseders & present:
                continue
            out.append(s)
        return out


__all__ = ["RetrievalPipeline", "SearchRequest", "SearchResponse"]
