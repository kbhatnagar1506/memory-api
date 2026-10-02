"""Hybrid search endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ...core.errors import NotFoundError, ValidationError
from ...core.ids import is_valid
from ...domain.chunking import chunk_closure
from ...domain.models import Scope, ScoredMemory, utcnow
from ...domain.retrieval.pipeline import SearchRequest
from ...service import MemoryService
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, SettingsDep, require_scope
from ..schemas import (
    ChunkDependency,
    ConfidenceBlock,
    DependencyRef,
    IntentBlock,
    SearchHit,
    SearchRequestBody,
    SearchResponseBody,
)

router = APIRouter(prefix="/spaces/{space_id}", tags=["search"])

#: Memories brought along per hit, nearest first: enough for a function and what it uses.
DEPENDENCIES_PER_HIT = 12


async def _with_dependencies(
    hit: SearchHit,
    scored: ScoredMemory,
    service: MemoryService,
    org_id: str,
    space_id: str,
    depth: int,
) -> SearchHit:
    """The hit plus what it depends on: the pieces of its memory the matched piece uses,
    and the memories it rests on, each dependencies first."""
    memory = scored.memory
    chunks = memory.chunks
    if scored.matched_chunk_id and not chunks:
        chunks = (await service.get_memory(org_id, space_id, memory.id)).chunks
    by_ordinal = {c.ordinal: c for c in chunks}
    matched = next((c for c in chunks if c.id == scored.matched_chunk_id), None)
    pieces: list[ChunkDependency] = []
    if matched is not None:
        graph = {c.ordinal: tuple(c.depends_on) for c in chunks}
        pieces = [
            ChunkDependency(ordinal=o, text=by_ordinal[o].text)
            for o in chunk_closure(graph, matched.ordinal)
            if o != matched.ordinal and o in by_ordinal
        ]
    try:
        found = await service.get_dependencies(
            org_id, space_id, memory.id, depth=depth, limit=DEPENDENCIES_PER_HIT
        )
    except NotFoundError:  # erased between the search and the walk
        return hit.model_copy(update={"chunk_dependencies": pieces, "dependencies": []})
    refs = [
        DependencyRef(
            id=m,
            summary=found.memories[m].summary,
            content=found.memories[m].content,
            depth=found.depth[m],
        )
        for m in found.order
        if m != memory.id
    ]
    return hit.model_copy(update={"chunk_dependencies": pieces, "dependencies": refs})


@router.post("/search", response_model=SearchResponseBody, summary="Hybrid search")
async def search(
    space_id: str,
    body: SearchRequestBody,
    service: ServiceDep,
    settings: SettingsDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SEARCH))],
) -> SearchResponseBody:
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")

    statuses = set(body.statuses)
    if body.include_superseded:
        from ...domain.models import MemoryStatus

        statuses.add(MemoryStatus.SUPERSEDED)

    filters = MemoryFilter(
        statuses=frozenset(statuses),
        kinds=frozenset(body.kinds),
        tags=tuple(t.casefold() for t in body.tags),
        metadata=tuple(sorted(body.metadata.items())),
        occurred_after=body.occurred_after,
        occurred_before=body.occurred_before,
        source=body.source,
    )
    request = SearchRequest(
        query=body.query,
        org_id=principal.org_id,
        space_id=space_id,
        limit=min(body.limit, settings.max_limit),
        filters=filters,
        mmr_lambda=body.mmr_lambda,
        half_life_days=body.half_life_days,
        use_decay=body.use_decay,
        use_rerank=body.use_rerank,
        use_mmr=body.use_mmr,
        include_superseded=body.include_superseded,
        candidate_multiplier=settings.candidate_multiplier,
        rerank_candidates=settings.rerank_candidates,
        rrf_k=settings.rrf_k,
        vector_weight=body.vector_weight,
        lexical_weight=body.lexical_weight,
        min_score=body.min_score,
        coverage=body.coverage,
        coverage_limit=settings.coverage_limit,
        max_per_source=settings.max_per_source,
        route_by_kind=settings.route_by_kind,
        # `asked_at` defaults to NOW rather than to None, which is the one place
        # this wiring changes behaviour on purpose.
        #
        # Stage 4b is gated on `asked_at is not None`, so leaving it unset kept
        # the temporal stage dead for every product request while the benchmark
        # harness -- the only caller that set it -- measured and documented it.
        # "The question was asked now" is true of every synchronous search, and a
        # caller replaying history can say otherwise.
        #
        # Safe because the stage is a BIAS: it multiplies in-window scores and
        # removes nothing, and it does nothing at all unless the query names a
        # window `extract_scope` can parse.
        asked_at=body.asked_at or utcnow(),
        use_temporal_scope=body.use_temporal_scope,
        tune_by_intent=body.tune_by_intent,
        use_expansion=body.use_expansion,
        use_entity_expansion=body.use_entity_expansion,
        entity_budget=body.entity_budget,
        entity_weight=body.entity_weight,
        known_speakers=tuple(body.known_speakers),
    )
    result = await service.search(request)
    hits = [SearchHit.from_domain(r, explain=body.explain) for r in result.results]
    if body.with_dependencies:
        hits = [
            await _with_dependencies(
                hit, scored, service, principal.org_id, space_id, body.dependency_depth
            )
            for hit, scored in zip(hits, result.results, strict=True)
        ]
    return SearchResponseBody(
        query=result.query,
        results=hits,
        count=len(result.results),
        total_candidates=result.total_candidates,
        strategies=result.strategies,
        rerank_degraded=result.rerank_degraded,
        timings_ms=result.timings_ms,
        conflicts=[list(pair) for pair in result.conflicts],
        intent=(
            IntentBlock(
                kind=str(result.intent.kind),
                source=result.intent.source,
                comprehensive=result.intent.comprehensive,
            )
            if result.intent
            else None
        ),
        confidence=(
            ConfidenceBlock(
                level=str(result.confidence.level),
                top_score=round(result.confidence.top_score, 4),
                margin=round(result.confidence.margin, 4),
                n_results=result.confidence.n_results,
                has_conflicts=result.confidence.has_conflicts,
                reason=result.confidence.reason,
                refusal_reason=(
                    str(result.confidence.refusal_reason)
                    if result.confidence.refusal_reason
                    else None
                ),
            )
            if result.confidence
            else None
        ),
    )


__all__ = ["router"]
