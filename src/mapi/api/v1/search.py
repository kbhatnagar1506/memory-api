"""Hybrid search endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ...core.errors import ValidationError
from ...core.ids import is_valid
from ...domain.models import Scope
from ...domain.retrieval.pipeline import SearchRequest
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, SettingsDep, require_scope
from ..schemas import (
    ConfidenceBlock,
    IntentBlock,
    SearchHit,
    SearchRequestBody,
    SearchResponseBody,
)

router = APIRouter(prefix="/spaces/{space_id}", tags=["search"])


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
    )
    result = await service.search(request)
    return SearchResponseBody(
        query=result.query,
        results=[SearchHit.from_domain(r, explain=body.explain) for r in result.results],
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
