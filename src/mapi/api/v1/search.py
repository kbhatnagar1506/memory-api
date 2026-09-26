"""Hybrid search, multi-space search, and search-by-stored-memory."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from ...config import Settings
from ...core.errors import ValidationError
from ...core.ids import is_valid
from ...domain.models import MemoryStatus, Scope, utcnow
from ...domain.retrieval.pipeline import SearchRequest, SearchResponse
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, SettingsDep, require_scope
from ..schemas import (
    ConfidenceBlock,
    IntentBlock,
    MultiSearchRequest,
    MultiSearchResponse,
    MultiSearchTarget,
    MultiSearchTargetResult,
    SearchHit,
    SearchRequestBody,
    SearchResponseBody,
    SimilarRequest,
    SimilarResponse,
)

router = APIRouter(prefix="/spaces/{space_id}", tags=["search"])
#: Routes that span spaces, so cannot live under one space's prefix.
multi_router = APIRouter(tags=["search"])


def _require_space_id(space_id: str, field: str) -> None:
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field=field)


def _filters(
    body: SearchRequestBody | MultiSearchTarget | SimilarRequest,
) -> MemoryFilter:
    """One place that turns a body's filter fields into a `MemoryFilter`.

    Search, each multi-search target and similar all filter the same way; three
    copies of this is how a new filter ends up honoured by two of them.
    """
    statuses: set[MemoryStatus] = {MemoryStatus.ACTIVE}
    kinds: frozenset[Any] = frozenset()
    source: str | None = None
    occurred_after = occurred_before = None
    if not isinstance(body, SimilarRequest):
        statuses = set(body.statuses)
        if body.include_superseded:
            statuses.add(MemoryStatus.SUPERSEDED)
        kinds = frozenset(body.kinds)
        source = body.source
        occurred_after = body.occurred_after
        occurred_before = body.occurred_before
    return MemoryFilter(
        statuses=frozenset(statuses),
        kinds=kinds,
        tags=tuple(t.casefold() for t in body.tags),
        metadata=tuple(sorted(body.metadata.items())),
        exclude_metadata=tuple(sorted(body.exclude_metadata.items())),
        occurred_after=occurred_after,
        occurred_before=occurred_before,
        source=source,
    )


def _max_per_source(requested: int | None, settings: Settings) -> int:
    return settings.max_per_source if requested is None else requested


def _response_fields(
    result: SearchResponse,
    *,
    explain: bool,
    include_content: bool,
    snippet_chars: int | None,
    auth_ms: float | None,
) -> dict[str, Any]:
    """The body of a search response, shared by `/search` and each multi-search target."""
    timings = dict(result.timings_ms)
    if auth_ms is not None:
        timings["auth_ms"] = round(auth_ms, 2)
    return {
        "query": result.query,
        "results": [
            SearchHit.from_domain(
                r,
                explain=explain,
                include_content=include_content,
                snippet_chars=snippet_chars,
            )
            for r in result.results
        ],
        "count": len(result.results),
        "total_candidates": result.total_candidates,
        "strategies": result.strategies,
        "rerank_degraded": result.rerank_degraded,
        "timings_ms": timings,
        "conflicts": [list(pair) for pair in result.conflicts],
        "intent": (
            IntentBlock(
                kind=str(result.intent.kind),
                source=result.intent.source,
                comprehensive=result.intent.comprehensive,
            )
            if result.intent
            else None
        ),
        "confidence": (
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
    }


@router.post("/search", response_model=SearchResponseBody, summary="Hybrid search")
async def search(
    space_id: str,
    body: SearchRequestBody,
    http: Request,
    service: ServiceDep,
    settings: SettingsDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SEARCH))],
) -> SearchResponseBody:
    _require_space_id(space_id, "space_id")
    filters = _filters(body)
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
        max_per_source=_max_per_source(body.max_per_source, settings),
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
    return SearchResponseBody(
        **_response_fields(
            result,
            explain=body.explain,
            include_content=body.include_content,
            snippet_chars=body.snippet_chars,
            auth_ms=getattr(http.state, "auth_ms", None),
        )
    )


@multi_router.post(
    "/multi-search",
    response_model=MultiSearchResponse,
    summary="One question, up to four spaces, one embedding",
)
async def multi_search(
    body: MultiSearchRequest,
    http: Request,
    service: ServiceDep,
    settings: SettingsDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SEARCH))],
) -> MultiSearchResponse:
    """Search several spaces with one question.

    The question is embedded ONCE and the vector shared, so asking a person's
    own memory and an event directory costs one provider call instead of one
    per space. Every `space_id` is checked against the caller's organization
    before anything runs; one that is foreign or missing fails the whole
    request with 404, exactly as a single `/search` would.
    """
    for i, target in enumerate(body.targets):
        _require_space_id(target.space_id, f"targets[{i}].space_id")
    asked_at = body.asked_at or utcnow()
    requests = [
        SearchRequest(
            query=body.query,
            org_id=principal.org_id,
            space_id=target.space_id,
            limit=min(target.limit, settings.max_limit),
            filters=_filters(target),
            half_life_days=target.half_life_days,
            use_decay=target.use_decay,
            include_superseded=target.include_superseded,
            candidate_multiplier=settings.candidate_multiplier,
            rerank_candidates=settings.rerank_candidates,
            rrf_k=settings.rrf_k,
            vector_weight=target.vector_weight,
            lexical_weight=target.lexical_weight,
            min_score=target.min_score,
            coverage=target.coverage,
            coverage_limit=settings.coverage_limit,
            max_per_source=_max_per_source(target.max_per_source, settings),
            route_by_kind=settings.route_by_kind,
            asked_at=asked_at,
        )
        for target in body.targets
    ]
    results, degraded, timings = await service.multi_search(body.query, requests)
    auth_ms = getattr(http.state, "auth_ms", None)
    if auth_ms is not None:
        timings["auth_ms"] = round(auth_ms, 2)
    return MultiSearchResponse(
        query=body.query,
        targets=[
            MultiSearchTargetResult(
                space_id=target.space_id,
                **_response_fields(
                    result,
                    explain=target.explain,
                    include_content=target.include_content,
                    snippet_chars=target.snippet_chars,
                    auth_ms=None,
                ),
            )
            for target, result in zip(body.targets, results, strict=True)
        ],
        degraded=degraded,
        timings_ms=timings,
    )


@router.post(
    "/similar",
    response_model=SimilarResponse,
    summary="Nearest neighbours of a stored memory",
)
async def similar(
    space_id: str,
    body: SimilarRequest,
    http: Request,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SEARCH))],
) -> SimilarResponse:
    """Memories in this space nearest to a memory stored in `source_space_id`.

    Searches with the source's stored vector, so no embedding call is made.
    Both spaces must belong to the caller's organization; a source anywhere
    else is a 404, indistinguishable from one that does not exist.
    """
    _require_space_id(space_id, "space_id")
    _require_space_id(body.source_space_id, "source_space_id")
    if body.source_memory_id is not None and not is_valid(body.source_memory_id, "memory"):
        raise ValidationError(
            f"{body.source_memory_id!r} is not a valid memory id", field="source_memory_id"
        )
    memory_id, result = await service.similar(
        principal.org_id,
        space_id,
        source_space_id=body.source_space_id,
        source_memory_id=body.source_memory_id,
        source_key=body.source_key,
        limit=body.limit,
        filters=_filters(body),
        max_per_source=body.max_per_source,
        min_score=body.min_score,
    )
    timings = dict(result.timings_ms)
    auth_ms = getattr(http.state, "auth_ms", None)
    if auth_ms is not None:
        timings["auth_ms"] = round(auth_ms, 2)
    return SimilarResponse(
        source_space_id=body.source_space_id,
        source_memory_id=memory_id,
        results=[
            SearchHit.from_domain(
                r,
                explain=body.explain,
                include_content=body.include_content,
                snippet_chars=body.snippet_chars,
            )
            for r in result.results
        ],
        count=len(result.results),
        timings_ms=timings,
    )


__all__ = ["multi_router", "router"]
