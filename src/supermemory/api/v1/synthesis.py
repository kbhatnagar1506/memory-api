"""Derivation and profiles: answers computed from episodes, with provenance."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ...core.errors import ValidationError
from ...core.ids import is_valid
from ...domain.models import Scope
from ..deps import Principal, ServiceDep, require_scope
from ..schemas import (
    ConsolidateResponse,
    DerivedRowResponse,
    DeriveRequest,
    DeriveResponse,
    MemoryResponse,
    ProfileResponse,
)

router = APIRouter(prefix="/spaces/{space_id}", tags=["synthesis"])


def _validate_space_id(space_id: str) -> str:
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")
    return space_id


@router.post("/derive", response_model=DeriveResponse, summary="Derive an answer from episodes")
async def derive(
    space_id: str,
    body: DeriveRequest,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> DeriveResponse:
    """Compute an answer that may exist in no single memory — a count, an
    ordering, a span — by extracting grounded rows per episode and reducing in
    code. With `materialize=true` the answer is stored as a `derived` memory
    with `derived_from` edges to every source, joins the given profile
    `bucket`, and goes STALE automatically if any source is later erased,
    deleted, or superseded. Requires a configured synthesis backend (503
    otherwise)."""
    _validate_space_id(space_id)
    derived, stored = await service.derive(
        principal.org_id,
        space_id,
        question=body.question,
        k=body.k,
        materialize=body.materialize,
        bucket=body.bucket,
    )
    return DeriveResponse(
        answer=derived.answer,
        kind=str(derived.kind),
        computed=derived.computed,
        table=[
            DerivedRowResponse(
                date=row.date.isoformat() if row.date else None,
                fact=row.fact,
                quote=row.quote,
                source_id=row.source_id,
            )
            for row in derived.table
        ],
        source_ids=list(derived.source_ids),
        memory=MemoryResponse.from_domain(stored) if stored else None,
    )


@router.get(
    "/profiles/{bucket}",
    response_model=ProfileResponse,
    summary="Current facts in a profile bucket",
)
async def get_profile(
    space_id: str,
    bucket: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_READ))],
) -> ProfileResponse:
    """ACTIVE derived facts tagged into this bucket. Superseded facts are
    history (see versions/lineage); stale facts are pending re-derivation;
    neither is returned, because neither is current truth."""
    _validate_space_id(space_id)
    facts = await service.get_profile(principal.org_id, space_id, bucket)
    return ProfileResponse(bucket=bucket, facts=[MemoryResponse.from_domain(m) for m in facts])


@router.post(
    "/consolidate",
    response_model=ConsolidateResponse,
    summary="Re-derive facts whose sources changed",
)
async def consolidate(
    space_id: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
    budget: int = 20,
) -> ConsolidateResponse:
    """Finds derived facts marked STALE (their sources were erased, deleted or
    superseded) and recomputes them from the surviving evidence, superseding
    the stale fact with the fresh one.

    The same idea competitors run as a background "dream cycle", minus the
    destructive half: nothing is merged away, nothing expires, episodes are
    never touched. A derivation whose evidence no longer supports an answer
    stays stale rather than being deleted or silently resurrected. `budget`
    caps LLM spend per call."""
    _validate_space_id(space_id)
    report = await service.refresh_stale_derivations(
        principal.org_id, space_id, budget=max(1, min(budget, 200))
    )
    return ConsolidateResponse(**report)  # type: ignore[arg-type]
