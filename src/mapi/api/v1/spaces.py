"""Space management."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from ...core.errors import NotFoundError, ValidationError
from ...core.ids import is_valid
from ...domain.models import Scope, Space, utcnow
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, StoreDep, require_scope
from ..schemas import (
    CreateSpaceRequest,
    PurgeResponse,
    SpaceListResponse,
    SpaceResponse,
)

router = APIRouter(prefix="/spaces", tags=["spaces"])


def _to_response(space: Space, count: int | None = None) -> SpaceResponse:
    return SpaceResponse(
        id=space.id,
        slug=space.slug,
        name=space.name,
        description=space.description,
        metadata=space.metadata,
        created_at=space.created_at,
        memory_count=count,
    )


@router.post(
    "",
    response_model=SpaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a space",
)
async def create_space(
    body: CreateSpaceRequest,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_WRITE))],
) -> SpaceResponse:
    space = await service.create_space(
        principal.org_id,
        slug=body.slug,
        name=body.name,
        description=body.description,
        metadata=body.metadata,
    )
    return _to_response(space, count=0)


@router.get("", response_model=SpaceListResponse, summary="List spaces")
async def list_spaces(
    store: StoreDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_READ))],
) -> SpaceListResponse:
    spaces = await store.list_spaces(principal.org_id)
    items = []
    for space in spaces:
        count = await store.count_memories(space.org_id, space.id, filters=MemoryFilter())
        items.append(_to_response(space, count))
    return SpaceListResponse(items=items)


@router.get("/{space_id}", response_model=SpaceResponse, summary="Fetch a space")
async def get_space(
    space_id: str,
    store: StoreDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_READ))],
) -> SpaceResponse:
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")
    space = await store.get_space(principal.org_id, space_id)
    if space is None:
        raise NotFoundError(f"space {space_id} not found", field="space_id")
    count = await store.count_memories(space.org_id, space.id, filters=MemoryFilter())
    return _to_response(space, count)


@router.delete(
    "/{space_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a space and everything in it",
)
async def delete_space(
    space_id: str,
    store: StoreDep,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_WRITE))],
) -> None:
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")
    deleted = await store.delete_space(principal.org_id, space_id)
    # Evicted either way: the space lookup cache must never outlive the row.
    service.forget_space(principal.org_id, space_id)
    if not deleted:
        raise NotFoundError(f"space {space_id} not found", field="space_id")


@router.post(
    "/{space_id}/purge",
    response_model=PurgeResponse,
    summary="Destroy a space and all of its history",
)
async def purge_space(
    space_id: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_WRITE))],
) -> PurgeResponse:
    """The right-to-erasure form of DELETE. DELETE removes the live rows and
    keeps version history as an audit trail, so a deleted space's content
    stayed readable through `/versions`; purge removes versions, edges,
    chunks, memories and the space in one transaction and reports each
    count. Idempotent: purging again -- or purging a space DELETE already
    removed -- returns 200, with zeros once nothing is left."""
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")
    report = await service.purge_space(principal.org_id, space_id)
    return PurgeResponse(
        space_id=space_id,
        spaces=report.spaces,
        memories=report.memories,
        chunks=report.chunks,
        relation_edges=report.relation_edges,
        memory_versions=report.memory_versions,
        purged_at=utcnow().isoformat(),
    )


__all__ = ["router"]
