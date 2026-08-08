"""Space management."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from ...core.errors import NotFoundError, ValidationError
from ...core.ids import is_valid
from ...domain.models import Scope, Space
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, StoreDep, require_scope
from ..schemas import (
    CreateSpaceRequest,
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
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_WRITE))],
) -> None:
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")
    if not await store.delete_space(principal.org_id, space_id):
        raise NotFoundError(f"space {space_id} not found", field="space_id")


__all__ = ["router"]
