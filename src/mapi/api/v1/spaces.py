"""Space management."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, status
from fastapi import Response as HttpResponse

from ...core.errors import ConflictError, NotFoundError, ValidationError
from ...core.ids import is_valid
from ...domain.models import Scope, Space, utcnow
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, StoreDep, require_scope
from ..schemas import (
    CreateSpaceRequest,
    EnsureSpaceRequest,
    PurgeResponse,
    SpaceListResponse,
    SpaceResponse,
)

#: The slug rule from `Space`, restated for the path parameter so a malformed
#: slug is a 422 at the edge rather than a validation error from the model.
SlugPath = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$")]

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
    # Two queries whatever the org's size. This was a count per space, which
    # is fine for a team with five spaces and a thousand round trips for an
    # event with one space per attendee.
    spaces = await store.list_spaces(principal.org_id)
    counts = await store.count_memories_by_space(principal.org_id)
    return SpaceListResponse(items=[_to_response(s, counts.get(s.id, 0)) for s in spaces])


@router.put(
    "/by-slug/{slug}",
    response_model=SpaceResponse,
    responses={
        200: {"description": "The space already existed"},
        201: {"description": "Created"},
    },
    summary="Create a space by slug, or return the one that exists",
)
async def ensure_space(
    slug: SlugPath,
    response: HttpResponse,
    service: ServiceDep,
    store: StoreDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_WRITE))],
    body: EnsureSpaceRequest | None = None,
) -> SpaceResponse:
    """Idempotent: the same PUT always ends with the same space, 201 the first time.

    A client that keys its spaces by a stable name of its own (one space per
    attendee, a `directory` space per event) needs "the space called X" and
    had to list every space to find it, then race another worker to create
    it. Here a concurrent create that loses the race on the unique slug reads
    the winner back instead of failing, so N workers calling this at once get
    one space and N-1 of them a 200.
    """
    existing = await store.get_space_by_slug(principal.org_id, slug)
    if existing is None:
        wanted = body or EnsureSpaceRequest()
        try:
            created = await service.create_space(
                principal.org_id,
                slug=slug,
                name=wanted.name or slug,
                description=wanted.description,
                metadata=wanted.metadata,
            )
        except ConflictError:
            existing = await store.get_space_by_slug(principal.org_id, slug)
            if existing is None:
                raise
        else:
            response.status_code = status.HTTP_201_CREATED
            return _to_response(created, count=0)
    count = await store.count_memories(existing.org_id, existing.id, filters=MemoryFilter())
    return _to_response(existing, count)


@router.get("/by-slug/{slug}", response_model=SpaceResponse, summary="Fetch a space by slug")
async def get_space_by_slug(
    slug: SlugPath,
    store: StoreDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.SPACES_READ))],
) -> SpaceResponse:
    space = await store.get_space_by_slug(principal.org_id, slug)
    if space is None:
        # Same 404 for "absent" and "another org's": the slug is not a probe.
        raise NotFoundError(f"space {slug!r} not found", field="slug")
    count = await store.count_memories(space.org_id, space.id, filters=MemoryFilter())
    return _to_response(space, count)


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
