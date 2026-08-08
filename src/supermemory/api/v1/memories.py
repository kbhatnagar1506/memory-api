"""Memory CRUD, bulk ingest and relation endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status

from ...core.errors import ValidationError
from ...core.ids import is_valid
from ...domain.models import MemoryStatus, Scope
from ...store.base import MemoryFilter
from ..deps import CurrentPrincipal, ServiceDep, SettingsDep, require_scope
from ..schemas import (
    BulkCreateMemoryRequest,
    BulkCreateMemoryResponse,
    CreateMemoryRequest,
    CreateMemoryResponse,
    LinkRequest,
    MemoryListResponse,
    MemoryResponse,
)

router = APIRouter(prefix="/spaces/{space_id}/memories", tags=["memories"])


def _validate_space_id(space_id: str) -> str:
    if not is_valid(space_id, "space"):
        raise ValidationError(
            f"{space_id!r} is not a valid space id", field="space_id"
        )
    return space_id


def _to_response(result) -> CreateMemoryResponse:
    return CreateMemoryResponse(
        memory=MemoryResponse.from_domain(result.memory),
        created=result.created,
        duplicate_of=result.duplicate_of,
        duplicate_kind=str(result.duplicate_kind),
        similarity=round(result.similarity, 6),
        superseded=result.superseded,
        chunk_count=result.chunk_count,
    )


@router.post(
    "",
    response_model=CreateMemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a memory",
)
async def create_memory(
    space_id: str,
    body: CreateMemoryRequest,
    service: ServiceDep,
    response: Response,
    principal: Annotated[object, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> CreateMemoryResponse:
    _validate_space_id(space_id)
    result = await service.ingest(
        org_id=principal.org_id,  # type: ignore[attr-defined]
        space_id=space_id,
        content=body.content,
        summary=body.summary,
        metadata=body.metadata,
        tags=body.tags,
        source=body.source,
        occurred_at=body.occurred_at,
        dedupe=body.dedupe,
        auto_supersede=body.auto_supersede,
    )
    # A deduplicated write did not create anything; 200 says so honestly.
    if not result.created:
        response.status_code = status.HTTP_200_OK
    response.headers["etag"] = f'W/"{result.memory.version}"'
    return _to_response(result)


@router.post(
    "/bulk",
    response_model=BulkCreateMemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create up to 100 memories",
)
async def bulk_create(
    space_id: str,
    body: BulkCreateMemoryRequest,
    service: ServiceDep,
    principal: Annotated[object, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> BulkCreateMemoryResponse:
    _validate_space_id(space_id)
    items: list[CreateMemoryResponse] = []
    # Sequential rather than gathered: concurrent ingest of a batch containing
    # duplicates of each other would race the dedup check and store both.
    for item in body.items:
        result = await service.ingest(
            org_id=principal.org_id,  # type: ignore[attr-defined]
            space_id=space_id,
            content=item.content,
            summary=item.summary,
            metadata=item.metadata,
            tags=item.tags,
            source=item.source,
            occurred_at=item.occurred_at,
            dedupe=item.dedupe,
            auto_supersede=item.auto_supersede,
        )
        items.append(_to_response(result))
    return BulkCreateMemoryResponse(
        items=items,
        created=sum(1 for i in items if i.created),
        duplicates=sum(1 for i in items if not i.created),
    )


@router.get("", response_model=MemoryListResponse, summary="List memories")
async def list_memories(
    space_id: str,
    service: ServiceDep,
    settings: SettingsDep,
    principal: Annotated[object, Depends(require_scope(Scope.MEMORIES_READ))],
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    cursor: str | None = None,
    tag: Annotated[list[str] | None, Query()] = None,
    source: str | None = None,
    status_filter: Annotated[list[MemoryStatus] | None, Query(alias="status")] = None,
) -> MemoryListResponse:
    _validate_space_id(space_id)
    limit = min(limit, settings.max_limit)
    filters = MemoryFilter(
        statuses=frozenset(status_filter or [MemoryStatus.ACTIVE]),
        tags=tuple(t.casefold() for t in (tag or [])),
        source=source,
    )
    page = await service.list_memories(
        principal.org_id,  # type: ignore[attr-defined]
        space_id,
        filters=filters,
        limit=limit,
        cursor=cursor,
    )
    return MemoryListResponse(
        items=[MemoryResponse.from_domain(m) for m in page.items],
        next_cursor=page.next_cursor,
        total=page.total,
    )


@router.get("/{memory_id}", response_model=MemoryResponse, summary="Fetch a memory")
async def get_memory(
    space_id: str,
    memory_id: str,
    service: ServiceDep,
    response: Response,
    principal: Annotated[object, Depends(require_scope(Scope.MEMORIES_READ))],
) -> MemoryResponse:
    _validate_space_id(space_id)
    if not is_valid(memory_id, "memory"):
        raise ValidationError(
            f"{memory_id!r} is not a valid memory id", field="memory_id"
        )
    memory = await service.get_memory(
        principal.org_id, space_id, memory_id  # type: ignore[attr-defined]
    )
    response.headers["etag"] = f'W/"{memory.version}"'
    return MemoryResponse.from_domain(memory)


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a memory",
)
async def delete_memory(
    space_id: str,
    memory_id: str,
    service: ServiceDep,
    principal: Annotated[object, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> None:
    _validate_space_id(space_id)
    if not is_valid(memory_id, "memory"):
        raise ValidationError(
            f"{memory_id!r} is not a valid memory id", field="memory_id"
        )
    await service.delete_memory(
        principal.org_id, space_id, memory_id  # type: ignore[attr-defined]
    )


@router.post(
    "/{memory_id}/relations",
    response_model=MemoryResponse,
    summary="Relate one memory to another",
)
async def link_memory(
    space_id: str,
    memory_id: str,
    body: LinkRequest,
    service: ServiceDep,
    principal: Annotated[object, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> MemoryResponse:
    _validate_space_id(space_id)
    for value, kind, field in (
        (memory_id, "memory", "memory_id"),
        (body.target_id, "memory", "target_id"),
    ):
        if not is_valid(value, kind):
            raise ValidationError(f"{value!r} is not a valid memory id", field=field)
    updated = await service.link(
        principal.org_id,  # type: ignore[attr-defined]
        space_id,
        source_id=memory_id,
        target_id=body.target_id,
        relation=body.relation,
        reason=body.reason,
    )
    return MemoryResponse.from_domain(updated)


__all__ = ["router"]
