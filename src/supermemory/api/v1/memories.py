"""Memory CRUD, bulk ingest and relation endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status

from ...core.errors import ValidationError
from ...core.ids import is_valid
from ...domain.models import MemoryStatus, Scope
from ...service import IngestResult
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, SettingsDep, require_scope
from ..schemas import (
    BulkCreateMemoryRequest,
    BulkCreateMemoryResponse,
    CreateMemoryRequest,
    CreateMemoryResponse,
    EraseAttestation,
    LineageResponse,
    LinkRequest,
    MemoryContextResponse,
    MemoryListResponse,
    MemoryResponse,
    MemoryVersionListResponse,
    MemoryVersionResponse,
    RelationListResponse,
    RelationResponse,
)

router = APIRouter(prefix="/spaces/{space_id}/memories", tags=["memories"])


def _validate_space_id(space_id: str) -> str:
    if not is_valid(space_id, "space"):
        raise ValidationError(f"{space_id!r} is not a valid space id", field="space_id")
    return space_id


def _validate_memory_id(memory_id: str) -> str:
    if not is_valid(memory_id, "memory"):
        raise ValidationError(f"{memory_id!r} is not a valid memory id", field="memory_id")
    return memory_id


def _to_response(result: IngestResult) -> CreateMemoryResponse:
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
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> CreateMemoryResponse:
    _validate_space_id(space_id)
    result = await service.ingest(
        org_id=principal.org_id,
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
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> BulkCreateMemoryResponse:
    _validate_space_id(space_id)
    items: list[CreateMemoryResponse] = []
    # Sequential rather than gathered: concurrent ingest of a batch containing
    # duplicates of each other would race the dedup check and store both.
    for item in body.items:
        result = await service.ingest(
            org_id=principal.org_id,
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
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_READ))],
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
        principal.org_id,
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
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_READ))],
    as_of: Annotated[
        datetime | None,
        Query(description="Return the memory as it stood at this instant (system time)."),
    ] = None,
) -> MemoryResponse:
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    if as_of is not None:
        # Point-in-time read: the memory as our database understood it then.
        # Chunks are not versioned, so chunk_count is 0 on a historical read.
        historical = await service.get_memory_as_of(
            principal.org_id, space_id, memory_id, as_of
        )
        response.headers["etag"] = f'W/"{historical.version}"'
        return MemoryResponse.from_domain(historical)

    memory = await service.get_memory(
        principal.org_id,
        space_id,
        memory_id,
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
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> None:
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    await service.delete_memory(
        principal.org_id,
        space_id,
        memory_id,
    )


@router.post(
    "/{memory_id}/relations",
    response_model=RelationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Relate one memory to another",
)
async def link_memory(
    space_id: str,
    memory_id: str,
    body: LinkRequest,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> RelationResponse:
    _validate_space_id(space_id)
    for value, kind, field in (
        (memory_id, "memory", "memory_id"),
        (body.target_id, "memory", "target_id"),
    ):
        if not is_valid(value, kind):
            raise ValidationError(f"{value!r} is not a valid memory id", field=field)
    edge = await service.link(
        principal.org_id,
        space_id,
        source_id=memory_id,
        target_id=body.target_id,
        relation=body.relation,
        reason=body.reason,
    )
    return RelationResponse.from_domain(edge)


@router.get(
    "/{memory_id}/relations",
    response_model=RelationListResponse,
    summary="List a memory's relations",
)
async def get_relations(
    space_id: str,
    memory_id: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_READ))],
    direction: Annotated[str, Query(pattern="^(out|in)$")] = "out",
) -> RelationListResponse:
    """`out` = relations this memory asserts. `in` = relations pointing at it,
    which is how you ask "what supersedes this"."""
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    edges = await service.list_relations(
        principal.org_id, space_id, memory_id, direction=direction
    )
    return RelationListResponse(items=[RelationResponse.from_domain(e) for e in edges])


@router.get(
    "/{memory_id}/lineage",
    response_model=LineageResponse,
    summary="Where a memory sits in its supersession chain",
)
async def get_lineage(
    space_id: str,
    memory_id: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_READ))],
) -> LineageResponse:
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    lineage = await service.get_lineage(principal.org_id, space_id, memory_id)
    return LineageResponse(**lineage)  # type: ignore[arg-type]


@router.get(
    "/{memory_id}/context",
    response_model=MemoryContextResponse,
    summary="A memory with its entire relation neighborhood",
)
async def get_memory_context(
    space_id: str,
    memory_id: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_READ))],
) -> MemoryContextResponse:
    """One call instead of four: the memory, whether it is still current (and
    what replaced it), what it replaced, its provenance (`derived_from`), what
    was derived from it, and what it references or contradicts — each resolved
    to full content, not bare ids. This is the read an agent wants before
    trusting a fact."""
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    context = await service.get_memory_context(principal.org_id, space_id, memory_id)
    as_response = MemoryResponse.from_domain
    return MemoryContextResponse(
        memory=as_response(context.memory),
        is_current=context.is_current,
        current_head=[as_response(m) for m in context.current_head],
        replaced=[as_response(m) for m in context.replaced],
        derived_from=[as_response(m) for m in context.derived_from],
        derivatives=[as_response(m) for m in context.derivatives],
        references=[as_response(m) for m in context.references],
        contradicts=[as_response(m) for m in context.contradicts],
    )


@router.get(
    "/{memory_id}/versions",
    response_model=MemoryVersionListResponse,
    summary="Full version history of a memory",
)
async def get_versions(
    space_id: str,
    memory_id: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_READ))],
) -> MemoryVersionListResponse:
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    versions = await service.list_memory_versions(principal.org_id, space_id, memory_id)
    return MemoryVersionListResponse(
        memory_id=memory_id,
        items=[MemoryVersionResponse.from_domain(v) for v in versions],
    )


@router.post(
    "/{memory_id}/erase",
    response_model=EraseAttestation,
    summary="Right-to-erasure purge with attestation",
)
async def erase_memory(
    space_id: str,
    memory_id: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> EraseAttestation:
    """Destroys the memory everywhere it can be reached: live row, chunks,
    edges, and the full version history — point-in-time reads included. This
    is the compliance path; plain DELETE preserves the audit trail instead.
    Returns proof of what was destroyed."""
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    attestation = await service.erase_memory(principal.org_id, space_id, memory_id)
    return EraseAttestation(**attestation)  # type: ignore[arg-type]


__all__ = ["router"]
