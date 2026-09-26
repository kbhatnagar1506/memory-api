"""Memory CRUD, bulk ingest and relation endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status

from ...core.errors import ProviderError, ValidationError
from ...core.ids import is_valid
from ...domain.models import MAX_KEY_CHARS, MemoryKind, MemoryStatus, RelationType, Scope
from ...service import BulkOutcome, IngestItem, IngestResult
from ...store.base import MemoryFilter
from ..deps import Principal, ServiceDep, SettingsDep, require_scope
from ..schemas import (
    BulkCreateMemoryRequest,
    BulkCreateMemoryResponse,
    BulkEraseResponse,
    BulkItemResponse,
    CreateMemoryRequest,
    CreateMemoryResponse,
    EraseAttestation,
    EraseByTagRequest,
    ItemError,
    LineageResponse,
    LinkRequest,
    MemoryContextResponse,
    MemoryListResponse,
    MemoryResponse,
    MemoryVersionListResponse,
    MemoryVersionResponse,
    RelationListResponse,
    RelationResponse,
    RetiredKey,
    RetireRequest,
    RetireResponse,
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
        contradicts=result.contradicts,
        chunk_count=result.chunk_count,
        erased=result.erased,
    )


def _to_bulk_item(outcome: BulkOutcome, *, expose_detail: bool) -> BulkItemResponse:
    if outcome.error is not None or outcome.result is None:
        error = outcome.error or ValidationError("item produced no result")
        return BulkItemResponse(
            index=outcome.index,
            status=error.status_code,
            error=ItemError.from_error(error, expose_detail=expose_detail),
        )
    single = _to_response(outcome.result)
    return BulkItemResponse(
        index=outcome.index,
        status=status.HTTP_201_CREATED if single.created else status.HTTP_200_OK,
        **single.model_dump(),
    )


def _validate_key(key: str) -> str:
    if not key or len(key) > MAX_KEY_CHARS or any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValidationError(f"{key[:40]!r} is not a valid memory key", field="key")
    return key


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
    settings: SettingsDep,
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
        key=body.key,
        replace=body.replace,
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
    settings: SettingsDep,
    response: Response,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> BulkCreateMemoryResponse:
    """Every item's chunks are embedded in one batched pass, then the items
    are written in order. Each item carries its own `status` and, if it
    failed, its own `error`: one bad item no longer fails the request.

    201 when anything was created, 200 when nothing was. If EVERY item
    failed and a provider was among the causes, the request fails as a whole
    with that provider error, so a client's backoff sees the upstream
    trouble instead of a 200 full of item errors.
    """
    _validate_space_id(space_id)
    outcomes = await service.ingest_many(
        org_id=principal.org_id,
        space_id=space_id,
        items=[
            IngestItem(
                content=item.content,
                summary=item.summary,
                metadata=item.metadata,
                tags=item.tags,
                source=item.source,
                occurred_at=item.occurred_at,
                key=item.key,
                replace=item.replace,
            )
            for item in body.items
        ],
    )
    if outcomes and all(o.error is not None for o in outcomes):
        upstream = next((o.error for o in outcomes if isinstance(o.error, ProviderError)), None)
        if upstream is not None:
            raise upstream
    items = [_to_bulk_item(o, expose_detail=settings.debug_errors) for o in outcomes]
    created = sum(1 for i in items if i.created)
    if not created:
        response.status_code = status.HTTP_200_OK
    return BulkCreateMemoryResponse(
        items=items,
        created=created,
        duplicates=sum(1 for i in items if i.error is None and not i.created),
        failed=sum(1 for i in items if i.error is not None),
    )


@router.post(
    "/erase-by-tag",
    response_model=BulkEraseResponse,
    summary="Erase every memory a tag has ever been on",
)
async def erase_by_tag(
    space_id: str,
    body: EraseByTagRequest,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> BulkEraseResponse:
    """Right-to-erasure over a tag, in one transaction: every status and
    kind, live rows and version history alike. Idempotent -- a retry
    reports zeros. The tag is matched as stored (trimmed, case-folded)."""
    _validate_space_id(space_id)
    report = await service.erase_by_tag(principal.org_id, space_id, body.tag)
    return BulkEraseResponse(**report)  # type: ignore[arg-type]


@router.post(
    "/retire",
    response_model=RetireResponse,
    summary="Archive the memories held under some keys",
)
async def retire_keys(
    space_id: str,
    body: RetireRequest,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> RetireResponse:
    """For a sync client whose source dropped these facts: they stop being
    searchable but stay in history. Use DELETE .../by-key/{key} to erase."""
    _validate_space_id(space_id)
    retired, missing = await service.retire_keys(
        principal.org_id, space_id, body.keys, status=MemoryStatus(body.status)
    )
    return RetireResponse(
        retired=[
            RetiredKey(key=m.key or "", memory_id=m.id, version=m.version) for m in retired
        ],
        missing=missing,
    )


@router.delete(
    "/by-key/{key:path}",
    response_model=BulkEraseResponse,
    summary="Erase everything ever held under a key",
)
async def erase_by_key(
    space_id: str,
    key: str,
    service: ServiceDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.MEMORIES_WRITE))],
) -> BulkEraseResponse:
    """Erasure, not the audit-preserving delete: the ACTIVE memory, every
    superseded or archived one before it, and all their version history.
    `{key:path}` so keys containing '/' survive routing; percent-encode the
    rest. Idempotent -- a retry reports zeros."""
    _validate_space_id(space_id)
    report = await service.erase_by_key(principal.org_id, space_id, _validate_key(key))
    return BulkEraseResponse(**report)  # type: ignore[arg-type]


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
    kind: Annotated[list[MemoryKind] | None, Query()] = None,
) -> MemoryListResponse:
    _validate_space_id(space_id)
    limit = min(limit, settings.max_limit)
    filters = MemoryFilter(
        statuses=frozenset(status_filter or [MemoryStatus.ACTIVE]),
        kinds=frozenset(kind or []),
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
    type: RelationType | None = None,
) -> RelationListResponse:
    """`out` = relations this memory asserts. `in` = relations pointing at it,
    which is how you ask "what supersedes this".

    `type` narrows to one kind of edge. Worth having now that consolidation
    runs on every write: a memory accumulates association edges by itself,
    and "what does this replace" should not mean paging all of them.
    """
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    edges = await service.list_relations(
        principal.org_id, space_id, memory_id, direction=direction, type=type
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
    Returns proof of what was destroyed. Idempotent: an id with nothing left
    to erase returns 200 with `already_erased: true`, never 404."""
    _validate_space_id(space_id)
    _validate_memory_id(memory_id)
    attestation = await service.erase_memory(principal.org_id, space_id, memory_id)
    return EraseAttestation(**attestation)  # type: ignore[arg-type]


__all__ = ["router"]
