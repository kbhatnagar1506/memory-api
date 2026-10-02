"""API key management. Requires the admin scope."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from ...core.errors import NotFoundError, ValidationError
from ...core.ids import is_valid
from ...core.security import build_api_key, default_scopes
from ...domain.models import ApiKey, Scope
from ..deps import Principal, SettingsDep, StoreDep, require_scope
from ..schemas import (
    ApiKeyListResponse,
    ApiKeyResponse,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
)

router = APIRouter(prefix="/keys", tags=["keys"])


def _to_response(key: ApiKey) -> ApiKeyResponse:
    return ApiKeyResponse(
        id=key.id,
        name=key.name,
        prefix=key.prefix,
        scopes=sorted(key.scopes),
        space_ids=sorted(key.space_ids) if key.space_ids is not None else None,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        expires_at=key.expires_at,
        revoked_at=key.revoked_at,
    )


@router.post(
    "",
    response_model=CreateApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Mint an API key",
)
async def create_key(
    body: CreateApiKeyRequest,
    store: StoreDep,
    settings: SettingsDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.ADMIN))],
) -> CreateApiKeyResponse:
    scopes = frozenset(body.scopes) if body.scopes else default_scopes()
    spaces: frozenset[str] | None = None
    if body.space_ids is not None:
        if Scope.ADMIN in scopes:
            # An admin key mints keys, and a minted key could reach any space.
            raise ValidationError("an admin key can't be limited to spaces", field="space_ids")
        for space_id in body.space_ids:
            if not is_valid(space_id, "space") or not await store.get_space(
                principal.org_id, space_id
            ):
                raise ValidationError(f"space {space_id} not found", field="space_ids")
        spaces = frozenset(body.space_ids)
    record, plaintext = build_api_key(
        org_id=principal.org_id,
        name=body.name,
        pepper=settings.api_key_pepper,
        scopes=scopes,
        expires_at=body.expires_at,
        space_ids=spaces,
    )
    stored = await store.create_api_key(record)
    # The only time the plaintext ever leaves this process.
    return CreateApiKeyResponse(key=_to_response(stored), plaintext=plaintext)


@router.get("", response_model=ApiKeyListResponse, summary="List API keys")
async def list_keys(
    store: StoreDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.ADMIN))],
) -> ApiKeyListResponse:
    keys = await store.list_api_keys(principal.org_id)
    return ApiKeyListResponse(items=[_to_response(k) for k in keys])


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key",
)
async def revoke_key(
    key_id: str,
    store: StoreDep,
    principal: Annotated[Principal, Depends(require_scope(Scope.ADMIN))],
) -> None:
    if not await store.revoke_api_key(principal.org_id, key_id):
        raise NotFoundError(f"api key {key_id} not found or already revoked")


__all__ = ["router"]
