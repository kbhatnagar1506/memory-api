"""API key management. Requires the admin scope."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from ...core.errors import NotFoundError
from ...core.security import build_api_key, default_scopes
from ...domain.models import ApiKey, Scope
from ..deps import SettingsDep, StoreDep, require_scope
from ..schemas import (
    ApiKeyListResponse,
    ApiKeyResponse,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
)

router = APIRouter(prefix="/keys", tags=["keys"])


def _to_response(key: ApiKey) -> ApiKeyResponse:
    return ApiKeyResponse(
        id=key.id, name=key.name, prefix=key.prefix, scopes=sorted(key.scopes),
        created_at=key.created_at, last_used_at=key.last_used_at,
        expires_at=key.expires_at, revoked_at=key.revoked_at,
    )


@router.post(
    "", response_model=CreateApiKeyResponse, status_code=status.HTTP_201_CREATED,
    summary="Mint an API key",
)
async def create_key(
    body: CreateApiKeyRequest,
    store: StoreDep,
    settings: SettingsDep,
    principal: Annotated[object, Depends(require_scope(Scope.ADMIN))],
) -> CreateApiKeyResponse:
    record, plaintext = build_api_key(
        org_id=principal.org_id,  # type: ignore[attr-defined]
        name=body.name,
        pepper=settings.api_key_pepper,
        scopes=frozenset(body.scopes) if body.scopes else default_scopes(),
        expires_at=body.expires_at,
    )
    stored = await store.create_api_key(record)
    # The only time the plaintext ever leaves this process.
    return CreateApiKeyResponse(key=_to_response(stored), plaintext=plaintext)


@router.get("", response_model=ApiKeyListResponse, summary="List API keys")
async def list_keys(
    store: StoreDep,
    principal: Annotated[object, Depends(require_scope(Scope.ADMIN))],
) -> ApiKeyListResponse:
    keys = await store.list_api_keys(principal.org_id)  # type: ignore[attr-defined]
    return ApiKeyListResponse(items=[_to_response(k) for k in keys])


@router.delete(
    "/{key_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke an API key",
)
async def revoke_key(
    key_id: str,
    store: StoreDep,
    principal: Annotated[object, Depends(require_scope(Scope.ADMIN))],
) -> None:
    if not await store.revoke_api_key(principal.org_id, key_id):  # type: ignore[attr-defined]
        raise NotFoundError(f"api key {key_id} not found or already revoked")


__all__ = ["router"]
