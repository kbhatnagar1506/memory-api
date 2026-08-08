"""Request-scoped dependencies: authentication, scope checks, rate limiting.

Authentication resolves an API key to a principal once per request and caches it
on `request.state`, so a handler needing both the org id and a scope check does
not hash the key twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from ..config import Settings
from ..core.errors import ForbiddenError, RateLimitedError, UnauthorizedError
from ..core.logging import org_id_var
from ..core.metrics import RATE_LIMITED
from ..core.security import hash_key, looks_like_key, parse_authorization
from ..domain.models import ApiKey, Scope, utcnow
from ..service import MemoryService
from ..store.base import MemoryStore


@dataclass(frozen=True, slots=True)
class Principal:
    org_id: str
    key_id: str
    scopes: frozenset[Scope]

    def require(self, scope: Scope) -> None:
        if Scope.ADMIN not in self.scopes and scope not in self.scopes:
            raise ForbiddenError(
                f"this key lacks the {scope.value!r} scope",
                extra={"required_scope": scope.value},
            )


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_store(request: Request) -> MemoryStore:
    return request.app.state.store  # type: ignore[no-any-return]


def get_service(request: Request) -> MemoryService:
    return request.app.state.service  # type: ignore[no-any-return]


async def authenticate(request: Request) -> Principal:
    cached = getattr(request.state, "principal", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    settings: Settings = request.app.state.settings
    store: MemoryStore = request.app.state.store

    raw = parse_authorization(
        request.headers.get("authorization") or request.headers.get("x-api-key")
    )
    if not raw:
        raise UnauthorizedError(
            "provide an API key via 'Authorization: Bearer <key>' or 'X-API-Key'"
        )
    # Reject obviously malformed keys before touching the database.
    if not looks_like_key(raw):
        raise UnauthorizedError("malformed API key")

    record: ApiKey | None = await store.get_api_key_by_hash(
        hash_key(raw, settings.api_key_pepper)
    )
    # Identical message and status for unknown, revoked and expired keys: the
    # difference between them is information an attacker can use.
    if record is None or not record.is_active():
        raise UnauthorizedError("invalid or expired API key")

    principal = Principal(
        org_id=record.org_id, key_id=record.id, scopes=frozenset(record.scopes)
    )
    request.state.principal = principal
    org_id_var.set(principal.org_id)

    try:
        await store.touch_api_key(record.id, utcnow())
    except Exception:  # noqa: BLE001 - last-used tracking must never 500 a request
        pass
    return principal


async def enforce_rate_limit(
    request: Request, principal: Annotated[Principal, Depends(authenticate)]
) -> Principal:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return principal
    decision = await limiter.check(principal.org_id)
    request.state.rate_limit = decision
    if not decision.allowed:
        RATE_LIMITED.inc()
        raise RateLimitedError(
            f"rate limit of {limiter.per_minute}/min exceeded for this organization",
            retry_after=decision.retry_after,
        )
    return principal


def require_scope(scope: Scope):
    """Dependency factory: authenticate, rate limit, then check one scope."""

    async def _dep(
        principal: Annotated[Principal, Depends(enforce_rate_limit)],
    ) -> Principal:
        principal.require(scope)
        return principal

    return _dep


CurrentPrincipal = Annotated[Principal, Depends(enforce_rate_limit)]
ServiceDep = Annotated[MemoryService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
StoreDep = Annotated[MemoryStore, Depends(get_store)]

__all__ = [
    "CurrentPrincipal", "Principal", "ServiceDep", "SettingsDep", "StoreDep",
    "authenticate", "enforce_rate_limit", "get_service", "get_settings_dep",
    "get_store", "require_scope",
]
