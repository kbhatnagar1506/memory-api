"""Typed failures, mapped from the API's problem+json responses.

Every error carries the server's `request_id`. That is the single most useful
thing a client can hand someone debugging a production issue, and it is
invisible unless the SDK surfaces it -- so it goes in the message, not just
an attribute nobody prints.
"""

from __future__ import annotations

from typing import Any


class MapiError(Exception):
    """Base for every failure this client raises."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str = "",
        request_id: str = "",
        field: str = "",
    ) -> None:
        self.status = status
        self.code = code
        self.request_id = request_id
        self.field = field
        detail = message
        if field:
            detail = f"{detail} (field: {field})"
        if request_id:
            detail = f"{detail} [request_id: {request_id}]"
        super().__init__(detail)


class AuthenticationError(MapiError):
    """The API key is missing, malformed, or revoked."""


class PermissionError_(MapiError):
    """The key is valid but lacks the scope for this call."""


class NotFoundError(MapiError):
    """No such space or memory.

    Also what you get for a resource belonging to another organization: the
    API answers 404 rather than 403 there deliberately, since a 403 would
    confirm the resource exists.
    """


class ConflictError(MapiError):
    """The write collides with something already stored."""


class ValidationError(MapiError):
    """The request was rejected before anything was stored."""


class RateLimitError(MapiError):
    """Too many requests. `retry_after` is seconds, when the server said."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class ServerError(MapiError):
    """The service failed. Retried automatically before you see this."""


class ConnectionError_(MapiError):
    """The service could not be reached at all."""


#: Mapped on the server's own `code`, not on the status, because the status
#: alone cannot distinguish "wrong key" from "key lacks this scope".
_BY_CODE: dict[str, type[MapiError]] = {
    "unauthorized": AuthenticationError,
    "forbidden": PermissionError_,
    "not_found": NotFoundError,
    "conflict": ConflictError,
    "idempotency_conflict": ConflictError,
    "validation_error": ValidationError,
    "bad_request": ValidationError,
    "payload_too_large": ValidationError,
    "unsupported_media_type": ValidationError,
    "rate_limited": RateLimitError,
}

_BY_STATUS: dict[int, type[MapiError]] = {
    401: AuthenticationError,
    403: PermissionError_,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    429: RateLimitError,
}


def from_response(status: int, payload: dict[str, Any], retry_after: float | None) -> MapiError:
    """Build the right exception from a problem+json body.

    Falls back to the status when the body is not problem+json at all --
    a proxy returning an HTML 502 is a real thing that happens, and it must
    not surface as a JSON decode error that hides the actual status.
    """
    code = str(payload.get("code") or "")
    kind = _BY_CODE.get(code) or _BY_STATUS.get(status) or ServerError
    message = str(payload.get("detail") or payload.get("title") or f"HTTP {status}")
    if kind is RateLimitError:
        return RateLimitError(
            message,
            retry_after=retry_after,
            status=status,
            code=code,
            request_id=str(payload.get("request_id") or ""),
        )
    return kind(
        message,
        status=status,
        code=code,
        request_id=str(payload.get("request_id") or ""),
        field=str(payload.get("field") or ""),
    )


__all__ = [
    "AuthenticationError",
    "ConflictError",
    "ConnectionError_",
    "MapiError",
    "NotFoundError",
    "PermissionError_",
    "RateLimitError",
    "ServerError",
    "ValidationError",
    "from_response",
]
