"""Error taxonomy and RFC 9457 (problem+json) responses.

Every failure the API can produce is one of these types. Two rules:

  * The client always learns *what* went wrong and *which field*, never an
    internal stack trace. `debug_errors` can add detail locally; production
    validation refuses to boot with it enabled.
  * Errors carry a stable machine-readable `type` slug. Clients branch on that,
    never on the human-readable message, so wording can change freely.
"""

from __future__ import annotations

from typing import Any

CONTENT_TYPE = "application/problem+json"
_DOC_BASE = "https://docs.supermemory.dev/errors"


class SupermemoryError(Exception):
    """Base for every expected failure. Unexpected ones become 500s."""

    status_code: int = 500
    slug: str = "internal_error"
    title: str = "Internal server error"

    def __init__(
        self,
        detail: str | None = None,
        *,
        field: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail or self.title
        self.field = field
        self.extra = extra or {}
        super().__init__(self.detail)

    def to_problem(self, *, instance: str | None = None) -> dict[str, Any]:
        problem: dict[str, Any] = {
            "type": f"{_DOC_BASE}/{self.slug}",
            "code": self.slug,
            "title": self.title,
            "status": self.status_code,
            "detail": self.detail,
        }
        if self.field:
            problem["field"] = self.field
        if instance:
            problem["instance"] = instance
        problem.update(self.extra)
        return problem


# -- 4xx ---------------------------------------------------------------------


class ValidationError(SupermemoryError):
    status_code = 422
    slug = "validation_error"
    title = "Request failed validation"


class BadRequestError(SupermemoryError):
    status_code = 400
    slug = "bad_request"
    title = "Malformed request"


class UnauthorizedError(SupermemoryError):
    status_code = 401
    slug = "unauthorized"
    title = "Missing or invalid credentials"


class ForbiddenError(SupermemoryError):
    status_code = 403
    slug = "forbidden"
    title = "Insufficient scope for this operation"


class NotFoundError(SupermemoryError):
    status_code = 404
    slug = "not_found"
    title = "Resource not found"


class ConflictError(SupermemoryError):
    status_code = 409
    slug = "conflict"
    title = "Resource conflict"


class IdempotencyConflictError(ConflictError):
    slug = "idempotency_conflict"
    title = "Idempotency key reused with a different request body"


class PayloadTooLargeError(SupermemoryError):
    status_code = 413
    slug = "payload_too_large"
    title = "Request body exceeds the configured limit"


class UnsupportedMediaTypeError(SupermemoryError):
    status_code = 415
    slug = "unsupported_media_type"
    title = "Unsupported content type"


class RateLimitedError(SupermemoryError):
    status_code = 429
    slug = "rate_limited"
    title = "Rate limit exceeded"

    def __init__(self, detail: str | None = None, *, retry_after: float = 1.0) -> None:
        super().__init__(detail, extra={"retry_after": round(retry_after, 3)})
        self.retry_after = retry_after


# -- 5xx ---------------------------------------------------------------------


class ProviderError(SupermemoryError):
    """An upstream embedding or LLM provider failed."""

    status_code = 502
    slug = "provider_error"
    title = "Upstream provider failed"


class ProviderTimeoutError(ProviderError):
    status_code = 504
    slug = "provider_timeout"
    title = "Upstream provider timed out"


class StoreError(SupermemoryError):
    status_code = 503
    slug = "store_unavailable"
    title = "Storage backend unavailable"


class ConfigurationError(SupermemoryError):
    status_code = 500
    slug = "configuration_error"
    title = "Service is misconfigured"


__all__ = [
    "CONTENT_TYPE",
    "BadRequestError",
    "ConfigurationError",
    "ConflictError",
    "ForbiddenError",
    "IdempotencyConflictError",
    "NotFoundError",
    "PayloadTooLargeError",
    "ProviderError",
    "ProviderTimeoutError",
    "RateLimitedError",
    "StoreError",
    "SupermemoryError",
    "UnauthorizedError",
    "UnsupportedMediaTypeError",
    "ValidationError",
]
