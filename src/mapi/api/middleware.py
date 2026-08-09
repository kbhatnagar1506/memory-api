"""HTTP middleware: correlation ids, access logs, metrics, body limits.

Order matters and is set in main.py: the body-size guard runs outermost so an
oversized upload is rejected before anything else allocates for it.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ..core.errors import CONTENT_TYPE, PayloadTooLargeError
from ..core.logging import get_logger, org_id_var, request_id_var
from ..core.metrics import REQUEST_LATENCY, REQUESTS

log = get_logger("mapi.http")

REQUEST_ID_HEADER = "x-request-id"

Next = Callable[[Request], Awaitable[Response]]


def _route_template(request: Request) -> str:
    """The route pattern, not the resolved path.

    Metrics labelled with resolved paths create one time series per memory id,
    which is how a Prometheus instance runs out of memory.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Next) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        # Trust an inbound id only if it looks like one; otherwise it is a log
        # injection vector.
        request_id = (
            incoming
            if incoming and 8 <= len(incoming) <= 128 and incoming.isprintable()
            else uuid.uuid4().hex
        )
        token = request_id_var.set(request_id)
        org_token = org_id_var.set(None)
        request.state.request_id = request_id
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            limit = getattr(request.state, "rate_limit", None)
            if limit is not None:
                response.headers["x-ratelimit-limit"] = str(limit.limit)
                response.headers["x-ratelimit-remaining"] = str(limit.remaining)
                response.headers["x-ratelimit-reset"] = str(round(limit.reset_after, 3))
            return response
        finally:
            elapsed = time.perf_counter() - started
            template = _route_template(request)
            REQUESTS.labels(request.method, template, str(status)).inc()
            REQUEST_LATENCY.labels(request.method, template).observe(elapsed)
            if template != "/health" and template != "/metrics":
                log.info(
                    "request",
                    method=request.method,
                    path=template,
                    status=status,
                    duration_ms=round(elapsed * 1000, 2),
                )
            request_id_var.reset(token)
            org_id_var.reset(org_token)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies by Content-Length before reading them."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        raw = request.headers.get("content-length")
        if raw:
            try:
                declared = int(raw)
            except ValueError:
                declared = 0
            if declared > self.max_bytes:
                err = PayloadTooLargeError(
                    f"body of {declared} bytes exceeds the {self.max_bytes} byte limit"
                )
                return JSONResponse(
                    err.to_problem(instance=str(request.url.path)),
                    status_code=err.status_code,
                    media_type=CONTENT_TYPE,
                )
        return await call_next(request)


__all__ = [
    "REQUEST_ID_HEADER",
    "BodySizeLimitMiddleware",
    "RequestContextMiddleware",
]
