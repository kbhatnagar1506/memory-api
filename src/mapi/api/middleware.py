"""HTTP middleware: correlation ids, access logs, metrics, body limits, deadlines.

Order matters and is set in main.py: the body-size guard runs outermost so an
oversized upload is rejected before anything else allocates for it, and the
deadline runs innermost so a timed-out request still gets a request id, an
access-log line and a metric like any other.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable

import orjson
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..core.errors import CONTENT_TYPE, PayloadTooLargeError, RequestTimeoutError
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


class RequestDeadlineMiddleware:
    """Answer 504 when a request has not begun its response within `timeout_s`.

    `request_timeout_s` was declared and read by nothing, so the only deadline
    a request had was whatever the platform in front imposed -- 120 s on Cloud
    Run, none at all behind a plain VM port. A caller with its own 1.2 s budget
    (facemash's question path) abandons the request long before that, and the
    server kept embedding, querying and holding a pool connection for an answer
    nobody would read. Under load that is how a slow provider turns into an
    exhausted connection pool.

    The deadline covers the time until the response STARTS, not its whole
    body. A JSON endpoint starts and finishes in one step, so for those it is
    the total; a streamed response that has already started is left alone,
    because cutting one mid-body produces a truncated document with a 200
    status, which is worse than either outcome.

    Pure ASGI rather than `BaseHTTPMiddleware`: the work runs as its own task
    so it can be cancelled, and cancellation propagates into the store and the
    provider client, which is what actually releases the connection.
    """

    def __init__(self, app: ASGIApp, timeout_s: float) -> None:
        self.app = app
        #: 0 disables the deadline, for deployments that bound requests
        #: somewhere else and for tests that measure without one.
        self.timeout_s = timeout_s

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.timeout_s <= 0:
            await self.app(scope, receive, send)
            return

        started = asyncio.Event()

        async def tracking_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                started.set()
            await send(message)

        work = asyncio.ensure_future(self.app(scope, receive, tracking_send))
        began = asyncio.ensure_future(started.wait())
        try:
            await asyncio.wait(
                {work, began}, timeout=self.timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            began.cancel()

        if work.done() or started.is_set():
            # Finished, or the response is under way: no deadline applies any
            # more, and awaiting re-raises whatever the app raised.
            await work
            return

        work.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await work
        if started.is_set():
            # The app began a response while being cancelled; a second start
            # message would be a protocol violation.
            return

        path = scope.get("path", "")
        log.warning("request_deadline_exceeded", path=path, timeout_s=self.timeout_s)
        error = RequestTimeoutError(
            f"no response within the {self.timeout_s:g} second request deadline"
        )
        problem = error.to_problem(instance=str(path))
        request_id = (scope.get("state") or {}).get("request_id")
        if request_id:
            problem["request_id"] = request_id
        body = orjson.dumps(problem)
        await send(
            {
                "type": "http.response.start",
                "status": error.status_code,
                "headers": [
                    (b"content-type", CONTENT_TYPE.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


__all__ = [
    "REQUEST_ID_HEADER",
    "BodySizeLimitMiddleware",
    "RequestContextMiddleware",
    "RequestDeadlineMiddleware",
]
