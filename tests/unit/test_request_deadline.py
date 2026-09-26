"""The request deadline: `request_timeout_s`, which used to be read by nothing.

Behaviour that matters to a caller: a slow request is answered 504 with the
usual problem document instead of hanging; the slow work is actually
cancelled rather than left holding a connection; a response that has already
started is never cut off mid-body; and 0 turns the whole thing off.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from mapi.api.middleware import RequestContextMiddleware, RequestDeadlineMiddleware
from mapi.config import Settings
from mapi.core.errors import CONTENT_TYPE
from mapi.main import create_app


def _app(timeout_s: float, state: dict[str, object]) -> Starlette:
    async def fast(_: Request) -> Response:
        return JSONResponse({"ok": True})

    async def slow(_: Request) -> Response:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return JSONResponse({"ok": "too late"})

    async def stream(_: Request) -> Response:
        async def body():
            yield b"first,"
            await asyncio.sleep(0.2)  # well past the deadline, after the start
            yield b"second"

        return StreamingResponse(body(), media_type="text/plain")

    async def broken(_: Request) -> Response:
        raise RuntimeError("boom")

    app = Starlette(
        routes=[
            Route("/fast", fast),
            Route("/slow", slow),
            Route("/stream", stream),
            Route("/broken", broken),
        ]
    )
    app.add_middleware(RequestDeadlineMiddleware, timeout_s=timeout_s)
    app.add_middleware(RequestContextMiddleware)
    return app


def _client(app: Starlette) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://t",
    )


async def test_a_slow_request_gets_a_504_problem_document() -> None:
    state: dict[str, object] = {}
    async with _client(_app(0.05, state)) as client:
        started = asyncio.get_running_loop().time()
        response = await client.get("/slow")
        elapsed = asyncio.get_running_loop().time() - started

    assert response.status_code == 504
    assert response.headers["content-type"] == CONTENT_TYPE
    body = response.json()
    assert body["code"] == "request_timeout"
    assert body["status"] == 504
    assert body["instance"] == "/slow"
    # Wrapped by the context middleware, so it is traceable like any response.
    assert body["request_id"] == response.headers["x-request-id"]
    assert elapsed < 2, "the deadline did not bound the request"


async def test_the_slow_work_is_cancelled_not_abandoned() -> None:
    """The point: a pool connection held for an answer nobody reads is the
    failure this exists to prevent, and only cancellation releases it."""
    state: dict[str, object] = {}
    async with _client(_app(0.05, state)) as client:
        await client.get("/slow")
    assert state.get("cancelled") is True


async def test_fast_requests_are_untouched() -> None:
    async with _client(_app(0.5, {})) as client:
        response = await client.get("/fast")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


async def test_a_started_response_is_never_cut_off() -> None:
    """A stream that began in time runs to its end, deadline or not: cutting it
    would deliver a truncated body under a 200."""
    async with _client(_app(0.05, {})) as client:
        response = await client.get("/stream")
    assert response.status_code == 200
    assert response.text == "first,second"


async def test_zero_disables_the_deadline() -> None:
    state: dict[str, object] = {}
    app = _app(0, state)

    async def quick_slow(_: Request) -> Response:
        await asyncio.sleep(0.1)
        return JSONResponse({"ok": "finished"})

    app.router.routes.append(Route("/quick-slow", quick_slow))
    async with _client(app) as client:
        response = await client.get("/quick-slow")
    assert response.status_code == 200


async def test_application_errors_still_propagate() -> None:
    """The deadline must not swallow a failure into a timeout, or a success."""
    async with _client(_app(1.0, {})) as client:
        response = await client.get("/broken")
    assert response.status_code == 500


def test_the_setting_accepts_zero_and_refuses_negatives() -> None:
    assert Settings(request_timeout_s=0).request_timeout_s == 0
    with pytest.raises(ValueError):
        Settings(request_timeout_s=-1)


def test_the_deadline_is_opt_in() -> None:
    """Nothing enforced the old declared 30 s, so every existing deployment
    runs without a deadline; upgrading must not start cutting requests off."""
    assert Settings().request_timeout_s == 0


async def test_the_real_app_has_no_deadline_unless_asked(monkeypatch) -> None:
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=64,
        api_key_pepper="test-pepper",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        store = app.state.store
        original = store.ping

        async def _slowish() -> bool:
            await asyncio.sleep(0.1)
            return await original()

        monkeypatch.setattr(store, "ping", _slowish)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as client:
            assert (await client.get("/ready")).status_code == 200


async def test_the_real_app_enforces_its_setting(monkeypatch) -> None:
    """Wired into create_app, inside the context middleware."""
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=64,
        api_key_pepper="test-pepper",
        request_timeout_s=0.05,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        store = app.state.store

        async def _hang() -> bool:
            await asyncio.sleep(5)
            return True

        monkeypatch.setattr(store, "ping", _hang)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as client:
            response = await client.get("/ready")
            assert response.status_code == 504
            assert response.json()["code"] == "request_timeout"
            assert response.headers.get("x-request-id")
            assert (await client.get("/health")).status_code == 200
