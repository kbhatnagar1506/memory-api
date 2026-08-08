"""Liveness, readiness and metrics.

Liveness and readiness are genuinely different. Liveness answers "is this
process wedged" — if it fails, restart me. Readiness answers "can I serve
traffic right now" — if it fails, take me out of the load balancer but do not
restart, because the dependency will come back. Conflating them turns a brief
database blip into a cluster-wide restart loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ... import __version__
from ...core.metrics import REGISTRY, STORE_UP
from ..schemas import HealthResponse

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse, summary="Liveness")
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok", version=__version__,
        environment=str(settings.environment), checks={"process": True},
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness")
async def ready(request: Request, response: Response) -> HealthResponse:
    store = request.app.state.store
    settings = request.app.state.settings
    try:
        store_ok = await store.ping()
    except Exception:  # noqa: BLE001 - a readiness probe must never raise
        store_ok = False
    STORE_UP.set(1 if store_ok else 0)

    checks = {"store": store_ok}
    healthy = all(checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        environment=str(settings.environment),
        checks=checks,
    )


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


__all__ = ["router"]
