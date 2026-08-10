"""Application factory and lifespan.

`create_app` builds a fully wired application from settings and nothing else, so
tests construct one with an in-memory store and no network, and production
constructs one with Postgres and Gemini, through the same code path.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import (
    RerankBackend,
    Settings,
    get_settings,
)
from .core.errors import (
    CONTENT_TYPE,
    ConfigurationError,
    NotFoundError,
    SupermemoryError,
    ValidationError,
)
from .core.gcp import materialize_adc
from .core.logging import configure_logging, get_logger
from .core.ratelimit import build_rate_limiter
from .domain.embeddings import build_embedder
from .domain.retrieval.rerank import (
    HeuristicReranker,
    NoopReranker,
    Reranker,
)
from .service import MemoryService
from .store import build_store

log = get_logger("mapi")

DESCRIPTION = """\
A memory API for AI agents.

**Hybrid retrieval** fuses dense vector search with BM25 lexical search using
Reciprocal Rank Fusion, then reranks, applies recency decay, and diversifies
with Maximal Marginal Relevance.

**Belief revision** is the part most memory APIs lack: memories form a typed
graph (`supersedes`, `contradicts`), so a fact that has been overtaken stops
being returned instead of sitting next to the fact that replaced it.

Every result carries its full score provenance under `explain`.
"""


def build_reranker(settings: Settings) -> Reranker:
    backend = settings.rerank_backend
    if backend is RerankBackend.NONE:
        return NoopReranker()
    if backend is RerankBackend.HEURISTIC:
        return HeuristicReranker()
    if backend is RerankBackend.LLM:
        from .domain.retrieval.rerank import LLMReranker

        try:
            return LLMReranker(
                model=settings.rerank_model,
                timeout_s=settings.rerank_timeout_s,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                api_key=settings.gemini_api_key,
            )
        except Exception as exc:
            # Reranking is an enhancement. Refusing to boot because an optional
            # provider is unreachable would be the wrong trade.
            log.warning("llm_reranker_unavailable", error=str(exc)[:200])
            return HeuristicReranker()
    raise ConfigurationError(f"unknown rerank backend: {backend}")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    problems = settings.validate_production()
    if problems:
        for problem in problems:
            log.error("production_config_problem", problem=problem)
        raise ConfigurationError(f"refusing to start in production: {'; '.join(problems)}")

    store = build_store(settings)
    await store.initialize()
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)

    app.state.store = store
    app.state.embedder = embedder
    app.state.reranker = reranker
    from .domain.synthesis.completer import build_completer, build_extractor

    app.state.service = MemoryService(
        store,
        embedder,
        reranker,
        settings,
        completer=build_completer(settings),
        extractor=build_extractor(settings),
    )
    app.state.rate_limiter = build_rate_limiter(
        per_minute=settings.rate_limit_per_minute,
        burst=settings.rate_limit_burst,
        redis_url=settings.redis_url,
    )

    if settings.bootstrap_admin_key:
        await _seed_bootstrap(app, settings)

    log.info(
        "started",
        version=__version__,
        environment=str(settings.environment),
        store=str(settings.store_backend),
        embeddings=str(settings.embedding_backend),
        rerank=reranker.name,
        dimensions=settings.embedding_dimensions,
    )
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await store.aclose()
        with contextlib.suppress(Exception):
            await embedder.aclose()
        with contextlib.suppress(Exception):
            await app.state.rate_limiter.aclose()
        log.info("stopped")


async def _seed_bootstrap(app: FastAPI, settings: Settings) -> None:
    """Create a first organization, space and admin key for local development.

    Production validation rejects `bootstrap_admin_key`, so this path cannot run
    against a real deployment.
    """
    from .core.security import hash_key
    from .domain.models import ApiKey, Organization, Scope, Space

    store = app.state.store
    key_hash = hash_key(settings.bootstrap_admin_key or "", settings.api_key_pepper)
    if await store.get_api_key_by_hash(key_hash) is not None:
        return

    org = await store.create_organization(Organization(name="Bootstrap"))
    space = await store.create_space(Space(org_id=org.id, slug="default", name="Default space"))
    await store.create_api_key(
        ApiKey(
            org_id=org.id,
            name="bootstrap-admin",
            key_hash=key_hash,
            prefix=(settings.bootstrap_admin_key or "")[:11],
            scopes=frozenset(Scope.all()),
        )
    )
    app.state.bootstrap = {"org_id": org.id, "space_id": space.id}
    log.warning(
        "bootstrap_seeded",
        org_id=org.id,
        space_id=space.id,
        note="development only; never enabled in production",
    )


def _problem(request: Request, exc: SupermemoryError) -> JSONResponse:
    settings: Settings = request.app.state.settings
    payload = exc.to_problem(instance=str(request.url.path))
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        payload["request_id"] = request_id
    headers: dict[str, str] = {}
    if exc.status_code == 429:
        headers["retry-after"] = str(int(max(getattr(exc, "retry_after", 1.0), 1)))
    if exc.status_code == 401:
        headers["www-authenticate"] = 'Bearer realm="mapi"'
    if not settings.debug_errors and exc.status_code >= 500:
        payload["detail"] = exc.title
    return JSONResponse(
        payload, status_code=exc.status_code, media_type=CONTENT_TYPE, headers=headers
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SupermemoryError)
    async def _handle(request: Request, exc: SupermemoryError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("request_failed", code=exc.slug, detail=exc.detail)
        return _problem(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        wrapped = ValidationError(
            first.get("msg", "request failed validation"),
            field=location or None,
            extra={
                "errors": [
                    {
                        "field": ".".join(str(p) for p in e.get("loc", ()) if p != "body"),
                        "message": e.get("msg", ""),
                        "type": e.get("type", ""),
                    }
                    for e in exc.errors()[:20]
                ]
            },
        )
        return _problem(request, wrapped)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        mapped: SupermemoryError
        if exc.status_code == 404:
            mapped = NotFoundError(str(exc.detail))
        else:
            mapped = SupermemoryError(str(exc.detail))
            mapped.status_code = exc.status_code
            mapped.slug = f"http_{exc.status_code}"
            mapped.title = str(exc.detail)
        return _problem(request, mapped)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Anything reaching here is a bug. Log it fully, tell the client nothing.
        log.exception("unhandled_exception", error=str(exc)[:500])
        return _problem(request, SupermemoryError("an unexpected error occurred"))


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_json)
    # Before any provider client is constructed: on a platform without gcloud
    # the service-account key arrives as a config var, and google-auth wants a
    # path. No-op locally, where real ADC already exists.
    materialize_adc()

    app = FastAPI(
        title="Mapi",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "Mapi"},
        license_info={"name": "MIT"},
    )
    app.state.settings = resolved

    from .api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
    from .api.v1.health import router as health_router
    from .api.v1.router import api_router

    # Added last runs first: the size guard rejects oversized bodies before the
    # context middleware allocates anything for them.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=resolved.max_request_bytes)

    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router)

    # The product surface: pages for a person, cookie-authenticated. Mounted
    # after the API so a versioned route always wins a path collision.
    from fastapi.staticfiles import StaticFiles

    from .api.web import router as web_router

    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "api" / "static")),
        name="static",
    )
    app.include_router(web_router)

    @app.get("/graph", include_in_schema=False, response_class=HTMLResponse)
    async def graph_ui() -> str:
        """Read-only viewer for a space's memory graph.

        Served by the app rather than shipped as a separate front end: it is a
        debugging surface for the relation graph, and a viewer needing its own
        build step is a viewer nobody runs. Requires ?space=&key= -- it holds
        no credentials of its own.
        """
        from .api.graph_ui import PAGE

        return PAGE

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        """Browsers ask for this on every page; without it the logs fill with
        404s that look like real misses."""
        return FileResponse(Path(__file__).parent / "api" / "static" / "mapi-logo.png")

    @app.get("/meta", include_in_schema=False)
    async def meta() -> dict[str, Any]:
        return {
            "name": "mapi",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app  # uvicorn --factory mapi.main:app

__all__ = ["build_reranker", "create_app", "lifespan", "register_error_handlers"]
