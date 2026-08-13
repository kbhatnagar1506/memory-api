"""Browser routes: sign in, choose an organization, open its dashboard.

Separate from the versioned API because these are pages for a person, not
endpoints for a program. They authenticate with a cookie rather than a bearer
key, return HTML rather than JSON, and redirect rather than raising 401 --
a browser handed a JSON error is a dead end.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ..config import Settings
from ..core.errors import ConflictError, NotFoundError
from ..core.logging import get_logger
from ..core.security import build_api_key
from ..domain.models import Memory, Scope, User
from ..identity import MAX_ORGS_PER_USER, IdentityService
from ..store.base import MemoryFilter, MemoryStore
from . import pages
from .session import (
    COOKIE_NAME,
    STATE_COOKIE,
    authorize_url,
    build_google_auth,
    exchange_code,
    issue,
    new_state,
    read,
)

log = get_logger(__name__)
router = APIRouter(include_in_schema=False)


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _identity(request: Request) -> IdentityService:
    return IdentityService(request.app.state.store)


async def current_user(request: Request) -> User | None:
    """The signed-in person, or None. Never raises: pages redirect instead."""
    settings = _settings(request)
    if not settings.session_secret:
        return None
    user_id = read(
        settings.session_secret,
        request.cookies.get(COOKIE_NAME, ""),
        max_age_days=settings.session_max_age_days,
    )
    if not user_id:
        return None
    # Resolved from the store per request rather than trusted from the
    # cookie: the cookie carries an id and nothing else, so a removed user or
    # a changed email takes effect on the next click rather than at expiry.
    store: MemoryStore = request.app.state.store
    return await store.get_user(user_id)


CurrentUser = Annotated[User | None, Depends(current_user)]


# -- pages -------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request, user: CurrentUser) -> HTMLResponse:
    settings = _settings(request)
    auth = build_google_auth(
        settings.google_client_id, settings.google_client_secret, settings.public_base_url
    )
    return HTMLResponse(pages.landing(user is not None, auth is not None))


@router.get("/orgs", response_class=HTMLResponse)
async def list_orgs(request: Request, user: CurrentUser, error: str = "") -> Response:
    if user is None:
        return RedirectResponse("/auth/google/login", status_code=303)
    summaries = await _identity(request).list_organizations(user)
    return HTMLResponse(
        pages.organizations(
            user.email,
            [
                {
                    "id": s.organization.id,
                    "name": s.organization.name,
                    "role": s.role.value,
                    "spaces": s.spaces,
                }
                for s in summaries
            ],
            cap=MAX_ORGS_PER_USER,
            error=error,
        )
    )


@router.post("/orgs")
async def create_org(
    request: Request, user: CurrentUser, name: Annotated[str, Form()]
) -> Response:
    if user is None:
        return RedirectResponse("/auth/google/login", status_code=303)
    try:
        await _identity(request).create_organization(user, name)
    except ConflictError as exc:
        # Back to the list with the reason, rather than a JSON error page a
        # browser cannot act on.
        from urllib.parse import quote

        return RedirectResponse(f"/orgs?error={quote(str(exc))}", status_code=303)
    return RedirectResponse("/orgs", status_code=303)


# -- dashboard JSON, cookie-authenticated ------------------------------------
#
# Deliberately NOT the versioned API. That surface takes a bearer key and
# belongs to programs; these belong to a signed-in person and take a cookie.
# Keeping them apart means the public API never has to learn about sessions,
# and a browser bug cannot widen what a key can do.


async def _member_or_404(request: Request, user: User | None, org_id: str) -> User:
    if user is None:
        raise NotFoundError(f"organization {org_id} not found", field="org_id")
    await _identity(request).require_member(user, org_id)
    return user


@router.get("/orgs/{org_id}/api/spaces")
async def api_spaces(request: Request, user: CurrentUser, org_id: str) -> dict[str, object]:
    await _member_or_404(request, user, org_id)
    store = request.app.state.store
    items = []
    for space in await store.list_spaces(org_id):
        items.append(
            {
                "id": space.id,
                "name": space.name,
                "memory_count": await store.count_memories(
                    org_id, space.id, filters=MemoryFilter()
                ),
                "metadata": space.metadata,
            }
        )
    return {"items": items}


@router.get("/orgs/{org_id}/api/graph")
async def api_graph(
    request: Request, user: CurrentUser, org_id: str, space: str, limit: int = 400
) -> dict[str, object]:
    await _member_or_404(request, user, org_id)
    graph = await request.app.state.service.get_graph(org_id, space, limit=limit)
    return {
        "space_id": graph.space_id,
        "nodes": [
            {
                "id": m.id,
                "content": m.content[:400],
                "kind": m.kind.value,
                "status": m.status.value,
                "occurred_at": m.occurred_at.isoformat(),
                "tags": m.tags,
                "degree": graph.degree.get(m.id, 0),
            }
            for m in graph.memories
        ],
        "edges": [
            {
                "source": e.source_id,
                "target": e.target_id,
                "type": e.type.value,
                "reason": e.reason,
                "confidence": e.confidence,
            }
            for e in graph.edges
        ],
        "counts": graph.counts,
    }


@router.get("/orgs/{org_id}/api/context/{memory_id}")
async def api_context(
    request: Request, user: CurrentUser, org_id: str, memory_id: str, space: str
) -> dict[str, object]:
    await _member_or_404(request, user, org_id)
    ctx = await request.app.state.service.get_memory_context(org_id, space, memory_id)

    def brief(items: Sequence[Memory]) -> list[dict[str, str]]:
        return [
            {
                "id": m.id,
                "content": m.content[:400],
                "status": m.status.value,
                "occurred_at": m.occurred_at.isoformat(),
            }
            for m in items
        ]

    return {
        "memory": {
            "id": ctx.memory.id,
            "content": ctx.memory.content,
            "status": ctx.memory.status.value,
            "kind": ctx.memory.kind.value,
            "occurred_at": ctx.memory.occurred_at.isoformat(),
        },
        "is_current": ctx.is_current,
        "current_head": brief(ctx.current_head),
        "replaced": brief(ctx.replaced),
        "derived_from": brief(ctx.derived_from),
        "derivatives": brief(ctx.derivatives),
        "references": brief(ctx.references),
        "contradicts": brief(ctx.contradicts),
    }


@router.get("/orgs/{org_id}/api/keys")
async def api_keys(request: Request, user: CurrentUser, org_id: str) -> dict[str, object]:
    await _member_or_404(request, user, org_id)
    return {
        "items": [
            {
                "id": k.id,
                "name": k.name,
                "created_at": k.created_at.isoformat(),
                "scopes": sorted(s.value for s in k.scopes),
            }
            for k in await request.app.state.store.list_api_keys(org_id)
        ]
    }


@router.get("/orgs/{org_id}", response_class=HTMLResponse)
async def dashboard(request: Request, user: CurrentUser, org_id: str) -> Response:
    """The organization's dashboard: graph, replay and keys in one shell.

    Serves the same document as the standalone viewer. It reads the org from
    the path and authenticates with the session cookie, so no key appears in
    a URL -- a key in a query string ends up in history, referrers and logs.
    """
    if user is None:
        return RedirectResponse("/auth/google/login", status_code=303)
    # 404 for a non-member, as everywhere else: a 403 confirms the
    # organization exists to anyone who can guess an id.
    await _identity(request).require_member(user, org_id)
    from .graph_ui import PAGE

    return HTMLResponse(PAGE)


@router.post("/orgs/{org_id}/keys.json")
async def create_key_json(request: Request, user: CurrentUser, org_id: str) -> dict[str, str]:
    """Mint a key for the dashboard's own use. Returns the plaintext ONCE."""
    await _member_or_404(request, user, org_id)
    payload = await request.json()
    name = str(payload.get("name") or "key").strip()[:120] or "key"
    record, plaintext = build_api_key(
        org_id=org_id,
        name=name,
        pepper=_settings(request).api_key_pepper,
        scopes=frozenset(Scope.all()),
    )
    await request.app.state.store.create_api_key(record)
    # Never logged and never stored: only the hash is kept, so this response
    # is the single opportunity to see it.
    return {"key": plaintext, "name": name}


@router.post("/orgs/{org_id}/spaces")
async def create_space(
    request: Request, user: CurrentUser, org_id: str, name: Annotated[str, Form()]
) -> Response:
    if user is None:
        return RedirectResponse("/auth/google/login", status_code=303)
    await _identity(request).require_member(user, org_id)
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().casefold()).strip("-") or "space"
    await request.app.state.service.create_space(
        org_id, slug=slug[:64], name=name.strip()[:200]
    )
    return RedirectResponse(f"/orgs/{org_id}", status_code=303)


# -- auth --------------------------------------------------------------------


@router.get("/auth/google/login")
async def google_login(request: Request) -> Response:
    settings = _settings(request)
    auth = build_google_auth(
        settings.google_client_id, settings.google_client_secret, settings.public_base_url
    )
    if auth is None or not settings.session_secret:
        return HTMLResponse(pages.landing(False, False), status_code=503)
    state = new_state()
    response = RedirectResponse(authorize_url(auth, state), status_code=307)
    # The state is echoed by Google and compared on return: without it, an
    # attacker can complete a sign-in in someone else's browser.
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
    )
    return response


@router.get("/auth/google/callback")
async def google_callback(
    request: Request, code: str = "", state: str = "", error: str = ""
) -> Response:
    settings = _settings(request)
    auth = build_google_auth(
        settings.google_client_id, settings.google_client_secret, settings.public_base_url
    )
    expected = request.cookies.get(STATE_COOKIE, "")
    if error or not code or not state or state != expected or auth is None:
        log.warning("oauth_callback_rejected", has_code=bool(code), state_ok=state == expected)
        return RedirectResponse("/?auth=failed", status_code=303)

    claims = await exchange_code(auth, code)
    user = await _identity(request).sign_in_with_google(claims)

    response = RedirectResponse("/orgs", status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        issue(settings.session_secret or "", user.id),
        max_age=settings.session_max_age_days * 86_400,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
    )
    response.delete_cookie(STATE_COOKIE)
    return response


@router.get("/auth/logout")
async def logout() -> Response:
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response


__all__ = ["router"]
