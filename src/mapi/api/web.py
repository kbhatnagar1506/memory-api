"""Browser routes: sign in, choose an organization, open its dashboard.

Separate from the versioned API because these are pages for a person, not
endpoints for a program. They authenticate with a cookie rather than a bearer
key, return HTML rather than JSON, and redirect rather than raising 401 --
a browser handed a JSON error is a dead end.
"""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ..config import Settings
from ..core.errors import ConflictError, NotFoundError
from ..core.logging import get_logger
from ..core.security import build_api_key
from ..domain.models import Scope, User
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


@router.get("/orgs/{org_id}", response_class=HTMLResponse)
async def dashboard(request: Request, user: CurrentUser, org_id: str) -> Response:
    if user is None:
        return RedirectResponse("/auth/google/login", status_code=303)
    identity = _identity(request)
    # 404 for a non-member, same as everywhere else: a 403 would confirm the
    # organization exists to anyone who can guess an id.
    await identity.require_member(user, org_id)

    store = request.app.state.store
    org = await store.get_organization(org_id)
    if org is None:
        raise NotFoundError(f"organization {org_id} not found", field="org_id")

    spaces = []
    for space in await store.list_spaces(org_id):
        count = await store.count_memories(org_id, space.id, filters=MemoryFilter())
        spaces.append({"id": space.id, "name": space.name, "memories": count})

    keys = [
        {
            "name": k.name,
            "prefix": k.id[:12],
            "created": k.created_at.date().isoformat(),
        }
        for k in await store.list_api_keys(org_id)
    ]
    # Shown once, then gone: the plaintext is never stored, so a page reload
    # cannot reveal it again.
    revealed = request.query_params.get("key", "")
    return HTMLResponse(
        pages.dashboard(org.name, org.id, user.email, spaces, keys, new_key=revealed)
    )


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


@router.post("/orgs/{org_id}/keys")
async def create_key(
    request: Request, user: CurrentUser, org_id: str, name: Annotated[str, Form()]
) -> Response:
    if user is None:
        return RedirectResponse("/auth/google/login", status_code=303)
    await _identity(request).require_member(user, org_id)
    settings = _settings(request)
    record, plaintext = build_api_key(
        org_id=org_id,
        name=name.strip()[:120] or "key",
        pepper=settings.api_key_pepper,
        scopes=frozenset(Scope.all()),
    )
    await request.app.state.store.create_api_key(record)
    # Carried in the redirect so it is shown exactly once. Not logged, and not
    # recoverable afterwards -- only the hash is kept.
    from urllib.parse import quote

    return RedirectResponse(f"/orgs/{org_id}?key={quote(plaintext)}", status_code=303)


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
