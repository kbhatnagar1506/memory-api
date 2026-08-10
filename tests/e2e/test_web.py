"""The browser surface: sign-in gating, org isolation, and key reveal."""

from __future__ import annotations

import httpx
import pytest

from mapi.api.session import COOKIE_NAME, issue
from mapi.config import Settings
from mapi.identity import IdentityService
from mapi.main import create_app

SECRET = "test-session-secret"

CLAIMS = {"sub": "u1", "email": "a@x.test", "email_verified": True, "name": "A"}
OTHER = {"sub": "u2", "email": "b@x.test", "email_verified": True, "name": "B"}


@pytest.fixture
async def app_and_client():
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=128,
        api_key_pepper="test-pepper",
        session_secret=SECRET,
        google_client_id="cid",
        google_client_secret="csecret",
        public_base_url="http://test",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=False
        ) as client:
            yield app, client


def _as(user_id: str) -> dict[str, str]:
    return {COOKIE_NAME: issue(SECRET, user_id)}


async def test_landing_is_public(app_and_client) -> None:
    _, client = app_and_client
    r = await client.get("/")
    assert r.status_code == 200
    assert "Continue with Google" in r.text


async def test_orgs_requires_sign_in(app_and_client) -> None:
    _, client = app_and_client
    r = await client.get("/orgs")
    assert r.status_code == 303
    assert r.headers["location"] == "/auth/google/login"


async def test_a_member_sees_their_dashboard(app_and_client) -> None:
    app, client = app_and_client
    identity = IdentityService(app.state.store)
    user = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(user, "Acme")

    r = await client.get(f"/orgs/{org.id}", cookies=_as(user.id))
    assert r.status_code == 200
    assert "Acme" in r.text
    assert "API keys" in r.text


async def test_a_non_member_gets_404_not_403(app_and_client) -> None:
    """403 would confirm the organization exists to anyone guessing ids."""
    app, client = app_and_client
    identity = IdentityService(app.state.store)
    owner = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(owner, "Private")
    outsider = await identity.sign_in_with_google(OTHER)

    r = await client.get(f"/orgs/{org.id}", cookies=_as(outsider.id))
    assert r.status_code == 404


async def test_a_forged_cookie_does_not_sign_you_in(app_and_client) -> None:
    _, client = app_and_client
    forged = issue("not-the-secret", "usr_whoever")
    r = await client.get("/orgs", cookies={COOKIE_NAME: forged})
    assert r.status_code == 303


async def test_a_created_key_is_revealed_once(app_and_client) -> None:
    app, client = app_and_client
    identity = IdentityService(app.state.store)
    user = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(user, "Acme")

    r = await client.post(f"/orgs/{org.id}/keys", data={"name": "sdk"}, cookies=_as(user.id))
    assert r.status_code == 303
    assert "key=" in r.headers["location"]

    # and a plain reload never shows it again
    again = await client.get(f"/orgs/{org.id}", cookies=_as(user.id))
    assert "Copy this key now" not in again.text


async def test_a_non_member_cannot_mint_a_key(app_and_client) -> None:
    app, client = app_and_client
    identity = IdentityService(app.state.store)
    owner = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(owner, "Private")
    outsider = await identity.sign_in_with_google(OTHER)

    r = await client.post(
        f"/orgs/{org.id}/keys", data={"name": "theirs"}, cookies=_as(outsider.id)
    )
    assert r.status_code == 404
    assert await app.state.store.list_api_keys(org.id) == []


async def test_spaces_created_from_the_dashboard_appear(app_and_client) -> None:
    app, client = app_and_client
    identity = IdentityService(app.state.store)
    user = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(user, "Acme")

    await client.post(
        f"/orgs/{org.id}/spaces", data={"name": "Ada Lovelace"}, cookies=_as(user.id)
    )
    page = await client.get(f"/orgs/{org.id}", cookies=_as(user.id))
    assert "Ada Lovelace" in page.text
