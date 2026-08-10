"""Signed browser sessions, and the Google sign-in exchange.

Two authentication paths coexist on purpose, because they answer different
questions. The API takes a bearer key, which says an organization authorised
this call. The dashboard takes a cookie, which says a PERSON is here. A key
in a browser would be readable by any script on the page and could not be
revoked per person; a cookie in an SDK would be meaningless.

The session holds only a user id. Everything else -- email, name, which
organizations they belong to -- is read from the store per request, so
removing someone's membership takes effect on their next click rather than
whenever their cookie happens to expire.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..core.errors import ConfigurationError, ForbiddenError
from ..core.logging import get_logger

log = get_logger(__name__)

COOKIE_NAME = "mapi_session"
STATE_COOKIE = "mapi_oauth_state"

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

#: Only what identifies a person. No Drive, no Gmail, nothing that would make
#: the consent screen ask for more than the product needs.
SCOPES = "openid email profile"


@dataclass(frozen=True, slots=True)
class GoogleAuth:
    client_id: str
    client_secret: str
    redirect_uri: str


def serializer(secret: str) -> URLSafeTimedSerializer:
    # Salted so a value signed for one purpose cannot be replayed as another:
    # the session cookie and the OAuth state share a secret but not a domain.
    return URLSafeTimedSerializer(secret, salt="mapi.session")


def issue(secret: str, user_id: str) -> str:
    return serializer(secret).dumps({"uid": user_id})


def read(secret: str, token: str, *, max_age_days: int) -> str | None:
    """The user id in a cookie, or None if it is absent, forged or stale."""
    if not token:
        return None
    try:
        data = serializer(secret).loads(token, max_age=max_age_days * 86_400)
    except SignatureExpired:
        return None
    except BadSignature:
        # Worth a line: a forged cookie is either a bug in our signing or
        # somebody trying, and both are things you want to see.
        log.warning("session_bad_signature")
        return None
    return str(data.get("uid") or "") or None


def authorize_url(auth: GoogleAuth, state: str) -> str:
    """Where to send the browser to sign in."""
    from urllib.parse import urlencode

    params = {
        "client_id": auth.client_id,
        "redirect_uri": auth.redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        # Ask for a fresh choice rather than silently reusing whichever
        # account the browser happens to be signed into.
        "prompt": "select_account",
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


def new_state() -> str:
    return secrets.token_urlsafe(24)


async def exchange_code(auth: GoogleAuth, code: str) -> dict[str, Any]:
    """Trade an authorization code for the caller's identity claims.

    The ID token comes back over TLS from Google's token endpoint, in a
    response only someone holding our client secret could have obtained. That
    is what makes the authorization-code flow safe without separately
    verifying the token's signature -- the check exists for tokens received
    from elsewhere, such as a client passing one to a backend, which is not
    what happens here. Google documents exactly this exemption.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": auth.client_id,
                "client_secret": auth.client_secret,
                "redirect_uri": auth.redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if response.status_code != 200:
        # Never log the body: it echoes the code and can carry the secret.
        log.warning("oauth_token_exchange_failed", status=response.status_code)
        raise ForbiddenError("Google rejected the sign-in attempt")

    id_token = str(response.json().get("id_token") or "")
    if not id_token:
        raise ForbiddenError("Google returned no identity token")
    return _claims(id_token)


def _claims(id_token: str) -> dict[str, Any]:
    """The payload of a JWT whose signature was established by transport."""
    import base64
    import json

    try:
        payload = id_token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (IndexError, ValueError) as exc:
        raise ForbiddenError("Google returned a malformed identity token") from exc
    if not isinstance(claims, dict):
        raise ForbiddenError("Google returned a malformed identity token")
    return claims


def build_google_auth(
    client_id: str | None, client_secret: str | None, base_url: str | None
) -> GoogleAuth | None:
    """The OAuth config, or None when sign-in is not set up.

    None rather than raising: the API is fully usable on bearer keys without
    Google configured, and a missing dashboard should not stop the service
    from booting.
    """
    if not (client_id and client_secret and base_url):
        return None
    return GoogleAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"{base_url.rstrip('/')}/auth/google/callback",
    )


def require_session_secret(secret: str | None) -> str:
    if not secret:
        raise ConfigurationError("session_secret is required to sign in")
    return secret


__all__ = [
    "COOKIE_NAME",
    "STATE_COOKIE",
    "GoogleAuth",
    "authorize_url",
    "build_google_auth",
    "exchange_code",
    "issue",
    "new_state",
    "read",
    "require_session_secret",
]
