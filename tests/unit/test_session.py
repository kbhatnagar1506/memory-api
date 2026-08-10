"""Browser sessions: signed, expiring, and carrying only an id."""

from __future__ import annotations

import time

import pytest

from mapi.api.session import (
    GoogleAuth,
    _claims,
    authorize_url,
    build_google_auth,
    issue,
    read,
)
from mapi.core.errors import ForbiddenError

SECRET = "a-test-secret"


def test_a_session_round_trips() -> None:
    assert read(SECRET, issue(SECRET, "usr_1"), max_age_days=14) == "usr_1"


def test_a_cookie_signed_with_another_secret_is_refused() -> None:
    forged = issue("attacker-secret", "usr_admin")
    assert read(SECRET, forged, max_age_days=14) is None


def test_a_tampered_cookie_is_refused() -> None:
    token = issue(SECRET, "usr_1")
    assert read(SECRET, token[:-3] + "aaa", max_age_days=14) is None


def test_an_expired_cookie_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    token = issue(SECRET, "usr_1")
    # Capture the real clock first: a lambda calling the patched name would
    # call itself.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 15 * 86_400)
    assert read(SECRET, token, max_age_days=14) is None


def test_an_absent_cookie_is_not_an_error() -> None:
    assert read(SECRET, "", max_age_days=14) is None


def test_the_authorize_url_carries_state_and_asks_for_a_choice() -> None:
    auth = GoogleAuth("cid", "secret", "https://x.test/auth/google/callback")
    url = authorize_url(auth, "st4te")
    assert "state=st4te" in url
    assert "prompt=select_account" in url
    assert "scope=openid+email+profile" in url
    assert "response_type=code" in url
    # the secret must never reach the browser
    assert "secret" not in url


def test_sign_in_is_optional() -> None:
    """The API works on bearer keys with no Google configured; a missing
    dashboard must not stop the service booting."""
    assert build_google_auth(None, None, None) is None
    assert build_google_auth("id", None, "https://x.test") is None
    auth = build_google_auth("id", "sec", "https://x.test/")
    assert auth is not None
    assert auth.redirect_uri == "https://x.test/auth/google/callback"


def test_malformed_identity_tokens_are_refused() -> None:
    for junk in ("", "not-a-jwt", "a.b", "a.!!!.c"):
        with pytest.raises(ForbiddenError):
            _claims(junk)
