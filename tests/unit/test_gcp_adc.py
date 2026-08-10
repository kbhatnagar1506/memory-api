"""Materializing ADC from a config var.

ADC is a resolution order, not one mechanism. On a dyno there is no gcloud to
have logged in, so the only path that exists is a service-account key at
GOOGLE_APPLICATION_CREDENTIALS -- and a platform can only hand us the blob.
"""

from __future__ import annotations

import json

import pytest

from mapi.core.gcp import JSON_ENV, PATH_ENV, materialize_adc

KEY = json.dumps(
    {
        "type": "service_account",
        "project_id": "p",
        "client_email": "sa@p.iam.gserviceaccount.com",
        "private_key": "-----BEGIN PRIVATE KEY-----\nx\n-----END PRIVATE KEY-----\n",
    }
)


def test_writes_the_key_and_points_adc_at_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JSON_ENV, KEY)
    monkeypatch.delenv(PATH_ENV, raising=False)
    path = materialize_adc()
    assert path is not None and path.exists()
    assert json.loads(path.read_text())["client_email"] == "sa@p.iam.gserviceaccount.com"
    import os

    assert os.environ[PATH_ENV] == str(path)


def test_the_key_is_not_world_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JSON_ENV, KEY)
    monkeypatch.delenv(PATH_ENV, raising=False)
    path = materialize_adc()
    assert path is not None
    assert path.stat().st_mode & 0o077 == 0, "a private key must not be group/world readable"


def test_local_adc_is_never_disturbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The normal local case: real ADC exists and must be left alone."""
    monkeypatch.delenv(JSON_ENV, raising=False)
    assert materialize_adc() is None


def test_an_explicit_path_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Someone who mounted a real key file meant it."""
    monkeypatch.setenv(JSON_ENV, KEY)
    monkeypatch.setenv(PATH_ENV, "/mnt/secrets/key.json")
    assert materialize_adc() is None


def test_a_malformed_blob_degrades_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad config var must surface as a provider error at first use, not as
    a boot crash that takes the whole app down with no way in."""
    for junk in ("not json", "{}", '{"type":"service_account"}', "[]"):
        monkeypatch.setenv(JSON_ENV, junk)
        monkeypatch.delenv(PATH_ENV, raising=False)
        assert materialize_adc() is None
