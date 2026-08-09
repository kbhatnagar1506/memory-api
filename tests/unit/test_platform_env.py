"""Adopting the host platform's connection strings.

Heroku attaches addons by setting DATABASE_URL and REDIS_URL and knows
nothing about our MAPI_ prefix. Worse, it still issues `postgres://`, a
scheme SQLAlchemy dropped -- so the platform's own URL is unusable verbatim
and the failure appears as an opaque dialect error at first connection, in
production, on a deploy that passed every test.
"""

from __future__ import annotations

import pytest

from mapi.config import Settings


def test_heroku_postgres_url_is_rewritten_for_asyncpg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@ec2.compute.amazonaws.com:5432/d1")
    monkeypatch.delenv("MAPI_DATABASE_URL", raising=False)
    assert (
        Settings().database_url == "postgresql+asyncpg://u:p@ec2.compute.amazonaws.com:5432/d1"
    )


def test_bare_postgresql_scheme_is_also_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/d")
    monkeypatch.delenv("MAPI_DATABASE_URL", raising=False)
    assert Settings().database_url == "postgresql+asyncpg://u:p@host:5432/d"


def test_an_explicit_setting_always_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adoption is a fallback. An operator who set ours meant it."""
    monkeypatch.setenv("DATABASE_URL", "postgres://platform/db")
    monkeypatch.setenv("MAPI_DATABASE_URL", "postgresql+asyncpg://mine/db")
    assert Settings().database_url == "postgresql+asyncpg://mine/db"


def test_redis_url_is_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "rediss://:tok@host:6380")
    monkeypatch.delenv("MAPI_REDIS_URL", raising=False)
    assert Settings().redis_url == "rediss://:tok@host:6380"


def test_nothing_is_invented_when_the_platform_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("MAPI_DATABASE_URL", raising=False)
    monkeypatch.delenv("MAPI_REDIS_URL", raising=False)
    s = Settings()
    assert s.database_url is None
    assert s.redis_url is None
