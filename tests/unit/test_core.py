"""Ids, security, rate limiting, configuration and error shapes."""

from __future__ import annotations

import pytest

from supermemory.config import Settings
from supermemory.core.errors import (
    NotFoundError,
    RateLimitedError,
    SupermemoryError,
    ValidationError,
)
from supermemory.core.ids import PREFIXES, is_valid, kind_of, new_id
from supermemory.core.ratelimit import InMemoryRateLimiter
from supermemory.core.security import (
    build_api_key,
    generate_key,
    hash_key,
    looks_like_key,
    parse_authorization,
    verify_key,
)
from supermemory.domain.models import Scope

# -- ids -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(PREFIXES))
def test_ids_round_trip(kind: str) -> None:
    value = new_id(kind)
    assert is_valid(value, kind)
    assert kind_of(value) == kind


def test_ids_are_unique() -> None:
    assert len({new_id("memory") for _ in range(5000)}) == 5000


def test_ids_are_time_sortable() -> None:
    ids = [new_id("memory", now_ms=t) for t in (1_000, 2_000, 3_000)]
    assert ids == sorted(ids)


def test_id_of_wrong_kind_is_rejected() -> None:
    assert not is_valid(new_id("space"), "memory")


@pytest.mark.parametrize("bad", ["", "nope", "mem_", "mem_short", 42, None, "MEM_ABC"])
def test_malformed_ids_rejected(bad: object) -> None:
    assert not is_valid(bad)


def test_unknown_id_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown id kind"):
        new_id("nonsense")


def test_negative_timestamp_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        new_id("memory", now_ms=-1)


# -- security ------------------------------------------------------------------


def test_key_verification_round_trip() -> None:
    generated = generate_key("pepper")
    assert verify_key(generated.plaintext, generated.key_hash, "pepper")


def test_key_verification_fails_with_a_different_pepper() -> None:
    """This is the property that protects a leaked database."""
    generated = generate_key("pepper")
    assert not verify_key(generated.plaintext, generated.key_hash, "other-pepper")


def test_keys_are_unique_and_prefixed() -> None:
    keys = [generate_key("p").plaintext for _ in range(200)]
    assert len(set(keys)) == 200
    assert all(k.startswith("sm_") for k in keys)


def test_hash_is_stable_and_not_the_plaintext() -> None:
    digest = hash_key("sm_abc", "pepper")
    assert digest == hash_key("sm_abc", "pepper")
    assert "sm_abc" not in digest
    assert len(digest) == 64


@pytest.mark.parametrize(
    "value,ok",
    [("sm_" + "a" * 30, True), ("nope", False), ("sm_short", False), ("", False)],
)
def test_key_shape_check(value: str, ok: bool) -> None:
    assert looks_like_key(value) is ok


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("Token abc", "abc"),
        ("abc", "abc"),
        ("", None),
        ("   ", None),
        (None, None),
        ("Bearer   ", None),
    ],
)
def test_authorization_parsing(header: str | None, expected: str | None) -> None:
    assert parse_authorization(header) == expected


def test_build_api_key_never_stores_plaintext() -> None:
    record, plaintext = build_api_key(org_id="org_x", name="k", pepper="p")
    assert plaintext not in record.key_hash
    assert record.prefix == plaintext[:11]
    assert record.is_active()


def test_admin_scope_implies_every_other_scope() -> None:
    record, _ = build_api_key(org_id="o", name="k", pepper="p", scopes=frozenset({Scope.ADMIN}))
    assert record.allows(Scope.MEMORIES_WRITE)
    assert record.allows(Scope.SEARCH)


def test_revoked_key_is_inactive() -> None:
    from supermemory.domain.models import utcnow

    record, _ = build_api_key(org_id="o", name="k", pepper="p")
    assert not record.model_copy(update={"revoked_at": utcnow()}).is_active()


def test_expired_key_is_inactive() -> None:
    from datetime import timedelta

    from supermemory.domain.models import utcnow

    record, _ = build_api_key(org_id="o", name="k", pepper="p")
    expired = record.model_copy(update={"expires_at": utcnow() - timedelta(seconds=1)})
    assert not expired.is_active()


# -- rate limiting -------------------------------------------------------------


async def test_bucket_allows_burst_then_rejects() -> None:
    limiter = InMemoryRateLimiter(per_minute=60, burst=3)
    outcomes = [(await limiter.check("k")).allowed for _ in range(5)]
    assert outcomes == [True, True, True, False, False]


async def test_buckets_are_per_key() -> None:
    limiter = InMemoryRateLimiter(per_minute=60, burst=1)
    assert (await limiter.check("a")).allowed
    assert (await limiter.check("b")).allowed


async def test_rejection_reports_retry_after() -> None:
    limiter = InMemoryRateLimiter(per_minute=60, burst=1)
    await limiter.check("k")
    decision = await limiter.check("k")
    assert not decision.allowed
    assert decision.retry_after > 0


async def test_tokens_refill_over_time() -> None:
    limiter = InMemoryRateLimiter(per_minute=6000, burst=1)
    assert (await limiter.check("k")).allowed
    import asyncio

    await asyncio.sleep(0.05)
    assert (await limiter.check("k")).allowed


@pytest.mark.parametrize("per_minute,burst", [(0, 1), (1, 0), (-1, 1)])
def test_invalid_rate_limit_configuration(per_minute: int, burst: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryRateLimiter(per_minute=per_minute, burst=burst)


# -- configuration -------------------------------------------------------------


def test_overlap_must_be_smaller_than_chunk_target() -> None:
    with pytest.raises(ValueError, match="non-terminating"):
        Settings(chunk_target_tokens=100, chunk_overlap_tokens=100)


def test_postgres_backend_requires_a_url() -> None:
    with pytest.raises(ValueError, match="requires database_url"):
        Settings(store_backend="postgres", database_url=None)


def test_max_limit_must_not_be_below_default() -> None:
    with pytest.raises(ValueError, match="max_limit"):
        Settings(default_limit=50, max_limit=10)


def test_production_validation_catches_insecure_defaults() -> None:
    problems = Settings(
        environment="production",
        store_backend="postgres",
        database_url="postgresql+asyncpg://u:p@h/db",
        debug_errors=True,
    ).validate_production()
    joined = " ".join(problems)
    assert "api_key_pepper" in joined
    assert "debug_errors" in joined
    assert "deterministic" in joined
    assert "redis_url" in joined


def test_local_environment_has_no_production_complaints() -> None:
    assert Settings(environment="local").validate_production() == []


# -- errors --------------------------------------------------------------------


def test_problem_document_shape() -> None:
    problem = NotFoundError("missing", field="memory_id").to_problem(instance="/v1/x")
    assert problem["status"] == 404
    assert problem["code"] == "not_found"
    assert problem["field"] == "memory_id"
    assert problem["instance"] == "/v1/x"
    assert problem["type"].startswith("https://")


def test_rate_limited_carries_retry_after() -> None:
    problem = RateLimitedError("slow down", retry_after=2.5).to_problem()
    assert problem["retry_after"] == 2.5


def test_validation_error_status() -> None:
    assert ValidationError().status_code == 422


def test_base_error_defaults_to_500() -> None:
    assert SupermemoryError().status_code == 500
