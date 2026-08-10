"""Per-tenant quotas.

Rate limiting caps how fast a tenant calls. These cap how much they
accumulate, which is the cost nobody notices until the invoice arrives.
"""

from __future__ import annotations

import pytest

from mapi.config import Settings
from mapi.core.quota import (
    Quota,
    QuotaExceededError,
    check_bytes,
    check_memories,
    check_writes,
)
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

# -- the pure checks ------------------------------------------------------


def test_nothing_configured_means_nothing_enforced() -> None:
    """The shipped default must cost zero, not merely allow everything."""
    assert not Quota().enforced


@pytest.mark.parametrize(
    "quota",
    [
        Quota(max_memories_per_org=1),
        Quota(max_bytes_per_org=1),
        Quota(max_writes_per_day=1),
    ],
)
def test_any_single_limit_turns_enforcement_on(quota: Quota) -> None:
    assert quota.enforced


def test_the_memory_limit_admits_up_to_the_line() -> None:
    quota = Quota(max_memories_per_org=10)
    check_memories(quota, 9)  # the tenth write is allowed
    with pytest.raises(QuotaExceededError):
        check_memories(quota, 10)


def test_the_byte_limit_counts_the_incoming_write() -> None:
    """A limit checked against stored bytes alone always overshoots by one."""
    quota = Quota(max_bytes_per_org=1000)
    check_bytes(quota, 900, 100)
    with pytest.raises(QuotaExceededError):
        check_bytes(quota, 900, 101)


def test_the_write_limit_is_per_day() -> None:
    quota = Quota(max_writes_per_day=5)
    check_writes(quota, 4)
    with pytest.raises(QuotaExceededError):
        check_writes(quota, 5)


def test_unlimited_never_raises_however_large() -> None:
    check_memories(Quota(), 10_000_000)
    check_bytes(Quota(), 10**12, 10**12)
    check_writes(Quota(), 10_000_000)


def test_the_error_names_which_limit_was_hit() -> None:
    """402, not 429: retrying will not help, and a client must be able to tell."""
    with pytest.raises(QuotaExceededError) as caught:
        check_memories(Quota(max_memories_per_org=1), 1)
    error = caught.value
    assert error.status_code == 402
    assert error.limit_name == "max_memories_per_org"
    assert error.extra["limit"] == 1
    assert error.extra["current"] == 1


# -- enforcement on the real write path -----------------------------------


async def _service(**limits: int) -> tuple[MemoryService, Organization, Space]:
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=32,
        rerank_backend="heuristic",
        api_key_pepper="x" * 32,
        **limits,
    )
    store = InMemoryStore()
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=32), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="Quota Test"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="Space"))
    return service, org, space


async def test_ingest_refuses_past_the_memory_limit() -> None:
    service, org, space = await _service(max_memories_per_org=3)
    for i in range(3):
        await service.ingest(org_id=org.id, space_id=space.id, content=f"fact {i}")

    with pytest.raises(QuotaExceededError) as caught:
        await service.ingest(org_id=org.id, space_id=space.id, content="one too many")
    assert caught.value.limit_name == "max_memories_per_org"


async def test_ingest_refuses_past_the_byte_limit() -> None:
    service, org, space = await _service(max_bytes_per_org=200)
    await service.ingest(org_id=org.id, space_id=space.id, content="a" * 150)

    with pytest.raises(QuotaExceededError) as caught:
        await service.ingest(org_id=org.id, space_id=space.id, content="b" * 100)
    assert caught.value.limit_name == "max_bytes_per_org"


async def test_ingest_refuses_past_the_daily_write_limit() -> None:
    service, org, space = await _service(max_writes_per_day=2)
    await service.ingest(org_id=org.id, space_id=space.id, content="first")
    await service.ingest(org_id=org.id, space_id=space.id, content="second")

    with pytest.raises(QuotaExceededError) as caught:
        await service.ingest(org_id=org.id, space_id=space.id, content="third")
    assert caught.value.limit_name == "max_writes_per_day"


async def test_the_limit_spans_spaces_within_one_org() -> None:
    """A per-space limit is escaped by creating another space."""
    service, org, space = await _service(max_memories_per_org=2)
    other = await service.store.create_space(
        Space(org_id=org.id, slug="other", name="Other")
    )
    await service.ingest(org_id=org.id, space_id=space.id, content="first")
    await service.ingest(org_id=org.id, space_id=other.id, content="second")

    with pytest.raises(QuotaExceededError):
        await service.ingest(org_id=org.id, space_id=other.id, content="third")


async def test_one_tenant_hitting_its_limit_does_not_affect_another() -> None:
    service, org, space = await _service(max_memories_per_org=1)
    await service.ingest(org_id=org.id, space_id=space.id, content="theirs")

    other_org = await service.store.create_organization(Organization(name="Other"))
    other_space = await service.store.create_space(
        Space(org_id=other_org.id, slug="s", name="Space")
    )
    result = await service.ingest(
        org_id=other_org.id, space_id=other_space.id, content="mine"
    )
    assert result.created


async def test_unlimited_writes_are_never_counted() -> None:
    """Enforcement off must mean no usage query at all, not a permissive one."""
    service, org, space = await _service()
    called = False
    original = service.store.tenant_usage

    async def spy(*args: object, **kw: object):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return await original(*args, **kw)  # type: ignore[arg-type]

    service.store.tenant_usage = spy  # type: ignore[method-assign]
    await service.ingest(org_id=org.id, space_id=space.id, content="anything")
    assert not called


async def test_the_quota_is_checked_before_the_embedder_is_called() -> None:
    """Embedding is where the money is; a refused write must not spend it."""
    service, org, space = await _service(max_memories_per_org=1)
    await service.ingest(org_id=org.id, space_id=space.id, content="first")

    calls = 0
    original_embed = service.embedder.embed

    async def counting(texts):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return await original_embed(texts)

    service.embedder.embed = counting  # type: ignore[method-assign]
    with pytest.raises(QuotaExceededError):
        await service.ingest(org_id=org.id, space_id=space.id, content="second")
    assert calls == 0
