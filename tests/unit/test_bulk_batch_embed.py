"""Bulk ingest: one embedding pass for the whole request, per-item outcomes.

Before this, `POST .../memories/bulk` ingested items one after another, and
each item made its own embedding call: a 100-item sync was ~100 sequential
round trips (18 s or more at the measured latency), and the first item to
fail threw away the response for all the others that had already landed.

The contract pinned here:

  * the embedding call count is ceil(total chunks / batch size), however the
    chunks fall into items;
  * batches run at most `embedding_concurrency` at a time;
  * items that need no vectors -- an exact restatement, an unchanged keyed
    payload -- cost no embedding at all;
  * writes still happen in request order, so item 7 sees item 6;
  * one bad item fails alone, with its own status and problem document.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from mapi.config import Settings
from mapi.core.errors import PayloadTooLargeError, ProviderError, ValidationError
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import MemoryStatus, Organization, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import IngestItem, MemoryService
from mapi.store.base import MemoryFilter
from mapi.store.memory import InMemoryStore

DIMS = 64


class Gauge(DeterministicEmbedder):
    """Counts provider calls and the texts in each, and the peak in flight."""

    def __init__(self, *, batch_size: int, fail_calls: frozenset[int] = frozenset()) -> None:
        super().__init__(dimensions=DIMS, batch_size=batch_size)
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.in_flight = 0
        self.peak = 0
        self.fail_calls = fail_calls

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        call = self.calls
        self.batch_sizes.append(len(texts))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            if call in self.fail_calls:
                raise ProviderError("scripted quota wall", extra={"retry_after": 12.0})
            return await super()._embed_batch(texts)
        finally:
            self.in_flight -= 1


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "store_backend": "memory",
        "embedding_backend": "deterministic",
        "embedding_dimensions": DIMS,
        "rerank_backend": "none",
        "api_key_pepper": "test-pepper",
        "chunk_target_tokens": 64,
        "chunk_overlap_tokens": 8,
        "embedding_cache_size": 0,
        "max_content_bytes": 20_000,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def _service(
    *, batch_size: int = 4, concurrency: int = 2, fail_calls: frozenset[int] = frozenset()
) -> tuple[MemoryService, Gauge, Organization, Space]:
    store = InMemoryStore()
    org = await store.create_organization(Organization(name="Bulk"))
    space = await store.create_space(Space(org_id=org.id, slug="bulk", name="Bulk"))
    embedder = Gauge(batch_size=batch_size, fail_calls=fail_calls)
    service = MemoryService(
        store,
        embedder,
        HeuristicReranker(),
        _settings(embedding_concurrency=concurrency, embedding_batch_size=batch_size),
    )
    return service, embedder, org, space


def _facts(n: int, *, prefix: str = "fact") -> list[IngestItem]:
    # Distinct vocabularies, so no item is a near-duplicate of another.
    topics = ["robotics", "pottery", "sailing", "compilers", "birding", "chess", "baking"]
    return [
        IngestItem(
            content=f"{prefix} {i}: works on {topics[i % len(topics)]} project {i * 7919}"
        )
        for i in range(n)
    ]


async def test_embed_calls_are_ceil_chunks_over_batch_size() -> None:
    service, embedder, org, space = await _service(batch_size=4)
    outcomes = await service.ingest_many(org_id=org.id, space_id=space.id, items=_facts(10))
    assert all(o.error is None and o.result and o.result.created for o in outcomes)
    chunks = sum(o.result.chunk_count for o in outcomes if o.result)
    assert chunks == 10
    assert embedder.calls == math.ceil(chunks / 4) == 3
    assert embedder.batch_sizes == [4, 4, 2]


async def test_chunks_are_packed_across_item_boundaries() -> None:
    """Multi-chunk items do not each round up to their own batch."""
    service, embedder, org, space = await _service(batch_size=5)
    long_text = " ".join(f"sentence {n} about distributed consensus." for n in range(60))
    items = [
        IngestItem(content=f"{long_text} variant {v}", tags=[f"v{v}"]) for v in range(3)
    ] + _facts(4)
    outcomes = await service.ingest_many(org_id=org.id, space_id=space.id, items=items)
    assert all(o.error is None for o in outcomes)
    chunks = sum(o.result.chunk_count for o in outcomes if o.result)
    assert chunks > len(items), "the long items should have split into several chunks"
    assert embedder.calls == math.ceil(chunks / 5)


async def test_a_hundred_items_at_batch_fifty_is_two_calls() -> None:
    """The live smoke's arithmetic, offline: 100 one-chunk items, batch 50."""
    service, embedder, org, space = await _service(batch_size=50)
    outcomes = await service.ingest_many(org_id=org.id, space_id=space.id, items=_facts(100))
    assert sum(1 for o in outcomes if o.result and o.result.created) == 100
    assert embedder.calls == 2


async def test_batches_in_flight_never_exceed_the_concurrency_setting() -> None:
    service, embedder, org, space = await _service(batch_size=2, concurrency=2)
    await service.ingest_many(org_id=org.id, space_id=space.id, items=_facts(12))
    assert embedder.calls == 6
    assert embedder.peak == 2

    serial, serial_embedder, org2, space2 = await _service(batch_size=2, concurrency=1)
    await serial.ingest_many(org_id=org2.id, space_id=space2.id, items=_facts(6))
    assert serial_embedder.peak == 1


async def test_what_needs_no_vectors_costs_no_embedding() -> None:
    service, embedder, org, space = await _service(batch_size=10)
    await service.ingest(org_id=org.id, space_id=space.id, content="already stored text")
    await service.ingest(
        org_id=org.id, space_id=space.id, content="stuck on: CUDA", key="card:stuck"
    )
    before = embedder.calls
    outcomes = await service.ingest_many(
        org_id=org.id,
        space_id=space.id,
        items=[
            IngestItem(content="already stored text"),
            IngestItem(content="stuck on: CUDA", key="card:stuck"),
        ],
    )
    assert embedder.calls == before, "neither item needed a vector"
    assert [o.result.created for o in outcomes if o.result] == [False, False]
    assert all(o.result and o.result.duplicate_of for o in outcomes)


async def test_duplicates_inside_one_request_fold_into_the_first() -> None:
    service, embedder, org, space = await _service(batch_size=10)
    outcomes = await service.ingest_many(
        org_id=org.id,
        space_id=space.id,
        items=[
            IngestItem(content="the demo is at 3pm"),
            IngestItem(content="The demo is at 3pm", tags=["later"]),
        ],
    )
    first, second = (o.result for o in outcomes)
    assert first is not None and second is not None
    assert first.created and not second.created
    assert second.duplicate_of == first.memory.id
    assert "later" in second.memory.tags, "the restatement's tags merge into the original"
    assert embedder.calls == 1


async def test_writes_happen_in_request_order() -> None:
    """Item 2 replaces item 1 under the same key; it could only if item 1
    had already been written when item 2 was."""
    service, _embedder, org, space = await _service(batch_size=10)
    outcomes = await service.ingest_many(
        org_id=org.id,
        space_id=space.id,
        items=[
            IngestItem(content="room: Klaus 1116", key="fm:room"),
            IngestItem(content="room: CULC 144", key="fm:room"),
            IngestItem(content="room: CULC 144", key="fm:room"),
        ],
    )
    first, second, third = (o.result for o in outcomes)
    assert first is not None and second is not None and third is not None
    assert second.superseded == [first.memory.id]
    assert not third.created and third.duplicate_of == second.memory.id
    active = await service.store.get_active_by_keys(org.id, space.id, ["fm:room"])
    assert active["fm:room"].content == "room: CULC 144"
    old = await service.store.get_memory(org.id, space.id, first.memory.id)
    assert old is not None and old.status is MemoryStatus.SUPERSEDED


async def test_a_stale_stage_one_snapshot_never_resurrects_a_row() -> None:
    """Stage 1 found an ACTIVE row to fold into; by write time something had
    superseded it. Folding into the snapshot would write it back ACTIVE --
    the A -> B -> A bug again, inside one request."""
    service, embedder, org, space = await _service(batch_size=10)
    original = await service.ingest(org_id=org.id, space_id=space.id, content="lives in Austin")
    store = service.store
    lookup = store.find_by_content_hashes

    async def then_superseded(*args: object, **kwargs: object) -> dict[str, object]:
        found = await lookup(*args, **kwargs)  # type: ignore[arg-type]
        row = await store.get_memory(org.id, space.id, original.memory.id)
        assert row is not None
        await store.upsert_memory(
            row.model_copy(
                update={"status": MemoryStatus.SUPERSEDED, "version": row.version + 1}
            )
        )
        return found  # type: ignore[return-value]

    store.find_by_content_hashes = then_superseded  # type: ignore[method-assign]
    before = embedder.calls
    outcomes = await service.ingest_many(
        org_id=org.id, space_id=space.id, items=[IngestItem(content="lives in Austin")]
    )
    result = outcomes[0].result
    assert result is not None and result.created, "a new ACTIVE row, not a fold"
    assert embedder.calls == before + 1, "it had to embed after all"
    old = await store.get_memory(org.id, space.id, original.memory.id)
    assert old is not None and old.status is MemoryStatus.SUPERSEDED


async def test_one_bad_item_fails_alone() -> None:
    service, _embedder, org, space = await _service(batch_size=10)
    outcomes = await service.ingest_many(
        org_id=org.id,
        space_id=space.id,
        items=[
            IngestItem(content="fine item one"),
            IngestItem(content="   "),
            IngestItem(content="x" * 30_000),
            IngestItem(content="fine item two", key="tab\there"),
            IngestItem(content="fine item three"),
        ],
    )
    assert [o.index for o in outcomes] == [0, 1, 2, 3, 4]
    assert outcomes[0].result and outcomes[0].result.created
    assert isinstance(outcomes[1].error, ValidationError)
    assert isinstance(outcomes[2].error, PayloadTooLargeError)
    assert isinstance(outcomes[3].error, ValidationError)
    assert outcomes[4].result and outcomes[4].result.created


async def test_a_failed_batch_fails_only_the_items_in_it() -> None:
    service, embedder, org, space = await _service(
        batch_size=2, concurrency=1, fail_calls=frozenset({2})
    )
    outcomes = await service.ingest_many(org_id=org.id, space_id=space.id, items=_facts(5))
    assert embedder.calls == 3
    landed = [o.index for o in outcomes if o.result and o.result.created]
    failed = [o.index for o in outcomes if o.error is not None]
    assert landed == [0, 1, 4] and failed == [2, 3]
    assert all(isinstance(outcomes[i].error, ProviderError) for i in failed)
    page = await service.store.list_memories(org.id, space.id, filters=MemoryFilter(), limit=10)
    assert len(page.items) == 3, "the failed items wrote nothing"


# -- HTTP -----------------------------------------------------------------------------------


async def test_http_bulk_reports_each_item(client, space_id) -> None:
    url = f"/v1/spaces/{space_id}/memories/bulk"
    response = await client.post(
        url,
        json={
            "items": [
                {"content": "bulk alpha fact"},
                {"content": "bulk alpha fact"},
                {"content": "   x   ", "key": "k:1"},
            ]
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["created"] == 2 and body["duplicates"] == 1 and body["failed"] == 0
    items = body["items"]
    assert [i["index"] for i in items] == [0, 1, 2]
    assert [i["status"] for i in items] == [201, 200, 201]
    assert items[1]["duplicate_of"] == items[0]["memory"]["id"]
    assert items[2]["memory"]["key"] == "k:1"
    assert all(i["error"] is None for i in items)
    timing = response.headers["server-timing"].split(", ")
    assert [part.split(";")[0] for part in timing] == ["db", "embed", "cpu", "total"]


async def test_http_bulk_carries_per_item_errors(client, space_id, app_context) -> None:
    service = app_context["app"].state.service
    service.settings = service.settings.model_copy(update={"max_content_bytes": 100})
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/bulk",
        json={"items": [{"content": "small and fine"}, {"content": "y" * 500}]},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["created"] == 1 and body["failed"] == 1
    bad = body["items"][1]
    assert bad["status"] == 413 and bad["memory"] is None
    assert bad["error"]["code"] == "payload_too_large"
    assert bad["error"]["field"] == "content"


async def test_http_bulk_with_nothing_created_is_200(client, space_id) -> None:
    url = f"/v1/spaces/{space_id}/memories/bulk"
    await client.post(url, json={"items": [{"content": "said once"}]})
    again = await client.post(url, json={"items": [{"content": "said once"}]})
    assert again.status_code == 200
    assert again.json()["created"] == 0 and again.json()["duplicates"] == 1


async def test_http_bulk_that_only_hit_a_provider_wall_is_a_502(
    client, space_id, app_context
) -> None:
    """Every item failed upstream: the request fails as a whole, so a
    client's backoff sees the provider trouble -- with its retry hint."""
    service = app_context["app"].state.service
    wall = Gauge(batch_size=32, fail_calls=frozenset(range(1, 100)))
    service.embedder = wall
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/bulk",
        json={"items": [{"content": "first thing"}, {"content": "second thing"}]},
    )
    assert response.status_code == 502
    assert response.json()["code"] == "provider_error"
    assert response.json()["retry_after"] == 12.0


async def test_http_bulk_item_errors_carry_the_retry_hint(
    client, space_id, app_context
) -> None:
    service = app_context["app"].state.service
    service.embedder = Gauge(batch_size=1, fail_calls=frozenset({2}))
    service.settings = service.settings.model_copy(update={"embedding_concurrency": 1})
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/bulk",
        json={"items": [{"content": "lands fine"}, {"content": "hits the wall"}]},
    )
    assert response.status_code == 201
    failed = response.json()["items"][1]
    assert failed["status"] == 502
    assert failed["error"]["retry_after"] == 12.0
    # 5xx detail stays server-side unless debug_errors is on.
    assert failed["error"]["detail"] == failed["error"]["title"]


@pytest.mark.parametrize("n", [101])
async def test_http_bulk_still_caps_the_request(client, space_id, n: int) -> None:
    response = await client.post(
        f"/v1/spaces/{space_id}/memories/bulk",
        json={"items": [{"content": f"item {i}"} for i in range(n)]},
    )
    assert response.status_code == 422
