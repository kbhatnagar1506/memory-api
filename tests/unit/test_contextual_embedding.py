"""Contextual embedding: richer vector, identical stored text.

The whole safety of this rests on one invariant -- the header reaches the
embedder and nothing else. If it ever reached storage, every search result,
every grounded quote and every dashboard row would carry a synthetic prefix
the caller never wrote.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mapi.config import Settings
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.embeddings.context import MAX_HEADER_CHARS, build_header, for_embedding
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

WHEN = datetime(2023, 5, 20, 14, 30, tzinfo=UTC)


def test_a_header_carries_date_source_and_topic() -> None:
    header = build_header(
        occurred_at=WHEN, source="user", tags=("fishing",), metadata={"title": "Lake trip"}
    )
    assert header == "2023-05-20 · Lake trip · user · fishing"


def test_nothing_to_say_means_no_header() -> None:
    """A corpus with no dates or sources is left exactly as it was."""
    assert build_header() == ""
    assert for_embedding("plain text", "") == "plain text"


def test_the_date_is_a_date_not_a_timestamp() -> None:
    """Seconds are precision no question asks for, inside the chunk's budget."""
    assert "14:30" not in build_header(occurred_at=WHEN)


def test_a_header_cannot_crowd_out_the_chunk() -> None:
    header = build_header(
        occurred_at=WHEN, source="s" * 400, metadata={"title": "t" * 400}
    )
    assert len(header) <= MAX_HEADER_CHARS


def test_the_header_is_marked_as_framing_not_prose() -> None:
    out = for_embedding("yeah, three of them", "2023-05-20 · user")
    assert out.startswith("<context>\n")
    assert out.endswith("\nyeah, three of them")


def test_duplicate_parts_appear_once() -> None:
    header = build_header(source="fishing", tags=("fishing",), metadata={"title": "fishing"})
    assert header == "fishing"


async def _service(**overrides):
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=64,
        rerank_backend="heuristic",
        api_key_pepper="x" * 32,
        **overrides,
    )
    store = InMemoryStore()
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=64), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    return service, store, org.id, space.id


async def test_the_stored_text_never_carries_the_header() -> None:
    """THE invariant. A leak here corrupts every downstream surface."""
    service, _store, org_id, space_id = await _service()
    result = await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content="yeah, three of them",
        source="user",
        tags=["fishing"],
        occurred_at=WHEN,
    )
    stored = await service.get_memory(org_id, space_id, result.memory.id)
    assert stored.content == "yeah, three of them"
    for chunk in stored.chunks:
        assert "<context>" not in chunk.text
        assert "2023-05-20" not in chunk.text


async def test_the_header_does_reach_the_embedder() -> None:
    """Otherwise the feature is a no-op wearing a config flag."""
    service, _, org_id, space_id = await _service()
    seen: list[str] = []
    original = service.embedder.embed

    async def spy(texts):
        seen.extend(texts)
        return await original(texts)

    service.embedder.embed = spy  # type: ignore[method-assign]
    await service.ingest(
        org_id=org_id, space_id=space_id, content="yeah, three of them",
        source="user", occurred_at=WHEN,
    )
    assert any("<context>" in t and "2023-05-20" in t for t in seen)


async def test_turning_it_off_restores_the_previous_behaviour_exactly() -> None:
    service, _, org_id, space_id = await _service(contextual_embedding=False)
    seen: list[str] = []
    original = service.embedder.embed

    async def spy(texts):
        seen.extend(texts)
        return await original(texts)

    service.embedder.embed = spy  # type: ignore[method-assign]
    await service.ingest(
        org_id=org_id, space_id=space_id, content="yeah, three of them",
        source="user", occurred_at=WHEN,
    )
    assert seen == ["yeah, three of them"]


async def test_it_adds_no_retrieval_units() -> None:
    """The property that separates this from the extraction arms.

    Extraction turned one document into ~13 units and cost 21-54 questions to
    crowding. This changes vectors only -- the chunk count is untouched.
    """
    with_ctx, _, org_a, space_a = await _service()
    without, _, org_b, space_b = await _service(contextual_embedding=False)
    passage = "I caught three bass at the lake on Saturday morning with the new rod."

    a = await with_ctx.ingest(
        org_id=org_a, space_id=space_a, content=passage, source="user", occurred_at=WHEN
    )
    b = await without.ingest(
        org_id=org_b, space_id=space_b, content=passage, source="user", occurred_at=WHEN
    )
    assert a.chunk_count == b.chunk_count
