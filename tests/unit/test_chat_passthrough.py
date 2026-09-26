"""/chat searches the way /search does, and quotes what it matched.

Two gaps between the chat path and the search it is built on:

  * `service.chat` built a bare `SearchRequest`, so the deployment's retrieval
    settings (max_per_source, coverage_limit, route_by_kind, ...) tuned /search
    and silently not the answers built on it, and with no `asked_at` the
    temporal stage never ran for a chat question.
  * A memory longer than the prompt's per-memory cap reached the model as its
    first 4,000 characters, whether or not the passage it was retrieved for
    was among them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from tests.support.factories import memory as build_memory

from mapi.config import Settings
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Organization, ScoredMemory, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.domain.synthesis.chat import (
    EXCERPT_PREFIX,
    MAX_MEMORY_CHARS,
    TRUNCATION_MARKER,
    format_memories,
)
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

ORG, SPACE = "org_01k000000000000000000000", "spc_01k000000000000000000000"


def _hit(content: str, matched_text: str = "") -> ScoredMemory:
    return ScoredMemory(
        memory=build_memory(org_id=ORG, space_id=SPACE, content=content),  # type: ignore[arg-type]
        score=1.0,
        matched_text=matched_text,
    )


async def _service(prompts: list[str], **overrides: Any):
    async def complete(prompt: str) -> str:
        prompts.append(prompt)
        return "From memory [1]."

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
        store,
        DeterministicEmbedder(dimensions=64),
        HeuristicReranker(),
        settings,
        completer=complete,
    )
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    return service, org.id, space.id


# -- the search behind chat ---------------------------------------------------------


async def test_chat_searches_with_the_deployments_settings_and_a_clock() -> None:
    prompts: list[str] = []
    service, org_id, space_id = await _service(
        prompts,
        max_per_source=2,
        coverage_limit=40,
        route_by_kind=True,
        candidate_multiplier=9,
        rerank_candidates=12,
        rrf_k=30,
    )
    await service.ingest(org_id=org_id, space_id=space_id, content="The demo is on Sunday.")

    seen = []
    original = service.search

    async def spy(request):
        seen.append(request)
        return await original(request)

    service.search = spy  # type: ignore[method-assign]
    before = datetime.now(UTC)
    await service.chat(org_id, space_id, message="When is the demo?", k=5)

    (request,) = seen
    assert request.limit == 5
    assert request.max_per_source == 2
    assert request.coverage_limit == 40
    assert request.route_by_kind is True
    assert request.candidate_multiplier == 9
    assert request.rerank_candidates == 12
    assert request.rrf_k == 30
    assert request.asked_at is not None
    assert before - timedelta(seconds=1) <= request.asked_at <= datetime.now(UTC)


async def test_chat_answers_from_the_passage_deep_in_a_long_memory() -> None:
    """End to end: the needle sits past the old 4,000-character clip."""
    prompts: list[str] = []
    service, org_id, space_id = await _service(
        prompts, chunk_target_tokens=128, chunk_overlap_tokens=0
    )
    filler = " ".join(f"Routine log line {i} about nothing in particular." for i in range(120))
    needle = "The locker combination for the hardware room is 4417."
    await service.ingest(org_id=org_id, space_id=space_id, content=f"{filler} {needle}")
    assert len(filler) > MAX_MEMORY_CHARS

    await service.chat(org_id, space_id, message="locker combination hardware room")

    (prompt,) = prompts
    assert "4417" in prompt
    assert EXCERPT_PREFIX in prompt


# -- what of a memory the model sees ------------------------------------------------


def test_a_memory_that_fits_is_shown_whole_even_with_a_match() -> None:
    rendered = format_memories([_hit("the whole short fact", matched_text="short fact")])
    assert rendered.endswith("the whole short fact")
    assert EXCERPT_PREFIX not in rendered


def test_an_over_long_memory_shows_the_passage_that_matched() -> None:
    content = "x" * (MAX_MEMORY_CHARS + 500) + " the answer is 42"
    rendered = format_memories([_hit(content, matched_text="the answer is 42")])
    assert "the answer is 42" in rendered
    assert rendered.endswith(f"{EXCERPT_PREFIX}the answer is 42{TRUNCATION_MARKER}")


def test_an_over_long_match_is_still_capped() -> None:
    content = "y" * (MAX_MEMORY_CHARS * 2)
    rendered = format_memories([_hit(content, matched_text="z" * (MAX_MEMORY_CHARS + 100))])
    assert rendered.count("z") == MAX_MEMORY_CHARS
    assert TRUNCATION_MARKER in rendered


def test_without_a_match_the_opening_is_the_fallback() -> None:
    rendered = format_memories([_hit("w" * (MAX_MEMORY_CHARS + 10))])
    assert EXCERPT_PREFIX not in rendered
    assert rendered.endswith("w" * 10 + TRUNCATION_MARKER)
