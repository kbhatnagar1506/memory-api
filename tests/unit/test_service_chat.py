"""`service.chat`: the orchestration around the grounded answer.

`test_chat.py` covers the synthesis -- formatting, citations, the decline. This
covers the service method that wraps it, which had no test and one behaviour
worth being careful about: `remember=True` writes the user's MESSAGE back as a
memory, and a question stored as a memory is a claim nobody made.

    "is Krishna on an F-1 visa?"  ->  stored  ->  retrieved later as evidence

That is the failure mode the flag defaults off for, and it is the reason this
file exists separately from the synthesis tests.

The other property worth pinning: chat uses the ORDINARY search pipeline rather
than a special path, so it inherits question classification, coverage,
supersession suppression and conflict surfacing. A future refactor that gave chat
its own retrieval would silently drop all four, and the tests below would not
notice unless they assert it -- so one of them does.
"""

from __future__ import annotations

import pytest

from mapi.core.errors import NotFoundError, ProviderError
from mapi.domain.models import Organization, Space
from mapi.domain.synthesis.chat import Turn
from mapi.service import MemoryService

FACTS = [
    "Our primary database is Postgres 16 running on Cloud SQL in us-central1.",
    "Redis 7 backs the rate limiter across every process.",
    "The office lease renews in June.",
]


@pytest.fixture
async def stocked(service: MemoryService, org: Organization, space: Space) -> None:
    for fact in FACTS:
        await service.ingest(org_id=org.id, space_id=space.id, content=fact, extract=False)


def _replying(text: str = "Postgres 16 [1]"):
    async def complete(prompt: str) -> str:
        return text

    return complete


# -- the guard rails -------------------------------------------------------


async def test_chat_without_a_synthesis_backend_is_a_502(
    service: MemoryService, org: Organization, space: Space
) -> None:
    """`ProviderError`, not a 500 and not a cheerful empty answer.

    There is no degraded chat: answering without the model would mean inventing
    the answer, which is the one thing the module forbids.
    """
    service.completer = None
    with pytest.raises(ProviderError) as caught:
        await service.chat(org.id, space.id, message="what database?")
    assert caught.value.status_code == 502


async def test_an_unknown_space_is_a_404_not_an_empty_answer(
    service: MemoryService, org: Organization
) -> None:
    """Checked BEFORE the model call, so a typo'd space id does not cost a
    completion. And 404 rather than 403, so another org's space id is
    indistinguishable from one that does not exist."""
    service.completer = _replying()
    with pytest.raises(NotFoundError):
        await service.chat(org.id, "spc_01k000000000000000000000", message="hello")


# -- remember, which is the dangerous flag ---------------------------------


async def test_remember_is_off_by_default(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """A chat that silently records every question turns a question into a fact.

    "is Krishna on an F-1 visa?" stored as a memory is a claim nobody made, and
    it will come back later as evidence.
    """
    from mapi.store.base import MemoryFilter

    service.completer = _replying()
    before = await service.store.count_memories(org.id, space.id, filters=MemoryFilter())
    await service.chat(org.id, space.id, message="is the lease renewing in June?")
    after = await service.store.count_memories(org.id, space.id, filters=MemoryFilter())
    assert after == before


async def test_remember_true_stores_the_users_message(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """Opt-in, and when opted into it stores the message verbatim -- not the
    answer, which would record the model's output as a remembered fact."""
    from mapi.store.base import MemoryFilter

    service.completer = _replying()
    await service.chat(
        org.id, space.id, message="I moved the standup to 10:15am.", remember=True
    )
    listing = await service.list_memories(
        org.id, space.id, filters=MemoryFilter(), limit=50, cursor=None
    )
    contents = [m.content for m in listing.items]
    assert "I moved the standup to 10:15am." in contents
    assert "Postgres 16 [1]" not in contents, "the model's reply must not be stored"


# -- what chat inherits ----------------------------------------------------


async def test_chat_returns_the_search_response_it_used(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """Both halves come back, so a caller can see the evidence AND the answer.

    Without the response a caller cannot tell a refusal caused by empty retrieval
    from one caused by the model declining -- two different problems with the same
    reply text.
    """
    service.completer = _replying()
    answer, response = await service.chat(org.id, space.id, message="what database?")
    assert answer.considered == [hit.memory.id for hit in response.results]


async def test_chat_uses_the_ordinary_pipeline_and_inherits_its_signals(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """Not a special retrieval path.

    Chat inherits question classification, coverage, supersession suppression and
    conflict surfacing because it calls `self.search`. A refactor that gave chat
    its own retrieval would drop all four silently, so the presence of the
    pipeline's own reporting is asserted here.
    """
    service.completer = _replying()
    _answer, response = await service.chat(org.id, space.id, message="what database?")
    assert response.intent is not None, "no question classification -- not the pipeline"
    assert response.confidence is not None
    assert response.timings_ms


async def test_the_k_argument_bounds_the_evidence(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    service.completer = _replying()
    _answer, response = await service.chat(org.id, space.id, message="what database?", k=1)
    assert len(response.results) <= 1


async def test_history_is_passed_through_to_the_prompt(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    seen: dict[str, str] = {}

    async def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return "yes [1]"

    service.completer = complete
    await service.chat(
        org.id,
        space.id,
        message="and the cache?",
        history=[Turn(role="user", content="what database?")],
    )
    assert "what database?" in seen["prompt"]


async def test_an_empty_space_declines_without_calling_the_model(
    service: MemoryService, org: Organization, space: Space
) -> None:
    """Retrieval returned nothing, so there is nothing to ground an answer in.

    The model is never called -- which also means an empty space costs no tokens,
    and the reply is the honest one rather than whatever a model would produce
    from an empty context.
    """
    called = False

    async def complete(prompt: str) -> str:
        nonlocal called
        called = True
        return "I know all about it"

    service.completer = complete
    answer, response = await service.chat(org.id, space.id, message="anything?")
    assert not called
    assert response.results == []
    assert answer.cited == []
    assert "don't have" in answer.reply
