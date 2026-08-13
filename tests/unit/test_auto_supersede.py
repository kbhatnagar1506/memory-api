"""The ingest-time auto-supersede path, and the floor that guards it.

This path had no test at all, and that is exactly how the defect it now
guards against survived: `propose_supersessions` ranks candidates by
confidence and documents that the caller decides what to do with them, and
the caller applied every one. Seeding a graph from real LongMemEval chat
histories made it visible -- 94 supersessions fired across six people's
histories, only 2 of which had any revision language, median confidence
0.43. Ninety-four true memories flipped to SUPERSEDED, which default
retrieval hides.

The asymmetry is the whole argument for a floor. A false supersession
removes a true memory from every default read with no error raised
anywhere. A missed supersession leaves both memories active and rankable,
where recency still favours the newer one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mapi.config import Settings
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Memory, MemoryStatus, Organization, RelationType, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

BASE = datetime(2024, 3, 1, tzinfo=UTC)


async def _confirming(prompt: str) -> str:
    """An adjudicator that agrees the first candidate was replaced.

    Supersession is no longer decided by similarity. The cosine proposer
    SHORTLISTS and a model confirms, because similarity cannot distinguish
    "the standup moved to 10:15" from "the standup is in the main room" --
    both are about the standup, only one replaces anything. Audited against
    a real corpus, similarity alone hid four true memories out of five.
    """
    # Names the attribute and BOTH values, because that is the real contract:
    # a verdict that cannot quote two different values is rejected in code, so
    # a stub that omits them would be testing a path production never takes.
    return (
        '[{"n": 1, "attribute": "start time", "old_value": "9:30am",'
        ' "new_value": "10:15am", "confidence": 0.95}]'
    )


async def _refusing(prompt: str) -> str:
    """An adjudicator that confirms nothing."""
    return "[]"


async def _service(
    store: InMemoryStore,
    settings: Settings,
    adjudicator: object | None = None,
) -> MemoryService:
    return MemoryService(
        store,
        DeterministicEmbedder(dimensions=128),
        HeuristicReranker(),
        settings,
        extractor=adjudicator,  # type: ignore[arg-type]
    )


def _settings(floor: float) -> Settings:
    return Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=128,
        rerank_backend="heuristic",
        api_key_pepper="test-pepper",
        chunk_target_tokens=64,
        chunk_overlap_tokens=8,
        supersede_min_confidence=floor,
    )


async def _space(store: InMemoryStore) -> tuple[str, str]:
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    return org.id, space.id


#: Near-restatement, one week later. High similarity under any embedder.
OLD = "The team standup is at 9:30am every weekday in the main room."
NEW = "The team standup is now at 10:15am every weekday in the main room."


async def test_high_confidence_supersession_is_applied() -> None:
    store = InMemoryStore()
    org_id, space_id = await _space(store)
    service = await _service(store, _settings(0.0), _confirming)

    first = await service.ingest(
        org_id=org_id, space_id=space_id, content=OLD, occurred_at=BASE
    )
    second = await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content=NEW,
        occurred_at=BASE + timedelta(days=7),
        auto_supersede=True,
    )

    assert first.memory.id in second.superseded
    replaced = await service.get_memory(org_id, space_id, first.memory.id)
    assert replaced.status is MemoryStatus.SUPERSEDED


async def test_floor_declines_low_confidence_and_reports_it() -> None:
    """A floor above the proposal's confidence must decline, not silently drop."""
    store = InMemoryStore()
    org_id, space_id = await _space(store)
    # 1.0 is above the proposer's ceiling of 0.99, so nothing can clear it.
    service = await _service(store, _settings(1.0))

    first = await service.ingest(
        org_id=org_id, space_id=space_id, content=OLD, occurred_at=BASE
    )
    second = await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content=NEW,
        occurred_at=BASE + timedelta(days=7),
        auto_supersede=True,
    )

    assert second.superseded == []
    # Declined, and reported: the caller can see the system considered it.
    assert first.memory.id in second.supersede_declined

    survivor = await service.get_memory(org_id, space_id, first.memory.id)
    assert survivor.status is MemoryStatus.ACTIVE, (
        "a declined proposal must leave the older memory readable; hiding it "
        "is the expensive half of the asymmetry"
    )


async def test_declined_proposal_writes_no_edge() -> None:
    """Declining means no graph claim either, not just no status flip."""
    store = InMemoryStore()
    org_id, space_id = await _space(store)
    service = await _service(store, _settings(1.0))

    await service.ingest(org_id=org_id, space_id=space_id, content=OLD, occurred_at=BASE)
    await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content=NEW,
        occurred_at=BASE + timedelta(days=7),
        auto_supersede=True,
    )

    graph = await service.get_graph(org_id, space_id, limit=50)
    assert [e for e in graph.edges if e.type is RelationType.SUPERSEDES] == []


async def test_declined_memory_stays_retrievable() -> None:
    """The point of the floor: the older memory still answers questions.

    A false supersession is invisible in the graph and fatal in the results,
    so this asserts on the read path rather than on status alone.
    """
    store = InMemoryStore()
    org_id, space_id = await _space(store)
    service = await _service(store, _settings(1.0))

    first = await service.ingest(
        org_id=org_id, space_id=space_id, content=OLD, occurred_at=BASE
    )
    await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content=NEW,
        occurred_at=BASE + timedelta(days=7),
        auto_supersede=True,
    )

    found = await service.search(
        SearchRequest(org_id=org_id, space_id=space_id, query=OLD, limit=10)
    )
    assert first.memory.id in {r.memory.id for r in found.results}


async def test_without_an_adjudicator_nothing_is_ever_hidden() -> None:
    """Fails closed. Similarity alone is not trusted to hide a memory.

    This is the whole safety property: supersession is the only operation
    that removes a memory from default search, so when the model that
    decides is unavailable, the answer is no. A misconfigured or unreachable
    backend costs a missed revision, never a deleted fact.
    """
    store = InMemoryStore()
    org_id, space_id = await _space(store)
    service = await _service(store, _settings(0.0))  # no adjudicator

    first = await service.ingest(
        org_id=org_id, space_id=space_id, content=OLD, occurred_at=BASE
    )
    result = await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content=NEW,
        occurred_at=BASE + timedelta(days=7),
    )
    assert result.superseded == []
    survivor = await service.get_memory(org_id, space_id, first.memory.id)
    assert survivor.status is MemoryStatus.ACTIVE


async def test_the_adjudicator_can_refuse_a_shortlisted_supersession() -> None:
    """Two memories about the same subject, neither replacing the other."""
    store = InMemoryStore()
    org_id, space_id = await _space(store)
    service = await _service(store, _settings(0.0), _refusing)

    first = await service.ingest(
        org_id=org_id, space_id=space_id, content=OLD, occurred_at=BASE
    )
    result = await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content=NEW,
        occurred_at=BASE + timedelta(days=7),
    )
    assert result.superseded == []
    assert first.memory.id in result.supersede_declined
    survivor = await service.get_memory(org_id, space_id, first.memory.id)
    assert survivor.status is MemoryStatus.ACTIVE


@pytest.mark.parametrize("floor", [-0.1, 1.1])
def test_floor_must_be_a_probability(floor: float) -> None:
    with pytest.raises(ValueError):
        _settings(floor)


# -- a document is not a fact ----------------------------------------------
#
# A long memory asserts many things at once, and supersession is all-or-
# nothing: it hides the whole row and cascades `_mark_derivations_stale` over
# everything extracted from it. Measured on a 10-session trace, two whole
# session records were superseded because one number inside them changed
# later, and 8 supersessions removed 49 of 131 memories -- 37% of the corpus.


def _mem(content: str, when: datetime) -> Memory:
    from mapi.core.ids import new_id

    return Memory(
        org_id=new_id("org"),
        space_id=new_id("space"),
        content=content,
        occurred_at=when,
    )


def test_a_long_document_is_never_proposed_for_supersession() -> None:
    from mapi.domain.consolidation import propose_supersessions

    session = _mem(
        "Reviewed spend this morning. Cloud SQL is 340 dollars a month, Redis "
        "is 45, Heroku dynos come to 100. " + ("Discussion of alerting. " * 22),
        BASE,
    )
    assert len(session.content) > 500
    later = _mem(
        "Reviewed spend again. Cloud SQL is 410 dollars a month, and the second "
        "dyno took Heroku to 175. " + ("More discussion of the same. " * 20),
        BASE + timedelta(days=60),
    )
    # Inside the supersession band [0.72, 0.97): above the ceiling a pair is a
    # near-duplicate, which is a different verdict entirely.
    vector, close = [1.0, 0.0, 0.0], [0.9, 0.436, 0.0]

    assert propose_supersessions(later, vector, [(session, close)]) == []


def test_a_short_fact_is_still_eligible() -> None:
    """The rule must aim supersession, not disable it."""
    from mapi.domain.consolidation import propose_supersessions

    old = _mem("I live in Berlin.", BASE)
    new = _mem("I live in Madrid now.", BASE + timedelta(days=60))
    vector, close = [1.0, 0.0, 0.0], [0.9, 0.436, 0.0]

    assert propose_supersessions(new, vector, [(old, close)])
