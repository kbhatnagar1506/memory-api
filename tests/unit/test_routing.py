"""Kind routing: episodes for what happened, claims for what is true.

The split these tests encode was measured, not assumed. Turning on write-time
extraction moved six LongMemEval capabilities, and the sign matched the kind
of memory each capability needs, six for six -- every semantic one held or
improved, every episodic one lost.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mapi.domain.models import Memory, MemoryKind, ScoredMemory
from mapi.domain.retrieval.routing import allocate, apply_allocation
from mapi.domain.synthesis.classify import QuestionKind

WHEN = datetime(2023, 5, 1, tzinfo=UTC)


def _m(kind: MemoryKind, score: float) -> ScoredMemory:
    return ScoredMemory(
        memory=Memory(
            org_id="org_1",
            space_id="spc_1",
            content=f"{kind.value} {score}",
            kind=kind,
            occurred_at=WHEN,
        ),
        score=score,
    )


def kinds(results: list[ScoredMemory]) -> list[str]:
    return [s.memory.kind.value for s in results]


# -- allocation --------------------------------------------------------------


def test_ordering_questions_reserve_the_window_for_episodes() -> None:
    """The order IS the information, and a claim strips the narrative.

    Real LongMemEval phrasing. Note the classifier needs the explicit
    ordering cue: "what is the order of the museums I visited" WITHOUT
    "from earliest to latest" falls through to DIRECT, which is a known
    gap in `classify`, not in routing.
    """
    a = allocate("What is the order of the six museums I visited from earliest to latest?", 10)
    assert a.episodic == 7
    assert a.derived == 3


def test_date_arithmetic_reserves_for_episodes() -> None:
    a = allocate("How many days passed between the two appointments?", 10)
    assert a.episodic > a.derived


def test_advice_reserves_for_claims() -> None:
    """Advice uses remembered preferences and never needs a timeline;
    preference was the one capability that improved with episodes removed."""
    a = allocate("Can you recommend a restaurant I would like?", 10)
    assert a.derived > a.episodic


def test_neither_kind_is_ever_starved() -> None:
    """A shape is a guess. Starving one kind turns a wrong guess into an
    unanswerable question rather than a worse-ordered one."""
    for question in (
        "What is the order of the six museums I visited?",
        "Can you recommend a restaurant I would like?",
    ):
        a = allocate(question, 10)
        assert a.episodic >= 2
        assert a.derived >= 2


def test_allocation_always_sums_to_the_limit() -> None:
    for limit in (2, 3, 5, 10, 20, 50):
        a = allocate("What is the order of my trips from earliest to latest?", limit)
        assert a.episodic + a.derived == limit


def test_a_single_slot_goes_to_the_preferred_kind() -> None:
    assert allocate("Which did I do first, the gala or the run?", 1).episodic == 1
    assert allocate("Can you suggest something I would enjoy?", 1).derived == 1


def test_an_unclassified_question_splits_evenly() -> None:
    """The classifier returns DIRECT when unsure, so its failure mode is
    'no routing', never 'wrong routing'."""
    a = allocate("", 10, kind=QuestionKind.DIRECT)
    assert a.episodic == 5
    assert a.derived == 5


# -- application -------------------------------------------------------------


def test_claims_cannot_crowd_out_every_episode() -> None:
    """The regression this exists to fix: claims outscore episodes, so a
    purely score-ordered window can hold no episodes at all."""
    scored = [_m(MemoryKind.DERIVED, 1.0 - i / 100) for i in range(9)]
    scored += [_m(MemoryKind.EPISODIC, 0.2), _m(MemoryKind.EPISODIC, 0.1)]

    routed = apply_allocation(
        scored, allocate("Which did I visit first, the museum or the gallery?", 10), 10
    )

    assert len(routed) == 10
    assert kinds(routed).count("episodic") == 2, "every episode available was kept"


def test_unused_slots_are_backfilled() -> None:
    """A space holding only claims must still return a full window for an
    ordering question -- allocation, not a quota."""
    scored = [_m(MemoryKind.DERIVED, 1.0 - i / 100) for i in range(10)]
    routed = apply_allocation(
        scored, allocate("Which did I visit first, the museum or the gallery?", 10), 10
    )
    assert len(routed) == 10
    assert set(kinds(routed)) == {"derived"}


def test_results_stay_in_score_order() -> None:
    """Routing decides membership, not ranking."""
    scored = [_m(MemoryKind.DERIVED, 0.9), _m(MemoryKind.EPISODIC, 0.8)]
    scored += [_m(MemoryKind.DERIVED, 0.7), _m(MemoryKind.EPISODIC, 0.6)]
    routed = apply_allocation(scored, allocate("Which came first, A or B?", 4), 4)
    assert [s.score for s in routed] == sorted((s.score for s in routed), reverse=True)


def test_routing_is_recorded_in_explain() -> None:
    scored = [_m(MemoryKind.EPISODIC, 0.9), _m(MemoryKind.DERIVED, 0.8)]
    routed = apply_allocation(scored, allocate("Which came first, A or B?", 2), 2)
    assert all(any("routing:" in note for note in s.explain) for s in routed)


def test_never_returns_more_than_the_limit() -> None:
    scored = [_m(MemoryKind.DERIVED, 1.0 - i / 100) for i in range(20)]
    scored += [_m(MemoryKind.EPISODIC, 0.5 - i / 100) for i in range(20)]
    assert len(apply_allocation(scored, allocate("Which came first, A or B?", 10), 10)) == 10
