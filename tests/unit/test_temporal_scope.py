"""Date-scoped retrieval.

`extract_scope` has existed for a while and was called in exactly one place:
inside derive.py, AFTER retrieval had already chosen what to read. So a
question naming a month searched the whole corpus and applied the window to
whatever survived -- backwards for the one capability our retrieval actually
caps. Measured on LongMemEval, temporal-reasoning retrieves at
`full_recall@k` 0.910 against 0.970-1.000 everywhere else.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from mapi.core.ids import new_id
from mapi.domain.models import Memory, ScoredMemory
from mapi.domain.retrieval.temporal import IN_WINDOW, NEAR_WINDOW, apply_scope
from mapi.domain.synthesis.scope import DateRange

MARCH = DateRange(date(2023, 3, 1), date(2023, 3, 31), "in March 2023")


def scored(when: datetime | None, score: float = 1.0) -> ScoredMemory:
    memory = Memory(
        org_id=new_id("org"),
        space_id=new_id("space"),
        content="a memory",
        occurred_at=when or datetime(2020, 1, 1, tzinfo=UTC),
    )
    return ScoredMemory(memory=memory, score=score)


def test_no_scope_changes_nothing() -> None:
    """Inert for every question that does not name a window."""
    items = [scored(datetime(2023, 3, 15, tzinfo=UTC), 0.5)]
    assert apply_scope(items, None)[0].score == 0.5


def test_in_window_memories_are_raised() -> None:
    inside = scored(datetime(2023, 3, 15, tzinfo=UTC), 0.5)
    apply_scope([inside], MARCH)
    assert inside.score == 0.5 * IN_WINDOW


def test_a_memory_just_outside_keeps_part_of_the_boost() -> None:
    """The boundary is the asker's approximation, not a fact.

    "In March" is often satisfied by something logged on 2 April.
    """
    near = scored(datetime(2023, 4, 5, tzinfo=UTC), 0.5)
    apply_scope([near], MARCH)
    assert near.score == 0.5 * NEAR_WINDOW


def test_a_distant_memory_is_untouched() -> None:
    far = scored(datetime(2023, 9, 1, tzinfo=UTC), 0.5)
    apply_scope([far], MARCH)
    assert far.score == 0.5


def test_nothing_is_ever_removed() -> None:
    """A bias, not a filter.

    Event time is caller-supplied and may be absent or approximate. Filtering
    turns a ranking problem into missing data, and makes a question whose
    answer sits outside the window it named unanswerable.
    """
    items = [
        scored(datetime(2023, 3, 15, tzinfo=UTC)),
        scored(datetime(2019, 1, 1, tzinfo=UTC)),
        scored(datetime(2024, 12, 31, tzinfo=UTC)),
    ]
    assert len(apply_scope(items, MARCH)) == 3


def test_the_window_reorders() -> None:
    """A slightly-less-relevant in-window memory should overtake."""
    out_of_window = scored(datetime(2023, 9, 1, tzinfo=UTC), 0.60)
    in_window = scored(datetime(2023, 3, 15, tzinfo=UTC), 0.55)
    out_of_window.memory = out_of_window.memory.model_copy(update={"id": "mem_" + "a" * 26})
    in_window.memory = in_window.memory.model_copy(update={"id": "mem_" + "b" * 26})

    ordered = apply_scope([out_of_window, in_window], MARCH)
    assert ordered[0] is in_window, "0.55 x 1.25 = 0.6875 beats 0.60"


def test_a_date_cannot_beat_relevance_outright() -> None:
    """The boost is modest on purpose.

    A bigger multiplier here would be the deep-boost-stack mistake this
    codebase has documented once already -- where a 2.0x let any boosted
    candidate in the first 60 ranks outrank an unboosted rank-1 result.
    """
    strong_out = scored(datetime(2023, 9, 1, tzinfo=UTC), 1.0)
    weak_in = scored(datetime(2023, 3, 15, tzinfo=UTC), 0.5)
    ordered = apply_scope([strong_out, weak_in], MARCH)
    assert ordered[0] is strong_out


def test_a_memory_with_no_event_time_is_skipped_not_crashed() -> None:
    items = [scored(None, 0.5)]
    apply_scope(items, MARCH)
    assert items[0].score == 0.5


def test_the_reason_is_recorded_in_explain() -> None:
    inside = scored(datetime(2023, 3, 15, tzinfo=UTC), 0.5)
    apply_scope([inside], MARCH)
    assert any("in March 2023" in note for note in inside.explain)
