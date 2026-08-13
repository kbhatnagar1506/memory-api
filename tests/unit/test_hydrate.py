"""Context hydration for the answer path.

The measured problem: retrieval delivers complete evidence for 97.2% of
LongMemEval questions and accuracy is 0.8255, so roughly 17.5% are answered
wrong with the evidence already in the context. A memory is one chat turn --
"yeah, three of them" retrieves correctly and cannot be answered from.

The property that makes this safe to ship is that it changes only what is
ASSEMBLED, never what is retrieved. These tests hold that line.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mapi.domain.synthesis.derive import SourceDoc
from mapi.domain.synthesis.hydrate import Neighbourhood, assemble

BASE = datetime(2026, 3, 1, tzinfo=UTC)


def doc(n: int, text: str = "", chars: int = 0) -> SourceDoc:
    body = text or ("x" * chars if chars else f"turn {n}")
    return SourceDoc(id=f"mem_{n}", text=body, occurred_at=BASE + timedelta(minutes=n))


def test_an_anchor_with_no_neighbours_passes_through() -> None:
    out = assemble([Neighbourhood(anchor=doc(5))])
    assert [d.id for d in out] == ["mem_5"]


def test_neighbours_are_included_around_the_anchor() -> None:
    hood = Neighbourhood(anchor=doc(5), before=(doc(3), doc(4)), after=(doc(6),))
    assert [d.id for d in assemble([hood])] == ["mem_3", "mem_4", "mem_5", "mem_6"]


def test_output_is_chronological_not_ranked() -> None:
    """An elision resolves backwards.

    Presenting the reply before the question it answers puts the work back on
    the model, which is the work this exists to remove.
    """
    hoods = [
        Neighbourhood(anchor=doc(9), before=(doc(8),)),
        Neighbourhood(anchor=doc(2), before=(doc(1),)),
    ]
    assert [d.id for d in assemble(hoods)] == ["mem_1", "mem_2", "mem_8", "mem_9"]


def test_a_memory_shared_by_two_anchors_appears_once() -> None:
    hoods = [
        Neighbourhood(anchor=doc(4), after=(doc(5),)),
        Neighbourhood(anchor=doc(5), before=(doc(4),)),
    ]
    assert [d.id for d in assemble(hoods)] == ["mem_4", "mem_5"]


def test_every_anchor_survives_however_tight_the_budget() -> None:
    """Hydration must never lose the thing retrieval found.

    A budget too small for the anchors alone degrades to exactly today's
    behaviour rather than dropping a retrieved memory to make room for
    somebody else's context.
    """
    hoods = [
        Neighbourhood(anchor=doc(n, chars=500), before=(doc(n + 100, chars=500),))
        for n in (1, 2, 3)
    ]
    out = assemble(hoods, budget_chars=10)
    assert {d.id for d in out} == {"mem_1", "mem_2", "mem_3"}


def test_a_tight_budget_spreads_context_rather_than_stacking_it() -> None:
    """One turn everywhere beats four turns on the first result."""
    # `before` runs oldest -> newest, so its LAST entry is the turn
    # immediately preceding the anchor. Anchors sit at 100, 200, 300.
    hoods = [
        Neighbourhood(
            anchor=doc(n, chars=100),
            before=(doc(n - 2, chars=100), doc(n - 1, chars=100)),
        )
        for n in (100, 200, 300)
    ]
    # Room for the three anchors plus three neighbours, not six.
    out = assemble(hoods, budget_chars=600)
    ids = {d.id for d in out}
    assert {"mem_100", "mem_200", "mem_300"} <= ids
    # The immediately-preceding turn is admitted for every anchor before any
    # anchor gets its second.
    assert {"mem_99", "mem_199", "mem_299"} <= ids
    assert not ids & {"mem_98", "mem_198", "mem_298"}


def test_an_empty_result_set_hydrates_to_nothing() -> None:
    assert assemble([]) == []


def test_the_anchor_count_is_never_reduced() -> None:
    """The invariant retrieval metrics depend on."""
    hoods = [Neighbourhood(anchor=doc(n)) for n in range(12)]
    out = assemble(hoods, budget_chars=1)
    assert len(out) == 12
