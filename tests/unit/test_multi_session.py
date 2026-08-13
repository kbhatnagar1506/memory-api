"""Multi-session questions: evidence that lives in more than one place.

The measured shape of this problem, from the lme-full run: accuracy falls about
12.3 points for every additional session an answer needs. One evidence session
answers at ~94%, 2.59 sessions at ~74%, and retrieval stays flat at 0.97-1.00
across that whole range. The evidence is THERE; what degrades is assembling it.

So these tests are about the two ways assembly breaks, both of which are
invisible in a `hit@k` number:

  * COVERAGE -- a question needing five facts gets a ranked few, and the ranking
    has no opinion about completeness. A count over four of six sessions is not
    a slightly worse answer, it is a wrong one that looks confident.
  * SCOPE -- a question naming a window aggregates over everything retrieved
    instead of everything in the window. That error goes both ways, which is
    the signature of a missing filter rather than a missing fact.

The single-session case is here too, and it is not the easy one: a lone session
holding the whole answer has to survive competition from many sessions of
near-miss context, and that is the case where a corpus-wide sweep hurts.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.synthesis.scope import extract_scope
from mapi.service import MemoryService


def _when(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, tzinfo=UTC)


#: One fact per session, all answering the same question, deliberately spread
#: over eight months so no window contains all of them. This is the shape the
#: accuracy curve above is about: nothing here is hard to retrieve
#: individually, and the answer requires every row.
GYM_VISITS = [
    ("s01", "Went to the climbing gym on the way home.", _when(2026, 1, 14)),
    ("s02", "Climbing session after the standup, top-roped four routes.", _when(2026, 2, 3)),
    ("s03", "Back at the climbing wall, worked on the overhang.", _when(2026, 3, 19)),
    ("s04", "Climbed with Priya at the new bouldering place.", _when(2026, 4, 8)),
    ("s06", "Climbing gym again, finally sent the pink route.", _when(2026, 6, 21)),
]

#: Topically adjacent sessions that answer nothing. A corpus-wide sweep has to
#: survive these, and `n05` is the one that matters: it mentions the gym and
#: records a NON-event. A count that keys on topic rather than on what happened
#: picks it up and returns six for a corpus containing five, which is the exact
#: over-count shape the scope module measured ("weddings this year", gold 3,
#: answered 5).
NOISE = [
    ("n01", "Bought new climbing shoes online, half a size down.", _when(2026, 1, 20)),
    ("n02", "Read a long article about climbing injuries.", _when(2026, 2, 11)),
    ("n03", "The climbing gym raised its membership price.", _when(2026, 3, 2)),
    ("n04", "Watched a climbing documentary with Aaditi.", _when(2026, 4, 15)),
    ("n05", "Skipped the climbing gym, went for a run instead.", _when(2026, 5, 2)),
    ("n06", "The gym is closed for resurfacing next week.", _when(2026, 5, 9)),
]


@pytest.fixture
async def spread(service: MemoryService, org: Organization, space: Space) -> None:
    """Every memory carries its own session id and event date."""
    for session, content, occurred in GYM_VISITS + NOISE:
        await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content=content,
            occurred_at=occurred,
            metadata={"session_id": session},
        )


def _sessions(response: object) -> set[str]:
    return {
        str(hit.memory.metadata.get("session_id"))
        for hit in response.results  # type: ignore[attr-defined]
    }


# -- coverage across sessions ----------------------------------------------


async def test_a_count_question_reaches_every_session_holding_evidence(
    service: MemoryService, org: Organization, space: Space, spread: None
) -> None:
    """The failure this guards is a confident count over a subset.

    A count is conjunctive: missing one row is a wrong answer, not a slightly
    worse one, and nothing in the returned scores says a row is missing.
    """
    response = await service.search(
        SearchRequest(
            query="How many times did I go climbing?",
            org_id=org.id,
            space_id=space.id,
            limit=10,
        )
    )
    found = _sessions(response)
    evidence = {session for session, _, _ in GYM_VISITS}
    missing = evidence - found
    assert not missing, f"count would be computed over a subset; missing {sorted(missing)}"


async def test_a_comprehensive_question_widens_past_the_ranked_few(
    service: MemoryService, org: Organization, space: Space, spread: None
) -> None:
    """Comprehensiveness is a different request from relevance."""
    response = await service.search(
        SearchRequest(
            query="List all the times I went climbing.",
            org_id=org.id,
            space_id=space.id,
            limit=5,
        )
    )
    assert len(response.results) > 5, "a list-all question must not stop at limit"
    assert _sessions(response) >= {session for session, _, _ in GYM_VISITS}


# -- the window, which is where multi-session counts actually go wrong -----


@pytest.mark.parametrize(
    ("question", "in_window", "outside"),
    [
        ("How many times did I climb in the last 3 months?", {"s04", "s06"}, {"s01", "s02"}),
        (
            "How many climbing sessions between January and March 2026?",
            {"s01", "s02", "s03"},
            {"s04", "s06"},
        ),
        ("How many times did I climb in Feb 2026?", {"s02"}, {"s01", "s03", "s04", "s06"}),
        ("How many climbing sessions since April 2026?", {"s04", "s06"}, {"s01", "s02", "s03"}),
    ],
)
def test_the_window_a_question_names_partitions_the_evidence(
    question: str, in_window: set[str], outside: set[str]
) -> None:
    """Parsed with the question's OWN reference date, not the wall clock.

    Asserted against `contains` rather than against a retrieval result, because
    this is the deterministic half: whatever the ranker returns, the aggregate
    must run over the rows inside the window. An unbounded window here is the
    failure that hides -- the count is computed over all of history and comes
    back looking exactly as confident as a correct one.
    """
    reference = _when(2026, 6, 30).date()
    scope = extract_scope(question, reference)
    assert scope is not None, "a question naming a window must produce one"

    dated = {session: occurred.date() for session, _, occurred in GYM_VISITS}
    inside = {session for session, day in dated.items() if scope.contains(day)}
    assert in_window <= inside, f"{scope.label} dropped rows it should count"
    assert not (outside & inside), f"{scope.label} counted rows outside itself"


def test_an_undated_row_is_not_counted_inside_a_window() -> None:
    """Under-counting beats fabricating: an unknown date cannot be confirmed."""
    scope = extract_scope("How many in March 2026?", _when(2026, 6, 30).date())
    assert scope is not None
    assert not scope.contains(None)


# -- ordering across sessions ----------------------------------------------


async def test_the_earliest_and_latest_are_found_across_sessions(
    service: MemoryService, org: Organization, space: Space, spread: None
) -> None:
    """An ORDER question wants the endpoints, which are rarely the best-ranked.

    The top hit for "when did I first go climbing" is whatever most resembles
    the question. The answer is whichever row is oldest, so both endpoints have
    to be present in what comes back before the ordering can be right.
    """
    response = await service.search(
        SearchRequest(
            query="When did I first go climbing, and when was the most recent time?",
            org_id=org.id,
            space_id=space.id,
            limit=10,
        )
    )
    dated = [
        (hit.memory.occurred_at, hit.memory.metadata.get("session_id"))
        for hit in response.results
        if hit.memory.occurred_at is not None
    ]
    assert dated, "ordering needs event times"
    visits = [(when, session) for when, session in dated if session in evidence_ids()]
    assert visits, "ordering needs the evidence rows, not just any dated row"
    assert min(visits)[1] == "s01", f"earliest visit not retrieved (got {min(visits)[1]})"
    assert max(visits)[1] == "s06", f"latest visit not retrieved (got {max(visits)[1]})"


def evidence_ids() -> set[str]:
    return {session for session, _, _ in GYM_VISITS}


# -- the single-session case, which is not the easy one --------------------


async def test_one_session_holding_the_answer_outranks_adjacent_noise(
    service: MemoryService, org: Organization, space: Space, spread: None
) -> None:
    """Widening for multi-session questions must not drown single-session ones.

    Five of the eleven memories here are about climbing and answer nothing. A
    question whose answer lives in exactly one session has to put that session
    first, or the breadth that helps a count has cost precision on a lookup.
    """
    response = await service.search(
        SearchRequest(
            query="Which route did I finally send?",
            org_id=org.id,
            space_id=space.id,
            limit=5,
        )
    )
    assert response.results, "a lookup with a stored answer must return something"
    assert response.results[0].memory.metadata.get("session_id") == "s06"


async def test_a_lookup_is_not_widened_into_a_sweep(
    service: MemoryService, org: Organization, space: Space, spread: None
) -> None:
    """A single-answer question keeps its ranked shape."""
    response = await service.search(
        SearchRequest(
            query="Who did I go bouldering with?",
            org_id=org.id,
            space_id=space.id,
            limit=5,
        )
    )
    assert len(response.results) <= 5
    assert "Priya" in response.results[0].memory.content
