"""Date arithmetic: the half of a subtraction that is not in the table.

`temporal-reasoning` is the one capability where retrieval is the ceiling --
`full_recall@k` 0.910 against 0.970-1.000 everywhere else. That framing sent the
first fix to retrieval: bias candidates whose event time falls inside the window
the question named. Measured against the actual questions afterwards, that was
aimed at the wrong thing. Only 7 of the 133 temporal questions name a window at
all. What they ask instead is:

    How many weeks ago did I attend the friends and family sale?     gold 4
    How many months have passed since I last visited a museum?       gold 5
    How many days passed between my visit to MoMA and the concert?    gold 7 days
    Which event happened first, the wedding or the engagement party?

Two defects, both of which produce a wrong answer from correct retrieval and
correct dates -- the most expensive kind, because nothing upstream looks broken:

  * 25 of 133 measure from an event TO THE PRESENT ("ago", "since", "have
    passed"). One endpoint of that subtraction is the day the question was
    asked. The reducer required two dates in the TABLE and returned None for
    every one of them, and the compose prompt did not state the reference date
    either, so the model it fell through to could not compute them from what it
    was given.
  * The reducer returned "N days" for every subtraction, whatever unit the
    question asked for. "How many weeks ago" answered "28 days" when the gold
    answer is "4".

Both are arithmetic, so both are tested here rather than measured on a paid run.
"""

from __future__ import annotations

from datetime import date

import pytest

from mapi.domain.synthesis.classify import QuestionKind, classify
from mapi.domain.synthesis.derive import Extraction, _date_span

#: The day the question was asked. Every expectation below is relative to it,
#: never to the wall clock -- a test that drifts with the calendar is a test
#: that will fail on a day nobody is looking.
ASKED = date(2023, 6, 15)


def _row(when: date, fact: str = "attended the sale") -> Extraction:
    return Extraction(date=when, fact=fact, quote=fact, source_id="doc")


# -- measuring to the present ----------------------------------------------


@pytest.mark.parametrize(
    ("question", "event", "expected"),
    [
        ("How many weeks ago did I attend the sale?", date(2023, 5, 18), "4 weeks"),
        ("How many days ago did I attend the service?", date(2023, 6, 11), "4 days"),
        ("How many months have passed since I visited?", date(2023, 1, 20), "4 months"),
        ("How many years ago did I move here?", date(2019, 6, 1), "4 years"),
        ("How long ago did I start the course?", date(2023, 6, 1), "14 days"),
    ],
)
def test_an_event_is_measured_against_the_day_the_question_was_asked(
    question: str, event: date, expected: str
) -> None:
    assert _date_span(question, [_row(event)], ASKED) == expected


def test_without_a_reference_date_there_is_nothing_to_measure_to() -> None:
    """Silence, not a guess. The other endpoint genuinely does not exist."""
    assert _date_span("How many weeks ago was it?", [_row(date(2023, 5, 18))], None) is None


def test_an_event_dated_after_the_question_is_refused() -> None:
    """A negative age means the extraction or the reference date is wrong."""
    assert (
        _date_span("How many days ago did it happen?", [_row(date(2023, 7, 1))], ASKED) is None
    )


def test_a_to_present_question_over_several_dates_defers_to_the_model() -> None:
    """The row being asked about is not identifiable from dates alone.

    The old code took max-minus-min here and answered confidently. The model
    can read the facts and pick the right row; arithmetic over the wrong pair
    cannot be recovered from downstream.
    """
    rows = [_row(date(2023, 5, 18)), _row(date(2023, 4, 2), "a different outing")]
    assert _date_span("How many weeks ago did I attend the sale?", rows, ASKED) is None


# -- measuring between two events ------------------------------------------


@pytest.mark.parametrize(
    ("question", "first", "second", "expected"),
    [
        ("How many days passed between X and Y?", date(2023, 6, 1), date(2023, 6, 8), "7 days"),
        ("How long between the two events?", date(2023, 5, 1), date(2023, 5, 15), "14 days"),
        ("How many weeks passed between them?", date(2023, 4, 1), date(2023, 5, 6), "5 weeks"),
        ("How many months between the two?", date(2023, 1, 1), date(2023, 4, 1), "3 months"),
    ],
)
def test_two_dated_rows_are_subtracted_in_the_unit_asked_for(
    question: str, first: date, second: date, expected: str
) -> None:
    assert _date_span(question, [_row(first), _row(second)], ASKED) == expected


def test_one_date_and_no_to_present_wording_computes_nothing() -> None:
    """ "Between" needs two endpoints and the table holds one."""
    assert _date_span("How many days between them?", [_row(date(2023, 6, 1))], ASKED) is None


def test_an_undated_table_computes_nothing() -> None:
    rows = [Extraction(date=None, fact="something happened", quote="q", source_id="d")]
    assert _date_span("How many days ago?", rows, ASKED) is None


# -- the unit, which is what turns a right answer into a wrong one ---------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many days ago?", "28 days"),
        ("How many weeks ago?", "4 weeks"),
        ("How much time in days has passed since then?", "28 days"),
        # No unit named anywhere: days is the reasonable default.
        ("How long ago did that happen?", "28 days"),
    ],
)
def test_the_same_span_is_reported_in_the_unit_the_question_names(
    question: str, expected: str
) -> None:
    assert _date_span(question, [_row(date(2023, 5, 18))], ASKED) == expected


def test_an_unrecognised_unit_falls_back_to_days_rather_than_guessing() -> None:
    """ "How many visits ago" is not a time unit and must not become one."""
    assert (
        _date_span("How many visits ago was it?", [_row(date(2023, 5, 18))], ASKED) == "28 days"
    )


# -- the router has to send these here in the first place ------------------


@pytest.mark.parametrize(
    "question",
    [
        "How many weeks ago did I attend the friends and family sale at Nordstrom?",
        "How many months have passed since I last visited a museum with a friend?",
        "How many days passed between my visit to MoMA and the concert?",
        "How long ago did I start watering the herb garden?",
        "How many days ago did I attend the Maundy Thursday service?",
    ],
)
def test_real_temporal_question_shapes_route_to_date_arithmetic(question: str) -> None:
    """Phrasings taken from the failing capability, not invented for the test."""
    assert classify(question) is QuestionKind.DATE_ARITH
