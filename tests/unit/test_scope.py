"""Temporal scope: filter, then aggregate.

Every case here is drawn from a real multi-session failure. The errors ran
BOTH directions — "weddings this year" gold 3 answered 5, "bikes in March"
gold 2 answered 3, "plants last month" gold 3 answered 2 — which is the
signature of a missing window rather than a missing fact.
"""

from __future__ import annotations

from datetime import date

import pytest

from supermemory.domain.synthesis.scope import extract_scope

REF = date(2023, 4, 15)  # a Saturday, mid-month, mid-year


def test_this_year_runs_from_january_to_the_question_date() -> None:
    """ "How many weddings have I attended this year" must not count last
    year's, and must not count into the future either."""
    scope = extract_scope("How many weddings have I attended in this year?", REF)
    assert scope is not None
    assert scope.start == date(2023, 1, 1)
    assert scope.end == REF


def test_last_year_is_the_whole_previous_calendar_year() -> None:
    scope = extract_scope("How many trips did I take last year?", REF)
    assert scope is not None
    assert (scope.start, scope.end) == (date(2022, 1, 1), date(2022, 12, 31))


def test_last_month_is_the_previous_calendar_month() -> None:
    scope = extract_scope("How many plants did I acquire in the last month?", REF)
    assert scope is not None
    assert scope.start <= date(2023, 3, 20) <= scope.end


def test_bare_month_resolves_to_the_most_recent_one_already_past() -> None:
    """ "In March" asked in April is this year's March."""
    scope = extract_scope("How many bikes did I service in March?", REF)
    assert scope is not None
    assert (scope.start, scope.end) == (date(2023, 3, 1), date(2023, 3, 31))


def test_a_month_still_ahead_this_year_resolves_to_last_year() -> None:
    """ "In March", asked in February, is LAST March. Reading it as a future
    March gives an empty window and a confident zero."""
    scope = extract_scope("How many bikes did I service in March?", date(2023, 2, 10))
    assert scope is not None
    assert scope.start.year == 2022


def test_explicit_month_and_year_beats_the_bare_forms() -> None:
    scope = extract_scope("What did I buy in March 2021?", REF)
    assert scope is not None
    assert (scope.start, scope.end) == (date(2021, 3, 1), date(2021, 3, 31))


def test_explicit_year() -> None:
    scope = extract_scope("How many races did I run in 2022?", REF)
    assert scope is not None
    assert (scope.start, scope.end) == (date(2022, 1, 1), date(2022, 12, 31))


def test_last_n_units() -> None:
    scope = extract_scope("How many books in the last 3 months?", REF)
    assert scope is not None
    assert (REF - scope.start).days == 90


def test_a_price_is_not_mistaken_for_a_year() -> None:
    """ "in 800" would otherwise parse as the year 800 and empty the window."""
    assert extract_scope("What did I buy in 800 dollars?", REF) is None


def test_unbounded_questions_have_no_scope() -> None:
    for question in (
        "How many bikes do I own?",
        "Where does my sister live?",
        "What degree did I graduate with?",
    ):
        assert extract_scope(question, REF) is None


@pytest.mark.parametrize(
    "value,inside",
    [(date(2023, 3, 15), True), (date(2023, 4, 1), False), (None, False)],
)
def test_undated_rows_are_never_in_scope(value: date | None, inside: bool) -> None:
    """A row whose date is unknown cannot be confirmed inside the window;
    counting it would inflate an aggregate the user asked to bound."""
    scope = extract_scope("How many bikes did I service in March?", REF)
    assert scope is not None
    assert scope.contains(value) is inside
