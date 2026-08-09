"""Temporal scope: the window a question restricts its answer to.

Measured motivation. Of 40 multi-session failures, 8 asked for a count inside
an explicit window — "how many weddings have I attended **this year**", "how
many bikes did I service **in March**" — and nothing enforced the window. The
errors go BOTH ways, which is the signature of a missing filter rather than a
missing fact:

    weddings this year   gold 3  ->  5   (counted outside the window)
    bikes in March       gold 2  ->  3   (counted outside the window)
    plants last month    gold 3  ->  2   (missed some inside it)

So the reduce stage must run over rows inside the window, not over everything
the map stage found. This is the deterministic half of compositional
derivation — filter, then aggregate — and it is deliberately built first: a
date range parsed by code cannot hallucinate a window, and the roadmap's kill
criterion for model-generated plans is to fall back to exactly this.

Every relative scope resolves against the question's OWN reference date, not
against today. "This year" asked in April 2023 means 2023, whatever the clock
says when the benchmark is re-run — which is also what makes it reproducible.
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass(frozen=True, slots=True)
class DateRange:
    """Inclusive window. `label` is what the question actually said."""

    start: date
    end: date
    label: str

    def contains(self, value: date | None) -> bool:
        """Undated rows are NOT in scope.

        A row whose date is unknown cannot be confirmed inside the window, and
        counting it would inflate an aggregate the user asked to bound. This
        under-counts rather than fabricates, the same trade the dedup stage
        makes.
        """
        return value is not None and self.start <= value <= self.end


def _month_window(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _previous_month(reference: date) -> tuple[date, date]:
    first_of_this = reference.replace(day=1)
    last_of_prev = first_of_this - timedelta(days=1)
    return _month_window(last_of_prev.year, last_of_prev.month)


#: Ordered: the most specific pattern must win. "in March 2023" is a month in
#: a stated year, not the bare year 2023 and not the bare month March.
_EXPLICIT_MONTH_YEAR = re.compile(
    r"\bin\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE
)
_EXPLICIT_YEAR = re.compile(r"\bin\s+(\d{4})\b")
_BARE_MONTH = re.compile(r"\b(?:in|during)\s+(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
_LAST_N = re.compile(
    r"\b(?:in|over|during|within)?\s*the\s+(?:last|past)\s+(\d+|a|one)\s+"
    r"(day|week|month|year)s?\b",
    re.IGNORECASE,
)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def extract_scope(question: str, reference: date) -> DateRange | None:
    """The window a question restricts its answer to, or None if unbounded.

    `reference` is the date the question was asked. Relative phrasings are
    meaningless without it — and resolving them against the wall clock instead
    would make the same question produce different answers on different days.
    """
    match = _EXPLICIT_MONTH_YEAR.search(question)
    if match:
        month, year = _MONTHS[match.group(1).lower()], int(match.group(2))
        start, end = _month_window(year, month)
        return DateRange(start, end, match.group(0).strip())

    match = _EXPLICIT_YEAR.search(question)
    if match:
        year = int(match.group(1))
        # Reject anything that is obviously not a year (a price, a count).
        if 1900 <= year <= 2200:
            return DateRange(date(year, 1, 1), date(year, 12, 31), match.group(0).strip())

    match = _LAST_N.search(question)
    if match:
        raw = match.group(1).lower()
        count = 1 if raw in {"a", "one"} else int(raw)
        days = count * _UNIT_DAYS[match.group(2).lower()]
        return DateRange(reference - timedelta(days=days), reference, match.group(0).strip())

    lowered = question.lower()
    if "last month" in lowered:
        start, end = _previous_month(reference)
        return DateRange(start, end, "last month")
    if "this month" in lowered:
        start, end = _month_window(reference.year, reference.month)
        return DateRange(start, min(end, reference), "this month")
    if "last week" in lowered:
        return DateRange(
            reference - timedelta(days=14), reference - timedelta(days=7), "last week"
        )
    if "this week" in lowered:
        return DateRange(reference - timedelta(days=7), reference, "this week")
    if "last year" in lowered:
        year = reference.year - 1
        return DateRange(date(year, 1, 1), date(year, 12, 31), "last year")
    if "this year" in lowered:
        return DateRange(date(reference.year, 1, 1), reference, "this year")

    match = _BARE_MONTH.search(question)
    if match:
        month = _MONTHS[match.group(1).lower()]
        # A bare month means the most recent one that has already happened.
        # "In March", asked in February, is last March -- reading it as a
        # future March would produce an empty window and a confident zero.
        year = reference.year if month <= reference.month else reference.year - 1
        start, end = _month_window(year, month)
        return DateRange(start, min(end, reference), match.group(0).strip())

    return None


__all__ = ["DateRange", "extract_scope"]
