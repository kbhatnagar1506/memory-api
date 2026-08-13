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

#: A date parser has to know month names, so this is a lexicon and not a
#: heuristic word list: there is no structural property of a string that makes
#: it March. What a lexicon CAN be is incomplete, and this one was -- it held
#: only the full names, so "in Jan 2026", "in Sept" and "in Dec 2025" produced
#: no window at all. That is the failure this module's docstring calls the one
#: that hides: the aggregate is then computed over all of history and comes
#: back looking perfectly confident.
_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

#: Abbreviations, generated rather than typed out. Three letters is the common
#: form, plus the four-letter "sept" and the full name. Deriving them keeps the
#: two spellings of a month from drifting apart, which is the usual way a
#: hand-maintained second list goes wrong.
_MONTHS: dict[str, int] = {}
for _index, _name in enumerate(_MONTH_NAMES, start=1):
    _MONTHS[_name] = _index
    _MONTHS[_name[:3]] = _index
_MONTHS["sept"] = 9

#: Longest-first, so "march" is not matched as "mar" with a trailing "ch" that
#: then fails the word boundary, and "june" is not shadowed by "jun".
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))


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
_EXPLICIT_MONTH_YEAR = re.compile(r"\bin\s+(" + _MONTH_ALT + r")\s+(\d{4})\b", re.IGNORECASE)
_EXPLICIT_YEAR = re.compile(r"\bin\s+(\d{4})\b")

#: A NAMED DAY, which asks for a one-day window and used to get none. Both
#: orders, because "15 March 2026" and "March 15, 2026" are the same request.
#: No leading preposition is required: a day, a month and a year together are
#: unambiguous without one, unlike a bare month.
_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_ALT + r")\.?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTH_DAY_YEAR = re.compile(
    r"\b(" + _MONTH_ALT + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)

#: NUMERIC DATES. Checked before `_EXPLICIT_YEAR`, which is the bug they also
#: fix: `\bin\s+(\d{4})\b` matched the year inside "in 2026-03" and returned
#: the whole of 2026 for a question that named one month -- a window twelve
#: times too wide, reported with a label ("in 2026") that looks deliberate.
#:
#: Day-first and month-first numeric forms (03/04/2026) are deliberately NOT
#: parsed. They are genuinely ambiguous between conventions, and a silently
#: wrong window is worse here than no window: an unbounded aggregate is at
#: least obviously unbounded.
_ISO_DAY = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{1,2})\b")
_SLASH_MONTH = re.compile(r"\b(\d{1,2})/(\d{4})\b")


def _valid_day(year: int, month: int, day: int) -> bool:
    return 1 <= month <= 12 and 1 <= day <= monthrange(year, month)[1]


_BARE_MONTH = re.compile(r"\b(?:in|during)\s+(" + _MONTH_ALT + r")\b", re.IGNORECASE)
_LAST_N = re.compile(
    r"\b(?:in|over|during|within)?\s*the\s+(?:last|past)\s+(\d+|a|an|one)?\s*"
    r"(day|week|fortnight|month|quarter|year|decade)s?\b",
    re.IGNORECASE,
)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

#: Everything below widens the same idea: a question can name its window in
#: more ways than "in March 2023" and "the last 6 months". Each pattern here
#: was a question shape that previously produced an UNBOUNDED window, which
#: is the failure that hides: the answer is computed over all of history and
#: looks perfectly confident.

#: "since January", "since 2021", "since the move" -- an open-ended window
#: running to today. Only the datable forms are handled; "since the move"
#: needs an event lookup and correctly falls through to unbounded.
_SINCE = re.compile(
    r"\bsince\s+(?:(" + _MONTH_ALT + r")(?:\s+(\d{4}))?|(\d{4}))\b",
    re.IGNORECASE,
)
#: "before 2022", "after March 2021", "up to 2020", "until June".
_BEFORE_AFTER = re.compile(
    r"\b(before|after|prior to|up to|until|through)\s+"
    r"(?:(" + _MONTH_ALT + r")\s+)?(\d{4})\b",
    re.IGNORECASE,
)
#: "between March and June", "from January to April", with an optional year
#: on either side.
_BETWEEN = re.compile(
    r"\b(?:between|from)\s+(" + _MONTH_ALT + r")(?:\s+(\d{4}))?"
    r"\s+(?:and|to|through|-|until)\s+(" + _MONTH_ALT + r")(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
#: Calendar quarters, which is how anything with a finance or planning
#: vocabulary names a window: "in Q3", "Q4 2023", "this quarter".
_QUARTER = re.compile(r"\bq([1-4])(?:\s+(\d{4}))?\b", re.IGNORECASE)
#: Northern-hemisphere seasons. Approximate by construction -- a season is a
#: three-month band, and a question saying "last summer" is not asking for
#: astronomical precision.
_SEASONS = {
    "winter": (12, 2),
    "spring": (3, 5),
    "summer": (6, 8),
    "fall": (9, 11),
    "autumn": (9, 11),
}
_SEASON = re.compile(r"\b(last|this|past)\s+(" + "|".join(_SEASONS) + r")\b", re.IGNORECASE)
#: "in the last quarter", "over the past fortnight", and the two units the
#: numeric pattern above cannot express.
_UNIT_DAYS_EXTRA = {"quarter": 91, "fortnight": 14, "decade": 3650}


def _quarter_window(year: int, quarter: int) -> tuple[date, date]:
    first_month = (quarter - 1) * 3 + 1
    last_month = first_month + 2
    return (
        date(year, first_month, 1),
        date(year, last_month, monthrange(year, last_month)[1]),
    )


def extract_scope(question: str, reference: date) -> DateRange | None:
    """The window a question restricts its answer to, or None if unbounded.

    `reference` is the date the question was asked. Relative phrasings are
    meaningless without it — and resolving them against the wall clock instead
    would make the same question produce different answers on different days.
    """
    # A single named or ISO day is the narrowest window there is, so it is
    # tried before anything that would widen it to the containing month.
    for pattern, order in ((_DAY_MONTH_YEAR, "dmy"), (_MONTH_DAY_YEAR, "mdy")):
        match = pattern.search(question)
        if match:
            raw_day, raw_month = (
                (match.group(1), match.group(2))
                if order == "dmy"
                else (match.group(2), match.group(1))
            )
            year, month, day = int(match.group(3)), _MONTHS[raw_month.lower()], int(raw_day)
            if _valid_day(year, month, day):
                point = date(year, month, day)
                return DateRange(point, point, match.group(0).strip())

    match = _ISO_DAY.search(question)
    if match:
        year, month, day = (int(g) for g in match.groups())
        if _valid_day(year, month, day):
            point = date(year, month, day)
            return DateRange(point, point, match.group(0))

    match = _EXPLICIT_MONTH_YEAR.search(question)
    if match:
        month, year = _MONTHS[match.group(1).lower()], int(match.group(2))
        start, end = _month_window(year, month)
        return DateRange(start, end, match.group(0).strip())

    # Numeric month-in-year, before the bare-year pattern that would otherwise
    # swallow the year half of it and return twelve times the window asked for.
    for pattern, flipped in ((_ISO_MONTH, False), (_SLASH_MONTH, True)):
        match = pattern.search(question)
        if match:
            first, second = int(match.group(1)), int(match.group(2))
            year, month = (second, first) if flipped else (first, second)
            if 1 <= month <= 12 and 1900 <= year <= 2200:
                start, end = _month_window(year, month)
                return DateRange(start, end, match.group(0))

    match = _EXPLICIT_YEAR.search(question)
    if match:
        year = int(match.group(1))
        # Reject anything that is obviously not a year (a price, a count).
        if 1900 <= year <= 2200:
            return DateRange(date(year, 1, 1), date(year, 12, 31), match.group(0).strip())

    match = _LAST_N.search(question)
    if match:
        raw = (match.group(1) or "one").lower()
        count = 1 if raw in {"a", "an", "one"} else int(raw)
        unit = match.group(2).lower()
        days = count * (_UNIT_DAYS.get(unit) or _UNIT_DAYS_EXTRA[unit])
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

    match = _BETWEEN.search(question)
    if match:
        start_month = _MONTHS[match.group(1).lower()]
        end_month = _MONTHS[match.group(3).lower()]
        # A year stated on EITHER side governs both unless both state one:
        # "from January to April 2023" is 2023 throughout, and defaulting the
        # start to the current year produced a range running backwards.
        start_year = int(match.group(2)) if match.group(2) else 0
        end_year = int(match.group(4)) if match.group(4) else 0
        if not start_year and not end_year:
            start_year = end_year = reference.year
        else:
            start_year = start_year or end_year
            end_year = end_year or start_year
        # "between November and February" crosses a new year when no years
        # are stated. Reading it as a backwards range would produce an empty
        # window and a confident zero.
        if end_year == start_year and end_month < start_month:
            end_year += 1
        start = _month_window(start_year, start_month)[0]
        end = _month_window(end_year, end_month)[1]
        return DateRange(start, end, match.group(0).strip())

    match = _SINCE.search(question)
    if match:
        month_name, month_year, bare_year = match.groups()
        if bare_year:
            start = date(int(bare_year), 1, 1)
        else:
            month = _MONTHS[month_name.lower()]
            year = int(month_year) if month_year else reference.year
            if not month_year and month > reference.month:
                # "since November", asked in March, means last November.
                year -= 1
            start = _month_window(year, month)[0]
        if start <= reference:
            return DateRange(start, reference, match.group(0).strip())

    match = _BEFORE_AFTER.search(question)
    if match:
        direction, month_name, year_text = match.groups()
        year = int(year_text)
        month = _MONTHS[month_name.lower()] if month_name else 0
        if direction.lower() == "after":
            start = _month_window(year, month)[0] if month else date(year, 1, 1)
            return DateRange(start, reference, match.group(0).strip())
        boundary = _month_window(year, month)[1] if month else date(year, 12, 31)
        # An open START is unrepresentable, so anchor it far enough back that
        # nothing real falls outside. `date.min` would be honest but makes
        # every downstream span calculation absurd.
        return DateRange(date(1900, 1, 1), boundary, match.group(0).strip())

    match = _QUARTER.search(question)
    if match:
        quarter = int(match.group(1))
        year = int(match.group(2)) if match.group(2) else reference.year
        start, end = _quarter_window(year, quarter)
        if start <= reference:
            return DateRange(start, min(end, reference), match.group(0).strip())

    match = _SEASON.search(question)
    if match:
        which, season = match.group(1).lower(), match.group(2).lower()
        first_month, last_month = _SEASONS[season]
        year = reference.year
        if which in {"last", "past"} or first_month > reference.month:
            year -= 1
        if first_month > last_month:
            # Winter straddles the new year.
            start = date(year, first_month, 1)
            end = date(year + 1, last_month, monthrange(year + 1, last_month)[1])
        else:
            start = date(year, first_month, 1)
            end = date(year, last_month, monthrange(year, last_month)[1])
        return DateRange(start, min(end, reference), match.group(0).strip())

    if "this quarter" in lowered:
        quarter = (reference.month - 1) // 3 + 1
        start, end = _quarter_window(reference.year, quarter)
        return DateRange(start, min(end, reference), "this quarter")
    if "last quarter" in lowered:
        quarter = (reference.month - 1) // 3 + 1
        year = reference.year if quarter > 1 else reference.year - 1
        start, end = _quarter_window(year, quarter - 1 if quarter > 1 else 4)
        return DateRange(start, end, "last quarter")

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
