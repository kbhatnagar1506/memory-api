"""Let a question's own date window bias what gets retrieved.

`scope.extract_scope` turns "in March 2023", "since January", "over the past
six months" into a concrete date range. It has existed for a while and is
called in exactly one place: inside `derive.py`, AFTER retrieval has already
chosen what to read. So a question that names a month searches the whole
corpus, and the window is applied to whatever survived.

That is backwards for the one capability our retrieval actually caps.
Measured on LongMemEval: `temporal-reasoning` retrieves at `full_recall@k`
0.910 against 0.970-1.000 everywhere else -- 9% of those questions are
missing evidence before the model reads a word, so no amount of better
reading can fix them. It is the only capability where the ceiling is
retrieval.

WHY A BIAS AND NOT A FILTER. A hard filter is the obvious move and it is
dangerous: event time is what the caller supplied, a memory can be recorded
without one, and "in March" can be the asker's approximation of something
logged on 2 April. Filtering makes every one of those unanswerable and turns
a ranking problem into missing data. Boosting in-window candidates keeps them
all reachable and moves the right ones up.

It also stays OFF unless the question actually names a window -- no scope, no
change, so this is inert for every question that does not ask for one.
"""

from __future__ import annotations

from datetime import date, datetime

from ..models import ScoredMemory
from ..synthesis.scope import DateRange

#: Multiplier for a memory whose event time falls inside the asked-for window.
#:
#: Modest on purpose. The signal is real but not decisive: a question about
#: March is usually answered by a March memory, and sometimes by one from
#: April that describes March. 1.25 reorders at the margin without letting a
#: date beat relevance outright -- and a bigger number here would be the
#: multiplicative-boost mistake this codebase has already documented once.
IN_WINDOW = 1.25

#: Memories just outside the window keep a small share of the boost, because
#: the boundary is the asker's approximation rather than a fact. A question
#: about "March" is often satisfied by something on 2 April.
NEAR_WINDOW = 1.10
NEAR_DAYS = 14


def _as_date(value: datetime | date | None) -> date | None:
    if value is None:
        return None
    return value.date() if isinstance(value, datetime) else value


def apply_scope(
    scored: list[ScoredMemory],
    scope: DateRange | None,
) -> list[ScoredMemory]:
    """Raise in-window memories, then re-sort. No scope means no change.

    Nothing is removed. Every candidate stays in the list at its own score,
    so a question whose answer sits outside the window it named is still
    answerable -- which is the difference between a bias and a filter, and
    the reason this is safe to run on every question.
    """
    if scope is None or not scored:
        return scored

    for item in scored:
        occurred = _as_date(item.memory.occurred_at)
        if occurred is None:
            continue
        if scope.start <= occurred <= scope.end:
            item.score *= IN_WINDOW
            item.explain.append(f"within {scope.label} x{IN_WINDOW}")
            continue
        gap = min(abs((occurred - scope.start).days), abs((occurred - scope.end).days))
        if gap <= NEAR_DAYS:
            item.score *= NEAR_WINDOW
            item.explain.append(f"{gap}d outside {scope.label} x{NEAR_WINDOW}")

    scored.sort(key=lambda s: (-s.score, s.memory.id))
    return scored


__all__ = ["IN_WINDOW", "NEAR_DAYS", "NEAR_WINDOW", "apply_scope"]
