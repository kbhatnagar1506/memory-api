"""Question classification: which answers must be computed, not recalled.

Regex-first and deterministic. These exact patterns partitioned the 264
real failures-with-complete-evidence from the lme-full run — 45% COUNT, 12%
ORDER, 6% DATE-ARITH, 3% COMPARE — so they are validated against the failure
distribution they exist to fix, not invented. An LLM fallback can widen
coverage later; a misclassification is cheap either way, because every derive
kind fails open to the direct path.

Order matters: COUNT before COMPARE ("how many more..." is a count), and
DATE_ARITH before ORDER ("how long after the first..." is arithmetic).
"""

from __future__ import annotations

import re
from enum import StrEnum


class QuestionKind(StrEnum):
    #: Answer is (probably) literally present in one place. Fast path.
    DIRECT = "direct"
    #: "How many / how often / total" — the answer is a cardinality that
    #: exists in no session. len() over a grounded table.
    COUNT = "count"
    #: "First / last / latest / most recent" — max/min over dated rows.
    ORDER = "order"
    #: "How long / how many days between" — timedelta, or an explicit
    #: duration already stated in some episode.
    DATE_ARITH = "date_arith"
    #: "Which is cheaper / closer / more" — needs judgment over the table,
    #: so this one kind still composes through the model.
    COMPARE = "compare"
    #: "List all / what are the" — the table itself is the answer.
    LIST_ALL = "list_all"


_RULES: tuple[tuple[QuestionKind, re.Pattern[str]], ...] = (
    (
        QuestionKind.DATE_ARITH,
        re.compile(
            r"how (long|much time)|how many (days|weeks|months|years)"
            r"|days (between|since|until)|how old",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionKind.COUNT,
        re.compile(
            r"how many|how often|how much (money|did i spend)"
            r"|(the )?(total|combined) (number|amount|cost|spend)"
            r"|number of times",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionKind.ORDER,
        re.compile(
            r"\b(first|last|latest|earliest|most recent|newest|oldest)\b"
            r"|before or after|which came (first|last)",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionKind.LIST_ALL,
        re.compile(
            r"\blist (all|the|every)|what are (all|the .+s i)|which ones\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionKind.COMPARE,
        re.compile(
            r"\b(more|less|fewer|cheaper|closer|bigger|smaller|longer|shorter)"
            r" than\b|\b(cheapest|closest|most expensive|difference between)\b",
            re.IGNORECASE,
        ),
    ),
)


def classify(question: str) -> QuestionKind:
    """Route a question to the machinery that can actually answer it."""
    for kind, pattern in _RULES:
        if pattern.search(question):
            return kind
    return QuestionKind.DIRECT


__all__ = ["QuestionKind", "classify"]
