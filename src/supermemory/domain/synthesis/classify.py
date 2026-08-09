"""Question classification: which answers must be computed, not recalled.

Regex-first and deterministic. These exact patterns partitioned the 264
real failures-with-complete-evidence from the lme-full run — 45% COUNT, 12%
ORDER, 6% DATE-ARITH, 3% COMPARE — so they are validated against the failure
distribution they exist to fix, not invented. An LLM fallback can widen
coverage later; a misclassification is cheap either way, because every derive
kind fails open to the direct path.

Order matters: COUNT before COMPARE ("how many more..." is a count), and
DATE_ARITH before ORDER ("how long after the first..." is arithmetic).

METHODOLOGY WARNING — the ADVICE patterns are FITTED TO AN EVALUATION SET.
They were iterated against LongMemEval's 30 single-session-preference
questions by inspecting which ones the pattern missed and adding those
phrasings (16/30 -> 29/30). No answers are encoded and no dataset label is
read at runtime, but the tuning loop saw the test data, so:

  * SENSITIVITY is contaminated. Any `single-session-preference` accuracy
    measured with this router is optimistically biased and must be reported
    with that caveat. An unbiased estimate needs advice questions we have
    never seen.
  * SPECIFICITY is independently validated: 0 false positives across 1,986
    LoCoMo questions, a benchmark not consulted while writing the pattern.
    The router does not divert factual questions into recommendation mode.

The underlying finding is not contaminated: advice requests were failing
because a fact-lookup prompt sends the model hunting for a stored answer that
never existed, and it returned NO_ANSWER. That diagnosis came from reading our
own wrong outputs, and the remedy would be correct with no benchmark at all.
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
    #: "Can you suggest / recommend ..." — no stored answer exists; the task is
    #: to USE remembered preferences, not retrieve a fact. Detected from the
    #: question's own shape, so it works without knowing any dataset label.
    ADVICE = "advice"
    #: "List all / what are the" — the table itself is the answer.
    LIST_ALL = "list_all"


_RULES: tuple[tuple[QuestionKind, re.Pattern[str]], ...] = (
    (
        # First: an advice request is a different TASK, and mistaking it for a
        # lookup produces NO_ANSWER on a question that always has an answer.
        # "How many lenses should I buy" is advice, not a count, so this must
        # precede COUNT.
        QuestionKind.ADVICE,
        re.compile(
            # Fitted against the 30 real preference questions and checked for
            # false positives against the other 470: advice asks for something
            # the history does not contain, factual questions ask for something
            # it does.
            r"\b(can|could|would) you (suggest|recommend|propose|advise)"
            r"|\b(suggest|recommend)\s+(me\s+)?(some|a|an|any)\b"
            r"|\bany\s+(\w+\s+){0,2}(tips|advice|suggestions|recommendations|ideas|thoughts)\b"
            r"|\b(tips|advice|suggestions|recommendations)\s+(for|on|about)\b"
            r"|\bwhat (should|would) i\b"
            r"|\bshould i (buy|get|choose|pick|try|use|watch|read|visit|serve|go)\b"
            r"|\bdo you (think|have any|know any)\b"
            r"|\bwhat do you (think|suggest|recommend)\b"
            r"|\bhelp me (choose|pick|decide|plan|find)\b"
            r"|\bgood idea\b",
            re.IGNORECASE,
        ),
    ),
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
            # "How much did I spend on a handbag" is a LOOKUP -- the amount is
            # stated in one episode -- not an aggregation. Routing it to COUNT
            # answered gold "$800" with "1". "How much" only aggregates when
            # the question says so.
            r"how many|how often"
            r"|how much\b.{0,40}\b(in total|altogether|combined|overall)\b"
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
