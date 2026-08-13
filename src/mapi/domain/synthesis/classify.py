"""Question classification: which answers must be computed, not recalled.

Regex-first and deterministic. These exact patterns partitioned the 264
real failures-with-complete-evidence from the lme-full run — 45% COUNT, 12%
ORDER, 6% DATE-ARITH, 3% COMPARE — so they are validated against the failure
distribution they exist to fix, not invented. An LLM fallback can widen
coverage later; a misclassification is cheap either way, because every derive
kind fails open to the direct path.

Order matters: COUNT before COMPARE ("how many more..." is a count), and
DATE_ARITH before ORDER ("how long after the first..." is arithmetic).

METHODOLOGY — the ADVICE rule was fitted to an evaluation set, and is not any
more. The original was eleven alternations found by inspecting which of
LongMemEval's 30 single-session-preference questions the pattern missed and
adding each missed phrasing (16/30 -> 29/30). No answer was ever encoded and no
dataset label is read at runtime, but the tuning LOOP saw the test data, which
makes a sensitivity number measured with it optimistically biased.

It is now four grammatical frames -- a directive to the assistant, a bare
imperative, first-person deliberation about a future action, and a request for
an advice noun -- each justifiable without reference to any benchmark, and each
tested against held-out phrasings written for this repository rather than drawn
from an eval set. See `tests/unit/test_classify_preference.py`.

What that does and does not buy:

  * The RULE is no longer fitted, so it has a reason to generalise. The
    held-out set is the evidence for that, and it is small (n=40).
  * A sensitivity number measured on LongMemEval's 30 questions is STILL not
    unbiased, because those 30 questions informed the original pattern and the
    rewrite preserves their coverage by design. Reporting
    `single-session-preference` accuracy still needs that caveat.
  * SPECIFICITY was independently validated at 0 false positives across 1,986
    LoCoMo questions, a benchmark not consulted while writing the pattern. The
    rewrite widened the rule, so that figure is a prior and not a measurement
    of the current pattern -- the held-out negatives in the test file are.

Also worth knowing: this regex is the FALLBACK. `synthesis/understand.py`
classifies with a model and only falls back here on timeout or vendor failure,
so on the live path the fitted-ness of the regex is not what decides intent.

The underlying finding was never contaminated: advice requests were failing
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
            # FOUR GRAMMATICAL FRAMES, not a list of phrasings.
            #
            # This rule was originally eleven alternations discovered by
            # inspecting which of LongMemEval's 30 preference questions the
            # pattern missed and adding each one. It covered them (16/30 ->
            # 29/30) and it was fitted, so it could not be trusted to hold on
            # anything else. Rewritten as the frames English actually uses to
            # request a recommendation, each one justified on its own and
            # tested against held-out phrasings written without the eval set
            # open. Coverage of the fitted cases is preserved; what changes is
            # that there is now a reason for each branch.
            #
            # 1. DIRECTIVE TO THE ASSISTANT. A modal, the second person, and a
            #    verb of suggestion. The intervening words are for "can you
            #    suggest", "could you please recommend", "would you maybe
            #    propose".
            r"\b(?:can|could|would|will)\s+you\s+(?:\w+\s+){0,2}"
            r"(?:suggest|recommend|propose|advise|help)\b"
            # 2. BARE IMPERATIVE. Same verbs with the modal dropped, which is
            #    how the request is usually typed: "suggest a few", "recommend
            #    me something", "advise on a route".
            r"|\b(?:suggest|recommend|advise|propose)\s+(?:me\s+)?"
            r"(?:some|a|an|any|the|something|anything|\d)\b"
            # 3. FIRST-PERSON DELIBERATION about a future action. The asker is
            #    weighing a choice, so no stored answer exists -- and this is
            #    the frame, not a list of verbs, so it covers any verb after
            #    the modal. Bounded to a short window so it does not swallow a
            #    factual question that happens to contain "I".
            r"|\b(?:what|which|who|where|how)\s+(?:\w+\s+){0,3}"
            r"(?:should|ought)\s+(?:i|we)\b"
            r"|\b(?:should|ought)\s+(?:i|we)\b"
            r"|\bhelp\s+me\s+\w+"
            # 4. AN ADVICE NOUN IN A REQUESTING CONTEXT. The nouns are a small
            #    closed set -- English has no further words for "a
            #    recommendation" -- but the noun ALONE is not the frame, and
            #    treating it as one was wrong: "what was the advice the lawyer
            #    gave me in March" and "what tips did the instructor give"
            #    REPORT advice received, which is recall, and routing them to
            #    ADVICE makes the model invent a suggestion instead of finding
            #    what the corpus says. Caught by the held-out negatives in
            #    tests/unit/test_preference.py, on the first run.
            #
            #    So the noun has to be requested: introduced by an indefinite
            #    determiner ("any tips", "a few pointers"), governed by a verb
            #    of wanting ("looking for suggestions", "need guidance"),
            #    attached to its topic ("advice about the commute"), or
            #    addressed to the assistant ("your thoughts").
            r"|\b(?:any|some|a\s+few|a\s+couple\s+of)\s+(?:\w+\s+){0,2}"
            r"(?:tips|advice|suggestions?|recommendations?|ideas|thoughts|pointers|guidance)\b"
            r"|\b(?:looking\s+for|need|want|would\s+(?:love|like))\s+(?:some\s+|any\s+)?"
            r"(?:tips|advice|suggestions?|recommendations?|ideas|thoughts|pointers|guidance)\b"
            r"|\b(?:tips|advice|suggestions?|recommendations?|ideas|pointers|guidance)"
            r"\s+(?:for|on|about)\b"
            r"|\byour\s+(?:thoughts|advice|suggestions?|recommendations?)\b"
            # 5. EVALUATION OF A PROPOSAL. "Is it worth", "would it be a good
            #    idea", "does it make sense to" -- asking for a judgement on a
            #    course of action rather than for a fact about the past.
            r"|\b(?:a\s+)?good\s+idea\b"
            r"|\b(?:is|would)\s+it\s+(?:be\s+)?worth\b"
            r"|\bmakes?\s+sense\s+to\b"
            r"|\bwhat\s+do\s+you\s+(?:think|suggest|recommend)\b",
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
            # Comprehensiveness markers, added for the PRODUCT rather than for
            # any benchmark: "what is our entire infrastructure" read as a
            # lookup and returned an arbitrary ten of twenty-five facts. These
            # are generic English -- entire, whole, everything, all of our --
            # and carry none of the eval-set fitting the ADVICE rule above
            # warns about.
            r"\blist (all|the|every)|what are (all|the .+s i)|which ones\b"
            r"|\b(the )?(entire|whole|complete|full)\s+\w+"
            r"|\beverything (about|we|i|you|they)\b"
            r"|\ball (of )?(our|my|the|their)\b"
            r"|\bwhat (all|else)\b"
            r"|\bgive me (a rundown|an overview|the full)\b"
            r"|\b(overview|rundown|summary) of\b",
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
