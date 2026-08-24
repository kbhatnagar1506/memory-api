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
            r"(?:suggest|recommend|propose|advise|help|plan|pick|choose)\b"
            # 2. BARE IMPERATIVE. Same verbs with the modal dropped, which is
            #    how the request is usually typed: "suggest a few", "recommend
            #    me something", "advise on a route".
            r"|\b(?:suggest|recommend|advise|propose)\s+(?:me\s+)?"
            r"(?:some|a|an|any|the|something|anything|\d)\b"
            # 2b. THE SAME IMPERATIVE, for verbs that are also common nouns.
            #     Measured, not anticipated: "Plan how I should get to the
            #     Edinburgh review" classified DIRECT, took the fact-lookup
            #     contract, and answered "I don't have that in memory" -- a
            #     refusal on a request that always has an answer, which is the
            #     exact failure ADVICE exists to prevent.
            #
            #     These cannot join frame 2, because there the verb is
            #     identified by the determiner after it and "plan" takes the
            #     same shape as a NOUN doing so: "the plan the architect gave
            #     me" would match "plan"+"the" and route a recall question to
            #     advice. Anchoring to the start of the string identifies the
            #     imperative by POSITION instead -- a sentence-initial verb is
            #     not a noun phrase, and "what was the plan for March" keeps
            #     its DIRECT routing because `plan` is not where a verb goes.
            r"|^\s*(?:please\s+)?(?:plan|pick|choose)\b"
            # 3. FIRST-PERSON DELIBERATION about a future action. The asker is
            #    weighing a choice, so no stored answer exists -- and this is
            #    the frame, not a list of verbs, so it covers any verb after
            #    the modal. Bounded to a short window so it does not swallow a
            #    factual question that happens to contain "I".
            r"|\b(?:what|which|who|where|how)\s+(?:\w+\s+){0,3}"
            r"(?:should|ought)\s+(?:i|we)\b"
            r"|\b(?:should|ought)\s+(?:i|we)\b"
            r"|\bhelp\s+me\s+\w+"
            # 3b. THE SAME DELIBERATION IN AN EMBEDDED CLAUSE. English inverts
            #     the subject and modal in a direct question ("what should I
            #     wear") and does not in a subordinate one ("anything I should
            #     request", "advice on what I should wear"). It is one frame
            #     with two word orders, and only the inverted one was covered.
            #
            #     `(?!have|had)` is what keeps it out of the past: "what I
            #     should have done in March" and "when we should have renewed"
            #     REPORT a missed obligation, which is recall. `(?<!\bsay)` and
            #     its siblings block reported advice -- "what did the doctor say
            #     I should do" is a question about the corpus, not a request for
            #     a recommendation, and routing it here makes the model invent
            #     one instead of looking it up.
            #     The lookbehinds are a closed class of REPORTING verbs, the
            #     same justification frame 4 uses for its nouns: "she told me
            #     what I should charge" embeds the deliberation under someone
            #     else's past speech, so it is a question about the corpus.
            #     They are separate assertions because Python requires each
            #     lookbehind to be fixed-width, and "told me " is not "said ".
            r"|(?<!\bsaid )(?<!\bsays )(?<!\btold )(?<!\btold me )(?<!\basked )"
            r"(?<!\badvised )(?<!\bsaid that )"
            #     The subject is any PERSONAL pronoun, not just first person.
            #     "What should he put in my tea", "anything she ought to tell
            #     them when she books" -- somebody else acting on the user's
            #     behalf is still a request to apply the user's preferences,
            #     and it was the single most common miss in the lab corpus.
            #     Restricted to pronouns on purpose: "what should the invoice
            #     threshold have been" must stay a lookup.
            #     No REPORTING VERB may sit inside the window either. The
            #     lookbehinds above guard the position before the wh-word
            #     ("told me what I should charge"); this guards between it and
            #     the pronoun ("what did the doctor SAY I should do"), which is
            #     where the verb lands in the commoner phrasing and where
            #     widening the window to five words first let it through.
            r"\b(?:anything|something|what|which|where|how|whether)\s+"
            r"(?:(?!(?:say|says|said|tell|tells|told|ask|asks|asked|advise|advises"
            r"|advised|recommend|recommends|recommended|suggest|suggests|suggested)\b)"
            r"\w+\s+){0,5}(?:i|we|he|she|they)\s+"
            r"(?:should|ought\s+to|could)\s+(?!have\b|had\b)"
            r"|\b(?:should|ought\s+to)\s+(?:he|she|they)\b(?!\s+have\b)"
            # 3c. IMPERATIVE WITH A WH-COMPLEMENT. Frame 2 identifies its verbs
            #     by the determiner after them, so "recommend a hotel" is caught
            #     and "recommend WHAT KIND of hotel" is not -- same verb, same
            #     request, different complement.
            #     `when` is deliberately absent: after these verbs it is almost
            #     always a temporal adverbial rather than a complement, and
            #     including it routed "which wine did Priya SUGGEST WHEN we
            #     met" -- a lookup -- to advice.
            r"|\b(?:suggest|recommend|advise|propose|plan|pick|choose)\s+"
            r"(?:me\s+)?(?:what|which|how|where|whether)\b"
            # 3d. SUPERLATIVE. "What is the best way to get there" asks for a
            #     judgement, not a stored fact. Present tense only: "what WAS
            #     the best month for sales" is a lookup over history.
            r"|\b(?:what|which|where|who)(?:\s+\w+){0,2}\s*"
            r"(?:'s|\bis|\bare|\bwould\s+be)\s+"
            r"the\s+(?:best|better|right|ideal|safest|cheapest)\b"
            # 3e. INDEFINITE + EVALUATIVE, which is how consumers actually ask.
            #     3d covers "what's THE BEST way" and missed "what's A GOOD
            #     place to eat" -- definite-superlative but not
            #     indefinite-evaluative. Measured on consumer-register phrasing:
            #     2 of 12 quality-seeking requests routed correctly, so "what's
            #     a good place to eat", "where's a good spot for brunch" and
            #     "any good cafes" all took the DECLINE contract and answered
            #     "I don't have that in memory" to a request that always has an
            #     answer.
            #
            #     The frame is the INDEFINITE DETERMINER doing the work: asking
            #     for AN instance of a category is a recommendation request,
            #     while asking for THE stored particular is recall. "what's a
            #     good place" against "what's my usual coffee" -- same wh-word,
            #     different determiner, different task.
            #
            #     CONTRACTED FORM ONLY, deliberately. "what's a good place"
            #     matches; "what IS a good place" does not. Consumers contract,
            #     and the uncontracted form is where definitional questions live
            #     -- "what is a good faith estimate" would take the advice
            #     contract and answer a general-knowledge question with a guess
            #     dressed as a preference. The elided-apostrophe pass below
            #     means "whats a good place" reaches this frame too, which is
            #     the register that actually needed covering.
            r"|\b(?:what|where|who|which)(?:'s|\bis|\bare)\s+"
            r"(?:a|an|any|some)\s+(?:\w+\s+){0,2}"
            r"(?:good|great|nice|decent|solid|cheap|affordable|reliable|easy|"
            r"quick|quiet|fun|safe|better|best)\b"
            #     And the bare form with no wh-word at all: "any good cafes",
            #     "know any good places". Same indefinite-plus-evaluative shape.
            r"|\b(?:any|know\s+any|got\s+any|got\s+a)\s+(?:\w+\s+){0,1}"
            r"(?:good|great|nice|decent|solid|cheap|reliable|fun)\b"
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


#: Elided apostrophes, which informal writing drops as a matter of course.
#: "whats the best way" classified DIRECT while "what's the best way"
#: classified ADVICE -- the same question, routed to opposite contracts by a
#: punctuation mark. Restricted to wh-words because those are unambiguous:
#: "whats" and "wheres" are not words, whereas "its", "id" and "ill" are, and
#: expanding those would invent meanings the writer did not intend.
_ELIDED_APOSTROPHE = re.compile(r"\b(what|where|how|who|when|why)s\b", re.IGNORECASE)


def classify(question: str) -> QuestionKind:
    """Route a question to the machinery that can actually answer it."""
    question = _ELIDED_APOSTROPHE.sub(r"\1's", question)
    for kind, pattern in _RULES:
        if pattern.search(question):
            return kind
    return QuestionKind.DIRECT


__all__ = ["QuestionKind", "classify"]
