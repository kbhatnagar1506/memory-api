"""Question classification by a small model, with the regex as the floor.

`classify.py` says it plainly: "An LLM fallback can widen coverage later."
This is that. The regexes were written against a real failure distribution
and they hold up on the phrasings they were written for -- but they are
phrasings, and questions are not. "What is our entire infrastructure" read
as a lookup and returned an arbitrary ten of twenty-five stored facts,
because no pattern happened to contain the word `entire`. Adding the word
fixes that sentence and not the next one. A model reads the sentence.

Three properties make this safe to put on the read path, which is otherwise
LLM-free by design:

  * It FAILS OPEN. Timeout, vendor error, an unparseable answer, no backend
    configured -- every one of them falls through to `classify()`. The
    degraded mode is exactly today's behaviour, so an outage at the model
    vendor costs ranking quality and never availability.
  * It is BOUNDED. One sentence in, one word out: `max_output_tokens` in the
    tens, thinking off, and a hard `asyncio.wait_for` around the call. A slow
    vendor cannot make search slow; it can only make search dumber.
  * It is CACHED and CIRCUIT-BROKEN. Repeat questions never call out at all,
    and once the vendor has failed a few times in a row we stop calling until
    a cooldown passes, so a broken backend costs one timeout rather than one
    timeout per search.

Disagreements with the regex are logged with both labels. That is the whole
evaluation story for this component: the rules were validated against 264
real failures and 1,986 LoCoMo questions with zero false positives, so
replacing them wholesale on faith would be trading a measured thing for an
unmeasured one. Logging both means the swap can be checked against traffic
instead of argued about.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass

from ...core.logging import get_logger
from .classify import QuestionKind, classify
from .derive import CompleteFn

log = get_logger(__name__)

#: The model gets one job and one word. Every kind is described by what the
#: ANSWER has to be, not by how the question is phrased -- phrasing is what
#: the regex already does, and asking a model to imitate a regex gets a worse
#: regex. The examples are chosen for the boundaries that are genuinely
#: ambiguous: "how much did I spend on a handbag" is a lookup and "how much
#: did I spend in total" is arithmetic; "how many lenses should I buy" is
#: advice wearing a count's clothes.
UNDERSTAND_PROMPT = """\
Label what the ANSWER to this question would have to be. Reply with exactly \
one label and nothing else.

direct     — a fact stated somewhere in the history. Recall it.
list_all   — the complete set. Everything, the whole picture, all of them. \
Naming only the top few would be a wrong answer.
count      — a number nobody wrote down. Counting or summing is required.
order      — first, last, latest, earliest. Sorting by time is required.
date_arith — an interval. Subtracting dates is required.
compare    — a judgement between two or more remembered things.
advice     — a recommendation. The history holds preferences, not the answer; \
the answer has to be produced, not found.

Examples:
what did I name my dog -> direct
what is our entire infrastructure -> list_all
walk me through everything you know about my diet -> list_all
how much did I spend on the handbag -> direct
how much did I spend on clothes altogether -> count
how many times have I been to Lisbon -> count
what was the last thing I ordered -> order
how long after the wedding did they move -> date_arith
which apartment was cheaper -> compare
any tips for my trip to Rome -> advice
how many lenses should I buy -> advice

Question: {question}
Label:"""

_LABELS = {k.value: k for k in QuestionKind}

#: Consecutive failures before we stop calling, and how long we stay stopped.
#: Small numbers on purpose: the point is to notice an outage within a few
#: requests, not to ride one out.
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 30.0


@dataclass(frozen=True, slots=True)
class QueryIntent:
    """What the question is asking for, and who decided."""

    kind: QuestionKind
    #: "llm", "rules", or "cache". Surfaced in explain output so a surprising
    #: result set can always be traced back to the decision that shaped it.
    source: str

    @property
    def comprehensive(self) -> bool:
        """Whether answering with the top few would answer a different question."""
        return self.kind is QuestionKind.LIST_ALL


class QueryUnderstanding:
    """Classify a question with a small model; fall back to the regex.

    Construct one per process and share it -- the cache and the breaker are
    instance state, and both are worthless if every request makes its own.
    """

    def __init__(
        self,
        complete: CompleteFn | None,
        *,
        timeout_s: float = 2.0,
        cache_size: int = 2048,
    ) -> None:
        self._complete = complete
        self._timeout_s = timeout_s
        self._cache_size = cache_size
        self._cache: OrderedDict[str, QuestionKind] = OrderedDict()
        self._consecutive_failures = 0
        self._blocked_until = 0.0

    @property
    def enabled(self) -> bool:
        return self._complete is not None

    async def intent(self, question: str) -> QueryIntent:
        text = question.strip()
        if not text:
            return QueryIntent(QuestionKind.DIRECT, "rules")

        key = text.casefold()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return QueryIntent(cached, "cache")

        if self._complete is None or self._breaker_open():
            return QueryIntent(classify(text), "rules")

        kind = await self._ask(text)
        if kind is None:
            return QueryIntent(classify(text), "rules")

        rules_kind = classify(text)
        if kind is not rules_kind:
            # Not an error -- widening coverage is the point. Logged so the
            # swap is measurable against real traffic rather than asserted.
            log.info("intent_disagreement", llm=kind.value, rules=rules_kind.value)
        self._remember(key, kind)
        return QueryIntent(kind, "llm")

    # -- internals ---------------------------------------------------------

    def _breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._blocked_until:
            # Cooldown elapsed: let one request through to probe the vendor.
            self._consecutive_failures = 0
            return False
        return True

    async def _ask(self, question: str) -> QuestionKind | None:
        assert self._complete is not None
        try:
            raw = await asyncio.wait_for(
                self._complete(UNDERSTAND_PROMPT.format(question=question)),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            self._record_failure("timeout")
            return None
        except Exception as exc:  # never fail a search on this
            self._record_failure(str(exc)[:160])
            return None

        kind = parse_label(raw)
        if kind is None:
            # A reply we cannot read is a vendor problem like any other: it
            # counts toward the breaker, so a model that has started
            # answering in prose does not get asked five hundred more times.
            self._record_failure(f"unparseable: {raw.strip()[:60]!r}")
            return None
        self._consecutive_failures = 0
        return kind

    def _record_failure(self, reason: str) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._blocked_until = time.monotonic() + _BREAKER_COOLDOWN_S
        log.warning(
            "query_understanding_failed",
            reason=reason,
            consecutive=self._consecutive_failures,
        )

    def _remember(self, key: str, kind: QuestionKind) -> None:
        self._cache[key] = kind
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def parse_label(raw: str) -> QuestionKind | None:
    """The first known label anywhere in the reply.

    Deliberately forgiving about wrapping: models add a trailing period, a
    stray backtick, or a leading "Label:" no matter how the prompt is worded,
    and refusing those would throw away a correct answer over punctuation. It
    is NOT forgiving about content -- an unknown word returns None and the
    regex decides, rather than something being guessed into `direct`.
    """
    for token in raw.lower().split():
        cleaned = token.strip(" \t\n`'\"*.,:;-()[]")
        if cleaned in _LABELS:
            return _LABELS[cleaned]
    return None


__all__ = ["UNDERSTAND_PROMPT", "QueryIntent", "QueryUnderstanding", "parse_label"]
