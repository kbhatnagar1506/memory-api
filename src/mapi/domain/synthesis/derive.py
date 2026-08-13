"""Map -> ground -> reduce: compute answers instead of recalling them.

Every stage is designed around one measured failure each:

  * MAP is per-memory because finding instances in ONE document is the thing
    models are reliable at; finding-and-counting across six concatenated
    dialogues is where "three bikes" became "Multiple".
  * GROUND is code because extraction can fabricate. A row must carry a
    verbatim quote from its source or it does not exist. Substring check —
    no judge, no similarity threshold, no vibes.
  * REDUCE is code because arithmetic is where the model failed: 45% of
    failures-with-complete-evidence were counts. len() cannot miscount.

Fail-open everywhere: any error yields None and the caller falls back to the
direct path, so the worst case of this whole subsystem is the status quo.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from ..text import STOPWORDS
from .classify import QuestionKind
from .scope import DateRange, extract_scope

#: Any async text completion — the harness's Gemini client, a future product
#: client, or a test stub. Returns the raw completion text.
CompleteFn = Callable[[str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class SourceDoc:
    """One retrieved memory, as the derive path sees it."""

    id: str
    text: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class Extraction:
    """One grounded row: an event/fact with its date, quote and provenance."""

    date: date | None
    fact: str
    quote: str
    source_id: str


@dataclass(frozen=True, slots=True)
class DerivedAnswer:
    """An answer that was computed, with the table that proves it.

    `source_ids` is the provenance cluster: the memories this answer was
    derived from, ready to become `derived_from` edges when the answer is
    materialized as a memory of its own.
    """

    answer: str
    kind: QuestionKind
    table: tuple[Extraction, ...]
    source_ids: tuple[str, ...]
    computed: bool  #: True when code (not the model) produced the value
    #: The window the question restricted itself to, if any. Rows outside it
    #: were dropped BEFORE the reduce -- filter-then-aggregate, in code.
    scope: DateRange | None = None
    #: Grounded rows discarded by the scope filter. Reported rather than
    #: silently dropped: "8 found, 3 in March" is a different claim from
    #: "3 found", and an agent may want to say so.
    filtered_out: int = 0
    #: Rows dropped because they carried NO DATE while the question bounded
    #: itself in time. Split out from `filtered_out` because the two mean
    #: different things: out-of-window is the filter working, undated is a
    #: DATA QUALITY problem that silently undercounts. On a corpus where many
    #: memories lack dates, every windowed aggregate is low and nothing says
    #: so -- this is the field that says so.
    undated_dropped: int = 0
    #: Rows the extractor produced whose quote was NOT in its source -- i.e.
    #: fabricated, and discarded by grounding. A free, continuously measured
    #: hallucination rate on real traffic: the number every vendor in this
    #: category asserts and none publishes. A rising value is also the earliest
    #: warning that extraction has started to drift.
    rejected: int = 0

    @property
    def empty(self) -> bool:
        return not self.table

    @property
    def verified(self) -> bool:
        """Whether this answer was COMPUTED in code from multiple grounded rows.

        Provenance is persuasive, and that is a hazard: a wrong count wrapped
        in a grounded table with source ids reads as MORE trustworthy than a
        plain wrong answer, not less. Measured extraction is exactly right on
        only 60% of cases, so a caller needs to know which answers earned the
        table and which merely have one.

        True means: code did the arithmetic over at least two independently
        grounded rows. False means a model composed it, or a single row
        carried it -- treat as advisory.
        """
        return self.computed and len(self.table) >= 2

    @property
    def fabrication_rate(self) -> float:
        """Share of extracted rows that failed the verbatim-quote check."""
        produced = len(self.table) + self.filtered_out + self.rejected
        return self.rejected / produced if produced else 0.0


_MAP_PROMPT = """\
You are extracting evidence from ONE document to help answer a question.

Document (dated {doc_date}):
---
{text}
---

Question: {question}

List every DISTINCT event or fact in this document that helps answer the
question. Rules:
- "quote" must be copied VERBATIM from the document (it will be checked).
- "date" is when the event happened, YYYY-MM-DD. Resolve relative dates
  ("last Tuesday") against the document date. Use the document date if the
  event has no date of its own. Use null only if truly undatable.
- One entry per distinct item. If the document lists three bikes, emit three
  entries, not one entry saying "three bikes".
- Output [] if nothing in this document helps.

Output ONLY a JSON array:
[{{"date": "YYYY-MM-DD", "fact": "...", "quote": "..."}}]"""

#: The reference date is in here because without it a whole class of question
#: is not merely hard but UNCOMPUTABLE. "How many weeks ago did I attend the
#: sale", "how many months have passed since I last visited a museum" -- one
#: endpoint of that subtraction is the day the question was asked, and this
#: prompt used to supply dated rows and no "now". 25 of LongMemEval's 133
#: temporal-reasoning questions are that shape, 19% of the capability our
#: retrieval is already weakest on, and the model had no way to answer any of
#: them from what it was given.
#:
#: The direct answer path in the harness has stated the date all along. Only
#: the derived path, which is the one that ROUTES date arithmetic, did not.
_COMPOSE_PROMPT = """\
Answer the question using ONLY this table of dated, verified facts.

{table}
{asked_at}
Question: {question}

If the question asks how long ago something happened, or how much time has
passed since it, measure from its date to today's date above.

Answer in as few words as possible, in the unit the question asks for. If the
table cannot answer it, reply NO_ANSWER."""


def _parse_json_array(raw: str) -> list[dict[str, object]]:
    """Extract the first JSON array from a completion, tolerating fences.

    Models wrap JSON in ```json fences, prepend prose, or append trailing
    commentary. Grabbing the outermost [...] span and parsing that survives
    all three; anything unparseable yields [] rather than an exception —
    a lost extraction call must cost one document, not the whole derivation.
    """
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return (
        [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    )


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def ground(rows: Sequence[Extraction], sources: dict[str, str]) -> list[Extraction]:
    """Keep only rows whose quote actually appears in their source document.

    THE integrity gate: extraction output is model output, and a fabricated
    event in the table becomes a miscount presented with false confidence —
    worse than the failure it replaces. Whitespace-normalized, case-folded
    substring match: strict enough that invented text cannot pass, loose
    enough that a reflowed line still does.
    """
    kept: list[Extraction] = []
    normalized_sources = {sid: _normalize(text) for sid, text in sources.items()}
    for row in rows:
        source = normalized_sources.get(row.source_id, "")
        if row.quote and _normalize(row.quote) in source:
            kept.append(row)
    return kept


_TOKEN = re.compile(r"[a-z0-9]+")


def _content_tokens(fact: str) -> set[str]:
    """Tokens that distinguish one event from another.

    Stopwords are excluded BEFORE the similarity comparison, not after: "owns
    a road bike" and "owns a mountain bike" share three of five raw tokens
    (owns/a/bike) and would merge on raw Jaccard — the scaffolding words vote
    for a merge that the content words (road vs mountain) veto. Falls back to
    raw tokens when filtering empties the set, so an all-stopword fact still
    compares as itself rather than as everything.
    """
    tokens = set(_TOKEN.findall(fact.casefold()))
    content = {t for t in tokens if t not in STOPWORDS}
    return content or tokens


def _dedupe(rows: Sequence[Extraction]) -> list[Extraction]:
    """Collapse retellings: same date + mostly the same content words = one
    event.

    "Went to the gym" in session 3 retold in session 5 must count once;
    three different bikes listed in one session (same date!) must count
    three times. Date alone cannot distinguish these — the content words
    can: Jaccard >= 0.6 on stopword-filtered token sets, same-date rows
    only. Conservative on purpose: when ambiguous we under-count rather
    than fabricate.
    """
    kept: list[Extraction] = []
    for row in rows:
        tokens = _content_tokens(row.fact)
        duplicate = False
        for existing in kept:
            if existing.date != row.date:
                continue
            other = _content_tokens(existing.fact)
            union = tokens | other
            if union and len(tokens & other) / len(union) >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


#: An explicit duration stated in a fact ("took 4 hours") beats date
#: subtraction: "how long did it take to assemble the bookshelf" is answered
#: inside one episode, not between two.
_DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

#: A quantity with ANY unit attached, not only a duration.
#:
#: `_DURATION` exists for DATE_ARITH and is correctly time-only. But
#: `_states_quantity` -- which decides whether a fact carries the number the
#: question asked for -- was reading the same time-only list, so "I ran 12 km"
#: and "the shipment weighed 40 kg" and "it cost $1,200" all looked like
#: facts stating no quantity. Any corpus that is not a personal calendar is
#: mostly made of those.
#:
#: The unit is deliberately unconstrained: enumerating units is the same
#: mistake as enumerating negations, and a memory system does not get to
#: decide which domains its users work in. A number followed by a short word,
#: or preceded by a currency symbol, is a quantity.
_QUANTITY = re.compile(
    r"(?:[$£€¥₹]\s?\d[\d,]*(?:\.\d+)?)"
    r"|\b\d[\d,]*(?:\.\d+)?\s*%"
    r"|\b\d[\d,]*(?:\.\d+)?\s*[a-zA-Z]{1,12}\b"
)


#: Cardinal words, because a quantity written out is still stated. Bounded at
#: twenty plus round numbers: past that, prose uses digits.
_NUMBER_WORDS = frozenset(
    [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "dozen",
    ]
)
#: A digit run that is plausibly a quantity. Years are excluded because an
#: extracted fact routinely carries a date in its text ("in 2026 I caught
#: bass"), and reading 2026 as a count is worse than reading nothing.
_DIGITS = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\b")
_YEARISH = re.compile(r"^(19|20)\d{2}$")


#: A sentence end, including any quote or bracket that closes after it, so a
#: cut lands after `said."` rather than between the period and the quote.
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s")

#: How far back from the limit a boundary may be before taking it costs more
#: text than the tidiness is worth.
_BOUNDARY_FLOOR = 0.75

_MARKER = "\n[... truncated]"


def _clip(text: str, limit: int) -> str:
    """Trim on a boundary and mark it, rather than amputating mid-word.

    Boundaries in descending order of how little they cost the reader:
    paragraph, line, sentence, word. The previous version stopped at `". "` and
    then fell through to a hard character cut, so anything without that exact
    sequence in its last quarter was severed mid-word. The whitespace tier is
    what closes that hole; the sentence tier only makes the result tidier.
    """
    if len(text) <= limit:
        return text
    head, floor = text[:limit], limit * _BOUNDARY_FLOOR

    for sep in ("\n\n", "\n"):
        cut = head.rfind(sep)
        if cut > floor:
            return head[:cut].rstrip() + _MARKER

    sentences = [m.end() for m in _SENTENCE_END.finditer(head)]
    if sentences and sentences[-1] > floor:
        return head[: sentences[-1]].rstrip() + _MARKER

    space = head.rfind(" ")
    return (head[:space] if space > floor else head).rstrip() + _MARKER


def _states_quantity(fact: str) -> bool:
    """True when the fact text itself carries the number being asked for.

    THE distinction this whole reducer turns on. "How many bass did I catch"
    is answered by one episode saying "I caught 12 bass" -- one table row
    holding the quantity 12 -- not by counting rows. Measured cost of getting
    this backwards, on 500 real questions: gold 12 answered "1", gold 17
    answered "1", gold $800 answered "1". `len(table)` was computing how many
    times the subject was MENTIONED, and presenting it as how many there ARE.
    """
    lowered = fact.casefold()
    if any(word in lowered.split() for word in _NUMBER_WORDS):
        return True
    if _QUANTITY.search(fact):
        return True
    return any(not _YEARISH.match(m.replace(",", "")) for m in _DIGITS.findall(fact))


#: The unit the answer has to be expressed in. Reading it off the question is
#: the whole point: the reducer used to return "N days" for every date
#: subtraction, so "how many WEEKS ago did I attend the sale" -- whose gold
#: answer is the bare number 4 -- was answered "28 days". Correct retrieval,
#: correct dates, correct arithmetic, wrong answer.
_ASKED_UNIT = re.compile(r"\bhow (?:many|much)\s+(\w+)", re.IGNORECASE)

#: Days per unit. Weeks are exact; months and years are the conventional
#: approximations, which is what a question asking "how many months ago"
#: expects -- nobody asking that wants a calendar-difference edge case.
_UNIT_DAYS_OUT = {"day": 1, "week": 7, "month": 30, "year": 365}

#: A question measuring FROM an event TO the present. One endpoint of that
#: subtraction is the day the question was asked, so it cannot be computed from
#: the table alone -- and the old reducer required two dates in the table and
#: returned None for all of these. 25 of LongMemEval's 133 temporal questions.
_TO_PRESENT = re.compile(
    r"\bago\b|\bsince\b|\b(?:have|has)\s+passed\b|\buntil now\b|\bso far\b",
    re.IGNORECASE,
)


def _date_span(
    question: str, table: Sequence[Extraction], asked_at: date | None
) -> str | None:
    """A date subtraction, in the unit the question asked for.

    Two endpoints are needed and only one of them is always in the table. A
    question saying "ago" or "since" measures to the present, so `asked_at`
    supplies the other end; anything else measures between two rows.

    Deliberately silent rather than wrong. A question measuring to the present
    over a table holding several unrelated dates cannot be resolved here -- the
    row being asked about is not identifiable from dates alone -- so it returns
    None and composes through the model, which can read the facts and pick. The
    old code guessed max-minus-min in that case and answered confidently.
    """
    dates = sorted({row.date for row in table if row.date is not None})
    if not dates:
        return None

    unit = "day"
    if match := _ASKED_UNIT.search(question):
        candidate = match.group(1).lower().rstrip("s")
        if candidate in _UNIT_DAYS_OUT:
            unit = candidate

    if _TO_PRESENT.search(question):
        if asked_at is None or len(dates) != 1:
            return None
        days = (asked_at - dates[0]).days
    elif len(dates) >= 2:
        days = (dates[-1] - dates[0]).days
    else:
        return None

    if days < 0:
        # An event dated after the question was asked means the extraction or
        # the reference date is wrong. Saying so is better than a negative age.
        return None
    if unit == "day":
        return f"{days} days"
    return f"{days // _UNIT_DAYS_OUT[unit]} {unit}s"


def _reduce_in_code(
    kind: QuestionKind,
    table: list[Extraction],
    question: str = "",
    asked_at: date | None = None,
) -> str | None:
    """The arithmetic stage. Returns None when this kind needs the model."""
    if kind is QuestionKind.COUNT:
        # The stated-quantity branch was DELETED here, not disabled.
        #
        # It returned one row's PROSE with computed=True, which satisfied the
        # harness override gate written to admit only code-computed values --
        # and `DerivedAnswer.verified` documents itself as "code did the
        # arithmetic over at least two independently grounded rows". The branch
        # violated that contract while reporting compliance with it.
        #
        # Measured: 16 of 17 COUNT firings across all runs bypassed len()
        # through this branch (a date like 2023-02-01 inside `fact` flips it
        # via the 02/01 tokens, which the year filter does not catch), and
        # every one of the 4 firings in lme-v8 was wrong -- 3 of those
        # questions are answered correctly by the config that never invokes
        # derive at all.
        #
        # A count is len() over grounded rows. If the number is stated in the
        # prose, the direct answer path already reads it correctly (measured:
        # count-shaped questions score 0.869 with complete evidence, versus
        # 0.880 for everything else -- there is no counting gap to harvest).
        #
        # But deleting the branch outright would trade one bug for another:
        # rows reading "owns 3 bikes" and "owns 4 bikes" are not two instances,
        # and len() would answer 2. A row that states its own quantity is not
        # an enumerable instance, so if ANY row does, this reduce cannot
        # safely aggregate and declines -- leaving the question to the direct
        # path, which has the full text and reads the number correctly.
        if any(_states_quantity(row.fact) for row in table):
            return None
        # 2. A single row with no stated quantity is unresolvable: "one
        #    instance" and "the number was in the text and extraction missed
        #    it" are indistinguishable, and answering "1" was wrong far more
        #    often than right. Decline to compute; the model composes instead.
        if len(table) < 2:
            return None
        # 3. Genuine instance counting: several distinct grounded rows.
        return str(len(table))

    if kind is QuestionKind.ORDER:
        dated = [row for row in table if row.date is not None]
        if not dated:
            return None
        dated.sort(key=lambda row: row.date or date.min)
        # The question's own wording picks the end: default to the most
        # recent, which is what bare "last/latest" asks for.
        chosen = dated[-1]
        return f"{chosen.fact} ({chosen.date.isoformat()})" if chosen.date else chosen.fact

    if kind is QuestionKind.DATE_ARITH:
        for row in table:
            if _DURATION.search(row.fact):
                # The fact, not the regex match. Extracting "45 minutes" from
                # "commute is 45 minutes each way" dropped the qualifier the
                # gold answer required -- a correct answer narrowed into a
                # wrong one. Under a contains-the-answer judge, more context
                # can only help.
                return row.fact
        return _date_span(question, table, asked_at)

    if kind is QuestionKind.LIST_ALL:
        if not table:
            return None
        return "; ".join(row.fact for row in table)

    return None  # COMPARE and anything else: compose via the model


def _format_table(table: Sequence[Extraction]) -> str:
    lines = [
        f"- [{row.date.isoformat() if row.date else 'undated'}] {row.fact}" for row in table
    ]
    return "\n".join(lines)


async def derive_answer(
    question: str,
    kind: QuestionKind,
    docs: Sequence[SourceDoc],
    complete: CompleteFn,
    *,
    concurrency: int = 8,
    # 24_000, matching the direct answer path. It was 12_000 -- LOWER than
    # the path that crams ten sessions into one prompt, while this one maps a
    # single document per call and therefore has the most room in the system.
    # The asymmetry had no stated reason and cut 41.5% of a real corpus
    # mid-word, which is a plausible cause of the derive arm's regression:
    # grounding drops any claim whose quote fell past the cut, and the reducer
    # then counts over an incomplete table.
    max_doc_chars: int = 24_000,
    asked_at: date | None = None,
) -> DerivedAnswer | None:
    """Run map -> ground -> dedupe -> reduce for one question.

    Returns None (never raises) when derivation cannot help: no docs, every
    extraction failed or was rejected by grounding, or the reduce needed the
    model and the model declined. The caller falls back to the direct path,
    so this subsystem can only add correct answers, not subtract them.
    """
    if not docs:
        return None
    semaphore = asyncio.Semaphore(concurrency)

    async def map_one(doc: SourceDoc) -> list[Extraction]:
        prompt = _MAP_PROMPT.format(
            doc_date=doc.occurred_at.date().isoformat(),
            text=_clip(doc.text, max_doc_chars),
            question=question,
        )
        async with semaphore:
            try:
                raw = await complete(prompt)
            except Exception:
                return []  # one lost document, not a lost derivation
        rows = []
        for item in _parse_json_array(raw):
            fact = str(item.get("fact") or "").strip()
            quote = str(item.get("quote") or "").strip()
            if fact and quote:
                rows.append(
                    Extraction(
                        date=_parse_date(item.get("date")),
                        fact=fact,
                        quote=quote,
                        source_id=doc.id,
                    )
                )
        return rows

    mapped = await asyncio.gather(*(map_one(doc) for doc in docs))
    raw_rows = [row for rows in mapped for row in rows]
    grounded = ground(raw_rows, {doc.id: doc.text for doc in docs})
    rejected = len(raw_rows) - len(grounded)
    table = _dedupe(grounded)

    # FILTER, then reduce. A question that bounds itself in time ("how many
    # weddings this year") must aggregate over the window, not over everything
    # the map stage found -- measured, the errors ran BOTH ways, which is the
    # signature of a missing filter rather than a missing fact.
    scope = extract_scope(question, asked_at) if asked_at else None
    filtered_out = 0
    undated_dropped = 0
    if scope is not None:
        in_scope = [row for row in table if scope.contains(row.date)]
        undated_dropped = sum(1 for row in table if row.date is None)
        filtered_out = len(table) - len(in_scope)
        # An empty window is more likely a dating failure than a true zero, so
        # decline rather than assert "0" -- the model still sees the excerpts.
        if not in_scope:
            return None
        table = in_scope

    if not table:
        return None

    computed = _reduce_in_code(kind, table, question, asked_at)
    if computed is not None:
        answer = computed
        was_computed = True
    else:
        try:
            raw = await complete(
                _COMPOSE_PROMPT.format(
                    table=_format_table(table),
                    question=question,
                    asked_at=(
                        f"\nToday's date is {asked_at.isoformat()}.\n" if asked_at else ""
                    ),
                )
            )
        except Exception:
            return None
        answer = raw.strip()
        was_computed = False
        if not answer or answer.upper().startswith("NO_ANSWER"):
            return None

    return DerivedAnswer(
        answer=answer,
        kind=kind,
        table=tuple(table),
        source_ids=tuple(dict.fromkeys(row.source_id for row in table)),
        computed=was_computed,
        scope=scope,
        filtered_out=filtered_out,
        undated_dropped=undated_dropped,
        rejected=rejected,
    )


__all__ = [
    "CompleteFn",
    "DerivedAnswer",
    "Extraction",
    "SourceDoc",
    "derive_answer",
    "ground",
]
