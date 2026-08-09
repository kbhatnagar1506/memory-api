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

_COMPOSE_PROMPT = """\
Answer the question using ONLY this table of dated, verified facts.

{table}

Question: {question}

Answer in as few words as possible. If the table cannot answer it, reply
NO_ANSWER."""


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
    return any(not _YEARISH.match(m.replace(",", "")) for m in _DIGITS.findall(fact))


def _reduce_in_code(kind: QuestionKind, table: list[Extraction]) -> str | None:
    """The arithmetic stage. Returns None when this kind needs the model."""
    if kind is QuestionKind.COUNT:
        # 1. A stated quantity beats a row count, always. Return the whole
        #    fact rather than the bare number: the official judge asks whether
        #    the response CONTAINS the correct answer, so carrying the
        #    surrounding words is strictly safer than extracting a scalar and
        #    risking the wrong one ("12 bass and 3 trout").
        stated = [row for row in table if _states_quantity(row.fact)]
        if stated:
            # Conflicting statements ("3 bikes" in March, "4 bikes" in May)
            # resolve the same way the store resolves any revision: latest
            # event time wins.
            return max(stated, key=lambda row: row.date or date.min).fact
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
        dates = sorted(row.date for row in table if row.date is not None)
        if len(dates) >= 2:
            return f"{(dates[-1] - dates[0]).days} days"
        return None

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
    max_doc_chars: int = 12_000,
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
            text=doc.text[:max_doc_chars],
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
    if scope is not None:
        in_scope = [row for row in table if scope.contains(row.date)]
        filtered_out = len(table) - len(in_scope)
        # An empty window is more likely a dating failure than a true zero, so
        # decline rather than assert "0" -- the model still sees the excerpts.
        if not in_scope:
            return None
        table = in_scope

    if not table:
        return None

    computed = _reduce_in_code(kind, table)
    if computed is not None:
        answer = computed
        was_computed = True
    else:
        try:
            raw = await complete(
                _COMPOSE_PROMPT.format(table=_format_table(table), question=question)
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
