"""Contradiction adjudication: the pairs seven strings cannot see.

`consolidation.propose_contradictions` finds a conflict when two topically
similar memories differ by a negation, an antonym, or a figure. That works,
and it is free, and it is bounded by a seven-item list:

    not · never · cannot · can't · won't · will not · stopped

So "I stopped eating meat" conflicts with "I eat meat", and *no longer*,
*quit*, *gave up*, *switched away from*, *used to*, *isn't anymore*,
*discontinued* and every other way English negates a thing all read as
agreement. Extending the list moves the boundary; it does not remove it.

This adds a second pass over exactly the pairs the list could not explain --
already filtered to high topical similarity, so the model sees a handful of
plausible conflicts per write and not the corpus. The lexical pass keeps its
job: it is free, it is right about the cases it catches, and it runs when no
backend is configured.

FAILS CLOSED, unlike every other model call in this system. Elsewhere a
vendor failure degrades a ranking and the user gets slightly worse results.
Here a failure that invented conflicts would put CONTRADICTS edges between
memories that agree, and the module this supports says why that is the worse
direction: "a false contradiction erodes trust faster than a missed one." A
timeout, a parse failure, or an unreachable vendor therefore yields no
proposals at all -- the same output as having no backend, which is the
behaviour this feature has always had.
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from ...core.logging import get_logger
from .derive import CompleteFn
from .extract import _salvage_objects

log = get_logger(__name__)

#: Pairs sent in one call. The similarity gate upstream rarely admits more,
#: and a prompt holding fifty candidate memories is one where the model
#: starts skimming.
MAX_CANDIDATES = 8

#: Below this the verdict is discarded. The prompt asks for a confidence and
#: the floor is high because the default answer to "do these conflict" should
#: be no -- most pairs that survive an 0.82 similarity gate are restatements,
#: not disagreements.
MIN_CONFIDENCE = 0.7

ADJUDICATE_PROMPT = """\
Decide which of the numbered statements CONTRADICT the new statement.

A contradiction means both cannot be true of the same subject at the same \
time. Be strict:

  * A statement that ADDS detail is not a contradiction. "I run on Tuesdays" \
and "I run on Tuesdays and Fridays" agree.
  * A statement about a DIFFERENT subject is not a contradiction, however \
similar the wording.
  * A CHANGE OVER TIME is a contradiction only if both are stated as current. \
"I used to live in Berlin" and "I live in Madrid" agree.
  * A restatement in other words is not a contradiction.
  * Opposites, reversals and abandonments ARE contradictions however they are \
phrased: stopped, quit, gave up, no longer, switched away from, cancelled, \
changed my mind.

New statement:
{statement}

Existing statements:
{candidates}

Reply with a JSON array and nothing else. Include ONLY the ones that \
contradict; reply with [] if none do.

[{{"n": <number>, "reason": "<six words or fewer>", "confidence": <0.0-1.0>}}]"""


SUPERSEDE_PROMPT = """\
Decide which of the numbered statements the NEW statement REPLACES.

A replacement means the same attribute of the same subject now has a \
different value, so the old statement is no longer true. Be strict:

  * A DIFFERENT attribute of the same subject is not a replacement. \
"Runs Postgres 16" and "runs Redis for rate limiting" are both true.
  * A different subject is not a replacement, however similar the wording \
or the topic.
  * More detail about the same thing is not a replacement. "Deploys on \
Heroku" and "deploys on Heroku with two dynos" agree.
  * Being in the same subject area is NOT a replacement. Two facts about \
one deployment, one codebase or one person usually both hold.
  * A restatement in other words IS a replacement, but ONLY when the new \
statement carries everything the old one did. A shorter, vaguer version of \
the same fact replaces nothing: "affiliated with Reakon Labs" does NOT \
replace "Principal Engineer and co-founder of Reakon Labs since May 2026", \
it just says less.

  * A statement that CONFIRMS the old value replaces nothing. "Redis is \
unchanged at 45 dollars" does NOT replace "Redis costs 45 dollars" -- it says \
the old statement is still true.
  * Mentioning the same THING is not replacing a fact about it. "We are not \
moving off Redis" does not replace "Redis backs the rate limiter".

Replacing hides the old statement from search, so when in doubt, do not.

New statement:
{statement}

Existing statements:
{candidates}

Reply with a JSON array and nothing else. Include ONLY the ones the new \
statement replaces; reply with [] if it replaces none.

For each one, name the ATTRIBUTE that changed and quote BOTH values. If you \
cannot fill `old_value` and `new_value` with two DIFFERENT values, it is not \
a replacement and must not be listed.

[{{"n": <number>, "attribute": "<what changed>", "old_value": "<before>",
  "new_value": "<after>", "confidence": <0.0-1.0>}}]"""


#: A statement replacing another must not say strictly LESS than it.
#:
#: Measured on real data: "Krishna is affiliated with Reakon Labs" was applied
#: as superseding "Krishna is Principal Engineer and co-founder of Reakon Labs
#: Pvt. Ltd. since May 2026", hiding the specific fact behind the vague one.
#: The model called it a restatement, which it is -- a lossy one.
#:
#: This is a code guard rather than another prompt line because the direction
#: of information loss is a property of the two strings, and a rule that can
#: be checked should not be delegated to a judge that can be talked out of it.
#: How much shorter a statement must be before it counts as a summary.
_SUMMARY_LENGTH_RATIO = 0.7
#: How many content words `old` can lose before the loss is material, when
#: none of them is a number.
_MIN_LOST_TOKENS = 4


#: Shortest token worth comparing. Drops "a", "of", "is" and initials without
#: needing a stopword list, which would be one more list to keep extending.
_MIN_TOKEN_CHARS = 2


def _tokens(text: str) -> set[str]:
    """Content words, lowercased, stripped of punctuation and symbols.

    Punctuation is dropped by Unicode general category rather than matched by
    a character class, because the guards built on this compare a value quoted
    by a model against the memory it came from, and the model does not quote
    punctuation back consistently. "$45.00" and "45.00" have to tokenise the
    same or a real revision is rejected; "n/a" and "???" have to reduce to
    nothing or a placeholder is accepted as a value.

    A category test does that without an enumeration, and it costs one
    `unicodedata` lookup per character on strings that are one sentence long.
    """
    out: set[str] = set()
    for word in text.lower().split():
        clean = "".join(c for c in word if not unicodedata.category(c).startswith(("P", "S")))
        if len(clean) > _MIN_TOKEN_CHARS:
            out.add(clean)
    return out


def _is_less_specific(new: str, old: str) -> bool:
    """True when `new` is a lossy summary of `old`, which may never replace it.

    Measured on real data: "Krishna is affiliated with Reakon Labs" was
    applied as superseding "Krishna is Principal Engineer and co-founder of
    Reakon Labs Pvt. Ltd. since May 2026", hiding the specific fact behind
    the vague one. The model called it a restatement, which it is -- a lossy
    one, and the direction of the loss is what makes it wrong.

    Token OVERLAP is the obvious test and it does not work: a summary
    paraphrases, so its own words ("affiliated") are absent from the original
    and coverage reads low. What identifies a summary is what the other
    statement KEEPS -- the dates, quantities and qualifiers that the shorter
    one dropped.

    So: substantially shorter, sharing a subject, and the discarded remainder
    contains either a number or several content words. Conservative by
    construction, because a false positive here only means a real revision
    goes unrecorded, while a false negative hides a true memory.

    A code guard rather than another prompt line: the direction of
    information loss is a property of the two strings, and a rule that can be
    checked should not be delegated to a judge that can be talked out of it.
    """
    new_tokens, old_tokens = _tokens(new), _tokens(old)
    if not new_tokens or not old_tokens:
        return False
    if not new_tokens & old_tokens:
        return False  # different subjects entirely; not a summary of anything
    if len(new_tokens) >= len(old_tokens) * _SUMMARY_LENGTH_RATIO:
        return False
    lost = old_tokens - new_tokens
    return any(t.isdigit() for t in lost) or len(lost) >= _MIN_LOST_TOKENS


@dataclass(frozen=True, slots=True)
class Verdict:
    """One adjudicated conflict."""

    memory_id: str
    reason: str
    confidence: float


def format_candidates(candidates: Sequence[tuple[str, str]]) -> str:
    """Number the candidates for the prompt.

    Numbered rather than keyed by id: a 30-character ULID in the output is
    30 chances to transpose a character into an id that exists, and a
    mistyped id would attach a real conflict to the wrong memory.
    """
    return "\n".join(f"{i + 1}. {text}" for i, (_, text) in enumerate(candidates))


def parse_verdicts(raw: str, candidates: Sequence[tuple[str, str]]) -> list[Verdict]:
    """Read the model's array back into verdicts, dropping anything doubtful.

    Every failure mode here resolves to "no conflict": an out-of-range index,
    a missing confidence, a number that is not a number. The cost of dropping
    a real conflict is that the memory stays unflagged, which is where it
    started.
    """
    verdicts: list[Verdict] = []
    seen: set[int] = set()
    for obj in _decode(raw):
        if not isinstance(obj, dict):
            continue
        try:
            index = int(obj["n"]) - 1
            confidence = float(obj.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= index < len(candidates) or index in seen:
            continue
        if confidence < MIN_CONFIDENCE:
            continue
        seen.add(index)
        reason = str(obj.get("reason") or "").strip() or "adjudicated conflict"
        verdicts.append(
            Verdict(
                memory_id=candidates[index][0],
                reason=reason[:120],
                confidence=min(confidence, 1.0),
            )
        )
    return verdicts


def _decode(raw: str) -> list[object]:
    text = raw.strip()
    if text.startswith("```"):
        # Fenced output despite the instruction. Strip the fence rather than
        # discard a good answer over formatting.
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Same salvage the extractor uses: a truncated array still holds
        # complete objects, and the alternative is reading "cut off" as
        # "nothing conflicts".
        return list(_salvage_objects(text))
    if isinstance(parsed, list):
        return list(parsed)
    if isinstance(parsed, dict):
        return [parsed]
    return []


async def adjudicate_contradictions(
    statement: str,
    candidates: Sequence[tuple[str, str]],
    complete: CompleteFn,
    *,
    timeout_s: float = 10.0,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Verdict]:
    """Which of `candidates` contradict `statement`, as judged by the model.

    `candidates` is (memory_id, content), pre-filtered by the caller to pairs
    that are topically close and that the lexical pass could not explain.
    Returns an empty list on any failure -- see the module docstring for why
    this one fails closed.
    """
    usable = [(mid, text) for mid, text in candidates if text.strip()][:max_candidates]
    if not statement.strip() or not usable:
        return []

    prompt = ADJUDICATE_PROMPT.format(
        statement=statement.strip(), candidates=format_candidates(usable)
    )
    try:
        raw = await asyncio.wait_for(complete(prompt), timeout=timeout_s)
    except TimeoutError:
        log.warning("adjudication_timeout", candidates=len(usable))
        return []
    except Exception as exc:
        log.warning("adjudication_failed", error=str(exc)[:160])
        return []

    verdicts = parse_verdicts(raw, usable)
    if verdicts:
        log.info("adjudicated_conflicts", found=len(verdicts), considered=len(usable))
    return verdicts


#: A quoted value may be at most this fraction of the statement it came from.
#:
#: When the model cannot isolate an attribute it echoes the whole sentence
#: into both slots -- "Redis status: Redis is unchanged at 45 dollars. -> The
#: user is NOT moving off Redis." Those pass a difference check trivially and
#: mean nothing: a value is a value, not a restatement of the claim. Measured
#: on a 10-session trace, this shape was 3 of the 5 surviving false
#: supersessions.
_MAX_VALUE_SHARE = 0.6


def _numbers(value: object) -> list[float]:
    """Every number in `value`, as numbers. Thousands separators removed."""
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))]


def _is_quotable(value: object, source: str) -> bool:
    """Whether `value` can actually be read out of `source`.

    THE RULE: a memory may not be hidden for a value it does not contain. The
    model is asked to quote the old value out of the memory it wants to
    replace, so a quote that is not in there is not a quote -- it is the model
    filling a required field, and the supersession rests on nothing.

    This replaces a thirteen-item list of placeholder words ("unknown", "n/a",
    "not stated", ...). The list worked on the cases it named and was blind to
    every one it did not: TBD, "???", "not recorded", "wasn't said",
    "unstated", "omitted", "the memory does not say". Enumerating the ways a
    model can admit it does not know is unbounded, and the enumeration is the
    wrong shape of solution -- what all of them have in common is that the
    quoted value is not in the text being quoted.

    It is also strictly stronger. On the same trace, it additionally rejects
    "primary database provider: Heroku Postgres -> Cloud SQL" against a memory
    reading "The user's primary database is Postgres 16 running on Cloud SQL
    in us-central1" -- both statements say Cloud SQL, so the old value was
    never in there and the "revision" was one fact restated with provenance.

    Numbers anchor, words corroborate: a value carrying a number needs that
    number present and one word to agree, while a value with no number has
    nothing but its words and needs all of them.

    Applied to `old_value` only. The NEW value is legitimately inferred --
    "the user cancelled the add-on" implies 0 dollars -- and requiring it to
    be quotable would reject real revisions.
    """
    tokens = _tokens(str(value))
    if not tokens:
        return False
    if numbers := _numbers(value):
        return set(numbers) <= set(_numbers(source)) and bool(tokens & _tokens(source))
    return tokens <= _tokens(source)


def _echoes_statement(value: object, statement: str) -> bool:
    """Whether a quoted value is just the statement copied back.

    A value is a value -- "45 dollars", "Heroku", "9 pages". When the model
    cannot isolate the attribute it pastes the whole sentence into the slot,
    which passes a difference check and asserts nothing.
    """
    text, source = str(value).strip(), statement.strip()
    if not text or not source:
        return False
    return len(text) >= len(source) * _MAX_VALUE_SHARE and (
        text[:40].lower() in source.lower() or source[:40].lower() in text.lower()
    )


def _same_value(left: object, right: object) -> bool:
    """Whether two quoted values say the same thing.

    Numbers are compared AS NUMBERS, because "$45.00" and "45 dollars" are one
    value wearing two formats and a string comparison calls them different --
    which would let a confirmation through as a replacement, the exact failure
    this check exists to stop.

    That numeric path is also why the currency and unit list this used to
    carry ("dollars", "usd", "eur", "per month") was doing nothing: it only
    ever ran when NEITHER side had a number, and "45 dollars a month" against
    "45 dollars" resolves on the numbers long before any word is compared. It
    was an English word list guarding a branch it could not reach.
    """
    left_n, right_n = _numbers(left), _numbers(right)
    if left_n and right_n:
        return left_n == right_n
    return _tokens(str(left)) == _tokens(str(right))


def parse_supersede_verdicts(
    raw: str,
    candidates: Sequence[tuple[str, str]],
    statement: str = "",
) -> list[Verdict]:
    """Read supersession verdicts, rejecting any that fail their own evidence.

    The model is made to name the attribute and quote BOTH values, and then the
    values are checked here. That is the whole mechanism: prose rules in the
    prompt did not hold, and the failures were all one shape -- a new statement
    that CONFIRMED or merely mentioned the old one being read as replacing it.

    Audited on a 10-session trace before this check existed: 5 of 9 applied
    supersessions were wrong, and the clearest was "Redis is unchanged at 45"
    hiding "Redis costs 45 dollars a month". A model that has to write
    old_value and new_value cannot express that as a replacement without
    writing the same value twice -- which is exactly what this rejects.
    """
    verdicts: list[Verdict] = []
    seen: set[int] = set()
    for obj in _decode(raw):
        if not isinstance(obj, dict):
            continue
        try:
            index = int(obj["n"]) - 1
            confidence = float(obj.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= index < len(candidates) or index in seen:
            continue
        if confidence < MIN_CONFIDENCE:
            continue

        old_value = obj.get("old_value")
        new_value = obj.get("new_value")
        if old_value is None or new_value is None:
            # Could not name what changed, so nothing demonstrably changed.
            log.info("supersede_rejected", why="no values", index=index)
            continue
        if not _is_quotable(old_value, candidates[index][1]):
            log.info(
                "supersede_rejected",
                why="old value not in the memory it would hide",
                index=index,
                value=str(old_value)[:40],
            )
            continue
        if _echoes_statement(old_value, candidates[index][1]) or _echoes_statement(
            new_value, statement
        ):
            log.info("supersede_rejected", why="echoed the statement", index=index)
            continue
        if _same_value(old_value, new_value):
            log.info("supersede_rejected", why="same value", value=str(old_value)[:40])
            continue

        seen.add(index)
        attribute = str(obj.get("attribute") or "attribute").strip()[:60]
        verdicts.append(
            Verdict(
                memory_id=candidates[index][0],
                reason=f"{attribute}: {old_value} -> {new_value}"[:120],
                confidence=min(confidence, 1.0),
            )
        )
    return verdicts


async def adjudicate_supersessions(
    statement: str,
    candidates: Sequence[tuple[str, str]],
    complete: CompleteFn,
    *,
    timeout_s: float = 10.0,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Verdict]:
    """Which of `candidates` the new statement actually replaces.

    Supersession is the only operation in this system that HIDES a memory,
    and it was deciding to do so from cosine similarity plus shared tags.
    Audited against a real corpus that hid four true facts out of five:

        "The API runs on Heroku with two web dynos."
          replaced by "The Python runtime is 3.13 on the heroku-24 stack."

    Both true, both about the same deployment, similarity 0.87, same tag.
    Embedding distance can say two statements are about the same AREA. It
    cannot say one supersedes the other, because that is a question about
    what changed, and nothing about the geometry encodes change.

    So the cosine proposer becomes a SHORTLIST and this makes the decision.
    Fails closed, like its contradiction sibling and more so: an unreachable
    vendor hides nothing, which is the safe direction when the alternative
    is deleting true memories from every future answer.
    """
    usable = [
        (mid, text)
        for mid, text in candidates
        if text.strip() and not _is_less_specific(statement, text)
    ][:max_candidates]
    if not statement.strip() or not usable:
        return []

    prompt = SUPERSEDE_PROMPT.format(
        statement=statement.strip(), candidates=format_candidates(usable)
    )
    try:
        raw = await asyncio.wait_for(complete(prompt), timeout=timeout_s)
    except TimeoutError:
        log.warning("supersede_adjudication_timeout", candidates=len(usable))
        return []
    except Exception as exc:
        log.warning("supersede_adjudication_failed", error=str(exc)[:160])
        return []

    verdicts = parse_supersede_verdicts(raw, usable, statement)
    log.info("adjudicated_supersessions", confirmed=len(verdicts), shortlisted=len(usable))
    return verdicts


__all__ = [
    "ADJUDICATE_PROMPT",
    "MAX_CANDIDATES",
    "MIN_CONFIDENCE",
    "SUPERSEDE_PROMPT",
    "Verdict",
    "adjudicate_contradictions",
    "adjudicate_supersessions",
    "format_candidates",
    "parse_supersede_verdicts",
    "parse_verdicts",
]
