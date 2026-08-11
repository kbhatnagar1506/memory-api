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

Replacing hides the old statement from search, so when in doubt, do not.

New statement:
{statement}

Existing statements:
{candidates}

Reply with a JSON array and nothing else. Include ONLY the ones the new \
statement replaces; reply with [] if it replaces none.

[{{"n": <number>, "reason": "<six words or fewer>", "confidence": <0.0-1.0>}}]"""


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


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


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

    verdicts = parse_verdicts(raw, usable)
    log.info(
        "adjudicated_supersessions", confirmed=len(verdicts), shortlisted=len(usable)
    )
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
    "parse_verdicts",
]
