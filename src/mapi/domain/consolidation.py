"""Consolidation: deduplication and belief revision.

A memory store that only appends is wrong within a week of real use. The same
fact arrives twice from two sources; a decision is reversed; a preference
changes. Without consolidation, retrieval returns the stale answer next to the
current one and the agent has no way to tell which is which.

Three mechanisms, cheapest first:

  1. **Exact duplicate** — normalized content hash. Free, catches re-ingestion
     of the same document, which is by far the most common case.
  2. **Near duplicate** — cosine similarity at or above a threshold. Catches
     reformatting and light paraphrase.
  3. **Supersession** — a new memory that covers the same subject as an older
     one and conflicts with it marks the older one superseded.

Only (1) and (2) run automatically. Supersession is proposed, not applied,
unless the caller opts in: silently hiding a user's data because a similarity
score crossed a threshold is a bad default, and the failure mode (the true fact
is hidden, the stale one survives) is invisible until someone notices the wrong
answer. `apply_supersession` exists for callers who want it; the API exposes it
per-request and per-space.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .embeddings.base import Vector, cosine_similarity
from .models import (
    Memory,
    MemoryStatus,
    RelationEdge,
    RelationType,
    content_hash,
)


class DuplicateKind(StrEnum):
    NONE = "none"
    EXACT = "exact"
    NEAR = "near"


@dataclass(frozen=True, slots=True)
class DuplicateVerdict:
    kind: DuplicateKind
    existing_id: str | None = None
    similarity: float = 0.0

    @property
    def is_duplicate(self) -> bool:
        return self.kind is not DuplicateKind.NONE


@dataclass(frozen=True, slots=True)
class SupersessionProposal:
    """A claim that `new_id` replaces `old_id`, with why."""

    new_id: str
    old_id: str
    similarity: float
    reason: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ContradictionProposal:
    """A claim that two memories cannot both be true, with why.

    Distinct from supersession, and the distinction is the point. Supersession
    is REVISION: a newer memory replaces an older one, and time orders them.
    Contradiction is DISAGREEMENT: two memories conflict and nothing about
    their timestamps says which wins — the user said "the meeting is Tuesday"
    and "the meeting is Thursday" in the same hour, or two sources disagree.

    Neither memory is marked superseded. A contradicted memory stays ACTIVE
    because it may be the true one, and hiding either would answer the
    question with silence. What retrieval does instead is SURFACE the conflict:
    every competitor resolves it invisibly (newest timestamp wins), which is
    indistinguishable from having no conflict at all. An agent told "these two
    disagree" can ask; an agent handed the winner cannot.
    """

    left_id: str
    right_id: str
    similarity: float
    reason: str
    confidence: float


def detect_exact_duplicate(content: str, existing: Sequence[Memory]) -> DuplicateVerdict:
    """Match on normalized content hash: whitespace and case are not meaning."""
    digest = content_hash(content)
    for memory in existing:
        if memory.status is MemoryStatus.ARCHIVED:
            continue
        if memory.content_sha256 == digest:
            return DuplicateVerdict(DuplicateKind.EXACT, memory.id, 1.0)
    return DuplicateVerdict(DuplicateKind.NONE)


#: Tokens that distinguish one record from another: numbers, identifiers,
#: and anything carrying a digit. Two texts differing in these are different
#: facts however close their embeddings sit.
_DISTINCTIVE = re.compile(r"\b\w*\d[\w.-]*\b")


def differs_materially(left: str, right: str) -> bool:
    """Whether two texts state different FACTS, not merely different words.

    Near-duplicate detection exists to collapse restatements, and it decides
    from cosine similarity alone. That is safe for prose and destructive for
    records, because an embedding barely moves when one digit changes:

        "component number 2: service-2"
        "component number 3: service-3"      cosine 0.980

    Measured on the live service, eight distinct facts written in that shape
    became four -- half the corpus silently discarded at write time, with the
    caller told "duplicate". Any corpus of invoices, SKUs, log lines, ticket
    numbers or contact records is mostly pairs like that.

    So a duplicate must also agree on the tokens that carry identity. This is
    deliberately a NARROW test -- numbers and identifiers only -- because its
    job is to veto a merge, and a veto that fires too often merely keeps two
    copies of a restatement, which costs a row. The failure it prevents costs
    a fact.
    """
    return set(_DISTINCTIVE.findall(left.lower())) != set(_DISTINCTIVE.findall(right.lower()))


def detect_near_duplicate(
    embedding: Vector,
    candidates: Sequence[tuple[str, Vector]],
    *,
    threshold: float = 0.97,
    text: str = "",
    texts: Mapping[str, str] | None = None,
) -> DuplicateVerdict:
    """Highest-similarity candidate at or above `threshold`, if any.

    When `text` and `texts` are supplied, a candidate is additionally
    required not to differ materially -- see `differs_materially` for the
    half-a-corpus this loses without it. They are optional so the pure
    similarity behaviour stays available and testable.
    """
    if not embedding or not candidates:
        return DuplicateVerdict(DuplicateKind.NONE)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")

    best_id: str | None = None
    best_sim = -1.0
    for memory_id, vector in candidates:
        if len(vector) != len(embedding):
            # A dimension change means the index was built by a different
            # model. Skip rather than crash; the caller re-embeds.
            continue
        sim = cosine_similarity(embedding, vector)
        if texts is not None and text and differs_materially(text, texts.get(memory_id, "")):
            # Close in embedding space, different in the tokens that carry
            # identity. Not a duplicate, whatever the cosine says.
            continue
        if sim > best_sim:
            best_sim, best_id = sim, memory_id

    if best_id is not None and best_sim >= threshold:
        return DuplicateVerdict(DuplicateKind.NEAR, best_id, best_sim)
    return DuplicateVerdict(DuplicateKind.NONE, None, max(best_sim, 0.0))


#: Similarity band in which two memories are about the same subject but are not
#: restatements of each other. Below it they are unrelated; above it they are
#: duplicates, handled by dedup rather than supersession.
SUPERSEDE_LOW = 0.72
SUPERSEDE_HIGH = 0.97

#: Surface markers of a corrected or reversed statement. Deliberately a weak
#: signal: it raises confidence, it never decides on its own.
_REVISION_MARKERS = (
    "no longer",
    "instead of",
    "replaced",
    "replaces",
    "superseded",
    "updated",
    "correction",
    "actually",
    "changed to",
    "moved to",
    "now uses",
    "reverted",
    "cancelled",
    "canceled",
    "deprecated",
    "rescinded",
)
#: English negation is a CLOSED grammatical class. That is what makes this an
#: inventory rather than a list somebody has to keep extending: not, the
#: contracted n't, and the negative quantifiers are the whole of it, and no
#: further phrasing can be added because the language has no more.
#:
#: This used to be seven substrings tested with `in`, which had two defects.
#: "stopped" was in it -- a lexical verb, not a negator, so "stopped by the
#: store" read as a negation while "gave up", "quit" and "no longer" did not.
#: Cessation is open-class semantics and belongs to the adjudicator, which
#: already receives every pair this cannot explain. And substring testing has
#: no word boundary, so the class is now matched as words.
_NEGATION = re.compile(
    r"\b(?:not|no|none|never|neither|nor|nobody|nothing|nowhere|cannot)\b|n't\b",
    re.IGNORECASE,
)


def propose_supersessions(
    new_memory: Memory,
    new_embedding: Vector,
    candidates: Sequence[tuple[Memory, Vector]],
    *,
    low: float = SUPERSEDE_LOW,
    high: float = SUPERSEDE_HIGH,
) -> list[SupersessionProposal]:
    """Propose that `new_memory` supersedes older, similar, conflicting memories.

    Only older memories are eligible: a memory cannot supersede its own future.
    Confidence is heuristic, and the caller decides what to do with it.
    """
    if not new_embedding:
        return []
    if low > high:
        raise ValueError("low must not exceed high")

    text = new_memory.content.casefold()
    has_marker = any(m in text for m in _REVISION_MARKERS)
    has_negation = _NEGATION.search(text) is not None

    proposals: list[SupersessionProposal] = []
    for memory, vector in candidates:
        if memory.id == new_memory.id or same_lineage(new_memory, memory):
            continue
        if memory.status is not MemoryStatus.ACTIVE:
            continue
        if len(memory.content) > _DOCUMENT_CHARS:
            # A DOCUMENT IS NOT A FACT, and cannot be replaced wholesale.
            #
            # A long memory asserts many things at once. A later memory that
            # revises ONE of them does not replace the rest, and supersession
            # is all-or-nothing: it hides the whole row and cascades
            # `_mark_derivations_stale` across everything extracted from it.
            #
            # Measured on a 10-session trace: two whole session records were
            # superseded because a single number inside them changed later, and
            # 8 supersessions removed 49 of 131 memories -- 37% of the corpus
            # -- from default search. One wrong verdict took 19 true claims
            # with it.
            #
            # `_is_less_specific` already blocks a short CLAIM from replacing a
            # long document. It does not block a long document from replacing
            # another long document: s07 at 822 chars against s03 at 1,123 is a
            # token ratio of 0.8, above that guard's threshold, so it reached
            # the adjudicator and was confirmed. Length is the signal the
            # adjudicator cannot see.
            #
            # A document that genuinely disagrees gets CONTRADICTS, which hides
            # nothing. Supersession stays available for the case it was built
            # for -- one fact replacing one fact.
            continue
        # Strictly older by event time. Equal timestamps are ambiguous, so we
        # decline rather than guess which one wins.
        if _as_naive(memory.occurred_at) >= _as_naive(new_memory.occurred_at):
            continue
        if len(vector) != len(new_embedding):
            continue

        sim = cosine_similarity(new_embedding, vector)
        if not (low <= sim < high):
            continue

        confidence = 0.35 + 0.45 * ((sim - low) / max(high - low, 1e-9))
        reasons = [f"same subject (similarity {sim:.2f})"]
        if has_marker:
            confidence += 0.15
            reasons.append("revision language in the new memory")
        if has_negation:
            confidence += 0.05
            reasons.append("negation in the new memory")
        if new_memory.tags and set(new_memory.tags) & set(memory.tags):
            confidence += 0.05
            reasons.append("shared tags")

        proposals.append(
            SupersessionProposal(
                new_id=new_memory.id,
                old_id=memory.id,
                similarity=sim,
                reason="; ".join(reasons),
                confidence=min(confidence, 0.99),
            )
        )

    proposals.sort(key=lambda p: (-p.confidence, p.old_id))
    return proposals


def _as_naive(value: datetime) -> float:
    """Comparable epoch seconds regardless of tzinfo awareness."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


def apply_supersession(
    new_memory: Memory,
    old_memory: Memory,
    proposal: SupersessionProposal,
) -> tuple[RelationEdge, Memory]:
    """Build the supersession edge and the updated (superseded) old memory.

    Returns the edge to persist and an updated copy of the old memory; nothing
    is mutated in place, so a caller that fails to persist one side has not
    corrupted the other. The NEW memory is not returned because it no longer
    changes: the relation lives on the edge, and attaching an edge is a fact
    about the graph, not an edit to the memory's own content.
    """
    if new_memory.id == old_memory.id:
        raise ValueError("a memory cannot supersede itself")

    edge = RelationEdge(
        org_id=new_memory.org_id,
        space_id=new_memory.space_id,
        source_id=new_memory.id,
        target_id=old_memory.id,
        type=RelationType.SUPERSEDES,
        reason=proposal.reason,
        confidence=proposal.confidence,
    )
    updated_old = old_memory.model_copy(
        update={
            "status": MemoryStatus.SUPERSEDED,
            "version": old_memory.version + 1,
            "updated_at": new_memory.updated_at,
        }
    )
    return edge, updated_old


def merge_duplicate(existing: Memory, incoming: Memory) -> Memory:
    """Fold a duplicate write into the memory already stored.

    Metadata and tags union; the newer event time wins; content is left alone
    because the two are equivalent by definition. The version bumps so a caller
    holding an ETag learns something changed.
    """
    merged_meta = {**existing.metadata, **incoming.metadata}
    merged_tags = list(dict.fromkeys([*existing.tags, *incoming.tags]))
    occurred = max(existing.occurred_at, incoming.occurred_at, key=_as_naive)
    return existing.model_copy(
        update={
            "metadata": merged_meta,
            "tags": merged_tags,
            "occurred_at": occurred,
            "updated_at": incoming.updated_at,
            "version": existing.version + 1,
            "source": incoming.source or existing.source,
        }
    )


__all__ = [
    "SUPERSEDE_HIGH",
    "SUPERSEDE_LOW",
    "ContradictionProposal",
    "DuplicateKind",
    "DuplicateVerdict",
    "SupersessionProposal",
    "apply_supersession",
    "detect_exact_duplicate",
    "detect_near_duplicate",
    "differs_materially",
    "merge_duplicate",
    "propose_contradictions",
    "propose_supersessions",
    "same_lineage",
    "unexplained_pairs",
]


#: Contradiction needs a TIGHTER similarity floor than supersession. Two
#: memories that merely share a topic are not in conflict; they have to be
#: about the same specific claim before disagreeing about it is meaningful.
CONTRADICT_LOW = 0.82

#: Stems whose presence on opposite sides of two similar statements is evidence
#: they conflict.
#:
#: Antonymy is irreducibly lexical -- there is no structural property of two
#: strings that makes them opposites, so this is a lexicon and stays one. What
#: it is NOT allowed to be is wrong, and it was: the pairs were tested with
#: `in`, so they matched inside longer words. Measured on ten innocent pairs,
#: nine were reported as conflicting:
#:
#:     "runs on Python 3.13"    / "the flag is off"    on   inside pyth-ON
#:     "dislikes the new one"   / "dislikes the old"   like inside dis-LIKE-s
#:     "Backups upload nightly" / "the download job"   up   inside UP-load
#:     "The window seat"        / "they close at ten"  win  inside WIN-dow
#:     "Yesterday it was paid"  / "there is no invoice" yes inside YES-terday
#:     "The seller confirmed"   / "we will buy two"    sell inside SELL-er
#:
#: A false CONTRADICTS edge is the expensive direction here, and worse, these
#: never reach the adjudicator: `unexplained_pairs` only escalates what the
#: lexical pass could NOT explain, so a spurious match writes an edge with no
#: model review at all. The `dislike` row is the sharpest -- two statements
#: that AGREE, both saying dislike, flagged as opposites because one word
#: contains the other.
#:
#: Five pairs are also gone, and word boundaries are not what saves them:
#: `before`/`after` and `more`/`less` are relational terms that co-occur
#: innocently and mean nothing without a shared dimension; `up`/`down` and
#: `on`/`off` are too polysemous to carry a claim even matched as whole words,
#: because "runs ON Python" and "the flag is OFF" are both correct uses of
#: those words and are not a disagreement; and `always`/`never`, `yes`/`no`
#: are already the negation class above, where they were counted twice.
_ANTONYM_STEMS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("accept", "reject"),
    ("accept", "decline"),
    ("approve", "deny"),
    ("enable", "disable"),
    ("start", "stop"),
    ("open", "close"),
    ("win", "lose"),
    ("buy", "sell"),
    ("hire", "fire"),
    ("add", "remove"),
    ("include", "exclude"),
    ("agree", "disagree"),
    ("like", "dislike"),
    ("prefer", "avoid"),
    ("confirm", "cancel"),
)

#: Suffixes a stem may carry and still be the same term. Doubled-consonant
#: forms are here because "cancelled" and "stopped" are the shapes these stems
#: actually appear in, and a lexicon that cannot match its own entries in
#: running text is a lexicon that does nothing.
_INFLECTIONS = r"(?:s|es|d|ed|ing|led|ling|ped|ping)?"


def _term(stem: str) -> re.Pattern[str]:
    """A stem, word-bounded, tolerating inflection.

    Stems of two characters take a plural at most: applying the full suffix set
    to "on" would match "ones", which is how this class of bug starts.
    """
    suffix = _INFLECTIONS if len(stem) > 2 else r"s?"
    return re.compile(rf"\b{stem}{suffix}\b", re.IGNORECASE)


_ANTONYMS: tuple[tuple[str, str, re.Pattern[str], re.Pattern[str]], ...] = tuple(
    (positive, negative, _term(positive), _term(negative))
    for positive, negative in _ANTONYM_STEMS
)

#: Above this, a memory is treated as a DOCUMENT rather than a single fact,
#: and is never superseded. Chosen well above a restated fact ("I live in
#: Madrid now" is 20 chars, a spend line ~45) and well below a session (the
#: trace corpus averaged 1,026). Anything in between is ambiguous, and the
#: safe reading of ambiguity here is "do not hide it".
_DOCUMENT_CHARS = 500

#: Numbers and dates are the highest-signal disagreement: two near-identical
#: sentences differing only in a figure are almost always in conflict.
_FIGURE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\b")
_WEEKDAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)


def _polarity_conflict(left: str, right: str) -> str | None:
    """A negation or antonym present on one side and absent on the other."""
    if bool(_NEGATION.search(left)) != bool(_NEGATION.search(right)):
        return "one statement is negated and the other is not"
    for positive, negative, positive_re, negative_re in _ANTONYMS:
        left_has = positive_re.search(left), negative_re.search(left)
        right_has = positive_re.search(right), negative_re.search(right)
        # One side asserts the positive term, the other the negative term --
        # and neither side asserts BOTH, which is what a statement discussing
        # the change itself does ("moved the flag from on to off"). Reading
        # that as half of a disagreement is how a changelog entry ends up
        # contradicting the thing it describes.
        if left_has[0] and left_has[1]:
            continue
        if right_has[0] and right_has[1]:
            continue
        if (left_has[0] and right_has[1]) or (left_has[1] and right_has[0]):
            return f"opposing terms ({positive}/{negative})"
    return None


def _figure_conflict(left: str, right: str) -> str | None:
    """Same claim, different number or weekday."""
    left_figures, right_figures = set(_FIGURE.findall(left)), set(_FIGURE.findall(right))
    if left_figures and right_figures and not (left_figures & right_figures):
        return "same subject, different figures"
    left_days = {d.casefold() for d in _WEEKDAY.findall(left)}
    right_days = {d.casefold() for d in _WEEKDAY.findall(right)}
    if left_days and right_days and not (left_days & right_days):
        return "same subject, different days"
    return None


def same_lineage(left: Memory, right: Memory) -> bool:
    """Whether two memories are facets of a single original statement.

    Write-time extraction turns one paragraph into several atomic claims, and
    every one of them is near-identical to its parent and to its siblings --
    they are the same sentence viewed at different resolutions. Feeding those
    pairs to the revision checks produces nonsense in both directions, and it
    did, on real data:

        "Krishna is on an F-1 visa."
          flagged as CONTRADICTING
        "Krishna is on an F-1 visa. He needs CPT authorization ..."

        "Krishna is affiliated with Reakon Labs."
          flagged as SUPERSEDING
        "Krishna is Principal Engineer and co-founder of Reakon Labs since
         May 2026."

    The second is the dangerous one: a vaguer claim hid the specific fact it
    was extracted from. Nothing about similarity can catch this, because the
    similarity is real -- the pair genuinely IS about the same thing. What
    disqualifies it is PROVENANCE, which the `extracted_from` metadata records
    exactly.

    A claim never contradicts or replaces the passage it came from, and two
    claims from one passage never contradict or replace each other. They are
    joined by `derived_from` and by association instead, which is what those
    edges are for.
    """
    left_parent = (left.metadata or {}).get("extracted_from")
    right_parent = (right.metadata or {}).get("extracted_from")
    if left_parent and left_parent == right.id:
        return True
    if right_parent and right_parent == left.id:
        return True
    return bool(left_parent) and left_parent == right_parent


def unexplained_pairs(
    new_memory: Memory,
    new_embedding: Vector,
    candidates: Sequence[tuple[Memory, Vector]],
    *,
    low: float = CONTRADICT_LOW,
    high: float = SUPERSEDE_HIGH,
) -> list[tuple[Memory, float]]:
    """Pairs close enough to conflict that the lexical signals could not judge.

    The same similarity gate as `propose_contradictions`, minus the memories
    it already explained. These are the interesting ones: topically almost
    identical, and yet no flipped negation, no antonym, no differing figure.
    Either they agree -- which is the common case, and why the caller must
    treat this as candidates rather than conflicts -- or they disagree in a
    way seven strings cannot express.

    Pure and synchronous like the rest of this module. Judging them needs a
    model, and a model needs I/O, so that decision belongs to the caller.
    """
    if not new_embedding:
        return []
    if low > high:
        raise ValueError("low must not exceed high")

    out: list[tuple[Memory, float]] = []
    for memory, vector in candidates:
        if memory.id == new_memory.id or same_lineage(new_memory, memory):
            continue
        if memory.status is not MemoryStatus.ACTIVE:
            continue
        if len(vector) != len(new_embedding):
            continue
        similarity = cosine_similarity(new_embedding, vector)
        if not (low <= similarity < high):
            continue
        if _polarity_conflict(new_memory.content, memory.content):
            continue
        if _figure_conflict(new_memory.content, memory.content):
            continue
        out.append((memory, similarity))
    out.sort(key=lambda pair: (-pair[1], pair[0].id))
    return out


def propose_contradictions(
    new_memory: Memory,
    new_embedding: Vector,
    candidates: Sequence[tuple[Memory, Vector]],
    *,
    low: float = CONTRADICT_LOW,
    high: float = SUPERSEDE_HIGH,
) -> list[ContradictionProposal]:
    """Propose that `new_memory` conflicts with existing memories.

    Requires BOTH high topical similarity and a concrete disagreement signal —
    a flipped negation, opposing terms, or a differing figure/day. Similarity
    alone is not evidence of conflict; it is evidence of relatedness, and
    treating one as the other is how a paraphrase gets flagged as a
    contradiction. A false contradiction erodes trust faster than a missed one,
    so the bar is deliberately high and the output is a PROPOSAL.

    Unlike supersession this does NOT require the candidate to be older.
    Disagreement has no direction: two statements made in the same minute can
    conflict, and that is precisely the case supersession cannot express.
    """
    if not new_embedding:
        return []
    if low > high:
        raise ValueError("low must not exceed high")

    proposals: list[ContradictionProposal] = []
    for memory, vector in candidates:
        if memory.id == new_memory.id or same_lineage(new_memory, memory):
            continue
        if memory.status is not MemoryStatus.ACTIVE:
            continue
        if len(vector) != len(new_embedding):
            continue
        similarity = cosine_similarity(new_embedding, vector)
        if not (low <= similarity < high):
            continue

        reasons = [
            r
            for r in (
                _polarity_conflict(new_memory.content, memory.content),
                _figure_conflict(new_memory.content, memory.content),
            )
            if r
        ]
        if not reasons:
            continue

        # Two independent signals is much stronger evidence than one.
        confidence = min(0.45 + 0.2 * (similarity - low) / max(high - low, 1e-9), 0.95)
        if len(reasons) > 1:
            confidence = min(confidence + 0.2, 0.95)
        proposals.append(
            ContradictionProposal(
                left_id=new_memory.id,
                right_id=memory.id,
                similarity=similarity,
                reason=f"same subject (similarity {similarity:.2f}); " + "; ".join(reasons),
                confidence=confidence,
            )
        )

    proposals.sort(key=lambda p: (-p.confidence, p.right_id))
    return proposals
