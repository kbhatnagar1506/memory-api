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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .embeddings.base import Vector, cosine_similarity
from .models import Memory, MemoryStatus, RelationEdge, RelationType, content_hash


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


def detect_near_duplicate(
    embedding: Vector,
    candidates: Sequence[tuple[str, Vector]],
    *,
    threshold: float = 0.97,
) -> DuplicateVerdict:
    """Highest-similarity candidate at or above `threshold`, if any."""
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
_NEGATIONS = ("not ", "never ", "won't", "will not", "cannot", "can't", "stopped")


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
    has_negation = any(n in text for n in _NEGATIONS)

    proposals: list[SupersessionProposal] = []
    for memory, vector in candidates:
        if memory.id == new_memory.id or same_lineage(new_memory, memory):
            continue
        if memory.status is not MemoryStatus.ACTIVE:
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

#: Pairs whose presence on opposite sides of two similar statements is
#: evidence they conflict. Ordered longest-first inside each pair so
#: "not going" is not read as containing "going".
_ANTONYMS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("accept", "reject"),
    ("accepted", "declined"),
    ("approve", "deny"),
    ("enable", "disable"),
    ("start", "stop"),
    ("open", "closed"),
    ("before", "after"),
    ("more", "less"),
    ("always", "never"),
    ("yes", "no"),
    ("on", "off"),
    ("up", "down"),
    ("win", "lose"),
    ("buy", "sell"),
    ("hire", "fire"),
    ("add", "remove"),
    ("include", "exclude"),
    ("agree", "disagree"),
    ("like", "dislike"),
    ("prefer", "avoid"),
    ("confirmed", "cancelled"),
)

#: Numbers and dates are the highest-signal disagreement: two near-identical
#: sentences differing only in a figure are almost always in conflict.
_FIGURE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\b")
_WEEKDAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)


def _polarity_conflict(left: str, right: str) -> str | None:
    """A negation or antonym present on one side and absent on the other."""
    lower_left, lower_right = left.casefold(), right.casefold()
    left_neg = any(n in lower_left for n in _NEGATIONS)
    right_neg = any(n in lower_right for n in _NEGATIONS)
    if left_neg != right_neg:
        return "one statement is negated and the other is not"
    for positive, negative in _ANTONYMS:
        in_left = positive in lower_left, negative in lower_left
        in_right = positive in lower_right, negative in lower_right
        # One side asserts the positive term, the other the negative term.
        if (in_left[0] and in_right[1]) or (in_left[1] and in_right[0]):
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
