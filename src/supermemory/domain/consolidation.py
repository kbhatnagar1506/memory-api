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

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .embeddings.base import Vector, cosine_similarity
from .models import Memory, MemoryStatus, Relation, RelationType, content_hash


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


def detect_exact_duplicate(
    content: str, existing: Sequence[Memory]
) -> DuplicateVerdict:
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
    "no longer", "instead of", "replaced", "replaces", "superseded", "updated",
    "correction", "actually", "changed to", "moved to", "now uses", "reverted",
    "cancelled", "canceled", "deprecated", "rescinded",
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
        if memory.id == new_memory.id:
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
        from datetime import timezone

        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def apply_supersession(
    new_memory: Memory,
    old_memory: Memory,
    proposal: SupersessionProposal,
) -> tuple[Memory, Memory]:
    """Record the relation on both sides and mark the old memory superseded.

    Returns updated copies; nothing is mutated in place, so a caller that fails
    to persist one side has not corrupted the other.
    """
    if new_memory.id == old_memory.id:
        raise ValueError("a memory cannot supersede itself")

    already = any(
        r.type is RelationType.SUPERSEDES and r.target_id == old_memory.id
        for r in new_memory.relations
    )
    new_relations = list(new_memory.relations)
    if not already:
        new_relations.append(
            Relation(
                type=RelationType.SUPERSEDES,
                target_id=old_memory.id,
                reason=proposal.reason,
                confidence=proposal.confidence,
            )
        )

    updated_new = new_memory.model_copy(
        update={"relations": new_relations, "version": new_memory.version + 1}
    )
    updated_old = old_memory.model_copy(
        update={
            "status": MemoryStatus.SUPERSEDED,
            "version": old_memory.version + 1,
            "updated_at": new_memory.updated_at,
        }
    )
    return updated_new, updated_old


def merge_duplicate(existing: Memory, incoming: Memory) -> Memory:
    """Fold a duplicate write into the memory already stored.

    Metadata and tags union; the newer event time wins; content is left alone
    because the two are equivalent by definition. The version bumps so a caller
    holding an ETag learns something changed.
    """
    merged_meta = {**existing.metadata, **incoming.metadata}
    merged_tags = list(dict.fromkeys([*existing.tags, *incoming.tags]))
    occurred = max(
        existing.occurred_at, incoming.occurred_at, key=_as_naive
    )
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
    "DuplicateKind",
    "DuplicateVerdict",
    "SupersessionProposal",
    "apply_supersession",
    "detect_exact_duplicate",
    "detect_near_duplicate",
    "merge_duplicate",
    "propose_supersessions",
]
