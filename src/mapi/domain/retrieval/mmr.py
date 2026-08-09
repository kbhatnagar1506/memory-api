"""Maximal Marginal Relevance: relevance traded against redundancy.

Pure top-k over a memory store returns the same fact five times, phrased five
ways, because a corpus that has accumulated over months contains near-duplicates
by construction. That wastes the agent's context window on repetition.

MMR selects greedily:

    next = argmax over candidates of
             lambda * relevance(d) - (1 - lambda) * max similarity(d, selected)

lambda = 1.0 is plain relevance; lambda = 0.0 is pure diversity. The default of
0.7 leans toward relevance while suppressing outright restatements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..embeddings.base import Vector, cosine_similarity


@dataclass(frozen=True, slots=True)
class MMRCandidate:
    id: str
    relevance: float
    embedding: Vector | None = None


@dataclass(frozen=True, slots=True)
class MMRSelection:
    id: str
    relevance: float
    #: Highest similarity to anything already selected, at selection time.
    redundancy: float
    mmr_score: float
    rank: int


def _normalize_relevance(candidates: Sequence[MMRCandidate]) -> dict[str, float]:
    """Put relevance on [0,1] so lambda trades against similarity on one scale."""
    if not candidates:
        return {}
    values = [c.relevance for c in candidates]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {c.id: 1.0 for c in candidates}
    span = hi - lo
    return {c.id: (c.relevance - lo) / span for c in candidates}


def maximal_marginal_relevance(
    candidates: Sequence[MMRCandidate],
    *,
    limit: int,
    lambda_: float = 0.7,
) -> list[MMRSelection]:
    """Greedy MMR. O(limit * n) similarity evaluations.

    Candidates without an embedding are treated as maximally distinct: we cannot
    prove they are redundant, and dropping them would lose recall.
    """
    if limit <= 0:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be within [0, 1]")
    if not candidates:
        return []

    seen: set[str] = set()
    unique: list[MMRCandidate] = []
    for c in candidates:
        if c.id in seen:
            continue
        seen.add(c.id)
        unique.append(c)

    normalized = _normalize_relevance(unique)
    remaining = {c.id: c for c in unique}
    selected: list[MMRSelection] = []
    selected_vectors: list[Vector] = []

    while remaining and len(selected) < limit:
        best_id: str | None = None
        best_score = float("-inf")
        best_redundancy = 0.0

        for cid, cand in remaining.items():
            redundancy = 0.0
            if cand.embedding is not None and selected_vectors:
                redundancy = max(cosine_similarity(cand.embedding, v) for v in selected_vectors)
            score = lambda_ * normalized[cid] - (1.0 - lambda_) * redundancy
            # Ties break on id so paging is stable across identical requests.
            if score > best_score or (
                score == best_score and (best_id is None or cid < best_id)
            ):
                best_id, best_score, best_redundancy = cid, score, redundancy

        assert best_id is not None  # remaining is non-empty
        chosen = remaining.pop(best_id)
        selected.append(
            MMRSelection(
                id=chosen.id,
                relevance=chosen.relevance,
                redundancy=best_redundancy,
                mmr_score=best_score,
                rank=len(selected) + 1,
            )
        )
        if chosen.embedding is not None:
            selected_vectors.append(chosen.embedding)

    return selected


__all__ = ["MMRCandidate", "MMRSelection", "maximal_marginal_relevance"]
