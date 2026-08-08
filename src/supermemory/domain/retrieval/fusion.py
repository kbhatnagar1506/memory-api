"""Rank fusion.

Vector search and lexical search fail in different, complementary ways. Vector
search misses exact identifiers ("error TS2345", an order number, a person's
surname) because those carry little semantic signal. Lexical search misses
paraphrase. Fusing them beats either alone on almost every corpus.

Reciprocal Rank Fusion is the default because it combines *ranks*, not scores.
Cosine similarity and BM25 are on incomparable scales, and any attempt to blend
the raw numbers ends up silently dominated by whichever has the larger variance.
RRF sidesteps the calibration problem entirely and is remarkably hard to beat.

    score(d) = sum over lists L of  1 / (k + rank_L(d))

`k` damps the contribution of top ranks; the literature's default of 60 means a
document must do well in several lists to outrank one that is first in a single
list.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RankedList:
    """One retrieval strategy's opinion, best first."""

    name: str
    ids: Sequence[str]
    #: Raw per-id scores, retained only for explanation.
    scores: dict[str, float] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"{self.name}: weight must be non-negative")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"{self.name}: duplicate ids in a ranked list")


@dataclass(slots=True)
class FusedItem:
    id: str
    score: float
    #: list name -> 1-based rank in that list
    ranks: dict[str, int] = field(default_factory=dict)
    #: list name -> that list's raw score
    raw: dict[str, float] = field(default_factory=dict)

    def explain(self) -> list[str]:
        parts = [f"{name} rank {rank}" for name, rank in sorted(self.ranks.items())]
        parts.append(f"fused {self.score:.5f}")
        return parts


def reciprocal_rank_fusion(
    lists: Iterable[RankedList],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[FusedItem]:
    """Fuse ranked lists. Deterministic: ties break on id for stable paging."""
    if k < 1:
        raise ValueError("rrf k must be >= 1")

    accumulated: dict[str, FusedItem] = {}
    for ranked in lists:
        if ranked.weight == 0:
            continue
        for position, doc_id in enumerate(ranked.ids, start=1):
            item = accumulated.get(doc_id)
            if item is None:
                item = FusedItem(id=doc_id, score=0.0)
                accumulated[doc_id] = item
            item.score += ranked.weight / (k + position)
            item.ranks[ranked.name] = position
            if doc_id in ranked.scores:
                item.raw[ranked.name] = ranked.scores[doc_id]

    ordered = sorted(accumulated.values(), key=lambda i: (-i.score, i.id))
    return ordered[:limit] if limit is not None else ordered


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    """Normalize to [0,1]. A flat list maps to 1.0, not to a divide-by-zero."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def weighted_score_fusion(
    lists: Iterable[RankedList],
    *,
    limit: int | None = None,
) -> list[FusedItem]:
    """Alternative fusion: min-max normalize each list, then weighted-sum.

    Kept because it preserves score *margins* — the gap between a 0.95 and a
    0.55 match — which RRF discards. Useful when one strategy is known to be far
    stronger than another. Sensitive to outliers, hence not the default.
    """
    materialized = [ranked for ranked in lists if ranked.weight != 0]
    accumulated: dict[str, FusedItem] = {}
    total_weight = sum(r.weight for r in materialized) or 1.0

    for ranked in materialized:
        source = ranked.scores or {
            doc_id: 1.0 / position
            for position, doc_id in enumerate(ranked.ids, start=1)
        }
        normalized = _min_max(source)
        for position, doc_id in enumerate(ranked.ids, start=1):
            item = accumulated.get(doc_id)
            if item is None:
                item = FusedItem(id=doc_id, score=0.0)
                accumulated[doc_id] = item
            item.score += ranked.weight * normalized.get(doc_id, 0.0) / total_weight
            item.ranks[ranked.name] = position
            if doc_id in ranked.scores:
                item.raw[ranked.name] = ranked.scores[doc_id]

    ordered = sorted(accumulated.values(), key=lambda i: (-i.score, i.id))
    return ordered[:limit] if limit is not None else ordered


__all__ = [
    "DEFAULT_RRF_K",
    "FusedItem",
    "RankedList",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
