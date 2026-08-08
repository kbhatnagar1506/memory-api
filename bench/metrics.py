"""Retrieval metrics, scored against labelled evidence.

No LLM anywhere in this file, so every number is exactly reproducible and
carries no judge variance.

`full_recall_at_k` is the headline, not `hit_at_k`, and that correction came
from our own data. On LoCoMo multi-hop questions (mean 3.1 evidence turns):

    ANY evidence found  = 74%      <- what hit@k measures
    ALL evidence found  = 22%      <- what the answer actually needs
    end-to-end accuracy = 38%

Accuracy tracks full recall, not hit rate, because answering a three-hop
question with one of three facts is a wrong answer. Reporting hit@k as the
headline meant optimising a metric that did not predict our own outcome.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of evidence present in the top k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def full_recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 only if EVERY evidence item is in the top k. The headline metric."""
    if not relevant:
        return 0.0
    return 1.0 if relevant.issubset(set(retrieved[:k])) else 0.0


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if ANY evidence is in the top k. Reported, but not the headline:
    it overstates readiness on multi-evidence questions."""
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first evidence item."""
    if not relevant:
        return 0.0
    for position, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG. Rewards ranking evidence high, not merely including it."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Sane at 0/n and n/n, unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


__all__ = [
    "full_recall_at_k",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "wilson",
]
