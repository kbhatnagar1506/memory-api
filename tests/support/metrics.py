"""Retrieval metrics over gold ids. Pure functions, no model, no store.

`bench/metrics.py` already has `recall_at_k`, `full_recall_at_k`, `mrr`,
`ndcg_at_k` and `wilson`, and those are the ones the benchmark harness reports.
This module is not a replacement -- it adds the three the capability families
need and `bench` has no reason to carry, and it is importable from the test
suite without dragging in the harness.

The additions, and why each exists:

  * **`joint_recall_at_k`** -- 1.0 only when EVERY required gold id is in the
    top k. Same idea as `bench`'s `full_recall_at_k`, restated here so the
    capability suite does not depend on `bench/` (which is not an installed
    package). This is the metric the multi-needle literature is about: plain
    `recall@k` of 0.9 on a 10-fact question reads like near-success and is a
    wrong answer, because a count over 9 of 10 is simply incorrect.
  * **`sufficient_at_k`** -- joint recall renamed for the question it answers:
    "could a perfect reader have got this right from what we returned?" Google's
    *Sufficient Context* work found 45.2% of real RAG instances insufficient,
    with models then hallucinating 15-40% instead of abstaining. Tracking it
    separates a retrieval bug from a reading bug in every incident, and it costs
    zero model calls.
  * **`unique_information_at_k`** -- how many DISTINCT gold facts are in the top
    k, not how many rows. Near-duplicate flooding returns k copies of one fact
    and scores a perfect `recall@k`; this is the metric that notices.
  * **`effective_x`** -- NoLiMa's definition, generalized: the largest sweep
    point at which a metric still holds at least `ratio` of its baseline. Turns
    a decay curve into one reportable number ("effective corpus size 3000").
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _prefix(retrieved: Sequence[str], k: int) -> set[str]:
    if k < 0:
        raise ValueError("k must not be negative")
    return set(retrieved[:k])


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of gold ids present in the top k. 0.0 when nothing is relevant."""
    gold = set(relevant)
    if not gold:
        return 0.0
    return len(_prefix(retrieved, k) & gold) / len(gold)


def joint_recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 only if EVERY gold id is in the top k. Conjunctive, deliberately.

    The metric `recall@k` hides: a question needing four facts that gets three
    scores 0.75 and is answered wrong. There is no partial credit for an
    incomplete evidence set, so the metric should not offer any.
    """
    gold = set(relevant)
    if not gold:
        return 0.0
    return 1.0 if gold <= _prefix(retrieved, k) else 0.0


def sufficient_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> bool:
    """Whether a perfect reader could have answered from the top k.

    Identical arithmetic to `joint_recall_at_k`, different question, and worth a
    separate name: this one is reported per hop count, where it localizes the
    failure ("2-hop sufficiency 0.94, 3-hop 0.41").
    """
    return joint_recall_at_k(retrieved, relevant, k) == 1.0


def unique_information_at_k(retrieved: Sequence[str], fact_of: dict[str, str], k: int) -> int:
    """Distinct FACTS in the top k, not distinct rows.

    `fact_of` maps memory id -> gold fact id, so ten near-duplicates of one
    fact count once. Without this, a store that floods top-k with paraphrases of
    a single memory scores a perfect recall while returning one fact's worth of
    information.
    """
    return len({fact_of[mid] for mid in retrieved[:k] if mid in fact_of})


def rank_of(retrieved: Sequence[str], target: str) -> int | None:
    """1-based rank, or None when absent. 1-based because ranks are read aloud."""
    for i, mid in enumerate(retrieved, start=1):
        if mid == target:
            return i
    return None


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first gold hit, over the whole list."""
    gold = set(relevant)
    if not gold:
        return 0.0
    for i, mid in enumerate(retrieved, start=1):
        if mid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG. Ideal DCG is over `min(len(relevant), k)` positions."""
    gold = set(relevant)
    if not gold or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(pos + 1)
        for pos, mid in enumerate(retrieved[:k], start=1)
        if mid in gold
    )
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def effective_x(
    measurements: Sequence[tuple[float, float]], *, ratio: float = 0.85
) -> float | None:
    """The largest sweep point still holding `ratio` of the baseline.

    `measurements` is `[(x, metric)]`; the baseline is the metric at the
    SMALLEST x, which is the regime the system is expected to be best in.
    Returns None when even the baseline point fails its own threshold, which
    means the sweep has no usable region and reporting a number would be
    misleading.

    From NoLiMa, which used it to show that models advertising 32K contexts had
    an *effective* length of 2K-8K. The same gap exists for a memory store, and
    this is how it gets a number instead of a vibe.
    """
    if not measurements:
        return None
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be within (0, 1]")
    ordered = sorted(measurements, key=lambda pair: pair[0])
    baseline = ordered[0][1]
    threshold = baseline * ratio
    best: float | None = None
    for x, value in ordered:
        if value >= threshold:
            best = x
        else:
            # Stop at the first failure: the metric is expected to decay
            # monotonically, and honouring a later recovery would report a
            # cliff as if it had not happened.
            break
    return best


__all__ = [
    "effective_x",
    "joint_recall_at_k",
    "mrr",
    "ndcg_at_k",
    "rank_of",
    "recall_at_k",
    "sufficient_at_k",
    "unique_information_at_k",
]
