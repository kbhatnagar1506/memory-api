"""Recency weighting.

A memory system that ignores time returns last year's plan alongside this
week's. But hard time filters are worse: a two-year-old fact that is still the
only answer should still be findable. So recency is a *multiplier* on relevance,
never a filter.

Exponential decay with a configurable half-life:

    factor = floor + (1 - floor) * 0.5 ** (age_days / half_life_days)

The floor is what stops the multiplier from ever reaching zero — an old but
uniquely relevant memory keeps a meaningful share of its score instead of being
erased by arithmetic.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

DEFAULT_HALF_LIFE_DAYS = 180.0
DEFAULT_FLOOR = 0.25


def _as_utc(value: datetime) -> datetime:
    """Naive datetimes are assumed UTC rather than rejected.

    Mixed awareness is the single most common bug in time-handling code, and
    subtracting a naive from an aware datetime raises TypeError deep inside a
    scoring loop where the traceback explains nothing.
    """
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def age_days(occurred_at: datetime, *, now: datetime | None = None) -> float:
    reference = _as_utc(now or datetime.now(UTC))
    delta = reference - _as_utc(occurred_at)
    return max(0.0, delta.total_seconds() / 86_400.0)


def recency_factor(
    occurred_at: datetime,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    floor: float = DEFAULT_FLOOR,
) -> float:
    """Multiplier in [floor, 1.0]. Future timestamps are treated as now."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be within [0, 1]")

    days = age_days(occurred_at, now=now)
    if days > 36_500:  # a century; avoid a pointless underflow computation
        return floor
    decayed = math.pow(0.5, days / half_life_days)
    return floor + (1.0 - floor) * decayed


def apply_decay(
    score: float,
    occurred_at: datetime,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    floor: float = DEFAULT_FLOOR,
) -> tuple[float, float]:
    """Return (decayed_score, factor). Factor is surfaced for explainability."""
    factor = recency_factor(occurred_at, now=now, half_life_days=half_life_days, floor=floor)
    return score * factor, factor


__all__ = [
    "DEFAULT_FLOOR",
    "DEFAULT_HALF_LIFE_DAYS",
    "access_factor",
    "age_days",
    "apply_decay",
    "recency_factor",
]


#: How much a frequently-retrieved memory may be boosted, at most. Deliberately
#: small: reconsolidation is a tie-breaker among comparable results, not a
#: popularity contest that lets a much-read memory outrank a better match.
MAX_ACCESS_BOOST = 0.10
#: Accesses at which the boost is effectively saturated. Logarithmic, so the
#: first few retrievals matter and the thousandth does not.
ACCESS_SATURATION = 20.0


def access_factor(access_count: int, *, max_boost: float = MAX_ACCESS_BOOST) -> float:
    """Multiplier for how often a memory has actually been retrieved.

    Biological memory strengthens traces that get replayed, and every competing
    system implements that as *forgetting* — Supermemory's dream cycle prunes,
    Mem0 expires by `forgetAfter`, the research literature calls it
    interference-based decay. The one independent cost study (arXiv 2603.04814)
    is why we do not: lossy consolidation measurably loses recall, and a memory
    system that deletes on a heuristic fails invisibly.

    So retrieval frequency is a RANKING signal and nothing else. It can move a
    memory up a result list; it can never hide one, mark one stale, or delete
    one. The bound is small and the growth logarithmic so that a much-retrieved
    memory cannot outrank a genuinely better match — it only wins ties.

    Returns a multiplier in [1.0, 1.0 + max_boost].
    """
    if access_count <= 0:
        return 1.0
    # log1p(n)/log1p(saturation) reaches 1.0 at the saturation point and grows
    # slowly past it; clamped so an outlier cannot escape the bound.
    strength = min(math.log1p(access_count) / math.log1p(ACCESS_SATURATION), 1.0)
    return 1.0 + max_boost * strength
