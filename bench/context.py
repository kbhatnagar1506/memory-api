"""Context assembly for the LoCoMo benchmark (v2).

v1 built one window per retrieved turn and concatenated them. Two measured
wastes: overlapping windows emitted the same turns twice (hits cluster, so
overlap is common), and every turn restated its date bracket. v2 merges
overlapping intervals so each turn appears exactly once, and emits a date
header only when the date changes. Identical information, fewer tokens —
deduplication can only remove exact repetition, so the quality risk is zero
by construction.

Optional extractive compression (`keep_terms`) prunes neighbour turns that
share no analyzed term with the query or any hit turn. That one CAN cost
accuracy — narrative glue sometimes matters — so it is a separate flag,
measured separately, never silently on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

#: Marks a pruned run of turns. One token spent to preserve the signal that
#: the dialogue is discontinuous there — silently stitching non-adjacent turns
#: together invites the answer model to invent causality between them.
ELLIPSIS = "..."


def merge_intervals(
    intervals: Sequence[tuple[int, int]], *, gap: int = 0
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent [lo, hi] integer intervals.

    Adjacent means contiguous turns (next lo == hi + 1): keeping them separate
    would print a block separator between consecutive dialogue lines, which is
    noise. `gap` widens that tolerance — gap=1 also bridges a single missing
    turn (adding it) rather than splitting the narrative.
    """
    if gap < 0:
        raise ValueError("gap must be non-negative")
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for lo, hi in ordered:
        if hi < lo:
            raise ValueError(f"interval ({lo}, {hi}) is inverted")
        if merged and lo <= merged[-1][1] + 1 + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def render_context(
    turns: Sequence[tuple[str, str]],
    merged: Sequence[tuple[int, int]],
    centers: set[int],
    *,
    keep_terms: set[str] | None = None,
    analyze: Callable[[str], list[str]] | None = None,
    keep_adjacent: int = 1,
) -> str:
    """Render merged intervals as dialogue blocks.

    `turns` is the full ordered conversation as (date, content) pairs. A date
    header line is emitted only when the date changes — dialogue runs for many
    turns inside one session, and per-turn date brackets were pure repetition.

    With `keep_terms`, non-center turns inside a block are pruned unless they
    are within `keep_adjacent` of a center (the immediate conversational
    context of a hit is always kept) or share at least one analyzed term with
    `keep_terms`. Pruned runs leave an ELLIPSIS marker.
    """
    if keep_terms is not None and analyze is None:
        raise ValueError("keep_terms requires an analyze function")

    blocks: list[str] = []
    for lo, hi in merged:
        lines: list[str] = []
        last_date: str | None = None
        pruning = False
        for index in range(lo, hi + 1):
            date, content = turns[index]
            keep = True
            if keep_terms is not None:
                near_center = any(abs(index - c) <= keep_adjacent for c in centers)
                if not near_center:
                    assert analyze is not None
                    keep = bool(set(analyze(content)) & keep_terms)
            if not keep:
                if not pruning:
                    lines.append(ELLIPSIS)
                    pruning = True
                continue
            pruning = False
            if date != last_date:
                lines.append(f"[{date}]")
                last_date = date
            lines.append(content)
        blocks.append("\n".join(lines))
    return "\n---\n".join(blocks)


__all__ = ["ELLIPSIS", "merge_intervals", "render_context"]
