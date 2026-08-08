"""Benchmark context assembly v2: merging, compaction, pruning."""

from __future__ import annotations

import pytest
from bench.context import ELLIPSIS, merge_intervals, render_context

from supermemory.domain.text import analyze

TURNS = [
    ("2023-05-01", "A: turn zero"),
    ("2023-05-01", "B: turn one"),
    ("2023-05-01", "A: turn two about kafka"),
    ("2023-05-02", "B: turn three"),
    ("2023-05-02", "A: turn four"),
    ("2023-05-02", "B: turn five"),
    ("2023-05-03", "A: turn six"),
    ("2023-05-03", "B: turn seven"),
]


# -- merging -------------------------------------------------------------------


def test_overlapping_intervals_merge() -> None:
    assert merge_intervals([(0, 4), (2, 6)]) == [(0, 6)]


def test_adjacent_intervals_merge() -> None:
    """Contiguous turns must not get a block separator between them."""
    assert merge_intervals([(0, 2), (3, 5)]) == [(0, 5)]


def test_disjoint_intervals_stay_separate() -> None:
    assert merge_intervals([(0, 1), (4, 5)]) == [(0, 1), (4, 5)]


def test_gap_bridges_a_single_missing_turn() -> None:
    assert merge_intervals([(0, 1), (3, 4)], gap=1) == [(0, 4)]
    assert merge_intervals([(0, 1), (3, 4)], gap=0) == [(0, 1), (3, 4)]


def test_merge_is_order_insensitive() -> None:
    assert merge_intervals([(4, 6), (0, 2), (5, 7)]) == [(0, 2), (4, 7)]


def test_merge_rejects_inverted_and_negative_gap() -> None:
    with pytest.raises(ValueError, match="inverted"):
        merge_intervals([(3, 1)])
    with pytest.raises(ValueError, match="non-negative"):
        merge_intervals([(0, 1)], gap=-1)


def test_merge_empty() -> None:
    assert merge_intervals([]) == []


# -- rendering -----------------------------------------------------------------


def test_each_turn_emitted_exactly_once_after_merge() -> None:
    """The v1 duplication bug: overlapping windows repeated turns."""
    merged = merge_intervals([(0, 3), (2, 5)])
    text = render_context(TURNS, merged, centers={1, 4})
    assert text.count("turn two about kafka") == 1
    assert text.count("turn three") == 1


def test_date_header_only_on_change() -> None:
    text = render_context(TURNS, [(0, 4)], centers={2})
    assert text.count("[2023-05-01]") == 1
    assert text.count("[2023-05-02]") == 1


def test_blocks_separated_by_marker() -> None:
    text = render_context(TURNS, [(0, 1), (6, 7)], centers={0, 7})
    assert "\n---\n" in text


def test_pruning_keeps_center_adjacency_regardless_of_terms() -> None:
    keep = set(analyze("kafka"))
    text = render_context(TURNS, [(0, 5)], centers={2}, keep_terms=keep, analyze=analyze)
    # +-1 of the center survives even with zero term overlap.
    assert "turn one" in text
    assert "turn three" in text
    # Distant, no-overlap turns are pruned to an ellipsis.
    assert "turn five" not in text
    assert ELLIPSIS in text


def test_pruning_keeps_term_matches_far_from_center() -> None:
    keep = set(analyze("turn five"))
    text = render_context(TURNS, [(0, 5)], centers={2}, keep_terms=keep, analyze=analyze)
    assert "turn five" in text


def test_pruned_run_collapses_to_single_ellipsis() -> None:
    keep = set(analyze("zzz-no-match"))
    text = render_context(TURNS, [(0, 7)], centers={0}, keep_terms=keep, analyze=analyze)
    # turns 2..7 all pruned -> exactly one marker, not six.
    assert text.count(ELLIPSIS) == 1


def test_keep_terms_without_analyzer_rejected() -> None:
    with pytest.raises(ValueError, match="analyze"):
        render_context(TURNS, [(0, 1)], centers=set(), keep_terms={"x"})


def test_no_pruning_by_default() -> None:
    text = render_context(TURNS, [(0, 7)], centers=set())
    for _, content in TURNS:
        assert content in text
    assert ELLIPSIS not in text
