"""Statistics. These are the numbers that decide whether the product exists."""

from __future__ import annotations

import math

import pytest

from mapi_ablation.score import (
    accuracy,
    best_text_arm,
    cluster_bootstrap_ci,
    cluster_failures,
    cost_table,
    mcnemar,
    outcome_counts,
)


def row(arm, qid, outcome, seed=0, model="m", cond="cold", **kw):
    base = {
        "arm": arm, "qid": qid, "outcome": outcome, "seed": seed,
        "model_key": model, "condition": cond, "n_nodes": 40,
        "qtype": "root_cause", "question": "q", "expected": "N01",
        "parsed": "N01" if outcome == "correct" else "N02",
        "input_tokens": 100, "latency_ms": 10.0, "raw_text": "N01",
    }
    base.update(kw)
    return base


def test_accuracy_basic():
    rows = [row("a", "1", "correct"), row("a", "2", "wrong"),
            row("a", "3", "unparseable"), row("a", "4", "error")]
    assert accuracy(rows) == 0.25
    assert outcome_counts(rows) == {
        "correct": 1, "wrong": 1, "unparseable": 1, "error": 1
    }


def test_unparseable_is_not_counted_as_correct():
    assert accuracy([row("a", "1", "unparseable")]) == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    rows = [row("a", str(i), "correct" if i % 4 else "wrong", seed=i % 5)
            for i in range(100)]
    ci = cluster_bootstrap_ci(rows, resamples=2000)
    assert ci.lo <= ci.point <= ci.hi
    assert 0.0 <= ci.lo and ci.hi <= 1.0
    assert ci.n == 100


def test_bootstrap_is_deterministic():
    rows = [row("a", str(i), "correct" if i % 3 else "wrong", seed=i % 4)
            for i in range(60)]
    a = cluster_bootstrap_ci(rows, resamples=1000)
    b = cluster_bootstrap_ci(rows, resamples=1000)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_bootstrap_refuses_to_invent_an_interval_from_one_cluster():
    """One seed means a cluster bootstrap is undefined; say so, don't fake it."""
    rows = [row("a", str(i), "correct", seed=0) for i in range(30)]
    ci = cluster_bootstrap_ci(rows, resamples=500)
    assert ci.point == 1.0
    assert math.isnan(ci.lo) and math.isnan(ci.hi)


def test_clustered_bootstrap_is_wider_than_naive_would_be():
    """Correlated-within-seed data must not produce a falsely tight interval."""
    # Every query in a seed shares its outcome: maximal within-cluster
    # correlation, so the true uncertainty is about the 4 seeds, not the 80 rows.
    rows = []
    for s in range(4):
        for i in range(20):
            rows.append(row("a", f"{s}-{i}", "correct" if s < 2 else "wrong", seed=s))
    ci = cluster_bootstrap_ci(rows, resamples=4000)
    assert ci.point == pytest.approx(0.5)
    # Resampling 4 all-or-nothing clusters must admit values far from 0.5.
    assert ci.lo <= 0.1 and ci.hi >= 0.9


def test_mcnemar_pairs_only_identical_questions():
    a = [row("canvas", "q1", "correct"), row("canvas", "q2", "correct")]
    b = [row("json", "q1", "wrong"), row("json", "q3", "correct")]
    res = mcnemar(a, b, "canvas", "json")
    assert res.n_pairs == 1  # q2/q3 are unpaired and must be dropped
    assert res.b_count == 1 and res.c_count == 0


def test_mcnemar_symmetric_case_is_not_significant():
    a = [row("canvas", f"q{i}", "correct" if i < 5 else "wrong") for i in range(10)]
    b = [row("json", f"q{i}", "wrong" if i < 5 else "correct") for i in range(10)]
    res = mcnemar(a, b, "canvas", "json")
    assert res.b_count == 5 and res.c_count == 5
    assert res.p_value == pytest.approx(1.0)


def test_mcnemar_lopsided_case_is_significant():
    a = [row("canvas", f"q{i}", "correct") for i in range(12)]
    b = [row("json", f"q{i}", "wrong") for i in range(12)]
    res = mcnemar(a, b, "canvas", "json")
    assert res.b_count == 12 and res.c_count == 0
    assert res.p_value < 0.001


def test_mcnemar_ignores_concordant_pairs():
    """Both right or both wrong carries no information about a difference."""
    a = [row("canvas", f"q{i}", "correct") for i in range(50)]
    b = [row("json", f"q{i}", "correct") for i in range(50)]
    res = mcnemar(a, b, "canvas", "json")
    assert res.discordant == 0
    assert res.p_value == 1.0


def test_no_discordant_pairs_gives_p_of_one():
    assert mcnemar([], [], "a", "b").p_value == 1.0


def test_best_text_arm_is_chosen_empirically():
    rows = (
        [row("prose", f"p{i}", "wrong") for i in range(10)]
        + [row("json", f"j{i}", "correct" if i < 5 else "wrong") for i in range(10)]
        + [row("layout_text", f"l{i}", "correct" if i < 8 else "wrong")
           for i in range(10)]
    )
    assert best_text_arm(rows, ("prose", "json", "layout_text")) == "layout_text"


def test_cost_table_tokens_per_correct_penalises_a_cheap_wrong_arm():
    cheap_wrong = [row("a", str(i), "wrong", input_tokens=10) for i in range(10)]
    dear_right = [row("b", str(i), "correct", input_tokens=100) for i in range(10)]
    t = cost_table(cheap_wrong + dear_right)
    assert t["a"]["tokens_per_correct"] is None  # nothing correct at any price
    assert t["b"]["tokens_per_correct"] == 100.0


def test_cost_table_reports_none_rather_than_zero_when_nothing_is_correct():
    t = cost_table([row("a", "1", "wrong", input_tokens=10)])
    assert t["a"]["tokens_per_correct"] is None


def test_failure_clustering_separates_unparseable_from_wrong():
    fails = [
        {"outcome": "unparseable", "pixel_dist": 10, "hops": 1, "t_gap": None,
         "crosses_domain": False},
        {"outcome": "wrong", "pixel_dist": 1200, "hops": 1, "t_gap": None,
         "crosses_domain": False},
        {"outcome": "wrong", "pixel_dist": 50, "hops": 4, "t_gap": None,
         "crosses_domain": False},
    ]
    buckets = cluster_failures(fails)
    assert buckets["answered in prose rather than an ID"] == 1
    assert buckets["referenced nodes far apart on canvas (>900px)"] == 1
    assert buckets["long causal chain (>=3 hops)"] == 1
