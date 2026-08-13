"""The lab scorer, pinned against the three answers it used to grade wrong.

A broken scorer is worse than no scorer: it converts correct answers into fake
findings, and the format experiment spent eleven arms confirming three of them.
Every case below that carries a model quote is verbatim from that run.
"""

from __future__ import annotations

import pytest
from bench.lab.scoring import Expected, score

# -- the three answers the old scorer graded wrong -------------------------


def test_a_terse_yes_is_correct_when_yes_is_the_answer() -> None:
    """expected 'yes, 180 vs 210'; the model said 'Yes'. Wrong verdict, old
    scorer -- the prompt DEMANDS terseness, so punishing it measures the
    disagreement between prompt and scorer, not the system."""
    expected = Expected.parse("yes + 180 210")
    graded = score("Yes", expected)
    assert graded.correct
    assert graded.completeness == 0.0, "the figures are completeness, not correctness"

    fuller = score("Yes, 180 versus 210 a night.", expected)
    assert fuller.correct and fuller.completeness == 1.0


def test_a_comma_in_the_expectation_cannot_break_matching() -> None:
    """expected 'aisle, no red-eye' -- the token 'aisle,' with its comma matched
    nothing, ever. The model's 'aisle seat... avoid red-eye flights' was right."""
    expected = Expected.parse("aisle; red-eye|red eye")
    verbatim = "You should book an aisle seat and avoid red-eye flights."
    assert score(verbatim, expected).correct


def test_the_vendor_without_the_amount_is_correct_but_incomplete() -> None:
    """expected 'Datadog 890'; the model said 'Datadog'. Right answer to "what
    is this invoice for", missing the supporting figure -- two different
    measurements, now reported separately."""
    expected = Expected.parse("datadog + 890")
    graded = score("Datadog", expected)
    assert graded.correct
    assert graded.completeness == 0.0
    assert not graded.complete


# -- matching discipline ---------------------------------------------------


def test_numbers_match_on_word_boundaries() -> None:
    """'18' inside '180' is a WRONG number scored right -- the one direction a
    scorer must never err in."""
    expected = Expected.parse("18")
    assert not score("the answer is 180", expected).correct
    assert score("the answer is 18", expected).correct


def test_punctuation_and_case_are_not_meaning() -> None:
    expected = Expected.parse("1500; cloud sql")
    assert score("It costs $1,500 on Cloud SQL.", expected).correct


def test_alternatives_within_a_group() -> None:
    expected = Expected.parse("red-eye|red eye|redeye")
    for surface in ("a red-eye flight", "the red eye", "no redeye for me"):
        assert score(surface, expected).correct, surface


def test_every_group_is_required() -> None:
    expected = Expected.parse("aisle; window")
    graded = score("always the aisle", expected)
    assert not graded.correct
    assert graded.missing == ("window",)


def test_an_empty_answer_fails_with_everything_missing() -> None:
    expected = Expected.parse("yes + 180")
    graded = score("", expected)
    assert not graded.correct
    assert graded.completeness == 0.0


def test_a_spec_with_no_required_groups_is_an_authoring_error() -> None:
    """`+ 180` alone would accept ANY answer -- a floor of zero wearing a spec's
    clothes. Refused at parse time, where the author is looking."""
    with pytest.raises(ValueError, match="no required groups"):
        Expected.parse("+ 180 210")


def test_no_bonus_means_completeness_is_full() -> None:
    """A question with no supporting figures is complete when it is correct --
    reporting 0.0 would make terse questions look permanently half-answered."""
    graded = score("Priya", Expected.parse("priya"))
    assert graded.complete
