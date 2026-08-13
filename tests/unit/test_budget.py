"""A documented negative, kept honest.

`budget.decide` is dead code by intent. Its own docstring says so: "MEASURED
RESULT: this does not work as specified. Keep it only as a documented negative."
Replayed offline over real ranked lists it bought delivery by taking THREE TIMES
more evidence -- the one move already measured harmful, since k=20 on LongMemEval
costs 8 questions to dilution.

So why test it at all. Two reasons, and neither is coverage:

**The tell must stay visible.** The evidence that the mechanism is inert is that
focused questions took exactly 16.0 documents on BOTH corpora -- a rule claiming
to read concentration off the result set cannot land on the same number for
12,000-character sessions and 400-character turns. That happens because
similarity scores are flat enough that "at least 80% of the top" admits nearly
everything. `test_a_flat_distribution_admits_nearly_everything` reproduces it in
four lines, so the finding survives without needing the offline replay.

**Dead code rots into live code.** A module with no callers and no tests is one
import away from being used by someone who reads the first half of the docstring.
`test_decide_still_has_no_callers` fails the moment it is wired up, which forces
whoever does it to read the measurement first.

The rest pins the arithmetic, so that if it IS ever revived the revival starts
from a known baseline rather than from a re-derivation.
"""

from __future__ import annotations

import pytest

from mapi.domain.retrieval.budget import (
    _CLIFF_BROAD,
    _CLIFF_FOCUSED,
    _FLOOR_BROAD,
    _FLOOR_FOCUSED,
    _HARD_CEILING,
    decide,
)

#: Classified DIRECT and ADVICE respectively -- the two focused shapes.
FOCUSED_Q = "what database do we use"
BROAD_Q = "how many invoices did I process in January"


# -- the finding that outlived the rule ------------------------------------


def test_a_flat_distribution_admits_nearly_everything() -> None:
    """The tell. This is why the mechanism is inert.

    Real similarity distributions are flat: twenty results inside a few percent
    of each other. "At least 80% of the top score" then admits all twenty, so the
    budget is set by how many candidates arrived rather than by where the evidence
    stops -- a fixed number wearing an adaptive costume.
    """
    flat = [0.90 - i * 0.001 for i in range(20)]
    assert decide(FOCUSED_Q, flat).take == 20


def test_a_steep_cliff_is_the_shape_the_rule_was_designed_for() -> None:
    """And it does work here -- which is the trap.

    On a synthetic distribution with a real cliff the rule looks excellent. The
    offline replay found that real distributions are not this shape, which is a
    thing no unit test can tell you and this docstring has to.
    """
    steep = [0.90, 0.88, 0.20, 0.19, 0.18]
    assert decide(FOCUSED_Q, steep).take == 2


def test_the_two_question_families_get_different_cliffs() -> None:
    """A focused question stops at the first real drop; an aggregate keeps
    taking comparable hits. The one part of the design that survived contact:
    the two corpora need OPPOSITE policies."""
    scores = [1.0, 0.70, 0.60, 0.50, 0.40]
    focused = decide(FOCUSED_Q, scores)
    broad = decide(BROAD_Q, scores)
    assert focused.cliff_ratio == _CLIFF_FOCUSED
    assert broad.cliff_ratio == _CLIFF_BROAD
    assert broad.take > focused.take


# -- the arithmetic --------------------------------------------------------


def test_nothing_retrieved_takes_nothing() -> None:
    budget = decide(FOCUSED_Q, [])
    assert budget.take == 0
    assert budget.reason == "nothing retrieved"


@pytest.mark.parametrize("top", [0.0, -0.5])
def test_unusable_scores_fall_back_to_the_floor(top: float) -> None:
    """A zero or negative top score means the ratio is meaningless -- `top * 0.8`
    is not a threshold when top is 0. Take the floor rather than everything or
    nothing."""
    budget = decide(FOCUSED_Q, [top, top, top, top])
    assert budget.take == _FLOOR_FOCUSED
    assert budget.reason == "no usable scores"


def test_the_floor_protects_a_question_that_needs_corroboration() -> None:
    """A steep cliff must not starve an aggregate down to one result."""
    steep = [1.0, 0.10, 0.09, 0.08, 0.07, 0.06]
    assert decide(BROAD_Q, steep).take == _FLOOR_BROAD


def test_the_floor_cannot_exceed_what_was_retrieved() -> None:
    """`min(floor, len(scores))` -- a budget larger than the result set would
    have the caller slice past the end."""
    assert decide(BROAD_Q, [1.0, 0.05]).take == 2


def test_the_ceiling_guards_a_pathological_distribution() -> None:
    """Not a tuning knob. A perfectly flat 500-result set must not become a
    500-document prompt."""
    flat = [0.5] * (_HARD_CEILING + 50)
    assert decide(BROAD_Q, flat).take == _HARD_CEILING


def test_a_single_result_is_taken() -> None:
    assert decide(FOCUSED_Q, [0.9]).take == 1


def test_the_threshold_is_inclusive() -> None:
    """A score exactly at `top * cliff` counts. Pinned because the boundary
    decides one result, and one result is the difference between the floor and
    the cliff on a short list."""
    at_threshold = [1.0, _CLIFF_FOCUSED]
    assert decide(FOCUSED_Q, at_threshold).take == 2
    below = [1.0, _CLIFF_FOCUSED - 0.01, 0.1]
    assert decide(FOCUSED_Q, below).take == _FLOOR_FOCUSED


def test_the_reason_names_the_shape_and_the_numbers() -> None:
    """It goes into an explain trace, so it has to be readable on its own."""
    budget = decide(BROAD_Q, [0.9, 0.8, 0.7])
    assert "broad question" in budget.reason
    assert "55%" in budget.reason
    assert "0.900" in budget.reason


def test_focused_is_reported_so_a_caller_can_see_the_classification() -> None:
    assert decide(FOCUSED_Q, [0.9]).focused is True
    assert decide(BROAD_Q, [0.9]).focused is False


def test_scores_past_the_ceiling_are_never_examined() -> None:
    """The loop is bounded before the floor and ceiling clamps, so a million
    scores cost the same as a hundred."""
    scores = [1.0] * 50 + [0.0] * 10_000
    assert decide(BROAD_Q, scores).take == 50


# -- the guard that matters ------------------------------------------------


def test_decide_still_has_no_callers() -> None:
    """Dead by intent. This test is what makes reviving it deliberate.

    A module with no callers and no tests is one import away from being used by
    someone who read the first half of a docstring and not the MEASURED RESULT
    paragraph. If `decide` is wired up, this fails, and whoever wired it has to
    come here and read why it was left alone.

    Searched as source text rather than by import graph, because an import inside
    a function body would not show up any other way.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    callers: list[str] = []
    for path in (root / "src").rglob("*.py"):
        if path.name == "budget.py":
            continue
        text = path.read_text()
        if re.search(r"\bfrom\b.*\bbudget\b import|\bbudget\.decide\b|\bdecide\(", text):
            callers.append(str(path.relative_to(root)))
    assert not callers, (
        f"budget.decide is now referenced by {callers}. Read the MEASURED RESULT "
        "paragraph in budget.py before keeping this: replayed offline it bought "
        "delivery by taking 3x more evidence, and k=20 on LongMemEval already "
        "cost 8 questions to dilution. If reviving it is right, delete this test "
        "in the same commit and say why."
    )
