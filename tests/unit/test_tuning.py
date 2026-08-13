"""Fusion parameters keyed on question shape.

These assert the SHAPE of the policy, not specific constants -- a test that
pins 1.3 would have to be rewritten the moment anyone tunes it, and would
stop meaning anything. What must hold is the reasoning: enumeration leans
lexical and flattens the curve, lookup leans vector.
"""

from __future__ import annotations

import pytest

from mapi.domain.retrieval.tuning import Fusion, fusion_for
from mapi.domain.synthesis.classify import QuestionKind


@pytest.mark.parametrize("kind", [QuestionKind.LIST_ALL, QuestionKind.COUNT])
def test_enumeration_leans_lexical_and_flattens_the_curve(kind: QuestionKind) -> None:
    """Every instance counts, and the surface form repeats."""
    f = fusion_for(kind)
    assert f.lexical > f.vector
    assert f.rrf_k > Fusion().rrf_k, "a flat curve is what lets the tail survive"


@pytest.mark.parametrize("kind", [QuestionKind.DIRECT, QuestionKind.ADVICE])
def test_lookup_leans_vector(kind: QuestionKind) -> None:
    """One right answer, phrased in the asker's words, not the memory's."""
    f = fusion_for(kind)
    assert f.vector > f.lexical


def test_advice_leans_hardest_on_paraphrase() -> None:
    """A preference is almost never worded like the request for it."""
    assert fusion_for(QuestionKind.ADVICE).vector > fusion_for(QuestionKind.DIRECT).vector


def test_every_kind_has_a_policy_or_the_default() -> None:
    for kind in QuestionKind:
        f = fusion_for(kind)
        assert f.vector > 0 and f.lexical > 0 and f.rrf_k >= 1


def test_the_weights_stay_small() -> None:
    """A fusion weight is not a boost.

    It scales an arm before ranks merge, so it shifts ordering at the margin.
    A stack of large multipliers is how a ranking layer stops being
    attributable -- and how a competitor's 2.0x silently made their default
    search return only one class of result for months.
    """
    for kind in QuestionKind:
        f = fusion_for(kind)
        assert 0.8 <= f.vector <= 1.5
        assert 0.8 <= f.lexical <= 1.5
