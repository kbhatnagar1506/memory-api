"""Retrieval confidence: computed from evidence, not asked of a model.

Measured motivation: on 500 questions the system failed in BOTH calibration
directions at once — 24 questions declined while the evidence sat in the
delivered context, and 8 unanswerable questions answered anyway. A single
accuracy number hides both, and asking the answering model how sure it is
measures its tone rather than the data.
"""

from __future__ import annotations

from mapi.domain.retrieval.confidence import (
    ConfidenceLevel,
    RefusalReason,
    assess,
)


def test_nothing_retrieved_is_none_and_names_correct_silence() -> None:
    """Declining because the space holds nothing is CORRECT behaviour, not a
    memory failure. The refusal reason is what lets a caller tell them apart."""
    c = assess([])
    assert c.level is ConfidenceLevel.NONE
    assert c.refusal_reason is RefusalReason.NO_RELEVANT_MEMORY
    assert c.is_answerable is False


def test_strong_separated_match_is_high() -> None:
    c = assess([0.81, 0.42, 0.30])
    assert c.level is ConfidenceLevel.HIGH
    assert c.refusal_reason is None
    assert c.is_answerable is True


def test_a_flat_distribution_is_low_however_high_the_top_score() -> None:
    """Ten results all scoring 0.80 means the retriever found a neighbourhood,
    not an answer — the exact shape that produces a confident wrong answer."""
    c = assess([0.80, 0.79, 0.78, 0.77])
    assert c.level is ConfidenceLevel.LOW
    assert c.refusal_reason is RefusalReason.WEAK_EVIDENCE
    assert "no clear best match" in c.reason


def test_weak_top_score_is_low() -> None:
    c = assess([0.18, 0.05])
    assert c.level is ConfidenceLevel.LOW
    assert c.refusal_reason is RefusalReason.WEAK_EVIDENCE


def test_conflicting_evidence_is_low_regardless_of_scores() -> None:
    """A perfect match to two memories that contradict each other is the worst
    case, not the best: answering picks a side silently."""
    c = assess([0.95, 0.40], has_conflicts=True)
    assert c.level is ConfidenceLevel.LOW
    assert c.refusal_reason is RefusalReason.CONFLICTING_EVIDENCE


def test_a_single_result_is_not_treated_as_perfectly_separated() -> None:
    """With one result there is nothing to be flat against, so margin is 0.0 —
    an absence of evidence about separation, not evidence of separation."""
    c = assess([0.75])
    assert c.margin == 0.0
    assert c.level is ConfidenceLevel.HIGH  # judged on the top score alone


def test_moderate_match_is_medium() -> None:
    c = assess([0.44, 0.20])
    assert c.level is ConfidenceLevel.MEDIUM
    assert c.refusal_reason is None


def test_raw_signals_are_returned_for_recalibration() -> None:
    """Callers with a different risk tolerance must be able to re-threshold
    without re-running the search."""
    c = assess([0.62, 0.31, 0.10])
    assert c.top_score == 0.62
    assert round(c.margin, 4) == 0.31
    assert c.n_results == 3
