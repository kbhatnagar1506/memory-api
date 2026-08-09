"""Retrieval confidence: how much should an agent trust this result set?

Measured motivation. On 500 questions our system failed in BOTH calibration
directions at once: 24 questions were declined while the evidence was sitting
in the delivered context, and 8 unanswerable questions were answered anyway.
Those are opposite errors and a single accuracy number hides both.

The failures also have different shapes. Over-confidence concentrates on
aggregate questions — asked "how many museums did I visit in December" with no
museum visits retrieved, a model will still produce a number, because a count
is always producible. Under-confidence concentrates on direct lookups, where a
model that cannot find the phrasing it expects declines rather than infers.

So confidence is computed HERE, from retrieval evidence, rather than asked of
the model that is about to answer. A model's own certainty is a property of
its tone; the shape of the retrieved distribution is a property of the data.

No LLM, no extra round trip: every signal below is already computed by the
pipeline. The output is a coarse level plus the raw signals, deliberately not
a precise-looking float — a spurious 0.7413 would imply a calibration we have
not earned.

THRESHOLD HONESTY: `_STRONG_SCORE` and `_WEAK_SCORE` are absolute cosine
values and are therefore embedding-model dependent, and they are NOT fitted to
labelled data — they are structural defaults. The relative signals (margin
ratio, result count, conflicts) are scale-free and carry most of the weight
for exactly that reason. Fitting the absolute floors needs a labelled
score-distribution study; until then, treat HIGH/LOW as ordering, not as
probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: Cosine similarity above which a top hit is a strong lexical/semantic match.
#: Structural default, not fitted -- see the module docstring.
_STRONG_SCORE = 0.55
_WEAK_SCORE = 0.30
#: A top result that barely beats the runner-up means the retriever found a
#: neighbourhood, not an answer. Expressed as a RATIO of the top score so it
#: stays meaningful across embedding models and score scales.
_CLEAR_MARGIN_RATIO = 0.12


class ConfidenceLevel(StrEnum):
    #: Strong top match, clear separation, no contradictions. Safe to assert.
    HIGH = "high"
    MEDIUM = "medium"
    #: Weak, flat, or self-contradicting evidence. An agent should ask rather
    #: than assert, and an answer built on this deserves a hedge.
    LOW = "low"
    #: Nothing usable was retrieved. Declining here is CORRECT behaviour, not
    #: a memory failure -- the distinction the `reason` field exists to carry.
    NONE = "none"


class RefusalReason(StrEnum):
    #: The space has nothing matching at all. Not a system failure.
    NO_RELEVANT_MEMORY = "no_relevant_memory"
    #: Memories came back, but weakly and without separation.
    WEAK_EVIDENCE = "weak_evidence"
    #: Retrieved memories contradict each other; answering would pick a side
    #: silently. The one case where surfacing beats resolving.
    CONFLICTING_EVIDENCE = "conflicting_evidence"


@dataclass(frozen=True, slots=True)
class RetrievalConfidence:
    """A coarse trust level plus the raw signals that produced it.

    The signals are returned alongside the level on purpose: a caller with a
    different risk tolerance can re-threshold them without re-running the
    search, and a caller debugging a bad answer can see which signal was weak.
    """

    level: ConfidenceLevel
    top_score: float
    #: Absolute gap between the best and second-best result. 0.0 when only one
    #: result came back (no separation is measurable, not "perfect separation").
    margin: float
    n_results: int
    has_conflicts: bool
    reason: str
    #: Set only when `level` is NONE or LOW: why an answer may not be
    #: supportable. Lets a caller distinguish "we have nothing" (correct
    #: silence) from "we have something but it is thin" (a memory gap).
    refusal_reason: RefusalReason | None = None

    @property
    def is_answerable(self) -> bool:
        """Whether the evidence supports asserting an answer at all."""
        return self.level is not ConfidenceLevel.NONE


def assess(scores: list[float], *, has_conflicts: bool = False) -> RetrievalConfidence:
    """Grade a result set from its score distribution.

    `scores` must be ordered best-first, as the pipeline returns them.
    """
    if not scores:
        return RetrievalConfidence(
            level=ConfidenceLevel.NONE,
            top_score=0.0,
            margin=0.0,
            n_results=0,
            has_conflicts=False,
            reason="nothing retrieved",
            refusal_reason=RefusalReason.NO_RELEVANT_MEMORY,
        )

    top = scores[0]
    margin = top - scores[1] if len(scores) > 1 else 0.0
    # A single result cannot be "flat" -- there is nothing to be flat against.
    flat = len(scores) > 1 and margin < top * _CLEAR_MARGIN_RATIO

    if has_conflicts:
        return RetrievalConfidence(
            level=ConfidenceLevel.LOW,
            top_score=top,
            margin=margin,
            n_results=len(scores),
            has_conflicts=True,
            reason="retrieved memories contradict each other",
            refusal_reason=RefusalReason.CONFLICTING_EVIDENCE,
        )

    if top < _WEAK_SCORE:
        return RetrievalConfidence(
            level=ConfidenceLevel.LOW,
            top_score=top,
            margin=margin,
            n_results=len(scores),
            has_conflicts=False,
            reason=f"best match is weak ({top:.2f})",
            refusal_reason=RefusalReason.WEAK_EVIDENCE,
        )

    if flat:
        return RetrievalConfidence(
            level=ConfidenceLevel.LOW,
            top_score=top,
            margin=margin,
            n_results=len(scores),
            has_conflicts=False,
            reason=(
                f"no clear best match (top {top:.2f}, margin {margin:.2f}) — "
                "the retriever found a neighbourhood, not an answer"
            ),
            refusal_reason=RefusalReason.WEAK_EVIDENCE,
        )

    if top >= _STRONG_SCORE:
        return RetrievalConfidence(
            level=ConfidenceLevel.HIGH,
            top_score=top,
            margin=margin,
            n_results=len(scores),
            has_conflicts=False,
            reason=f"strong match ({top:.2f}) with clear separation ({margin:.2f})",
        )

    return RetrievalConfidence(
        level=ConfidenceLevel.MEDIUM,
        top_score=top,
        margin=margin,
        n_results=len(scores),
        has_conflicts=False,
        reason=f"moderate match ({top:.2f})",
    )


__all__ = ["ConfidenceLevel", "RefusalReason", "RetrievalConfidence", "assess"]
