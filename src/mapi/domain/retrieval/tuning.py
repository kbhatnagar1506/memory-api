"""Fusion parameters that vary with what the question is asking for.

One set of weights served every question: `vector_weight` 1.0,
`lexical_weight` 1.0, `rrf_k` 60, for a lookup and an enumeration alike. That
is a reasonable default and a poor answer to two questions with opposite
needs:

    "what did I name my dog"       one right memory exists. Precision.
    "how many times did I visit"   every instance counts. Breadth.

A lookup wants the vector arm, which is good at "the same thing said
differently". An enumeration wants the lexical arm, which is good at "this
exact surface form, everywhere it appears" -- and a larger `rrf_k`, because
small k concentrates score in the first few ranks and enumeration needs the
tail.

WHERE THESE NUMBERS COME FROM, stated plainly because it matters: they are
reasoned from what each arm is good at, NOT fitted to a benchmark. Nothing
here was chosen by running LongMemEval and keeping what scored higher. That
distinction is the difference between a design decision and a number that
will not survive contact with somebody else's corpus -- and this module sits
next to `classify.py`, whose ADVICE patterns carry a contamination warning
for exactly the mistake this avoids.

The multipliers are deliberately small. A fusion weight is not a boost: it
scales an arm's contribution before ranks are merged, so 1.3 shifts ordering
at the margin rather than reordering wholesale. A ten-deep multiplicative
boost stack is how a ranking layer becomes unattributable, and the cost of
that is not hypothetical -- it is how a competitor's `compiled_truth 2.0x`
silently made their default search return only compiled truth for months.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..synthesis.classify import QuestionKind

#: RRF damping. Larger k flattens the curve, so rank 20 keeps more of its
#: weight relative to rank 1 -- which is what a question needing the whole
#: tail wants, and what a question needing one row does not.
_DEFAULT_K = 60


@dataclass(frozen=True, slots=True)
class Fusion:
    """How much each retrieval arm counts, and how flat the rank curve is."""

    vector: float = 1.0
    lexical: float = 1.0
    rrf_k: int = _DEFAULT_K


#: Per shape. Anything absent uses the default, so a new QuestionKind is
#: inert here until somebody has a reason for it.
_BY_KIND: dict[QuestionKind, Fusion] = {
    # Enumeration and counting: every instance matters and the surface form
    # repeats, which is the lexical arm's strength. Flatter curve so the
    # tail survives fusion.
    QuestionKind.LIST_ALL: Fusion(vector=1.0, lexical=1.3, rrf_k=90),
    QuestionKind.COUNT: Fusion(vector=1.0, lexical=1.3, rrf_k=90),
    # Ordering and intervals need the endpoints, which are usually stated in
    # similar words at both ends. Mild lexical lean, standard curve.
    QuestionKind.ORDER: Fusion(vector=1.0, lexical=1.15),
    QuestionKind.DATE_ARITH: Fusion(vector=1.0, lexical=1.15),
    # A lookup has one right answer, phrased in the asker's words rather than
    # the memory's. Lean on the arm that handles paraphrase.
    QuestionKind.DIRECT: Fusion(vector=1.15, lexical=1.0),
    # Advice retrieves preferences, which are almost never phrased the way
    # the request is. Strongest vector lean of the set.
    QuestionKind.ADVICE: Fusion(vector=1.25, lexical=0.9),
    # Comparison needs both sides present before anything can be compared,
    # so it is closer to enumeration than to lookup.
    QuestionKind.COMPARE: Fusion(vector=1.0, lexical=1.15, rrf_k=75),
}


def fusion_for(kind: QuestionKind) -> Fusion:
    """Fusion parameters for a question shape."""
    return _BY_KIND.get(kind, Fusion())


__all__ = ["Fusion", "fusion_for"]
