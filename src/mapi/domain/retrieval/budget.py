"""How much evidence to hand the answerer, decided by the system.

Neither a document count nor a token count transfers between corpora, which
is why every attempt to pick one number has been corpus-specific tuning:

    LongMemEval   one "document" is a 12,000-char session   k=10 -> 27,000 tok
    LoCoMo        one "document" is a 400-char turn         k=10 ->    954 tok

Same k, two orders of magnitude of actual evidence. Measured saturation
differs the same way: LongMemEval's delivery saturates at k=10 (0.972 and
buying the last 2.8% at k=20 costs more in dilution than it returns), while
LoCoMo is still climbing at k=50 (0.668 -> 0.833).

So the invariant is not k and not tokens. It is **how concentrated the
evidence is**. A question whose answer sits in one place has one dominant
hit and a steep score cliff behind it; a question whose answer is spread
across sessions has a flat run of comparable scores. Reading that shape off
the score distribution adapts to the corpus automatically, which is exactly
what a fixed number cannot do.

MEASURED RESULT: this does not work as specified. Keep it only as a
documented negative.

Replayed offline over real ranked lists on both corpora (delivery only, no
answerer, no judge):

    LongMemEval   fixed k=10  0.958 delivery @ 10 docs
                  fixed k=20  0.992 @ 20
                  score-cliff 1.000 @ 31.9      <- perfect, by taking 3x more
    LoCoMo        fixed k=50  0.859 @ 50
                  score-cliff 0.773 @ 18.1      <- worse than the fixed rule

It buys delivery by taking MORE, which is the one move already measured
harmful: k=20 on LongMemEval costs 8 questions to dilution, so 32 documents
is far worse than 0.972 at 10. And the tell that the mechanism is inert is
that focused questions took exactly 16.0 documents on BOTH corpora -- a rule
reading concentration off the result set cannot land on the same number for
12,000-char sessions and 400-char turns. Similarity scores are flat enough
that "at least 80% of the top score" admits nearly everything, so this is a
fixed number wearing an adaptive costume.

The deeper finding, which outlived the rule: the two corpora need OPPOSITE
policies. On LongMemEval delivery already saturates (0.992 at k=20) and the
binding constraint is dilution, so the objective is the MINIMUM that
delivers. On LoCoMo delivery is still climbing at k=50, so the objective is
more. "Stop when delivery saturates" optimises the wrong quantity on one of
them, and no single stopping rule tried so far expresses both.

Do not re-derive this without a mechanism that discriminates on something
other than score ratio.

Two signals combine:

  * the QUESTION's shape bounds how many distinct pieces of evidence it can
    plausibly need (an advice request needs one; a count needs several) --
    measured, our label-free classifier predicts gold-evidence count with
    median 1 for advice/direct and 2+ for aggregates;
  * the RESULT SET's shape says where the evidence actually stops.

Neither alone is enough. Shape alone is the fixed-number problem again; score
alone cannot tell a genuinely broad question from a vague one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..synthesis.classify import QuestionKind, classify

#: Fraction of the top score below which a result is treated as off the cliff
#: rather than more evidence. Two values, because the two question families
#: have genuinely different profiles: a single-evidence question should stop
#: at the first real drop, an aggregate should keep taking comparable hits.
_CLIFF_FOCUSED = 0.80
_CLIFF_BROAD = 0.55

#: Floors, so a steep cliff cannot starve a question that needs corroboration.
_FLOOR_FOCUSED = 2
_FLOOR_BROAD = 4

#: Absolute backstop. Not a tuning knob -- a guard against a pathological
#: flat distribution returning the whole corpus. Deliberately far above any
#: budget observed in practice (LongMemEval saturates at 10, LoCoMo at ~50).
_HARD_CEILING = 100

#: Question shapes whose answer lives in one place. Everything else -- counts,
#: orderings, spans, comparisons, lists -- aggregates over several.
_FOCUSED = frozenset({QuestionKind.DIRECT, QuestionKind.ADVICE})


@dataclass(frozen=True, slots=True)
class Budget:
    """How many results to keep, and why. `reason` is for explain traces."""

    take: int
    cliff_ratio: float
    focused: bool
    reason: str


def decide(question: str, scores: list[float]) -> Budget:
    """How many of `scores` (best-first) to hand the answerer.

    Corpus-independent: the caller supplies no k, no token budget, and no
    knowledge of how large a document is. The stopping point comes from where
    this particular result set falls off relative to its own best hit.
    """
    focused = classify(question) in _FOCUSED
    cliff = _CLIFF_FOCUSED if focused else _CLIFF_BROAD
    floor = _FLOOR_FOCUSED if focused else _FLOOR_BROAD

    if not scores:
        return Budget(0, cliff, focused, "nothing retrieved")

    top = scores[0]
    if top <= 0:
        return Budget(min(floor, len(scores)), cliff, focused, "no usable scores")

    threshold = top * cliff
    take = 0
    for score in scores[:_HARD_CEILING]:
        if score < threshold:
            break
        take += 1

    # The floor protects against a steep cliff on a question that needs
    # corroboration; the ceiling protects against a flat distribution that
    # never falls off. Both are guards, not the mechanism.
    take = max(take, min(floor, len(scores)))
    take = min(take, len(scores), _HARD_CEILING)

    shape = "focused" if focused else "broad"
    return Budget(
        take,
        cliff,
        focused,
        f"{shape} question; kept {take} results scoring >= {cliff:.0%} of the best ({top:.3f})",
    )


__all__ = ["Budget", "decide"]
