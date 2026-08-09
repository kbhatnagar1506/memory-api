"""Which KIND of memory a question wants.

Write-time extraction stores two things per document: the episode as it was
said, and the atomic claims it makes. Retrieval had no way to prefer one, and
a 500-question run measured what that costs -- cleanly split along a line
nobody had drawn in code:

    knowledge-update            0.872 -> 0.923   +5.1   semantic
    single-session-preference   0.667 -> 0.700   +3.3   semantic
    single-session-assistant    0.946 -> 0.964   +1.8   semantic
    single-session-user         0.943 -> 0.886   -5.7   EPISODIC
    temporal-reasoning          0.797 -> 0.722   -7.5   EPISODIC
    multi-session               0.729 -> 0.639   -9.0   EPISODIC

Six for six. Every semantic capability held or improved; every episodic one
lost. The `only` arm made the mechanism unambiguous: it retrieved BETTER than
`add` (full_recall 0.950 vs 0.948) and answered WORSE (0.762 vs 0.781). The
right sessions were found and then could not be answered from, because what
the answerer read had changed. Extraction preserves WHAT IS TRUE and discards
WHAT HAPPENED.

So a claim is not a better memory, it is a different one:

    "Ran the charity 5K in 27:12 on 20 May 2023"   answers "what is their PB"
    the conversation it came from                  answers "what did they do
                                                    before the tournament"

An ordering question needs episodes because the order IS the information, and
six disconnected claims each stamped with a timestamp is a worse basis for it
than three coherent sessions.

ALLOCATION, NOT FILTERING. A misclassified question under a hard filter gets
an empty result set; under allocation it gets a worse ordering and still
answers. The classifier is regex-first and deliberately conservative, so it
returns DIRECT when unsure -- and DIRECT is the balanced split, which means
the failure mode of the classifier is "no routing", not "wrong routing".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import MemoryKind, ScoredMemory
from ..synthesis.classify import QuestionKind, classify


@dataclass(frozen=True, slots=True)
class Allocation:
    """How many of `limit` slots each memory kind may take."""

    episodic: int
    derived: int
    reason: str


#: Fraction of the window reserved for EPISODES, by question shape.
#:
#: ORDER and DATE_ARITH are the two shapes whose answer is a relationship
#: BETWEEN events, so they need the events. COMPARE needs the surrounding
#: detail a claim strips. COUNT sits at 0.5: counting needs every instance
#: (claims are better at not hiding one inside a paragraph) but also needs
#: enough context to tell two mentions of one event from two events.
#:
#: ADVICE is the extreme on the other side -- it uses remembered preferences
#: and never needs a timeline, and preference was the one capability that
#: improved when episodes were removed entirely.
_EPISODIC_SHARE: dict[QuestionKind, float] = {
    QuestionKind.ORDER: 0.7,
    QuestionKind.DATE_ARITH: 0.7,
    QuestionKind.COMPARE: 0.6,
    QuestionKind.COUNT: 0.5,
    QuestionKind.LIST_ALL: 0.4,
    QuestionKind.DIRECT: 0.5,
    QuestionKind.ADVICE: 0.2,
}

#: Neither side ever gets zero. A shape is a guess about what the question
#: needs, and starving one kind entirely turns a wrong guess into an
#: unanswerable question rather than a worse-ordered one.
_MIN_PER_KIND = 2


def allocate(question: str, limit: int, *, kind: QuestionKind | None = None) -> Allocation:
    """Split `limit` slots between episodes and claims for this question."""
    shape = kind if kind is not None else classify(question)
    share = _EPISODIC_SHARE.get(shape, 0.5)

    if limit <= 1:
        # Nothing to split. Give the single slot to the preferred kind.
        return Allocation(
            episodic=1 if share >= 0.5 else 0,
            derived=0 if share >= 0.5 else 1,
            reason=f"{shape.value}: single slot",
        )

    floor = min(_MIN_PER_KIND, limit // 2)
    episodic = round(limit * share)
    episodic = max(floor, min(limit - floor, episodic))
    return Allocation(
        episodic=episodic,
        derived=limit - episodic,
        reason=f"{shape.value}: {episodic} episodic / {limit - episodic} claim slots",
    )


def apply_allocation(
    scored: list[ScoredMemory], allocation: Allocation, limit: int
) -> list[ScoredMemory]:
    """Fill each kind's slots best-first, then backfill any it could not use.

    Backfilling is what makes this allocation rather than a quota: a space
    holding only episodes must still return `limit` results for a question
    that would have preferred claims. The reserve only ever changes WHICH
    results are dropped when both kinds are competing for the same window.
    """
    budget = {MemoryKind.EPISODIC: allocation.episodic, MemoryKind.DERIVED: allocation.derived}
    kept: list[ScoredMemory] = []
    overflow: list[ScoredMemory] = []

    for s in scored:
        remaining = budget.get(s.memory.kind, 0)
        if remaining > 0:
            budget[s.memory.kind] = remaining - 1
            s.explain.append(f"routing: {allocation.reason}")
            kept.append(s)
        else:
            overflow.append(s)

    if len(kept) < limit:
        # Unused slots go back to the pool in score order, so an unbalanced
        # corpus is never punished for not matching the question's shape.
        for s in overflow:
            if len(kept) >= limit:
                break
            s.explain.append("routing: backfilled an unused slot")
            kept.append(s)

    # Restore global score order: allocation decides membership, not ranking.
    kept.sort(key=lambda s: s.score, reverse=True)
    return kept[:limit]


__all__ = ["Allocation", "allocate", "apply_allocation"]
