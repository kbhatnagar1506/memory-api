"""Give a retrieved fragment back the conversation it came from.

Retrieval returns the right memory and then hands the answerer a fragment
stripped of everything that made it meaningful. On a conversational corpus a
memory is one turn of about fifty tokens:

    "I caught 12 bass"          self-contained
    "yeah, three of them"       meaningless alone

Both retrieve correctly. Only the first can be answered from.

This is where our measured headroom is. Retrieval delivers complete evidence
for 97.2% of LongMemEval questions at full_recall@k 0.968, and accuracy is
0.8255 -- so about 17.5% of questions are answered wrong while the evidence
is already in the context, against a retrieval-side ceiling of 2.8%. Every
ranking idea competes for the small number. This one goes after the large
one.

The asymmetry is what makes it safe to try: hydration changes only what is
ASSEMBLED after retrieval, never what is retrieved, so `full_recall@k` and
MRR are computed on an unchanged set. Upside and downside land on different
metrics, and one run tells you unambiguously which moved.

WHAT COUNTS AS A NEIGHBOUR. Same source, nearest in event time. `source` is
the caller's own grouping -- a conversation, a document, a thread -- so this
reassembles exactly the unit the caller wrote and never invents one. A memory
with no source has no neighbours, and passes through alone.

BUDGETED, because the failure this fixes has a mirror image. Padding every
anchor with context dilutes the prompt, and a model reading twenty turns to
answer from one is a model with more chances to answer from the wrong one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .derive import SourceDoc

#: Turns either side of an anchor. Two is enough to resolve a pronoun or an
#: elision ("three of them") without pulling in an unrelated topic shift.
DEFAULT_NEIGHBOURS = 2

#: Characters of assembled context. A budget in characters rather than tokens
#: because the conversion is corpus-dependent and the point is a ceiling, not
#: an accounting -- see `budget.py` for why token counts do not transfer.
DEFAULT_BUDGET_CHARS = 24_000


@dataclass(frozen=True, slots=True)
class Neighbourhood:
    """An anchor and the turns around it, in the order they occurred."""

    anchor: SourceDoc
    before: tuple[SourceDoc, ...] = ()
    after: tuple[SourceDoc, ...] = ()

    def ordered(self) -> list[SourceDoc]:
        return [*self.before, self.anchor, *self.after]


def _length(doc: SourceDoc) -> int:
    return len(doc.text)


def assemble(
    neighbourhoods: Sequence[Neighbourhood],
    *,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> list[SourceDoc]:
    """Flatten neighbourhoods into one deduplicated, budgeted document list.

    ANCHORS ARE PAID FOR FIRST. Every anchor is admitted before any neighbour
    is, so a tight budget degrades to exactly today's behaviour -- the
    retrieved memories, unexpanded -- rather than dropping a retrieved memory
    to make room for somebody else's context. Hydration must never be able to
    lose the thing retrieval found.

    Neighbours are then added a ring at a time, closest first across all
    anchors, so a small budget spreads one turn of context everywhere rather
    than four turns onto the first result.
    """
    if not neighbourhoods:
        return []

    chosen: dict[str, SourceDoc] = {}
    spent = 0

    for hood in neighbourhoods:
        if hood.anchor.id in chosen:
            continue
        chosen[hood.anchor.id] = hood.anchor
        spent += _length(hood.anchor)

    # Ring 0 is the immediate neighbour, ring 1 the one beyond it. Widening
    # in lockstep keeps the context even when the budget runs out mid-way.
    depth = max((max(len(h.before), len(h.after)) for h in neighbourhoods), default=0)
    for ring in range(depth):
        for hood in neighbourhoods:
            for side in (hood.before[::-1], hood.after):
                if ring >= len(side):
                    continue
                doc = side[ring]
                if doc.id in chosen:
                    continue
                if spent + _length(doc) > budget_chars:
                    continue
                chosen[doc.id] = doc
                spent += _length(doc)

    # Chronological, so the answerer reads a conversation rather than a
    # relevance ranking. An elision resolves backwards; presenting the reply
    # before the question it answers puts the work back on the model.
    return sorted(chosen.values(), key=lambda d: (d.occurred_at, d.id))


__all__ = [
    "DEFAULT_BUDGET_CHARS",
    "DEFAULT_NEIGHBOURS",
    "Neighbourhood",
    "assemble",
]
