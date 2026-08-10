"""Association: the edge that says two memories are about the same thing.

Every other edge in this system is a claim about TRUTH. `supersedes` says one
memory replaced another, `contradicts` says they cannot both hold,
`derived_from` says one was computed from the other. All three are rare by
construction, because most facts about a person neither replace nor disagree
with the rest.

Measured on a real space of 55 memories about one person: zero edges. Every
statement was true, none replaced another, and the graph was 55 isolated
points. That is correct behaviour and a useless picture. What was missing is
the ordinary relation — these two are about the same subject — which nothing
in the system modelled.

WHY NOT COSINE. The obvious implementation is "link anything above a
similarity threshold", and it is wrong. Measured with the production
embedder on that same space:

    0.847   "Prefers window seats on long flights"
         ↔  "Reads science fiction on the commute"

    0.669   "Prefers window seats on long flights"
         ↔  "Allergic to shellfish"

The first pair shares nothing but a sentence shape -- two preferences, no
common subject -- and outscores pairs that are genuinely related elsewhere in
the corpus. A cosine threshold low enough to catch real associations is low
enough to catch that, and the result is a hairball that says "everything is
related to everything", which is the same amount of information as no edges
at all.

So the signal is SHARED REFERENTS, not proximity:

  * a shared entity -- a name, a product, an identifier both memories mention
  * a shared tag, but only when the two are also topically close, because a
    tag applied to half the corpus is a folder, not a relationship

Similarity is kept as a tie-break and a floor, never as the reason.

Degree is capped. An association graph where one popular memory links to
forty others is a graph nobody can read, and the cap is what keeps the
picture legible as the corpus grows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .embeddings.base import Vector, cosine_similarity
from .models import Memory, MemoryStatus
from .retrieval.entities import extract_entities

#: Most associations one new memory may declare. Chosen for READABILITY
#: rather than recall: the graph is a picture a person looks at, and past a
#: handful of edges per node it stops being one.
MAX_ASSOCIATIONS = 4

#: A shared tag only counts as evidence when the two memories are also this
#: close. Below it, the tag is filing, not relatedness -- "work" on two
#: memories about different jobs in different years is a folder name.
TAG_SIMILARITY_FLOOR = 0.60

#: Below this, nothing links however much vocabulary is shared. Guards
#: against a single incidental token ("Monday", "2024") joining two memories
#: with nothing else in common.
MIN_SIMILARITY = 0.45

#: At or above this the pair is a candidate for supersession or contradiction
#: instead, and those edges say something stronger. Associating them as well
#: would draw two edges for one relationship.
MAX_SIMILARITY = 0.97


@dataclass(frozen=True, slots=True)
class AssociationProposal:
    """Two memories that share a referent, with what they share."""

    left_id: str
    right_id: str
    similarity: float
    reason: str
    confidence: float


def _entities(text: str) -> set[str]:
    return set(extract_entities(text, max_entities=12))


def propose_associations(
    new_memory: Memory,
    new_embedding: Vector,
    candidates: Sequence[tuple[Memory, Vector]],
    *,
    max_edges: int = MAX_ASSOCIATIONS,
) -> list[AssociationProposal]:
    """Which existing memories this one is about the same thing as.

    `candidates` is the nearest-neighbour set the write path already fetched
    for deduplication and supersession, so this costs no extra query -- it is
    a second reading of a lookup that had to happen anyway.
    """
    if not new_embedding or max_edges <= 0:
        return []

    new_entities = _entities(new_memory.content)
    new_tags = {t.casefold() for t in new_memory.tags}

    scored: list[tuple[int, float, AssociationProposal]] = []
    for memory, vector in candidates:
        if memory.id == new_memory.id or memory.status is not MemoryStatus.ACTIVE:
            continue
        if len(vector) != len(new_embedding):
            continue
        similarity = cosine_similarity(new_embedding, vector)
        if not (MIN_SIMILARITY <= similarity < MAX_SIMILARITY):
            continue

        shared_entities = new_entities & _entities(memory.content)
        shared_tags = new_tags & {t.casefold() for t in memory.tags}

        if shared_entities:
            named = ", ".join(sorted(shared_entities)[:3])
            reason, weight = f"both mention {named}", len(shared_entities) + 1
        elif shared_tags and similarity >= TAG_SIMILARITY_FLOOR:
            named = ", ".join(sorted(shared_tags)[:3])
            reason, weight = f"same subject ({named})", 1
        else:
            # Close in embedding space and nothing in common. This is exactly
            # the 0.847 case -- two unrelated preferences that happen to be
            # phrased alike -- and it is the pair this function exists to
            # NOT draw.
            continue

        scored.append(
            (
                weight,
                similarity,
                AssociationProposal(
                    left_id=new_memory.id,
                    right_id=memory.id,
                    similarity=similarity,
                    reason=reason,
                    # Shared referents are the evidence; similarity only
                    # sharpens a proposal that already has a reason.
                    confidence=min(1.0, 0.5 + 0.15 * weight + 0.2 * similarity),
                ),
            )
        )

    # Strongest evidence first, then closest. The id breaks ties so the same
    # corpus always produces the same graph.
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].right_id))
    return [proposal for _, _, proposal in scored[:max_edges]]


__all__ = [
    "MAX_ASSOCIATIONS",
    "MAX_SIMILARITY",
    "MIN_SIMILARITY",
    "TAG_SIMILARITY_FLOOR",
    "AssociationProposal",
    "propose_associations",
]
