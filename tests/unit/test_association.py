"""Association: the one edge type with a producer and no tests.

`propose_associations` runs on every write with belief revision enabled, so it
is the edge most likely to exist in a real graph -- and it was imported by
nothing in the suite.

Its whole design is a rejection of the obvious implementation. "Link anything
above a cosine threshold" was measured on a real 55-memory space and produced:

    0.847   "Prefers window seats on long flights"
         <->  "Reads science fiction on the commute"

Two preferences sharing a sentence shape and no subject, outscoring pairs that
are genuinely related. A threshold low enough to catch real associations catches
that too, and the result says "everything relates to everything" -- the same
information as no edges at all.

So the signal is SHARED REFERENTS and similarity is only a floor and a
tie-break. That inversion is what these tests are about: the 0.847 pair must not
be drawn, and a pair sharing a name must be drawn even when it scores lower.

Vectors are constructed with `at_cosine` rather than embedded, because every
threshold here is a similarity band and the test needs to sit a pair exactly
inside or outside one. `MIN_SIMILARITY` is 0.45 and `TAG_SIMILARITY_FLOOR` is
0.60; landing on 0.44 versus 0.46 by hoping a hash embedder cooperates is not a
test.
"""

from __future__ import annotations

import pytest
from tests.support.factories import memory as build_memory
from tests.support.vectors import ANCHOR, at_cosine, axis

from mapi.domain.association import (
    MAX_ASSOCIATIONS,
    MAX_SIMILARITY,
    MIN_SIMILARITY,
    TAG_SIMILARITY_FLOOR,
    propose_associations,
)
from mapi.domain.models import MemoryStatus

ORG, SPACE = "org_01k000000000000000000000", "spc_01k000000000000000000000"


def _mem(content: str, **kwargs: object) -> object:
    return build_memory(org_id=ORG, space_id=SPACE, content=content, **kwargs)  # type: ignore[arg-type]


def _at(cosine: float, off: int = 1) -> list[float]:
    return at_cosine(cosine, off=off)


# -- the case this function exists to NOT draw ------------------------------


def test_close_but_sharing_nothing_is_not_an_association() -> None:
    """The 0.847 measurement, as a test.

    Two preferences phrased alike, no common subject, no common tag. High
    similarity is not evidence of relatedness and must not produce an edge.
    """
    new = _mem("Prefers window seats on long flights.")
    other = _mem("Reads science fiction on the commute.")
    assert propose_associations(new, axis(ANCHOR), [(other, _at(0.847))]) == []


def test_a_shared_entity_links_even_at_a_lower_similarity() -> None:
    """The inversion: 0.50 with a shared name beats 0.847 with nothing.

    This is the whole design in one assertion -- the reason is the referent, and
    similarity only decides whether the pair is close enough to consider.
    """
    new = _mem("The founding engineer offer was accepted by Priya.")
    related = _mem("We heard that Priya starts on the first of June.")
    unrelated = _mem("Reads science fiction on the commute.")

    proposals = propose_associations(
        new, axis(ANCHOR), [(related, _at(0.50, 1)), (unrelated, _at(0.847, 2))]
    )
    assert [p.right_id for p in proposals] == [related.id]
    # Lowercased, because `extract_entities` casefolds. The reason string is
    # user-visible ("both mention priya"), so this pins the casing rather than
    # asserting a prettier version that does not exist.
    assert "priya" in proposals[0].reason


def test_a_name_that_starts_a_memory_is_not_seen_as_an_entity() -> None:
    """A real limitation of association reach, found while writing these tests.

    `extract_entities` requires a capitalised word to be sighted MID-SENTENCE
    before it counts, because sentence-initial capitalisation carries no
    information -- every sentence starts capitalised. Measured:

        "Priya accepted the founding engineer offer."   -> []
        "The offer was accepted by Priya."              -> ['priya']

    That is the right call for entity extraction and it has a consequence for
    association nobody wrote down: "Priya accepted the offer" and "Priya starts
    in June" -- the most natural way to record either fact -- share NO entity, so
    they associate only if they also share a tag and clear 0.60. Two memories
    about one person, both beginning with her name, are not linked.

    Pinned rather than fixed: the fix belongs in `entities.py` (a name seen
    mid-sentence anywhere in the corpus should count sentence-initially too), and
    changing extraction affects retrieval bridging as well as this. Recorded here
    so the limit is discoverable from the association tests, which is where
    somebody debugging a sparse graph would look.
    """
    from mapi.domain.retrieval.entities import extract_entities

    assert extract_entities("Priya accepted the founding engineer offer.") == []

    initial_new = _mem("Priya accepted the founding engineer offer.")
    initial_other = _mem("Priya starts on the first of June.")
    assert propose_associations(initial_new, axis(ANCHOR), [(initial_other, _at(0.80))]) == []

    # The same two facts, one word rearranged, do link.
    mid_new = _mem("The founding engineer offer was accepted by Priya.")
    mid_other = _mem("We heard that Priya starts on the first of June.")
    assert propose_associations(mid_new, axis(ANCHOR), [(mid_other, _at(0.80))])


# -- similarity band -------------------------------------------------------


def test_below_the_floor_nothing_links_however_much_is_shared() -> None:
    """Guards against one incidental token joining two unrelated memories.

    Both mention Priya; at 0.30 they are not about the same thing, and a name
    appearing in two distant memories is a coincidence the graph should not
    render as a relationship.
    """
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("A restaurant in Lisbon was recommended by Priya.")
    below = MIN_SIMILARITY - 0.15
    assert propose_associations(new, axis(ANCHOR), [(other, _at(below))]) == []


def test_exactly_at_the_floor_links() -> None:
    """`MIN_SIMILARITY <= similarity`, inclusive. Pinned because an inclusive
    bound flipping to exclusive is a silent one-pair-wide behaviour change."""
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("We heard that Priya starts on the first of June.")
    proposals = propose_associations(new, axis(ANCHOR), [(other, _at(MIN_SIMILARITY))])
    assert len(proposals) == 1


def test_at_or_above_the_ceiling_the_pair_belongs_to_a_stronger_edge() -> None:
    """0.97 is the supersession and near-duplicate band.

    Drawing an association there would put two edges on one relationship, and
    the other edge says something stronger.
    """
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("The founding engineer role was accepted by Priya.")
    assert propose_associations(new, axis(ANCHOR), [(other, _at(MAX_SIMILARITY))]) == []


def test_just_below_the_ceiling_still_links() -> None:
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("We heard that Priya starts on the first of June.")
    proposals = propose_associations(new, axis(ANCHOR), [(other, _at(MAX_SIMILARITY - 0.01))])
    assert len(proposals) == 1


# -- tags: filing versus relatedness ---------------------------------------


def test_a_shared_tag_links_only_when_the_pair_is_also_close() -> None:
    """A tag on half the corpus is a folder, not a relationship.

    "work" on two memories about different jobs in different years shares a
    filing decision and nothing else, so the tag needs corroboration.
    """
    new = _mem("Shipped the billing rewrite.", tags=["work"])
    other = _mem("Renewed the office lease.", tags=["work"])

    below = propose_associations(new, axis(ANCHOR), [(other, _at(TAG_SIMILARITY_FLOOR - 0.05))])
    assert below == [], "a tag below the floor is filing, not evidence"

    above = propose_associations(new, axis(ANCHOR), [(other, _at(TAG_SIMILARITY_FLOOR + 0.05))])
    assert len(above) == 1
    assert "same subject" in above[0].reason
    assert "work" in above[0].reason


def test_exactly_at_the_tag_floor_links() -> None:
    new = _mem("Shipped the billing rewrite.", tags=["work"])
    other = _mem("Renewed the office lease.", tags=["work"])
    proposals = propose_associations(new, axis(ANCHOR), [(other, _at(TAG_SIMILARITY_FLOOR))])
    assert len(proposals) == 1


def test_tags_are_matched_case_insensitively() -> None:
    new = _mem("Shipped the billing rewrite.", tags=["Work"])
    other = _mem("Renewed the office lease.", tags=["WORK"])
    proposals = propose_associations(new, axis(ANCHOR), [(other, _at(0.75))])
    assert len(proposals) == 1


def test_an_entity_outranks_a_tag() -> None:
    """A shared name is stronger evidence than a shared folder, so when both
    apply the reason names the entity and the weight reflects it."""
    new = _mem("The billing rewrite was shipped by Priya.", tags=["work"])
    other = _mem("The office lease was renewed by Priya.", tags=["work"])
    proposals = propose_associations(new, axis(ANCHOR), [(other, _at(0.75))])
    assert len(proposals) == 1
    assert "both mention" in proposals[0].reason


# -- what is eligible at all -----------------------------------------------


def test_a_memory_never_associates_with_itself() -> None:
    new = _mem("The offer was accepted by Priya.")
    assert propose_associations(new, axis(ANCHOR), [(new, _at(0.80))]) == []


@pytest.mark.parametrize(
    "status", [MemoryStatus.SUPERSEDED, MemoryStatus.ARCHIVED, MemoryStatus.STALE]
)
def test_only_active_memories_are_associated(status: MemoryStatus) -> None:
    """An edge to a hidden memory draws a line to a node the picture omits."""
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("Priya starts on the first of June.", status=status)
    assert propose_associations(new, axis(ANCHOR), [(other, _at(0.80))]) == []


def test_a_dimension_mismatch_is_skipped_not_crashed() -> None:
    """An index built by a different model. Skip the candidate; the write must
    still succeed, because the alternative is a failed ingest on a config
    change."""
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("We heard that Priya starts on the first of June.")
    assert propose_associations(new, axis(ANCHOR), [(other, [0.1, 0.2])]) == []


def test_no_embedding_proposes_nothing() -> None:
    new = _mem("The offer was accepted by Priya.")
    other = _mem("We heard Priya starts in June.")
    assert propose_associations(new, [], [(other, _at(0.80))]) == []


def test_no_candidates_proposes_nothing() -> None:
    assert propose_associations(_mem("anything"), axis(ANCHOR), []) == []


# -- degree cap and determinism -------------------------------------------


def test_the_degree_is_capped() -> None:
    """A node with forty edges is a graph nobody can read, and readability is
    the stated reason this feature exists at all."""
    new = _mem("The founding engineer offer was accepted by Priya.")
    candidates = [
        (_mem(f"We saw Priya do thing number {i}."), _at(0.70, off=i + 1)) for i in range(12)
    ]
    proposals = propose_associations(new, axis(ANCHOR), candidates)
    assert len(proposals) == MAX_ASSOCIATIONS


def test_max_edges_zero_disables_it() -> None:
    new = _mem("The offer was accepted by Priya.")
    other = _mem("We heard Priya starts in June.")
    assert propose_associations(new, axis(ANCHOR), [(other, _at(0.80))], max_edges=0) == []


def test_more_shared_entities_rank_higher_than_closer_similarity() -> None:
    """Evidence first, proximity second. Two shared names beat one shared name
    at a higher cosine, because the reason is what the edge asserts."""
    new = _mem("We saw Priya and Aaditi review the Datadog invoice.")
    two = _mem("We saw Priya and Aaditi meet on Tuesday.")
    one = _mem("The report was sent by Priya.")

    proposals = propose_associations(
        new, axis(ANCHOR), [(one, _at(0.90, 1)), (two, _at(0.60, 2))]
    )
    assert next(p.right_id for p in proposals) == two.id


def test_the_same_corpus_produces_the_same_graph() -> None:
    """Ties break on id, so a picture does not reshuffle between runs."""
    new = _mem("The invoice was reviewed by Priya.")
    candidates = [(_mem(f"We saw Priya do thing {i}."), _at(0.70, off=i + 1)) for i in range(8)]
    first = [p.right_id for p in propose_associations(new, axis(ANCHOR), candidates)]
    second = [p.right_id for p in propose_associations(new, axis(ANCHOR), list(candidates))]
    assert first == second


# -- confidence ------------------------------------------------------------


def test_confidence_rises_with_evidence_and_stays_bounded() -> None:
    new = _mem("We saw Priya and Aaditi review the Datadog invoice together.")
    two = _mem("We saw Priya and Aaditi meet about Datadog on Tuesday.")
    proposals = propose_associations(new, axis(ANCHOR), [(two, _at(0.80))])
    assert len(proposals) == 1
    assert 0.0 < proposals[0].confidence <= 1.0


def test_a_tag_only_association_is_less_confident_than_a_named_one() -> None:
    """The confidence has to reflect which branch produced the edge, or the
    field carries no information about how much to trust the reason."""
    tagged_new = _mem("Shipped the billing rewrite.", tags=["work"])
    tagged_other = _mem("Renewed the office lease.", tags=["work"])
    named_new = _mem("The billing rewrite was shipped by Priya.")
    named_other = _mem("The office lease was renewed by Priya.")

    tagged = propose_associations(tagged_new, axis(ANCHOR), [(tagged_other, _at(0.80))])
    named = propose_associations(named_new, axis(ANCHOR), [(named_other, _at(0.80))])
    assert tagged and named
    assert tagged[0].confidence < named[0].confidence


def test_the_proposal_records_the_similarity_it_decided_on() -> None:
    """So a reviewer can see the number without recomputing it."""
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("We heard that Priya starts on the first of June.")
    proposals = propose_associations(new, axis(ANCHOR), [(other, _at(0.73))])
    assert proposals[0].similarity == pytest.approx(0.73, abs=1e-6)


def test_the_edge_points_from_the_new_memory() -> None:
    new = _mem("The founding engineer offer was accepted by Priya.")
    other = _mem("We heard that Priya starts on the first of June.")
    proposals = propose_associations(new, axis(ANCHOR), [(other, _at(0.80))])
    assert proposals[0].left_id == new.id
    assert proposals[0].right_id == other.id
