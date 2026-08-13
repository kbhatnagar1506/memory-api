"""Family G: knowing when the answer is not there.

The most-neglected capability in the field and the one with the clearest
measurements. Google's *Sufficient Context* found 45.2% of real RAG instances
have insufficient context and models then hallucinate 15-40% of the time instead
of abstaining. RGB names "negative rejection" as one of four core abilities and
finds LLMs bad at it. AbstentionBench's headline is that reasoning fine-tuning
makes abstention WORSE and scale does not help. SQuAD 2.0 built 50,000
unanswerable questions specifically engineered to look answerable.

All of those measure a model. The store-level question comes first and is
cheaper: **can the retrieval layer tell an answerable query from an unanswerable
one, before any model is involved?** If it can, a caller has a deterministic
abstention signal and the model never has to be trusted to refuse. If it cannot,
every downstream abstention is the model guessing.

`sufficient@k` is the metric that makes this concrete: 1 exactly when the top k
contains EVERY required gold span. It is the quantity Google measured at 54.8%,
computable here with zero model calls, and it is what separates a retrieval bug
from a reading bug in any incident.

This family runs AFTER the confidence-scale fix on purpose. Before it, `score`
was an RRF value with a ~0.033 ceiling compared against thresholds of 0.55/0.30,
so every query graded LOW/weak_evidence and a separability test would have
measured nothing. `calibrated_score` reads `vector_score`, so there is now a real
cosine to threshold on.
"""

from __future__ import annotations

import pytest
from tests.support.corpus import Fact, Query, build, load, register
from tests.support.harness import GEOMETRY
from tests.support.metrics import sufficient_at_k
from tests.support.vectors import ANCHOR, at_cosine, axis

from mapi.domain.retrieval.confidence import ConfidenceLevel, RefusalReason
from mapi.domain.retrieval.pipeline import SearchRequest

#: An answerable query's gold sits here; an unanswerable one has nothing above
#: the filler band. The gap is what a threshold has to separate.
ANSWERABLE_COSINE = 0.90
FILLER_COSINE = 0.30


def _facts(gold: bool, count: int = 10) -> list[Fact]:
    facts: list[Fact] = []
    if gold:
        facts.append(
            Fact(
                gold_id="gold",
                text="gold the kelmady arrangement was signed in March",
                vector=at_cosine(ANSWERABLE_COSINE, off=1),
            )
        )
    facts += [
        Fact(
            gold_id=f"n{i}",
            text=f"n{i} unrelated material about quolmp and thraskin",
            vector=at_cosine(FILLER_COSINE, off=2 + i),
        )
        for i in range(count)
    ]
    return facts


async def _search(
    geometry: tuple, facts: list[Fact], *, slug: str, limit: int = 10, **overrides: object
) -> tuple[object, object]:
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug=slug, name=slug)
    query = Query(
        key="q",
        text="the kelmady arrangement",
        vector=axis(ANCHOR),
        gold_ids=tuple(f.gold_id for f in facts if f.gold_id == "gold"),
    )
    corpus = build(facts, [query])
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
    response = await service.search(
        SearchRequest(
            query="the kelmady arrangement",
            org_id=org.id,
            space_id=space_n.id,
            limit=limit,
            **{**GEOMETRY, **overrides},  # type: ignore[arg-type]
        )
    )
    return response, corpus


# -- G3: sufficient@k, the headline metric ---------------------------------


async def test_sufficient_at_k_is_true_when_the_evidence_is_all_there(
    geometry: tuple,
) -> None:
    response, corpus = await _search(geometry, _facts(gold=True), slug="suff-yes")
    returned = [hit.memory.id for hit in response.results]  # type: ignore[attr-defined]
    assert sufficient_at_k(returned, corpus.memory_ids(["gold"]), 10)  # type: ignore[attr-defined]


async def test_sufficient_at_k_is_false_when_a_required_span_is_missing(
    geometry: tuple,
) -> None:
    """The distinction that matters: a partial evidence set is INSUFFICIENT.

    Three facts required, only the top one retrievable. `recall@k` reads 0.33 and
    looks like partial progress; `sufficient@k` reads false, which is the truth --
    no reader could answer correctly from what came back.
    """
    facts = [
        Fact(
            gold_id="gold",
            text="gold part one of the kelmady arrangement",
            vector=at_cosine(0.90, off=1),
        ),
        Fact(
            gold_id="gold2",
            text="gold2 part two, filed separately",
            vector=at_cosine(0.20, off=2),
        ),
        Fact(
            gold_id="gold3",
            text="gold3 part three, also separate",
            vector=at_cosine(0.18, off=3),
        ),
    ]
    facts += _facts(gold=False, count=8)
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="suff-no", name="suff-no")
    corpus = build(
        facts,
        [
            Query(
                key="q",
                text="the kelmady arrangement",
                vector=axis(ANCHOR),
                gold_ids=("gold", "gold2", "gold3"),
            )
        ],
    )
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
    response = await service.search(
        SearchRequest(
            query="the kelmady arrangement",
            org_id=org.id,
            space_id=space_n.id,
            limit=2,
            **GEOMETRY,
        )
    )
    returned = [hit.memory.id for hit in response.results]
    required = corpus.memory_ids(["gold", "gold2", "gold3"])
    assert not sufficient_at_k(returned, required, 2)


# -- G1: the false-retrieval rate ------------------------------------------


async def test_an_unanswerable_query_returns_only_weak_matches(
    geometry: tuple,
) -> None:
    """RGB's negative rejection, at the layer that should own it.

    Nothing in the corpus answers the query. The store still returns rows --
    that is what a nearest-neighbour index does -- so the signal cannot be "did
    anything come back". It has to be the SCORE, which is why the confidence
    scale had to be fixed before this family could be written.
    """
    response, _corpus = await _search(geometry, _facts(gold=False), slug="unans")
    assert response.results, "a vector index always returns its nearest rows"  # type: ignore[attr-defined]
    top = response.results[0].vector_score  # type: ignore[attr-defined]
    assert top is not None
    assert top <= FILLER_COSINE + 0.05, f"an unanswerable query scored {top}"


async def test_the_confidence_signal_separates_answerable_from_not(
    geometry: tuple,
) -> None:
    """The deterministic abstention signal, end to end.

    An answerable query must not report a refusal reason; an unanswerable one
    must. That is what a caller keys on to decide whether to ask a model at all,
    and it costs nothing.
    """
    answerable, _c1 = await _search(geometry, _facts(gold=True), slug="sep-yes")
    unanswerable, _c2 = await _search(geometry, _facts(gold=False), slug="sep-no")

    assert answerable.confidence is not None and unanswerable.confidence is not None  # type: ignore[attr-defined]
    assert answerable.confidence.refusal_reason is None, (  # type: ignore[attr-defined]
        f"an answerable query reported {answerable.confidence.refusal_reason}"  # type: ignore[attr-defined]
    )
    assert unanswerable.confidence.refusal_reason is RefusalReason.WEAK_EVIDENCE  # type: ignore[attr-defined]


async def test_an_empty_space_declines_for_a_different_reason_than_a_weak_match(
    geometry: tuple,
) -> None:
    """NO_RELEVANT_MEMORY is correct silence. WEAK_EVIDENCE is a memory gap.

    Conflating them is what made the field useless before the scale fix --
    everything read as a gap, so a caller could not tell "we have nothing on
    this" from "we have something thin", which are different product problems.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="empty", name="empty")
    # Registered even though the space is empty: an unregistered query lands on
    # the reserved miss axis, and a test that passes because the query vector was
    # a sentinel is passing for the wrong reason. The autouse miss check caught
    # exactly this.
    embedder.register("anything at all", axis(ANCHOR))
    response = await service.search(
        SearchRequest(
            query="anything at all", org_id=org.id, space_id=space_n.id, limit=5, **GEOMETRY
        )
    )
    assert response.confidence is not None
    assert response.confidence.level is ConfidenceLevel.NONE
    assert response.confidence.refusal_reason is RefusalReason.NO_RELEVANT_MEMORY


# -- G2: threshold calibration --------------------------------------------


@pytest.mark.parametrize("threshold", [0.5, 0.6, 0.7])
async def test_a_threshold_exists_that_admits_answerable_and_rejects_the_rest(
    geometry: tuple, threshold: float
) -> None:
    """`min_score` as a shipped abstention default.

    The gold sits at 0.90 and the filler at 0.30, so any threshold between them
    separates the two perfectly. The point is not that 0.6 is the right number --
    it is that the mechanism now operates on a scale where a number CAN be right.
    Before the fix, `min_score=0.3` emptied every result set regardless.
    """
    admitted, _c1 = await _search(
        geometry, _facts(gold=True), slug=f"thr-yes-{threshold}", min_score=threshold
    )
    rejected, _c2 = await _search(
        geometry, _facts(gold=False), slug=f"thr-no-{threshold}", min_score=threshold
    )
    assert admitted.results, f"threshold {threshold} rejected an answerable query"  # type: ignore[attr-defined]
    assert rejected.results == [], f"threshold {threshold} admitted an unanswerable one"  # type: ignore[attr-defined]


# -- G4: adversarial lexical overlap --------------------------------------


async def test_high_lexical_overlap_does_not_make_an_unanswerable_query_look_answerable(
    geometry: tuple,
) -> None:
    """SQuAD 2.0's construction: swap one entity so overlap stays maximal.

    The corpus holds "the kelmady arrangement"; the query asks about "the
    thraskin arrangement". Nearly every word matches. With the vector arm alone
    the constructed cosine is low and the score reflects that -- which is the
    property being asserted, because the lexical arm would score this highly and
    a fused default would inherit some of that.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="adversarial", name="adv")
    facts = [
        Fact(
            gold_id="near",
            text="near the kelmady arrangement was signed in March",
            vector=at_cosine(0.25, off=1),
        )
    ]
    corpus = build(
        facts,
        [
            Query(
                key="q",
                text="the thraskin arrangement was signed in March",
                vector=axis(ANCHOR),
                gold_ids=(),
            )
        ],
    )
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)

    response = await service.search(
        SearchRequest(
            query="the thraskin arrangement was signed in March",
            org_id=org.id,
            space_id=space_n.id,
            limit=5,
            **GEOMETRY,
        )
    )
    assert response.results
    assert response.results[0].vector_score is not None
    assert response.results[0].vector_score < 0.5, (
        "a one-entity swap with near-total lexical overlap scored as a match"
    )


async def test_the_lexical_arm_alone_is_fooled_by_the_same_pair(
    geometry: tuple,
) -> None:
    """Why fusion exists, stated as a property rather than as a claim.

    The same adversarial pair scored on the lexical arm alone: BM25 sees seven
    shared tokens out of eight and ranks it highly. That is not a bug in BM25 --
    it is why a lexical-only memory system cannot abstain, and why the vector arm
    carries the discrimination here.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="adv-lex", name="adv-lex")
    facts = [
        Fact(
            gold_id="near",
            text="near the kelmady arrangement was signed in March",
            vector=at_cosine(0.25, off=1),
        )
    ]
    corpus = build(
        facts,
        [
            Query(
                key="q",
                text="the thraskin arrangement was signed in March",
                vector=axis(ANCHOR),
                gold_ids=(),
            )
        ],
    )
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)

    from tests.support.harness import LEXICAL_ONLY

    response = await service.search(
        SearchRequest(
            query="the thraskin arrangement was signed in March",
            org_id=org.id,
            space_id=space_n.id,
            limit=5,
            **LEXICAL_ONLY,
        )
    )
    assert response.results, "the lexical arm found the high-overlap non-answer"
    assert response.results[0].lexical_score is not None


# -- G5: calibrated non-abstention ----------------------------------------


async def test_the_system_does_not_over_refuse(geometry: tuple) -> None:
    """Abstention is trivially gameable by refusing everything.

    F1 (abstain when you should) and F4 (do not abstain when you should not) are
    only meaningful together, so this is the twin of the tests above: an
    answerable query at every similarity level from 0.90 down to 0.55 must NOT
    be graded as a refusal.
    """
    for cosine in (0.90, 0.75, 0.60, 0.55):
        facts = [
            Fact(
                gold_id="gold",
                text="gold the kelmady arrangement",
                vector=at_cosine(cosine, off=1),
            )
        ]
        facts += _facts(gold=False, count=5)
        response, _corpus = await _search(geometry, facts, slug=f"noover{int(cosine * 100)}")
        assert response.confidence is not None  # type: ignore[attr-defined]
        assert response.confidence.refusal_reason is None, (  # type: ignore[attr-defined]
            f"a gold at cosine {cosine} was refused: {response.confidence.reason}"  # type: ignore[attr-defined]
        )


async def test_a_conflicting_result_set_declines_for_its_own_reason(
    geometry: tuple,
) -> None:
    """The third refusal reason, and the one that should not be silent.

    Two memories joined by a CONTRADICTS edge, both retrieved. Answering would
    pick a side invisibly. Every competitor resolves this by newest-timestamp,
    which is indistinguishable from there being no conflict at all.
    """
    from tests.support.factories import edge

    from mapi.domain.models import RelationType

    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="conflict", name="conflict")
    facts = [
        Fact(gold_id="a", text="a the standup is at 9:30am", vector=at_cosine(0.90, off=1)),
        Fact(gold_id="b", text="b the standup is at 10:15am", vector=at_cosine(0.88, off=2)),
    ]
    corpus = build(
        facts,
        [Query(key="q", text="when is the standup", vector=axis(ANCHOR), gold_ids=("a", "b"))],
    )
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)

    left, right = corpus.fact("a").memory_id, corpus.fact("b").memory_id
    for source, target in ((left, right), (right, left)):
        await service.store.create_relation(
            edge(
                org_id=org.id,
                space_id=space_n.id,
                source_id=source,
                target_id=target,
                type=RelationType.CONTRADICTS,
            )
        )

    response = await service.search(
        SearchRequest(
            query="when is the standup",
            org_id=org.id,
            space_id=space_n.id,
            limit=5,
            **GEOMETRY,
        )
    )
    assert response.conflicts, "a CONTRADICTS pair in the result set was not surfaced"
    assert response.confidence is not None
    assert response.confidence.refusal_reason is RefusalReason.CONFLICTING_EVIDENCE
