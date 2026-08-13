"""Family B: what crowds the answer out, and what does not.

The most counter-intuitive result in the retrieval literature is Cuconasu et
al.'s: RANDOM noise in the prompt IMPROVED accuracy by up to 35%, while the
retriever's own highest-scoring non-answer documents hurt it. Chroma's Context
Rot found one distractor already measurably hurts, and that a topically coherent
haystack is worse than a shuffled one in 18 of 18 models. RULER found sets
returned with duplicates and gaps as needle count grows.

Every one of those is a claim about a distribution, so every one needs the
similarity structure specified rather than observed. That is what this family is
for and why it could not have been written before `vectors.py`:

    band(0.61, 8)     eight distractors at EXACTLY 0.61 from the query
    cone(0.88, 0.95, 10)   ten near-duplicates, close to the query AND to each other

The one design note worth repeating: B1's tiers are a lexical-overlap x cosine
CROSS, not a semantic ladder. `DeterministicEmbedder` is a lexical embedder, so
"does it know car means automobile" is unanswerable with it and is a provider
question anyway. Specifying both axes independently is strictly more informative:
tier T2 has ZERO shared content words and a high cosine, which tests exactly
whether the vector arm survives lexical zero-overlap -- a property of the
pipeline, not of a hash function.
"""

from __future__ import annotations

import pytest
from tests.support.corpus import Fact, Query, build, load, register
from tests.support.harness import GEOMETRY
from tests.support.metrics import rank_of, recall_at_k, unique_information_at_k
from tests.support.vectors import ANCHOR, at_cosine, axis, band, cone

from mapi.domain.retrieval.pipeline import SearchRequest

GOLD_COSINE = 0.90


async def _ranked(
    geometry: tuple, facts: list[Fact], query_text: str, *, limit: int = 20, **overrides: object
) -> tuple[list[str], object]:
    """Load a corpus, search once, return (gold ids in rank order, corpus)."""
    service, org, space, embedder = geometry
    query = Query(
        key="q",
        text=query_text,
        vector=axis(ANCHOR),
        gold_ids=tuple(f.gold_id for f in facts if f.gold_id.startswith("gold")),
    )
    corpus = build(facts, [query])
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space.id)
    response = await service.search(
        SearchRequest(
            query=query_text,
            org_id=org.id,
            space_id=space.id,
            limit=limit,
            **{**GEOMETRY, **overrides},  # type: ignore[arg-type]
        )
    )
    returned = [hit.memory.id for hit in response.results]
    return corpus.gold_of(returned), corpus


# -- B1: the lexical-overlap x cosine cross --------------------------------


#: Four tiers. `shared` is how many content words the memory shares with the
#: query; `cosine` is its constructed similarity. The two vary INDEPENDENTLY,
#: which is what a semantic ladder cannot express.
TIERS: tuple[tuple[str, str, float], ...] = (
    ("T0_verbatim", "the vorn kelmady arrangement was signed", 0.95),
    ("T1_synonym", "the vorn agreement was countersigned", 0.85),
    ("T2_paraphrase", "quolmp zhoulk brenth thraskin plemmoth", 0.80),
    ("T3_onehop", "zhoulk brenth mentioned kelmady once", 0.45),
)


@pytest.mark.parametrize(("tier", "text", "cosine"), TIERS, ids=[t[0] for t in TIERS])
async def test_each_overlap_tier_is_retrievable(
    geometry: tuple, tier: str, text: str, cosine: float
) -> None:
    """The gold at each tier, against eight distractors well below it.

    T2 is the one that matters: zero shared content words with the query, high
    cosine. If it fails, the vector arm is not contributing and the system is
    lexical-only in practice however it is configured.
    """
    facts = [Fact(gold_id="gold", text=f"gold {text}", vector=at_cosine(cosine, off=1))]
    facts += [
        Fact(gold_id=f"d{i}", text=f"d{i} unrelated filler", vector=at_cosine(0.35, off=2 + i))
        for i in range(8)
    ]
    gold, _corpus = await _ranked(geometry, facts, "the vorn kelmady arrangement was signed")
    assert "gold" in gold, f"{tier} at cosine {cosine} was not retrieved at all"


async def test_the_lexical_dependence_gap_is_zero_on_the_vector_arm(
    geometry: tuple,
) -> None:
    """`recall(T0) - recall(T2)`, the headline number of this family.

    With only the vector arm running, a memory sharing NO words with the query
    must be as retrievable as a verbatim one, since cosine is all that is being
    consulted. A non-zero gap here means something lexical leaked into a path
    configured not to use it.
    """
    service, org, _space, embedder = geometry
    recalls = {}
    for tier, text, cosine in (TIERS[0], TIERS[2]):
        space_n = await service.create_space(org.id, slug=f"gap{tier.lower()}", name=tier)
        facts = [Fact(gold_id="gold", text=f"gold {text}", vector=at_cosine(cosine, off=1))]
        facts += [
            Fact(gold_id=f"d{i}", text=f"d{i} filler", vector=at_cosine(0.35, off=2 + i))
            for i in range(8)
        ]
        corpus = build(
            facts,
            [
                Query(
                    key="q",
                    text="the vorn kelmady arrangement was signed",
                    vector=axis(ANCHOR),
                    gold_ids=("gold",),
                )
            ],
        )
        register(corpus, embedder)
        await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
        response = await service.search(
            SearchRequest(
                query="the vorn kelmady arrangement was signed",
                org_id=org.id,
                space_id=space_n.id,
                limit=10,
                **GEOMETRY,
            )
        )
        returned = [hit.memory.id for hit in response.results]
        recalls[tier] = recall_at_k(returned, [corpus.fact("gold").memory_id], 10)

    gap = recalls["T0_verbatim"] - recalls["T2_paraphrase"]
    assert gap == 0.0, f"lexical-dependence gap {gap}: {recalls}"


# -- B2: the similarity floor ---------------------------------------------


@pytest.mark.parametrize("gold_cosine", [0.95, 0.80, 0.65, 0.50])
async def test_the_gold_outranks_a_lower_band(geometry: tuple, gold_cosine: float) -> None:
    """Stratified by similarity, so the floor is a measurement not a guess.

    The distractors sit 0.15 below the gold at every level. Rank 1 must hold all
    the way down, because the ordering is what cosine is for -- if it fails at
    0.50 the system has an absolute floor nobody documented.
    """
    facts = [
        Fact(
            gold_id="gold",
            text="gold the kelmady arrangement",
            vector=at_cosine(gold_cosine, off=1),
        )
    ]
    facts += [
        Fact(
            gold_id=f"d{i}",
            text=f"d{i} filler",
            vector=at_cosine(gold_cosine - 0.15, off=2 + i),
        )
        for i in range(6)
    ]
    gold, _corpus = await _ranked(geometry, facts, "the kelmady arrangement")
    assert gold and gold[0] == "gold", f"gold at {gold_cosine} did not rank first: {gold}"


# -- B3: the distractor gradient ------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 4, 8, 16, 64])
async def test_the_gold_holds_rank_one_against_n_hard_negatives(
    geometry: tuple, count: int
) -> None:
    """Chroma's finding was that ONE distractor already hurts.

    Here they are hard: 0.85 against a gold at 0.90, a five-point margin, and
    there can be sixty-four of them. Rank must not move. 127 orthogonal
    off-directions is exactly why `DIMS` is 128.
    """
    facts = [
        Fact(
            gold_id="gold",
            text="gold the kelmady arrangement",
            vector=at_cosine(GOLD_COSINE, off=1),
        )
    ]
    facts += [
        Fact(gold_id=f"d{i}", text=f"d{i} near miss about kelmady", vector=vector)
        for i, vector in enumerate(band(0.85, count, first_off=2))
    ]
    gold, corpus = await _ranked(geometry, facts, "the kelmady arrangement", limit=70)
    assert gold and gold[0] == "gold", f"{count} hard negatives displaced the gold"
    assert rank_of(corpus.gold_of([f.memory_id for f in corpus.facts]), "gold") is not None


# -- B4: noise type asymmetry ---------------------------------------------


async def test_random_noise_does_not_move_the_gold_but_near_misses_are_the_test(
    geometry: tuple,
) -> None:
    """The Power-of-Noise result, inverted into an assertion.

    Two arms at equal corpus size: one filled with vectors far from the query,
    one with near-misses just below the gold. Random noise must not move the
    gold's rank at all -- if it does, the store has a density or normalization
    problem independent of semantics. Near-misses are allowed to compress the
    margin, and that is the only thing that should.
    """
    service, org, _space, embedder = geometry
    ranks = {}
    for label, distractor_cosine in (("random", 0.10), ("near_miss", 0.87)):
        space_n = await service.create_space(org.id, slug=f"noise{label}", name=label)
        facts = [
            Fact(
                gold_id="gold",
                text="gold the kelmady arrangement",
                vector=at_cosine(GOLD_COSINE, off=1),
            )
        ]
        facts += [
            Fact(
                gold_id=f"d{i}",
                text=f"d{i} filler",
                vector=at_cosine(distractor_cosine, off=2 + i),
            )
            for i in range(12)
        ]
        corpus = build(
            facts,
            [
                Query(
                    key="q",
                    text="the kelmady arrangement",
                    vector=axis(ANCHOR),
                    gold_ids=("gold",),
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
                limit=20,
                **GEOMETRY,
            )
        )
        returned = corpus.gold_of([hit.memory.id for hit in response.results])
        ranks[label] = rank_of(returned, "gold")

    assert ranks["random"] == 1, f"random noise moved the gold to rank {ranks['random']}"
    assert ranks["near_miss"] == 1, f"near misses moved the gold to rank {ranks['near_miss']}"


# -- B5: haystack coherence ----------------------------------------------


async def test_a_coherent_haystack_is_no_worse_than_a_shuffled_one(
    geometry: tuple,
) -> None:
    """Context Rot found coherent haystacks worse in 18 of 18 models.

    At the store layer the filler's topical coherence is expressed as how close
    the distractors are to EACH OTHER, which `band` and `cone` control
    separately from how close they are to the query. Both arms hold the
    distractors at 0.60 from the query; the coherent arm additionally clusters
    them tightly together.
    """
    service, org, _space, embedder = geometry
    ranks = {}
    arms = {
        "shuffled": band(0.60, 10, first_off=2),
        "coherent": cone(0.60, 0.95, 10, first_off=2),
    }
    for label, vectors in arms.items():
        space_n = await service.create_space(org.id, slug=f"coh{label}", name=label)
        facts = [
            Fact(
                gold_id="gold",
                text="gold the kelmady arrangement",
                vector=at_cosine(GOLD_COSINE, off=1),
            )
        ]
        facts += [
            Fact(gold_id=f"d{i}", text=f"d{i} filler", vector=v) for i, v in enumerate(vectors)
        ]
        corpus = build(
            facts,
            [
                Query(
                    key="q",
                    text="the kelmady arrangement",
                    vector=axis(ANCHOR),
                    gold_ids=("gold",),
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
                limit=20,
                **GEOMETRY,
            )
        )
        ranks[label] = rank_of(corpus.gold_of([h.memory.id for h in response.results]), "gold")

    assert ranks["coherent"] == ranks["shuffled"] == 1, ranks


# -- B6: near-duplicate flooding -----------------------------------------


@pytest.mark.parametrize("copies", [0, 3, 10])
async def test_a_near_duplicate_flood_does_not_eat_the_window(
    geometry: tuple, copies: int
) -> None:
    """RULER's "duplicated answers without the complete set".

    One gold plus `copies` near-duplicates of a DIFFERENT fact, all clustered at
    0.88 from the query and 0.95 from each other. `recall@k` cannot see this
    failure -- ten copies of one fact score perfectly. `unique_information@k`
    counts distinct FACTS, which is what a caller actually receives.
    """
    facts = [
        Fact(
            gold_id="gold",
            text="gold the kelmady arrangement",
            vector=at_cosine(GOLD_COSINE, off=1),
        ),
        Fact(
            gold_id="other", text="other the thraskin schedule", vector=at_cosine(0.70, off=90)
        ),
    ]
    facts += [
        Fact(gold_id=f"dup{i}", text=f"dup{i} the thraskin schedule restated", vector=v)
        for i, v in enumerate(cone(0.88, 0.95, copies, first_off=2))
    ]
    service, org, space, embedder = geometry
    query = Query(
        key="q", text="the kelmady arrangement", vector=axis(ANCHOR), gold_ids=("gold",)
    )
    corpus = build(facts, [query])
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space.id)
    response = await service.search(
        SearchRequest(
            query="the kelmady arrangement",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            **GEOMETRY,
        )
    )
    returned = [hit.memory.id for hit in response.results]
    assert corpus.fact("gold").memory_id in returned, (
        f"{copies} near-duplicates crowded out the gold"
    )
    # Distinct facts, not distinct rows: with copies present the top-5 must still
    # carry more than one fact's worth of information.
    distinct = unique_information_at_k(returned, corpus.fact_of, 5)
    assert distinct >= 2 or len(returned) < 2, f"top-5 carried {distinct} distinct facts"


# -- the embedder's own character, asserted rather than assumed -----------


async def test_the_repo_embedder_is_lexical_not_semantic() -> None:
    """A regression guard, and an honest label on the harness.

    `DeterministicEmbedder` is blake2b feature hashing over word and character
    n-grams. It has no notion of meaning, so "car" is closer to "cart" than to
    "automobile" -- and every capability family above constructs its vectors
    precisely because of that.

    Asserted rather than left implicit: if someone swaps in a semantic embedder,
    retrieval behaviour changes everywhere and this is the test that says so
    first.
    """
    from mapi.domain.embeddings.base import cosine_similarity
    from mapi.domain.embeddings.deterministic import DeterministicEmbedder

    embedder = DeterministicEmbedder(dimensions=128)
    car, automobile, cart = (await embedder.embed(["car", "automobile", "cart"])).vectors

    assert cosine_similarity(car, cart) > cosine_similarity(car, automobile), (
        "the embedder now prefers a synonym over a shared prefix -- it has become "
        "semantic, and the capability families' constructed vectors should be "
        "revisited"
    )
