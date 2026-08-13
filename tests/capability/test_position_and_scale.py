"""Family A: does WHERE a memory sits change whether it comes back?

From the needle-in-a-haystack literature and its descendants. NIAH found
retrieval accuracy is a function of (context length x needle depth), worst at
7-50% depth; "Lost in the Middle" found a gold document in the middle can score
BELOW the no-document baseline; RULER found only half of models claiming 32K
contexts actually hold 32K; NoLiMa found effective context is 2K-8K once lexical
overlap is removed.

Those are all claims about a MODEL reading a prompt. The store-level analogue is
sharper and nobody had tested it: a memory's rank must depend on its content and
nothing else. Not on when it was written, not on how many neighbours it has, not
on where in a long memory the matching sentence sits.

WHY THIS IS THE FAMILY THAT NEEDS CONSTRUCTED VECTORS MOST. Every test here holds
similarity FIXED and varies something else -- position, corpus size, chunk offset.
With a hash embedder you cannot hold similarity fixed while changing the text, so
the experiment is impossible: any rank change could be the position or could be
the n-grams. `at_cosine` makes the gold's similarity identical across every arm
by construction, so a rank difference has exactly one possible cause.

Nothing here asserts a number from a paper. The assertions are invariances --
"rank does not move" -- because for a store, unlike for a model, position
sensitivity is not a limitation to be characterised. It is a bug.
"""

from __future__ import annotations

import pytest
from tests.support.corpus import Fact, Query, build, load, register
from tests.support.harness import GEOMETRY
from tests.support.metrics import effective_x, rank_of
from tests.support.vectors import ANCHOR, at_cosine, axis

from mapi.domain.retrieval.pipeline import SearchRequest

#: The gold sits here; everything else is filler at a clearly lower cosine. The
#: gap is wide (0.90 vs 0.55) so nothing in these tests is decided by a margin
#: that float noise could close.
GOLD_COSINE = 0.90
FILLER_COSINE = 0.55


def _corpus(gold_at: int, size: int, seed: int = 1) -> object:
    """A corpus of `size` facts with the gold inserted at index `gold_at`.

    Insertion order is what varies; content and geometry do not. Filler facts
    all sit at the same cosine as each other, so the only thing distinguishing
    the gold is its similarity -- which is identical in every arm.
    """
    facts = []
    for i in range(size):
        if i == gold_at:
            facts.append(
                Fact(
                    gold_id="gold",
                    text="fact gold: the vorn kelmady arrangement",
                    vector=at_cosine(GOLD_COSINE, off=1),
                )
            )
        else:
            facts.append(
                Fact(
                    gold_id=f"f{i:04d}",
                    text=f"fact f{i:04d}: unrelated filler about quolmp {i}",
                    vector=at_cosine(FILLER_COSINE, off=2 + (i % 100)),
                )
            )
    query = Query(
        key="q", text="the vorn kelmady arrangement", vector=axis(ANCHOR), gold_ids=("gold",)
    )
    return build(facts, [query], seed=seed)


async def _rank_of_gold(geometry: tuple, corpus: object, limit: int = 20) -> int | None:
    service, org, space, embedder = geometry
    register(corpus, embedder)  # type: ignore[arg-type]
    await load(service.store, corpus, org_id=org.id, space_id=space.id)  # type: ignore[arg-type]
    response = await service.search(
        SearchRequest(
            query=corpus.queries["q"].text,  # type: ignore[attr-defined]
            org_id=org.id,
            space_id=space.id,
            limit=limit,
            **GEOMETRY,
        )
    )
    returned = [hit.memory.id for hit in response.results]
    return rank_of(returned, corpus.fact("gold").memory_id)  # type: ignore[attr-defined]


# -- A1: insertion position ------------------------------------------------


@pytest.mark.parametrize("fraction", [0.0, 0.07, 0.25, 0.5, 0.75, 0.93, 1.0])
async def test_insertion_position_does_not_change_rank(
    geometry: tuple, fraction: float
) -> None:
    """NIAH's depth sweep, at the store layer.

    The gold is written first, last, or anywhere between, with identical content
    and identical similarity. Its rank must be 1 every time. A U-shape here --
    the model-side finding -- would mean insertion order leaked into ranking,
    which for a store is not a characteristic but a defect.
    """
    size = 30
    position = min(int(fraction * (size - 1)), size - 1)
    rank = await _rank_of_gold(geometry, _corpus(position, size))
    assert rank == 1, f"gold at insertion position {position}/{size} ranked {rank}"


async def test_the_rank_spread_across_positions_is_zero(geometry: tuple) -> None:
    """The aggregate form, which is what a regression would show up in.

    Run as one test over one service so the comparison is within a single
    corpus-shaped run rather than across fixtures.
    """
    service, org, _space, embedder = geometry
    ranks = []
    for position in (0, 5, 15, 29):
        # Fresh space per arm: the same gold id in one space would dedupe.
        space_n = await service.create_space(
            org.id, slug=f"pos{position}", name=f"Position {position}"
        )
        corpus = _corpus(position, 30)
        register(corpus, embedder)
        await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
        response = await service.search(
            SearchRequest(
                query=corpus.queries["q"].text,
                org_id=org.id,
                space_id=space_n.id,
                limit=20,
                **GEOMETRY,
            )
        )
        returned = [hit.memory.id for hit in response.results]
        ranks.append(rank_of(returned, corpus.fact("gold").memory_id))
    assert set(ranks) == {1}, f"rank varied with insertion position: {ranks}"


# -- A3: corpus size -------------------------------------------------------


@pytest.mark.parametrize("size", [10, 50, 200])
async def test_the_gold_survives_a_growing_haystack(geometry: tuple, size: int) -> None:
    """RULER's "claimed versus effective" question, restated for a store.

    Same gold, same similarity, more filler. The filler is all at 0.55 and the
    gold at 0.90, so no amount of it should displace the gold -- and if it does,
    the cause is crowding in the candidate fetch rather than similarity.
    """
    rank = await _rank_of_gold(geometry, _corpus(size // 2, size))
    assert rank == 1, f"gold ranked {rank} in a corpus of {size}"


async def test_effective_corpus_size_is_reported_not_assumed(geometry: tuple) -> None:
    """The metric, computed rather than asserted at a guessed value.

    `effective_x` is NoLiMa's definition -- the largest sweep point still holding
    85% of the baseline. Reported as a floor so the number is visible in the
    ledger and a regression fails, without pinning an exact value that any
    unrelated ranking change would move.
    """
    service, org, _space, embedder = geometry
    curve: list[tuple[float, float]] = []
    for size in (10, 50, 200):
        space_n = await service.create_space(org.id, slug=f"eff{size}", name=f"Eff {size}")
        corpus = _corpus(size // 2, size)
        register(corpus, embedder)
        await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
        response = await service.search(
            SearchRequest(
                query=corpus.queries["q"].text,
                org_id=org.id,
                space_id=space_n.id,
                limit=10,
                **GEOMETRY,
            )
        )
        returned = [hit.memory.id for hit in response.results]
        found = 1.0 if corpus.fact("gold").memory_id in returned else 0.0
        curve.append((size, found))

    effective = effective_x(curve)
    assert effective is not None, f"gold not retrievable even at the smallest size: {curve}"
    assert effective >= 200, f"effective corpus size only {effective}: {curve}"


# -- A4: position WITHIN one memory ---------------------------------------


@pytest.mark.parametrize("offset", [0, 1, 3, 7])
async def test_the_matching_chunk_can_sit_anywhere_in_a_long_memory(
    geometry: tuple, offset: int
) -> None:
    """The embedding-side position bias, and the one that is genuinely ours.

    A long memory is chunked, and only one chunk holds the answer. Published
    findings put dense retrievers down 15.6% when the relevant span moves later
    in a passage, and embedding cosine down 7-12% for start-edits.

    Here the matching chunk is placed at ordinal 0, 1, 3 or 7 of an eight-chunk
    memory, at identical similarity. The memory must be found regardless --
    `vector_search` scores a memory by its BEST chunk, so chunk ordinal must not
    matter, and if it did the failure would look like "long memories are worse"
    rather than like a position bug.
    """
    from tests.support.factories import memory as build_memory
    from tests.support.factories import tenant

    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug=f"chunk{offset}", name="Chunked")
    assert tenant is not None  # imported for symmetry with other families

    texts = [f"paragraph {i} about quolmp filler material" for i in range(8)]
    texts[offset] = "paragraph holding the vorn kelmady arrangement"
    vectors = [at_cosine(FILLER_COSINE, off=10 + i) for i in range(8)]
    vectors[offset] = at_cosine(GOLD_COSINE, off=1)

    stored = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="\n\n".join(texts),
        chunk_texts=texts,
        chunk_vectors=vectors,
    )
    await service.store.upsert_memory(stored)
    embedder.register("the vorn kelmady arrangement", axis(ANCHOR))

    response = await service.search(
        SearchRequest(
            query="the vorn kelmady arrangement",
            org_id=org.id,
            space_id=space_n.id,
            limit=5,
            **GEOMETRY,
        )
    )
    assert [hit.memory.id for hit in response.results] == [stored.id], (
        f"a memory whose matching chunk is at ordinal {offset} was not found"
    )


async def test_the_matched_chunk_is_the_one_that_matched(geometry: tuple) -> None:
    """Not just the memory -- the right chunk.

    `matched_text` is what a caller shows a user and what chat quotes. Returning
    the memory with the wrong chunk attached would cite paragraph 1 for a fact
    that is in paragraph 7.
    """
    from tests.support.factories import memory as build_memory

    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="matched", name="Matched")

    texts = [f"paragraph {i} about quolmp filler" for i in range(6)]
    texts[4] = "paragraph holding the vorn kelmady arrangement"
    vectors = [at_cosine(FILLER_COSINE, off=20 + i) for i in range(6)]
    vectors[4] = at_cosine(GOLD_COSINE, off=1)

    stored = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="\n\n".join(texts),
        chunk_texts=texts,
        chunk_vectors=vectors,
    )
    await service.store.upsert_memory(stored)
    embedder.register("the vorn kelmady arrangement", axis(ANCHOR))

    response = await service.search(
        SearchRequest(
            query="the vorn kelmady arrangement",
            org_id=org.id,
            space_id=space_n.id,
            limit=5,
            **GEOMETRY,
        )
    )
    assert response.results
    assert "vorn kelmady" in response.results[0].matched_text


# -- A5: the silent-truncation canary --------------------------------------


async def test_a_fact_at_the_very_end_of_a_long_memory_is_still_found(
    geometry: tuple,
) -> None:
    """The LongEmbed failure: the embedder is the real context limit.

    A provider that silently truncates its input drops the tail of a long
    document, and the signature is a cliff to zero recall at a particular length
    -- not a gradual decline. `ScriptedEmbedder` cannot truncate, so what this
    actually verifies is that OUR chunking reaches the end of a long memory and
    that the last chunk is searchable. The canary is worth having in place: swap
    in a real provider and the same test becomes the truncation detector.
    """
    from tests.support.factories import memory as build_memory

    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="tail", name="Tail")

    texts = [f"section {i} of a long document about quolmp" for i in range(40)]
    texts[-1] = "the final section names the vorn kelmady arrangement"
    vectors = [at_cosine(FILLER_COSINE, off=30 + (i % 90)) for i in range(40)]
    vectors[-1] = at_cosine(GOLD_COSINE, off=1)

    stored = build_memory(
        org_id=org.id,
        space_id=space_n.id,
        content="\n\n".join(texts),
        chunk_texts=texts,
        chunk_vectors=vectors,
    )
    await service.store.upsert_memory(stored)
    embedder.register("the vorn kelmady arrangement", axis(ANCHOR))

    response = await service.search(
        SearchRequest(
            query="the vorn kelmady arrangement",
            org_id=org.id,
            space_id=space_n.id,
            limit=5,
            **GEOMETRY,
        )
    )
    assert [hit.memory.id for hit in response.results] == [stored.id]
    assert "final section" in response.results[0].matched_text
