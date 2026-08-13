"""Family D: answers that need a chain, and the gate that keeps the tests honest.

Multi-hop QA is where retrieval and reasoning are easiest to confuse. MuSiQue's
authors built a "DiRe" probe because they found models answering supposedly
multi-hop questions from a single hop -- the datasets leaked. FRAMES measured a
+32-point gap between naive and oracle retrieval on multi-step questions. MemGPT's
nested key-value task, where the value of one key IS the next key, put GPT-4 at
0% at three levels of nesting.

D3 is the important one and it is not a test of the product. It is a test of the
TESTS: delete the bridge memory and check that the answer memory is no longer
retrievable at rank 1. If it still is, the case was secretly single-hop and every
other assertion in the file was measuring nothing. It runs at corpus-construction
time on every case, which is the only place it can catch a bad case before that
case produces a reassuring green tick.

What this family can and cannot show. The store has no iterative retrieval loop,
so "multi-hop" here means: **is the whole evidence chain present in one top-k?**
That is `joint_recall@k` over the chain, and it is the quantity a reader needs --
a chain missing its middle link cannot be followed by any model. D4 simulates the
agentic loop deterministically (concatenate the rank-1 text onto the query, search
again) to show whether the embedding space would support one at all.
"""

from __future__ import annotations

import pytest
from tests.support.corpus import Fact, Query, build, load, register
from tests.support.harness import GEOMETRY
from tests.support.metrics import joint_recall_at_k, rank_of, sufficient_at_k
from tests.support.vectors import ANCHOR, at_cosine, axis

from mapi.domain.retrieval.pipeline import SearchRequest

#: A chain: the query names A, hop 1 links A to B, hop 2 links B to C, and the
#: answer hangs off the last link. Each hop is less similar to the QUERY than the
#: one before it, which is exactly why multi-hop is hard for a single retrieval
#: pass -- the later links look less relevant to what was asked.
HOP_COSINES = (0.90, 0.62, 0.48, 0.40)

#: Filler sits ABOVE every hop after the first, and that is the whole point.
#:
#: The first version put filler at 0.30, far below the later links, and D3 caught
#: it: with the bridge removed the answer link still ranked first, because nothing
#: in the corpus was competitive with it. A chain built that way is not multi-hop
#: at all -- the query reaches the answer directly, the intervening links are
#: decoration, and every joint-recall assertion over it is vacuous.
#:
#: For a case to be genuinely multi-hop, each later link must be
#: INDISTINGUISHABLE from filler by similarity to the query. Then the only way to
#: reach it is through the bridge, which is what "multi-hop" means.
FILLER_COSINE = 0.70

#: Few enough that a generous k still admits the whole chain past them. With four
#: fillers above hops 1-3, the chain occupies ranks 1 and 6-8 of a 10-result
#: window -- present, but only because k is generous, which is the honest picture.
FILLER_COUNT = 4


def _chain(hops: int) -> list[Fact]:
    """`hops` links plus filler. Link i is at HOP_COSINES[i] from the query."""
    names = ["vorn", "kelmady", "thraskin", "quolmp", "zhoulk"]
    facts = []
    for i in range(hops):
        facts.append(
            Fact(
                gold_id=f"hop{i}",
                text=f"hop{i} {names[i]} is associated with {names[i + 1]}",
                vector=at_cosine(HOP_COSINES[i], off=1 + i),
            )
        )
    facts += [
        Fact(
            gold_id=f"f{i}",
            text=f"f{i} unrelated plemmoth material number {i}",
            vector=at_cosine(FILLER_COSINE, off=20 + i),
        )
        for i in range(FILLER_COUNT)
    ]
    return facts


async def _load_chain(geometry: tuple, hops: int, *, slug: str) -> tuple[object, object]:
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug=slug, name=slug)
    facts = _chain(hops)
    query = Query(
        key="q",
        text="what is vorn ultimately associated with",
        vector=axis(ANCHOR),
        gold_ids=tuple(f"hop{i}" for i in range(hops)),
    )
    corpus = build(facts, [query])
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
    return corpus, space_n


async def _search(
    geometry: tuple, space_n: object, query: str, *, limit: int = 10
) -> list[str]:
    service, org, _space, _embedder = geometry
    response = await service.search(
        SearchRequest(
            query=query,
            org_id=org.id,
            space_id=space_n.id,  # type: ignore[attr-defined]
            limit=limit,
            **GEOMETRY,
        )
    )
    return [hit.memory.id for hit in response.results]


# -- D3: the suite-validity gate. FIRST, because it validates the rest -----


@pytest.mark.parametrize("hops", [2, 3, 4])
async def test_a_case_is_genuinely_multi_hop(geometry: tuple, hops: int) -> None:
    """The DiRe probe: delete the bridge and the answer must become unreachable.

    Run BEFORE any multi-hop claim, because a case that is secretly single-hop
    produces a green tick that means nothing. MuSiQue's authors found exactly this
    contamination in existing datasets -- questions answerable from one hop while
    labelled as needing several.

    Concretely: with the middle link removed, the LAST link must not be rank 1 for
    the original query. If it is, the query reaches the answer directly and the
    intervening hops are decoration.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug=f"gate{hops}", name=f"gate{hops}")

    # Load the chain WITHOUT its first link -- the one the query names.
    facts = [f for f in _chain(hops) if f.gold_id != "hop0"]
    corpus = build(
        facts,
        [
            Query(
                key="q",
                text="what is vorn ultimately associated with",
                vector=axis(ANCHOR),
                gold_ids=tuple(f"hop{i}" for i in range(1, hops)),
            )
        ],
    )
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)

    returned = await _search(geometry, space_n, "what is vorn ultimately associated with")
    gold = corpus.gold_of(returned)
    last = f"hop{hops - 1}"
    assert not gold or gold[0] != last, (
        f"with hop0 removed, {last} still ranks first -- this case is not "
        "genuinely multi-hop and every assertion built on it is vacuous"
    )


# -- D1 / D2: joint recall over the chain, by hop count --------------------


@pytest.mark.parametrize("hops", [1, 2, 3, 4])
async def test_the_whole_chain_is_retrieved_in_one_pass(geometry: tuple, hops: int) -> None:
    """`joint_recall@k` over every link. The quantity a reader needs.

    A chain missing its middle link cannot be followed by any model, however
    capable -- so partial recall is not partial success. k is generous (10 over a
    14-memory corpus) because the question is whether the links are REACHABLE,
    not whether they outrank the filler.
    """
    corpus, space_n = await _load_chain(geometry, hops, slug=f"d1h{hops}")
    returned = await _search(geometry, space_n, "what is vorn ultimately associated with")
    required = corpus.memory_ids([f"hop{i}" for i in range(hops)])  # type: ignore[attr-defined]
    assert joint_recall_at_k(returned, required, 10) == 1.0, (
        f"{hops}-hop chain incomplete: retrieved {sorted(corpus.gold_of(returned))}"  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("hops", [2, 3, 4])
async def test_sufficiency_is_reported_per_hop_count(geometry: tuple, hops: int) -> None:
    """Split by hops, which is where the failure localizes.

    "2-hop sufficiency 0.94, 3-hop 0.41" is actionable; a single aggregate over
    all hop counts hides which depth broke.
    """
    corpus, space_n = await _load_chain(geometry, hops, slug=f"d2h{hops}")
    returned = await _search(geometry, space_n, "what is vorn ultimately associated with")
    required = corpus.memory_ids([f"hop{i}" for i in range(hops)])  # type: ignore[attr-defined]
    assert sufficient_at_k(returned, required, 10)


async def test_the_later_hops_rank_below_the_first(geometry: tuple) -> None:
    """Why multi-hop is hard, as a property rather than an assumption.

    The chain is built so each link is less similar to the QUERY than the one
    before. That ordering must show up in the ranks -- it is what makes a naive
    top-k truncate a chain, and it is the reason `joint_recall` at a small k
    would fail where a generous k succeeds.
    """
    corpus, space_n = await _load_chain(geometry, 4, slug="d2order")
    returned = await _search(geometry, space_n, "what is vorn ultimately associated with")
    gold = corpus.gold_of(returned)  # type: ignore[attr-defined]
    ranks = [rank_of(gold, f"hop{i}") for i in range(4)]
    present = [r for r in ranks if r is not None]
    assert present == sorted(present), f"hop ranks not monotonic in chain order: {ranks}"


async def test_a_short_k_truncates_a_long_chain(geometry: tuple) -> None:
    """The failure this family exists to measure, made explicit.

    At k=2 a 4-hop chain cannot be complete. Asserted so the metric is shown
    discriminating rather than only ever passing -- a `joint_recall` that never
    fails is not measuring anything.
    """
    corpus, space_n = await _load_chain(geometry, 4, slug="d2short")
    returned = await _search(
        geometry, space_n, "what is vorn ultimately associated with", limit=2
    )
    required = corpus.memory_ids([f"hop{i}" for i in range(4)])  # type: ignore[attr-defined]
    assert joint_recall_at_k(returned, required, 2) == 0.0


# -- D4: the agentic loop, simulated without a model ----------------------


async def test_a_second_pass_seeded_with_the_first_hit_reaches_further(
    geometry: tuple,
) -> None:
    """Whether the embedding space would support iterative retrieval at all.

    Round 1 is the query. Round 2 is the query CONCATENATED with the rank-1
    memory's text -- a deterministic rewrite, no model. If hop 2's rank improves,
    an agentic loop has something to work with; if it does not, adding one would
    not help and the answer is a graph traversal instead.

    Registered explicitly because the rewritten query is a new string, and the
    autouse miss check would otherwise catch it landing on the sentinel axis.
    """
    _service, _org, _space, embedder = geometry
    corpus, space_n = await _load_chain(geometry, 3, slug="d4")

    first = await _search(geometry, space_n, "what is vorn ultimately associated with")
    gold_first = corpus.gold_of(first)  # type: ignore[attr-defined]
    assert gold_first, "round one retrieved nothing"

    # The rewritten query sits closer to hop1 than the original did: it now
    # carries hop0's text, which names the bridge entity.
    rewritten = "what is vorn ultimately associated with hop0 vorn is associated with kelmady"
    embedder.register(rewritten, at_cosine(0.95, off=2))

    second = await _search(geometry, space_n, rewritten)
    gold_second = corpus.gold_of(second)  # type: ignore[attr-defined]

    before = rank_of(gold_first, "hop1")
    after = rank_of(gold_second, "hop1")
    assert after is not None, "the second pass lost hop1 entirely"
    assert before is not None
    assert after <= before, f"seeding with the first hit made hop1 worse: {before} -> {after}"
