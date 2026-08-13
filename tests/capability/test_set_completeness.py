"""Family C: when one missing row makes the answer wrong.

The multi-needle literature's central finding is that SET COMPLETENESS collapses
long before single-item recall does, and asymmetrically by position -- LangChain's
multi-needle work found the tail of the context retrieved and the head dropped.
RULER found sets returned with duplicates and gaps.

For a memory API this is the difference between a metric that flatters and one
that tells the truth. A question needing four facts that gets three scores
`recall@k` 0.75, which reads like near-success. The answer is a wrong count. There
is no partial credit for an incomplete evidence set, so the metric must not offer
any -- hence `joint_recall@k`, which is 1.0 only when every required row is
present and 0.0 otherwise.

The practical payoff is C3: it finds the (M, k) frontier where coverage stops
being complete. That number tells you when top-k truncation has made correct
aggregation IMPOSSIBLE downstream, no matter how good the reader is -- which is
the point at which the answer is an aggregation endpoint rather than a better
ranker.
"""

from __future__ import annotations

import pytest
from tests.support.corpus import Fact, Query, build, load, register
from tests.support.harness import GEOMETRY
from tests.support.metrics import joint_recall_at_k, recall_at_k
from tests.support.vectors import ANCHOR, at_cosine, axis, band

from mapi.domain.retrieval.pipeline import SearchRequest

#: Every member of the gold set sits at the same cosine, because a question
#: needing all of them has no reason to prefer one. That is also the hardest
#: case for a ranker: nothing distinguishes the members, so any dropped one is
#: dropped arbitrarily.
MEMBER_COSINE = 0.85
FILLER_COSINE = 0.40


async def _load_set(
    geometry: tuple, members: int, *, slug: str, fillers: int = 10
) -> tuple[object, object]:
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug=slug, name=slug)
    facts = [
        Fact(gold_id=f"g{i:03d}", text=f"g{i:03d} expense entry for March, item {i}", vector=v)
        for i, v in enumerate(band(MEMBER_COSINE, members, first_off=1))
    ]
    facts += [
        Fact(
            gold_id=f"f{i:03d}",
            text=f"f{i:03d} unrelated quolmp material",
            vector=at_cosine(FILLER_COSINE, off=1 + members + i),
        )
        for i in range(fillers)
    ]
    query = Query(
        key="q",
        text="every expense entry for March",
        vector=axis(ANCHOR),
        gold_ids=tuple(f"g{i:03d}" for i in range(members)),
    )
    corpus = build(facts, [query])
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)
    return corpus, space_n


async def _returned(geometry: tuple, space_n: object, *, limit: int) -> list[str]:
    service, org, _space, _embedder = geometry
    response = await service.search(
        SearchRequest(
            query="every expense entry for March",
            org_id=org.id,
            space_id=space_n.id,  # type: ignore[attr-defined]
            limit=limit,
            **GEOMETRY,
        )
    )
    return [hit.memory.id for hit in response.results]


# -- C1: the completeness curve -------------------------------------------


@pytest.mark.parametrize("members", [1, 2, 4, 8, 16])
async def test_a_gold_set_is_complete_at_k_equals_n(geometry: tuple, members: int) -> None:
    """`joint_recall@n` for a set of n. The baseline claim.

    If a question needs n facts and the caller asks for exactly n, all n must
    come back -- the members outscore every filler by 0.45 of cosine, so nothing
    but a ranking bug can drop one.
    """
    corpus, space_n = await _load_set(geometry, members, slug=f"c1n{members}")
    returned = await _returned(geometry, space_n, limit=members)
    required = corpus.memory_ids([f"g{i:03d}" for i in range(members)])  # type: ignore[attr-defined]
    assert joint_recall_at_k(returned, required, members) == 1.0, (
        f"{members}-member set incomplete at k={members}: "
        f"{recall_at_k(returned, required, members):.2f} recall"
    )


@pytest.mark.parametrize("members", [4, 8, 16])
async def test_completeness_holds_at_k_equals_2n(geometry: tuple, members: int) -> None:
    """Headroom does not hurt. The multi-needle finding was that MORE context
    made completeness worse, so doubling k must not lose a member."""
    corpus, space_n = await _load_set(geometry, members, slug=f"c1x2n{members}")
    returned = await _returned(geometry, space_n, limit=members * 2)
    required = corpus.memory_ids([f"g{i:03d}" for i in range(members)])  # type: ignore[attr-defined]
    assert joint_recall_at_k(returned, required, members * 2) == 1.0


async def test_the_completeness_capacity_is_reported(geometry: tuple) -> None:
    """The number this family exists to produce.

    "Completeness capacity" is the largest n where `joint_recall@2n` still holds.
    Asserted as a floor rather than an exact value, and printed on failure, so a
    regression fails without an unrelated ranking change failing the build.
    """
    curve: list[tuple[int, float]] = []
    for members in (2, 4, 8, 16):
        corpus, space_n = await _load_set(geometry, members, slug=f"cap{members}")
        returned = await _returned(geometry, space_n, limit=members * 2)
        required = corpus.memory_ids([f"g{i:03d}" for i in range(members)])  # type: ignore[attr-defined]
        curve.append((members, joint_recall_at_k(returned, required, members * 2)))

    complete = [n for n, score in curve if score == 1.0]
    assert complete, f"no set size was complete: {curve}"
    assert max(complete) >= 16, f"completeness capacity only {max(complete)}: {curve}"


async def test_one_missing_member_is_a_zero_not_a_high_score(geometry: tuple) -> None:
    """The metric's whole point, demonstrated rather than asserted abstractly.

    Ask for k=3 on a 4-member set. `recall@k` reads 0.75 -- which in a report
    looks like near-success -- and `joint_recall@k` reads 0.0, which is what a
    count over three of four facts actually is.
    """
    corpus, space_n = await _load_set(geometry, 4, slug="onemissing")
    returned = await _returned(geometry, space_n, limit=3)
    required = corpus.memory_ids([f"g{i:03d}" for i in range(4)])  # type: ignore[attr-defined]
    assert recall_at_k(returned, required, 3) == pytest.approx(0.75)
    assert joint_recall_at_k(returned, required, 3) == 0.0


# -- C2: is the dropped member the earliest-inserted? ---------------------


async def test_a_truncated_set_does_not_systematically_drop_the_earliest(
    geometry: tuple,
) -> None:
    """The multi-needle asymmetry: "retrieves the tail, ignores the head".

    Deliberately truncate an 8-member set to 4 and check WHICH four survive. If
    the survivors are always the last-inserted, insertion order is leaking into
    ranking -- a bug that a `joint_recall` number alone would report as "the set
    was incomplete" without saying why.
    """
    corpus, space_n = await _load_set(geometry, 8, slug="c2drop")
    returned = await _returned(geometry, space_n, limit=4)
    survivors = sorted(corpus.gold_of(returned))  # type: ignore[attr-defined]
    assert len(survivors) == 4
    # The members are equidistant from the query, so ties break on memory id --
    # which is ULID-ordered, i.e. insertion-ordered. What must NOT happen is the
    # reverse: the LAST four surviving while the first four are dropped.
    latest_four = sorted(f"g{i:03d}" for i in range(4, 8))
    assert survivors != latest_four, (
        "the four latest-inserted members survived and the four earliest were "
        f"dropped, which is the head-dropping bias: {survivors}"
    )


# -- C3: the coverage frontier -------------------------------------------


@pytest.mark.parametrize(("members", "limit"), [(20, 20), (20, 40), (40, 40), (60, 60)])
async def test_coverage_is_complete_across_the_m_by_k_frontier(
    geometry: tuple, members: int, limit: int
) -> None:
    """Where top-k truncation makes correct aggregation impossible.

    This is the test that tells you when the answer is an aggregation endpoint
    rather than a better ranker: if coverage cannot reach 1.0 at any k, no reader
    downstream can produce a correct count, however good it is.

    Also the practical regression test for the coverage-cap fix -- these sizes
    are above the old 32 ceiling.
    """
    corpus, space_n = await _load_set(geometry, members, slug=f"c3m{members}k{limit}")
    returned = await _returned(geometry, space_n, limit=limit)
    required = corpus.memory_ids([f"g{i:03d}" for i in range(members)])  # type: ignore[attr-defined]
    covered = len(set(returned) & set(required)) / members
    assert covered == 1.0, f"coverage {covered:.2f} at M={members}, k={limit}"


async def test_a_comprehensive_question_reaches_a_whole_large_set(
    geometry: tuple,
) -> None:
    """The coverage path, on a set larger than the old cap.

    A question phrased comprehensively must return the whole territory rather
    than a ranked few, and 60 members is where the pre-fix behaviour capped at
    32. Uses the shipped `coverage` decision rather than a large explicit limit,
    so it tests the feature and not just the parameter.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="c3cov", name="coverage")
    members = 60
    facts = [
        Fact(gold_id=f"g{i:03d}", text=f"g{i:03d} infrastructure component {i}", vector=v)
        for i, v in enumerate(band(MEMBER_COSINE, members, first_off=1))
    ]
    corpus = build(
        facts,
        [
            Query(
                key="q",
                text="list all of our infrastructure components",
                vector=axis(ANCHOR),
                gold_ids=tuple(f.gold_id for f in facts),
            )
        ],
    )
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)

    response = await service.search(
        SearchRequest(
            query="list all of our infrastructure components",
            org_id=org.id,
            space_id=space_n.id,
            limit=10,
            coverage=True,
            coverage_limit=100,
            use_rerank=False,
            use_decay=False,
            use_mmr=False,
            lexical_weight=0.0,
            tune_by_intent=False,
        )
    )
    assert len(response.results) == members, (
        f"a comprehensive question returned {len(response.results)} of {members}"
    )


# -- C4: cross-query non-interference ------------------------------------


async def test_two_queries_over_one_corpus_do_not_interfere(geometry: tuple) -> None:
    """RULER's multi-query needle test, restated.

    Two distinct queries, each with its own single gold, resolved against the
    same corpus. Each must find its own -- a shared cache or a mutated request
    object would show up as one query returning the other's answer.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="c4", name="multiquery")

    facts = [
        Fact(
            gold_id="alpha", text="alpha the kelmady arrangement", vector=at_cosine(0.90, off=1)
        ),
        Fact(gold_id="beta", text="beta the thraskin schedule", vector=at_cosine(0.90, off=2)),
    ]
    queries = [
        Query(
            key="qa",
            text="the kelmady arrangement",
            vector=at_cosine(0.99, off=1),
            gold_ids=("alpha",),
        ),
        Query(
            key="qb",
            text="the thraskin schedule",
            vector=at_cosine(0.99, off=2),
            gold_ids=("beta",),
        ),
    ]
    corpus = build(facts, queries)
    register(corpus, embedder)
    await load(service.store, corpus, org_id=org.id, space_id=space_n.id)

    for key, expected in (("qa", "alpha"), ("qb", "beta")):
        response = await service.search(
            SearchRequest(
                query=corpus.queries[key].text,
                org_id=org.id,
                space_id=space_n.id,
                limit=2,
                **GEOMETRY,
            )
        )
        top = corpus.gold_of([hit.memory.id for hit in response.results])
        assert top and top[0] == expected, f"{key} returned {top}, expected {expected} first"
