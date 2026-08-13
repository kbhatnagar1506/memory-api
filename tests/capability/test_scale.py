"""Family I: what happens as the store fills up, and what it costs.

The survey of memory-benchmark literature names this as one of five systematic
gaps: **no benchmark reports efficiency alongside accuracy.** Mem0 is the
exception and reports it as a headline (average retrieved tokens per call, p50/p95
search latency), and it is the right instinct -- a memory system that answers
perfectly at 25,000 tokens per query is a different product from one that answers
nearly as well at 7,000.

The other half is capacity degradation. Every accuracy number is quoted at some
corpus size, and the SLOPE matters more than the point: a system at 0.95 on 500
memories and 0.60 on 5,000 is worse than one flat at 0.85, and a single figure
cannot tell them apart.

MARKED SLOW, and the reason is not wall clock. Measured on this machine the whole
file runs in well under a second -- direct store writes are 0.04ms each, and a
3,000-memory sweep builds in 0.11s. It is marked because latency assertions are
machine-dependent by nature and have no business gating somebody's inner loop,
and because a multi-arm sweep is a measurement rather than an invariant. Run it
with `-m slow`; the default lane skips it.

WHAT THE NUMBERS MEAN HERE. `InMemoryStore.vector_search` is an exact scan, so
linear query cost in N is CORRECT rather than a defect -- and pinning it is how a
regression to something worse than linear gets noticed. Postgres uses HNSW and
would be sub-linear; the conformance suite is where the two are compared, and
`brute_force_topk` in the harness is the recall oracle for that comparison.
"""

from __future__ import annotations

import time

import pytest
from tests.support.factories import memory as build_memory
from tests.support.harness import GEOMETRY
from tests.support.metrics import effective_x, rank_of
from tests.support.vectors import ANCHOR, at_cosine, axis

from mapi.domain.retrieval.pipeline import SearchRequest

pytestmark = pytest.mark.slow

GOLD_COSINE = 0.92
FILLER_COSINE = 0.50

#: The sweep. Chosen so the largest is well past any corpus the unit tests use
#: and small enough that the whole file stays under a second.
SIZES = (100, 500, 1500, 3000)


async def _fill(geometry: tuple, size: int, *, slug: str) -> tuple[object, str]:
    """`size` memories with exactly one gold, written straight to the store.

    Direct writes rather than `service.ingest`: ingest interposes dedupe, a
    neighbours lookup, supersession, contradiction and association -- 7.4ms per
    write against 0.04ms -- and none of that is what a capacity sweep is
    measuring. The write path at scale gets its own test below.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug=slug, name=slug)
    embedder.register("the kelmady arrangement", axis(ANCHOR))

    gold_id = ""
    for i in range(size):
        is_gold = i == size // 2
        marker = "goldrow" if is_gold else f"f{slug}{i}"
        vector = (
            at_cosine(GOLD_COSINE, off=1)
            if is_gold
            else at_cosine(FILLER_COSINE, off=2 + (i % 120))
        )
        embedder.register_marker(marker, vector)
        stored = build_memory(
            org_id=org.id,
            space_id=space_n.id,
            content=f"{marker} corpus entry number {i}",
            vector=vector,
        )
        await service.store.upsert_memory(stored)
        if is_gold:
            gold_id = stored.id
    return space_n, gold_id


async def _timed_search(
    geometry: tuple, space_n: object, *, runs: int = 5
) -> tuple[list[str], float]:
    """Returns the ids and the mean per-query wall time in milliseconds."""
    service, org, _space, _embedder = geometry
    ids: list[str] = []
    started = time.perf_counter()
    for _ in range(runs):
        response = await service.search(
            SearchRequest(
                query="the kelmady arrangement",
                org_id=org.id,
                space_id=space_n.id,  # type: ignore[attr-defined]
                limit=10,
                **GEOMETRY,
            )
        )
        ids = [hit.memory.id for hit in response.results]
    elapsed = (time.perf_counter() - started) / runs * 1000
    return ids, elapsed


# -- I1: capacity degradation ---------------------------------------------


@pytest.mark.parametrize("size", SIZES)
async def test_the_gold_is_still_rank_one_at_every_size(geometry: tuple, size: int) -> None:
    """The slope, one point at a time.

    A single accuracy figure quoted at one corpus size says nothing about the
    curve. Same gold, same similarity, 30x more filler -- rank must not move,
    because nothing about the geometry changed.
    """
    space_n, gold_id = await _fill(geometry, size, slug=f"cap{size}")
    ids, _ms = await _timed_search(geometry, space_n)
    assert rank_of(ids, gold_id) == 1, f"gold fell to rank {rank_of(ids, gold_id)} at N={size}"


async def test_effective_corpus_size_covers_the_whole_sweep(geometry: tuple) -> None:
    """The curve as one reportable number, via NoLiMa's definition.

    `effective_x` is the largest sweep point still holding 85% of the baseline. If
    it comes back below the largest size, there is a degradation cliff and the
    number says where.
    """
    curve: list[tuple[float, float]] = []
    for size in SIZES:
        space_n, gold_id = await _fill(geometry, size, slug=f"eff{size}")
        ids, _ms = await _timed_search(geometry, space_n, runs=1)
        curve.append((size, 1.0 if rank_of(ids, gold_id) == 1 else 0.0))

    effective = effective_x(curve)
    assert effective == max(SIZES), (
        f"effective corpus size {effective}, not {max(SIZES)}: {curve}"
    )


# -- I3: latency ----------------------------------------------------------


async def test_query_cost_grows_no_worse_than_linearly(geometry: tuple) -> None:
    """An exact scan is linear, and linear is the contract to hold.

    `InMemoryStore.vector_search` scans every candidate, so O(N) is CORRECT here
    rather than a defect -- measured 4.3ms at 500 and 26.7ms at 3000, about
    8.9us per memory. What this catches is a regression to something WORSE than
    linear: an accidental quadratic in a filter or a per-result store round trip
    would show up as the ratio exploding.

    Generous tolerance (3x the linear expectation) because wall-clock timing on a
    shared machine is noisy, and a flaky performance test gets deleted.
    """
    timings: dict[int, float] = {}
    for size in (500, 3000):
        space_n, _gold = await _fill(geometry, size, slug=f"lat{size}")
        _ids, ms = await _timed_search(geometry, space_n)
        timings[size] = ms

    growth = timings[3000] / max(timings[500], 1e-9)
    linear = 3000 / 500
    assert growth < linear * 3, (
        f"query cost grew {growth:.1f}x for a 6x corpus, which is worse than linear: {timings}"
    )


async def test_a_coverage_query_is_not_pathologically_slower(geometry: tuple) -> None:
    """The cost the coverage fix incurred, measured rather than assumed.

    Lifting the hydration cap from 32 to `effective_limit` means a comprehensive
    question now reranks up to 200 candidates instead of 32. That is the right
    trade -- the feature was returning 32 while advertising 100 -- but it is a real
    cost and it should be bounded rather than discovered in production.
    """
    service, org, _space, embedder = geometry
    space_n, _gold = await _fill(geometry, 500, slug="covcost")
    embedder.register("list all of the corpus entries", axis(ANCHOR))

    started = time.perf_counter()
    narrow = await service.search(
        SearchRequest(
            query="the kelmady arrangement",
            org_id=org.id,
            space_id=space_n.id,
            limit=10,
            **GEOMETRY,
        )
    )
    narrow_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    wide = await service.search(
        SearchRequest(
            query="list all of the corpus entries",
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
    wide_ms = (time.perf_counter() - started) * 1000

    assert len(wide.results) > len(narrow.results), "the coverage window did not widen"
    assert wide_ms < max(narrow_ms * 10, 250), (
        f"a coverage query cost {wide_ms:.1f}ms against {narrow_ms:.1f}ms for a "
        "narrow one -- the widened hydration pool is not paying for itself"
    )


async def test_repeated_queries_do_not_slow_down(geometry: tuple) -> None:
    """No unbounded per-query state.

    An access counter, a growing explain list, or a cache that never evicts would
    show up as the tenth query costing more than the first. `decay.access_factor`
    exists in the tree for exactly that idea and is deliberately unwired.
    """
    space_n, _gold = await _fill(geometry, 500, slug="repeat")
    _ids, first = await _timed_search(geometry, space_n, runs=3)
    _ids, later = await _timed_search(geometry, space_n, runs=20)
    assert later < first * 3, f"queries got slower with repetition: {first} -> {later}"


# -- I2: what a query actually returns -----------------------------------


@pytest.mark.parametrize("size", (500, 3000))
async def test_the_result_set_does_not_grow_with_the_corpus(geometry: tuple, size: int) -> None:
    """Token efficiency, at the layer that determines it.

    Mem0 reports retrieved tokens per call as a headline number because it is
    what the caller pays for. The store-level invariant is that `limit` means
    limit: a 3,000-memory space must not return more rows than a 100-memory one
    for the same request, or the prompt grows with the corpus and cost scales with
    history rather than with the question.
    """
    space_n, _gold = await _fill(geometry, size, slug=f"tok{size}")
    ids, _ms = await _timed_search(geometry, space_n, runs=1)
    assert len(ids) == 10, f"limit=10 returned {len(ids)} at N={size}"


async def test_the_write_path_at_scale_stays_bounded_per_write(geometry: tuple) -> None:
    """`ingest` cost per write, which is the one that is NOT flat.

    Measured earlier: 2.6ms per write at N=50 rising to 7.4ms at N=500, because
    every ingest runs a `neighbours` lookup over the space for dedupe,
    supersession, contradiction and association. That is quadratic in the corpus
    and it is a known characteristic, not a surprise -- `consolidation_candidates`
    (64) bounds how many of those neighbours are examined.

    What this asserts is that the bound holds: per-write cost may grow, but the
    LOOKUP is capped, so it must not grow without limit. Ten writes at two corpus
    sizes, and the later batch must not be an order of magnitude worse.
    """
    service, org, _space, embedder = geometry
    space_n = await service.create_space(org.id, slug="writepath", name="writepath")
    embedder.register("the kelmady arrangement", axis(ANCHOR))

    async def batch(start: int) -> float:
        started = time.perf_counter()
        for i in range(start, start + 10):
            marker = f"w{i}"
            embedder.register_marker(marker, at_cosine(FILLER_COSINE, off=2 + (i % 120)))
            await service.ingest(
                org_id=org.id,
                space_id=space_n.id,
                content=f"{marker} written entry number {i}",
                extract=False,
            )
        return (time.perf_counter() - started) / 10 * 1000

    early = await batch(0)
    for i in range(10, 300):
        marker = f"w{i}"
        embedder.register_marker(marker, at_cosine(FILLER_COSINE, off=2 + (i % 120)))
        await service.store.upsert_memory(
            build_memory(
                org_id=org.id,
                space_id=space_n.id,
                content=f"{marker} bulk entry number {i}",
                vector=at_cosine(FILLER_COSINE, off=2 + (i % 120)),
            )
        )
    late = await batch(300)

    assert late < max(early * 15, 50.0), (
        f"per-write cost went {early:.2f}ms -> {late:.2f}ms over 300 memories; "
        "the consolidation_candidates cap is not bounding the neighbours lookup"
    )
