"""Pipeline contracts that no test held, and two defects that lived in the gap.

Every assertion here was written after reading the pipeline and finding that
something documented was not true. The two that mattered:

**Coverage was capped at 32.** `effective_limit` is widened for a comprehensive
question -- up to `coverage_limit` (100), bounded by `_MAX_COVERAGE_LIMIT` (200)
-- and `fetch` is widened with it, to as many as 600 candidates. Then the
hydration pool was sliced with `request.limit`, and every stage after it can
only shrink. So the whole feature was bounded by `rerank_candidates`, and stage
9 stamped `"coverage window 100"` on each of the 32 results it returned.
`test_coverage.py` could not see it: its fixture holds 20 facts.

**`min_score` and confidence graded the wrong number.** Both compared cosine
thresholds against `ScoredMemory.score`, which is an RRF score (~0.03 ceiling)
or an unbounded reranker sum depending on `use_rerank`. `min_score=0.3` with
rerank off dropped every result of every query. See `test_confidence_scale.py`.

The rest are properties the docstrings claim and nothing checked: which
`timings_ms` keys appear, that a memory deleted between search and hydrate is
skipped rather than emitted as a dangling id, that `rerank_degraded` propagates,
and that an embedding failure degrades to lexical-only instead of failing.
"""

from __future__ import annotations

import pytest

from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import (
    _MAX_CANDIDATE_FETCH,
    _MAX_COVERAGE_LIMIT,
    SearchRequest,
)
from mapi.service import MemoryService

#: Well above `rerank_candidates` (32), which is the cap the old slice imposed.
#: A 20-fact corpus -- what test_coverage.py uses -- cannot distinguish "capped
#: at 32" from "returned everything".
CORPUS = 60


@pytest.fixture
async def wide(service: MemoryService, org: Organization, space: Space) -> None:
    """One flat subject with more facets than the cap, so a coverage question
    genuinely needs a window wider than 32."""
    for i in range(CORPUS):
        await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content=f"Infrastructure component number {i}: service-{i} runs in the cluster.",
            tags=("infra",),
        )


# -- defect: coverage capped at 32 -----------------------------------------


async def test_a_coverage_question_returns_more_than_the_rerank_pool(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """The defect, at its narrowest.

    Before the fix this returned exactly 32 -- `max(rerank_candidates=32,
    limit=10)` -- for a question whose coverage window was 100.
    """
    response = await service.search(
        SearchRequest(
            query="What is our entire infrastructure?",
            org_id=org.id,
            space_id=space.id,
            limit=10,
            coverage=True,
            coverage_limit=100,
        )
    )
    assert len(response.results) > 32, (
        f"coverage returned {len(response.results)}; the hydration pool is still "
        "being cut with request.limit"
    )
    assert len(response.results) == CORPUS


async def test_the_coverage_window_it_advertises_is_the_one_it_delivers(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """The explain line was the tell: it claimed 100 while returning 32.

    An explain string that overstates what happened is worse than none, because
    it is what someone debugging a short result set would read and trust.
    """
    response = await service.search(
        SearchRequest(
            query="List all of our infrastructure.",
            org_id=org.id,
            space_id=space.id,
            limit=10,
            coverage=True,
            coverage_limit=100,
        )
    )
    windows = [
        line for hit in response.results for line in hit.explain if "coverage window" in line
    ]
    assert windows, "a comprehensive result must say so in explain"
    advertised = int(windows[0].split("coverage window ")[1].split()[0])
    assert len(response.results) <= advertised
    # And the corpus is smaller than the window, so it must be exhausted.
    assert len(response.results) == CORPUS


async def test_raising_the_limit_used_to_be_the_only_way_to_widen_coverage(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """Before the fix, `limit` and not `coverage_limit` controlled the ceiling.

    Kept as a regression test in the opposite direction: a coverage question
    must now return the same count whether the caller passed limit=10 or
    limit=50, because the coverage window governs.
    """
    counts = []
    for limit in (10, 50):
        response = await service.search(
            SearchRequest(
                query="What is our entire infrastructure?",
                org_id=org.id,
                space_id=space.id,
                limit=limit,
                coverage=True,
                coverage_limit=100,
            )
        )
        counts.append(len(response.results))
    assert counts[0] == counts[1], f"limit still governs the coverage window: {counts}"


async def test_a_specific_question_is_not_widened(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """The fix must not turn every search into a coverage sweep."""
    response = await service.search(
        SearchRequest(
            query="What is service-7?",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            coverage=False,
        )
    )
    assert len(response.results) <= 5


async def test_the_coverage_window_is_still_bounded(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """ "Everything" is a question shape, not a licence to export a space."""
    response = await service.search(
        SearchRequest(
            query="What is our entire infrastructure?",
            org_id=org.id,
            space_id=space.id,
            limit=10,
            coverage=True,
            coverage_limit=100_000,
        )
    )
    assert len(response.results) <= _MAX_COVERAGE_LIMIT
    assert _MAX_CANDIDATE_FETCH >= _MAX_COVERAGE_LIMIT


# -- min_score, which runs after MMR and truncation ------------------------


async def test_min_score_is_measured_on_the_cosine_scale(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """`min_score=0.3` used to empty every result set with rerank off.

    The threshold was compared against a fused RRF score whose ceiling is
    `2/(rrf_k+1)` ~= 0.033, so no result could ever clear 0.3 and the filter
    silently discarded everything.
    """
    response = await service.search(
        SearchRequest(
            query="Infrastructure component number 7: service-7 runs in the cluster.",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            use_rerank=False,
            min_score=0.3,
        )
    )
    assert response.results, "a verbatim query must survive a 0.3 cosine floor"


async def test_an_impossible_floor_still_empties_the_set(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """The filter must still filter; cosine cannot exceed 1.0."""
    response = await service.search(
        SearchRequest(
            query="service-7",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            use_rerank=False,
            min_score=1.01,
        )
    )
    assert response.results == []


# -- the timings contract --------------------------------------------------


async def test_timings_report_every_stage_that_ran(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    response = await service.search(
        SearchRequest(query="service-3", org_id=org.id, space_id=space.id, limit=5)
    )
    for key in ("embed_ms", "candidates_ms", "fusion_ms", "hydrate_ms", "rerank_ms"):
        assert key in response.timings_ms, f"{key} missing from timings"
    assert "entity_ms" not in response.timings_ms, "entity expansion did not run"


async def test_rerank_timing_is_absent_when_rerank_is_off(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """A timing for a stage that did not run is a lie about where time went."""
    response = await service.search(
        SearchRequest(
            query="service-3", org_id=org.id, space_id=space.id, limit=5, use_rerank=False
        )
    )
    assert "rerank_ms" not in response.timings_ms


async def test_an_empty_query_is_answered_without_touching_the_store(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    response = await service.search(
        SearchRequest(query="   ", org_id=org.id, space_id=space.id, limit=5)
    )
    assert response.results == []
    assert response.total_candidates == 0
    assert response.confidence is not None
    assert response.confidence.level.value == "none"


async def test_a_nonpositive_limit_returns_nothing_rather_than_everything(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    for limit in (0, -1):
        response = await service.search(
            SearchRequest(query="service-3", org_id=org.id, space_id=space.id, limit=limit)
        )
        assert response.results == []


async def test_every_empty_path_reports_a_confidence(
    service: MemoryService, org: Organization, space: Space
) -> None:
    """An empty space, an empty query and a zero limit are three different
    early returns, and a caller reading `confidence` should not have to know
    which one it hit."""
    for request in (
        SearchRequest(query="anything", org_id=org.id, space_id=space.id, limit=5),
        SearchRequest(query="", org_id=org.id, space_id=space.id, limit=5),
        SearchRequest(query="anything", org_id=org.id, space_id=space.id, limit=0),
    ):
        response = await service.search(request)
        assert response.results == []
        assert response.confidence is not None, (
            f"no confidence for {request.query!r}/{request.limit}"
        )


# -- degradation -----------------------------------------------------------


async def test_an_embedding_failure_degrades_to_lexical_rather_than_failing(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """Documented in the pipeline docstring, never tested.

    The vector arm is the better one; losing it should cost quality, not
    availability. A search that raises here takes the whole product down for the
    duration of a provider incident.
    """
    from mapi.core.errors import ProviderError

    async def broken(text: str) -> list[float]:
        raise ProviderError("vendor is down")

    # `embed_one`, not `embed_query`: the pipeline prefers `embed_query` when
    # the provider defines it, and `DeterministicEmbedder` does not -- so
    # patching the name that is absent would have patched nothing, and the test
    # would have passed while exercising a healthy vector arm.
    original = service.embedder.embed_one
    service.embedder.embed_one = broken  # type: ignore[method-assign]
    try:
        response = await service.search(
            SearchRequest(
                query="Infrastructure component number 3",
                org_id=org.id,
                space_id=space.id,
                limit=5,
            )
        )
    finally:
        service.embedder.embed_one = original  # type: ignore[method-assign]

    assert response.results, "lexical-only search returned nothing"
    assert "vector" not in response.strategies
    assert all(hit.vector_score is None for hit in response.results)


async def test_a_memory_deleted_between_search_and_hydrate_is_skipped(
    service: MemoryService, org: Organization, space: Space, wide: None
) -> None:
    """The hydration hole: `memories.get(item.id) is None`.

    Reproduced by deleting a memory from under a stale candidate list, which is
    what a concurrent delete does. The result must be short, not dangling.
    """
    from mapi.store.base import MemoryFilter

    listing = await service.list_memories(
        org.id, space.id, filters=MemoryFilter(), limit=5, cursor=None
    )
    victim = listing.items[0]

    original = service.store.get_memories

    async def hiding(org_id: str, space_id: str, memory_ids: object) -> dict[str, object]:
        found = await original(org_id, space_id, memory_ids)  # type: ignore[arg-type]
        found.pop(victim.id, None)
        return found  # type: ignore[return-value]

    service.store.get_memories = hiding  # type: ignore[method-assign, assignment]
    try:
        response = await service.search(
            SearchRequest(query="infrastructure", org_id=org.id, space_id=space.id, limit=10)
        )
    finally:
        service.store.get_memories = original  # type: ignore[assignment]

    assert victim.id not in [hit.memory.id for hit in response.results]
