"""Confidence graded a number that was never a similarity.

`_STRONG_SCORE = 0.55` and `_WEAK_SCORE = 0.30` are cosine thresholds and always
said so. `assess()` was being handed `ScoredMemory.score`, which is a cosine at
no point in the pipeline: with reranking off it is a fused RRF score whose
ceiling is `2/(rrf_k+1)` ~= 0.033, and with reranking on it is
`HeuristicReranker`'s unbounded IDF sum. Measured on one query against one
five-fact corpus:

    use_rerank=True   ->  top score 5.7835   level=high
    use_rerank=False  ->  top score 0.0352   level=low, refusal=weak_evidence

Same question, same corpus, same right answer, opposite verdicts -- decided by a
performance flag. And `vector_score` was 0.806 in both runs: a real cosine, sat
right there on the same object, ignored.

`min_score` had the same bug with a worse symptom. `min_score=0.3` with rerank
off could not admit any result from any query, and the only trace was a
debug-level log line.

The fix is `calibrated_score`, not new numbers: a threshold that has to be
re-derived per flag combination is not a threshold.
"""

from __future__ import annotations

import pytest
from tests.support.factories import memory as build_memory

from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.confidence import (
    _STRONG_SCORE,
    _WEAK_SCORE,
    ConfidenceLevel,
    calibrated_score,
)
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.service import MemoryService

FACTS = [
    "Our primary database is Postgres 16 running on Cloud SQL in us-central1.",
    "Redis 7 backs the rate limiter across every process.",
    "The API runs on Python 3.13 with FastAPI and uvicorn.",
    "Embeddings come from Vertex AI text-embedding-004 at 768 dimensions.",
    "We deploy on Heroku with a release phase that runs migrations.",
]


@pytest.fixture
async def stocked(service: MemoryService, org: Organization, space: Space) -> None:
    for fact in FACTS:
        await service.ingest(org_id=org.id, space_id=space.id, content=fact)


# -- the defect ------------------------------------------------------------


async def test_confidence_does_not_flip_when_reranking_is_toggled(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """The headline. One query, two flags, one verdict.

    Reranking is a latency/quality trade. It is not supposed to be an epistemic
    one, and a caller reading `confidence` to decide whether to assert an answer
    was getting a different answer depending on a setting they may not control.
    """
    verdicts = {}
    for rerank in (True, False):
        response = await service.search(
            SearchRequest(
                query=FACTS[0],
                org_id=org.id,
                space_id=space.id,
                limit=5,
                use_rerank=rerank,
            )
        )
        verdicts[rerank] = response.confidence
    assert verdicts[True] is not None and verdicts[False] is not None
    assert verdicts[True].level is verdicts[False].level, (
        f"rerank=True gave {verdicts[True].level.value}, "
        f"rerank=False gave {verdicts[False].level.value}"
    )


async def test_a_verbatim_query_is_not_graded_as_weak_evidence(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """With rerank off, EVERY query used to return LOW/weak_evidence.

    The RRF ceiling is ~0.033 and `_WEAK_SCORE` is 0.30, so the branch was
    unreachable in the other direction -- no result set could clear it.
    """
    response = await service.search(
        SearchRequest(
            query=FACTS[0], org_id=org.id, space_id=space.id, limit=5, use_rerank=False
        )
    )
    assert response.confidence is not None
    assert response.confidence.level is not ConfidenceLevel.LOW
    assert response.confidence.refusal_reason is None


async def test_the_graded_score_is_on_the_cosine_scale(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """`top_score` is now comparable against the thresholds beside it."""
    response = await service.search(
        SearchRequest(
            query=FACTS[1], org_id=org.id, space_id=space.id, limit=5, use_rerank=False
        )
    )
    assert response.confidence is not None
    assert -1.0 <= response.confidence.top_score <= 1.0
    assert response.confidence.top_score >= _WEAK_SCORE


async def test_min_score_admits_a_verbatim_match(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """`min_score=0.3` used to empty every result set with rerank off."""
    response = await service.search(
        SearchRequest(
            query=FACTS[2],
            org_id=org.id,
            space_id=space.id,
            limit=5,
            use_rerank=False,
            min_score=0.3,
        )
    )
    assert response.results


async def test_min_score_still_excludes_a_weak_match(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """The filter has to still filter, or the fix traded one bug for another."""
    loose = await service.search(
        SearchRequest(
            query="something entirely unrelated to any of this",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            use_rerank=False,
        )
    )
    strict = await service.search(
        SearchRequest(
            query="something entirely unrelated to any of this",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            use_rerank=False,
            min_score=0.95,
        )
    )
    assert len(strict.results) < len(loose.results) or not loose.results


# -- calibrated_score itself ----------------------------------------------


def test_calibrated_score_prefers_the_cosine() -> None:
    scored = build_memory(org_id="org_x", space_id="spc_x", content="a fact")
    from mapi.domain.models import ScoredMemory

    hit = ScoredMemory(memory=scored, score=5.78, vector_score=0.806)
    assert calibrated_score(hit) == pytest.approx(0.806)


def test_calibrated_score_falls_back_when_the_vector_arm_did_not_run() -> None:
    """Lexical-only search -- a zeroed `vector_weight`, or a degraded provider.

    There is no cosine to grade on, so the old behaviour is the best available
    and is what happens. Silently reporting 0.0 would turn a working
    lexical-only search into a permanent NO_RELEVANT_MEMORY.
    """
    from mapi.domain.models import ScoredMemory

    scored = build_memory(org_id="org_x", space_id="spc_x", content="a fact")
    hit = ScoredMemory(memory=scored, score=0.42, vector_score=None)
    assert calibrated_score(hit) == pytest.approx(0.42)


def test_the_thresholds_are_orderable_cosines() -> None:
    """A guard on the constants, so a future edit cannot silently un-fix this."""
    assert 0.0 < _WEAK_SCORE < _STRONG_SCORE <= 1.0


async def test_an_empty_result_set_declines_for_the_right_reason(
    service: MemoryService, org: Organization, space: Space
) -> None:
    """NO_RELEVANT_MEMORY is correct silence; WEAK_EVIDENCE is a memory gap.

    Conflating them is what made the field useless: everything read as a gap.
    """
    response = await service.search(
        SearchRequest(query="anything at all", org_id=org.id, space_id=space.id, limit=5)
    )
    assert response.confidence is not None
    assert response.confidence.level is ConfidenceLevel.NONE
    assert response.confidence.refusal_reason is not None
    assert response.confidence.refusal_reason.value == "no_relevant_memory"
