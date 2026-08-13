"""Comprehensive questions get the whole territory, not the best few.

The bug this exists to prevent, measured against a live space holding 25
infrastructure facts: "what is our entire infrastructure" returned 10 of
them with every score inside a 2% band -- 0.0143 to 0.0164 -- so which ten
came back was effectively arbitrary. The database, the cache and the runtime
were not among them. Ranking answers "which of these is most relevant"; this
question asks "what is all of this", and relevance has no opinion on
completeness.
"""

from __future__ import annotations

import pytest

from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.synthesis.classify import QuestionKind
from mapi.service import MemoryService

#: One space, one subject, many facets. Deliberately flat: no fact is a
#: better answer to "what is our infrastructure" than any other, which is
#: exactly the distribution a ranked top-k handles worst.
INFRASTRUCTURE = [
    "Our primary database is Postgres 16.",
    "We use pgvector for embedding storage.",
    "Redis backs the rate limiter.",
    "The API runs on Python 3.13 with FastAPI.",
    "Vector indexes use HNSW.",
    "Authentication is Google OAuth.",
    "Migrations run through Alembic.",
    "We deploy on Heroku with a release phase.",
    "Embeddings come from Vertex AI.",
    "Structured logging goes to stdout as JSON.",
    "The SDK is published to PyPI as mapi-sdk.",
    "Sessions are signed cookies via itsdangerous.",
    "SQLAlchemy 2.0 async is the ORM layer.",
    "Ruff handles linting and formatting.",
    "Mypy runs in strict mode.",
    "Tests run under pytest with asyncio mode.",
    "Reranking uses Gemini when enabled.",
    "The lexical index is a generated tsvector column.",
    "Rate limits are per API key.",
    "Search fuses vector and lexical with RRF.",
]


@pytest.fixture
async def stocked(service: MemoryService, org: Organization, space: Space) -> None:
    for fact in INFRASTRUCTURE:
        await service.ingest(org_id=org.id, space_id=space.id, content=fact)


async def test_a_specific_question_still_gets_a_ranked_few(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    result = await service.search(
        SearchRequest(
            query="what database do we use", org_id=org.id, space_id=space.id, limit=5
        )
    )
    assert result.intent is not None
    assert result.intent.kind is QuestionKind.DIRECT
    assert not result.intent.comprehensive
    assert len(result.results) <= 5, "a lookup must not turn into an export"


async def test_a_comprehensive_question_gets_the_whole_set(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    result = await service.search(
        SearchRequest(
            query="what is our entire infrastructure",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            coverage_limit=100,
        )
    )
    assert result.intent is not None
    assert result.intent.comprehensive
    assert len(result.results) > 5, "the window did not widen past the requested limit"


async def test_coverage_can_be_forced_on(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """An API told to return everything must not argue with its caller."""
    result = await service.search(
        SearchRequest(
            query="database",
            org_id=org.id,
            space_id=space.id,
            limit=3,
            coverage=True,
            coverage_limit=100,
        )
    )
    assert result.intent is not None
    assert result.intent.source == "explicit"
    assert len(result.results) > 3


async def test_coverage_can_be_forced_off(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    result = await service.search(
        SearchRequest(
            query="what is our entire infrastructure",
            org_id=org.id,
            space_id=space.id,
            limit=4,
            coverage=False,
            coverage_limit=100,
        )
    )
    assert result.intent is not None
    assert not result.intent.comprehensive
    assert len(result.results) <= 4


async def test_the_coverage_window_is_bounded(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """ "Everything" is a shape of question, not a licence to scan a space."""
    result = await service.search(
        SearchRequest(
            query="tell me everything about our stack",
            org_id=org.id,
            space_id=space.id,
            limit=1,
            coverage_limit=100_000,
        )
    )
    assert len(result.results) <= 200


async def test_the_widening_reaches_the_store_not_just_the_truncation(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """Widening after the fetch would widen a set already cut to size.

    The candidate pool has to be requested larger from the first query, so
    `total_candidates` -- the size of the fused pool, before any top-k -- is
    where the decision proves it happened early.
    """
    narrow = await service.search(
        SearchRequest(
            query="what database do we use", org_id=org.id, space_id=space.id, limit=3
        )
    )
    wide = await service.search(
        SearchRequest(
            query="what is our entire infrastructure",
            org_id=org.id,
            space_id=space.id,
            limit=3,
            coverage_limit=100,
        )
    )
    assert wide.total_candidates > narrow.total_candidates


async def test_explain_records_why_the_result_set_is_this_shape(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    result = await service.search(
        SearchRequest(
            query="list all of our infrastructure",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            coverage_limit=100,
        )
    )
    assert result.results
    assert any("coverage window" in note for note in result.results[0].explain)
