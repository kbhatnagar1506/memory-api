"""Fixtures for the capability families.

Two kinds of test live here and they need opposite things:

**Geometry tests** construct a similarity structure with `ScriptedEmbedder` and
assert what the pipeline does with it. For those, an unregistered string is a
silent failure -- the embedder returns a reserved orthogonal sentinel and the
test measures the sentinel -- so `assert_no_misses` is autouse.

**Stack tests** run the shipped embedder and the shipped lexical arm and measure
what the product actually does. The minimal-pair battery is one of these, and it
has to be: hand-specifying the vectors for "$500" and "$5,000" would assert that
a higher cosine ranks first, which is true by construction and tests nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from tests.support.embedder import ScriptedEmbedder
from tests.support.factories import tenant
from tests.support.vectors import DIMS

from mapi.config import Settings
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore


@pytest.fixture
def scripted() -> ScriptedEmbedder:
    return ScriptedEmbedder(dimensions=DIMS)


@pytest.fixture
def capability_settings() -> Settings:
    """Matches `tests/conftest.py::settings` on width, and keeps
    `contextual_embedding` on -- the marker lookup exists precisely so a corpus
    survives the header that flag injects, and turning it off here would test a
    configuration nobody runs."""
    return Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=DIMS,
        rerank_backend="heuristic",
        api_key_pepper="capability-pepper-value",
        chunk_target_tokens=64,
        chunk_overlap_tokens=8,
        rate_limit_per_minute=100_000,
        rate_limit_burst=100_000,
    )


@pytest.fixture
async def geometry(
    scripted: ScriptedEmbedder, capability_settings: Settings
) -> AsyncIterator[tuple[MemoryService, Organization, Space, ScriptedEmbedder]]:
    """A service whose similarity structure the test specifies.

    Yields the embedder too, so a test registers its vectors and the autouse
    miss check can find the same instance.
    """
    store = InMemoryStore()
    service = MemoryService(store, scripted, HeuristicReranker(), capability_settings)
    org, space = await tenant(store, name="Capability")
    yield service, org, space, scripted
    await store.reset()


@pytest.fixture(autouse=True)
def assert_no_misses(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    """Fail any geometry test that embedded an unregistered string.

    Without this, a typo in a marker makes the memory land on the reserved miss
    axis, the query finds nothing there, and the test passes for the wrong
    reason. Only applies when the test asked for `scripted`.
    """
    yield
    if "scripted" not in request.fixturenames:
        return
    embedder = request.getfixturevalue("scripted")
    assert embedder.misses == [], (
        "these strings were embedded without a registered vector, so they landed "
        f"on the reserved miss axis: {embedder.misses}"
    )
