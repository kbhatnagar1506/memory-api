"""Write-time extraction: decomposition, grounding, and the additive rule.

The passage used throughout is a real LongMemEval turn, unedited. It is one
memory today and one vector today -- the average of tour logistics,
photography gear, and owning a Nikon -- which is the failure this subsystem
exists to fix.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from mapi.config import Settings
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import MemoryKind, Organization, RelationType, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.domain.synthesis.extract import MIN_CHARS, extract_claims, parse_claims
from mapi.service import MemoryService
from mapi.store.base import MemoryFilter
from mapi.store.memory import InMemoryStore

PASSAGE = (
    "I'll definitely ask them about customized itineraries and Quetzal-focused "
    "tours. I've also been thinking about my photography gear and wanted to know "
    "if you have any recommendations for lenses or camera settings for capturing "
    "birds in flight. I've had some experience with my Nikon camera, but I'm not "
    "sure what would be the best setup for bird photography."
)

GOOD = json.dumps(
    [
        {
            "fact": "Owns a Nikon camera.",
            "quote": "I've had some experience with my Nikon camera",
        },
        {
            "fact": "Plans to ask about customized itineraries and Quetzal-focused tours.",
            "quote": "ask them about customized itineraries and Quetzal-focused tours",
        },
        {
            "fact": "Is interested in photographing birds in flight.",
            "quote": "camera settings for capturing birds in flight",
        },
    ]
)


def _completer(reply: str):
    async def complete(prompt: str) -> str:
        assert PASSAGE[:40] in prompt, "the passage must reach the model"
        return reply

    return complete


# -- parsing and grounding ---------------------------------------------------


async def test_splits_a_blob_into_atomic_claims() -> None:
    claims = await extract_claims(PASSAGE, _completer(GOOD))
    assert [c.fact for c in claims] == [
        "Owns a Nikon camera.",
        "Plans to ask about customized itineraries and Quetzal-focused tours.",
        "Is interested in photographing birds in flight.",
    ]


async def test_ungrounded_claim_is_discarded() -> None:
    """A quote that is not in the passage means the fact was invented."""
    reply = json.dumps(
        [
            {"fact": "Owns a Nikon camera.", "quote": "my Nikon camera"},
            {"fact": "Owns a Canon camera.", "quote": "I also own a Canon"},
        ]
    )
    claims = await extract_claims(PASSAGE, _completer(reply))
    assert [c.fact for c in claims] == ["Owns a Nikon camera."]


async def test_grounding_tolerates_reflowed_whitespace() -> None:
    reply = json.dumps(
        [{"fact": "Owns a Nikon camera.", "quote": "experience   with\n  my Nikon camera"}]
    )
    assert len(await extract_claims(PASSAGE, _completer(reply))) == 1


def test_near_identical_restatements_collapse() -> None:
    """Two memories for one fact double-count and vote twice in retrieval."""
    reply = json.dumps(
        [
            {"fact": "Owns a Nikon camera.", "quote": "my Nikon camera"},
            {"fact": "The speaker owns a Nikon camera.", "quote": "my Nikon camera"},
        ]
    )
    assert len(parse_claims(reply, PASSAGE)) == 1


def test_fenced_json_is_accepted() -> None:
    assert len(parse_claims(f"```json\n{GOOD}\n```", PASSAGE)) == 3


def test_unparseable_completion_yields_nothing() -> None:
    for junk in ("", "I'm sorry, I can't help with that.", "[{oops", "null", "{}"):
        assert parse_claims(junk, PASSAGE) == []


async def test_provider_failure_is_swallowed() -> None:
    """Fail-open: the worst case of this subsystem is the status quo."""

    async def boom(_: str) -> str:
        raise RuntimeError("vertex is down")

    assert await extract_claims(PASSAGE, boom) == []


async def test_short_text_skips_the_model_call() -> None:
    calls = []

    async def complete(prompt: str) -> str:
        calls.append(prompt)
        return GOOD

    assert await extract_claims("more", complete) == []
    assert calls == [], "a one-word turn must not cost a model call"
    assert len("more") < MIN_CHARS


# -- service wiring ----------------------------------------------------------


def _settings() -> Settings:
    return Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=128,
        rerank_backend="heuristic",
        api_key_pepper="test-pepper",
        chunk_target_tokens=64,
        chunk_overlap_tokens=8,
    )


async def _wired(reply: str = GOOD) -> tuple[MemoryService, str, str]:
    store = InMemoryStore()
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    service = MemoryService(
        store,
        DeterministicEmbedder(dimensions=128),
        HeuristicReranker(),
        _settings(),
        extractor=_completer(reply),
    )
    return service, org.id, space.id


async def test_ingest_writes_claims_and_keeps_the_parent() -> None:
    service, org_id, space_id = await _wired()
    result = await service.ingest(
        org_id=org_id,
        space_id=space_id,
        content=PASSAGE,
        occurred_at=datetime(2023, 5, 6, tzinfo=UTC),
        extract=True,
    )

    assert len(result.extracted) == 3
    # The parent is the memory the caller wrote, unchanged.
    assert result.memory.content == PASSAGE
    assert result.memory.kind is MemoryKind.EPISODIC

    page = await service.store.list_memories(
        org_id, space_id, filters=MemoryFilter(), limit=50, cursor=None
    )
    assert len(page.items) == 4, "three claims plus the untouched original"


async def test_claims_point_back_at_their_source() -> None:
    """The derived_from edge is what makes erasure propagate."""
    service, org_id, space_id = await _wired()
    result = await service.ingest(
        org_id=org_id, space_id=space_id, content=PASSAGE, extract=True
    )

    graph = await service.get_graph(org_id, space_id, limit=50)
    edges = [e for e in graph.edges if e.type is RelationType.DERIVED_FROM]
    assert len(edges) == 3
    assert {e.target_id for e in edges} == {result.memory.id}
    assert {e.source_id for e in edges} == set(result.extracted)


async def test_erasing_the_source_makes_its_claims_stale() -> None:
    """The whole reason claims are additive and linked rather than standalone."""
    service, org_id, space_id = await _wired()
    result = await service.ingest(
        org_id=org_id, space_id=space_id, content=PASSAGE, extract=True
    )

    await service.erase_memory(org_id, space_id, result.memory.id)

    for claim_id in result.extracted:
        claim = await service.store.get_memory(org_id, space_id, claim_id)
        assert claim is not None
        assert claim.status.value == "stale", (
            "a claim whose source was erased must not stay standing as an "
            "assertion nobody can trace"
        )


async def test_claims_carry_their_quote_and_parent() -> None:
    service, org_id, space_id = await _wired()
    result = await service.ingest(
        org_id=org_id, space_id=space_id, content=PASSAGE, extract=True
    )
    claim = await service.store.get_memory(org_id, space_id, result.extracted[0])
    assert claim is not None
    assert claim.kind is MemoryKind.DERIVED
    assert claim.metadata["extracted_from"] == result.memory.id
    assert claim.metadata["quote"] in PASSAGE


async def test_extraction_is_off_by_default() -> None:
    service, org_id, space_id = await _wired()
    result = await service.ingest(org_id=org_id, space_id=space_id, content=PASSAGE)
    assert result.extracted == []


async def test_no_completer_means_no_extraction() -> None:
    """Extraction needs a model; without one, ingest behaves exactly as before."""
    store = InMemoryStore()
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=128), HeuristicReranker(), _settings()
    )
    result = await service.ingest(
        org_id=org.id, space_id=space.id, content=PASSAGE, extract=True
    )
    assert result.extracted == []
    assert result.memory.content == PASSAGE


async def test_a_claim_that_restates_its_parent_is_not_self_linked() -> None:
    """A one-sentence turn is already atomic; dedupe resolves the claim back
    to the parent, and a memory cannot be derived from itself."""
    sentence = "The user set a personal best time of 27:12 in a charity 5K run in May 2023."
    reply = json.dumps([{"fact": sentence, "quote": sentence}])
    store = InMemoryStore()
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))

    async def complete(_: str) -> str:
        return reply

    service = MemoryService(
        store,
        DeterministicEmbedder(dimensions=128),
        HeuristicReranker(),
        _settings(),
        extractor=complete,
    )
    result = await service.ingest(
        org_id=org.id, space_id=space.id, content=sentence, extract=True
    )

    assert result.extracted == []
    graph = await service.get_graph(org.id, space.id, limit=20)
    assert [e for e in graph.edges if e.type is RelationType.DERIVED_FROM] == []
    assert len(graph.memories) == 1


def test_truncated_array_still_yields_its_complete_objects() -> None:
    """A completion cut at the token ceiling must not zero the document.

    Whole-array parsing needs a closing bracket, so without salvage the
    longest and densest passages -- the ones with the most to decompose --
    silently produce nothing at all.
    """
    truncated = (
        '[\n {"fact": "Owns a Nikon camera.", '
        '"quote": "I\'ve had some experience with my Nikon camera"},\n'
        ' {"fact": "Is interested in photographing birds in flight.", '
        '"quote": "camera settings for capturing birds in flight"},\n'
        ' {"fact": "Plans to ask about customized itin'
    )
    claims = parse_claims(truncated, PASSAGE)
    assert [c.fact for c in claims] == [
        "Owns a Nikon camera.",
        "Is interested in photographing birds in flight.",
    ]


def test_salvage_handles_braces_inside_quoted_values() -> None:
    """Depth counting must ignore braces inside strings, or one stray brace
    in a fact realigns every object after it."""
    text = PASSAGE + " I use {brackets} in my notes."
    raw = (
        '[{"fact": "Uses {brackets} in their notes.", "quote": "I use {brackets} in my notes."'
    )
    assert parse_claims(raw + "}", text)[0].fact == "Uses {brackets} in their notes."
