"""Model-dependent thresholds become settings: confidence bands, the header
cut-off for short texts, and the supersession ceiling.

All three were constants tuned on text-embedding-004. On gemini-embedding-001
unrelated text scores about 0.50-0.53, so WEAK (0.30) never fired and nearly
every answer graded HIGH; card-sized texts sharing a contextual header drifted
together; and a raised dedupe threshold left a band of changed facts that were
neither merged nor considered for supersession. Every default below keeps the
old behaviour -- a deployment opts in with fitted values.
"""

from __future__ import annotations

import pytest

from mapi.config import Settings
from mapi.domain.consolidation import CONTRADICT_LOW, SUPERSEDE_HIGH
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.embeddings.context import for_embedding, wants_header
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.confidence import (
    _STRONG_SCORE,
    _WEAK_SCORE,
    DEFAULT_BANDS,
    ConfidenceBands,
    ConfidenceLevel,
    RefusalReason,
    assess,
)
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

#: The bands facemash starts from on gemini-embedding-001.
FITTED = ConfidenceBands(strong=0.64, weak=0.57)


def _settings(**overrides) -> Settings:
    return Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=64,
        rerank_backend="heuristic",
        api_key_pepper="x" * 32,
        **overrides,
    )


async def _service(**overrides):
    store = InMemoryStore()
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=64), HeuristicReranker(), _settings(**overrides)
    )
    org = await store.create_organization(Organization(name="T"))
    space = await store.create_space(Space(org_id=org.id, slug="s", name="S"))
    return service, org.id, space.id


# -- confidence bands ---------------------------------------------------------------


def test_the_default_bands_are_the_old_constants() -> None:
    assert (DEFAULT_BANDS.strong, DEFAULT_BANDS.weak) == (_STRONG_SCORE, _WEAK_SCORE)
    settings = Settings()
    assert (settings.confidence_strong, settings.confidence_weak) == (0.55, 0.30)


@pytest.mark.parametrize(("strong", "weak"), [(0.5, 0.5), (0.4, 0.6), (1.2, 0.5), (0.6, -0.1)])
def test_inverted_or_out_of_range_bands_are_refused(strong: float, weak: float) -> None:
    with pytest.raises(ValueError):
        ConfidenceBands(strong=strong, weak=weak)


def test_settings_refuse_inverted_bands() -> None:
    with pytest.raises(ValueError, match="confidence_weak"):
        Settings(confidence_strong=0.5, confidence_weak=0.6)


def test_on_the_new_model_unrelated_text_no_longer_grades_high() -> None:
    """0.52 is a usable match under the old floors and merely the noise level
    of gemini-embedding-001 -- LOW, weak evidence -- under fitted ones; 0.60
    was HIGH and is now only MEDIUM."""
    scores = [0.60, 0.40]
    assert assess(scores).level is ConfidenceLevel.HIGH
    assert assess(scores, bands=FITTED).level is ConfidenceLevel.MEDIUM

    noise = [0.52, 0.30]
    assert assess(noise).level is ConfidenceLevel.MEDIUM
    graded = assess(noise, bands=FITTED)
    assert graded.level is ConfidenceLevel.LOW
    assert graded.refusal_reason is RefusalReason.WEAK_EVIDENCE

    strong = [0.80, 0.55]
    assert assess(strong, bands=FITTED).level is ConfidenceLevel.HIGH


async def test_the_service_hands_its_bands_to_the_pipeline() -> None:
    service, _, _ = await _service(confidence_strong=0.64, confidence_weak=0.57)
    assert service.pipeline.confidence_bands == FITTED
    default_service, _, _ = await _service()
    assert default_service.pipeline.confidence_bands == DEFAULT_BANDS


async def test_a_search_is_graded_with_the_configured_bands() -> None:
    """End to end: the same query and corpus, graded by two sets of bands."""
    facts = ["The launch moved to the ninth of March.", "Lunch is at the east cafe."]
    service, org_id, space_id = await _service()
    for fact in facts:
        await service.ingest(org_id=org_id, space_id=space_id, content=fact)
    request = SearchRequest(query=facts[0], org_id=org_id, space_id=space_id, limit=5)
    baseline = (await service.search(request)).confidence
    assert baseline is not None and baseline.level is not ConfidenceLevel.LOW
    top = baseline.top_score
    assert top < 0.98, "need headroom above the top score for the stricter bands"

    strict, org2, space2 = await _service(
        confidence_strong=round(top + 0.015, 4), confidence_weak=round(top + 0.01, 4)
    )
    for fact in facts:
        await strict.ingest(org_id=org2, space_id=space2, content=fact)
    graded = (
        await strict.search(
            SearchRequest(query=facts[0], org_id=org2, space_id=space2, limit=5)
        )
    ).confidence
    assert graded is not None
    assert graded.top_score == pytest.approx(top)
    assert graded.level is ConfidenceLevel.LOW
    assert graded.refusal_reason is RefusalReason.WEAK_EVIDENCE


# -- contextual header cut-off ------------------------------------------------------


def test_short_memories_skip_the_header_and_long_ones_keep_it() -> None:
    short = "Building a robot arm; stuck on inverse kinematics."
    assert not wants_header(short, min_chars=200)
    assert wants_header("x" * 250, min_chars=200)
    # The default keeps the header everywhere: no behaviour change unasked.
    assert wants_header(short, min_chars=0)
    assert wants_header("", min_chars=0)


def test_the_cut_off_counts_content_not_padding() -> None:
    assert not wants_header("  hi  " + " " * 500, min_chars=10)


def _spy_on_embedder(service: MemoryService) -> list[str]:
    seen: list[str] = []
    original = service.embedder.embed

    async def spy(texts):
        seen.extend(texts)
        return await original(texts)

    service.embedder.embed = spy  # type: ignore[method-assign]
    return seen


async def test_the_service_embeds_a_short_memory_bare() -> None:
    service, org_id, space_id = await _service(contextual_min_chars=200)
    seen = _spy_on_embedder(service)
    await service.ingest(
        org_id=org_id, space_id=space_id, content="Looking for a co-founder.", source="card"
    )
    assert seen == ["Looking for a co-founder."]

    seen.clear()
    long_text = "I have been building a scheduling assistant for clinics. " * 5
    await service.ingest(org_id=org_id, space_id=space_id, content=long_text, source="card")
    assert seen and all(t.startswith("<context>") for t in seen)


async def test_a_long_memorys_short_chunks_keep_their_header() -> None:
    """The cut-off is per memory. Per chunk it would strip the framing from
    every fragment of a long memory shorter than the cut-off -- its tail most
    of all, the chunk that needs the framing most."""
    cut_off = 300
    # 64 tokens is about 256 characters a chunk: each chunk alone is under
    # the cut-off, the memory as a whole is well over it.
    service, org_id, space_id = await _service(
        contextual_min_chars=cut_off, chunk_target_tokens=64, chunk_overlap_tokens=0
    )
    seen = _spy_on_embedder(service)
    sentences = [f"Day {i} of the hackathon we rewrote the matcher again." for i in range(12)]
    await service.ingest(
        org_id=org_id, space_id=space_id, content=" ".join([*sentences, "Then lunch."])
    )
    bodies = [t.split("</context>\n", 1)[-1] for t in seen]
    assert len(bodies) >= 2, "needs a multi-chunk memory"
    assert all(len(b) < cut_off for b in bodies), "each chunk alone is under the cut-off"
    assert all(t.startswith("<context>") for t in seen)


def test_embedding_input_is_unchanged_when_there_is_no_header() -> None:
    assert for_embedding("plain", "") == "plain"
    assert for_embedding("plain", "h") == "<context>\nh\n</context>\nplain"


def test_the_cut_off_setting_defaults_off_and_is_bounded() -> None:
    assert Settings().contextual_min_chars == 0
    with pytest.raises(ValueError):
        Settings(contextual_min_chars=-1)


# -- supersession ceiling follows the dedupe threshold ------------------------------


async def test_the_ceiling_is_the_dedupe_threshold() -> None:
    default, _, _ = await _service()
    assert default._similarity_ceiling() == SUPERSEDE_HIGH, "default behaviour moved"
    raised, _, _ = await _service(dedupe_threshold=0.995)
    assert raised._similarity_ceiling() == 0.995


async def test_a_very_low_dedupe_threshold_cannot_invert_a_band() -> None:
    low, _, _ = await _service(dedupe_threshold=0.5)
    assert low._similarity_ceiling() == CONTRADICT_LOW


async def test_every_proposer_is_given_the_ceiling(monkeypatch) -> None:
    """Supersession, contradiction and the unexplained-pair shortlist all used
    the constant; each must now receive the configured ceiling."""
    import mapi.service as service_module

    seen: dict[str, float] = {}

    def spying(name: str):
        original = getattr(service_module, name)

        def wrapper(*args, **kwargs):
            seen[name] = kwargs.get("high", -1.0)
            return original(*args, **kwargs)

        return wrapper

    for name in ("propose_supersessions", "propose_contradictions", "unexplained_pairs"):
        monkeypatch.setattr(service_module, name, spying(name))

    async def _confirm(prompt: str) -> str:
        return "[]"

    service, org_id, space_id = await _service(dedupe_threshold=0.995)
    service.extractor = _confirm
    await service.ingest(org_id=org_id, space_id=space_id, content="We deploy on Fridays.")
    await service.ingest(
        org_id=org_id, space_id=space_id, content="We no longer deploy on Fridays."
    )
    assert seen == {
        "propose_supersessions": 0.995,
        "propose_contradictions": 0.995,
        "unexplained_pairs": 0.995,
    }
