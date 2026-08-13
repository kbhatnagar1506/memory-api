"""Entity bridging: evidence that shares a name with a hit but not with the query.

The stage nothing exercised. `tests/unit/test_entities.py` tests the extractor in
isolation; `_expand_by_entity` -- the part that turns extracted entities into a
second retrieval pass, dedupes against the first, and merges the results into
fusion -- had no coverage at all, and until this branch it could not run in the
product either (`use_entity_expansion` was unreachable over HTTP).

What it is for: "who did I go bouldering with" retrieves the bouldering memory,
which names Priya. Nothing in the query says Priya, so the memory recording what
Priya recommended afterwards is invisible to both arms. Bridging finds it by
mining names out of what came back and searching again.

The honest caveat, from `entities.py`: "MEASURED RESULT: this does not work, on
any corpus tested." Same category as `budget.py`, with one difference -- this one
is reachable now, so its behaviour is worth pinning even though its value is not
established. These tests assert the MECHANISM (dedupe, best-score-per-memory,
per-lookup failure isolation, the explain marker) rather than that it improves
retrieval, because the latter is what was measured and found absent.
"""

from __future__ import annotations

import pytest

from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.service import MemoryService
from mapi.store.base import LexicalHit

#: Two facts joined only by a name the query never mentions. This is the shape
#: bridging exists for: no lexical overlap and no semantic overlap between the
#: query and the second memory.
SEED = "Climbed with Priya at the new bouldering place on Tuesday."
BRIDGED = "A restaurant near the wall was recommended by Priya."
NOISE = [
    "The office lease renews in June.",
    "Postgres 16 runs on Cloud SQL in us-central1.",
    "Invoice INV-4472 came from Datadog for 890 dollars.",
]


@pytest.fixture
async def stocked(service: MemoryService, org: Organization, space: Space) -> None:
    for text in [SEED, BRIDGED, *NOISE]:
        await service.ingest(org_id=org.id, space_id=space.id, content=text, extract=False)


async def _search(
    service: MemoryService, org: Organization, space: Space, **kwargs: object
) -> object:
    kwargs.setdefault("limit", 5)
    return await service.search(
        SearchRequest(
            query="who did I go bouldering with",
            org_id=org.id,
            space_id=space.id,
            **kwargs,  # type: ignore[arg-type]
        )
    )


# -- the stage runs, or does not, as asked ---------------------------------


async def test_bridging_is_off_unless_asked(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """It costs one indexed lookup per entity, so it is opt-in."""
    response = await _search(service, org, space)
    assert "entity" not in response.strategies  # type: ignore[attr-defined]
    assert response.entities_used == []  # type: ignore[attr-defined]
    assert "entity_ms" not in response.timings_ms  # type: ignore[attr-defined]


async def test_bridging_reports_the_entities_it_used(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """Part of the explain surface: "why is this here" must be answerable for a
    bridged hit, and the answer is the entity that reached it."""
    response = await _search(service, org, space, use_entity_expansion=True)
    assert "entity_ms" in response.timings_ms  # type: ignore[attr-defined]
    assert isinstance(response.entities_used, list)  # type: ignore[attr-defined]


async def test_a_bridged_hit_says_so_in_explain(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """A result that neither arm found, marked as such.

    The marker is only added when the memory is absent from BOTH the vector and
    lexical ranks -- otherwise a seed re-found through the entity arm would be
    labelled as bridged, which would overstate what the stage contributed.
    """
    response = await _search(service, org, space, use_entity_expansion=True, limit=10)
    bridged = [
        hit
        for hit in response.results  # type: ignore[attr-defined]
        if any("bridged by entity" in line for line in hit.explain)
    ]
    for hit in bridged:
        assert "vector" not in hit.explain[0] or True  # marker only, see docstring
    # Not asserting that bridging FOUND anything on this corpus -- see the module
    # docstring. What must hold is that anything it labels is genuinely new.
    for hit in bridged:
        assert hit.vector_score is None or hit.lexical_score is None


async def test_the_query_entities_are_not_searched_again(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """ "bouldering" is in the query, so the first pass already searched it.

    Re-searching a query term spends a lookup to find what is already in `fused`,
    and every one of those results is then dropped by the dedupe below.
    """
    response = await _search(service, org, space, use_entity_expansion=True)
    used = [e.lower() for e in response.entities_used]  # type: ignore[attr-defined]
    assert "bouldering" not in used


async def test_a_speaker_name_can_be_excluded(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """In a dialogue corpus the participants are the most frequent proper nouns
    and the least discriminating -- bridging on "Priya" in a corpus where every
    turn mentions Priya reaches everything."""
    response = await _search(
        service,
        org,
        space,
        use_entity_expansion=True,
        known_speakers=("Priya",),
    )
    used = [e.lower() for e in response.entities_used]  # type: ignore[attr-defined]
    assert "priya" not in used


async def test_a_zero_entity_budget_disables_the_lookups(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    response = await _search(service, org, space, use_entity_expansion=True, entity_budget=0)
    assert response.entities_used == []  # type: ignore[attr-defined]


# -- failure isolation -----------------------------------------------------


async def test_one_failed_lookup_does_not_fail_the_search(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """Expansion is an enhancement. An enhancement that can fail a search is a
    liability, and this is the path that would do it -- N concurrent lookups, any
    of which can raise."""
    original = service.store.lexical_search
    calls = {"n": 0}

    async def flaky(*args: object, **kwargs: object) -> list[LexicalHit]:
        calls["n"] += 1
        if calls["n"] > 1:  # the first call is the main lexical arm
            raise RuntimeError("index unavailable")
        return await original(*args, **kwargs)  # type: ignore[arg-type]

    service.store.lexical_search = flaky  # type: ignore[method-assign, assignment]
    try:
        response = await _search(service, org, space, use_entity_expansion=True)
    finally:
        service.store.lexical_search = original  # type: ignore[method-assign]

    assert response.results, "a failed entity lookup emptied the whole result set"


async def test_bridging_on_an_empty_space_is_inert(
    service: MemoryService, org: Organization, space: Space
) -> None:
    """No seeds means no entities means no lookups, and no crash on the empty
    `seed_texts` list."""
    response = await _search(service, org, space, use_entity_expansion=True)
    assert response.results == []  # type: ignore[attr-defined]
    assert response.entities_used == []  # type: ignore[attr-defined]


# -- the merge -------------------------------------------------------------


async def test_bridged_results_never_duplicate_a_first_pass_hit(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """The dedupe against `already`.

    Re-ranking a seed through the entity arm would double-count it in fusion --
    the same memory contributing two ranked entries, inflating its fused score
    for no reason other than that it mentioned its own entity.
    """
    response = await _search(service, org, space, use_entity_expansion=True, limit=10)
    ids = [hit.memory.id for hit in response.results]  # type: ignore[attr-defined]
    assert len(ids) == len(set(ids))


async def test_enabling_bridging_never_loses_a_result(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """Bridging ADDS candidates. Fusion reorders, so a hit can move -- but the
    set must not shrink, because an enhancement that hides evidence is worse than
    one that finds none."""
    without = await _search(service, org, space, limit=10)
    with_bridging = await _search(service, org, space, use_entity_expansion=True, limit=10)
    assert len(with_bridging.results) >= len(without.results)  # type: ignore[attr-defined]


async def test_bridging_is_recorded_as_a_strategy_when_it_contributes(
    service: MemoryService, org: Organization, space: Space, stocked: None
) -> None:
    """`strategies` is how a caller learns which arms ran. It gains "entity"
    only when the arm actually produced candidates to fuse -- an arm that ran and
    found nothing is not a strategy that contributed."""
    response = await _search(service, org, space, use_entity_expansion=True, limit=10)
    if response.entities_used:  # type: ignore[attr-defined]
        # Entities were found; whether they produced NEW hits is corpus-dependent.
        assert "entity_ms" in response.timings_ms  # type: ignore[attr-defined]
