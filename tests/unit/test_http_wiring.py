"""Which `SearchRequest` fields the HTTP route can actually set.

Eight of them could not. `tune_by_intent`, `asked_at`, `use_temporal_scope`,
`use_expansion`, `use_entity_expansion`, `entity_budget`, `entity_weight` and
`known_speakers` were set nowhere in `src/`, so they sat at their dataclass
defaults permanently. Two consequences, both invisible from either side:

  * Pipeline stage 4b -- temporal scope -- is gated on `asked_at is not None`.
    Nothing set it, so the stage was dead code in the product while being
    measured and written up by the benchmark harness, which was the only caller
    that did set it.
  * `use_expansion` and `use_entity_expansion` could not be turned on at all.
    Worse for expansion: `MemoryService` constructed the pipeline without an
    expander, so it defaulted to `NoopExpander` and `HydeExpander` was never
    instantiated anywhere in the tree. The flag was a no-op behind a no-op.

And `use_mmr` disagreed in the other direction: the request body defaulted True
while `SearchRequest` defaults False, with a measurement recorded on the field
saying MMR "changed no retrieval metric while costing 2.5x latency (61ms ->
110ms)". Every HTTP search paid for it.

THE POINT OF THIS FILE is that the wiring is now asserted, so a field becoming
reachable -- or stopping -- is a visible diff rather than something to discover
by reading two files and noticing an absence.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

from mapi.api.schemas import SearchRequestBody
from mapi.domain.retrieval.pipeline import SearchRequest

#: Set from the request body. A caller controls these.
FROM_BODY = frozenset(
    {
        "query",
        "limit",
        "filters",
        "mmr_lambda",
        "half_life_days",
        "use_decay",
        "use_rerank",
        "use_mmr",
        "include_superseded",
        "vector_weight",
        "lexical_weight",
        "min_score",
        "coverage",
        "asked_at",
        "use_temporal_scope",
        "tune_by_intent",
        "use_expansion",
        "use_entity_expansion",
        "entity_budget",
        "entity_weight",
        "known_speakers",
    }
)

#: Set from `Settings`, deliberately. These are operator knobs: a caller who
#: could raise `rerank_candidates` or `candidate_multiplier` could make one
#: request cost an arbitrary amount of work.
FROM_SETTINGS = frozenset(
    {
        "candidate_multiplier",
        "rerank_candidates",
        "rrf_k",
        "coverage_limit",
        "max_per_source",
        "route_by_kind",
    }
)

#: From the principal and the path, never from the body -- `org_id` comes off the
#: authenticated key so a caller cannot name another tenant's.
FROM_CONTEXT = frozenset({"org_id", "space_id"})


def test_every_search_request_field_is_accounted_for() -> None:
    """The whole surface, partitioned. A new field must be classified.

    This is the test that would have caught the original defect: eight fields
    belonged to no category and nothing said so.
    """
    known = {f.name for f in dataclass_fields(SearchRequest)}
    classified = FROM_BODY | FROM_SETTINGS | FROM_CONTEXT
    assert known - classified == set(), (
        f"unclassified SearchRequest fields: {sorted(known - classified)}. "
        "Add each to FROM_BODY, FROM_SETTINGS or FROM_CONTEXT -- an unclassified "
        "field is one nothing can set."
    )
    assert classified - known == set(), (
        f"classified fields that no longer exist: {sorted(classified - known)}"
    )


def test_the_body_exposes_every_field_it_claims_to() -> None:
    """`FROM_BODY` and the schema must agree, or the table above is fiction."""
    body_fields = set(SearchRequestBody.model_fields)
    # The body carries filter inputs and `explain` that do not map 1:1 onto
    # SearchRequest fields; everything else must be present.
    expected = FROM_BODY - {"query", "filters", "limit"}
    missing = expected - body_fields
    assert not missing, f"SearchRequestBody is missing {sorted(missing)}"


def test_the_three_categories_are_disjoint() -> None:
    assert not (FROM_BODY & FROM_SETTINGS)
    assert not (FROM_BODY & FROM_CONTEXT)
    assert not (FROM_SETTINGS & FROM_CONTEXT)


# -- defaults that must agree across the boundary --------------------------


def test_use_mmr_defaults_the_same_on_both_sides() -> None:
    """The disagreement that made every HTTP search pay for a measured no-op."""
    body_default = SearchRequestBody.model_fields["use_mmr"].default
    request_default = next(
        f.default for f in dataclass_fields(SearchRequest) if f.name == "use_mmr"
    )
    assert body_default == request_default is False


def test_expansion_flags_default_off() -> None:
    """Wiring is not enabling.

    Both cost real resources per request -- an LLM call for HyDE, N store lookups
    for entity bridging -- so making them reachable must not make them the
    default. A caller opts in.
    """
    for name in ("use_expansion", "use_entity_expansion"):
        assert SearchRequestBody.model_fields[name].default is False, name


def test_temporal_scope_defaults_on_because_it_is_a_bias() -> None:
    """The one flag that is safe to default on.

    `apply_scope` multiplies in-window scores and removes nothing, and it does
    nothing at all unless the query names a window `extract_scope` can parse. So
    the risk of leaving it on is bounded, and the cost of leaving it off was a
    dead pipeline stage.
    """
    assert SearchRequestBody.model_fields["use_temporal_scope"].default is True


def test_asked_at_is_optional_on_the_body() -> None:
    """Optional in, defaulted at the route.

    The route fills it with `utcnow()` when absent, which is what brings stage 4b
    to life for ordinary requests. A caller replaying history can still pass its
    own, which is the case that has to keep working.
    """
    assert SearchRequestBody.model_fields["asked_at"].default is None


def test_min_score_is_bounded_to_the_cosine_range() -> None:
    """It is compared against a cosine now, so >1.0 can never match anything.

    Bounding it turns a silently-empty result set into a 422 that says what is
    wrong. Before the confidence fix the field was unbounded because the scale it
    was compared against was unbounded too.
    """
    meta = SearchRequestBody.model_fields["min_score"].metadata
    assert any(getattr(m, "le", None) == 1.0 for m in meta), meta


# -- the expander, which was a no-op behind a no-op ------------------------


async def test_a_service_with_a_completer_gets_a_real_expander() -> None:
    """`use_expansion=True` used to do nothing even when set.

    `MemoryService` built the pipeline without an `expander`, so it fell back to
    `NoopExpander`, and `HydeExpander` -- which builds its own vendor client and
    therefore cannot be constructed by a deliberately vendor-free service -- was
    never instantiated anywhere in the tree.
    """
    from mapi.config import Settings
    from mapi.domain.embeddings.deterministic import DeterministicEmbedder
    from mapi.domain.retrieval.rerank import HeuristicReranker
    from mapi.service import MemoryService
    from mapi.store.memory import InMemoryStore

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=128,
        api_key_pepper="wiring-pepper-value",
    )

    async def complete(prompt: str) -> str:
        return "a hypothetical answer passage"

    with_model = MemoryService(
        InMemoryStore(),
        DeterministicEmbedder(dimensions=128),
        HeuristicReranker(),
        settings,
        completer=complete,
    )
    assert with_model.pipeline.expander.name == "hyde"

    without = MemoryService(
        InMemoryStore(),
        DeterministicEmbedder(dimensions=128),
        HeuristicReranker(),
        settings,
    )
    assert without.pipeline.expander.name == "none"
