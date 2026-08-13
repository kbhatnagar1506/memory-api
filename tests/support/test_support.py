"""Self-tests for the harness. The one file whose failure invalidates the rest.

`at_cosine` returning 0.79 when asked for 0.80 would not fail here loudly -- it
would make a hundred downstream capability assertions quietly meaningless, all
of them still green. So the constructions are checked against the definition
they claim to satisfy, and the metrics against hand-computed tables.

Also holds the two Phase 0 guards: that `bench` is importable (the CI collection
failure), and that every marker declared in pyproject.toml is actually used.
"""

from __future__ import annotations

import math

import pytest
from tests.support import metrics
from tests.support.corpus import Fact, Query, build, fictional, register
from tests.support.embedder import MISS_AXIS, ScriptedEmbedder
from tests.support.vectors import (
    ANCHOR,
    COSINE_TOLERANCE,
    DIMS,
    at_cosine,
    axis,
    band,
    brute_force_topk,
    cone,
    cosine_of,
    nudge,
)

from mapi.domain.embeddings.base import cosine_similarity

# -- Phase 0 guards ---------------------------------------------------------


def test_bench_is_importable() -> None:
    """The CI-red bug: bare `pytest` could not import `bench`.

    `tests/unit/test_bench.py` imports `bench.*`, `bench/` is not an installed
    package, and pytest inserts each module's basedir rather than the rootdir.
    So bare `pytest` -- which is what .github/workflows/ci.yml runs -- died at
    COLLECTION and took the whole run with it, while `python -m pytest` passed
    locally because it puts CWD on the path. Fixed with `pythonpath = ["."]`.
    """
    import bench.metrics

    assert callable(bench.metrics.full_recall_at_k)


#: Markers declared for work not yet landed. Each entry names the file that
#: will use it, and MUST be deleted when that file appears -- at which point
#: this test starts enforcing the marker for real. An entry that outlives its
#: phase is the rot this test exists to catch.
_MARKERS_PENDING = {
    "slow": "tests/capability/test_scale.py (Phase 5)",
    "live_llm": "tests/capability/test_semantic_embedder.py (Phase 4)",
}


def test_every_declared_marker_is_used() -> None:
    """`--strict-markers` catches undeclared markers, not unused declarations.

    `postgres` was declared and applied to nothing, so `-m "not postgres"`
    silently selected the whole suite -- a lever that looked available and was
    not. Gating happened inside a fixture's `pytest.skip` instead, which cannot
    be deselected.

    Anything genuinely not yet used belongs in `_MARKERS_PENDING` with the file
    that will use it, so the debt is named rather than invisible.
    """
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    declared = {
        entry.split(":", 1)[0] for entry in config["tool"]["pytest"]["ini_options"]["markers"]
    }

    used: set[str] = set()
    for path in (root / "tests").rglob("*.py"):
        used |= set(re.findall(r"pytest\.mark\.(\w+)", path.read_text()))

    unused = declared - used - set(_MARKERS_PENDING)
    assert not unused, f"markers declared but never applied: {sorted(unused)}"

    # And the reverse: a pending marker that IS now used must leave the list,
    # or the exemption silently protects a real regression later.
    landed = set(_MARKERS_PENDING) & used
    assert not landed, (
        f"these markers are now used and must be removed from _MARKERS_PENDING: "
        f"{sorted(landed)}"
    )


# -- vectors: the constructions ---------------------------------------------


def test_axes_are_orthonormal() -> None:
    for i in (0, 1, 7, DIMS - 1):
        assert cosine_of(axis(i), axis(i)) == pytest.approx(1.0, abs=COSINE_TOLERANCE)
    assert cosine_of(axis(0), axis(1)) == pytest.approx(0.0, abs=COSINE_TOLERANCE)
    assert cosine_of(axis(3), axis(9)) == pytest.approx(0.0, abs=COSINE_TOLERANCE)


@pytest.mark.parametrize("target", [0.0, 0.05, 0.45, 0.72, 0.8, 0.82, 0.97, 1.0])
def test_at_cosine_hits_its_target_exactly(target: float) -> None:
    """The load-bearing claim of the whole harness."""
    vector = at_cosine(target, off=5)
    assert cosine_of(vector, axis(ANCHOR)) == pytest.approx(target, abs=COSINE_TOLERANCE)


@pytest.mark.parametrize("target", [0.3, 0.8, 0.97])
def test_at_cosine_survives_the_provider_contract(target: float) -> None:
    """`_validate` re-normalizes; a construction must be invariant under that."""
    vector = at_cosine(target, off=2)
    norm = math.sqrt(sum(x * x for x in vector))
    assert norm == pytest.approx(1.0, abs=COSINE_TOLERANCE)
    renormalized = [x / norm for x in vector]
    assert cosine_of(renormalized, axis(ANCHOR)) == pytest.approx(target, abs=COSINE_TOLERANCE)


def test_at_cosine_agrees_with_the_production_cosine() -> None:
    """The harness and `embeddings.base.cosine_similarity` must not disagree."""
    for target in (0.15, 0.6, 0.9):
        vector = at_cosine(target, off=4)
        assert cosine_similarity(vector, axis(ANCHOR)) == pytest.approx(
            target, abs=COSINE_TOLERANCE
        )


def test_at_cosine_refuses_the_anchor_axis() -> None:
    """Axis 0 is the query direction; a distractor there is not a distractor."""
    with pytest.raises(ValueError, match="anchor"):
        at_cosine(0.5, off=ANCHOR)


@pytest.mark.parametrize("bad", [-1.5, 1.5])
def test_at_cosine_rejects_impossible_cosines(bad: float) -> None:
    with pytest.raises(ValueError, match="outside"):
        at_cosine(bad, off=1)


def test_band_members_are_equidistant_from_the_anchor() -> None:
    members = band(0.8, 5)
    for member in members:
        assert cosine_of(member, axis(ANCHOR)) == pytest.approx(0.8, abs=COSINE_TOLERANCE)


def test_band_members_are_c_squared_from_each_other() -> None:
    """The documented gotcha, asserted so nobody has to rediscover it.

    A band at 0.8 is an internally-0.64 cloud. A test that assumed members were
    mutually distant would be wrong in a way that only shows up as a confusing
    MMR or near-duplicate result.
    """
    members = band(0.8, 4)
    for i, left in enumerate(members):
        for right in members[i + 1 :]:
            assert cosine_of(left, right) == pytest.approx(0.64, abs=COSINE_TOLERANCE)


def test_band_refuses_to_exceed_the_dimension_budget() -> None:
    with pytest.raises(ValueError, match="dims"):
        band(0.5, DIMS + 1)


@pytest.mark.parametrize(("outer", "inner"), [(0.8, 0.9), (0.6, 0.5), (0.9, 0.95)])
def test_cone_controls_both_distances(outer: float, inner: float) -> None:
    """What a band cannot express: close to the query AND close to each other."""
    members = cone(outer, inner, 4)
    for member in members:
        assert cosine_of(member, axis(ANCHOR)) == pytest.approx(outer, abs=COSINE_TOLERANCE)
    for i, left in enumerate(members):
        for right in members[i + 1 :]:
            assert cosine_of(left, right) == pytest.approx(inner, abs=COSINE_TOLERANCE)


def test_cone_refuses_an_unreachable_inner_cosine() -> None:
    """Members at cosine c from the anchor cannot be further apart than c**2."""
    with pytest.raises(ValueError, match="unreachable"):
        cone(0.8, 0.5, 3)  # 0.5 < 0.8**2 = 0.64


def test_cone_refuses_a_degenerate_apex() -> None:
    with pytest.raises(ValueError, match="single point"):
        cone(1.0, 1.0, 3)


def test_nudge_perturbs_by_a_controlled_amount() -> None:
    base = at_cosine(0.8, off=3)
    moved = nudge(base, 1e-6, off=4)
    assert cosine_of(base, moved) == pytest.approx(1.0, abs=1e-9)
    assert moved != base


# -- vectors: the oracle ----------------------------------------------------


def test_brute_force_topk_orders_by_cosine() -> None:
    corpus = {
        "far": at_cosine(0.2, off=1),
        "near": at_cosine(0.9, off=2),
        "mid": at_cosine(0.6, off=3),
    }
    assert brute_force_topk(axis(ANCHOR), corpus, 3) == ["near", "mid", "far"]


def test_brute_force_topk_breaks_ties_on_id() -> None:
    """Must match `InMemoryStore.vector_search`, or a disagreement is spurious."""
    corpus = {"zeta": at_cosine(0.7, off=1), "alpha": at_cosine(0.7, off=2)}
    assert brute_force_topk(axis(ANCHOR), corpus, 2) == ["alpha", "zeta"]


async def test_brute_force_topk_agrees_with_the_in_memory_store() -> None:
    """The oracle is only an oracle if it matches the thing it audits."""
    from tests.support.factories import store_memory, tenant

    from mapi.store.memory import InMemoryStore

    store = InMemoryStore()
    org, space = await tenant(store)
    placed = {}
    for i, target in enumerate((0.95, 0.71, 0.44, 0.12), start=1):
        stored = await store_memory(
            store,
            org_id=org.id,
            space_id=space.id,
            content=f"fact number {i}",
            vector=at_cosine(target, off=i),
        )
        placed[stored.id] = at_cosine(target, off=i)

    from mapi.store.base import MemoryFilter

    hits = await store.vector_search(
        org_id=org.id,
        space_id=space.id,
        embedding=axis(ANCHOR),
        limit=4,
        filters=MemoryFilter(),
    )
    assert [h.memory_id for h in hits] == brute_force_topk(axis(ANCHOR), placed, 4)


# -- the scripted embedder --------------------------------------------------


async def test_scripted_embedder_returns_what_was_registered() -> None:
    embedder = ScriptedEmbedder()
    want = at_cosine(0.83, off=6)
    embedder.register("the exact query", want)
    got = await embedder.embed_one("the exact query")
    assert cosine_of(got, axis(ANCHOR)) == pytest.approx(0.83, abs=COSINE_TOLERANCE)
    assert embedder.misses == []


async def test_a_marker_survives_the_contextual_header() -> None:
    """The reason markers exist rather than exact strings.

    `service.ingest` embeds `for_embedding(text, header)`, not `text`. An exact
    lookup misses on every real write; a marker in the content does not.
    """
    from mapi.domain.embeddings.context import build_header, for_embedding

    embedder = ScriptedEmbedder()
    embedder.register_marker("g0007", at_cosine(0.91, off=8))

    header = build_header(
        occurred_at=None, source="user", tags=("infra",), metadata={"title": "Review"}
    )
    wrapped = for_embedding("fact g0007: the deployment moved", header)
    assert wrapped != "fact g0007: the deployment moved"

    got = await embedder.embed_one(wrapped)
    assert cosine_of(got, axis(ANCHOR)) == pytest.approx(0.91, abs=COSINE_TOLERANCE)
    assert embedder.misses == []


async def test_the_longest_marker_wins() -> None:
    embedder = ScriptedEmbedder()
    embedder.register_marker("g1", at_cosine(0.2, off=1))
    embedder.register_marker("g10", at_cosine(0.9, off=2))
    got = await embedder.embed_one("about g10 specifically")
    assert cosine_of(got, axis(ANCHOR)) == pytest.approx(0.9, abs=COSINE_TOLERANCE)


async def test_a_miss_is_recorded_and_does_not_raise() -> None:
    """The subtle safety property, and the reason for the autouse assertion.

    `_with_retries` wraps any exception in `ProviderError`, and
    `RetrievalPipeline._embed_query` catches that and returns None -- silently
    degrading to lexical-only. An embedder that raised on an unregistered query
    would produce a GREEN test that never ran vector search at all.
    """
    embedder = ScriptedEmbedder()
    got = await embedder.embed_one("never registered")
    assert embedder.misses == ["never registered"]
    assert cosine_of(got, axis(MISS_AXIS)) == pytest.approx(1.0, abs=COSINE_TOLERANCE)


async def test_a_miss_lands_orthogonal_to_the_corpus() -> None:
    """So a miss can never accidentally rank first and look like success."""
    embedder = ScriptedEmbedder()
    missed = await embedder.embed_one("unregistered")
    assert cosine_of(missed, axis(ANCHOR)) == pytest.approx(0.0, abs=COSINE_TOLERANCE)
    assert cosine_of(missed, at_cosine(0.9, off=3)) == pytest.approx(0.0, abs=COSINE_TOLERANCE)


def test_the_embedder_defines_no_query_side_transform() -> None:
    """`RetrievalPipeline` prefers `embed_query`; an asymmetric transform here
    would invalidate every cosine the harness claims to control."""
    assert not hasattr(ScriptedEmbedder, "embed_query")


def test_registering_a_wrong_width_vector_fails_loudly() -> None:
    embedder = ScriptedEmbedder()
    with pytest.raises(ValueError, match="dimensions"):
        embedder.register("x", [1.0, 0.0])


# -- corpus -----------------------------------------------------------------


def test_fictional_entities_avoid_real_world_tokens() -> None:
    names = fictional(seed=7, count=40)
    assert len(set(names)) == 40
    lowered = " ".join(names).lower()
    for real in ("postgres", "redis", "python", "google", "claude"):
        assert real not in lowered


def test_fictional_is_deterministic_in_its_seed() -> None:
    assert fictional(seed=3, count=10) == fictional(seed=3, count=10)
    assert fictional(seed=3, count=10) != fictional(seed=4, count=10)


def test_a_corpus_rejects_duplicate_gold_ids() -> None:
    facts = [
        Fact(gold_id="g1", text="a g1", vector=at_cosine(0.9, off=1)),
        Fact(gold_id="g1", text="b g1", vector=at_cosine(0.8, off=2)),
    ]
    with pytest.raises(ValueError, match="duplicate gold ids"):
        build(facts, [])


def test_a_corpus_rejects_a_query_naming_unknown_gold() -> None:
    facts = [Fact(gold_id="g1", text="a g1", vector=at_cosine(0.9, off=1))]
    queries = [Query(key="q", text="?", vector=axis(ANCHOR), gold_ids=("g2",))]
    with pytest.raises(ValueError, match="unknown gold"):
        build(facts, queries)


async def test_load_records_memory_ids_and_preserves_insertion_order() -> None:
    from tests.support.corpus import load
    from tests.support.factories import tenant

    from mapi.store.memory import InMemoryStore

    facts = [
        Fact(gold_id=f"g{i}", text=f"fact g{i} about a thing", vector=at_cosine(0.5, off=i))
        for i in range(1, 5)
    ]
    corpus = build(facts, [])
    store = InMemoryStore()
    org, space = await tenant(store)
    await load(store, corpus, org_id=org.id, space_id=space.id)

    assert all(f.memory_id for f in corpus.facts)
    assert corpus.gold_of([f.memory_id for f in corpus.facts]) == ["g1", "g2", "g3", "g4"]
    stored = await store.get_memory(
        org_id=org.id, space_id=space.id, memory_id=corpus.fact("g3").memory_id
    )
    assert stored is not None
    assert stored.metadata["insert_order"] == 2


async def test_register_binds_facts_by_marker_and_queries_exactly() -> None:
    facts = [Fact(gold_id="g9", text="fact g9 here", vector=at_cosine(0.77, off=3))]
    queries = [Query(key="q", text="what about it", vector=axis(ANCHOR), gold_ids=("g9",))]
    corpus = build(facts, queries)
    embedder = ScriptedEmbedder()
    register(corpus, embedder)

    # A fact matches through a header it was never registered with.
    got = await embedder.embed_one("<context>\n2026-01-01\n</context>\nfact g9 here")
    assert cosine_of(got, axis(ANCHOR)) == pytest.approx(0.77, abs=COSINE_TOLERANCE)
    assert embedder.misses == []


# -- metrics ----------------------------------------------------------------


def test_recall_at_k_counts_the_prefix_only() -> None:
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "c"], 2) == pytest.approx(0.5)
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "c"], 3) == pytest.approx(1.0)


def test_joint_recall_is_all_or_nothing() -> None:
    """The metric `recall@k` hides: three of four facts is a wrong answer."""
    assert metrics.joint_recall_at_k(["a", "b", "c"], ["a", "b", "c", "d"], 4) == 0.0
    assert metrics.joint_recall_at_k(["a", "b", "c", "d"], ["a", "b", "c", "d"], 4) == 1.0
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "b", "c", "d"], 4) == pytest.approx(0.75)


def test_sufficient_at_k_tracks_joint_recall() -> None:
    assert metrics.sufficient_at_k(["a", "b"], ["a", "b"], 2) is True
    assert metrics.sufficient_at_k(["a"], ["a", "b"], 2) is False


@pytest.mark.parametrize(
    "fn", [metrics.recall_at_k, metrics.joint_recall_at_k, metrics.ndcg_at_k]
)
def test_an_empty_gold_set_scores_zero_not_one(fn: object) -> None:
    """Matches `bench/metrics.py`: a question with no evidence scores 0.0.

    The tempting alternative -- "no requirements, so trivially satisfied" --
    would make an unlabelled question look like a success and inflate every
    aggregate it appears in.
    """
    assert fn(["a", "b"], [], 2) == 0.0  # type: ignore[operator]


def test_mrr_with_no_gold_scores_zero() -> None:
    assert metrics.mrr(["a", "b"], []) == 0.0


def test_unique_information_counts_facts_not_rows() -> None:
    """Ten paraphrases of one fact score a perfect recall and one fact."""
    fact_of = {"m1": "g1", "m2": "g1", "m3": "g1", "m4": "g2"}
    assert metrics.unique_information_at_k(["m1", "m2", "m3"], fact_of, 3) == 1
    assert metrics.unique_information_at_k(["m1", "m4"], fact_of, 2) == 2


def test_rank_of_is_one_based_and_none_when_absent() -> None:
    assert metrics.rank_of(["a", "b"], "a") == 1
    assert metrics.rank_of(["a", "b"], "b") == 2
    assert metrics.rank_of(["a", "b"], "z") is None


def test_mrr_uses_the_first_gold_hit() -> None:
    assert metrics.mrr(["x", "a"], ["a"]) == pytest.approx(0.5)
    assert metrics.mrr(["a", "x"], ["a"]) == pytest.approx(1.0)
    assert metrics.mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_is_one_for_a_perfect_ordering() -> None:
    assert metrics.ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(1.0 / math.log2(3))


def test_effective_x_reports_the_last_point_above_the_floor() -> None:
    """NoLiMa's definition: the largest sweep point holding 85% of baseline."""
    curve = [(100, 1.0), (1000, 0.95), (10_000, 0.86), (100_000, 0.40)]
    assert metrics.effective_x(curve) == 10_000


def test_effective_x_stops_at_the_first_cliff() -> None:
    """A later recovery must not erase a cliff that happened."""
    curve = [(100, 1.0), (1000, 0.10), (10_000, 0.99)]
    assert metrics.effective_x(curve) == 100


def test_effective_x_is_none_when_nothing_qualifies() -> None:
    assert metrics.effective_x([]) is None
