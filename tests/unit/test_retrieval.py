"""Fusion, MMR, decay and reranking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mapi.domain.retrieval.decay import (
    MAX_ACCESS_BOOST,
    access_factor,
    age_days,
    apply_decay,
    recency_factor,
)
from mapi.domain.retrieval.fusion import (
    RankedList,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from mapi.domain.retrieval.mmr import MMRCandidate, maximal_marginal_relevance
from mapi.domain.retrieval.rerank import (
    HeuristicReranker,
    LLMReranker,
    NoopReranker,
    RerankCandidate,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


# -- fusion --------------------------------------------------------------------


def test_rrf_rewards_agreement_across_lists() -> None:
    a = RankedList("vector", ["x", "y", "z"])
    b = RankedList("lexical", ["z", "x", "w"])
    fused = reciprocal_rank_fusion([a, b])
    # x is 1st and 2nd; z is 3rd and 1st. x should win on combined ranks.
    assert fused[0].id == "x"
    assert {i.id for i in fused} == {"x", "y", "z", "w"}


def test_rrf_is_deterministic_on_ties() -> None:
    a = RankedList("one", ["p", "q"])
    b = RankedList("two", ["q", "p"])
    first = [i.id for i in reciprocal_rank_fusion([a, b])]
    second = [i.id for i in reciprocal_rank_fusion([a, b])]
    assert first == second == sorted(first)


def test_rrf_handles_empty_and_zero_weight() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([RankedList("empty", [])]) == []
    fused = reciprocal_rank_fusion(
        [RankedList("keep", ["a"]), RankedList("drop", ["b"], weight=0.0)]
    )
    assert [i.id for i in fused] == ["a"]


def test_rrf_weight_shifts_the_winner() -> None:
    a = RankedList("vector", ["x", "y"], weight=1.0)
    b = RankedList("lexical", ["y", "x"], weight=10.0)
    assert reciprocal_rank_fusion([a, b])[0].id == "y"


def test_rrf_rejects_duplicate_ids_in_one_list() -> None:
    with pytest.raises(ValueError, match="duplicate ids"):
        RankedList("bad", ["a", "a"])


def test_rrf_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RankedList("bad", ["a"], weight=-1.0)


def test_rrf_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion([RankedList("a", ["x"])], k=0)


def test_rrf_limit_truncates() -> None:
    fused = reciprocal_rank_fusion([RankedList("a", list("abcdef"))], limit=2)
    assert len(fused) == 2


def test_fused_item_explains_itself() -> None:
    fused = reciprocal_rank_fusion(
        [RankedList("vector", ["a"], {"a": 0.9}), RankedList("lexical", ["a"])]
    )
    explanation = fused[0].explain()
    assert any("vector rank 1" in part for part in explanation)
    assert any("fused" in part for part in explanation)


def test_weighted_fusion_handles_flat_scores() -> None:
    """A list where every score is identical must not divide by zero."""
    ranked = RankedList("flat", ["a", "b"], {"a": 1.0, "b": 1.0})
    fused = weighted_score_fusion([ranked])
    assert len(fused) == 2
    assert all(item.score > 0 for item in fused)


# -- MMR -----------------------------------------------------------------------


def test_mmr_with_lambda_one_is_pure_relevance() -> None:
    candidates = [
        MMRCandidate("dup1", 1.0, [1.0, 0.0]),
        MMRCandidate("dup2", 0.99, [1.0, 0.0]),
        MMRCandidate("other", 0.5, [0.0, 1.0]),
    ]
    picked = [s.id for s in maximal_marginal_relevance(candidates, limit=2, lambda_=1.0)]
    assert picked == ["dup1", "dup2"]


def test_mmr_diversifies_away_from_near_duplicates() -> None:
    candidates = [
        MMRCandidate("dup1", 1.0, [1.0, 0.0]),
        MMRCandidate("dup2", 0.99, [1.0, 0.0]),
        MMRCandidate("other", 0.5, [0.0, 1.0]),
    ]
    picked = [s.id for s in maximal_marginal_relevance(candidates, limit=2, lambda_=0.5)]
    assert picked == ["dup1", "other"]


def test_mmr_keeps_candidates_without_embeddings() -> None:
    candidates = [MMRCandidate("a", 1.0), MMRCandidate("b", 0.5)]
    assert len(maximal_marginal_relevance(candidates, limit=2)) == 2


def test_mmr_deduplicates_repeated_ids() -> None:
    candidates = [MMRCandidate("a", 1.0), MMRCandidate("a", 0.9)]
    assert len(maximal_marginal_relevance(candidates, limit=5)) == 1


@pytest.mark.parametrize("limit", [0, -1])
def test_mmr_non_positive_limit_returns_empty(limit: int) -> None:
    assert maximal_marginal_relevance([MMRCandidate("a", 1.0)], limit=limit) == []


def test_mmr_empty_candidates() -> None:
    assert maximal_marginal_relevance([], limit=5) == []


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_mmr_rejects_lambda_outside_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="lambda_"):
        maximal_marginal_relevance([MMRCandidate("a", 1.0)], limit=1, lambda_=bad)


def test_mmr_limit_exceeding_candidates_returns_all() -> None:
    result = maximal_marginal_relevance([MMRCandidate("a", 1.0)], limit=10)
    assert len(result) == 1
    assert result[0].rank == 1


# -- decay ---------------------------------------------------------------------


def test_recency_factor_is_one_at_zero_age() -> None:
    assert recency_factor(NOW, now=NOW) == pytest.approx(1.0)


def test_recency_halves_at_the_half_life() -> None:
    factor = recency_factor(NOW - timedelta(days=180), now=NOW, half_life_days=180, floor=0.0)
    assert factor == pytest.approx(0.5, abs=1e-9)


def test_recency_never_falls_below_the_floor() -> None:
    factor = recency_factor(
        NOW - timedelta(days=100_000), now=NOW, half_life_days=30, floor=0.25
    )
    assert factor == pytest.approx(0.25)


def test_recency_is_monotonically_decreasing() -> None:
    factors = [recency_factor(NOW - timedelta(days=d), now=NOW) for d in (0, 30, 90, 365, 1000)]
    assert factors == sorted(factors, reverse=True)


def test_future_timestamps_are_clamped_to_now() -> None:
    assert recency_factor(NOW + timedelta(days=365), now=NOW) == pytest.approx(1.0)
    assert age_days(NOW + timedelta(days=5), now=NOW) == 0.0


def test_naive_datetimes_are_treated_as_utc() -> None:
    naive = datetime(2025, 7, 5)
    assert recency_factor(naive, now=NOW) > 0


def test_apply_decay_returns_score_and_factor() -> None:
    score, factor = apply_decay(
        1.0, NOW - timedelta(days=180), now=NOW, half_life_days=180, floor=0.0
    )
    assert score == pytest.approx(0.5, abs=1e-9)
    assert factor == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("half_life", [0, -1])
def test_invalid_half_life_rejected(half_life: float) -> None:
    with pytest.raises(ValueError, match="half_life"):
        recency_factor(NOW, now=NOW, half_life_days=half_life)


@pytest.mark.parametrize("floor", [-0.1, 1.1])
def test_invalid_floor_rejected(floor: float) -> None:
    with pytest.raises(ValueError, match="floor"):
        recency_factor(NOW, now=NOW, floor=floor)


# -- reranking -----------------------------------------------------------------

DOCS = [
    RerankCandidate("a", "the cat sat on the mat", 0.5),
    RerankCandidate("b", "quantum chromodynamics and gluon fields", 0.9),
    RerankCandidate("c", "my cat likes to sit on mats all day long", 0.4),
]


async def test_heuristic_ranks_topical_match_above_prior() -> None:
    results = await HeuristicReranker().rerank("where does the cat sit", DOCS, limit=3)
    assert results[0].id in {"a", "c"}
    assert results[-1].id == "b"


async def test_heuristic_handles_empty_candidates() -> None:
    assert await HeuristicReranker().rerank("q", [], limit=5) == []


async def test_heuristic_falls_back_on_stopword_only_query() -> None:
    results = await HeuristicReranker().rerank("the and of", DOCS, limit=3)
    assert all(not r.reranked for r in results)
    assert results[0].id == "b"  # highest prior


async def test_noop_preserves_prior_order() -> None:
    results = await NoopReranker().rerank("anything", DOCS, limit=3)
    assert [r.id for r in results] == ["b", "a", "c"]
    assert all(not r.reranked for r in results)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[2, 0, 1]", [2, 0, 1]),
        ("```json\n[1,0,2]\n```", [1, 0, 2]),
        ("Sure! [0, 2, 1] hope that helps", [0, 2, 1]),
        ('{"order": [2,1,0]}', [2, 1, 0]),
        ("[2]", [2, 0, 1]),  # partial: missing appended in order
        ("[1,1,1,0]", [1, 0, 2]),  # duplicates collapsed
        ("[9, 1, -4, 0]", [1, 0, 2]),  # out-of-range dropped
    ],
)
def test_llm_order_parsing_tolerates_malformed_output(raw, expected) -> None:
    assert LLMReranker.parse_order(raw, 3) == expected


@pytest.mark.parametrize("raw", ["", "I cannot help", "[]", "[true, false]"])
def test_llm_order_parsing_returns_none_when_unusable(raw: str) -> None:
    assert LLMReranker.parse_order(raw, 3) is None


def test_llm_order_parsing_guards_zero_length() -> None:
    assert LLMReranker.parse_order("[0]", 0) is None


def test_llm_sanitizer_strips_fences_and_control_characters() -> None:
    reranker = object.__new__(LLMReranker)
    reranker.max_doc_chars = 200
    cleaned = reranker._sanitize("ignore this ``` and \x01 that")
    assert "```" not in cleaned
    assert "\x01" not in cleaned


def test_llm_sanitizer_truncates_long_documents() -> None:
    reranker = object.__new__(LLMReranker)
    reranker.max_doc_chars = 50
    cleaned = reranker._sanitize("y" * 500)
    assert len(cleaned) < 100
    assert cleaned.endswith("...[truncated]")


# -- reconsolidation: access as a RANKING signal, never a deletion policy ------


def test_access_factor_is_neutral_for_never_retrieved_memories() -> None:
    assert access_factor(0) == 1.0
    assert access_factor(-5) == 1.0


def test_access_factor_is_bounded() -> None:
    """A much-retrieved memory must never outrank a genuinely better match.
    The boost is a tie-breaker, not a popularity contest."""
    assert access_factor(10_000) <= 1.0 + MAX_ACCESS_BOOST + 1e-9
    assert access_factor(1) < access_factor(5) < access_factor(20)


def test_access_factor_saturates_logarithmically() -> None:
    """The first few retrievals carry information; the thousandth does not."""
    early = access_factor(3) - access_factor(1)
    late = access_factor(1000) - access_factor(998)
    assert early > late


def test_access_factor_never_reaches_zero() -> None:
    """Nothing here can hide, stale, or delete a memory — it only reorders.
    Every competitor implements replay-strengthening as forgetting; the
    independent cost study (arXiv 2603.04814) is why we do not."""
    assert all(access_factor(n) >= 1.0 for n in range(0, 200))
