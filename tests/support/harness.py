"""Search presets, each documenting why a stage is on or off.

A capability test that constructs a geometry and then leaves the default
pipeline running is not measuring the geometry. Five stages will have rewritten
it before the assertion runs, and the resulting failure is unattributable. This
module names the combinations once, with the reason next to each switch, so a
test reads `**GEOMETRY` and a reader can find out what that bought.

The reasons are specific to this pipeline and were verified by reading it:

  * **`use_rerank`** -- `HeuristicReranker` OVERWRITES `ScoredMemory.score`
    with its own IDF-weighted lexical score. It does not adjust the fused
    score, it replaces it, so every constructed cosine is gone by stage 5. It
    is also unbounded: a top score of 5.78 was measured on a five-fact corpus,
    against confidence thresholds of 0.55/0.30.
  * **`use_decay`** -- multiplies by recency computed against the wall clock.
    Any test asserting a score value is non-hermetic with this on: it passes
    today and fails when the calendar moves.
  * **`use_mmr`** -- reorders for diversity, which is the opposite of what a
    ranking assertion wants.
  * **`lexical_weight = 0.0`** -- a true off switch, not a soft one:
    `fusion.reciprocal_rank_fusion` SKIPS a ranked list whose weight is zero
    entirely, so BM25 contributes nothing rather than contributing a little.
  * **`tune_by_intent`** -- rescales both arms by question shape
    (`retrieval/tuning.py`), so the same corpus ranks differently depending on
    how the query is phrased.
  * **`coverage = False`** -- otherwise a question that reads as comprehensive
    silently widens the result window and a `len(results)` assertion measures
    the classifier instead of the ranker.

ONE TRAP, recorded here because it has already cost time: with
`lexical_weight=0.0`, `SearchResponse.strategies` STILL contains `"lexical"`.
The list is built from whether the lexical arm returned hits, not from whether
its weight let them count. Assert on results and ranks, never on `strategies`.
"""

from __future__ import annotations

from typing import Any

#: Pure vector geometry. Everything that could rewrite a score is off.
GEOMETRY: dict[str, Any] = {
    "use_rerank": False,
    "use_decay": False,
    "use_mmr": False,
    "lexical_weight": 0.0,
    "tune_by_intent": False,
    "coverage": False,
}

#: The lexical arm alone, for the other half of a fusion claim.
LEXICAL_ONLY: dict[str, Any] = {
    "vector_weight": 0.0,
    "use_rerank": False,
    "use_decay": False,
    "use_mmr": False,
    "tune_by_intent": False,
    "coverage": False,
}

#: Both arms fused, nothing downstream rewriting the result. For testing
#: fusion itself rather than either arm.
FUSED: dict[str, Any] = {
    "use_rerank": False,
    "use_decay": False,
    "use_mmr": False,
    "tune_by_intent": False,
    "coverage": False,
}

#: Whatever the HTTP route actually sets. Deliberately empty: the defaults ARE
#: the configuration under test, and `tests/unit/test_http_wiring.py` pins which
#: fields the route can set at all.
PRODUCTION: dict[str, Any] = {}


def preset(name: str, **overrides: Any) -> dict[str, Any]:
    """A preset with per-test overrides, so a test can name its one exception."""
    presets = {
        "geometry": GEOMETRY,
        "lexical": LEXICAL_ONLY,
        "fused": FUSED,
        "production": PRODUCTION,
    }
    if name not in presets:
        raise ValueError(f"unknown preset {name!r}; known: {sorted(presets)}")
    return {**presets[name], **overrides}


__all__ = ["FUSED", "GEOMETRY", "LEXICAL_ONLY", "PRODUCTION", "preset"]
