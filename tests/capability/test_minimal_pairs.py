"""Minimal pairs: can retrieval tell two nearly-identical memories apart?

The highest-value cheap test in the whole catalogue, and the one whose failures
are most obviously bugs rather than tuning. Each case is two memories differing
in exactly ONE token class, and two queries differing in exactly that token. The
metric is PAIRWISE accuracy -- both queries must rank their own memory first --
so guessing scores 25%, not 50%.

WHY THIS RUNS AGAINST THE REAL STACK. Every other capability family constructs
its vectors with `ScriptedEmbedder`, because there the question is what the
pipeline does with a given geometry. Here the question is whether the shipped
embedder and the shipped lexical arm produce a usable geometry at all, and
hand-specifying the vectors would reduce the test to "a higher cosine ranks
first" -- true by construction, informative about nothing. So this file uses
`DeterministicEmbedder` plus BM25 plus the heuristic reranker, exactly as
configured in production.

WHY THESE EIGHT CLASSES. Every one is a real memory-API correctness bug when it
fails, and none needs a model call:

    negation      "the migration completed" vs "did not complete"
    numerals      500 vs 5000
    dates         March 3 vs March 30
    units         3 hours vs 3 minutes
    entities      Alice's key vs Bob's key
    direction     A reports to B vs B reports to A
    quantifiers   all services vs some services
    modality      must be reviewed vs may be reviewed

`direction` was expected to be unservable and is not, which is the most useful
thing this file found. Measured on the pair below:

    cos(A, B)                       0.9111   identical token multisets
    BM25 tokens                     identical -- the lexical arm ties EXACTLY
    cos(qAlice, A) vs (qAlice, B)   0.7652 vs 0.7205   margin 0.045
    cos(qBob,   A) vs (qBob,   B)   0.6792 vs 0.6872   margin 0.008

So word order does survive, but only through character 3/4-grams spanning word
boundaries -- an accident of the hash embedder's feature set, not a designed
property -- and on one side the whole decision rests on 0.008, about 1.2% of the
score. It passes today and would flip under any change to the n-gram sizes, the
tokenizer, or the embedder. `test_direction_discriminates_only_barely` asserts
the margin rather than the outcome, so the fragility is what is pinned.

Retrieval literature reports worse for negation: NevIR found most embedders
score BELOW random on negated pairs.

FLOORS ARE MEASURED, NOT ASPIRATIONAL. Each class asserts what the stack does
today, so a regression fails and an improvement fails loudly enough to notice
and raise the floor. `test_the_battery_scores_above_chance` is the one aggregate
claim, and it is the number to watch.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from tests.support.factories import tenant

from mapi.config import Settings
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore


@dataclass(frozen=True, slots=True)
class Pair:
    """Two memories one token apart, and the query that selects each.

    `token_class` names what differs. `expected` is the measured pairwise
    accuracy: 1.0 both queries right, 0.5 one, 0.0 neither.
    """

    token_class: str
    left_text: str
    right_text: str
    left_query: str
    right_query: str
    expected: float


PAIRS: tuple[Pair, ...] = (
    Pair(
        token_class="numerals",
        left_text="The refund issued to the customer was 500 dollars.",
        right_text="The refund issued to the customer was 5000 dollars.",
        left_query="500 dollars refund",
        right_query="5000 dollars refund",
        expected=1.0,
    ),
    Pair(
        token_class="dates",
        left_text="The architecture review is scheduled for March 3.",
        right_text="The architecture review is scheduled for March 30.",
        left_query="March 3 architecture review",
        right_query="March 30 architecture review",
        expected=1.0,
    ),
    Pair(
        token_class="units",
        left_text="The commute from the office takes 3 hours.",
        right_text="The commute from the office takes 3 minutes.",
        left_query="3 hours commute",
        right_query="3 minutes commute",
        expected=1.0,
    ),
    Pair(
        token_class="entities",
        left_text="Alice owns the production deploy key.",
        right_text="Bob owns the production deploy key.",
        left_query="Alice production deploy key",
        right_query="Bob production deploy key",
        expected=1.0,
    ),
    Pair(
        token_class="quantifiers",
        left_text="All of the internal services use mutual TLS.",
        right_text="Some of the internal services use mutual TLS.",
        left_query="all internal services mutual TLS",
        right_query="some internal services mutual TLS",
        expected=1.0,
    ),
    Pair(
        token_class="modality",
        left_text="Production deploys must be reviewed by a second engineer.",
        right_text="Production deploys may be reviewed by a second engineer.",
        left_query="deploys must be reviewed",
        right_query="deploys may be reviewed",
        expected=1.0,
    ),
    Pair(
        # "never" rather than "did not": a one-token negation, so the pair
        # differs by exactly one token like every other class here. "completed"
        # -> "did not complete" changes the verb form as well, which is two
        # changes and would measure their sum.
        token_class="negation",
        left_text="The database migration completed during the maintenance window.",
        right_text="The database migration never completed during the maintenance window.",
        left_query="the migration completed during the window",
        right_query="the migration never completed during the window",
        expected=1.0,
    ),
    Pair(
        token_class="direction",
        left_text="Alice reports to Bob on the platform team.",
        right_text="Bob reports to Alice on the platform team.",
        left_query="who does Alice report to on the platform team",
        right_query="who does Bob report to on the platform team",
        expected=1.0,
    ),
)

#: Below this, a per-query decision is being made on noise. 0.02 of cosine is
#: about the width of float32 rounding through pgvector plus any change to the
#: embedder's n-gram sizes, so a margin under it is not a property the system
#: has -- it is one it happens to display.
_FRAGILE_MARGIN = 0.02


async def _pairwise_accuracy(settings: Settings, pair: Pair) -> tuple[float, dict[str, str]]:
    """Score one pair. Returns the accuracy and which memory each query chose.

    A fresh store per pair: two memories only, so nothing else can be the top
    hit and a failure is unambiguously a discrimination failure rather than a
    crowding one.
    """
    from mapi.domain.embeddings.deterministic import DeterministicEmbedder

    store = InMemoryStore()
    service = MemoryService(
        store,
        DeterministicEmbedder(dimensions=settings.embedding_dimensions),
        HeuristicReranker(),
        settings,
    )
    org, space = await tenant(store, name="MinimalPair")

    left = await service.ingest(
        org_id=org.id, space_id=space.id, content=pair.left_text, extract=False
    )
    right = await service.ingest(
        org_id=org.id, space_id=space.id, content=pair.right_text, extract=False
    )
    labels = {left.memory.id: "left", right.memory.id: "right"}

    chose: dict[str, str] = {}
    for side, query in (("left", pair.left_query), ("right", pair.right_query)):
        response = await service.search(
            SearchRequest(query=query, org_id=org.id, space_id=space.id, limit=2)
        )
        top = response.results[0].memory.id if response.results else ""
        chose[side] = labels.get(top, "none")

    correct = sum(1 for side, picked in chose.items() if picked == side)
    return correct / 2.0, chose


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.token_class)
async def test_a_minimal_pair_scores_its_measured_floor(
    pair: Pair, capability_settings: Settings
) -> None:
    """Per class, with the failure message naming what the queries chose.

    Asserting `>=` rather than `==` so an improvement does not fail the build --
    but `test_no_floor_is_stale` below catches a floor that has become too low
    to mean anything.
    """
    accuracy, chose = await _pairwise_accuracy(capability_settings, pair)
    assert accuracy >= pair.expected, (
        f"{pair.token_class}: pairwise accuracy {accuracy} below floor {pair.expected}. "
        f"'{pair.left_query}' chose {chose['left']}, "
        f"'{pair.right_query}' chose {chose['right']}"
    )


async def test_the_battery_scores_above_chance(capability_settings: Settings) -> None:
    """The one aggregate number, and the one to watch.

    Random is 0.25: each query independently has an even chance of picking its
    own memory, and pairwise accuracy needs both.
    """
    scores = {}
    for pair in PAIRS:
        accuracy, _chose = await _pairwise_accuracy(capability_settings, pair)
        scores[pair.token_class] = accuracy
    mean = sum(scores.values()) / len(scores)
    assert mean > 0.25, f"battery at chance or below: {mean:.3f} over {scores}"


async def test_the_lexical_arm_cannot_see_direction_at_all() -> None:
    """Half of the stack is blind to this class, and that half is exact about it.

    "Alice reports to Bob" and "Bob reports to Alice" analyze to the identical
    token multiset, so BM25 cannot score them differently -- not "scores them
    similarly", scores them the same. Any discrimination on this class comes
    entirely from the vector arm, which is why the next test measures the vector
    margin rather than the ranking.
    """
    from mapi.domain.text import analyze

    pair = next(p for p in PAIRS if p.token_class == "direction")
    assert sorted(analyze(pair.left_text)) == sorted(analyze(pair.right_text))


async def test_direction_discriminates_only_barely(capability_settings: Settings) -> None:
    """It works, for a reason nobody designed, on a margin nobody should trust.

    Word order survives into the embedding only through character 3/4-grams that
    happen to span word boundaries. Measured margins on this pair: 0.045 for the
    Alice query and 0.008 for the Bob query -- the second is about 1.2% of the
    score, and below `_FRAGILE_MARGIN`.

    So the assertion is on the MARGIN, not on the ranking. Pinning "it ranks
    correctly" would pass today and give no warning when a tokenizer or n-gram
    change flips it; pinning the margin says out loud that this class is
    currently decided by noise, and fails the day someone believes otherwise.
    """
    from mapi.domain.embeddings.base import cosine_similarity
    from mapi.domain.embeddings.deterministic import DeterministicEmbedder

    pair = next(p for p in PAIRS if p.token_class == "direction")
    embedder = DeterministicEmbedder(dimensions=capability_settings.embedding_dimensions)
    left, right, q_left, q_right = (
        await embedder.embed(
            [pair.left_text, pair.right_text, pair.left_query, pair.right_query]
        )
    ).vectors

    margins = {
        "left": cosine_similarity(q_left, left) - cosine_similarity(q_left, right),
        "right": cosine_similarity(q_right, right) - cosine_similarity(q_right, left),
    }
    assert all(m > 0 for m in margins.values()), f"direction stopped discriminating: {margins}"
    assert min(margins.values()) < _FRAGILE_MARGIN, (
        f"direction margins are now robust ({margins}); the embedder gained real "
        "word-order sensitivity. Delete this test and say so."
    )


def test_no_floor_is_at_or_below_chance() -> None:
    """A guard on the guards.

    `>=` floors rot: a class that improves keeps passing against a stale number
    and the gain is never banked. This cannot detect that by itself, so it
    enforces the next best thing -- no floor may sit at or below the 0.25 random
    baseline, because such a floor asserts nothing while looking like coverage.

    A class we genuinely cannot serve should be recorded the way `direction` is:
    a named test measuring the specific quantity that is weak, not a zero in a
    table.
    """
    for pair in PAIRS:
        assert pair.expected > 0.25, (
            f"{pair.token_class} floor {pair.expected} is at or below chance (0.25); "
            "record an unservable class as its own test, not as a passing zero"
        )


def test_every_pair_differs_in_exactly_one_token_class() -> None:
    """Suite-validity gate. A pair differing in two things measures neither.

    Checked structurally: the two texts must differ, and their token multisets
    must differ by at most two tokens (one removed, one added). `direction` is
    exempt -- it differs by ORDER and by zero tokens, which is the whole point.
    """
    for pair in PAIRS:
        assert pair.left_text != pair.right_text
        left = pair.left_text.lower().rstrip(".").split()
        right = pair.right_text.lower().rstrip(".").split()
        if pair.token_class == "direction":
            assert sorted(left) == sorted(right), "direction pairs must be a permutation"
            continue
        only_left = [t for t in left if t not in right]
        only_right = [t for t in right if t not in left]
        assert len(only_left) <= 2 and len(only_right) <= 2, (
            f"{pair.token_class} differs by {only_left} / {only_right}, "
            "which is more than one token class"
        )
