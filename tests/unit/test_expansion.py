"""Query expansion: a feature that could not run, and the blend that carries it.

`expansion.py` had no tests and `HydeExpander` was never constructed anywhere in
the tree, so `use_expansion=True` was a no-op for every caller who found the
flag. The reason was structural rather than an oversight: `HydeExpander` builds
its own `google.genai` client, and `MemoryService` is deliberately vendor-free,
so the service could not wire it and nothing else tried. `CompletionExpander` --
HyDE over an injected `CompleteFn`, like every other model boundary here -- is
what closed that, and most of this file is about it.

The other half is `_mean_unit_vector`, which is where the expansion actually
takes effect. It averages the hypothetical passage's vector with the real query's
rather than replacing it, and the reason is bounded damage: a hypothesis that
goes off-topic would otherwise drag retrieval with it, and averaging limits the
harm to half the signal while keeping the search anchored to what was asked. That
argument is only true if the blend behaves, so it is tested directly -- including
the branch where two vectors cancel exactly, which is unreachable through any
realistic embedder and is precisely why nobody would find it by accident.

Every failure path degrades to the plain query. An enhancement that can fail a
search is a liability, so each of those paths gets its own test.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.support.vectors import ANCHOR, at_cosine, axis, cosine_of

from mapi.domain.retrieval.expansion import (
    CompletionExpander,
    NoopExpander,
)
from mapi.domain.retrieval.pipeline import _mean_unit_vector


def _replying(text: str):
    async def complete(prompt: str) -> str:
        return text

    return complete


# -- the noop, which is what the product used to always get -----------------


async def test_the_noop_expander_expands_nothing() -> None:
    assert await NoopExpander().expand("anything") == []
    assert NoopExpander().name == "none"


# -- CompletionExpander ----------------------------------------------------


async def test_a_passage_is_returned_and_whitespace_collapsed() -> None:
    """The passage is embedded, and an embedder is sensitive to the newlines a
    model likes to emit around prose."""
    expander = CompletionExpander(_replying("  I usually\n\nfly with an aisle seat.  "))
    assert await expander.expand("where do I like to sit") == [
        "I usually fly with an aisle seat."
    ]


async def test_the_prompt_asks_for_a_statement_not_an_answer() -> None:
    """HyDE works because the hypothesis is written in the register of the
    CORPUS, not the register of a question. If the prompt stopped asking for a
    first-person statement the vector would land in the wrong neighbourhood and
    the feature would quietly stop helping."""
    seen: dict[str, str] = {}

    async def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return "a passage"

    await CompletionExpander(complete).expand("what do I read on the commute")
    assert "first person" in seen["prompt"]
    assert "what do I read on the commute" in seen["prompt"]


async def test_a_long_passage_is_clipped() -> None:
    expander = CompletionExpander(_replying("word " * 1000), max_chars=100)
    passages = await expander.expand("q")
    assert len(passages[0]) <= 100


async def test_a_blank_query_never_calls_the_model() -> None:
    """There is nothing to hypothesise about, and the call costs money."""
    called = False

    async def complete(prompt: str) -> str:
        nonlocal called
        called = True
        return "x"

    assert await CompletionExpander(complete).expand("   ") == []
    assert not called


async def test_an_empty_completion_yields_no_passage() -> None:
    """Never an empty string.

    The embedder REJECTS blank input, so returning `[""]` would raise inside the
    embedding call and fail the whole search -- turning a skipped enhancement into
    an outage.
    """
    assert await CompletionExpander(_replying("")).expand("q") == []
    assert await CompletionExpander(_replying("   \n  ")).expand("q") == []


async def test_a_vendor_failure_degrades_to_no_expansion() -> None:
    async def broken(prompt: str) -> str:
        raise RuntimeError("vendor is down")

    assert await CompletionExpander(broken).expand("q") == []


async def test_a_timeout_degrades_to_no_expansion_and_does_not_block() -> None:
    """A slow expander must not hold a search open for its own timeout."""

    async def slow(prompt: str) -> str:
        await asyncio.sleep(10)
        return "too late"

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await CompletionExpander(slow, timeout_s=0.01).expand("q") == []
    assert loop.time() - started < 1.0


def test_the_completion_expander_reports_the_same_name_as_hyde() -> None:
    """`strategies` and explain traces read the name, and the mechanism is the
    same -- a caller should not have to know which class produced the passage."""
    assert CompletionExpander(_replying("x")).name == "hyde"


def test_the_vendor_coupled_expander_is_still_available_for_the_harness() -> None:
    """`HydeExpander` was not deleted. It owns its own client, which is right for
    the benchmark harness and wrong for the service."""
    from mapi.domain.retrieval.expansion import HydeExpander

    assert HydeExpander is not None
    fake = object()
    assert HydeExpander(client=fake).name == "hyde"


# -- the blend, which is where expansion takes effect ----------------------


def test_a_single_vector_is_returned_unchanged() -> None:
    """No expansion happened, so nothing should move."""
    query = at_cosine(0.9, off=3)
    assert _mean_unit_vector([query]) == query


def test_the_blend_lands_between_its_inputs() -> None:
    """The stated guarantee: the query keeps half the signal.

    Averaging the anchor with something orthogonal to it must land at cosine
    ~0.707 from the anchor -- closer than the passage, further than the query.
    That is the "bounded damage" claim, as a number.
    """
    blended = _mean_unit_vector([axis(ANCHOR), axis(5)])
    assert cosine_of(blended, axis(ANCHOR)) == pytest.approx(0.7071, abs=1e-4)
    assert cosine_of(blended, axis(5)) == pytest.approx(0.7071, abs=1e-4)


def test_the_blend_is_a_unit_vector() -> None:
    """Renormalized so cosine similarity reduces to a dot product downstream."""
    import math

    blended = _mean_unit_vector([axis(ANCHOR), at_cosine(0.3, off=2), at_cosine(0.6, off=4)])
    assert math.sqrt(sum(x * x for x in blended)) == pytest.approx(1.0, abs=1e-9)


def test_an_off_topic_passage_moves_the_query_but_does_not_replace_it() -> None:
    """The reason for averaging rather than substituting.

    A hypothesis that lands nowhere near the question still leaves the blend
    closer to the question than to nothing -- the damage is bounded at half the
    signal instead of being total.
    """
    query = axis(ANCHOR)
    off_topic = axis(40)
    blended = _mean_unit_vector([query, off_topic])
    assert cosine_of(blended, query) > 0.7


def test_diametrically_opposed_vectors_keep_the_query() -> None:
    """The unreachable branch, which is exactly why it needs a test.

    Two exactly opposed unit vectors average to zero, and a zero vector has no
    direction -- cosine similarity against it is undefined and the embedder
    contract rejects it. The code keeps the query, which is the component it
    trusts. No realistic embedder produces this, so nobody would ever hit it by
    accident and it would fail as a division by a zero norm.
    """
    query = axis(ANCHOR)
    opposed = [-x for x in query]
    assert _mean_unit_vector([query, opposed]) == query


def test_empty_vectors_are_ignored() -> None:
    query = at_cosine(0.8, off=1)
    assert _mean_unit_vector([query, []]) == query


def test_no_usable_vectors_raises() -> None:
    """Distinct from "no expansion": an empty list here means the caller lost
    the query too, which is a bug rather than a degraded path."""
    with pytest.raises(ValueError, match="no vectors"):
        _mean_unit_vector([])
    with pytest.raises(ValueError, match="no vectors"):
        _mean_unit_vector([[], []])


def test_mixed_dimensions_raise_rather_than_producing_a_garbage_vector() -> None:
    """A dimension mismatch means two different embedding models, and silently
    averaging the overlap would produce a vector that means nothing while
    looking valid."""
    with pytest.raises(ValueError, match="differing dimensions"):
        _mean_unit_vector([axis(ANCHOR), [0.1, 0.2, 0.3]])
