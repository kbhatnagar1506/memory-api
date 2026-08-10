"""Query understanding: a model on the read path that can never break it.

The tests are almost entirely about the failure modes. Whether the model
labels a question well is the model's problem; whether a bad vendor day can
slow down, break, or hang a search is ours.
"""

from __future__ import annotations

import asyncio

import pytest

from mapi.domain.synthesis.classify import QuestionKind
from mapi.domain.synthesis.understand import (
    QueryUnderstanding,
    parse_label,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("list_all", QuestionKind.LIST_ALL),
        ("  direct\n", QuestionKind.DIRECT),
        ("Label: count", QuestionKind.COUNT),
        ("`order`", QuestionKind.ORDER),
        ("advice.", QuestionKind.ADVICE),
        ("DATE_ARITH", QuestionKind.DATE_ARITH),
        ("**compare**", QuestionKind.COMPARE),
    ],
)
def test_parse_label_survives_the_wrapping_models_add(
    raw: str, expected: QuestionKind
) -> None:
    assert parse_label(raw) is expected


@pytest.mark.parametrize("raw", ["", "I think this is asking about your dog", "unknown"])
def test_parse_label_refuses_to_guess(raw: str) -> None:
    """An unreadable reply must fall through, not land on `direct` by accident."""
    assert parse_label(raw) is None


async def test_no_backend_uses_the_regex() -> None:
    intent = await QueryUnderstanding(None).intent("list all the databases we use")
    assert intent.kind is QuestionKind.LIST_ALL
    assert intent.source == "rules"


async def test_the_model_decides_when_it_answers() -> None:
    async def complete(prompt: str) -> str:
        return "list_all"

    intent = await QueryUnderstanding(complete).intent("walk me through the setup")
    assert intent.kind is QuestionKind.LIST_ALL
    assert intent.source == "llm"


async def test_a_timeout_falls_back_and_does_not_hang_the_search() -> None:
    async def slow(prompt: str) -> str:
        await asyncio.sleep(10)
        return "list_all"

    understanding = QueryUnderstanding(slow, timeout_s=0.01)
    loop = asyncio.get_running_loop()
    started = loop.time()
    intent = await understanding.intent("how many times did I visit")
    elapsed = loop.time() - started

    assert intent.kind is QuestionKind.COUNT, "regex still classified it"
    assert intent.source == "rules"
    assert elapsed < 1.0, "the timeout bounds the search, not the vendor"


async def test_a_vendor_error_falls_back() -> None:
    async def broken(prompt: str) -> str:
        raise RuntimeError("429 resource exhausted")

    intent = await QueryUnderstanding(broken).intent("what was the last thing I ordered")
    assert intent.kind is QuestionKind.ORDER
    assert intent.source == "rules"


async def test_prose_instead_of_a_label_falls_back() -> None:
    async def chatty(prompt: str) -> str:
        return "Sure! That question is asking about your travel history."

    intent = await QueryUnderstanding(chatty).intent("how many trips did I take")
    assert intent.source == "rules"
    assert intent.kind is QuestionKind.COUNT


async def test_repeat_questions_do_not_call_the_model_again() -> None:
    calls = 0

    async def complete(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "list_all"

    understanding = QueryUnderstanding(complete)
    first = await understanding.intent("What is our entire infrastructure?")
    second = await understanding.intent("what is our ENTIRE infrastructure?")

    assert first.source == "llm"
    assert second.source == "cache", "case and spacing must not miss the cache"
    assert second.kind is QuestionKind.LIST_ALL
    assert calls == 1


async def test_the_cache_is_bounded() -> None:
    async def complete(prompt: str) -> str:
        return "direct"

    understanding = QueryUnderstanding(complete, cache_size=4)
    for i in range(20):
        await understanding.intent(f"question number {i}")
    assert len(understanding._cache) == 4


async def test_a_broken_vendor_stops_being_called() -> None:
    """One outage should cost a handful of timeouts, not one per search."""
    calls = 0

    async def broken(prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("upstream is down")

    understanding = QueryUnderstanding(broken)
    for i in range(25):
        intent = await understanding.intent(f"unique question {i}")
        assert intent.source == "rules"

    assert calls <= 3, f"circuit breaker did not trip; called {calls} times"


async def test_the_breaker_reopens_after_the_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    healthy = False

    async def flaky(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if not healthy:
            raise RuntimeError("down")
        return "count"

    understanding = QueryUnderstanding(flaky)
    for i in range(10):
        await understanding.intent(f"q{i}")
    assert calls == 3

    healthy = True
    # Jump past the cooldown rather than sleeping through it.
    understanding._blocked_until = 0.0
    intent = await understanding.intent("a fresh question entirely")
    assert intent.source == "llm"
    assert intent.kind is QuestionKind.COUNT


async def test_understanding_never_raises_whatever_the_backend_does() -> None:
    """The read path degrades. It does not fail."""

    class Exploding:
        async def __call__(self, prompt: str) -> str:
            raise BaseExceptionGroup("chaos", [ValueError("a"), OSError("b")])

    intent = await QueryUnderstanding(Exploding()).intent("what did I name my dog")
    assert intent.kind is QuestionKind.DIRECT
    assert intent.source == "rules"
