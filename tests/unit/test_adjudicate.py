"""Contradiction adjudication: recall past the seven-string negation list.

The lexical pass finds a conflict when two similar memories differ by a
negation from a fixed list, an antonym from a fixed list, or a figure. These
tests are about the cases those lists cannot express, and about the one
property that makes putting a model on the write path safe here: it fails
CLOSED. A broken vendor proposes nothing, which is what this feature did
before a vendor existed.
"""

from __future__ import annotations

import pytest

from mapi.core.ids import new_id
from mapi.domain.consolidation import unexplained_pairs
from mapi.domain.models import Memory, MemoryStatus
from mapi.domain.synthesis.adjudicate import (
    MIN_CONFIDENCE,
    adjudicate_contradictions,
    format_candidates,
    parse_verdicts,
)

CANDIDATES = [("mem_a", "I eat meat"), ("mem_b", "I live in Berlin")]


def test_candidates_are_numbered_not_keyed_by_id() -> None:
    """A transposed character in a ULID would attach a conflict to the wrong row."""
    rendered = format_candidates(CANDIDATES)
    assert rendered == "1. I eat meat\n2. I live in Berlin"
    assert "mem_a" not in rendered


def test_parse_maps_indexes_back_to_ids() -> None:
    verdicts = parse_verdicts(
        '[{"n": 1, "reason": "vegetarian now", "confidence": 0.9}]', CANDIDATES
    )
    assert [v.memory_id for v in verdicts] == ["mem_a"]
    assert verdicts[0].reason == "vegetarian now"


def test_parse_survives_a_code_fence() -> None:
    raw = '```json\n[{"n": 2, "reason": "moved", "confidence": 0.95}]\n```'
    assert [v.memory_id for v in parse_verdicts(raw, CANDIDATES)] == ["mem_b"]


def test_parse_salvages_a_truncated_array() -> None:
    """A cut-off reply must not read as "nothing conflicts"."""
    raw = '[{"n": 1, "reason": "stopped eating meat", "confidence": 0.88}, {"n": 2, "rea'
    assert [v.memory_id for v in parse_verdicts(raw, CANDIDATES)] == ["mem_a"]


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "",
        "no conflicts found",
        '[{"n": 99, "confidence": 0.99}]',  # out of range
        '[{"n": 1}]',  # no confidence
        '[{"n": "one", "confidence": 0.9}]',  # not a number
        f'[{{"n": 1, "confidence": {MIN_CONFIDENCE - 0.01}}}]',  # under the floor
    ],
)
def test_doubtful_output_yields_no_conflict(raw: str) -> None:
    assert parse_verdicts(raw, CANDIDATES) == []


def test_a_duplicate_index_is_counted_once() -> None:
    raw = '[{"n": 1, "confidence": 0.9}, {"n": 1, "confidence": 0.95}]'
    assert len(parse_verdicts(raw, CANDIDATES)) == 1


async def test_it_finds_what_the_word_list_cannot() -> None:
    seen: dict[str, str] = {}

    async def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return '[{"n": 1, "reason": "no longer eats meat", "confidence": 0.92}]'

    verdicts = await adjudicate_contradictions(
        "I no longer eat meat", CANDIDATES, complete
    )
    assert [v.memory_id for v in verdicts] == ["mem_a"]
    assert "I no longer eat meat" in seen["prompt"]


async def test_a_vendor_failure_proposes_nothing() -> None:
    """Fails CLOSED. A false contradiction costs more than a missed one."""

    async def broken(prompt: str) -> str:
        raise RuntimeError("upstream is down")

    assert await adjudicate_contradictions("anything", CANDIDATES, broken) == []


async def test_a_timeout_proposes_nothing_and_does_not_block() -> None:
    import asyncio

    async def slow(prompt: str) -> str:
        await asyncio.sleep(10)
        return '[{"n": 1, "confidence": 0.99}]'

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await adjudicate_contradictions("x", CANDIDATES, slow, timeout_s=0.01) == []
    assert loop.time() - started < 1.0


async def test_the_candidate_list_is_capped() -> None:
    sent: dict[str, str] = {}

    async def complete(prompt: str) -> str:
        sent["prompt"] = prompt
        return "[]"

    many = [(f"mem_{i}", f"statement number {i}") for i in range(50)]
    await adjudicate_contradictions("x", many, complete, max_candidates=3)
    assert "3. statement number 2" in sent["prompt"]
    assert "4. statement number 3" not in sent["prompt"]


async def test_no_candidates_makes_no_call() -> None:
    called = False

    async def complete(prompt: str) -> str:
        nonlocal called
        called = True
        return "[]"

    assert await adjudicate_contradictions("x", [], complete) == []
    assert not called


# -- the seam: which pairs even reach the model ---------------------------


def _memory(content: str) -> Memory:
    return Memory(org_id=new_id("org"), space_id=new_id("space"), content=content)


def test_pairs_the_lexical_pass_already_explained_are_not_re_asked() -> None:
    """No point paying a model to confirm what a regex already found."""
    new = _memory("I do not eat meat")
    old = _memory("I eat meat")
    vector = [1.0, 0.0, 0.0]
    # Near-identical embeddings put the pair inside the contradiction band.
    close = [0.92, 0.39, 0.0]

    remaining = unexplained_pairs(new, vector, [(old, close)])
    assert remaining == [], "the negation list caught this one already"


def test_a_conflict_outside_the_word_list_is_handed_over() -> None:
    new = _memory("I gave up eating meat")
    old = _memory("I eat meat")
    vector = [1.0, 0.0, 0.0]
    close = [0.92, 0.39, 0.0]

    remaining = unexplained_pairs(new, vector, [(old, close)])
    assert [m.id for m, _ in remaining] == [old.id]


def test_distant_memories_never_reach_the_model() -> None:
    """The similarity gate is what keeps this affordable."""
    new = _memory("I gave up eating meat")
    unrelated = _memory("The API runs on Python")
    assert unexplained_pairs(new, [1.0, 0.0, 0.0], [(unrelated, [0.0, 1.0, 0.0])]) == []


def test_superseded_memories_are_not_adjudicated() -> None:
    new = _memory("I gave up eating meat")
    old = _memory("I eat meat")
    old.status = MemoryStatus.SUPERSEDED
    assert unexplained_pairs(new, [1.0, 0.0, 0.0], [(old, [0.92, 0.39, 0.0])]) == []
