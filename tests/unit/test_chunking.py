"""Chunking. The invariant that matters is that no content is ever lost."""

from __future__ import annotations

import pytest

from supermemory.domain.chunking import (
    chunk_text,
    estimate_tokens,
    normalize,
)


def _dense(text: str) -> str:
    return "".join(text.split())


@pytest.mark.parametrize("raw", ["", "   ", "\n\n\t  \n"])
def test_blank_input_yields_no_chunks(raw: str) -> None:
    assert chunk_text(raw) == []


def test_single_character() -> None:
    chunks = chunk_text("a")
    assert len(chunks) == 1
    assert chunks[0].text == "a"
    assert chunks[0].ordinal == 0


def test_normalize_strips_nul_bytes() -> None:
    """PostgreSQL rejects NUL in text columns; it must never reach the driver."""
    assert "\x00" not in normalize("before\x00after")
    assert normalize("before\x00after") == "beforeafter"


def test_normalize_unifies_line_endings() -> None:
    assert normalize("a\r\nb\rc") == "a\nb\nc"


def test_normalize_keeps_tabs_and_newlines() -> None:
    assert normalize("a\tb\nc") == "a\tb\nc"


@pytest.mark.parametrize(
    "text",
    [
        "x" * 5000,  # no whitespace at all
        "这是一个测试。" * 300,  # CJK, no spaces
        "The quick brown fox jumps over the dog. " * 200,
        "para one.\n\npara two.\n\n" * 60,
        "word " * 2000,
    ],
)
def test_chunking_never_loses_content(text: str) -> None:
    chunks = chunk_text(text, target_tokens=32, overlap_tokens=8)
    assert chunks, "expected at least one chunk"
    combined = _dense(" ".join(c.text for c in chunks))
    # Overlap means combined may be longer, never shorter.
    assert len(combined) >= len(_dense(normalize(text)))


def test_chunk_ordinals_are_contiguous() -> None:
    chunks = chunk_text("word " * 500, target_tokens=16, overlap_tokens=4)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_overlap_of_zero_produces_no_repetition() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta " * 20
    chunks = chunk_text(text, target_tokens=16, overlap_tokens=0)
    combined = _dense(" ".join(c.text for c in chunks))
    assert len(combined) == len(_dense(normalize(text)))


def test_single_token_longer_than_budget_still_terminates() -> None:
    """The obvious implementation loops forever here."""
    chunks = chunk_text("y" * 10_000, target_tokens=8, overlap_tokens=2)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def test_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValueError, match="smaller than target"):
        chunk_text("hello world", target_tokens=10, overlap_tokens=10)


@pytest.mark.parametrize("target,overlap", [(0, 0), (-1, 0)])
def test_invalid_target_rejected(target: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", target_tokens=target, overlap_tokens=overlap)


def test_negative_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        chunk_text("hello", target_tokens=10, overlap_tokens=-1)


def test_emoji_and_combining_characters_survive() -> None:
    text = "family 👨‍👩‍👧‍👦 and flag 🏳️‍🌈 and accents éàü"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "👨‍👩‍👧‍👦" in chunks[0].text


def test_estimate_tokens_never_zero_for_nonempty() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_large_document_is_fast() -> None:
    import time

    text = "lorem ipsum dolor sit amet. " * 20_000
    started = time.perf_counter()
    chunks = chunk_text(text, target_tokens=320, overlap_tokens=48)
    assert time.perf_counter() - started < 2.0
    assert len(chunks) > 100


def test_chunking_is_deterministic() -> None:
    text = "Some prose. " * 100
    a = chunk_text(text, target_tokens=32, overlap_tokens=8)
    b = chunk_text(text, target_tokens=32, overlap_tokens=8)
    assert [c.text for c in a] == [c.text for c in b]
