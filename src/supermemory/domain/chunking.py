"""Structure-aware chunking.

Naive fixed-width chunking cuts sentences in half, which costs retrieval quality
at exactly the boundary where a fact lives. This splits on the strongest
available boundary that still fits the budget — paragraph, then sentence, then
word, then hard character split as a last resort for input with no whitespace at
all (minified JSON, CJK text, a base64 blob).

Termination is guaranteed: every recursion either reduces the unit size or falls
through to a hard split, and `_hard_split` always makes progress. That matters
more than it sounds — the obvious implementation loops forever on a single token
longer than the chunk budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Rough characters-per-token for English prose. Chunk budgets are expressed in
#: tokens because that is what model context is priced in, but we avoid a
#: tokenizer dependency; the estimate is deliberately conservative.
CHARS_PER_TOKEN = 4

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|(?<=[。！？])\s*")
_WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate. Never returns 0 for non-empty input."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    ordinal: int
    start: int
    end: int
    token_estimate: int


def normalize(text: str) -> str:
    """Strip control characters and normalise line endings.

    NUL bytes in particular must go: PostgreSQL rejects them in text columns,
    so a single \\x00 in an upload would otherwise fail at the storage layer with
    an unhelpful driver error rather than at validation with a clear one.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32
    ).strip()


def _hard_split(text: str, budget_chars: int) -> list[str]:
    """Last resort: fixed-width slices. Always makes progress."""
    step = max(1, budget_chars)
    return [text[i : i + step] for i in range(0, len(text), step)]


def _split_units(text: str, budget_chars: int) -> list[str]:
    """Break text into the largest units that fit, escalating aggressiveness."""
    if len(text) <= budget_chars:
        return [text]

    for pattern in (_PARAGRAPH_RE, _SENTENCE_RE, _WHITESPACE_RE):
        parts = [p for p in pattern.split(text) if p and p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            for part in parts:
                # A single unit can still exceed the budget (one enormous
                # sentence); recurse on it with the next-weaker boundary.
                out.extend(
                    [part] if len(part) <= budget_chars
                    else _split_units(part, budget_chars)
                )
            return out
    return _hard_split(text, budget_chars)


def chunk_text(
    text: str,
    *,
    target_tokens: int = 320,
    overlap_tokens: int = 48,
) -> list[TextChunk]:
    """Split `text` into overlapping chunks near `target_tokens` each.

    Overlap is applied by carrying trailing units of one chunk into the next, so
    a fact that straddles a boundary appears whole in at least one chunk.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    cleaned = normalize(text)
    if not cleaned:
        return []

    budget = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    units = _split_units(cleaned, budget)
    if not units:
        return []

    # Index-based packing. Every unit lands in at least one chunk, and the start
    # index strictly increases each iteration, so this is lossless and always
    # terminates. An earlier version deduplicated chunk text and silently
    # dropped content whenever the source legitimately repeated itself --
    # 9000 characters of repeated prose collapsed to a single chunk.
    chunks: list[TextChunk] = []
    cursor = 0  # character offset for provenance
    i = 0
    n = len(units)

    while i < n:
        size = 0
        j = i
        while j < n and (size + len(units[j]) + 1) <= budget:
            size += len(units[j]) + 1
            j += 1
        if j == i:
            # One unit alone exceeds the budget. _split_units should prevent
            # this, but emitting it whole beats looping forever.
            j = i + 1

        body = " ".join(units[i:j]).strip()
        if body:
            start = cleaned.find(units[i], cursor)
            if start < 0:
                start = cursor
            chunks.append(
                TextChunk(
                    text=body,
                    ordinal=len(chunks),
                    start=start,
                    end=start + len(body),
                    token_estimate=estimate_tokens(body),
                )
            )
            cursor = start + max(len(units[i]), 1)

        if j >= n:
            break

        # Step back over trailing units to create the overlap, but never so far
        # that the next chunk starts at or before this one.
        back = 0
        carried = 0
        while back < (j - i - 1) and carried < overlap_chars:
            carried += len(units[j - 1 - back]) + 1
            back += 1
        i = max(i + 1, j - back)

    return chunks


__all__ = ["CHARS_PER_TOKEN", "TextChunk", "chunk_text", "estimate_tokens", "normalize"]
