"""Structure-aware chunking.

Naive fixed-width chunking cuts sentences in half, which costs retrieval quality
at exactly the boundary where a fact lives. This splits on the strongest
available boundary that still fits the budget — paragraph, then sentence, then
word, then hard character split as a last resort for input with no whitespace at
all (minified JSON, CJK text, a base64 blob).

Code and JSON are not prose, and a sentence splitter cuts them mid-structure: a
function's guards in one chunk and its write in another, each embedding as a
fragment of neither. `chunk_content` sizes their chunks to the structure
instead (`chunk_structured`):

    whole      content that fits the structured budget is ONE chunk: one function,
               one vector, however long it is up to that budget
    units      past it, the cuts fall only between structural units -- the members
               of a JSON object or array (recursing into a member still too big),
               top-level definitions in code (decorators kept with them), fenced
               code blocks -- and units pack into chunks as large as fit
    context    the content's leading description (and, inside nested JSON, the
               member's path) is embedded with every later chunk, so each piece says
               whose piece it is; a stored chunk is always an exact span of the
               caller's text

Termination is guaranteed: every recursion either reduces the unit size or falls
through to a hard split, and `_hard_split` always makes progress. That matters
more than it sounds — the obvious implementation loops forever on a single token
longer than the chunk budget.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

#: Rough characters-per-token for English prose. Chunk budgets are expressed in
#: tokens because that is what model context is priced in, but we avoid a
#: tokenizer dependency; the estimate is deliberately conservative.
CHARS_PER_TOKEN = 4

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# The fullwidth stops below are intentional: they end sentences in CJK text.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|(?<=[。！？])\s*")  # noqa: RUF001
_WHITESPACE_RE = re.compile(r"\s+")

#: A structured unit stays whole in one chunk up to this size: one function, one
#: vector. Under the embedding models' input windows (gemini-embedding-001 takes
#: 2048 tokens, and code tokenizes denser than the prose estimate assumes).
MAX_STRUCTURED_TOKENS = 1536
#: How much of the leading description rides along with each later chunk.
CONTEXT_TOKENS = 96

ContentKind = Literal["prose", "json", "code"]

_FENCE_RE = re.compile(r"^```[^\n]*\n.*?^```[ \t]*$", re.MULTILINE | re.DOTALL)
# A line that starts a top-level definition, in the languages agents write.
_DEFINITION_RE = re.compile(
    r"^(?:async\s+def|def|class|function|async\s+function|export|const|let|var|func|fn|"
    r"pub\s+fn|impl|interface|type|struct|enum|public|private|protected|static|module|"
    r"CREATE|SELECT|WITH)\b"
)
_DECORATOR_RE = re.compile(r"^@\w")
# Lines that look like code rather than prose, for detection only.
_CODE_LINE_RE = re.compile(
    r"^\s*(?:def |class |async def |import |from \S+ import |function |const |let |var |"
    r"export |return\b|if \(|for \(|while \(|#include|package |func |fn |public |private |@\w)"
    r"|[{};:]\s*$|^\s*[}\])]"
    # indented statements, assignments, calls, arrows: the bulk of a function body
    r"|^(?: {2,}|\t)\S|^\s*[\w.\[\]]+\s*[-+*/%|&]?=\s*\S|\w\([^()]*\)\s*$|=>|->"
)


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
    #: Embedded ahead of `text`, never stored: what a structured chunk is part of.
    context: str = ""


def normalize(text: str) -> str:
    """Strip control characters and normalise line endings.

    NUL bytes in particular must go: PostgreSQL rejects them in text columns,
    so a single \\x00 in an upload would otherwise fail at the storage layer with
    an unhelpful driver error rather than at validation with a clear one.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32).strip()


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
                    [part] if len(part) <= budget_chars else _split_units(part, budget_chars)
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


# -- structured content: code and JSON ----------------------------------------


def detect_kind(text: str) -> ContentKind:
    """JSON when the content is (or ends in) a JSON object or array, code when it has
    fenced blocks or reads mostly as code, else prose."""
    if _json_start(text) is not None:
        return "json"
    if _FENCE_RE.search(text):
        return "code"
    lines = [line for line in text.splitlines() if line.strip()]
    coded = sum(bool(_CODE_LINE_RE.search(x)) for x in lines)
    if len(lines) >= 3 and coded >= 0.5 * len(lines):
        return "code"
    return "prose"


def chunk_content(
    text: str,
    *,
    target_tokens: int = 320,
    overlap_tokens: int = 48,
    max_tokens: int = MAX_STRUCTURED_TOKENS,
    kind: ContentKind | None = None,
) -> list[TextChunk]:
    """Chunks sized to the content: prose by `chunk_text`, code and JSON by their
    structure (`chunk_structured`)."""
    cleaned = normalize(text)
    if not cleaned:
        return []
    kind = kind or detect_kind(cleaned)
    if kind == "prose":
        return chunk_text(cleaned, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    return chunk_structured(cleaned, kind, max_tokens=max_tokens)


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    path: str = ""


def chunk_structured(
    text: str, kind: ContentKind, *, max_tokens: int = MAX_STRUCTURED_TOKENS
) -> list[TextChunk]:
    """Whole when it fits `max_tokens`; else whole structural units packed into chunks as
    large as fit, each later chunk carrying the leading description as context."""
    if max_tokens <= CONTEXT_TOKENS:
        raise ValueError("max_tokens must exceed CONTEXT_TOKENS")
    text = normalize(text)
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [TextChunk(text, 0, 0, len(text), estimate_tokens(text))]

    head_end = _json_start(text) if kind == "json" else None
    if head_end is None and kind == "json":
        kind = "code"
    header = ""
    units: list[_Unit] = []
    budget = (max_tokens - CONTEXT_TOKENS) * CHARS_PER_TOKEN
    if kind == "json" and head_end is not None:
        header = text[:head_end].strip()
        if header:
            units += _prose_units(text, 0, head_end, budget)
        units += _json_units(text, head_end, _json_end(text, head_end), "", budget)
    else:
        units = _code_units(text, 0, len(text), budget)
        # The leading paragraph, when it describes rather than is code.
        lead = _PARAGRAPH_RE.split(text, maxsplit=1)[0].strip()
        if lead and not any(_CODE_LINE_RE.search(x) for x in lead.splitlines()):
            header = lead
    context_head = header[: CONTEXT_TOKENS * CHARS_PER_TOKEN].strip()

    chunks: list[TextChunk] = []
    i = 0
    while i < len(units):
        j = i + 1
        while j < len(units) and units[j].end - units[i].start <= budget:
            j += 1
        start, end = units[i].start, units[j - 1].end
        body = text[start:end]
        paths = list(dict.fromkeys(u.path for u in units[i:j] if u.path))
        parts = [] if (header and start < len(header)) else [context_head]
        if paths:
            parts.append("in " + ", ".join(paths))
        chunks.append(
            TextChunk(
                text=body,
                ordinal=len(chunks),
                start=start,
                end=end,
                token_estimate=estimate_tokens(body),
                context=" ".join(p for p in parts if p),
            )
        )
        i = j
    return chunks


def _json_start(text: str) -> int | None:
    """Where a JSON object or array that runs to the end of `text` begins: at the start, or
    after a leading description that ends in a blank line or a colon."""
    candidates = [0] + [m.end() for m in re.finditer(r"(?:\n\s*\n|:\s*\n)\s*", text)]
    for pos in candidates:
        if pos < len(text) and text[pos] in "{[":
            try:
                value = json.loads(text[pos:])
            except ValueError:
                continue
            if isinstance(value, dict | list) and value:
                return pos
    return None


def _json_end(text: str, start: int) -> int:
    return len(text.rstrip())


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _members(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Spans of the members of the JSON container text[start:end] (`{...}` or `[...]`):
    object members as `"key": value`, array elements, found by scanning the raw text."""
    spans: list[tuple[int, int]] = []
    depth, in_string, escaped, segment = 0, False, False, start + 1
    for i in range(start + 1, end - 1):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        elif c == "," and depth == 0:
            spans.append(_strip_span(text, segment, i))
            segment = i + 1
    spans.append(_strip_span(text, segment, end - 1))
    return [(a, b) for a, b in spans if b > a]


def _json_units(text: str, start: int, end: int, path: str, budget: int) -> list[_Unit]:
    """Whole members where they fit the budget; a member too big is opened up, its own
    members becoming units under its path."""
    out: list[_Unit] = []
    is_array = text[start] == "["
    for index, (a, b) in enumerate(_members(text, start, end)):
        if b - a <= budget:
            out.append(_Unit(a, b, path))
            continue
        name, value = _member_value(text, a, b)
        inner = f"{path}[{index}]" if is_array else (f"{path}.{name}" if path else name)
        if value is not None and text[value] in "{[":
            inside = _json_units(text, value, b, inner, budget)
        else:  # one enormous scalar: a long string, cut as text
            inside = [_Unit(x, y, inner) for x, y in _span_pieces(text, a, b, budget)]
        if inside:
            # The member's key and brackets go with its first and last pieces, so the
            # stored text says what it is and nothing between the pieces is dropped.
            inside[0] = _Unit(a, inside[0].end, inside[0].path)
            inside[-1] = _Unit(inside[-1].start, b, inside[-1].path)
        out += inside
    return out


def _member_value(text: str, start: int, end: int) -> tuple[str, int | None]:
    """An object member's key and where its value starts; for an array element, ("", start)."""
    if text[start] != '"':
        return "", start
    i, escaped = start + 1, False
    while i < end:
        c = text[i]
        if escaped:
            escaped = False
        elif c == "\\":
            escaped = True
        elif c == '"':
            break
        i += 1
    key = text[start + 1 : i]
    colon = text.find(":", i, end)
    if colon < 0:
        return key, None
    value = colon + 1
    while value < end and text[value].isspace():
        value += 1
    return key, value if value < end else None


def _code_units(text: str, start: int, end: int, budget: int) -> list[_Unit]:
    """Top-level definitions (with their decorators) and fenced blocks; a unit too big is
    split at blank lines, then lines, then characters."""
    fences = list(_FENCE_RE.finditer(text, start, end))
    if fences:
        out: list[_Unit] = []
        cursor = start
        for m in fences:
            if m.start() > cursor:
                out += _prose_units(text, cursor, m.start(), budget)
            out += (
                [_Unit(m.start(), m.end())]
                if m.end() - m.start() <= budget
                else _code_units(text, m.start(), m.end(), budget)
            )
            cursor = m.end()
        if cursor < end:
            out += _prose_units(text, cursor, end, budget)
        return out

    lines = _lines(text, start, end)
    bounds = [lines[0][0]] if lines else []
    for k, (a, b) in enumerate(lines):
        line = text[a:b]
        if k == 0 or not _DEFINITION_RE.match(line):
            continue
        first = k
        while first > 0:
            above = text[lines[first - 1][0] : lines[first - 1][1]]
            if not _DECORATOR_RE.match(above):
                break
            first -= 1
        if lines[first][0] > bounds[-1]:
            bounds.append(lines[first][0])
    bounds.append(end)
    out = []
    for a, b in pairwise(bounds):
        a, b = _strip_span(text, a, b)
        if b <= a:
            continue
        out += (
            [_Unit(a, b)]
            if b - a <= budget
            else [_Unit(x, y) for x, y in _span_pieces(text, a, b, budget)]
        )
    return out


def _prose_units(text: str, start: int, end: int, budget: int) -> list[_Unit]:
    a, b = _strip_span(text, start, end)
    if b <= a:
        return []
    return [_Unit(x, y) for x, y in _span_pieces(text, a, b, budget)]


def _lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    out, cursor = [], start
    while cursor < end:
        nl = text.find("\n", cursor, end)
        stop = end if nl < 0 else nl
        out.append((cursor, stop))
        cursor = stop + 1
    return out


def _span_pieces(text: str, start: int, end: int, budget: int) -> list[tuple[int, int]]:
    """text[start:end] cut into spans of at most `budget` characters, at the strongest
    boundary inside: a blank line, then a line break, then a space, else anywhere."""
    pieces: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > budget:
        limit = cursor + budget
        cut = -1
        for sep in ("\n\n", "\n", " "):
            found = text.rfind(sep, cursor + 1, limit)
            if found > cursor:
                cut = found
                break
        if cut <= cursor:
            cut = limit
        a, b = _strip_span(text, cursor, cut)
        if b > a:
            pieces.append((a, b))
        cursor = cut
    a, b = _strip_span(text, cursor, end)
    if b > a:
        pieces.append((a, b))
    return pieces


__all__ = [
    "CHARS_PER_TOKEN",
    "CONTEXT_TOKENS",
    "MAX_STRUCTURED_TOKENS",
    "ContentKind",
    "TextChunk",
    "chunk_content",
    "chunk_structured",
    "chunk_text",
    "detect_kind",
    "estimate_tokens",
    "normalize",
]
