"""Embed a chunk with the context that makes it mean something.

A chunk is embedded alone, so a chat turn becomes a vector for a sentence
that has no date, no speaker and no subject:

    "yeah, three of them"

Nothing in that string says when it was said, who said it, or what "them"
refers to, so nothing in its vector does either. Every query that could find
it has to match twelve characters of pronoun.

The fix is to embed the chunk WITH a short header and store the chunk
without one:

    <context>
    2023-05-20 · user
    </context>
    yeah, three of them

THE INVARIANT, and the only thing that makes this different from everything
we tried before: the header is used for the EMBEDDING INPUT ONLY. The text
persisted in `chunks.text`, returned by search, quoted in a grounded claim
and shown to a user is the original, unchanged. Nothing downstream can tell
this happened.

That is why it is not the extraction arm again. Extraction ADDED retrieval
units -- one document became ~13 claims, a few documents ate the whole
window, and `full_recall@k` fell 0.968 -> 0.948 while MRR rose to 0.987:
sharper retrieval, worse coverage, -21 to -54 questions. This adds zero
units. Same chunk count, same window, same crowding characteristics; only
the vector moves.

The header is built from metadata we already hold, so it costs no model call
and no extra latency. There is a richer version of this idea where a model
writes a per-chunk synopsis; that one is worth measuring separately and is
not what this is.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

#: Keys checked, in order, for a human-meaningful name for the unit a chunk
#: belongs to. First match wins.
_TITLE_KEYS = ("title", "session_title", "subject", "conversation", "topic")

#: Longest header we will build. A header competing with the chunk for the
#: embedding's attention is a header making retrieval worse.
MAX_HEADER_CHARS = 160


def build_header(
    *,
    occurred_at: datetime | None = None,
    source: str = "",
    tags: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """A one-line header for a chunk, from metadata we already have.

    Returns "" when there is nothing worth saying, and the caller then embeds
    the chunk exactly as before -- so a corpus carrying no dates, sources or
    tags is unaffected rather than prefixed with noise.
    """
    meta = metadata or {}
    parts: list[str] = []

    if occurred_at is not None:
        # Date only. A timestamp to the second is sixteen characters of
        # precision no question ever asks for, spent inside a budget the
        # chunk needs.
        parts.append(occurred_at.date().isoformat())

    for key in _TITLE_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            break

    if source:
        parts.append(source)

    # Tags last: they are the weakest signal and the easiest to have too many
    # of. Two is enough to place a chunk in a topic.
    parts.extend(t for t in tags[:2] if t)

    header = " · ".join(dict.fromkeys(parts))
    return header[:MAX_HEADER_CHARS]


def wants_header(content: str, *, min_chars: int) -> bool:
    """Whether a memory this long should be embedded with a header at all.

    The header is the right trade for a memory whose chunks never say when or
    about what they were written. On a short, self-describing memory -- a
    profile card, a one-line fact -- it is most of the embedding input, and
    two unrelated short texts sharing a header come out looking alike:
    measured on card-sized text, unrelated pairs rose from 0.745 to 0.895.

    Decided per MEMORY, not per chunk. A long memory's last chunk is often
    short, and it is the chunk that most needs the framing: a tail fragment
    is exactly the "yeah, three of them" this module exists for. Whitespace
    does not count toward the length. `min_chars=0` keeps the header on
    everything, which is the behaviour before the setting existed.
    """
    return len(content.strip()) >= min_chars


def for_embedding(text: str, header: str) -> str:
    """The string to embed. `text` is what gets stored, always.

    Callers must pass the result to the embedder and NEVER to the chunk they
    persist. The delimiter is explicit so the model reads the header as
    framing rather than as the start of the sentence.
    """
    if not header:
        return text
    return f"<context>\n{header}\n</context>\n{text}"


__all__ = ["MAX_HEADER_CHARS", "build_header", "for_embedding", "wants_header"]
