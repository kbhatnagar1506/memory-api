"""Sortable, prefixed, collision-resistant identifiers.

IDs are k-sortable (time-ordered prefix) so that index locality is good and
`ORDER BY id` approximates `ORDER BY created_at` without a second index. The
type prefix means an ID is self-describing in a log line, and passing a space
ID where a memory ID belongs is caught by validation rather than silently
querying nothing.
"""

from __future__ import annotations

import re
import secrets
import time
from typing import Final

_ALPHABET: Final = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford base32, no ILOU
_RANDOM_CHARS: Final = 16
_TIME_CHARS: Final = 10

PREFIXES: Final[dict[str, str]] = {
    "org": "org",
    "space": "spc",
    "memory": "mem",
    "chunk": "chk",
    "key": "key",
    "job": "job",
    "edge": "edg",
    "version": "ver",
    "user": "usr",
    "membership": "mbr",
}

_ID_RE: Final = re.compile(
    rf"^([a-z]{{3,5}})_([{_ALPHABET}]{{{_TIME_CHARS + _RANDOM_CHARS}}})$"
)


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        out.append(_ALPHABET[rem])
    return "".join(reversed(out))


def new_id(kind: str, *, now_ms: int | None = None) -> str:
    """Generate an identifier for `kind`. Monotonic prefix + 80 random bits."""
    if kind not in PREFIXES:
        raise ValueError(f"unknown id kind {kind!r}; known: {sorted(PREFIXES)}")
    ms = now_ms if now_ms is not None else int(time.time() * 1000)
    if ms < 0:
        raise ValueError("timestamp must be non-negative")
    ts = _encode(ms, _TIME_CHARS)
    rand = _encode(secrets.randbits(_RANDOM_CHARS * 5), _RANDOM_CHARS)
    return f"{PREFIXES[kind]}_{ts}{rand}"


def is_valid(value: object, kind: str | None = None) -> bool:
    if not isinstance(value, str):
        return False
    match = _ID_RE.match(value)
    if match is None:
        return False
    if kind is None:
        return match.group(1) in PREFIXES.values()
    return match.group(1) == PREFIXES.get(kind)


def kind_of(value: str) -> str | None:
    match = _ID_RE.match(value)
    if match is None:
        return None
    for kind, prefix in PREFIXES.items():
        if prefix == match.group(1):
            return kind
    return None


__all__ = ["PREFIXES", "is_valid", "kind_of", "new_id"]
