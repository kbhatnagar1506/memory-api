"""Response shapes, as plain dataclasses.

No pydantic. A client library that pins a validation framework forces its
version on every application that installs it, and this one has nothing to
validate -- the server already did. `raw` keeps the untouched payload so a
field added server-side is reachable the day it ships rather than the day
the SDK is updated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    content: str
    kind: str = "episodic"
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    occurred_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, data: dict[str, Any]) -> Memory:
        return cls(
            id=str(data.get("id", "")),
            content=str(data.get("content", "")),
            kind=str(data.get("kind", "episodic")),
            status=str(data.get("status", "active")),
            tags=list(data.get("tags") or []),
            metadata=dict(data.get("metadata") or {}),
            source=str(data.get("source", "")),
            occurred_at=_dt(data.get("occurred_at")),
            raw=data,
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result. `score` is comparable within a response, not across them."""

    memory: Memory
    score: float
    explain: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.memory.id

    @property
    def content(self) -> str:
        return self.memory.content

    @classmethod
    def parse(cls, data: dict[str, Any]) -> SearchHit:
        inner = data.get("memory")
        payload = inner if isinstance(inner, dict) else data
        return cls(
            memory=Memory.parse(payload),
            score=float(data.get("score", 0.0) or 0.0),
            explain=[str(x) for x in (data.get("explain") or [])],
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[SearchHit]
    query: str = ""
    #: Pairs of returned ids joined by a CONTRADICTS edge. Surfaced rather
    #: than resolved: either side may be the true one, and an agent told
    #: "these disagree" can ask, where one handed a winner cannot.
    conflicts: list[tuple[str, str]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, index: int) -> SearchHit:
        return self.hits[index]

    @classmethod
    def parse(cls, data: dict[str, Any]) -> SearchResult:
        return cls(
            hits=[SearchHit.parse(h) for h in (data.get("results") or [])],
            query=str(data.get("query", "")),
            conflicts=[
                (str(a), str(b))
                for a, b in (data.get("conflicts") or [])
                if isinstance(a, str) and isinstance(b, str)
            ],
            raw=data,
        )


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """A memory with its whole neighbourhood, resolved in one call."""

    memory: Memory
    is_current: bool = True
    current_head: list[Memory] = field(default_factory=list)
    replaced: list[Memory] = field(default_factory=list)
    derived_from: list[Memory] = field(default_factory=list)
    derivatives: list[Memory] = field(default_factory=list)
    references: list[Memory] = field(default_factory=list)
    contradicts: list[Memory] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, data: dict[str, Any]) -> MemoryContext:
        def many(key: str) -> list[Memory]:
            return [Memory.parse(m) for m in (data.get(key) or [])]

        return cls(
            memory=Memory.parse(data.get("memory") or {}),
            is_current=bool(data.get("is_current", True)),
            current_head=many("current_head"),
            replaced=many("replaced"),
            derived_from=many("derived_from"),
            derivatives=many("derivatives"),
            references=many("references"),
            contradicts=many("contradicts"),
            raw=data,
        )


@dataclass(frozen=True, slots=True)
class Space:
    id: str
    slug: str
    name: str
    memory_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, data: dict[str, Any]) -> Space:
        return cls(
            id=str(data.get("id", "")),
            slug=str(data.get("slug", "")),
            name=str(data.get("name", "")),
            memory_count=data.get("memory_count"),
            raw=data,
        )


__all__ = ["Memory", "MemoryContext", "SearchHit", "SearchResult", "Space"]
