"""The storage port.

Two implementations satisfy this interface — in-memory and PostgreSQL — and one
shared conformance suite runs against both. That is the point: the algorithms
above never learn which backend they are talking to, and a behavioural
difference between backends fails a test rather than surfacing in production.

Tenancy is enforced *here*, not in the API layer. Every method that touches
memories takes an `org_id` and a `space_id`, and implementations must filter on
both. Putting that check in a route handler means the one handler that forgets
leaks another tenant's data; putting it in the port means the leak has to be
written deliberately.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..domain.embeddings.base import Vector
from ..domain.models import ApiKey, Memory, MemoryStatus, Organization, Space


@dataclass(frozen=True, slots=True)
class VectorHit:
    memory_id: str
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class LexicalHit:
    memory_id: str
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class Page:
    """Cursor-paginated result. Cursors are opaque to the caller by contract."""

    items: list[Memory]
    next_cursor: str | None = None
    total: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryFilter:
    """Filters applied by every listing and search operation."""

    statuses: frozenset[MemoryStatus] = field(
        default_factory=lambda: frozenset({MemoryStatus.ACTIVE})
    )
    tags: tuple[str, ...] = ()
    #: Every key/value must match. Values compare as JSON equality.
    metadata: tuple[tuple[str, Any], ...] = ()
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None
    source: str | None = None

    def matches(self, memory: Memory) -> bool:
        """Reference semantics. SQL backends must reproduce this exactly."""
        if memory.status not in self.statuses:
            return False
        if self.tags and not set(self.tags).issubset(set(memory.tags)):
            return False
        for key, value in self.metadata:
            if memory.metadata.get(key) != value:
                return False
        if self.occurred_after is not None:
            if _epoch(memory.occurred_at) < _epoch(self.occurred_after):
                return False
        if self.occurred_before is not None:
            if _epoch(memory.occurred_at) > _epoch(self.occurred_before):
                return False
        return not (self.source is not None and memory.source != self.source)


def _epoch(value: datetime) -> float:
    if value.tzinfo is None:
        from datetime import timezone

        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


class MemoryStore(abc.ABC):
    """Persistence and retrieval primitives. Implementations must be async-safe."""

    # -- lifecycle ---------------------------------------------------------

    async def initialize(self) -> None:
        """Prepare the backend (create schema, warm pools). Idempotent."""
        return None

    async def aclose(self) -> None:
        return None

    @abc.abstractmethod
    async def ping(self) -> bool:
        """Cheap liveness probe. Must not raise."""

    # -- organizations & spaces -------------------------------------------

    @abc.abstractmethod
    async def create_organization(self, org: Organization) -> Organization: ...

    @abc.abstractmethod
    async def get_organization(self, org_id: str) -> Organization | None: ...

    @abc.abstractmethod
    async def create_space(self, space: Space) -> Space: ...

    @abc.abstractmethod
    async def get_space(self, org_id: str, space_id: str) -> Space | None: ...

    @abc.abstractmethod
    async def get_space_by_slug(self, org_id: str, slug: str) -> Space | None: ...

    @abc.abstractmethod
    async def list_spaces(self, org_id: str) -> list[Space]: ...

    @abc.abstractmethod
    async def delete_space(self, org_id: str, space_id: str) -> bool:
        """Delete a space and every memory in it. Returns False if absent."""

    # -- api keys ----------------------------------------------------------

    @abc.abstractmethod
    async def create_api_key(self, key: ApiKey) -> ApiKey: ...

    @abc.abstractmethod
    async def get_api_key_by_hash(self, key_hash: str) -> ApiKey | None: ...

    @abc.abstractmethod
    async def list_api_keys(self, org_id: str) -> list[ApiKey]: ...

    @abc.abstractmethod
    async def revoke_api_key(self, org_id: str, key_id: str) -> bool: ...

    @abc.abstractmethod
    async def touch_api_key(self, key_id: str, when: datetime) -> None:
        """Record last use. Best-effort: must never fail a request."""

    # -- memories ----------------------------------------------------------

    @abc.abstractmethod
    async def upsert_memory(self, memory: Memory) -> Memory:
        """Insert or replace by id. Chunks and embeddings are replaced wholesale."""

    @abc.abstractmethod
    async def get_memory(
        self, org_id: str, space_id: str, memory_id: str
    ) -> Memory | None: ...

    @abc.abstractmethod
    async def get_memories(
        self, org_id: str, space_id: str, memory_ids: Sequence[str]
    ) -> dict[str, Memory]:
        """Batch fetch. Missing ids are simply absent from the result."""

    @abc.abstractmethod
    async def delete_memory(self, org_id: str, space_id: str, memory_id: str) -> bool: ...

    @abc.abstractmethod
    async def list_memories(
        self,
        org_id: str,
        space_id: str,
        *,
        filters: MemoryFilter,
        limit: int,
        cursor: str | None = None,
    ) -> Page: ...

    @abc.abstractmethod
    async def count_memories(
        self, org_id: str, space_id: str, *, filters: MemoryFilter
    ) -> int: ...

    @abc.abstractmethod
    async def find_by_content_hash(
        self, org_id: str, space_id: str, digest: str
    ) -> Memory | None: ...

    # -- retrieval ---------------------------------------------------------

    @abc.abstractmethod
    async def vector_search(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        filters: MemoryFilter,
    ) -> list[VectorHit]:
        """Approximate nearest neighbours over chunks, best first.

        Scores are cosine similarity in [-1, 1]; higher is better.
        """

    @abc.abstractmethod
    async def lexical_search(
        self,
        org_id: str,
        space_id: str,
        query: str,
        *,
        limit: int,
        filters: MemoryFilter,
    ) -> list[LexicalHit]:
        """Full-text search over chunks, best first. Scores are backend-relative."""

    @abc.abstractmethod
    async def sample_embeddings(
        self, org_id: str, space_id: str, *, limit: int
    ) -> list[tuple[str, Vector]]:
        """(memory_id, embedding) pairs, for dedup and supersession checks."""


__all__ = [
    "LexicalHit",
    "MemoryFilter",
    "MemoryStore",
    "Page",
    "VectorHit",
]
