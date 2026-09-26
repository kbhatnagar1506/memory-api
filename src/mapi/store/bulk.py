"""The storage side of a batched bulk write: one transaction, few round trips.

`MemoryService.ingest_many` used to write a bulk request one item at a time
through the ordinary store methods, and every one of those opened its own
session -- BEGIN, set the tenant, a statement or three, COMMIT. Measured on
production that was ~40 round trips per item: a 91-section attendee memory
took ~19 s at 2.5 ms a round trip, almost none of it spent computing.

A `BulkSession` is the other shape. The service opens ONE, reads what the
whole request needs in a handful of statements (`prefetch`), decides every
item in process against an overlay of its own pending writes, and hands the
decisions back as an ordered list of operations (`apply`) that the backend
turns into a few multi-row statements.

The operations are the same calls the sequential path makes -- upsert, keyed
write, create edge -- in the same order, so a backend without a batched
implementation can simply replay them (`ReplayBulkSession`, what the
in-memory reference store uses). The Postgres backend compiles them instead.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from ..domain.embeddings.base import Vector, cosine_similarity
from ..domain.models import Memory, RelationEdge, RelationType, ReplaceMode

if TYPE_CHECKING:
    from .base import MemoryStore


@dataclass(frozen=True, slots=True)
class NeighbourHit:
    """One chunk an ANN prefetch returned for one query vector.

    `vector` is the chunk's embedding when the backend fetched it, and it is
    fetched for every memory's BEST chunk among a query's top `limit`
    memories -- the only chunks a caller can end up comparing against, since
    pending writes can push a stored memory down a ranking but never up.
    Vectors are shared objects across queries, so a similarity memo sees the
    same list twice.
    """

    memory_id: str
    similarity: float
    vector: Vector | None = None
    #: The chunk's id where the backend has one, to fetch a missing vector.
    chunk_id: str = ""


@dataclass(slots=True)
class BulkPrefetch:
    """Everything a bulk request reads, as of the start of its transaction."""

    #: ACTIVE row per key, for the keys asked about.
    active_by_key: dict[str, Memory] = field(default_factory=dict)
    #: The exact-duplicate gate's answer per digest (ACTIVE, unkeyed, lowest id).
    by_hash: dict[str, Memory] = field(default_factory=dict)
    #: Per query vector, in scan order (best first).
    hits: list[list[NeighbourHit]] = field(default_factory=list)
    #: Every memory a hit names. Chunks may come without vectors.
    memories: dict[str, Memory] = field(default_factory=dict)
    #: Ids among `active_by_key`'s rows that some memory is DERIVED_FROM.
    derived: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class UpsertOp:
    """`store.upsert_memory(memory, now=when)`. `fresh`: the row is new."""

    memory: Memory
    when: datetime
    fresh: bool


@dataclass(frozen=True, slots=True)
class KeyedOp:
    """`store.write_keyed(memory, mode=mode, reason=reason, now=when)`.

    Only ever a write that changes the key (an unchanged payload records no
    op) and only SUPERSEDE or a first write under the key: an ERASE that has
    anything to destroy takes the sequential path. `replaced` is the row it
    demotes, already in its demoted state, or None.
    """

    memory: Memory
    mode: ReplaceMode
    reason: str
    when: datetime
    replaced: Memory | None


@dataclass(frozen=True, slots=True)
class EdgeOp:
    """`store.create_relation(edge)`."""

    edge: RelationEdge


BulkOp = UpsertOp | KeyedOp | EdgeOp


class BulkSession(abc.ABC):
    """One tenant, one space, one transaction where the backend has them."""

    #: Width the backend stores, when it enforces one. The sequential path
    #: fails an item whose vectors disagree (`StoreError`); the batched path
    #: checks up front so the same item fails the same way, alone.
    dimensions: int | None = None
    #: How many CHUNKS the backend's neighbour scan reads per requested
    #: memory, when that cap shapes its answer (Postgres over-fetches 4x and
    #: keeps each memory's best). None: hits are exact, one per memory.
    chunk_cap_multiple: int | None = None

    def as_stored(self, memory: Memory) -> Memory:
        """`memory` as a later read of it would return it from this backend.

        A pending write is read back by later items of the same request; this
        makes that read look like the one the sequential path would have done
        against the database -- same timezone, same vector precision.
        """
        return memory

    def as_stored_vector(self, vector: Vector) -> Vector:
        """A vector as the backend stores it (float32 for pgvector)."""
        return vector

    async def lock_keys(self, keys: Sequence[str]) -> None:
        """Serialize other writers of these keys until the session ends."""
        return None

    @abc.abstractmethod
    async def prefetch(
        self,
        *,
        keys: Sequence[str],
        digests: Sequence[str],
        queries: Sequence[Vector],
        limit: int,
    ) -> BulkPrefetch:
        """Read everything the request needs, in as few round trips as the
        backend can manage."""

    @abc.abstractmethod
    async def neighbour_hits(
        self, queries: Sequence[Vector], limit: int
    ) -> list[list[NeighbourHit]]:
        """`prefetch`'s neighbour scan alone, for a vector it did not know
        about (an item that had to be embedded after all)."""

    async def chunk_vectors(self, chunk_ids: Sequence[str]) -> dict[str, Vector]:
        """Vectors for hits that came without one. Backends whose hits always
        carry vectors never get asked."""
        del chunk_ids
        return {}

    @abc.abstractmethod
    async def get_memories(self, ids: Sequence[str]) -> dict[str, Memory]:
        """Rows by id, for the rare read the prefetch did not anticipate."""

    @abc.abstractmethod
    async def apply(self, ops: Sequence[BulkOp]) -> None:
        """Write every decision, in order. Called at most once per session."""


class ReplayBulkSession(BulkSession):
    """A `BulkSession` over any `MemoryStore`'s public methods.

    No transaction and no batching: prefetch is the store's own lookups and
    `apply` replays each operation through the method it names -- which is
    exactly what the sequential path would have called, in the same order.
    The in-memory reference store uses this; its operations cost nothing, and
    replaying them keeps it the reference for what the batched Postgres
    implementation must agree with.
    """

    def __init__(self, store: MemoryStore, org_id: str, space_id: str) -> None:
        self.store = store
        self.org_id = org_id
        self.space_id = space_id

    async def prefetch(
        self,
        *,
        keys: Sequence[str],
        digests: Sequence[str],
        queries: Sequence[Vector],
        limit: int,
    ) -> BulkPrefetch:
        store, org, space = self.store, self.org_id, self.space_id
        out = BulkPrefetch()
        if keys:
            out.active_by_key = await store.get_active_by_keys(org, space, keys)
        if digests:
            out.by_hash = await store.find_by_content_hashes(org, space, digests)
        for query in queries:
            pairs = await store.neighbours(org, space, query, limit=limit)
            out.hits.append(self._hits(query, pairs))
            for memory, _ in pairs:
                out.memories.setdefault(memory.id, memory)
        derived: set[str] = set()
        for memory in out.active_by_key.values():
            edges = await store.list_relations(
                org, space, memory.id, direction="in", type=RelationType.DERIVED_FROM
            )
            if edges:
                derived.add(memory.id)
        out.derived = frozenset(derived)
        return out

    @staticmethod
    def _hits(query: Vector, pairs: Sequence[tuple[Memory, Vector]]) -> list[NeighbourHit]:
        return [
            NeighbourHit(memory.id, cosine_similarity(query, vector), vector)
            for memory, vector in pairs
        ]

    async def neighbour_hits(
        self, queries: Sequence[Vector], limit: int
    ) -> list[list[NeighbourHit]]:
        out = []
        for query in queries:
            pairs = await self.store.neighbours(self.org_id, self.space_id, query, limit=limit)
            out.append(self._hits(query, pairs))
        return out

    async def get_memories(self, ids: Sequence[str]) -> dict[str, Memory]:
        return await self.store.get_memories(self.org_id, self.space_id, ids)

    async def apply(self, ops: Sequence[BulkOp]) -> None:
        for op in ops:
            if isinstance(op, UpsertOp):
                await self.store.upsert_memory(op.memory, now=op.when)
            elif isinstance(op, KeyedOp):
                await self.store.write_keyed(
                    op.memory, mode=op.mode, reason=op.reason, now=op.when
                )
            else:
                await self.store.create_relation(op.edge)


@asynccontextmanager
async def replay_session(
    store: MemoryStore, org_id: str, space_id: str
) -> AsyncIterator[BulkSession]:
    yield ReplayBulkSession(store, org_id, space_id)


__all__ = [
    "BulkOp",
    "BulkPrefetch",
    "BulkSession",
    "EdgeOp",
    "KeyedOp",
    "NeighbourHit",
    "ReplayBulkSession",
    "UpsertOp",
    "replay_session",
]
