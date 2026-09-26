"""Batched bulk writes: the sequential write path, run against an overlay.

`MemoryService.ingest_many` decides each item with the same methods a single
write uses -- `_fold_exact_duplicate`, `_write_keyed`, `_write_unkeyed` --
and those methods talk to a store. Handing them a `BulkView` instead of the
real store is the whole trick: the view answers their reads from one
prefetch plus the request's own pending writes, records their writes as
operations, and the backend applies the lot in one transaction
(`BulkSession.apply`).

So there is one implementation of dedupe, keyed replacement, near-duplicate
merging and association, not two that must be kept in agreement. What the
view has to get right is narrower: that every read it answers is the read the
sequential path would have made at that point, against a database holding the
earlier items' writes. Each method says how it does that. Where it cannot --
an ERASE that would destroy history, a replaced row something was derived
from -- it raises `NeedsSequential` before anything is recorded, and the
service writes that item the old way.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .core.errors import ConflictError, StoreError, ValidationError
from .domain.consolidation import dedupe_eligible, keyed_unchanged
from .domain.embeddings.base import Vector
from .domain.models import (
    Memory,
    MemoryStatus,
    RelationEdge,
    RelationType,
    ReplaceMode,
    utcnow,
)
from .store.base import KeyedWrite
from .store.bulk import (
    BulkOp,
    BulkPrefetch,
    BulkSession,
    EdgeOp,
    KeyedOp,
    NeighbourHit,
    UpsertOp,
)


class WriteStore(Protocol):
    """The store methods the write path uses. `MemoryStore` and `BulkView` both
    satisfy it."""

    async def find_by_content_hash(
        self, org_id: str, space_id: str, digest: str
    ) -> Memory | None: ...

    async def get_active_by_keys(
        self,
        org_id: str,
        space_id: str,
        keys: Sequence[str],
        *,
        with_embeddings: bool = True,
    ) -> dict[str, Memory]: ...

    async def write_keyed(
        self,
        memory: Memory,
        *,
        mode: ReplaceMode,
        reason: str = "replaced under the same key",
        now: datetime | None = None,
    ) -> KeyedWrite: ...

    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory | None: ...

    async def get_memories(
        self,
        org_id: str,
        space_id: str,
        memory_ids: Sequence[str],
        *,
        with_embeddings: bool = True,
    ) -> dict[str, Memory]: ...

    async def upsert_memory(self, memory: Memory, *, now: datetime | None = None) -> Memory: ...

    async def create_relation(self, edge: RelationEdge) -> RelationEdge: ...

    async def list_relations(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        direction: str = "out",
        type: RelationType | None = None,
    ) -> list[RelationEdge]: ...

    async def neighbours(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        exclude_id: str = "",
    ) -> list[tuple[Memory, Vector]]: ...


class NeedsSequential(Exception):
    """The view met a write only the sequential path implements.

    Raised before anything is recorded for the item, so the caller can write
    that item (or, from `apply`, the whole segment) the old way.
    """


@dataclass(slots=True)
class BulkStats:
    """Where one bulk request's time went. Filled in by `ingest_many`."""

    items: int = 0
    #: Transactions the batched path opened (1 unless an item forced a split).
    segments: int = 0
    #: Items written by the sequential path: an ERASE with history, a
    #: replaced row with derivations, or everything when batching is off.
    sequential_items: int = 0
    #: Time awaiting the database: lookups, the bulk transaction's reads and
    #: writes, and any sequential writes.
    db_ms: float = 0.0
    #: Time awaiting the embedding provider.
    embed_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def cpu_ms(self) -> float:
        return max(0.0, self.total_ms - self.db_ms - self.embed_ms)


class TimedSession(BulkSession):
    """A `BulkSession` that charges every await to `BulkStats.db_ms`."""

    def __init__(self, inner: BulkSession, stats: BulkStats) -> None:
        self.inner = inner
        self.stats = stats
        self.dimensions = inner.dimensions
        self.chunk_cap_multiple = inner.chunk_cap_multiple

    def _charge(self, started: float) -> None:
        self.stats.db_ms += (time.perf_counter() - started) * 1000

    def as_stored(self, memory: Memory) -> Memory:
        return self.inner.as_stored(memory)

    def as_stored_vector(self, vector: Vector) -> Vector:
        return self.inner.as_stored_vector(vector)

    async def lock_keys(self, keys: Sequence[str]) -> None:
        started = time.perf_counter()
        try:
            await self.inner.lock_keys(keys)
        finally:
            self._charge(started)

    async def prefetch(
        self,
        *,
        keys: Sequence[str],
        digests: Sequence[str],
        queries: Sequence[Vector],
        limit: int,
    ) -> BulkPrefetch:
        started = time.perf_counter()
        try:
            return await self.inner.prefetch(
                keys=keys, digests=digests, queries=queries, limit=limit
            )
        finally:
            self._charge(started)

    async def neighbour_hits(
        self, queries: Sequence[Vector], limit: int
    ) -> list[list[NeighbourHit]]:
        started = time.perf_counter()
        try:
            return await self.inner.neighbour_hits(queries, limit)
        finally:
            self._charge(started)

    async def chunk_vectors(self, chunk_ids: Sequence[str]) -> dict[str, Vector]:
        started = time.perf_counter()
        try:
            return await self.inner.chunk_vectors(chunk_ids)
        finally:
            self._charge(started)

    async def get_memories(self, ids: Sequence[str]) -> dict[str, Memory]:
        started = time.perf_counter()
        try:
            return await self.inner.get_memories(ids)
        finally:
            self._charge(started)

    async def apply(self, ops: Sequence[BulkOp]) -> None:
        started = time.perf_counter()
        try:
            await self.inner.apply(ops)
        finally:
            self._charge(started)


@asynccontextmanager
async def timed_session(
    manager: AbstractAsyncContextManager[BulkSession], stats: BulkStats
) -> AsyncIterator[BulkSession]:
    """Enter a store's bulk session, charging its BEGIN and COMMIT to `db_ms`."""
    started = time.perf_counter()
    session = await manager.__aenter__()
    stats.db_ms += (time.perf_counter() - started) * 1000
    try:
        yield session
    except BaseException as exc:
        if not await manager.__aexit__(type(exc), exc, exc.__traceback__):
            raise
    else:
        started = time.perf_counter()
        await manager.__aexit__(None, None, None)
        stats.db_ms += (time.perf_counter() - started) * 1000


def _ranking_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.sumprod(vector, vector))


class BulkView:
    """A store as the sequential path would see it, mid-request.

    Holds the rows the request has read or written, in their CURRENT state
    and in the form the backend would read them back (`as_stored`), and the
    operations that produced that state. The database underneath is the one
    the session opened on: nothing reaches it until `apply`.
    """

    def __init__(self, session: BulkSession, org_id: str, space_id: str, *, limit: int) -> None:
        self.session = session
        self.org_id = org_id
        self.space_id = space_id
        self.limit = limit
        #: Current state of every row this request has read or written.
        self.rows: dict[str, Memory] = {}
        #: Rows this request created, current state, in creation order.
        self.created: dict[str, Memory] = {}
        #: Every operation, in the order the sequential path would have made it.
        self.ops: list[BulkOp] = []
        #: digest -> the prefetched exact-duplicate holder (None: none).
        self._holder: dict[str, str | None] = {}
        #: key -> id of its ACTIVE row (None: none). Keys never looked up are absent.
        self._key_active: dict[str, str | None] = {}
        #: Pre-existing rows whose incoming DERIVED_FROM edges were checked,
        #: and those that have some.
        self._derivation_checked: set[str] = set()
        self._derived: set[str] = set()
        #: Neighbour hits per prefetched query vector.
        self._hits: dict[tuple[float, ...], list[NeighbourHit]] = {}
        #: Edges written by this request, by (source, target, type).
        self._edges: dict[tuple[str, str, RelationType], RelationEdge] = {}
        #: Ranking norms, by vector identity (vectors held by `created`).
        self._norms: dict[int, float] = {}

    # -- loading ------------------------------------------------------------

    def _absorb(
        self,
        prefetch: BulkPrefetch,
        *,
        keys: Sequence[str],
        digests: Sequence[str],
        queries: Sequence[Vector],
    ) -> None:
        for memory in prefetch.memories.values():
            self.rows.setdefault(memory.id, memory)
        for key in keys:
            found = prefetch.active_by_key.get(key)
            if found is not None:
                self.rows.setdefault(found.id, found)
                self._derivation_checked.add(found.id)
            self._key_active.setdefault(key, found.id if found is not None else None)
        self._derived |= prefetch.derived
        for digest in digests:
            holder = prefetch.by_hash.get(digest)
            if holder is not None:
                self.rows.setdefault(holder.id, holder)
            self._holder.setdefault(digest, holder.id if holder is not None else None)
        for query, hits in zip(queries, prefetch.hits, strict=False):
            self._hits[tuple(query)] = hits

    async def load(
        self, *, keys: Sequence[str], digests: Sequence[str], queries: Sequence[Vector]
    ) -> None:
        """Everything the request is expected to read, in one prefetch."""
        keys = list(dict.fromkeys(keys))
        digests = list(dict.fromkeys(digests))
        prefetch = await self.session.prefetch(
            keys=keys, digests=digests, queries=queries, limit=self.limit
        )
        self._absorb(prefetch, keys=keys, digests=digests, queries=queries)

    async def must_go_sequential(self, memory: Memory, mode: ReplaceMode) -> bool:
        """Whether a keyed write needs what only the sequential path does.

        ERASE destroys every row and version ever held under the key, history
        the view does not hold; SUPERSEDE of a stored row that something is
        DERIVED_FROM marks those derivations stale, a transitive walk over
        edges the view does not hold either. Both are rare on a bulk write and
        both keep their exact sequential behaviour.
        """
        key = memory.key
        if key is None:
            return False
        if mode is ReplaceMode.ERASE:
            return True
        active = (await self.get_active_by_keys(self.org_id, self.space_id, [key])).get(key)
        if active is None or active.id in self.created:
            return False
        return active.id in self._derived or active.id not in self._derivation_checked

    def _put(self, memory: Memory, *, created: bool = False) -> Memory:
        stored = self.session.as_stored(memory)
        self.rows[stored.id] = stored
        if created or stored.id in self.created:
            self.created[stored.id] = stored
        return stored

    def _check_dimensions(self, memory: Memory) -> None:
        dims = self.session.dimensions
        if dims is None:
            return
        for chunk in memory.chunks:
            if chunk.embedding is not None and len(chunk.embedding) != dims:
                raise StoreError(
                    f"chunk {chunk.id} has {len(chunk.embedding)} dimensions, "
                    f"index expects {dims}"
                )

    async def _load_rows(self, ids: Sequence[str]) -> None:
        missing = [i for i in dict.fromkeys(ids) if i not in self.rows]
        if missing:
            for memory in (await self.session.get_memories(missing)).values():
                self.rows.setdefault(memory.id, memory)

    # -- the reads ------------------------------------------------------------

    async def find_by_content_hash(
        self, org_id: str, space_id: str, digest: str
    ) -> Memory | None:
        """The lowest-id ACTIVE unkeyed row with this digest, as the database
        would hold it now: the prefetched holder, or a row this request
        created. Nothing the batched path does can make a stored holder
        ineligible (that takes a supersession, which needs a model, or a
        stale-marking, which goes sequential), so the check below is a guard,
        not a case."""
        if digest not in self._holder:
            await self.load(keys=[], digests=[digest], queries=[])
        candidates: list[Memory] = []
        holder = self._holder.get(digest)
        if holder is not None:
            current = self.rows[holder]
            if not dedupe_eligible(current):
                raise NeedsSequential(f"exact-duplicate holder {holder} changed state")
            candidates.append(current)
        candidates += [
            m
            for m in self.created.values()
            if m.content_sha256 == digest and dedupe_eligible(m)
        ]
        return min(candidates, key=lambda m: m.id) if candidates else None

    async def get_active_by_keys(
        self,
        org_id: str,
        space_id: str,
        keys: Sequence[str],
        *,
        with_embeddings: bool = True,
    ) -> dict[str, Memory]:
        del with_embeddings
        unknown = [k for k in keys if k not in self._key_active]
        if unknown:
            await self.load(keys=unknown, digests=[], queries=[])
        out: dict[str, Memory] = {}
        for key in keys:
            active = self._key_active.get(key)
            if active is not None:
                out[key] = self.rows[active]
        return out

    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory | None:
        await self._load_rows([memory_id])
        return self.rows.get(memory_id)

    async def get_memories(
        self,
        org_id: str,
        space_id: str,
        memory_ids: Sequence[str],
        *,
        with_embeddings: bool = True,
    ) -> dict[str, Memory]:
        del with_embeddings
        await self._load_rows(memory_ids)
        return {i: self.rows[i] for i in memory_ids if i in self.rows}

    async def list_relations(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        direction: str = "out",
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        """Only the question the write path asks: what is DERIVED_FROM a row.

        For a row this request created, its edges are all here. For a stored
        row, only when the prefetch checked it and found none -- anything
        else is the sequential path's."""
        if direction != "in" or type is not RelationType.DERIVED_FROM:
            raise NeedsSequential(f"list_relations({direction!r}, {type!r})")
        if memory_id not in self.created and (
            memory_id in self._derived or memory_id not in self._derivation_checked
        ):
            raise NeedsSequential(f"derivations of stored memory {memory_id}")
        return [
            edge
            for (_, target, kind), edge in self._edges.items()
            if target == memory_id and kind is type
        ]

    async def neighbours(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        exclude_id: str = "",
    ) -> list[tuple[Memory, Vector]]:
        """The stored neighbours from the prefetch, merged with this request's
        new rows, ranked as the backend's own scan would rank them.

        Stored rows' embeddings do not change during the request (nothing the
        batched path does rewrites an existing row's chunks), so the
        prefetched hits are still right; what the sequential path would also
        have seen are the rows earlier items created. Those are scored here
        and merged in, and when the backend caps its scan by chunks (Postgres
        reads `limit * 4` chunks and keeps each memory's best) the merge
        applies the same cap, so a memory with many close chunks crowds out
        exactly what it would have crowded out in the database.
        """
        dims = self.session.dimensions
        if dims is not None and embedding and len(embedding) != dims:
            raise StoreError(
                f"query embedding has {len(embedding)} dimensions, index expects {dims}"
            )
        if not embedding or limit <= 0:
            return []
        hits = self._hits.get(tuple(embedding))
        if hits is None:
            hits = (await self.session.neighbour_hits([embedding], limit))[0]
            self._hits[tuple(embedding)] = hits
            await self._load_rows([h.memory_id for h in hits])

        # (similarity, memory id, vector or the stored hit that has it)
        candidates: list[tuple[float, str, Vector | NeighbourHit]] = [
            (h.similarity, h.memory_id, h)
            for h in hits
            if h.memory_id != exclude_id
            and h.memory_id not in self.created
            and self._visible(h.memory_id)
        ]
        query_norm = _ranking_norm(embedding)
        for memory in self.created.values():
            if memory.id == exclude_id or memory.status is MemoryStatus.ARCHIVED:
                continue
            for chunk in memory.chunks:
                vector = chunk.embedding
                if vector is None or len(vector) != len(embedding):
                    continue
                candidates.append(
                    (self._rank(embedding, query_norm, vector), memory.id, vector)
                )

        cap = self.session.chunk_cap_multiple
        if cap is not None:
            # Highest first, stable: the scan's ORDER BY distance LIMIT cap.
            candidates.sort(key=lambda row: -row[0])
            candidates = candidates[: limit * cap]
        best: dict[str, tuple[float, Vector | NeighbourHit]] = {}
        for similarity, memory_id, source in candidates:
            current = best.get(memory_id)
            if current is None or similarity > current[0]:
                best[memory_id] = (similarity, source)
        ordered = sorted(best.items(), key=lambda row: (-row[1][0], row[0]))[:limit]

        missing = [
            source.chunk_id
            for _, (_, source) in ordered
            if isinstance(source, NeighbourHit) and source.vector is None
        ]
        fetched = await self.session.chunk_vectors(missing) if missing else {}
        await self._load_rows([memory_id for memory_id, _ in ordered])
        out: list[tuple[Memory, Vector]] = []
        for memory_id, (_, source) in ordered:
            if isinstance(source, NeighbourHit):
                vector = source.vector or fetched.get(source.chunk_id)
                if vector is None:
                    raise NeedsSequential(f"no vector for chunk {source.chunk_id!r}")
            else:
                vector = source
            out.append((self.rows[memory_id], vector))
        return out

    def _visible(self, memory_id: str) -> bool:
        current = self.rows.get(memory_id)
        return current is None or current.status is not MemoryStatus.ARCHIVED

    def _rank(self, query: Vector, query_norm: float, vector: Vector) -> float:
        """Cosine for RANKING a pending row against stored ones.

        `math.sumprod` rather than the write path's exact loop: the database
        ranks stored rows with pgvector's own arithmetic, so the ranking was
        never bit-exact against Python anyway, and this is ~20x faster. The
        similarities the consolidation checks act on are still computed by
        them, exactly, from the vectors returned.
        """
        norm = self._norms.get(id(vector))
        if norm is None:
            norm = _ranking_norm(vector)
            self._norms[id(vector)] = norm
        if norm == 0.0 or query_norm == 0.0:
            return 0.0
        return max(-1.0, min(1.0, math.sumprod(query, vector) / (query_norm * norm)))

    # -- the writes -----------------------------------------------------------

    async def upsert_memory(self, memory: Memory, *, now: datetime | None = None) -> Memory:
        when = now or utcnow()
        fresh = memory.id not in self.rows
        if fresh:
            self._check_dimensions(memory)
        self.ops.append(UpsertOp(memory=memory, when=when, fresh=fresh))
        self._put(memory, created=fresh)
        return memory

    async def write_keyed(
        self,
        memory: Memory,
        *,
        mode: ReplaceMode,
        reason: str = "replaced under the same key",
        now: datetime | None = None,
    ) -> KeyedWrite:
        """`store.write_keyed`, for SUPERSEDE and for a key's first write.

        The request holds the advisory lock of every key it writes for its
        whole transaction, so the active row read here is the one the store
        would read under its own per-write lock."""
        key = memory.key
        if key is None:
            raise ValidationError("write_keyed needs a memory with a key", field="key")
        if await self.must_go_sequential(memory, mode):
            raise NeedsSequential(f"keyed {mode.value} of {key!r}")
        active = (await self.get_active_by_keys(self.org_id, self.space_id, [key])).get(key)
        if active is not None and keyed_unchanged(active, memory):
            return KeyedWrite(memory=active, unchanged=True, mode=mode)
        if active is not None and active.id == memory.id:
            raise ValidationError(
                "a keyed write replaces the active row with a NEW memory; "
                f"{memory.id} is the row it would replace",
                field="id",
            )
        if memory.id in self.rows:
            raise ConflictError(
                f"memory {memory.id} already exists; a keyed write creates a new one",
                field="id",
            )
        self._check_dimensions(memory)
        when = now or utcnow()
        replaced: Memory | None = None
        if active is not None:
            replaced = self._put(
                active.model_copy(
                    update={
                        "status": MemoryStatus.SUPERSEDED,
                        "version": active.version + 1,
                        "updated_at": when,
                    }
                )
            )
        self.ops.append(
            KeyedOp(memory=memory, mode=mode, reason=reason, when=when, replaced=replaced)
        )
        self._put(memory, created=True)
        self._key_active[key] = memory.id
        if replaced is not None:
            edge = RelationEdge(
                org_id=memory.org_id,
                space_id=memory.space_id,
                source_id=memory.id,
                target_id=replaced.id,
                type=RelationType.SUPERSEDES,
                reason=reason,
                confidence=1.0,
            )
            self._edges[(edge.source_id, edge.target_id, edge.type)] = edge
        return KeyedWrite(
            memory=memory,
            unchanged=False,
            mode=mode,
            replaced=[replaced.id] if replaced is not None else [],
        )

    async def create_relation(self, edge: RelationEdge) -> RelationEdge:
        """Idempotent within the request; against stored edges the backend's
        insert is (ON CONFLICT DO NOTHING / the store's own check)."""
        triple = (edge.source_id, edge.target_id, edge.type)
        existing = self._edges.get(triple)
        if existing is not None:
            return existing
        self._edges[triple] = edge
        self.ops.append(EdgeOp(edge=edge))
        return edge


__all__ = [
    "BulkStats",
    "BulkView",
    "NeedsSequential",
    "TimedSession",
    "WriteStore",
    "timed_session",
]
