"""The Postgres `BulkSession`: a whole bulk write in one transaction.

Round trips, not rows, are what a bulk write costs over a network. The
sequential path paid ~35 per item (a session per store call, each with its
own BEGIN, tenant `set_config` and COMMIT, plus the row, open-version and
chunk reads an upsert makes). This pays a fixed dozen for the request:

    BEGIN, tenant + HNSW settings           2
    advisory locks for the request's keys   1   (only with keys)
    every item's neighbour scan             1   (one LATERAL statement)
    rows by id / key / digest               2   (rows, then their chunks)
    UPDATE demoted and merged rows          1   (pipelined executemany)
    close their open versions               1
    INSERT memories, versions, chunks,
      edges                                 4   (multi-row)
    COMMIT                                  1

Semantics are the sequential path's by construction: the service decides
every item with the same code, against an overlay of this request's pending
writes, and `apply` writes those decisions -- the operations the sequential
path would have made, in order -- compiled into the statements above. Row
level security is unchanged: `app.org_id` is set once, for the one
transaction everything runs in.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, Integer, String, and_, exists, insert, or_, select, text, update
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ...core.errors import StoreError
from ...domain.embeddings.base import Vector
from ...domain.models import (
    Memory,
    MemoryStatus,
    MemoryVersion,
    RelationEdge,
    RelationType,
    ReplaceMode,
)
from ..bulk import BulkOp, BulkPrefetch, BulkSession, EdgeOp, KeyedOp, NeighbourHit, UpsertOp
from .models import ChunkRow, MemoryRow, MemoryVersionRow, RelationEdgeRow
from .vector import BinaryVector

if TYPE_CHECKING:
    from .store import PostgresStore

#: Chunks the neighbour scan reads per requested memory. Must match
#: `PostgresStore.neighbours`, whose answer the batched scan reproduces.
CHUNK_CAP_MULTIPLE = 4

#: Every query's neighbour scan, in one statement.
#:
#: `q` unpacks the request's query vectors from ONE float4[] parameter
#: (binary on the wire, no text parsing), and the LATERAL join runs
#: `neighbours`' own statement once per vector: same filters, same
#: `ORDER BY distance LIMIT cap`, under the same HNSW settings. Hits come back
#: as arrays, one row per query, because 91 queries x 256 chunks as rows is
#: ~23k rows to decode for a handful of strings each.
#:
#: The second branch fetches VECTORS, once per distinct chunk, and only for
#: each memory's best chunk among its query's top `limit` memories: the only
#: chunks a caller can end up comparing against. The sequential path hauled
#: every one of the `cap` chunk vectors per item, ~768 KB at 768 dimensions.
_NEIGHBOURS_SQL = text(
    """
    WITH q AS (
        SELECT g.i,
               CAST((CAST(:flat AS real[]))[(g.i - 1) * :dims + 1 : g.i * :dims] AS vector) AS v
        FROM generate_series(1, :n) AS g(i)
    ),
    hits AS (
        SELECT q.i, h.chunk_id, h.memory_id, h.distance
        FROM q
        CROSS JOIN LATERAL (
            SELECT c.id AS chunk_id, c.memory_id, c.embedding <=> q.v AS distance
            FROM chunks c
            JOIN memories m ON m.id = c.memory_id
            WHERE c.org_id = :org
              AND c.space_id = :space
              AND c.embedding IS NOT NULL
              AND m.status <> 'archived'
            ORDER BY c.embedding <=> q.v
            LIMIT :cap
        ) h
    ),
    best AS (
        SELECT DISTINCT ON (i, memory_id) i, memory_id, chunk_id, distance
        FROM hits
        ORDER BY i, memory_id, distance, chunk_id
    ),
    wanted AS (
        SELECT DISTINCT chunk_id
        FROM (
            SELECT chunk_id,
                   row_number() OVER (PARTITION BY i ORDER BY distance, memory_id) AS rn
            FROM best
        ) ranked
        WHERE rn <= :limit
    )
    SELECT i,
           array_agg(memory_id ORDER BY distance, chunk_id) AS memory_ids,
           array_agg(chunk_id ORDER BY distance, chunk_id) AS chunk_ids,
           array_agg(distance ORDER BY distance, chunk_id) AS distances,
           NULL AS chunk_id,
           NULL AS embedding
    FROM hits
    GROUP BY i
    UNION ALL
    SELECT NULL, NULL, NULL, NULL, c.id, c.embedding
    FROM chunks c
    JOIN wanted w ON w.chunk_id = c.id
    """
)

_CLOSE_VERSIONS_SQL = text(
    """
    UPDATE memory_versions AS v
    SET valid_to = c.w
    FROM unnest(CAST(:ids AS text[]), CAST(:ws AS timestamptz[])) AS c(id, w)
    WHERE v.memory_id = c.id AND v.valid_to IS NULL
    """
)

_LOCK_SQL = text(
    """
    SELECT count(pg_advisory_xact_lock(hashtextextended(t.k, 0)))
    FROM unnest(CAST(:locks AS text[])) WITH ORDINALITY AS t(k, n)
    """
)


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _as_jsonb(value: dict[str, Any]) -> dict[str, Any]:
    """A metadata dict as JSONB hands it back: tuples become lists, and so on."""
    try:
        decoded = json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return value  # the INSERT will refuse it, as the sequential path's would
    return decoded if isinstance(decoded, dict) else value


def _float32(vector: Sequence[float]) -> Vector:
    """What a float64 list reads back as after a trip through pgvector."""
    return array("f", vector).tolist()


def _vector(value: Any) -> Vector | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    to_list = getattr(value, "to_list", None)
    return list(to_list()) if to_list is not None else list(value)


class PostgresBulkSession(BulkSession):
    chunk_cap_multiple = CHUNK_CAP_MULTIPLE

    def __init__(self, store: PostgresStore, session: Any, org_id: str, space_id: str) -> None:
        self.store = store
        self.session = session
        self.org_id = org_id
        self.space_id = space_id
        self.dimensions = store.dimensions
        self._applied = False

    # -- how a pending write reads back ---------------------------------------

    def as_stored(self, memory: Memory) -> Memory:
        update_: dict[str, Any] = {
            "occurred_at": _utc(memory.occurred_at),
            "created_at": _utc(memory.created_at),
            "updated_at": _utc(memory.updated_at),
            "metadata": _as_jsonb(memory.metadata),
        }
        if any(c.embedding is not None for c in memory.chunks):
            update_["chunks"] = [
                c.model_copy(update={"embedding": _float32(c.embedding)})
                if c.embedding is not None
                else c
                for c in memory.chunks
            ]
        return memory.model_copy(update=update_)

    def as_stored_vector(self, vector: Vector) -> Vector:
        return _float32(vector)

    # -- reads --------------------------------------------------------------

    async def lock_keys(self, keys: Sequence[str]) -> None:
        # The same lock `write_keyed` takes, per key, in sorted order so two
        # bulk requests over overlapping keys cannot deadlock each other.
        wanted = sorted({f"mapi:key:{self.space_id}:{key}" for key in keys})
        if wanted:
            await self.session.execute(_LOCK_SQL, {"locks": wanted})

    async def prefetch(
        self,
        *,
        keys: Sequence[str],
        digests: Sequence[str],
        queries: Sequence[Vector],
        limit: int,
    ) -> BulkPrefetch:
        out = BulkPrefetch()
        hit_ids: list[str] = []
        if queries and limit > 0:
            out.hits = await self._neighbour_hits(queries, limit)
            hit_ids = list(dict.fromkeys(h.memory_id for hits in out.hits for h in hits))
        key_list = list(dict.fromkeys(keys))
        digest_list = list(dict.fromkeys(digests))
        if not (hit_ids or key_list or digest_list):
            return out

        conditions: list[Any] = []
        if hit_ids:
            conditions.append(MemoryRow.id.in_(hit_ids))
        if key_list:
            conditions.append(
                and_(
                    MemoryRow.memory_key.in_(key_list),
                    MemoryRow.status == MemoryStatus.ACTIVE.value,
                )
            )
        if digest_list:
            conditions.append(
                and_(
                    MemoryRow.content_sha256.in_(digest_list),
                    MemoryRow.status == MemoryStatus.ACTIVE.value,
                    MemoryRow.memory_key.is_(None),
                )
            )
        derived = (
            exists()
            .where(
                RelationEdgeRow.org_id == self.org_id,
                RelationEdgeRow.space_id == self.space_id,
                RelationEdgeRow.target_id == MemoryRow.id,
                RelationEdgeRow.type == RelationType.DERIVED_FROM.value,
            )
            .label("derived")
        )
        from .store import _CHUNKS_WITHOUT_VECTORS, PostgresStore

        stmt = (
            select(MemoryRow, derived)
            .where(
                MemoryRow.org_id == self.org_id,
                MemoryRow.space_id == self.space_id,
                or_(*conditions),
            )
            .options(_CHUNKS_WITHOUT_VECTORS)
            .order_by(MemoryRow.id)
        )
        rows = (await self.session.execute(stmt)).all()
        wanted_keys, wanted_digests, wanted_hits = set(key_list), set(digest_list), set(hit_ids)
        derived_ids: set[str] = set()
        for row, is_derived in rows:
            memory = PostgresStore._to_memory(row)
            active = memory.status is MemoryStatus.ACTIVE
            if memory.id in wanted_hits:
                out.memories[memory.id] = memory
            if active and memory.key is not None and memory.key in wanted_keys:
                out.active_by_key[memory.key] = memory
                if is_derived:
                    derived_ids.add(memory.id)
            if active and memory.key is None and memory.content_sha256 in wanted_digests:
                # Lowest id wins, as in `find_by_content_hashes`: rows arrive
                # in id order.
                out.by_hash.setdefault(memory.content_sha256, memory)
        out.derived = frozenset(derived_ids)
        # Domain objects are all anyone reads from here on; nothing this
        # session loaded may be flushed back behind `apply`'s statements.
        self.session.expunge_all()
        return out

    async def _neighbour_hits(
        self, queries: Sequence[Vector], limit: int
    ) -> list[list[NeighbourHit]]:
        for query in queries:
            if len(query) != self.dimensions:
                raise StoreError(
                    f"query embedding has {len(query)} dimensions, "
                    f"index expects {self.dimensions}"
                )
        flat = [float(x) for query in queries for x in query]
        stmt = _NEIGHBOURS_SQL.columns(
            i=Integer,
            memory_ids=ARRAY(String),
            chunk_ids=ARRAY(String),
            distances=ARRAY(Float),
            chunk_id=String,
            embedding=BinaryVector(self.dimensions),
        )
        result = await self.session.execute(
            stmt,
            {
                "flat": flat,
                "dims": self.dimensions,
                "n": len(queries),
                "org": self.org_id,
                "space": self.space_id,
                "cap": limit * CHUNK_CAP_MULTIPLE,
                "limit": limit,
            },
        )
        per_query: dict[int, tuple[list[str], list[str], list[float]]] = {}
        vectors: dict[str, Vector] = {}
        for i, memory_ids, chunk_ids, distances, chunk_id, embedding in result:
            if i is None:
                found = _vector(embedding)
                if found is not None:
                    vectors[chunk_id] = found
            else:
                per_query[int(i)] = (memory_ids, chunk_ids, distances)
        out: list[list[NeighbourHit]] = []
        for i in range(1, len(queries) + 1):
            memory_ids, chunk_ids, distances = per_query.get(i, ([], [], []))
            out.append(
                [
                    NeighbourHit(
                        memory_id=memory_id,
                        similarity=1.0 - float(distance),
                        vector=vectors.get(chunk_id),
                        chunk_id=chunk_id,
                    )
                    for memory_id, chunk_id, distance in zip(
                        memory_ids, chunk_ids, distances, strict=True
                    )
                ]
            )
        return out

    async def neighbour_hits(
        self, queries: Sequence[Vector], limit: int
    ) -> list[list[NeighbourHit]]:
        return await self._neighbour_hits(queries, limit)

    async def chunk_vectors(self, chunk_ids: Sequence[str]) -> dict[str, Vector]:
        wanted = list(dict.fromkeys(chunk_ids))
        if not wanted:
            return {}
        rows = await self.session.execute(
            select(ChunkRow.id, ChunkRow.embedding).where(
                ChunkRow.org_id == self.org_id,
                ChunkRow.space_id == self.space_id,
                ChunkRow.id.in_(wanted),
            )
        )
        out: dict[str, Vector] = {}
        for chunk_id, embedding in rows:
            found = _vector(embedding)
            if found is not None:
                out[chunk_id] = found
        return out

    async def get_memories(self, ids: Sequence[str]) -> dict[str, Memory]:
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return {}
        from .store import _CHUNKS_WITHOUT_VECTORS, PostgresStore

        rows = await self.session.scalars(
            select(MemoryRow)
            .where(
                MemoryRow.org_id == self.org_id,
                MemoryRow.space_id == self.space_id,
                MemoryRow.id.in_(wanted),
            )
            .options(_CHUNKS_WITHOUT_VECTORS)
        )
        found = {row.id: PostgresStore._to_memory(row) for row in rows}
        self.session.expunge_all()
        return found

    # -- writes -------------------------------------------------------------

    async def apply(self, ops: Sequence[BulkOp]) -> None:
        if self._applied:
            raise StoreError("a bulk session applies its operations once")
        self._applied = True
        plan = compile_ops(ops, dimensions=self.dimensions)
        session = self.session
        # Order matters twice over. Demotions land before the inserts, or the
        # one-ACTIVE-row-per-key index sees the new row beside the old one;
        # memories land before anything with a foreign key to them.
        if plan.updates:
            await session.execute(update(MemoryRow), plan.updates)
        if plan.close_ids:
            await session.execute(
                _CLOSE_VERSIONS_SQL, {"ids": plan.close_ids, "ws": plan.close_at}
            )
        if plan.inserts:
            await session.execute(insert(MemoryRow), plan.inserts)
        if plan.versions:
            await session.execute(insert(MemoryVersionRow), plan.versions)
        if plan.chunks:
            await session.execute(insert(ChunkRow), plan.chunks)
        if plan.edges:
            await session.execute(
                pg_insert(RelationEdgeRow).on_conflict_do_nothing(constraint="uq_edges_triple"),
                plan.edges,
            )


# -- compiling operations into statements -----------------------------------------------


@dataclass
class CompiledBulk:
    """The rows `apply` writes. Pure data, so it is testable without a database."""

    updates: list[dict[str, Any]] = field(default_factory=list)
    close_ids: list[str] = field(default_factory=list)
    close_at: list[datetime] = field(default_factory=list)
    inserts: list[dict[str, Any]] = field(default_factory=list)
    versions: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)


def _row(memory: Memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "org_id": memory.org_id,
        "space_id": memory.space_id,
        "content": memory.content,
        "summary": memory.summary,
        "meta": memory.metadata,
        "tags": memory.tags,
        "source": memory.source,
        "kind": memory.kind.value,
        "status": memory.status.value,
        "occurred_at": memory.occurred_at,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "content_sha256": memory.content_sha256,
        "version": memory.version,
        "memory_key": memory.key,
    }


def _version_row(memory: Memory, when: datetime) -> dict[str, Any]:
    snapshot = MemoryVersion.snapshot(memory, valid_from=when)
    return {
        "id": snapshot.id,
        "memory_id": snapshot.memory_id,
        "org_id": snapshot.org_id,
        "space_id": snapshot.space_id,
        "version": snapshot.version,
        "content": snapshot.content,
        "summary": snapshot.summary,
        "meta": snapshot.metadata,
        "tags": snapshot.tags,
        "source": snapshot.source,
        "kind": snapshot.kind.value,
        "status": snapshot.status.value,
        "occurred_at": snapshot.occurred_at,
        "valid_from": when,
        "valid_to": None,
        "memory_key": snapshot.key,
    }


def _edge_row(edge: RelationEdge, created_at: datetime | None = None) -> dict[str, Any]:
    return {
        "id": edge.id,
        "org_id": edge.org_id,
        "space_id": edge.space_id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "type": edge.type.value,
        "reason": edge.reason,
        "confidence": edge.confidence,
        "created_at": created_at or edge.created_at,
    }


def compile_ops(ops: Sequence[BulkOp], *, dimensions: int | None = None) -> CompiledBulk:
    """Replay `ops` in memory and return the final rows they amount to.

    The same rules `_upsert_in_session`, `_set_status_in_session` and
    `write_keyed` apply one call at a time:

      * a new row is INSERTed once, in its final state, with every version
        it passed through -- each closed at the next one's `when`;
      * an existing row is UPDATEd to its final state; its open version is
        closed at the first change and the new ones are appended;
      * a write that does not bump `version` overwrites the open snapshot in
        place rather than opening another (the double upsert of a created
        memory is exactly that);
      * chunks are written once, for new rows only: nothing the bulk path
        does to an existing row changes its content, so its chunks stay.
    """
    plan = CompiledBulk()
    created: dict[str, Memory] = {}
    first_written: dict[str, Memory] = {}
    updated: dict[str, Memory] = {}
    chains: dict[str, list[dict[str, Any]]] = {}

    def touch(memory: Memory, when: datetime, *, fresh: bool) -> None:
        mid = memory.id
        if fresh:
            if mid in created or mid in updated:
                raise StoreError(f"memory {mid} written as new twice in one bulk")
            first_written[mid] = memory
            created[mid] = memory
        elif mid in created:
            created[mid] = memory
        else:
            if mid not in updated:
                plan.close_ids.append(mid)
                plan.close_at.append(when)
            updated[mid] = memory
        chain = chains.setdefault(mid, [])
        previous = chain[-1] if chain else None
        if previous is not None and previous["version"] == memory.version:
            keep = {"id": previous["id"], "valid_from": previous["valid_from"]}
            previous.update(_version_row(memory, when))
            previous.update(keep)
        else:
            if previous is not None:
                previous["valid_to"] = when
            chain.append(_version_row(memory, when))

    for op in ops:
        if isinstance(op, UpsertOp):
            touch(op.memory, op.when, fresh=op.fresh)
        elif isinstance(op, KeyedOp):
            if op.replaced is not None:
                touch(op.replaced, op.when, fresh=False)
            touch(op.memory, op.when, fresh=True)
            if op.replaced is not None and op.mode is ReplaceMode.SUPERSEDE:
                plan.edges.append(
                    _edge_row(
                        RelationEdge(
                            org_id=op.memory.org_id,
                            space_id=op.memory.space_id,
                            source_id=op.memory.id,
                            target_id=op.replaced.id,
                            type=RelationType.SUPERSEDES,
                            reason=op.reason,
                            confidence=1.0,
                        ),
                        created_at=op.when,
                    )
                )
        elif isinstance(op, EdgeOp):
            plan.edges.append(_edge_row(op.edge))

    plan.updates = [_row(m) for m in updated.values()]
    plan.inserts = [_row(m) for m in created.values()]
    plan.versions = [row for chain in chains.values() for row in chain]
    for mid, memory in first_written.items():
        for chunk in memory.chunks:
            if (
                dimensions is not None
                and chunk.embedding is not None
                and len(chunk.embedding) != dimensions
            ):
                raise StoreError(
                    f"chunk {chunk.id} has {len(chunk.embedding)} dimensions, "
                    f"index expects {dimensions}"
                )
            plan.chunks.append(
                {
                    "id": chunk.id,
                    "memory_id": mid,
                    "org_id": memory.org_id,
                    "space_id": memory.space_id,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,
                    "token_estimate": chunk.token_estimate,
                    "embedding": chunk.embedding,
                }
            )
    return plan


@asynccontextmanager
async def postgres_bulk_session(
    store: PostgresStore, org_id: str, space_id: str, *, neighbour_limit: int
) -> AsyncIterator[BulkSession]:
    """One transaction for the whole request, tenant-scoped once."""
    if store._ef_search_supported is None or store._iterative_scan_supported is None:
        # Probe outside the transaction: a failed probe rolls back, and a
        # rollback inside `begin()` would end the transaction the bulk needs.
        async with store._session() as probe:
            await store._enable_iterative_scan(probe, wanted=neighbour_limit)
    async with store._session() as session, session.begin():
        await store._scope_for_ann(session, org_id, neighbour_limit)
        yield PostgresBulkSession(store, session, org_id, space_id)


__all__ = ["CHUNK_CAP_MULTIPLE", "CompiledBulk", "PostgresBulkSession", "compile_ops"]
