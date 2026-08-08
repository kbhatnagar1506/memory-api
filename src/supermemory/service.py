"""Application service: the transactional boundary between API and domain.

Route handlers do HTTP; the domain does algorithms; this does the workflow that
joins them. Keeping it here rather than in a route means the same ingest path is
reachable from a worker, a CLI or a test without spinning up FastAPI.

Ingestion is the interesting one:

    validate -> normalize -> exact-dup check -> chunk -> embed
             -> near-dup check -> optional supersession -> persist

Deduplication runs *before* embedding where it can (the exact-hash path),
because embedding is the expensive step and re-ingesting an unchanged document
is the single most common write a memory system sees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .config import Settings
from .core.errors import NotFoundError, PayloadTooLargeError, ValidationError
from .core.logging import get_logger
from .core.metrics import EMBEDDINGS, INGESTED, SEARCH_LATENCY, SEARCH_STAGE_LATENCY
from .domain.chunking import chunk_text, normalize
from .domain.consolidation import (
    DuplicateKind,
    apply_supersession,
    detect_near_duplicate,
    merge_duplicate,
    propose_supersessions,
)
from .domain.embeddings.base import EmbeddingProvider
from .domain.models import (
    Chunk,
    Memory,
    MemoryStatus,
    MemoryVersion,
    Organization,
    RelationEdge,
    RelationType,
    Space,
    utcnow,
)
from .domain.retrieval.pipeline import RetrievalPipeline, SearchRequest, SearchResponse
from .domain.retrieval.rerank import Reranker
from .store.base import MemoryFilter, MemoryStore, Page

log = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    memory: Memory
    created: bool
    duplicate_of: str | None = None
    duplicate_kind: DuplicateKind = DuplicateKind.NONE
    similarity: float = 0.0
    superseded: list[str] = field(default_factory=list)
    chunk_count: int = 0


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self.pipeline = RetrievalPipeline(store, embedder, reranker)

    # -- spaces ------------------------------------------------------------

    async def ensure_organization(self, name: str) -> Organization:
        org = Organization(name=name)
        return await self.store.create_organization(org)

    async def create_space(
        self,
        org_id: str,
        *,
        slug: str,
        name: str,
        description: str = "",
        metadata: dict[str, object] | None = None,
    ) -> Space:
        space = Space(
            org_id=org_id,
            slug=slug,
            name=name,
            description=description,
            metadata=metadata or {},
        )
        return await self.store.create_space(space)

    async def get_space_or_raise(self, org_id: str, space_id: str) -> Space:
        space = await self.store.get_space(org_id, space_id)
        if space is None:
            # A space in another org must be indistinguishable from one that
            # does not exist, or the 404/403 difference leaks its existence.
            raise NotFoundError(f"space {space_id} not found", field="space_id")
        return space

    # -- ingestion ---------------------------------------------------------

    async def ingest(
        self,
        *,
        org_id: str,
        space_id: str,
        content: str,
        summary: str = "",
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] = (),
        source: str = "",
        occurred_at: datetime | None = None,
        dedupe: bool = True,
        auto_supersede: bool = False,
    ) -> IngestResult:
        await self.get_space_or_raise(org_id, space_id)

        cleaned = normalize(content)
        if not cleaned:
            raise ValidationError("content is empty after normalization", field="content")
        size = len(cleaned.encode("utf-8"))
        if size > self.settings.max_content_bytes:
            raise PayloadTooLargeError(
                f"content is {size} bytes, limit is {self.settings.max_content_bytes}",
                field="content",
            )

        # -- exact duplicate: cheapest check, before any embedding ----------
        if dedupe:
            from .domain.models import content_hash

            existing = await self.store.find_by_content_hash(
                org_id, space_id, content_hash(cleaned)
            )
            if existing is not None:
                incoming = Memory(
                    org_id=org_id,
                    space_id=space_id,
                    content=cleaned,
                    metadata=dict(metadata or {}),
                    tags=list(tags),
                    source=source,
                    occurred_at=occurred_at or utcnow(),
                )
                merged = merge_duplicate(existing, incoming)
                await self.store.upsert_memory(merged)
                INGESTED.labels(outcome="duplicate_exact").inc()
                return IngestResult(
                    memory=merged,
                    created=False,
                    duplicate_of=existing.id,
                    duplicate_kind=DuplicateKind.EXACT,
                    similarity=1.0,
                    chunk_count=len(merged.chunks),
                )

        memory = Memory(
            org_id=org_id,
            space_id=space_id,
            content=cleaned,
            summary=summary,
            metadata=dict(metadata or {}),
            tags=list(tags),
            source=source,
            occurred_at=occurred_at or utcnow(),
        )

        # -- chunk and embed --------------------------------------------------
        pieces = chunk_text(
            cleaned,
            target_tokens=self.settings.chunk_target_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        if not pieces:
            raise ValidationError("content produced no chunks", field="content")

        try:
            result = await self.embedder.embed([p.text for p in pieces])
            EMBEDDINGS.labels(provider=self.embedder.name, outcome="ok").inc(len(pieces))
        except Exception:
            EMBEDDINGS.labels(provider=self.embedder.name, outcome="error").inc()
            INGESTED.labels(outcome="embed_error").inc()
            raise

        chunks = [
            Chunk(
                memory_id=memory.id,
                ordinal=piece.ordinal,
                text=piece.text,
                token_estimate=piece.token_estimate,
                embedding=vector,
            )
            for piece, vector in zip(pieces, result.vectors, strict=True)
        ]
        memory = memory.model_copy(update={"chunks": chunks})

        # -- near duplicate ----------------------------------------------------
        if dedupe and chunks:
            candidates = await self.store.sample_embeddings(org_id, space_id, limit=512)
            verdict = detect_near_duplicate(
                chunks[0].embedding or [],
                candidates,
                threshold=self.settings.dedupe_threshold,
            )
            if verdict.is_duplicate and verdict.existing_id:
                existing = await self.store.get_memory(org_id, space_id, verdict.existing_id)
                if existing is not None:
                    merged = merge_duplicate(existing, memory)
                    await self.store.upsert_memory(merged)
                    INGESTED.labels(outcome="duplicate_near").inc()
                    return IngestResult(
                        memory=merged,
                        created=False,
                        duplicate_of=existing.id,
                        duplicate_kind=DuplicateKind.NEAR,
                        similarity=verdict.similarity,
                        chunk_count=len(merged.chunks),
                    )

        # -- supersession ------------------------------------------------------
        superseded: list[str] = []
        if auto_supersede and chunks:
            page = await self.store.list_memories(
                org_id, space_id, filters=MemoryFilter(), limit=256
            )
            pairs = [
                (m, m.chunks[0].embedding)
                for m in page.items
                if m.chunks and m.chunks[0].embedding is not None
            ]
            proposals = propose_supersessions(
                memory,
                chunks[0].embedding or [],
                pairs,
            )
            # The memory must exist before an edge can point at it: the edge
            # has a foreign key to both endpoints.
            memory = await self.store.upsert_memory(memory)
            for proposal in proposals:
                old = await self.store.get_memory(org_id, space_id, proposal.old_id)
                if old is None:
                    continue
                edge, updated_old = apply_supersession(memory, old, proposal)
                await self.store.create_relation(edge)
                await self.store.upsert_memory(updated_old)
                superseded.append(proposal.old_id)

        stored = await self.store.upsert_memory(memory)
        INGESTED.labels(outcome="created").inc()
        return IngestResult(
            memory=stored,
            created=True,
            superseded=superseded,
            chunk_count=len(chunks),
        )

    # -- reads -------------------------------------------------------------

    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory:
        memory = await self.store.get_memory(org_id, space_id, memory_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")
        return memory

    async def delete_memory(self, org_id: str, space_id: str, memory_id: str) -> None:
        if not await self.store.delete_memory(org_id, space_id, memory_id):
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")

    async def list_memories(
        self,
        org_id: str,
        space_id: str,
        *,
        filters: MemoryFilter,
        limit: int,
        cursor: str | None,
    ) -> Page:
        await self.get_space_or_raise(org_id, space_id)
        return await self.store.list_memories(
            org_id, space_id, filters=filters, limit=limit, cursor=cursor
        )

    async def link(
        self,
        org_id: str,
        space_id: str,
        *,
        source_id: str,
        target_id: str,
        relation: RelationType,
        reason: str = "",
    ) -> RelationEdge:
        """Create a typed relation edge, applying its side effects.

        Returns the edge, not the source memory: the source memory is
        unchanged by this operation. Writing an edge is a statement about the
        graph, and bumping the source's version for it would produce a version
        history full of entries whose content is identical.
        """
        if source_id == target_id:
            raise ValidationError("a memory cannot relate to itself", field="target_id")
        # Both endpoints must exist and be visible to this tenant. Checked
        # before any write so a relation can never dangle.
        await self.get_memory(org_id, space_id, source_id)
        target = await self.get_memory(org_id, space_id, target_id)

        edge = await self.store.create_relation(
            RelationEdge(
                org_id=org_id,
                space_id=space_id,
                source_id=source_id,
                target_id=target_id,
                type=relation,
                reason=reason,
            )
        )

        if relation is RelationType.SUPERSEDES:
            if target.status is not MemoryStatus.SUPERSEDED:
                await self.store.upsert_memory(
                    target.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "version": target.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
        elif relation.symmetric:
            # `contradicts` is mutual; recording one direction only would make
            # the answer depend on which memory you happened to look at.
            # create_relation is idempotent, so the reverse edge is safe to
            # write unconditionally.
            await self.store.create_relation(
                RelationEdge(
                    org_id=org_id,
                    space_id=space_id,
                    source_id=target_id,
                    target_id=source_id,
                    type=relation,
                    reason=reason,
                )
            )
        return edge

    async def list_relations(
        self, org_id: str, space_id: str, memory_id: str, *, direction: str = "out"
    ) -> list[RelationEdge]:
        await self.get_memory(org_id, space_id, memory_id)
        return await self.store.list_relations(org_id, space_id, memory_id, direction=direction)

    async def get_lineage(
        self, org_id: str, space_id: str, memory_id: str
    ) -> dict[str, object]:
        """The supersession lineage around a memory, in both directions.

        `ancestors` is what this memory replaced, oldest last. `successors` is
        what replaced it — non-empty means this memory is stale, and the last
        entry is the current head of truth.
        """
        await self.get_memory(org_id, space_id, memory_id)
        backward = await self.store.walk_supersession_chain(
            org_id, space_id, memory_id, direction="backward"
        )
        forward = await self.store.walk_supersession_chain(
            org_id, space_id, memory_id, direction="forward"
        )
        return {
            "memory_id": memory_id,
            "ancestors": [e.target_id for e in backward],
            "successors": [e.source_id for e in forward],
            "is_current": not forward,
            "head": forward[-1].source_id if forward else memory_id,
        }

    async def get_memory_as_of(
        self, org_id: str, space_id: str, memory_id: str, as_of: datetime
    ) -> Memory:
        memory = await self.store.get_memory_as_of(org_id, space_id, memory_id, as_of)
        if memory is None:
            raise NotFoundError(
                f"memory {memory_id} did not exist at {as_of.isoformat()}",
                field="as_of",
            )
        return memory

    async def list_memory_versions(
        self, org_id: str, space_id: str, memory_id: str
    ) -> list[MemoryVersion]:
        versions = await self.store.list_memory_versions(org_id, space_id, memory_id)
        if not versions:
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")
        return versions

    async def erase_memory(
        self, org_id: str, space_id: str, memory_id: str
    ) -> dict[str, object]:
        """Right-to-erasure purge with an attestation of what was destroyed.

        Unlike delete (which preserves the audit trail), erase removes the
        content everywhere it can be reached: live row, chunks and embeddings,
        every edge touching it, and the full version history — so point-in-time
        reads cannot resurrect it. The attestation records the content hash
        (proof of WHICH content was destroyed, without retaining the content),
        counts of everything purged, and the ids of memories that had claimed
        derivation from the erased one, so a compliance process can review the
        blast radius.

        Works on live AND already-deleted memories: delete leaves history
        behind by design, and an erasure request must be able to purge that
        residue too.
        """
        memory = await self.store.get_memory(org_id, space_id, memory_id)
        versions = await self.store.list_memory_versions(org_id, space_id, memory_id)
        if memory is None and not versions:
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")

        # The hash must be captured BEFORE the purge; afterwards there is
        # nothing left to hash. Prefer the live row, fall back to the last
        # historical snapshot for the deleted-but-remembered case.
        if memory is not None:
            digest = memory.content_sha256
        else:
            from .domain.models import content_hash

            digest = content_hash(versions[-1].content)

        derivatives: list[str] = []
        if memory is not None:
            incoming = await self.store.list_relations(
                org_id, space_id, memory_id, direction="in"
            )
            derivatives = sorted(
                {e.source_id for e in incoming if e.type is RelationType.DERIVED_FROM}
            )

        report = await self.store.erase_memory(org_id, space_id, memory_id)
        INGESTED.labels(outcome="erased").inc()
        log.info(
            "memory_erased",
            memory_id=memory_id,
            versions_purged=report.versions_purged,
            edges_removed=report.edges_removed,
        )
        return {
            "memory_id": memory_id,
            "space_id": space_id,
            "content_sha256": digest,
            "chunks_removed": report.chunks_removed,
            "edges_removed": report.edges_removed,
            "versions_purged": report.versions_purged,
            "derived_memories_affected": derivatives,
            "erased_at": utcnow().isoformat(),
        }

    # -- search ------------------------------------------------------------

    async def search(self, request: SearchRequest) -> SearchResponse:
        await self.get_space_or_raise(request.org_id, request.space_id)
        with SEARCH_LATENCY.time():
            response = await self.pipeline.search(request)
        for stage, ms in response.timings_ms.items():
            SEARCH_STAGE_LATENCY.labels(stage=stage.removesuffix("_ms")).observe(ms / 1000.0)
        return response


__all__ = ["IngestResult", "MemoryService"]
