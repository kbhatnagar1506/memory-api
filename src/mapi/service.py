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
from .core.errors import (
    NotFoundError,
    PayloadTooLargeError,
    ProviderError,
    ValidationError,
)
from .core.logging import get_logger
from .core.metrics import EMBEDDINGS, INGESTED, SEARCH_LATENCY, SEARCH_STAGE_LATENCY
from .domain.chunking import chunk_text, normalize
from .domain.consolidation import (
    DuplicateKind,
    apply_supersession,
    detect_near_duplicate,
    merge_duplicate,
    propose_contradictions,
    propose_supersessions,
)
from .domain.embeddings.base import EmbeddingProvider
from .domain.models import (
    Chunk,
    Memory,
    MemoryKind,
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
from .domain.synthesis import classify
from .domain.synthesis.derive import CompleteFn, DerivedAnswer, SourceDoc, derive_answer
from .store.base import MemoryFilter, MemoryStore, Page

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """A memory resolved together with its whole relation neighborhood.

    `current_head` is non-empty exactly when `is_current` is false. Neighbors
    that no longer resolve (erased) are omitted, not stubbed.
    """

    memory: Memory
    is_current: bool
    current_head: list[Memory]
    replaced: list[Memory]
    derived_from: list[Memory]
    derivatives: list[Memory]
    references: list[Memory]
    contradicts: list[Memory]


@dataclass(slots=True)
class IngestResult:
    memory: Memory
    created: bool
    duplicate_of: str | None = None
    duplicate_kind: DuplicateKind = DuplicateKind.NONE
    similarity: float = 0.0
    superseded: list[str] = field(default_factory=list)
    #: Memories this write is judged to CONTRADICT. Recorded as edges and
    #: reported; never suppressed, because either side may be the true one.
    contradicts: list[str] = field(default_factory=list)
    chunk_count: int = 0


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        settings: Settings,
        completer: CompleteFn | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        #: Async prompt -> text, or None when synthesis is off. Injected so
        #: tests drive derivation deterministically and the service never
        #: imports a vendor SDK.
        self.completer = completer
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
        detect_conflicts: bool = False,
        kind: MemoryKind = MemoryKind.EPISODIC,
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
            kind=kind,
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

        # -- belief revision ---------------------------------------------------
        # Two different operations that both need the same candidate scan:
        # supersession (revision -- newer replaces older) and contradiction
        # (disagreement -- two ACTIVE memories that cannot both be true).
        #
        # Both are gated, but for different reasons. Supersession is gated on
        # SAFETY: it hides a memory, and hiding a user's data because a
        # similarity score crossed a threshold has an invisible failure mode.
        # Contradiction hides nothing, so it is gated purely on COST -- it
        # needs a candidate scan, and a cheap write path is the architectural
        # bet of this whole system. Enable it per request or per space.
        superseded: list[str] = []
        conflicts: list[str] = []
        if chunks and (auto_supersede or detect_conflicts):
            page = await self.store.list_memories(
                org_id, space_id, filters=MemoryFilter(), limit=256, cursor=None
            )
            pairs = [
                (m, m.chunks[0].embedding)
                for m in page.items
                if m.chunks and m.chunks[0].embedding is not None
            ]
            proposals = (
                propose_supersessions(memory, chunks[0].embedding or [], pairs)
                if auto_supersede
                else []
            )
            conflict_proposals = (
                propose_contradictions(memory, chunks[0].embedding or [], pairs)
                if detect_conflicts
                else []
            )
            # The memory must exist before an edge can point at it: the edge
            # has a foreign key to both endpoints.
            memory = await self.store.upsert_memory(memory)

            for conflict in conflict_proposals:
                # `contradicts` is symmetric, so both directions are written --
                # otherwise the answer to "does this conflict with anything"
                # would depend on which memory you happened to look up first.
                # Neither side is superseded or hidden: either may be true.
                for source_id, target_id in (
                    (memory.id, conflict.right_id),
                    (conflict.right_id, memory.id),
                ):
                    await self.store.create_relation(
                        RelationEdge(
                            org_id=org_id,
                            space_id=space_id,
                            source_id=source_id,
                            target_id=target_id,
                            type=RelationType.CONTRADICTS,
                            reason=conflict.reason,
                            confidence=conflict.confidence,
                        )
                    )
                conflicts.append(conflict.right_id)

            for proposal in proposals:
                old = await self.store.get_memory(org_id, space_id, proposal.old_id)
                if old is None:
                    continue
                edge, updated_old = apply_supersession(memory, old, proposal)
                await self.store.create_relation(edge)
                await self.store.upsert_memory(updated_old)
                superseded.append(proposal.old_id)
                # A superseded source invalidates what was computed from it:
                # the derivation may now describe a replaced state of the world.
                await self._mark_derivations_stale(
                    org_id,
                    space_id,
                    await self._direct_derivatives(org_id, space_id, proposal.old_id),
                )

        stored = await self.store.upsert_memory(memory)
        INGESTED.labels(outcome="created").inc()
        return IngestResult(
            memory=stored,
            created=True,
            superseded=superseded,
            contradicts=conflicts,
            chunk_count=len(chunks),
        )

    # -- reads -------------------------------------------------------------

    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory:
        memory = await self.store.get_memory(org_id, space_id, memory_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")
        return memory

    async def _mark_derivations_stale(
        self, org_id: str, space_id: str, source_ids: Sequence[str]
    ) -> list[str]:
        """Transitively mark every derivation of `source_ids` STALE.

        The invariant this enforces: a derivation must never outlive the
        evidence it was computed from. Erase a source episode and "you own 3
        bikes" derived from it is no longer known — serving it anyway would be
        a lie with provenance attached. STALE (not deletion) is the right
        response because the derivation is *recomputable* from the surviving
        sources, and its `derived_from` edges say exactly how.

        Transitive because derivations stack (a profile fact derived from a
        timeline derived from episodes): BFS over incoming DERIVED_FROM edges
        with a visited set, so diamond-shaped provenance terminates. Edges to
        the *removed* memory die with it in both backends, so callers collect
        the first hop BEFORE removal and pass it in; hops beyond the first are
        walked here, where the edges still exist.

        Only ACTIVE rows transition — a SUPERSEDED derivation stays superseded
        (its replacement is the current truth; do not resurrect it as merely
        stale). Returns the ids actually transitioned, for attestations.
        """
        staled: list[str] = []
        queue = list(dict.fromkeys(source_ids))
        visited: set[str] = set()
        while queue:
            derived_id = queue.pop(0)
            if derived_id in visited:
                continue
            visited.add(derived_id)
            derived = await self.store.get_memory(org_id, space_id, derived_id)
            if derived is None:
                continue
            if derived.status is MemoryStatus.ACTIVE:
                await self.store.upsert_memory(
                    derived.model_copy(
                        update={
                            "status": MemoryStatus.STALE,
                            "version": derived.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
                staled.append(derived_id)
            incoming = await self.store.list_relations(
                org_id, space_id, derived_id, direction="in"
            )
            queue.extend(e.source_id for e in incoming if e.type is RelationType.DERIVED_FROM)
        return staled

    async def _direct_derivatives(
        self, org_id: str, space_id: str, memory_id: str
    ) -> list[str]:
        """First-hop derivations of a memory, collected while its edges exist."""
        incoming = await self.store.list_relations(org_id, space_id, memory_id, direction="in")
        return sorted({e.source_id for e in incoming if e.type is RelationType.DERIVED_FROM})

    async def delete_memory(self, org_id: str, space_id: str, memory_id: str) -> None:
        # Collect BEFORE the delete: the edges naming the derivations die with
        # the memory (FK cascade on postgres, edge sweep in-memory).
        derivatives = await self._direct_derivatives(org_id, space_id, memory_id)
        if not await self.store.delete_memory(org_id, space_id, memory_id):
            raise NotFoundError(f"memory {memory_id} not found", field="memory_id")
        await self._mark_derivations_stale(org_id, space_id, derivatives)

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
                await self._mark_derivations_stale(
                    org_id,
                    space_id,
                    await self._direct_derivatives(org_id, space_id, target_id),
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

    async def get_memory_context(
        self, org_id: str, space_id: str, memory_id: str, *, limit_per_relation: int = 20
    ) -> MemoryContext:
        """The full relation neighborhood of one memory, resolved to content.

        This is the "entire context" read: one call answers what an agent
        otherwise assembles from four — the memory itself, whether it is still
        current (and what replaced it), what it replaced, what it was derived
        from, what was derived from it, and what it references or contradicts.

        Neighbors whose memory no longer resolves (erased, or outside this
        tenant's view) are silently omitted rather than surfaced as broken
        stubs: an erased source must not leak even its existence through the
        context of a surviving derivative.
        """
        memory = await self.get_memory(org_id, space_id, memory_id)
        lineage = await self.get_lineage(org_id, space_id, memory_id)

        async def resolve(ids: list[str]) -> list[Memory]:
            out: list[Memory] = []
            for candidate in ids[:limit_per_relation]:
                found = await self.store.get_memory(org_id, space_id, candidate)
                if found is not None:
                    out.append(found)
            return out

        outgoing = await self.store.list_relations(org_id, space_id, memory_id, direction="out")
        incoming = await self.store.list_relations(org_id, space_id, memory_id, direction="in")

        def ids_of(edges: list[RelationEdge], kind: RelationType, *, source: bool) -> list[str]:
            return [(e.source_id if source else e.target_id) for e in edges if e.type is kind]

        is_current = bool(lineage["is_current"])
        return MemoryContext(
            memory=memory,
            is_current=is_current,
            current_head=[] if is_current else await resolve([str(lineage["head"])]),
            replaced=await resolve(ids_of(outgoing, RelationType.SUPERSEDES, source=False)),
            derived_from=await resolve(
                ids_of(outgoing, RelationType.DERIVED_FROM, source=False)
            ),
            derivatives=await resolve(ids_of(incoming, RelationType.DERIVED_FROM, source=True)),
            references=await resolve(ids_of(outgoing, RelationType.REFERENCES, source=False)),
            contradicts=await resolve(
                ids_of(outgoing, RelationType.CONTRADICTS, source=False)
                + ids_of(incoming, RelationType.CONTRADICTS, source=True)
            ),
        )

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
            derivatives = await self._direct_derivatives(org_id, space_id, memory_id)

        report = await self.store.erase_memory(org_id, space_id, memory_id)
        # The derivations named in the attestation are not merely "affected":
        # they are marked STALE, transitively, so nothing computed from the
        # erased content is ever served as current truth again.
        await self._mark_derivations_stale(org_id, space_id, derivatives)
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
            "edges_bridged": report.edges_bridged,
            "versions_purged": report.versions_purged,
            "derived_memories_affected": derivatives,
            "erased_at": utcnow().isoformat(),
        }

    # -- derivation and profiles -------------------------------------------

    async def derive(
        self,
        org_id: str,
        space_id: str,
        *,
        question: str,
        k: int = 10,
        materialize: bool = False,
        bucket: str | None = None,
    ) -> tuple[DerivedAnswer, Memory | None]:
        """Compute an answer from episodes; optionally store it as a memory.

        The read half runs map -> ground -> reduce over the top-k retrieved
        episodes (code does the arithmetic; every table row carries a verbatim
        quote checked against its source). The write half — `materialize` —
        stores the answer as a `kind=derived` memory with `derived_from` edges
        to every source, which buys three things nothing else in the market
        has together:

          * provenance: the answer names the episodes that made it true;
          * a demand-driven index: the next similar question is a lookup, and
            a wrong guess cost nothing because the episodes are still there;
          * invalidation: erase/supersede a source and the stored fact goes
            STALE via the existing lifecycle — it can never outlive its
            evidence.

        `bucket` tags the fact into a profile ("preferences", "dietary", ...).
        Re-deriving the same bucket+question SUPERSEDES the previous fact, so
        profiles update through the same revision machinery as everything
        else — history, `?as_of=`, and lineage included.
        """
        if self.completer is None:
            raise ProviderError(
                "no synthesis backend configured; set synthesis_backend=gemini "
                "or inject a completer"
            )
        response = await self.search(
            SearchRequest(query=question, org_id=org_id, space_id=space_id, limit=k)
        )
        docs = [
            SourceDoc(
                id=hit.memory.id,
                text=hit.memory.content,
                occurred_at=hit.memory.occurred_at,
            )
            for hit in response.results
        ]
        kind = classify(question)
        derived = await derive_answer(question, kind, docs, self.completer)
        if derived is None:
            derived = DerivedAnswer(
                answer="", kind=kind, table=(), source_ids=(), computed=False
            )
        if not materialize or not derived.answer:
            return derived, None

        tags = [f"profile:{bucket}"] if bucket else []
        result = await self.ingest(
            org_id=org_id,
            space_id=space_id,
            content=derived.answer,
            summary=question,
            metadata={
                "question": question,
                "derive_kind": str(derived.kind),
                "computed": derived.computed,
                "table": [
                    {
                        "date": row.date.isoformat() if row.date else None,
                        "fact": row.fact,
                        "source_id": row.source_id,
                    }
                    for row in derived.table
                ],
            },
            tags=tags,
            source="derive",
            kind=MemoryKind.DERIVED,
            # A derived fact is not a duplicate of the episodes it summarises,
            # and near-dup collapsing against them would eat the derivation.
            dedupe=False,
        )
        stored = result.memory
        for source_id in derived.source_ids:
            await self.store.create_relation(
                RelationEdge(
                    org_id=org_id,
                    space_id=space_id,
                    source_id=stored.id,
                    target_id=source_id,
                    type=RelationType.DERIVED_FROM,
                    reason="derive: grounded extraction",
                )
            )
        # Same bucket + same question -> this fact replaces the previous one,
        # through the normal supersession machinery (status flip + edge), so
        # profile history is ordinary memory history.
        if bucket:
            previous = await self.list_memories(
                org_id,
                space_id,
                filters=MemoryFilter(tags=(f"profile:{bucket}".casefold(),)),
                limit=50,
                cursor=None,
            )
            for old in previous.items:
                if old.id == stored.id or old.kind is not MemoryKind.DERIVED:
                    continue
                if old.metadata.get("question") != question:
                    continue
                await self.store.create_relation(
                    RelationEdge(
                        org_id=org_id,
                        space_id=space_id,
                        source_id=stored.id,
                        target_id=old.id,
                        type=RelationType.SUPERSEDES,
                        reason="re-derived profile fact",
                    )
                )
                await self.store.upsert_memory(
                    old.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "version": old.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
        return derived, stored

    async def refresh_stale_derivations(
        self, org_id: str, space_id: str, *, budget: int = 20
    ) -> dict[str, object]:
        """Re-derive facts whose sources changed. Consolidation, done safely.

        Phase 2 marks a derivation STALE when any source is erased, deleted or
        superseded, but nothing re-computed it -- an open loop that left
        profiles permanently empty after the first invalidation. This closes
        it: find STALE derived memories, re-run the question that produced
        them against the surviving evidence, and supersede the stale fact with
        the fresh one.

        This is the same machinery competitors run as a "dream cycle", with the
        destructive half removed. Theirs merges facts into cleaner
        abstractions, resolves contradictions by newest timestamp, and deletes
        on `forgetAfter`. Ours only ADDS a recomputed fact and supersedes the
        one it replaces -- episodes are never touched, so a bad consolidation
        costs a re-derivation rather than the evidence.

        `budget` bounds LLM spend per call. Background consolidation is exactly
        how competitors' token bills became marketing liabilities, so the cost
        is capped and attributable rather than ambient.

        Safe to call repeatedly: a derivation that cannot be re-derived (its
        sources are gone entirely) stays STALE rather than being deleted or
        silently resurrected.
        """
        if self.completer is None:
            raise ProviderError("no synthesis backend configured")
        await self.get_space_or_raise(org_id, space_id)

        page = await self.store.list_memories(
            org_id,
            space_id,
            filters=MemoryFilter(statuses=frozenset({MemoryStatus.STALE})),
            limit=max(budget * 4, budget),
            cursor=None,
        )
        stale = [m for m in page.items if m.kind is MemoryKind.DERIVED][:budget]

        refreshed: list[str] = []
        abandoned: list[str] = []
        for memory in stale:
            question = str(memory.metadata.get("question") or memory.summary or "")
            if not question:
                # Nothing records what this fact answered, so it cannot be
                # recomputed. Leave it stale rather than guess.
                abandoned.append(memory.id)
                continue
            bucket = next(
                (t.split(":", 1)[1] for t in memory.tags if t.startswith("profile:")), None
            )
            try:
                _, stored = await self.derive(
                    org_id, space_id, question=question, materialize=True, bucket=bucket
                )
            except Exception:
                abandoned.append(memory.id)
                continue
            if stored is None:
                # The surviving evidence no longer supports an answer. Correct
                # outcome: the fact stays stale and the profile stays silent.
                abandoned.append(memory.id)
                continue
            if bucket is None:
                # Bucketed facts are superseded inside `derive`; unbucketed
                # ones need the link written here so lineage stays walkable.
                await self.store.create_relation(
                    RelationEdge(
                        org_id=org_id,
                        space_id=space_id,
                        source_id=stored.id,
                        target_id=memory.id,
                        type=RelationType.SUPERSEDES,
                        reason="re-derived after its sources changed",
                    )
                )
            current = await self.store.get_memory(org_id, space_id, memory.id)
            if current is not None and current.status is MemoryStatus.STALE:
                await self.store.upsert_memory(
                    current.model_copy(
                        update={
                            "status": MemoryStatus.SUPERSEDED,
                            "version": current.version + 1,
                            "updated_at": utcnow(),
                        }
                    )
                )
            refreshed.append(stored.id)

        return {
            "examined": len(stale),
            "refreshed": refreshed,
            "abandoned": abandoned,
            "budget": budget,
        }

    async def get_profile(
        self, org_id: str, space_id: str, bucket: str, *, limit: int = 100
    ) -> list[Memory]:
        """Current facts in one profile bucket.

        ACTIVE only, on purpose: superseded facts are history (visible via
        versions/lineage), stale facts are pending re-derivation, and neither
        is something an agent should act on as current truth.
        """
        await self.get_space_or_raise(org_id, space_id)
        page = await self.list_memories(
            org_id,
            space_id,
            filters=MemoryFilter(tags=(f"profile:{bucket}".casefold(),)),
            limit=limit,
            cursor=None,
        )
        return [m for m in page.items if m.kind is MemoryKind.DERIVED]

    # -- search ------------------------------------------------------------

    async def search(self, request: SearchRequest) -> SearchResponse:
        await self.get_space_or_raise(request.org_id, request.space_id)
        with SEARCH_LATENCY.time():
            response = await self.pipeline.search(request)
        for stage, ms in response.timings_ms.items():
            SEARCH_STAGE_LATENCY.labels(stage=stage.removesuffix("_ms")).observe(ms / 1000.0)
        return response


__all__ = ["IngestResult", "MemoryService"]
