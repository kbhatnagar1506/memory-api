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
from datetime import UTC, datetime
from typing import Any

from ..domain.embeddings.base import Vector
from ..domain.models import (
    ApiKey,
    Membership,
    Memory,
    MemoryKind,
    MemoryStatus,
    MemoryVersion,
    Organization,
    RelationEdge,
    RelationType,
    Space,
    User,
)


@dataclass(frozen=True, slots=True)
class VectorHit:
    memory_id: str
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class TenantUsage:
    """What one organization is currently consuming.

    One object because the three numbers are read together, by the quota
    check, on writes -- three round trips to answer one question would make
    the feature cost more than it saves.
    """

    memories: int
    bytes_stored: int
    writes_today: int


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
class EraseReport:
    """What `erase_memory` actually destroyed. The raw material of an attestation."""

    existed: bool
    chunks_removed: int = 0
    edges_removed: int = 0
    versions_purged: int = 0
    #: SUPERSEDES edges re-created around the removed memory so revision chains
    #: survive it. Removing the middle of A->B->C used to sever the path from A
    #: to C, and C — a fact the user explicitly replaced — resurfaced as
    #: current. The bridge (A->C) carries only the two surviving ids, never the
    #: removed memory's content, so erasure compliance is unaffected.
    edges_bridged: int = 0


@dataclass(frozen=True, slots=True)
class OrgPurgeReport:
    """What `purge_org` destroyed, counted per table.

    Counted rather than just done because the post-event purge is a promise to
    attendees, and "the command exited 0" is not evidence that 0 rows remain.
    `memory_versions` is its own count on purpose: it has no foreign key to
    `memories`, so it is the table a naive purge leaves full of content.
    """

    spaces: int = 0
    memories: int = 0
    chunks: int = 0
    edges: int = 0
    versions: int = 0


@dataclass(frozen=True, slots=True)
class MemoryFilter:
    """Filters applied by every listing and search operation."""

    statuses: frozenset[MemoryStatus] = field(
        default_factory=lambda: frozenset({MemoryStatus.ACTIVE})
    )
    #: Memory kinds to include. Empty means every kind.
    #:
    #: Measured need: write-time extraction stores claims alongside episodes,
    #: and four LongMemEval arms agree the loss comes from claims being
    #: RETRIEVED, not stored -- `only` retrieved better (0.950 vs 0.948) and
    #: answered worse (0.762 vs 0.781). Excluding them from the window is the
    #: configuration that keeps the graph without paying for it at answer time.
    kinds: frozenset[MemoryKind] = frozenset()
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
        if self.kinds and memory.kind not in self.kinds:
            return False
        if self.tags and not set(self.tags).issubset(set(memory.tags)):
            return False
        for key, value in self.metadata:
            if memory.metadata.get(key) != value:
                return False
        occurred = _epoch(memory.occurred_at)
        if self.occurred_after is not None and occurred < _epoch(self.occurred_after):
            return False
        if self.occurred_before is not None and occurred > _epoch(self.occurred_before):
            return False
        return not (self.source is not None and memory.source != self.source)


def _epoch(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
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

    # -- identity ----------------------------------------------------------
    #
    # Users are NOT scoped by org: a person exists before they belong to
    # anything, and belongs to several. Every other read in this interface
    # takes org_id first precisely because it must not cross a tenant
    # boundary; these three are the deliberate exception, and the boundary
    # they respect instead is the user's own id.

    @abc.abstractmethod
    async def upsert_user(self, user: User) -> User: ...

    @abc.abstractmethod
    async def get_user(self, user_id: str) -> User | None: ...

    @abc.abstractmethod
    async def get_user_by_google_sub(self, google_sub: str) -> User | None: ...

    @abc.abstractmethod
    async def create_membership(self, membership: Membership) -> Membership: ...

    @abc.abstractmethod
    async def list_memberships(self, user_id: str) -> list[Membership]: ...

    @abc.abstractmethod
    async def get_membership(self, user_id: str, org_id: str) -> Membership | None: ...

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
    async def upsert_memory(self, memory: Memory, *, now: datetime | None = None) -> Memory:
        """Insert or replace by id. Chunks and embeddings are replaced wholesale.

        Also records a `MemoryVersion` snapshot: closes the previous version's
        `valid_to` (if one exists) and opens a new one at `now`. Callers never
        need to remember to version separately — every write is bitemporal by
        construction. `now` defaults to the current time; it is a parameter
        only so tests can pin it.
        """

    @abc.abstractmethod
    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory | None: ...

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

    # -- relations: a typed, indexed graph ----------------------------------

    @abc.abstractmethod
    async def create_relation(self, edge: RelationEdge) -> RelationEdge:
        """Idempotent: an identical (source, target, type) edge is returned,
        not duplicated, so a retried request cannot create parallel edges."""

    @abc.abstractmethod
    async def list_relations(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        direction: str = "out",
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        """Edges touching `memory_id`. `direction` is "out" (memory_id is the
        source) or "in" (memory_id is the target, i.e. a reverse lookup)."""

    @abc.abstractmethod
    async def get_relations_between(
        self,
        org_id: str,
        space_id: str,
        memory_ids: Sequence[str],
        *,
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        """Edges whose source AND target are both within `memory_ids`.

        The induced subgraph over a result set. Note what this does NOT give
        you: transitive reachability. If A supersedes B supersedes C and only
        A and C are in `memory_ids`, this returns nothing, because neither
        edge has both endpoints in the set. Suppression that needs the
        transitive answer must walk the graph per candidate — see
        `reachable_superseders` — and this method is the cheap bulk prefilter,
        not the whole answer.
        """

    async def walk_supersession_chain(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        direction: str = "backward",
        max_depth: int = 20,
    ) -> list[RelationEdge]:
        """The supersession lineage starting at `memory_id`.

        "backward" (default) follows what `memory_id` supersedes, transitively
        — its ancestry, oldest facts it replaced. "forward" follows what
        supersedes `memory_id`, transitively — the path to the current head of
        truth.

        Concrete on the ABC, built once from `list_relations`, rather than
        duplicated per backend: a graph walk re-implemented in SQL and in
        Python is two places for cycle/depth-guard logic to quietly diverge.
        Depth-bounded and cycle-guarded because a malformed edge set (which
        should never happen, but "should never happen" is not a guarantee)
        must not hang a request.

        Follows a single strand: if a memory supersedes more than one target
        (a consolidation, not a simple replacement) this walks only the oldest
        edge at each step and the rest are not a "chain" at all but a DAG.
        Callers who need the full fan-out should use `list_relations` or
        `get_relations_between` directly rather than this convenience walker.
        """
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        list_direction = "out" if direction == "backward" else "in"
        chain: list[RelationEdge] = []
        visited: set[str] = {memory_id}
        current = memory_id
        for _ in range(max_depth):
            edges = await self.list_relations(
                org_id,
                space_id,
                current,
                direction=list_direction,
                type=RelationType.SUPERSEDES,
            )
            if not edges:
                break
            edge = edges[0]
            nxt = edge.target_id if direction == "backward" else edge.source_id
            if nxt in visited:
                break
            chain.append(edge)
            visited.add(nxt)
            current = nxt
        return chain

    async def reachable_superseders(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        max_depth: int = 20,
    ) -> set[str]:
        """Every memory that transitively supersedes `memory_id`.

        A breadth-first walk backwards along incoming SUPERSEDES edges, so
        A->B->C returns {A, B} for C even when B was never retrieved. This is
        the correct basis for suppression: "is this fact stale" is a question
        about the whole graph, not about which neighbours happened to match
        the query.

        Breadth-first over a `seen` set rather than a single strand, because
        a memory can be superseded by more than one successor (two people
        correcting the same fact independently) and any of them makes it
        stale.
        """
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        seen: set[str] = set()
        frontier = [memory_id]
        for _ in range(max_depth):
            if not frontier:
                break
            nxt: list[str] = []
            for current in frontier:
                edges = await self.list_relations(
                    org_id,
                    space_id,
                    current,
                    direction="in",
                    type=RelationType.SUPERSEDES,
                )
                for edge in edges:
                    if edge.source_id not in seen and edge.source_id != memory_id:
                        seen.add(edge.source_id)
                        nxt.append(edge.source_id)
            frontier = nxt
        return seen

    # -- bitemporal history --------------------------------------------------

    @abc.abstractmethod
    async def get_memory_as_of(
        self, org_id: str, space_id: str, memory_id: str, as_of: datetime
    ) -> Memory | None:
        """The memory's field values as our database understood them at `as_of`
        (system time). Returns None if the memory did not exist yet at that
        time. Chunks are not versioned — see `MemoryVersion`."""

    @abc.abstractmethod
    async def list_memory_versions(
        self, org_id: str, space_id: str, memory_id: str
    ) -> list[MemoryVersion]:
        """Every version of a memory, oldest first."""

    @abc.abstractmethod
    async def erase_memory(self, org_id: str, space_id: str, memory_id: str) -> EraseReport:
        """Compliance-grade purge, as distinct from `delete_memory`.

        `delete_memory` removes the live row and its edges but PRESERVES the
        version history — the audit-trail default, where history must outlive
        the row it describes. `erase_memory` is the right-to-erasure path: it
        removes the live row, its chunks and embeddings, every edge touching
        it, and every version snapshot, so the content is unrecoverable even
        through `get_memory_as_of` — the reconstructed past is scrubbed too.
        An erasure that survives point-in-time reads is not an erasure.

        Returns an EraseReport of exactly what was destroyed, so the caller
        can build an attestation. `existed=False` when the memory (and any
        residual history) was absent — erase is idempotent by design, since a
        retried compliance request must not fail on the second attempt.
        """

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
    async def tenant_usage(self, org_id: str, *, since: datetime) -> TenantUsage:
        """Memory count, bytes stored, and writes since `since`, for one org.

        Spans every space in the organization, because quotas are billed to
        the organization and a per-space limit is trivially escaped by
        creating another space.

        Called only when a quota is actually configured -- see `Quota.enforced`
        -- so a deployment with no limits pays nothing for this.
        """

    @abc.abstractmethod
    async def neighbours(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        exclude_id: str = "",
    ) -> list[tuple[Memory, Vector]]:
        """Memories nearest `embedding`, with the embedding that matched.

        The write path's consolidation checks -- near-duplicate, supersession,
        contradiction -- all ask the same question: which existing memories
        are close enough to this one to be about the same thing. This answers
        it with the ANN index.

        It replaces a `list_memories(limit=256)` scan that was wrong in two
        directions at once. It fetched the 256 NEWEST memories and their full
        embeddings on every write -- about 1.5MB of float32 across the wire
        for a 768-dimension model -- and past 256 memories in a space, a fact
        stated earlier could never again be superseded or contradicted, because
        it was never among the rows compared. Nearest is both cheaper and the
        question actually being asked.

        Returns (memory, embedding) rather than (memory, score) so the pure
        consolidation functions keep computing their own similarity. They are
        the tested surface; making them trust a number from the store would
        move that arithmetic into two backends that must then agree forever.
        """

    @abc.abstractmethod
    async def sample_embeddings(
        self, org_id: str, space_id: str, *, limit: int
    ) -> list[tuple[str, Vector]]:
        """(memory_id, embedding) pairs, for dedup and supersession checks."""

    # -- listing and operator helpers ----------------------------------------
    #
    # Kept together at the end: listing counts for the spaces endpoint, and
    # the organization-level operations behind `mapi admin`, which run from an
    # operator's shell rather than from any request.

    async def count_memories_by_space(self, org_id: str) -> dict[str, int]:
        """Active-memory count for every space in an org, keyed by space id.

        What `GET /v1/spaces` shows. It called `count_memories` once per space,
        a query per row of the listing, so an org holding one space per
        attendee paid a thousand round trips to list itself. Backends override
        this with one grouped query; this default is the reference semantics
        (`MemoryFilter()`: active memories only), and spaces holding nothing
        may be absent from the map.
        """
        return {
            space.id: await self.count_memories(org_id, space.id, filters=MemoryFilter())
            for space in await self.list_spaces(org_id)
        }

    @abc.abstractmethod
    async def list_organizations(self) -> list[Organization]:
        """Every organization, oldest id first. Operator use only: this is the
        one read in the interface that crosses tenants, and no route calls it."""

    @abc.abstractmethod
    async def rename_organization(self, org_id: str, name: str) -> Organization | None:
        """Set an organization's display name. None when it does not exist."""

    @abc.abstractmethod
    async def purge_org(
        self, org_id: str, *, keep_space_ids: frozenset[str] = frozenset()
    ) -> OrgPurgeReport:
        """Destroy every space in an org and everything in them, in one step.

        Removes memories, chunks, relation edges and version history, then the
        spaces themselves, except the spaces in `keep_space_ids`. History is
        purged by org and space rather than by memory, so versions left behind
        by earlier deletes (which preserve history by design) go too.

        The organization row and its API keys survive: this erases what the
        org stored, not the org. Idempotent -- a second run reports zeros.
        """


__all__ = [
    "EraseReport",
    "LexicalHit",
    "MemoryFilter",
    "MemoryStore",
    "OrgPurgeReport",
    "Page",
    "TenantUsage",
    "VectorHit",
]
