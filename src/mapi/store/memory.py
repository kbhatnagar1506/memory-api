"""In-memory reference implementation of MemoryStore.

Not a mock. It implements exact cosine k-NN and a real BM25 lexical index, so
the retrieval pipeline it serves is the same pipeline production runs — only the
index structures differ (exact scan versus HNSW, Python BM25 versus Postgres
tsvector). It is the backend for tests, CI and zero-infrastructure demos, and it
defines the semantics the Postgres backend is held to by the conformance suite.

Concurrency: every mutation takes a single lock. That is the right trade for a
reference implementation — correctness is the point, and the Postgres backend is
what carries production load.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime

from ..core.errors import ConflictError, ValidationError
from ..domain.embeddings.base import Vector, cosine_similarity
from ..domain.models import (
    ApiKey,
    Membership,
    Memory,
    MemoryStatus,
    MemoryVersion,
    Organization,
    RelationEdge,
    RelationType,
    Space,
    User,
    utcnow,
)
from ..domain.text import analyze, analyze_query
from .base import (
    EraseReport,
    LexicalHit,
    MemoryFilter,
    MemoryStore,
    Page,
    TenantUsage,
    VectorHit,
)

# BM25 parameters. k1 controls term-frequency saturation, b the strength of
# length normalization. These are the standard defaults.
_BM25_K1 = 1.2
_BM25_B = 0.75


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> str | None:
    try:
        padding = "=" * (-len(cursor) % 4)
        return base64.urlsafe_b64decode(cursor + padding).decode()
    except Exception:
        return None


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._orgs: dict[str, Organization] = {}
        self._users: dict[str, User] = {}
        self._users_by_sub: dict[str, User] = {}
        self._memberships: dict[str, Membership] = {}
        self._spaces: dict[str, Space] = {}
        self._keys: dict[str, ApiKey] = {}
        self._key_by_hash: dict[str, str] = {}
        self._memories: dict[str, Memory] = {}
        #: (org_id, space_id) -> ordered memory ids
        self._by_space: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._edges: dict[str, RelationEdge] = {}
        #: (space_id, memory_id) -> edge ids, indexed both directions so
        #: "what does X relate to" and "what relates to X" are both O(1).
        self._edges_by_source: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._edges_by_target: dict[tuple[str, str], list[str]] = defaultdict(list)
        #: memory_id -> versions, oldest first. The last entry has valid_to=None.
        self._versions: dict[str, list[MemoryVersion]] = defaultdict(list)

    async def ping(self) -> bool:
        return True

    async def reset(self) -> None:
        async with self._lock:
            self._orgs.clear()
            self._spaces.clear()
            self._keys.clear()
            self._key_by_hash.clear()
            self._memories.clear()
            self._by_space.clear()
            self._edges.clear()
            self._edges_by_source.clear()
            self._edges_by_target.clear()
            self._versions.clear()
            self._users.clear()
            self._users_by_sub.clear()
            self._memberships.clear()

    # -- identity ----------------------------------------------------------

    async def upsert_user(self, user: User) -> User:
        async with self._lock:
            existing = self._users_by_sub.get(user.google_sub)
            if existing is not None:
                # Keep the id stable across logins: memberships point at it,
                # and a new id on every sign-in would orphan them.
                merged = existing.model_copy(
                    update={
                        "email": user.email,
                        "name": user.name or existing.name,
                        "picture": user.picture or existing.picture,
                        "last_seen_at": user.last_seen_at,
                    }
                )
                self._users[merged.id] = merged
                self._users_by_sub[merged.google_sub] = merged
                return merged
            self._users[user.id] = user
            self._users_by_sub[user.google_sub] = user
            return user

    async def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    async def get_user_by_google_sub(self, google_sub: str) -> User | None:
        return self._users_by_sub.get(google_sub)

    async def create_membership(self, membership: Membership) -> Membership:
        async with self._lock:
            for existing in self._memberships.values():
                if (
                    existing.user_id == membership.user_id
                    and existing.org_id == membership.org_id
                ):
                    raise ConflictError("already a member of that organization")
            self._memberships[membership.id] = membership
            return membership

    async def list_memberships(self, user_id: str) -> list[Membership]:
        return sorted(
            (m for m in self._memberships.values() if m.user_id == user_id),
            key=lambda m: m.created_at,
        )

    async def get_membership(self, user_id: str, org_id: str) -> Membership | None:
        for m in self._memberships.values():
            if m.user_id == user_id and m.org_id == org_id:
                return m
        return None

    # -- organizations & spaces -------------------------------------------

    async def create_organization(self, org: Organization) -> Organization:
        async with self._lock:
            if org.id in self._orgs:
                raise ConflictError(f"organization {org.id} already exists")
            self._orgs[org.id] = org
            return org

    async def get_organization(self, org_id: str) -> Organization | None:
        return self._orgs.get(org_id)

    async def create_space(self, space: Space) -> Space:
        async with self._lock:
            if space.id in self._spaces:
                raise ConflictError(f"space {space.id} already exists")
            for existing in self._spaces.values():
                if existing.org_id == space.org_id and existing.slug == space.slug:
                    raise ConflictError(
                        f"slug {space.slug!r} is already used in this organization"
                    )
            self._spaces[space.id] = space
            self._by_space.setdefault((space.org_id, space.id), [])
            return space

    async def get_space(self, org_id: str, space_id: str) -> Space | None:
        space = self._spaces.get(space_id)
        return space if space and space.org_id == org_id else None

    async def get_space_by_slug(self, org_id: str, slug: str) -> Space | None:
        for space in self._spaces.values():
            if space.org_id == org_id and space.slug == slug:
                return space
        return None

    async def list_spaces(self, org_id: str) -> list[Space]:
        return sorted(
            (s for s in self._spaces.values() if s.org_id == org_id),
            key=lambda s: s.id,
        )

    async def delete_space(self, org_id: str, space_id: str) -> bool:
        async with self._lock:
            space = self._spaces.get(space_id)
            if space is None or space.org_id != org_id:
                return False
            for memory_id in self._by_space.pop((org_id, space_id), []):
                self._memories.pop(memory_id, None)
                self._remove_edges_touching(space_id, memory_id)
            del self._spaces[space_id]
            return True

    # -- api keys ----------------------------------------------------------

    async def create_api_key(self, key: ApiKey) -> ApiKey:
        async with self._lock:
            if key.key_hash in self._key_by_hash:
                raise ConflictError("api key hash collision")
            self._keys[key.id] = key
            self._key_by_hash[key.key_hash] = key.id
            return key

    async def get_api_key_by_hash(self, key_hash: str) -> ApiKey | None:
        key_id = self._key_by_hash.get(key_hash)
        return self._keys.get(key_id) if key_id else None

    async def list_api_keys(self, org_id: str) -> list[ApiKey]:
        return sorted(
            (k for k in self._keys.values() if k.org_id == org_id), key=lambda k: k.id
        )

    async def revoke_api_key(self, org_id: str, key_id: str) -> bool:
        async with self._lock:
            key = self._keys.get(key_id)
            if key is None or key.org_id != org_id or key.revoked_at is not None:
                return False
            from ..domain.models import utcnow

            self._keys[key_id] = key.model_copy(update={"revoked_at": utcnow()})
            return True

    async def touch_api_key(self, key_id: str, when: datetime) -> None:
        key = self._keys.get(key_id)
        if key is not None:
            self._keys[key_id] = key.model_copy(update={"last_used_at": when})

    # -- memories ----------------------------------------------------------

    async def upsert_memory(self, memory: Memory, *, now: datetime | None = None) -> Memory:
        async with self._lock:
            bucket = self._by_space.setdefault((memory.org_id, memory.space_id), [])
            if memory.id not in self._memories:
                bucket.append(memory.id)
                bucket.sort()
            self._memories[memory.id] = memory

            # See PostgresStore.upsert_memory: a write that does not bump
            # `version` is an in-place correction and overwrites the open
            # snapshot; only a version bump opens a new one. Both backends
            # apply this rule, and the conformance suite pins it.
            versions = self._versions[memory.id]
            when = now or utcnow()
            open_version = versions[-1] if versions and versions[-1].valid_to is None else None
            if open_version is not None and open_version.version == memory.version:
                versions[-1] = MemoryVersion.snapshot(
                    memory, valid_from=open_version.valid_from
                ).model_copy(update={"id": open_version.id})
            else:
                if open_version is not None:
                    versions[-1] = open_version.model_copy(update={"valid_to": when})
                versions.append(MemoryVersion.snapshot(memory, valid_from=when))
            return memory

    async def get_memory(self, org_id: str, space_id: str, memory_id: str) -> Memory | None:
        memory = self._memories.get(memory_id)
        if memory is None:
            return None
        # Tenancy check lives here so no caller can skip it.
        if memory.org_id != org_id or memory.space_id != space_id:
            return None
        return memory

    async def get_memories(
        self,
        org_id: str,
        space_id: str,
        memory_ids: Sequence[str],
        *,
        with_embeddings: bool = True,
    ) -> dict[str, Memory]:
        # `with_embeddings` is a transfer optimisation; vectors here are
        # already in process, so there is nothing to save by dropping them.
        del with_embeddings
        out: dict[str, Memory] = {}
        for memory_id in memory_ids:
            memory = await self.get_memory(org_id, space_id, memory_id)
            if memory is not None:
                out[memory_id] = memory
        return out

    async def delete_memory(self, org_id: str, space_id: str, memory_id: str) -> bool:
        async with self._lock:
            memory = self._memories.get(memory_id)
            if memory is None or memory.org_id != org_id or memory.space_id != space_id:
                return False
            del self._memories[memory_id]
            bucket = self._by_space.get((org_id, space_id))
            if bucket and memory_id in bucket:
                bucket.remove(memory_id)
            self._bridge_supersession_locked(space_id, memory_id)
            self._remove_edges_touching(space_id, memory_id)
            # Version history is a durable log, decoupled from the live table's
            # lifecycle on purpose: deleting a memory must not erase its audit
            # trail. It is left in self._versions.
            return True

    def _bridge_supersession_locked(self, space_id: str, memory_id: str) -> int:
        """Preserve revision chains across the removal of `memory_id`.

        Given A supersedes B supersedes C, removing B used to take the only
        path from A to C with it — and C, a fact the user explicitly replaced,
        resurfaced as current in every search. Found by erasing the middle of
        a "lives in Portland" -> "Austin" -> "Seattle" chain: Portland came
        back. For each (X -> memory, memory -> Y) pair of SUPERSEDES edges,
        insert X -> Y before the removal severs them.

        Only SUPERSEDES bridges. The other edge types do not compose: if B was
        derived from C, A deriving from B says nothing about C; a broken
        derivation should stay visibly broken.

        Confidence is the min of the two hops — a chain is as strong as its
        weakest link. The bridge reason names neither the removed memory nor
        its content, so the erasure path stays compliant.

        Caller must hold `self._lock` (both callers do).
        """
        incoming = [
            self._edges[eid]
            for eid in self._edges_by_target.get((space_id, memory_id), [])
            if self._edges[eid].type is RelationType.SUPERSEDES
        ]
        outgoing = [
            self._edges[eid]
            for eid in self._edges_by_source.get((space_id, memory_id), [])
            if self._edges[eid].type is RelationType.SUPERSEDES
        ]
        bridged = 0
        for upstream in incoming:
            for downstream in outgoing:
                if upstream.source_id == downstream.target_id:
                    continue  # would be a self-loop
                exists = any(
                    self._edges[eid].target_id == downstream.target_id
                    and self._edges[eid].type is RelationType.SUPERSEDES
                    for eid in self._edges_by_source.get((space_id, upstream.source_id), [])
                )
                if exists:
                    continue
                bridge = RelationEdge(
                    org_id=upstream.org_id,
                    space_id=space_id,
                    source_id=upstream.source_id,
                    target_id=downstream.target_id,
                    type=RelationType.SUPERSEDES,
                    reason="bridged across a removed intermediate revision",
                    confidence=min(upstream.confidence, downstream.confidence),
                )
                self._edges[bridge.id] = bridge
                self._edges_by_source[(space_id, bridge.source_id)].append(bridge.id)
                self._edges_by_target[(space_id, bridge.target_id)].append(bridge.id)
                bridged += 1
        return bridged

    def _remove_edges_touching(self, space_id: str, memory_id: str) -> int:
        """Cascade-delete edges when their source or target memory is gone.

        An edge whose endpoint no longer exists is worse than no edge: a
        lineage walk would dereference a memory id that resolves to nothing.
        Returns how many edges were removed, so erasure can attest to it.
        """
        removed = 0
        for edge_id in list(self._edges_by_source.pop((space_id, memory_id), [])):
            edge = self._edges.pop(edge_id, None)
            if edge is not None:
                removed += 1
                other = self._edges_by_target.get((space_id, edge.target_id))
                if other and edge_id in other:
                    other.remove(edge_id)
        for edge_id in list(self._edges_by_target.pop((space_id, memory_id), [])):
            edge = self._edges.pop(edge_id, None)
            if edge is not None:
                removed += 1
                other = self._edges_by_source.get((space_id, edge.source_id))
                if other and edge_id in other:
                    other.remove(edge_id)
        return removed

    async def erase_memory(self, org_id: str, space_id: str, memory_id: str) -> EraseReport:
        async with self._lock:
            memory = self._memories.get(memory_id)
            live = (
                memory is not None and memory.org_id == org_id and memory.space_id == space_id
            )
            # History may exist even when the live row is gone (delete_memory
            # preserves it). Erasure must purge that residue too, or a
            # point-in-time read resurrects content the caller was told is gone.
            residual = [
                v
                for v in self._versions.get(memory_id, [])
                if v.org_id == org_id and v.space_id == space_id
            ]
            if not live and not residual:
                return EraseReport(existed=False)

            chunks = len(memory.chunks) if live and memory is not None else 0
            edges = 0
            bridged = 0
            if live:
                del self._memories[memory_id]
                bucket = self._by_space.get((org_id, space_id))
                if bucket and memory_id in bucket:
                    bucket.remove(memory_id)
                bridged = self._bridge_supersession_locked(space_id, memory_id)
                edges = self._remove_edges_touching(space_id, memory_id)

            versions = len(residual)
            if memory_id in self._versions:
                remaining = [
                    v
                    for v in self._versions[memory_id]
                    if not (v.org_id == org_id and v.space_id == space_id)
                ]
                if remaining:
                    self._versions[memory_id] = remaining
                else:
                    del self._versions[memory_id]

            return EraseReport(
                existed=True,
                chunks_removed=chunks,
                edges_removed=edges,
                versions_purged=versions,
                edges_bridged=bridged,
            )

    def _space_memories(self, org_id: str, space_id: str) -> list[Memory]:
        return [
            self._memories[mid]
            for mid in self._by_space.get((org_id, space_id), [])
            if mid in self._memories
        ]

    async def list_memories(
        self,
        org_id: str,
        space_id: str,
        *,
        filters: MemoryFilter,
        limit: int,
        cursor: str | None = None,
    ) -> Page:
        if limit <= 0:
            return Page(items=[])
        matching = sorted(
            (m for m in self._space_memories(org_id, space_id) if filters.matches(m)),
            key=lambda m: m.id,
        )
        start = 0
        if cursor:
            after = _decode_cursor(cursor)
            if after is None:
                from ..core.errors import BadRequestError

                raise BadRequestError("malformed cursor", field="cursor")
            start = next((i for i, m in enumerate(matching) if m.id > after), len(matching))
        window = matching[start : start + limit]
        next_cursor = (
            _encode_cursor(window[-1].id) if window and start + limit < len(matching) else None
        )
        return Page(items=window, next_cursor=next_cursor, total=len(matching))

    async def count_memories(self, org_id: str, space_id: str, *, filters: MemoryFilter) -> int:
        return sum(1 for m in self._space_memories(org_id, space_id) if filters.matches(m))

    async def find_by_content_hash(
        self, org_id: str, space_id: str, digest: str
    ) -> Memory | None:
        for memory in self._space_memories(org_id, space_id):
            if memory.content_sha256 == digest and memory.status is not MemoryStatus.ARCHIVED:
                return memory
        return None

    # -- relations: a typed, indexed graph ----------------------------------

    async def create_relation(self, edge: RelationEdge) -> RelationEdge:
        async with self._lock:
            for existing_id in self._edges_by_source.get((edge.space_id, edge.source_id), []):
                existing = self._edges[existing_id]
                if existing.target_id == edge.target_id and existing.type == edge.type:
                    return existing
            self._edges[edge.id] = edge
            self._edges_by_source[(edge.space_id, edge.source_id)].append(edge.id)
            self._edges_by_target[(edge.space_id, edge.target_id)].append(edge.id)
            return edge

    async def list_relations(
        self,
        org_id: str,
        space_id: str,
        memory_id: str,
        *,
        direction: str = "out",
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        if direction not in ("out", "in"):
            # ValidationError (422), not a bare ValueError. Unreachable over HTTP
            # today -- the route constrains it with Query(pattern="^(out|in)$") --
            # but a non-MapiError escaping the store becomes an opaque 500 through
            # the catch-all handler, so any future caller that forwards an
            # unvalidated direction would get "an unexpected error occurred"
            # instead of a problem document naming the field.
            raise ValidationError(
                f"direction must be 'out' or 'in', got {direction!r}", field="direction"
            )
        index = self._edges_by_source if direction == "out" else self._edges_by_target
        edges = [self._edges[eid] for eid in index.get((space_id, memory_id), [])]
        edges = [e for e in edges if e.org_id == org_id]
        if type is not None:
            edges = [e for e in edges if e.type == type]
        # Oldest first: `walk_supersession_chain` relies on this to pick the
        # original supersession decision when a memory has more than one.
        return sorted(edges, key=lambda e: (e.created_at, e.id))

    async def get_relations_between(
        self,
        org_id: str,
        space_id: str,
        memory_ids: Sequence[str],
        *,
        type: RelationType | None = None,
    ) -> list[RelationEdge]:
        id_set = set(memory_ids)
        found: dict[str, RelationEdge] = {}
        for memory_id in id_set:
            for edge_id in self._edges_by_source.get((space_id, memory_id), []):
                edge = self._edges[edge_id]
                if (
                    edge.org_id == org_id
                    and edge.target_id in id_set
                    and (type is None or edge.type == type)
                ):
                    found[edge.id] = edge
        return sorted(found.values(), key=lambda e: (e.created_at, e.id))

    # -- bitemporal history --------------------------------------------------

    async def get_memory_as_of(
        self, org_id: str, space_id: str, memory_id: str, as_of: datetime
    ) -> Memory | None:
        as_of_aware = _aware(as_of)
        for version in reversed(self._versions.get(memory_id, [])):
            if version.org_id != org_id or version.space_id != space_id:
                continue
            valid_from = _aware(version.valid_from)
            valid_to = _aware(version.valid_to) if version.valid_to else None
            if valid_from <= as_of_aware and (valid_to is None or as_of_aware < valid_to):
                return Memory(
                    id=version.memory_id,
                    org_id=version.org_id,
                    space_id=version.space_id,
                    content=version.content,
                    summary=version.summary,
                    metadata=version.metadata,
                    tags=version.tags,
                    source=version.source,
                    kind=version.kind,
                    status=version.status,
                    occurred_at=version.occurred_at,
                    version=version.version,
                    chunks=[],
                )
        return None

    async def list_memory_versions(
        self, org_id: str, space_id: str, memory_id: str
    ) -> list[MemoryVersion]:
        return [
            v
            for v in self._versions.get(memory_id, [])
            if v.org_id == org_id and v.space_id == space_id
        ]

    # -- retrieval ---------------------------------------------------------

    async def vector_search(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        filters: MemoryFilter,
    ) -> list[VectorHit]:
        if not embedding or limit <= 0:
            return []
        hits: list[VectorHit] = []
        for memory in self._space_memories(org_id, space_id):
            if not filters.matches(memory):
                continue
            best: VectorHit | None = None
            for chunk in memory.chunks:
                if chunk.embedding is None or len(chunk.embedding) != len(embedding):
                    continue
                score = cosine_similarity(embedding, chunk.embedding)
                if best is None or score > best.score:
                    best = VectorHit(memory.id, chunk.id, score, chunk.text)
            if best is not None:
                hits.append(best)
        # Deterministic ordering: ties break on memory id so paging is stable.
        hits.sort(key=lambda h: (-h.score, h.memory_id))
        return hits[:limit]

    async def lexical_search(
        self,
        org_id: str,
        space_id: str,
        query: str,
        *,
        limit: int,
        filters: MemoryFilter,
    ) -> list[LexicalHit]:
        terms = analyze_query(query)
        if not terms or limit <= 0:
            return []

        # Build a BM25 index over the chunks visible to this tenant.
        docs: list[tuple[str, str, list[str], str]] = []
        for memory in self._space_memories(org_id, space_id):
            if not filters.matches(memory):
                continue
            for chunk in memory.chunks:
                docs.append((memory.id, chunk.id, analyze(chunk.text), chunk.text))
        if not docs:
            return []

        n = len(docs)
        avgdl = sum(len(d[2]) for d in docs) / n
        df: dict[str, int] = defaultdict(int)
        unique_terms = set(terms)
        for _, _, tokens, _ in docs:
            for term in unique_terms & set(tokens):
                df[term] += 1

        best_per_memory: dict[str, LexicalHit] = {}
        for memory_id, chunk_id, tokens, text in docs:
            if not tokens:
                continue
            dl = len(tokens)
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            score = 0.0
            for term in unique_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                # Standard BM25 IDF with the +0.5 smoothing that keeps it
                # positive for terms appearing in more than half the corpus.
                idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
                score += idf * (tf * (_BM25_K1 + 1)) / denom
            if score <= 0:
                continue
            current = best_per_memory.get(memory_id)
            if current is None or score > current.score:
                best_per_memory[memory_id] = LexicalHit(memory_id, chunk_id, score, text)

        hits = sorted(best_per_memory.values(), key=lambda h: (-h.score, h.memory_id))
        return hits[:limit]

    async def tenant_usage(self, org_id: str, *, since: datetime) -> TenantUsage:
        memories = bytes_stored = writes = 0
        for memory in self._memories.values():
            if memory.org_id != org_id or memory.status is MemoryStatus.ARCHIVED:
                continue
            memories += 1
            bytes_stored += len(memory.content.encode("utf-8"))
            if memory.created_at >= since:
                writes += 1
        return TenantUsage(memories=memories, bytes_stored=bytes_stored, writes_today=writes)

    async def neighbours(
        self,
        org_id: str,
        space_id: str,
        embedding: Vector,
        *,
        limit: int,
        exclude_id: str = "",
    ) -> list[tuple[Memory, Vector]]:
        """Brute force, which is what this backend is for.

        No index and no approximation: the reference implementation is the
        one the conformance suite checks Postgres against, so it computes the
        exact answer and lets the ANN backend be the thing that approximates.
        """
        if not embedding or limit <= 0:
            return []
        scored: list[tuple[float, Memory, Vector]] = []
        for memory in self._space_memories(org_id, space_id):
            if memory.id == exclude_id or memory.status is MemoryStatus.ARCHIVED:
                continue
            best: tuple[float, Vector] | None = None
            for chunk in memory.chunks:
                if chunk.embedding is None or len(chunk.embedding) != len(embedding):
                    continue
                score = cosine_similarity(embedding, chunk.embedding)
                if best is None or score > best[0]:
                    best = (score, chunk.embedding)
            if best is not None:
                scored.append((best[0], memory, best[1]))
        # Tie-break on id so a page of equal scores is stable across calls.
        scored.sort(key=lambda row: (-row[0], row[1].id))
        return [(memory, vector) for _, memory, vector in scored[:limit]]

    async def sample_embeddings(
        self, org_id: str, space_id: str, *, limit: int
    ) -> list[tuple[str, Vector]]:
        out: list[tuple[str, Vector]] = []
        for memory in self._space_memories(org_id, space_id):
            if memory.status is MemoryStatus.ARCHIVED:
                continue
            for chunk in memory.chunks:
                if chunk.embedding is not None:
                    out.append((memory.id, chunk.embedding))
                    break
            if len(out) >= limit:
                break
        return out

    # -- introspection, used by tests and the demo seeder ------------------

    def stats(self) -> dict[str, int]:
        return {
            "organizations": len(self._orgs),
            "spaces": len(self._spaces),
            "memories": len(self._memories),
            "api_keys": len(self._keys),
            "chunks": sum(len(m.chunks) for m in self._memories.values()),
        }

    def dump(self) -> str:
        return json.dumps(
            {"memories": [m.model_dump(mode="json") for m in self._memories.values()]},
            sort_keys=True,
        )


__all__ = ["InMemoryStore"]
