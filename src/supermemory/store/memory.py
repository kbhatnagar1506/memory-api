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
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime

from ..core.errors import ConflictError
from ..domain.embeddings.base import Vector, cosine_similarity
from ..domain.models import ApiKey, Memory, MemoryStatus, Organization, Space
from .base import LexicalHit, MemoryFilter, MemoryStore, Page, VectorHit

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# BM25 parameters. k1 controls term-frequency saturation, b the strength of
# length normalization. These are the standard defaults.
_BM25_K1 = 1.2
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


def _encode_cursor(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> str | None:
    try:
        padding = "=" * (-len(cursor) % 4)
        return base64.urlsafe_b64decode(cursor + padding).decode()
    except Exception:  # noqa: BLE001 - a malformed cursor is a client error
        return None


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._orgs: dict[str, Organization] = {}
        self._spaces: dict[str, Space] = {}
        self._keys: dict[str, ApiKey] = {}
        self._key_by_hash: dict[str, str] = {}
        self._memories: dict[str, Memory] = {}
        #: (org_id, space_id) -> ordered memory ids
        self._by_space: dict[tuple[str, str], list[str]] = defaultdict(list)

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

    async def upsert_memory(self, memory: Memory) -> Memory:
        async with self._lock:
            bucket = self._by_space.setdefault((memory.org_id, memory.space_id), [])
            if memory.id not in self._memories:
                bucket.append(memory.id)
                bucket.sort()
            self._memories[memory.id] = memory
            return memory

    async def get_memory(
        self, org_id: str, space_id: str, memory_id: str
    ) -> Memory | None:
        memory = self._memories.get(memory_id)
        if memory is None:
            return None
        # Tenancy check lives here so no caller can skip it.
        if memory.org_id != org_id or memory.space_id != space_id:
            return None
        return memory

    async def get_memories(
        self, org_id: str, space_id: str, memory_ids: Sequence[str]
    ) -> dict[str, Memory]:
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
            return True

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
            start = next(
                (i for i, m in enumerate(matching) if m.id > after), len(matching)
            )
        window = matching[start : start + limit]
        next_cursor = (
            _encode_cursor(window[-1].id)
            if window and start + limit < len(matching)
            else None
        )
        return Page(items=window, next_cursor=next_cursor, total=len(matching))

    async def count_memories(
        self, org_id: str, space_id: str, *, filters: MemoryFilter
    ) -> int:
        return sum(
            1 for m in self._space_memories(org_id, space_id) if filters.matches(m)
        )

    async def find_by_content_hash(
        self, org_id: str, space_id: str, digest: str
    ) -> Memory | None:
        for memory in self._space_memories(org_id, space_id):
            if memory.content_sha256 == digest and memory.status is not MemoryStatus.ARCHIVED:
                return memory
        return None

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
        terms = _tokenize(query)
        if not terms or limit <= 0:
            return []

        # Build a BM25 index over the chunks visible to this tenant.
        docs: list[tuple[str, str, list[str], str]] = []
        for memory in self._space_memories(org_id, space_id):
            if not filters.matches(memory):
                continue
            for chunk in memory.chunks:
                docs.append((memory.id, chunk.id, _tokenize(chunk.text), chunk.text))
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
                best_per_memory[memory_id] = LexicalHit(
                    memory_id, chunk_id, score, text
                )

        hits = sorted(
            best_per_memory.values(), key=lambda h: (-h.score, h.memory_id)
        )
        return hits[:limit]

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
