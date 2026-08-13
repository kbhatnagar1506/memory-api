"""Builders for domain objects, shared instead of copied six times.

The suite currently has six near-identical private memory builders --
`test_hydrate.py::_memory`, `test_consolidation.py::make` and `::_mem`,
`test_auto_supersede.py::_mem`, `test_adjudicate.py::_memory`,
`test_store_contract.py::_add` -- each re-solving the same two annoyances: a
`Memory` needs an org and a space it does not create, and attaching a chunk with
an embedding means building the chunk after the memory and then
`model_copy(update=...)`-ing it back in.

MIGRATION POLICY: new tests use this module; the six existing builders stay
where they are. Rewriting several files of passing tests is diff noise and
regression risk for no behavioural gain. Delete a local builder when its file is
being touched for some other reason.

One capability no local builder has, and the reason this module accepts it
explicitly: **`content_sha256` can be set directly.** `Memory._fill_hash` only
fills the field when it is empty, so a test for hash staleness otherwise has to
go through `model_copy`, which skips validation entirely and hides the very
mechanism under test.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from mapi.core.security import build_api_key
from mapi.domain.embeddings.base import Vector
from mapi.domain.models import (
    ApiKey,
    Chunk,
    Memory,
    MemoryStatus,
    MemoryVersion,
    Organization,
    RelationEdge,
    RelationType,
    Scope,
    ScoredMemory,
    Space,
)

#: Every default timestamp derives from one instant, so a test that builds ten
#: memories gets ten with the same `created_at` unless it says otherwise --
#: which makes ordering assertions depend on stated intent, not on how long the
#: builder loop took.
EPOCH = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def memory(
    *,
    org_id: str,
    space_id: str,
    content: str = "a remembered fact",
    vector: Vector | None = None,
    chunk_texts: Sequence[str] | None = None,
    chunk_vectors: Sequence[Vector] | None = None,
    **kwargs: Any,
) -> Memory:
    """A `Memory`, with chunks attached in one call.

    `vector` attaches a single chunk holding the whole content -- the shape
    almost every test wants. `chunk_texts` plus `chunk_vectors` builds several,
    for the intra-item position sweep where a long memory's chunks must sit at
    different similarities.

    Anything else (`tags`, `status`, `kind`, `occurred_at`, `metadata`,
    `content_sha256`, ...) passes straight through to the model, so its
    validators run exactly as they would in production.
    """
    if chunk_texts is not None and vector is not None:
        raise ValueError("pass vector for one chunk or chunk_texts for several, not both")

    built = Memory(org_id=org_id, space_id=space_id, content=content, **kwargs)

    if chunk_texts is not None:
        vectors = list(chunk_vectors or [None] * len(chunk_texts))  # type: ignore[list-item]
        if len(vectors) != len(chunk_texts):
            raise ValueError(f"{len(chunk_texts)} chunk texts but {len(vectors)} chunk vectors")
        chunks = [
            Chunk(memory_id=built.id, ordinal=i, text=text, embedding=vec)
            for i, (text, vec) in enumerate(zip(chunk_texts, vectors, strict=True))
        ]
    elif vector is not None:
        chunks = [Chunk(memory_id=built.id, ordinal=0, text=content, embedding=list(vector))]
    else:
        chunks = []

    return built.model_copy(update={"chunks": chunks}) if chunks else built


async def store_memory(store: Any, **kwargs: Any) -> Memory:
    """`memory()` then `upsert_memory`. The store-level equivalent of ingest.

    Deliberately NOT `service.ingest`: ingest interposes dedupe, a `neighbours`
    lookup, supersession, contradiction detection and association proposal, so
    what lands is not necessarily what was asked for. Tests about those
    behaviours call ingest; tests that need a corpus to exist call this.
    """
    built = memory(**kwargs)
    await store.upsert_memory(built)
    return built


def edge(
    *,
    org_id: str,
    space_id: str,
    source_id: str,
    target_id: str,
    type: RelationType = RelationType.REFERENCES,
    **kwargs: Any,
) -> RelationEdge:
    """A `RelationEdge`. For `SUPERSEDES`, `source_id` is the NEWER memory."""
    return RelationEdge(
        org_id=org_id,
        space_id=space_id,
        source_id=source_id,
        target_id=target_id,
        type=type,
        **kwargs,
    )


def version(
    source: Memory,
    *,
    valid_from: datetime,
    valid_to: datetime | None = None,
    **kwargs: Any,
) -> MemoryVersion:
    """A `MemoryVersion` snapshot of `source` over `[valid_from, valid_to)`.

    Half-open, matching both stores. Useful for constructing histories the write
    path cannot produce -- overlapping intervals, non-monotonic versions -- which
    is exactly where the two backends were found to disagree.
    """
    fields: dict[str, Any] = {
        "memory_id": source.id,
        "org_id": source.org_id,
        "space_id": source.space_id,
        "version": source.version,
        "content": source.content,
        "summary": source.summary,
        "metadata": dict(source.metadata),
        "tags": list(source.tags),
        "source": source.source,
        "kind": source.kind,
        "status": source.status,
        "occurred_at": source.occurred_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
    }
    fields.update(kwargs)
    return MemoryVersion(**fields)


async def tenant(
    store: Any, *, name: str = "Test Org", slug: str = ""
) -> tuple[Organization, Space]:
    """An org and a space in it. Slug derived from the org id when unset, so
    two calls against a shared Postgres store cannot collide."""
    org = await store.create_organization(Organization(name=name))
    space = await store.create_space(
        Space(org_id=org.id, slug=slug or f"s{org.id[-8:]}", name="Space")
    )
    return org, space


def api_key(
    org_id: str,
    *,
    pepper: str = "test-pepper",
    scopes: frozenset[Scope] | None = None,
    **kwargs: Any,
) -> tuple[ApiKey, str]:
    """A key record plus its plaintext, which exists only at creation time."""
    return build_api_key(
        org_id=org_id,
        name=kwargs.pop("name", "test key"),
        scopes=scopes if scopes is not None else frozenset(Scope.all()),
        pepper=pepper,
        **kwargs,
    )


def scored(
    source: Memory,
    score: float,
    **kwargs: Any,
) -> ScoredMemory:
    """A `ScoredMemory`, for testing rerank / MMR / confidence in isolation."""
    return ScoredMemory(memory=source, score=score, **kwargs)


def superseded(source: Memory) -> Memory:
    """`source` flipped to SUPERSEDED with its version bumped, as the write
    path does it -- so a test can build a hidden memory without replaying the
    consolidation that would hide it."""
    return source.model_copy(
        update={"status": MemoryStatus.SUPERSEDED, "version": source.version + 1}
    )


__all__ = [
    "EPOCH",
    "api_key",
    "edge",
    "memory",
    "scored",
    "store_memory",
    "superseded",
    "tenant",
    "version",
]
