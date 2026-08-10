"""Resource namespaces: `client.memories.add(...)`, `client.search.execute(...)`.

Grouped by resource rather than flattened onto the client, which is the shape
every generated SDK in this space converged on (OpenAI, Anthropic,
Supermemory). It is not fashion: a flat client with thirty methods stops
being discoverable, and `client.memories.` in an editor is a table of
contents for one concept rather than an undifferentiated list.

Sync and async are separate classes, as in every generated SDK, because a
resource does not merely forward a call -- it parses the result. A shared
body cannot both `return Memory.parse(data)` and `return Memory.parse(await
data)`. What IS shared is every URL and request body, built by the helpers
below and called from both, so the part that actually drifts cannot.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, TypeVar

from ._errors import MapiError
from ._models import Memory, MemoryContext, SearchResult, Space

T = TypeVar("T")

Relation = Literal["supersedes", "contradicts", "derived_from", "references"]

#: A request function. Sync clients return the payload; async clients return
#: an awaitable of it. Resources are written once against this signature and
#: simply never await -- the client that owns them decides.
Requester = Callable[..., Any]


def _memory_body(
    content: str,
    *,
    summary: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    source: str = "",
    occurred_at: datetime | None = None,
    extract: bool = False,
    auto_supersede: bool = False,
    detect_conflicts: bool = False,
    dedupe: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {"content": content, "dedupe": dedupe}
    if summary:
        body["summary"] = summary
    if tags:
        body["tags"] = tags
    if metadata:
        body["metadata"] = metadata
    if source:
        body["source"] = source
    if occurred_at is not None:
        body["occurred_at"] = occurred_at.isoformat()
    for flag, value in (
        ("extract", extract),
        ("auto_supersede", auto_supersede),
        ("detect_conflicts", detect_conflicts),
    ):
        if value:
            body[flag] = True
    return body


def _search_body(
    query: str,
    *,
    limit: int = 10,
    tags: list[str] | None = None,
    include_superseded: bool = False,
    min_score: float = 0.0,
    explain: bool = False,
    **options: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {"query": query, "limit": limit, **options}
    if tags:
        body["tags"] = tags
    if include_superseded:
        body["include_superseded"] = True
    if min_score:
        body["min_score"] = min_score
    if explain:
        body["explain"] = True
    return body


class _Resource:
    def __init__(self, request: Requester, space_id: Callable[[str], Any]) -> None:
        self._request = request
        self._space_id = space_id


# -- sync ---------------------------------------------------------------------


class Memories(_Resource):
    """Writing and reading individual memories."""

    def add(
        self,
        content: str,
        *,
        space: str,
        summary: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "",
        occurred_at: datetime | None = None,
        extract: bool = False,
        auto_supersede: bool = False,
        detect_conflicts: bool = False,
        dedupe: bool = True,
    ) -> Memory:
        """Store one memory.

        `occurred_at` is when it HAPPENED, not when you are writing it. The
        difference matters for anything asking what order things came in, and
        defaulting to now makes a backfill look like it all happened today.
        """
        data = self._request(
            "POST",
            f"/v1/spaces/{self._space_id(space)}/memories",
            json=_memory_body(
                content,
                summary=summary,
                tags=tags,
                metadata=metadata,
                source=source,
                occurred_at=occurred_at,
                extract=extract,
                auto_supersede=auto_supersede,
                detect_conflicts=detect_conflicts,
                dedupe=dedupe,
            ),
        )
        return Memory.parse((data or {}).get("memory", data) or {})

    def add_many(
        self, items: builtins.list[dict[str, Any]], *, space: str
    ) -> builtins.list[Memory]:
        """Store up to 100 memories in one request."""
        data = self._request(
            "POST",
            f"/v1/spaces/{self._space_id(space)}/memories/bulk",
            json={"items": items},
        )
        rows = (data or {}).get("items") or (data or {}).get("results") or []
        return [Memory.parse(r.get("memory", r)) for r in rows]

    def get(self, memory_id: str, *, space: str) -> Memory:
        data = self._request(
            "GET", f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}"
        )
        return Memory.parse(data or {})

    def list(
        self, *, space: str, limit: int = 50, cursor: str | None = None
    ) -> builtins.list[Memory]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = self._request(
            "GET", f"/v1/spaces/{self._space_id(space)}/memories", params=params
        )
        return [Memory.parse(m) for m in (data or {}).get("items", [])]

    def delete(self, memory_id: str, *, space: str) -> None:
        self._request("DELETE", f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}")

    def erase(self, memory_id: str, *, space: str) -> dict[str, Any]:
        """Right-to-erasure purge, with an attestation of what was destroyed.

        Unlike delete, this removes the content everywhere it can be reached
        and returns the blast radius -- including what had claimed derivation
        from it, since those are now standing on nothing.
        """
        result: dict[str, Any] = (
            self._request(
                "POST",
                f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/erase",
            )
            or {}
        )
        return result

    def context(self, memory_id: str, *, space: str) -> MemoryContext:
        """A memory with every typed relation touching it.

        The call that answers "why does the system believe this": what
        replaced it, what it replaced, what it was computed from, and what
        disagrees with it.
        """
        data = self._request(
            "GET", f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/context"
        )
        return MemoryContext.parse(data or {})

    def lineage(self, memory_id: str, *, space: str) -> dict[str, Any]:
        """Where this memory sits in its supersession chain."""
        result: dict[str, Any] = (
            self._request(
                "GET",
                f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/lineage",
            )
            or {}
        )
        return result

    def versions(self, memory_id: str, *, space: str) -> builtins.list[dict[str, Any]]:
        data = self._request(
            "GET", f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/versions"
        )
        return list((data or {}).get("items", []))

    def relate(
        self,
        memory_id: str,
        *,
        space: str,
        target_id: str,
        relation: Relation,
        reason: str = "",
    ) -> dict[str, Any]:
        result: dict[str, Any] = (
            self._request(
                "POST",
                f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/relations",
                json={"target_id": target_id, "type": relation, "reason": reason},
            )
            or {}
        )
        return result

    def relations(self, memory_id: str, *, space: str) -> builtins.list[dict[str, Any]]:
        data = self._request(
            "GET", f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/relations"
        )
        return list((data or {}).get("items", []))


class Search(_Resource):
    """Hybrid retrieval: vector and lexical, fused."""

    def execute(
        self,
        query: str,
        *,
        space: str,
        limit: int = 10,
        tags: list[str] | None = None,
        include_superseded: bool = False,
        min_score: float = 0.0,
        explain: bool = False,
        **options: Any,
    ) -> SearchResult:
        """Search a space.

        Superseded memories are excluded by default. A replaced fact is still
        stored and still reachable, but returning it beside its replacement is
        how an agent states last month's answer with this month's confidence.
        """
        data = self._request(
            "POST",
            f"/v1/spaces/{self._space_id(space)}/search",
            json=_search_body(
                query,
                limit=limit,
                tags=tags,
                include_superseded=include_superseded,
                min_score=min_score,
                explain=explain,
                **options,
            ),
        )
        return SearchResult.parse(data or {})


class Spaces:
    """Spaces. One space is one person's memory."""

    def __init__(self, request: Requester, cache: dict[str, str]) -> None:
        self._request = request
        self._cache = cache

    def list(self) -> builtins.list[Space]:
        data = self._request("GET", "/v1/spaces")
        spaces = [Space.parse(s) for s in (data or {}).get("items", [])]
        for space in spaces:
            self._cache[space.slug] = space.id
        return spaces

    def create(self, slug: str, name: str | None = None, **extra: Any) -> Space:
        data = self._request(
            "POST", "/v1/spaces", json={"slug": slug, "name": name or slug, **extra}
        )
        space = Space.parse(data or {})
        self._cache[space.slug] = space.id
        return space

    def get(self, space_id: str) -> Space:
        return Space.parse(self._request("GET", f"/v1/spaces/{space_id}") or {})

    def get_or_create(self, slug: str, name: str | None = None) -> Space:
        for space in self.list():
            if space.slug == slug:
                return space
        return self.create(slug, name)

    def delete(self, space_id: str) -> None:
        self._request("DELETE", f"/v1/spaces/{space_id}")


class Graph(_Resource):
    """The space as memories plus the typed edges between them."""

    def get(self, *, space: str, limit: int = 300) -> dict[str, Any]:
        result: dict[str, Any] = (
            self._request(
                "GET",
                f"/v1/spaces/{self._space_id(space)}/graph",
                params={"limit": limit},
            )
            or {}
        )
        return result


def resolve_slug(spaces: builtins.list[Space], slug: str, cache: dict[str, str]) -> str:
    for space in spaces:
        cache[space.slug] = space.id
    if slug not in cache:
        raise MapiError(
            f"no space with slug {slug!r}. Create it first: "
            f"client.spaces.create({slug!r})"
        )
    return cache[slug]





# -- async --------------------------------------------------------------------
#
# Separate classes rather than a shared body, because a resource parses its
# result: no single method can both `return Memory.parse(data)` and
# `return Memory.parse(await data)`. The URLs and request bodies come from the
# helpers above, so the part that would silently drift is shared and only the
# awaiting differs.


class AsyncMemories:
    """Writing and reading individual memories, awaited."""

    def __init__(self, request: Requester, space_id: Callable[[str], Any]) -> None:
        self._request = request
        self._space_id = space_id

    async def add(
        self,
        content: str,
        *,
        space: str,
        summary: str = "",
        tags: builtins.list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "",
        occurred_at: datetime | None = None,
        extract: bool = False,
        auto_supersede: bool = False,
        detect_conflicts: bool = False,
        dedupe: bool = True,
    ) -> Memory:
        data = await self._request(
            "POST",
            f"/v1/spaces/{await self._space_id(space)}/memories",
            json=_memory_body(
                content,
                summary=summary,
                tags=tags,
                metadata=metadata,
                source=source,
                occurred_at=occurred_at,
                extract=extract,
                auto_supersede=auto_supersede,
                detect_conflicts=detect_conflicts,
                dedupe=dedupe,
            ),
        )
        return Memory.parse((data or {}).get("memory", data) or {})

    async def add_many(
        self, items: builtins.list[dict[str, Any]], *, space: str
    ) -> builtins.list[Memory]:
        data = await self._request(
            "POST",
            f"/v1/spaces/{await self._space_id(space)}/memories/bulk",
            json={"items": items},
        )
        rows = (data or {}).get("items") or (data or {}).get("results") or []
        return [Memory.parse(r.get("memory", r)) for r in rows]

    async def get(self, memory_id: str, *, space: str) -> Memory:
        data = await self._request(
            "GET", f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}"
        )
        return Memory.parse(data or {})

    async def list(
        self, *, space: str, limit: int = 50, cursor: str | None = None
    ) -> builtins.list[Memory]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = await self._request(
            "GET", f"/v1/spaces/{await self._space_id(space)}/memories", params=params
        )
        return [Memory.parse(m) for m in (data or {}).get("items", [])]

    async def delete(self, memory_id: str, *, space: str) -> None:
        await self._request(
            "DELETE", f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}"
        )

    async def erase(self, memory_id: str, *, space: str) -> dict[str, Any]:
        result: dict[str, Any] = (
            await self._request(
                "POST",
                f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}/erase",
            )
            or {}
        )
        return result

    async def context(self, memory_id: str, *, space: str) -> MemoryContext:
        data = await self._request(
            "GET",
            f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}/context",
        )
        return MemoryContext.parse(data or {})

    async def lineage(self, memory_id: str, *, space: str) -> dict[str, Any]:
        result: dict[str, Any] = (
            await self._request(
                "GET",
                f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}/lineage",
            )
            or {}
        )
        return result

    async def versions(
        self, memory_id: str, *, space: str
    ) -> builtins.list[dict[str, Any]]:
        data = await self._request(
            "GET",
            f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}/versions",
        )
        return list((data or {}).get("items", []))

    async def relations(
        self, memory_id: str, *, space: str
    ) -> builtins.list[dict[str, Any]]:
        data = await self._request(
            "GET",
            f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}/relations",
        )
        return list((data or {}).get("items", []))

    async def relate(
        self,
        memory_id: str,
        *,
        space: str,
        target_id: str,
        relation: Relation,
        reason: str = "",
    ) -> dict[str, Any]:
        result: dict[str, Any] = (
            await self._request(
                "POST",
                f"/v1/spaces/{await self._space_id(space)}/memories/{memory_id}/relations",
                json={"target_id": target_id, "type": relation, "reason": reason},
            )
            or {}
        )
        return result


class AsyncSearch:
    """Hybrid retrieval, awaited."""

    def __init__(self, request: Requester, space_id: Callable[[str], Any]) -> None:
        self._request = request
        self._space_id = space_id

    async def execute(
        self,
        query: str,
        *,
        space: str,
        limit: int = 10,
        tags: builtins.list[str] | None = None,
        include_superseded: bool = False,
        min_score: float = 0.0,
        explain: bool = False,
        **options: Any,
    ) -> SearchResult:
        data = await self._request(
            "POST",
            f"/v1/spaces/{await self._space_id(space)}/search",
            json=_search_body(
                query,
                limit=limit,
                tags=tags,
                include_superseded=include_superseded,
                min_score=min_score,
                explain=explain,
                **options,
            ),
        )
        return SearchResult.parse(data or {})


class AsyncSpaces:
    """Spaces, awaited."""

    def __init__(self, request: Requester, cache: dict[str, str]) -> None:
        self._request = request
        self._cache = cache

    async def list(self) -> builtins.list[Space]:
        data = await self._request("GET", "/v1/spaces")
        spaces = [Space.parse(s) for s in (data or {}).get("items", [])]
        for space in spaces:
            self._cache[space.slug] = space.id
        return spaces

    async def create(self, slug: str, name: str | None = None, **extra: Any) -> Space:
        data = await self._request(
            "POST", "/v1/spaces", json={"slug": slug, "name": name or slug, **extra}
        )
        space = Space.parse(data or {})
        self._cache[space.slug] = space.id
        return space

    async def get(self, space_id: str) -> Space:
        return Space.parse(await self._request("GET", f"/v1/spaces/{space_id}") or {})

    async def get_or_create(self, slug: str, name: str | None = None) -> Space:
        for space in await self.list():
            if space.slug == slug:
                return space
        return await self.create(slug, name)

    async def delete(self, space_id: str) -> None:
        await self._request("DELETE", f"/v1/spaces/{space_id}")


class AsyncGraph:
    """The space as a graph, awaited."""

    def __init__(self, request: Requester, space_id: Callable[[str], Any]) -> None:
        self._request = request
        self._space_id = space_id

    async def get(self, *, space: str, limit: int = 300) -> dict[str, Any]:
        result: dict[str, Any] = (
            await self._request(
                "GET",
                f"/v1/spaces/{await self._space_id(space)}/graph",
                params={"limit": limit},
            )
            or {}
        )
        return result


__all__ = [
    "AsyncGraph",
    "AsyncMemories",
    "AsyncSearch",
    "AsyncSpaces",
    "Graph",
    "Memories",
    "Relation",
    "Search",
    "Spaces",
    "resolve_slug",
]
