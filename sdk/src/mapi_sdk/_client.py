"""The client. HTTP only — this package never imports the server.

That separation is the point, not an accident of layout: publishing a client
that imports the engine would publish the engine. There is a test asserting
this module's import graph touches nothing named `mapi.*`, so the property
survives someone reaching for a convenient shared helper later.
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime
from typing import Any, Literal, Self

import httpx

from ._errors import ConnectionError_, MapiError, from_response
from ._models import Memory, MemoryContext, SearchResult, Space

DEFAULT_BASE_URL = "https://memory-api-7b178bde9ecc.herokuapp.com"

#: Retried automatically. 429 and the gateway family are transient by
#: definition; 500 is not on this list, because a request that made the
#: server throw will usually make it throw again, and retrying hides it.
_RETRY_STATUS = frozenset({429, 502, 503, 504})


class _Transport:
    """Shared request logic. Retries, error mapping, and nothing else."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        key = api_key or os.getenv("MAPI_API_KEY", "")
        if not key:
            raise MapiError(
                "no API key: pass api_key=... or set MAPI_API_KEY. "
                "Create one in the dashboard under API keys."
            )
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Version in the agent so a server-side log can tell which client
            # produced a shape it did not expect.
            "User-Agent": "mapi-sdk/0.1.0 (python)",
        }

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, 30.0)
        # Jittered, so a fleet of clients retrying after one outage does not
        # arrive back in lockstep and cause the next one.
        delay: float = min(0.25 * (2**attempt), 8.0) * (0.5 + random.random())
        return delay

    def interpret(self, response: httpx.Response) -> Any:
        if response.status_code == 204 or not response.content:
            return None
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.is_success:
            return payload
        retry_after = _retry_after(response)
        raise from_response(
            response.status_code,
            payload if isinstance(payload, dict) else {},
            retry_after,
        )


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _memory_body(
    content: str,
    *,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    source: str,
    occurred_at: datetime | None,
    extract: bool,
    auto_supersede: bool,
    detect_conflicts: bool,
    dedupe: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {"content": content, "dedupe": dedupe}
    if tags:
        body["tags"] = tags
    if metadata:
        body["metadata"] = metadata
    if source:
        body["source"] = source
    if occurred_at is not None:
        body["occurred_at"] = occurred_at.isoformat()
    if extract:
        body["extract"] = True
    if auto_supersede:
        body["auto_supersede"] = True
    if detect_conflicts:
        body["detect_conflicts"] = True
    return body


class Mapi:
    """Synchronous client.

        mapi = Mapi(api_key="sm_...")
        mapi.add("Prefers window seats", space="ada")
        for hit in mapi.search("seating preference", space="ada"):
            print(hit.score, hit.content)

    `space` accepts a slug or an id. Slugs are resolved once and cached, so
    passing the human-readable name costs one extra request per process
    rather than one per call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._t = _Transport(api_key, base_url, timeout, max_retries)
        self._http = httpx.Client(base_url=self._t.base_url, timeout=timeout)
        self._spaces: dict[str, str] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -- plumbing ----------------------------------------------------------

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self._t.max_retries + 1):
            try:
                response = self._http.request(
                    method, path, headers=self._t.headers, **kw
                )
            except httpx.HTTPError as exc:
                last = ConnectionError_(f"could not reach {self._t.base_url}: {exc}")
                if attempt < self._t.max_retries:
                    time.sleep(self._t.backoff(attempt, None))
                    continue
                raise last from exc
            if response.status_code in _RETRY_STATUS and attempt < self._t.max_retries:
                time.sleep(self._t.backoff(attempt, _retry_after(response)))
                continue
            return self._t.interpret(response)
        raise last or MapiError("request failed")

    def _space_id(self, space: str) -> str:
        """Resolve a slug to an id, or pass an id straight through."""
        if space.startswith("spc_"):
            return space
        if space in self._spaces:
            return self._spaces[space]
        for item in self.spaces():
            self._spaces[item.slug] = item.id
        if space not in self._spaces:
            raise MapiError(
                f"no space with slug {space!r}. Create it first: "
                f"client.create_space({space!r})"
            )
        return self._spaces[space]

    # -- spaces ------------------------------------------------------------

    def spaces(self) -> list[Space]:
        data = self._request("GET", "/v1/spaces")
        return [Space.parse(s) for s in (data or {}).get("items", [])]

    def create_space(self, slug: str, name: str | None = None) -> Space:
        data = self._request(
            "POST", "/v1/spaces", json={"slug": slug, "name": name or slug}
        )
        space = Space.parse(data or {})
        self._spaces[space.slug] = space.id
        return space

    def get_or_create_space(self, slug: str, name: str | None = None) -> Space:
        for item in self.spaces():
            if item.slug == slug:
                self._spaces[item.slug] = item.id
                return item
        return self.create_space(slug, name)

    # -- memories ----------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        space: str,
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
        difference matters for anything asking "what order did these come
        in", and defaulting to now quietly makes a backfill look like it all
        happened today.
        """
        body = _memory_body(
            content,
            tags=tags,
            metadata=metadata,
            source=source,
            occurred_at=occurred_at,
            extract=extract,
            auto_supersede=auto_supersede,
            detect_conflicts=detect_conflicts,
            dedupe=dedupe,
        )
        data = self._request(
            "POST", f"/v1/spaces/{self._space_id(space)}/memories", json=body
        )
        payload = (data or {}).get("memory", data)
        return Memory.parse(payload or {})

    def add_many(self, items: list[dict[str, Any]], *, space: str) -> list[Memory]:
        """Store up to 100 memories in one request."""
        data = self._request(
            "POST",
            f"/v1/spaces/{self._space_id(space)}/memories/bulk",
            json={"items": items},
        )
        results = (data or {}).get("items") or (data or {}).get("results") or []
        return [Memory.parse(r.get("memory", r)) for r in results]

    def get(self, memory_id: str, *, space: str) -> Memory:
        data = self._request(
            "GET", f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}"
        )
        return Memory.parse(data or {})

    def delete(self, memory_id: str, *, space: str) -> None:
        self._request(
            "DELETE", f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}"
        )

    # -- reading -----------------------------------------------------------

    def search(
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
        """Hybrid search: vector and lexical, fused.

        Superseded memories are excluded by default -- a fact that has been
        replaced is still stored and still reachable, but returning it
        alongside its replacement is how an agent states last month's answer
        with this month's confidence.
        """
        body: dict[str, Any] = {"query": query, "limit": limit, **options}
        if tags:
            body["tags"] = tags
        if include_superseded:
            body["include_superseded"] = True
        if min_score:
            body["min_score"] = min_score
        if explain:
            body["explain"] = True
        data = self._request(
            "POST", f"/v1/spaces/{self._space_id(space)}/search", json=body
        )
        return SearchResult.parse(data or {})

    def context(self, memory_id: str, *, space: str) -> MemoryContext:
        """One memory plus every typed relation touching it.

        This is the call that answers "why does the system believe this":
        what replaced it, what it replaced, what it was computed from, and
        what disagrees with it.
        """
        data = self._request(
            "GET",
            f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/context",
        )
        return MemoryContext.parse(data or {})

    def relate(
        self,
        memory_id: str,
        *,
        space: str,
        target_id: str,
        relation: Literal["supersedes", "contradicts", "derived_from", "references"],
        reason: str = "",
    ) -> dict[str, Any]:
        edge: dict[str, Any] = (
            self._request(
                "POST",
                f"/v1/spaces/{self._space_id(space)}/memories/{memory_id}/relations",
                json={"target_id": target_id, "type": relation, "reason": reason},
            )
            or {}
        )
        return edge

    def graph(self, *, space: str, limit: int = 300) -> dict[str, Any]:
        """The space as memories plus the typed edges between them."""
        result = self._request(
            "GET", f"/v1/spaces/{self._space_id(space)}/graph", params={"limit": limit}
        )
        out: dict[str, Any] = dict(result or {})
        return out


__all__ = ["DEFAULT_BASE_URL", "Mapi"]
