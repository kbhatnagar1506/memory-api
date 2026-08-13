"""The clients. HTTP only — this package never imports the server.

That separation is the design, not an accident of layout: publishing a client
that imports the engine would publish the engine. Tests assert the import
graph, so the property survives someone reaching for a shared helper later.

`Mapi` and `AsyncMapi` are the same surface. The resource classes are written
once and handed a request callable, so URLs, bodies and parsing exist in one
place -- two hand-written implementations drift, and the async one always
drifts last and silently.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Self

import httpx

from ._errors import ConnectionError_, MapiError, from_response
from ._models import Space
from ._resources import (
    AsyncGraph,
    AsyncMemories,
    AsyncSearch,
    AsyncSpaces,
    Graph,
    Memories,
    Search,
    Spaces,
)

DEFAULT_BASE_URL = "https://memory-api-7b178bde9ecc.herokuapp.com"
__client_version__ = "0.1.1"

#: Retried automatically. 429 and the gateway family are transient by
#: definition. 500 is deliberately absent: a request that made the server
#: throw will usually throw again, and retrying only hides it.
RETRY_STATUS = frozenset({429, 502, 503, 504})


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _backoff(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, 30.0)
    # Jittered, so a fleet of clients retrying after one outage does not
    # arrive back in lockstep and cause the next one.
    delay: float = min(0.25 * (2**attempt), 8.0) * (0.5 + random.random())
    return delay


def _interpret(response: httpx.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return None
    try:
        payload = response.json()
    except ValueError:
        # Proxies return HTML 502s. Losing the status to a decode error would
        # replace a legible failure with a confusing one.
        payload = {}
    if response.is_success:
        return payload
    raise from_response(
        response.status_code,
        payload if isinstance(payload, dict) else {},
        _retry_after(response),
    )


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.getenv("MAPI_API_KEY", "")
    if not key:
        raise MapiError(
            "no API key: pass api_key=... or set MAPI_API_KEY. "
            "Create one in the dashboard under API keys."
        )
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Version in the agent, so a server log can identify which client
        # produced a shape it did not expect.
        "User-Agent": f"mapi-sdk/{__client_version__} (python)",
    }


class Mapi:
    """Synchronous client.

        from mapi_sdk import Mapi

        client = Mapi(api_key="sm_...")
        client.spaces.get_or_create("ada")
        client.memories.add("Prefers window seats", space="ada")

        for hit in client.search.execute("seating", space="ada"):
            print(hit.score, hit.content)

    `space` accepts a slug or an id everywhere. Slugs resolve once and cache,
    so the readable name costs one request per process rather than per call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = _resolve_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._space_cache: dict[str, str] = {}

        self.spaces = Spaces(self.request, self._space_cache)
        self.memories = Memories(self.request, self._space_id)
        self.search = Search(self.request, self._space_id)
        self.graph = Graph(self.request, self._space_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def request(self, method: str, path: str, **kw: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.request(
                    method, path, headers=_headers(self.api_key), **kw
                )
            except httpx.HTTPError as exc:
                last = ConnectionError_(f"could not reach {self.base_url}: {exc}")
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt, None))
                    continue
                raise last from exc
            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                time.sleep(_backoff(attempt, _retry_after(response)))
                continue
            return _interpret(response)
        raise last or MapiError("request failed")

    def _space_id(self, space: str) -> str:
        """Resolve a slug to an id, CREATING the space if it does not exist.

        Writing to a space that has not been created yet is not an error, it
        is the first write. Making the caller create one first is ceremony
        that exists only because the server keeps spaces in a table -- an
        implementation detail nobody should have to know about to store their
        first memory.

        Resolved once per process and cached, so the readable name costs one
        extra request on a cold client and nothing afterwards.
        """
        if space.startswith("spc_"):
            return space
        if space in self._space_cache:
            return self._space_cache[space]
        for existing in self.spaces.list():
            self._space_cache[existing.slug] = existing.id
        if space not in self._space_cache:
            created = self.spaces.create(space)
            self._space_cache[created.slug] = created.id
        return self._space_cache[space]

    # -- convenience -------------------------------------------------------
    # The two calls that make up most usage, without reaching through a
    # namespace. Thin delegates, so there is still one implementation.

    def add(self, content: str, *, space: str, **kw: Any) -> Any:
        return self.memories.add(content, space=space, **kw)

    def query(self, query: str, *, space: str, **kw: Any) -> Any:
        return self.search.execute(query, space=space, **kw)


class AsyncMapi:
    """Asynchronous client. Same surface as `Mapi`, awaited.

        client = AsyncMapi(api_key="sm_...")
        await client.memories.add("Prefers window seats", space="ada")
        hits = await client.search.execute("seating", space="ada")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = _resolve_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._space_cache: dict[str, str] = {}

        self.spaces = AsyncSpaces(self.request, self._space_cache)
        self.memories = AsyncMemories(self.request, self._space_id)
        self.search = AsyncSearch(self.request, self._space_id)
        self.graph = AsyncGraph(self.request, self._space_id)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def request(self, method: str, path: str, **kw: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http.request(
                    method, path, headers=_headers(self.api_key), **kw
                )
            except httpx.HTTPError as exc:
                last = ConnectionError_(f"could not reach {self.base_url}: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff(attempt, None))
                    continue
                raise last from exc
            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                await asyncio.sleep(_backoff(attempt, _retry_after(response)))
                continue
            return _interpret(response)
        raise last or MapiError("request failed")

    async def _space_id(self, space: str) -> str:
        """Resolve a slug to an id, creating the space if it does not exist."""
        if space.startswith("spc_"):
            return space
        if space in self._space_cache:
            return self._space_cache[space]
        payload = await self.request("GET", "/v1/spaces")
        for item in (payload or {}).get("items", []):
            existing = Space.parse(item)
            self._space_cache[existing.slug] = existing.id
        if space not in self._space_cache:
            created = await self.spaces.create(space)
            self._space_cache[created.slug] = created.id
        return self._space_cache[space]

    async def add(self, content: str, *, space: str, **kw: Any) -> Any:
        return await self.memories.add(content, space=space, **kw)

    async def query(self, query: str, *, space: str, **kw: Any) -> Any:
        return await self.search.execute(query, space=space, **kw)


__all__ = ["DEFAULT_BASE_URL", "AsyncMapi", "Mapi"]
