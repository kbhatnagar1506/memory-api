"""mapi — memory for AI agents.

    from mapi_sdk import Mapi

    client = Mapi(api_key="sm_...")            # or set MAPI_API_KEY
    client.spaces.get_or_create("ada")
    client.memories.add("Prefers window seats", space="ada")

    for hit in client.search.execute("seating preference", space="ada"):
        print(hit.score, hit.content)

Async is the same surface, awaited:

    from mapi_sdk import AsyncMapi

    client = AsyncMapi()
    await client.memories.add("Prefers window seats", space="ada")

This package speaks HTTP to the service and contains none of its
implementation. That is deliberate: publishing a client that imported the
engine would publish the engine.
"""

from ._client import DEFAULT_BASE_URL, AsyncMapi, Mapi
from ._errors import (
    AuthenticationError,
    ConflictError,
    ConnectionError_,
    MapiError,
    NotFoundError,
    PermissionError_,
    RateLimitError,
    ServerError,
    ValidationError,
)
from ._models import Memory, MemoryContext, SearchHit, SearchResult, Space
from ._resources import Graph, Memories, Search, Spaces

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BASE_URL",
    "AsyncMapi",
    "AuthenticationError",
    "ConflictError",
    "ConnectionError_",
    "Graph",
    "Mapi",
    "MapiError",
    "Memories",
    "Memory",
    "MemoryContext",
    "NotFoundError",
    "PermissionError_",
    "RateLimitError",
    "Search",
    "SearchHit",
    "SearchResult",
    "ServerError",
    "Space",
    "Spaces",
    "ValidationError",
    "__version__",
]
