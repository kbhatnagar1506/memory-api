"""mapi — memory for AI agents.

    from mapi_sdk import Mapi

    mapi = Mapi(api_key="sm_...")          # or set MAPI_API_KEY
    mapi.get_or_create_space("ada")
    mapi.add("Prefers window seats", space="ada")

    for hit in mapi.search("seating preference", space="ada"):
        print(hit.score, hit.content)

This package speaks HTTP to the service and contains none of its
implementation. That is deliberate: publishing a client that imported the
engine would publish the engine.
"""

from ._client import DEFAULT_BASE_URL, Mapi
from ._errors import (
    AuthenticationError,
    ConflictError,
    MapiError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from ._models import Memory, MemoryContext, SearchHit, SearchResult, Space

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BASE_URL",
    "AuthenticationError",
    "ConflictError",
    "Mapi",
    "MapiError",
    "Memory",
    "MemoryContext",
    "NotFoundError",
    "RateLimitError",
    "SearchHit",
    "SearchResult",
    "ServerError",
    "Space",
    "ValidationError",
    "__version__",
]
