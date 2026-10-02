"""v1 router assembly."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import space_access
from . import chat, keys, memories, search, spaces, synthesis

# A space-scoped key reaches only its spaces, on every route that names one.
api_router = APIRouter(prefix="/v1", dependencies=[Depends(space_access)])
api_router.include_router(spaces.router)
api_router.include_router(memories.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
api_router.include_router(synthesis.router)
api_router.include_router(keys.router)

__all__ = ["api_router"]
