"""v1 router assembly."""

from __future__ import annotations

from fastapi import APIRouter

from . import keys, memories, search, spaces

api_router = APIRouter(prefix="/v1")
api_router.include_router(spaces.router)
api_router.include_router(memories.router)
api_router.include_router(search.router)
api_router.include_router(keys.router)

__all__ = ["api_router"]
