"""Storage backends."""

from __future__ import annotations

from ..config import Settings, StoreBackend
from ..core.errors import ConfigurationError
from .base import LexicalHit, MemoryFilter, MemoryStore, Page, VectorHit
from .memory import InMemoryStore


def build_store(settings: Settings) -> MemoryStore:
    if settings.store_backend is StoreBackend.MEMORY:
        return InMemoryStore()
    if settings.store_backend is StoreBackend.POSTGRES:
        from .postgres.store import PostgresStore  # noqa: PLC0415

        if not settings.database_url:
            raise ConfigurationError("database_url is required for the postgres backend")
        return PostgresStore(
            settings.database_url,
            dimensions=settings.embedding_dimensions,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            statement_timeout_ms=settings.db_statement_timeout_ms,
        )
    raise ConfigurationError(f"unknown store backend: {settings.store_backend}")


__all__ = [
    "InMemoryStore", "LexicalHit", "MemoryFilter", "MemoryStore", "Page",
    "VectorHit", "build_store",
]
