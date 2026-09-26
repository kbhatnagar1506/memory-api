"""The default embedding model is one the Gemini Developer API still serves.

text-embedding-004 answers 404 on the Developer API. On the write path that
fails every ingest; on the read path it is worse, because a failed query
embedding degrades search to lexical-only without an error. So the default
must be a live model, and it must still fit the Vector(768) column the
migrations created: switching models is otherwise a re-embed and a migration.
"""

from __future__ import annotations

import inspect

import pytest

from mapi.config import Settings
from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.store.postgres.models import Base

LIVE_MODEL = "gemini-embedding-001"


def test_settings_default_to_a_live_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAPI_EMBEDDING_MODEL", raising=False)
    assert Settings(_env_file=None).embedding_model == LIVE_MODEL


def test_the_embedder_constructor_agrees_with_settings() -> None:
    """Callers that build GeminiEmbedder directly (bench, labs, the live test
    lane) must not silently get a different model than the server does."""
    default = inspect.signature(GeminiEmbedder.__init__).parameters["model"].default
    assert default == LIVE_MODEL


def test_the_default_dimensions_fit_the_migrated_vector_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAPI_EMBEDDING_DIMENSIONS", raising=False)
    column = Base.metadata.tables["chunks"].c.embedding
    assert Settings(_env_file=None).embedding_dimensions == column.type.dim == 768


def test_an_explicit_model_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAPI_EMBEDDING_MODEL", "text-embedding-005")
    assert Settings(_env_file=None).embedding_model == "text-embedding-005"
