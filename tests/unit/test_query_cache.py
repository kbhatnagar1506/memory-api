"""The query-embedding cache (B8).

Query vectors used to share the document LRU under a `__query__` prefix and
the exact query text. Three things were wrong with that, each tested here:

  * a bulk ingest -- hundreds of chunks -- evicted every cached question, and
    the matcher re-asks the same facet questions every few minutes;
  * a `list[float]` entry is ~31 KB at 768 dimensions, so the cache that
    fits in memory is a tenth the size it could be as float32;
  * "who knows  Rust?" and "who knows Rust? " were two cache entries and two
    provider calls for one question.
"""

from __future__ import annotations

from array import array
from typing import Any

import pytest

from mapi.domain.embeddings.base import EmbeddingProvider, Vector, normalize_query


class _Fake(EmbeddingProvider):
    """Document embeddings from a counter; no network."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="fake", dimensions=4, max_attempts=1, **kwargs)
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    async def _embed_batch(self, texts: list[str]) -> list[Vector]:
        self.calls += 1
        return [[1.0, float(len(t)), 0.5, 0.25] for t in texts]


# -- normalization ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("who knows  Rust?", "who knows Rust?"),
        ("  who knows Rust?\n", "who knows Rust?"),
        ("tab\tseparated\N{NO-BREAK SPACE}words", "tab separated words"),
        ("\uff32\uff55\uff53\uff54", "Rust"),  # full-width letters
        ("ﬁle systems", "file systems"),  # ligature
    ],
)
def test_normalization_folds_what_a_keyboard_varies(raw: str, expected: str) -> None:
    assert normalize_query(raw) == expected


def test_normalization_keeps_case() -> None:
    """ "Go" the language and "go" the verb embed differently; merging them is wrong."""
    assert normalize_query("Go") != normalize_query("go")


def test_normalization_of_whitespace_is_empty() -> None:
    assert normalize_query(" \t\n ") == ""


# -- the cache itself ------------------------------------------------------------


def test_entries_are_stored_as_float32() -> None:
    fake = _Fake(query_cache_size=4)
    fake._query_cache_put("q", [0.1, 0.2, 0.3, 0.4])
    stored = next(iter(fake._query_cache.values()))
    assert isinstance(stored, array)
    assert stored.typecode == "f"


def test_a_hit_is_a_fresh_list_the_caller_may_mutate() -> None:
    fake = _Fake(query_cache_size=4)
    fake._query_cache_put("q", [0.5, 0.25, 0.125, 1.0])
    first = fake._query_cache_get("q")
    assert first == [0.5, 0.25, 0.125, 1.0]  # exactly representable in float32
    assert first is not None
    first[0] = 99.0
    assert fake._query_cache_get("q") == [0.5, 0.25, 0.125, 1.0]


def test_float32_keeps_cosine_where_it_was() -> None:
    from mapi.domain.embeddings.base import cosine_similarity, l2_normalize

    fake = _Fake(query_cache_size=4)
    vec = l2_normalize([0.123456789, -0.987654321, 0.5555555, 0.3333333])
    fake._query_cache_put("q", vec)
    back = fake._query_cache_get("q")
    assert back is not None
    assert cosine_similarity(vec, back) == pytest.approx(1.0, abs=1e-6)


def test_the_cache_is_a_bounded_lru() -> None:
    fake = _Fake(query_cache_size=2)
    fake._query_cache_put("a", [1.0, 0, 0, 0])
    fake._query_cache_put("b", [0, 1.0, 0, 0])
    assert fake._query_cache_get("a") is not None  # refresh a
    fake._query_cache_put("c", [0, 0, 1.0, 0])
    assert fake._query_cache_get("b") is None
    assert fake._query_cache_get("a") is not None
    assert fake._query_cache_get("c") is not None


def test_size_zero_disables_it() -> None:
    fake = _Fake(query_cache_size=0)
    fake._query_cache_put("a", [1.0, 0, 0, 0])
    assert fake._query_cache_get("a") is None


async def test_document_ingest_cannot_evict_queries() -> None:
    fake = _Fake(cache_size=2, query_cache_size=2)
    fake._query_cache_put("who knows rust?", [1.0, 0, 0, 0])
    await fake.embed([f"document chunk {i}" for i in range(50)])
    assert fake._query_cache_get("who knows rust?") is not None
    assert len(fake._cache) == 2


# -- wired into the Gemini embedder ------------------------------------------------


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("google.genai")
    from mapi.domain.embeddings.gemini import TASK_QUERY, GeminiEmbedder

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    embedder = GeminiEmbedder(
        api_key="not-a-real-key", dimensions=4, cache_size=8, query_cache_size=8
    )
    sent: list[tuple[list[str], str]] = []

    async def fake_call(
        texts: list[str], task_type: str, *, sdk_retry: bool = True
    ) -> list[Vector]:
        sent.append((list(texts), task_type))
        return [[1.0, 2.0, 3.0, 4.0] for _ in texts]

    monkeypatch.setattr(embedder, "_call", fake_call)
    embedder.sent = sent  # type: ignore[attr-defined]
    embedder.task_query = TASK_QUERY  # type: ignore[attr-defined]
    return embedder


async def test_spellings_of_one_question_cost_one_call(gemini: Any) -> None:
    first = await gemini.embed_query("who knows  Rust?")
    second = await gemini.embed_query(" who knows Rust?\n")
    assert first == pytest.approx(second)
    assert len(gemini.sent) == 1


async def test_the_normalized_text_is_what_gets_embedded(gemini: Any) -> None:
    """Key and vector must describe the same string, or a hit could lie."""
    await gemini.embed_query("  who\tknows   Rust? ")
    assert gemini.sent == [(["who knows Rust?"], gemini.task_query)]


async def test_query_vectors_live_outside_the_document_cache(gemini: Any) -> None:
    await gemini.embed_query("what am I stuck on")
    assert len(gemini._query_cache) == 1
    assert not any("stuck" in key for key in gemini._cache)


async def test_a_blank_query_is_refused(gemini: Any) -> None:
    with pytest.raises(ValueError):
        await gemini.embed_query(" \N{NO-BREAK SPACE} ")
    assert gemini.sent == []


def test_the_factory_sizes_the_query_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("google.genai")
    from mapi.config import Settings
    from mapi.domain.embeddings import build_embedder

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    embedder = build_embedder(
        Settings(
            embedding_backend="gemini",
            gemini_api_key="not-a-real-key",
            query_embedding_cache_size=123,
        )
    )
    assert embedder._query_cache_size == 123
