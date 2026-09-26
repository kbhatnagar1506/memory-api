"""Read-path storage behaviour, against both backends (B7, B9, B13).

The store changes behind the faster search path, each held to what it replaced:

  * `exclude_metadata` must mean the same thing in SQL as in `MemoryFilter`.
  * Hydrating without vectors must return the same memories, minus only the
    vectors -- chunk count and text intact, since responses report both.
  * `chunk_embedding` (the source side of `/similar`) returns what was stored.
  * Binary vector transfer must round-trip exactly what text transfer did.
  * One `set_config` SELECT must set exactly what the four statements did.
  * OR-mode full-text search must never raise on anything a user types.

Postgres runs when MAPI_TEST_DATABASE_URL is set; everything that has an
in-memory meaning runs there too.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from sqlalchemy import event, text
from tests.support.factories import memory as build_memory
from tests.support.pg import DIMENSIONS, make_store, shared_postgres_store

from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Memory, Organization, Space
from mapi.store.base import MemoryFilter
from mapi.store.memory import InMemoryStore

_BACKENDS = [
    pytest.param("memory", id="memory"),
    pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
]

EMBEDDER = DeterministicEmbedder(dimensions=DIMENSIONS)


@pytest.fixture(params=_BACKENDS)
async def backend(request: pytest.FixtureRequest) -> Any:
    if request.param == "memory":
        return InMemoryStore()
    return await shared_postgres_store()


@pytest.fixture
async def postgres() -> Any:
    return await shared_postgres_store()


async def _tenant(store: Any) -> tuple[Organization, Space]:
    org = await store.create_organization(Organization(name="Read path"))
    space = await store.create_space(Space(org_id=org.id, slug=f"r{org.id[-10:]}", name="S"))
    return org, space


async def _add(store: Any, org: Organization, space: Space, text_: str, **kw: Any) -> Memory:
    chunk_texts = kw.pop("chunk_texts", None)
    if chunk_texts is not None:
        vectors = [await EMBEDDER.embed_one(t) for t in chunk_texts]
        built = build_memory(
            org_id=org.id,
            space_id=space.id,
            content=text_,
            chunk_texts=chunk_texts,
            chunk_vectors=vectors,
            **kw,
        )
    else:
        built = build_memory(
            org_id=org.id,
            space_id=space.id,
            content=text_,
            vector=await EMBEDDER.embed_one(text_),
            **kw,
        )
    return await store.upsert_memory(built)


# -- exclude_metadata parity ----------------------------------------------------------


async def test_exclusion_is_the_same_in_every_read(backend) -> None:
    org, space = await _tenant(backend)
    mine = await _add(backend, org, space, "rust lifetimes card", metadata={"user_id": "me"})
    both = await _add(
        backend, org, space, "rust async card", metadata={"user_id": "me", "event": "x"}
    )
    theirs = await _add(backend, org, space, "rust tooling card", metadata={"user_id": "you"})
    bare = await _add(backend, org, space, "rust without metadata")

    one = MemoryFilter(exclude_metadata=(("user_id", "me"),))
    pair = MemoryFilter(exclude_metadata=(("event", "x"), ("user_id", "me")))
    vector = await EMBEDDER.embed_one("rust card")

    for rule, expected in ((one, {theirs.id, bare.id}), (pair, {mine.id, theirs.id, bare.id})):
        listed = await backend.list_memories(org.id, space.id, filters=rule, limit=50)
        assert {m.id for m in listed.items} == expected
        hits = await backend.vector_search(org.id, space.id, vector, limit=50, filters=rule)
        assert {h.memory_id for h in hits} == expected
        lexical = await backend.lexical_search(org.id, space.id, "rust", limit=50, filters=rule)
        assert {h.memory_id for h in lexical} == expected
        assert await backend.count_memories(org.id, space.id, filters=rule) == len(expected)
    assert both.id not in {theirs.id, bare.id}


# -- hydrating without vectors --------------------------------------------------------


async def test_hydrate_without_vectors_keeps_everything_else(backend) -> None:
    org, space = await _tenant(backend)
    stored = await _add(
        backend,
        org,
        space,
        "first part. second part.",
        chunk_texts=["first part.", "second part."],
        tags=["t1"],
        metadata={"k": 1},
    )
    full = (await backend.get_memories(org.id, space.id, [stored.id]))[stored.id]
    lean = (await backend.get_memories(org.id, space.id, [stored.id], with_embeddings=False))[
        stored.id
    ]
    assert [c.text for c in lean.chunks] == [c.text for c in full.chunks]
    assert len(lean.chunks) == 2
    assert lean.model_copy(update={"chunks": []}) == full.model_copy(update={"chunks": []})
    assert all(c.embedding is not None for c in full.chunks)
    if not isinstance(backend, InMemoryStore):
        assert all(c.embedding is None for c in lean.chunks)


async def test_neighbours_carry_the_matched_vector_but_not_the_rest(postgres) -> None:
    org, space = await _tenant(postgres)
    stored = await _add(
        postgres, org, space, "alpha beta. gamma delta.", chunk_texts=["alpha beta.", "gamma."]
    )
    probe = await EMBEDDER.embed_one("alpha beta.")
    [(memory, vector)] = await postgres.neighbours(org.id, space.id, probe, limit=1)
    assert memory.id == stored.id
    assert len(vector) == DIMENSIONS
    assert _cos(vector, probe) == pytest.approx(1.0, abs=1e-5)
    assert [c.text for c in memory.chunks] == ["alpha beta.", "gamma."]
    assert all(c.embedding is None for c in memory.chunks)


# -- chunk_embedding --------------------------------------------------------------------


async def test_chunk_embedding_returns_the_stored_vector(backend) -> None:
    org, space = await _tenant(backend)
    stored = await _add(backend, org, space, "one. two.", chunk_texts=["one.", "two."])
    first = await backend.chunk_embedding(org.id, space.id, stored.id)
    second = await backend.chunk_embedding(org.id, space.id, stored.id, ordinal=1)
    assert first is not None and second is not None
    assert _cos(first, await EMBEDDER.embed_one("one.")) == pytest.approx(1.0, abs=1e-5)
    assert _cos(second, await EMBEDDER.embed_one("two.")) == pytest.approx(1.0, abs=1e-5)
    assert await backend.chunk_embedding(org.id, space.id, stored.id, ordinal=7) is None


async def test_chunk_embedding_is_tenant_scoped(backend) -> None:
    org, space = await _tenant(backend)
    stored = await _add(backend, org, space, "a secret card")
    stranger, _ = await _tenant(backend)
    assert await backend.chunk_embedding(stranger.id, space.id, stored.id) is None
    other_space = await backend.create_space(Space(org_id=org.id, slug="other", name="O"))
    assert await backend.chunk_embedding(org.id, other_space.id, stored.id) is None


# -- binary vectors -------------------------------------------------------------------------


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def test_vectors_round_trip_at_float32(postgres) -> None:
    assert postgres.binary_vectors is True
    org, space = await _tenant(postgres)
    stored = await _add(postgres, org, space, "round trip me")
    original = stored.chunks[0].embedding
    assert original is not None
    fetched = await postgres.get_memory(org.id, space.id, stored.id)
    assert fetched is not None
    back = fetched.chunks[0].embedding
    assert back is not None
    assert len(back) == len(original)
    assert max(abs(x - y) for x, y in zip(back, original, strict=True)) < 1e-6


async def test_a_text_vector_still_binds(postgres) -> None:
    """The codec parses pgvector's text form rather than refusing it."""
    literal = "[" + ",".join(["0.5"] * DIMENSIONS) + "]"
    async with postgres._session() as session:
        value = await session.scalar(
            text("SELECT vector_dims(CAST(:v AS vector))"), {"v": literal}
        )
    assert value == DIMENSIONS


async def test_the_text_path_still_works(postgres) -> None:
    """`db_binary_vectors=false` is a real escape hatch, not a dead flag."""
    text_store = make_store(binary_vectors=False)
    try:
        org, space = await _tenant(text_store)
        stored = await _add(text_store, org, space, "written as text")
        probe = await EMBEDDER.embed_one("written as text")
        hits = await text_store.vector_search(
            org.id, space.id, probe, limit=1, filters=MemoryFilter()
        )
        assert [h.memory_id for h in hits] == [stored.id]
        # And what one path wrote, the other reads.
        fetched = await postgres.get_memory(org.id, space.id, stored.id)
        assert fetched is not None and fetched.chunks[0].embedding is not None
    finally:
        await text_store.aclose()


# -- one set_config ----------------------------------------------------------------------


async def test_a_probed_ann_search_sends_one_setup_statement(postgres) -> None:
    org, space = await _tenant(postgres)
    await _add(postgres, org, space, "count my statements")
    probe = await EMBEDDER.embed_one("count my statements")
    # First search probes pgvector's capabilities.
    await postgres.vector_search(org.id, space.id, probe, limit=5, filters=MemoryFilter())

    statements: list[str] = []

    def record(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        statements.append(statement)

    engine = postgres._engine.sync_engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        hits = await postgres.vector_search(
            org.id, space.id, probe, limit=5, filters=MemoryFilter()
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert hits
    setup = [s for s in statements if "set_config" in s or s.lstrip().upper().startswith("SET")]
    assert len(setup) == 1, statements
    assert len(statements) == 2, statements  # the setup, then the search


async def test_the_combined_statement_sets_what_the_four_did(postgres) -> None:
    org, _ = await _tenant(postgres)
    postgres._ef_search_supported = True
    postgres._iterative_scan_supported = True
    async with postgres._session() as session:
        await postgres._scope_for_ann(session, org.id, 100)
        row = (
            await session.execute(
                text(
                    "SELECT current_setting('app.org_id'), "
                    "current_setting('hnsw.ef_search'), "
                    "current_setting('hnsw.iterative_scan'), "
                    "current_setting('hnsw.max_scan_tuples')"
                )
            )
        ).one()
    assert tuple(row) == (org.id, "300", "relaxed_order", "20000")


async def test_the_settings_are_transaction_local(postgres) -> None:
    """Pooled connections must not carry one tenant's scope into the next."""
    org, _ = await _tenant(postgres)
    async with postgres._session() as session:
        await postgres._scope_for_ann(session, org.id, 10)
        await session.commit()
        leaked = await session.scalar(text("SELECT current_setting('app.org_id', true)"))
    assert leaked in (None, "")


# -- OR-mode full text --------------------------------------------------------------------


@pytest.fixture
async def or_store(postgres) -> Any:
    store = make_store(lexical_mode="or")
    yield store
    await store.aclose()


async def test_or_mode_matches_any_word_and_ranks_by_how_many(postgres, or_store) -> None:
    org, space = await _tenant(postgres)
    one = await _add(postgres, org, space, "the rust compiler is strict")
    two = await _add(postgres, org, space, "rust lifetimes confuse the borrow checker")
    await _add(postgres, org, space, "figma files for the design review")
    question = "who can help me with rust borrow checker errors"

    strict = await postgres.lexical_search(
        org.id, space.id, question, limit=10, filters=MemoryFilter()
    )
    assert strict == []  # AND: nobody's card holds every word of a question

    loose = await or_store.lexical_search(
        org.id, space.id, question, limit=10, filters=MemoryFilter()
    )
    assert [h.memory_id for h in loose] == [two.id, one.id]


@pytest.mark.parametrize(
    "query",
    [
        "it's",
        "a&b|c",
        "!!!",
        "(unbalanced",
        "prefix:*",
        "back\\slash",
        "'quoted' \"double\"",
        "e-mail node.js C++",
        "the and of",  # stopwords only
        "\U0001f980 rust été",
        " ".join(f"word{i}" for i in range(200)),
    ],
)
async def test_or_mode_never_raises(postgres, or_store, query: str) -> None:
    org, space = await _tenant(postgres)
    await _add(postgres, org, space, "rust and e-mail and node.js and it's fine")
    await or_store.lexical_search(org.id, space.id, query, limit=5, filters=MemoryFilter())


async def test_and_mode_is_still_the_default(postgres) -> None:
    assert postgres.lexical_mode == "and"
