"""How the Postgres store talks to the database, checked without one (B7, B13).

Connection-pool policy, the binary vector switch and the OR-mode query text
are all decided before a statement runs, so all of them can be verified by
building the objects and reading what they would do. The round trips they
exist to save are measured in `tests/conformance/test_read_path_contract.py`,
which needs a database.
"""

from __future__ import annotations

import pytest
from pgvector import Vector as PgVector
from sqlalchemy.dialects import postgresql

from mapi.config import Settings
from mapi.store import build_store
from mapi.store.postgres.store import PostgresStore, _or_tsquery_text
from mapi.store.postgres.vector import BINARY_VECTORS_FLAG, BinaryVector, _encode

URL = "postgresql+asyncpg://user:pass@127.0.0.1:1/none"


# -- pool policy ---------------------------------------------------------------------


def test_the_pool_recycles_instead_of_pinging() -> None:
    store = PostgresStore(URL)
    pool = store._engine.sync_engine.pool
    assert pool._pre_ping is False
    assert pool._recycle == 1800


def test_the_settings_reach_the_store() -> None:
    settings = Settings(
        store_backend="postgres",
        database_url=URL,
        db_pool_recycle_s=300,
        db_pool_pre_ping=True,
        lexical_mode="or",
        db_binary_vectors=False,
    )
    store = build_store(settings)
    assert isinstance(store, PostgresStore)
    pool = store._engine.sync_engine.pool
    assert pool._pre_ping is True
    assert pool._recycle == 300
    assert store.lexical_mode == "or"
    assert store.binary_vectors is False


def test_an_unknown_lexical_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        PostgresStore(URL, lexical_mode="xor")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Settings(lexical_mode="xor")


def test_server_settings_cannot_override_the_statement_timeout() -> None:
    store = PostgresStore(
        URL,
        statement_timeout_ms=1234,
        server_settings={"statement_timeout": "0", "application_name": "mapi-read"},
    )
    assert store.server_settings == {
        "statement_timeout": "1234",
        "application_name": "mapi-read",
    }


# -- binary vectors ---------------------------------------------------------------------


def test_the_binary_flag_is_on_the_dialect_by_default() -> None:
    assert getattr(PostgresStore(URL)._engine.sync_engine.dialect, BINARY_VECTORS_FLAG) is True
    off = PostgresStore(URL, binary_vectors=False)
    assert getattr(off._engine.sync_engine.dialect, BINARY_VECTORS_FLAG, False) is False


def test_binds_pass_lists_through_when_binary() -> None:
    dialect = postgresql.dialect()
    setattr(dialect, BINARY_VECTORS_FLAG, True)
    process = BinaryVector(3).bind_processor(dialect)
    assert process is not None
    vec = [0.1, 0.2, 0.3]
    assert process(vec) is vec
    assert process((0.1, 0.2, 0.3)) == vec
    assert process(None) is None


def test_binds_render_text_when_not_binary() -> None:
    process = BinaryVector(3).bind_processor(postgresql.dialect())
    assert process is not None
    assert process([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"


def test_the_encoder_takes_every_shape_it_can_meet() -> None:
    expected = PgVector([1.0, 2.5, -3.0]).to_binary()
    assert _encode([1.0, 2.5, -3.0]) == expected
    assert _encode(PgVector([1.0, 2.5, -3.0])) == expected
    assert _encode("[1,2.5,-3]") == expected
    assert _encode((1.0, 2.5, -3.0)) == expected


def test_decoded_vectors_come_back_as_lists() -> None:
    process = BinaryVector(3).result_processor(postgresql.dialect(), None)
    decoded = PgVector.from_binary(PgVector([0.5, 0.25, 1.0]).to_binary())
    assert process(decoded) == [0.5, 0.25, 1.0]


# -- OR-mode query text -----------------------------------------------------------------


def test_each_word_is_quoted_and_ored() -> None:
    assert _or_tsquery_text("rust borrow checker") == "'rust' | 'borrow' | 'checker'"


def test_syntax_characters_are_inert_inside_quotes() -> None:
    assert _or_tsquery_text("a&b !c (d") == "'a&b' | '!c' | '(d'"


def test_quotes_and_backslashes_are_escaped() -> None:
    assert _or_tsquery_text("it's") == "'it''s'"
    assert _or_tsquery_text("back\\slash") == "'back\\\\slash'"


def test_repeats_collapse_and_the_length_is_bounded() -> None:
    assert _or_tsquery_text("rust rust Rust") == "'rust' | 'Rust'"
    many = _or_tsquery_text(" ".join(f"w{i}" for i in range(500)))
    assert many is not None
    assert many.count("|") == 63


def test_a_blank_query_builds_nothing() -> None:
    assert _or_tsquery_text("   ") is None
