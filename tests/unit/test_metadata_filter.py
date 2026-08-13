"""Metadata filtering: the reference semantics, and the SQL that must match it.

`MemoryFilter.matches` carries the line "Reference semantics. SQL backends must
reproduce this exactly." The Postgres backend reproduced it for two JSON types
out of eight, and the failure was silent in the worst direction: the filter
matched nothing, so a search returned zero results and looked like an empty
space rather than a broken predicate.

    filter value   stored JSON text   compared against   matched
    "prod"         'prod'             str("prod")        yes
    3              '3'                str(3)             yes
    True           'true'             str(True) 'True'   NO
    None           SQL NULL           'None'             NO
    1.0            '1'                '1.0'              NO
    {"b": 1}       '{"b": 1}'         "{'b': 1}" repr    NO
    ["a", "b"]     '["a", "b"]'       "['a', 'b']"       NO

Reachable from the product: `POST /search` takes `metadata` as `dict[str, Any]`
and the search body has no shallowness validator, so booleans, nulls, objects
and arrays all reach the store. A caller filtering on `{"verified": true}` got
the right hits in any in-memory dev run and zero rows against Postgres.

The conformance suite could not catch it. Its only metadata fixture is
`{"k": "v"}` -- one of the two types that agreed -- and it asserted round-tripping
rather than filtering.

WHY THIS FILE NEEDS NO DATABASE. SQLAlchemy compiles a predicate to SQL without
connecting, so the Postgres half is verified here by compiling the statement and
reading the operator. That runs everywhere, including on a laptop with no Docker.
The round-trip parity matrix in `tests/conformance/` covers the other half and
needs the container.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.support.factories import memory as build_memory

from mapi.domain.models import MemoryStatus
from mapi.store.base import MemoryFilter

#: Every JSON type a caller can put in a metadata filter, with a value that is
#: NOT its own Python repr wherever the two can differ. That divergence is the
#: whole bug, so the table is built to expose it rather than to avoid it.
VALUES: tuple[tuple[str, Any], ...] = (
    ("string", "production"),
    ("integer", 3),
    ("zero", 0),
    ("float", 1.5),
    ("bool_true", True),
    ("bool_false", False),
    ("none", None),
    ("empty_string", ""),
    ("nested_dict", {"region": "us-central1", "replicas": 2}),
    ("list", ["alpha", "beta"]),
    ("empty_list", []),
    ("unicode", "café-naïve"),
)


# -- the reference semantics ------------------------------------------------


@pytest.mark.parametrize(("label", "value"), VALUES, ids=[v[0] for v in VALUES])
def test_the_reference_filter_matches_a_stored_value(label: str, value: Any) -> None:
    """`MemoryFilter.matches` is the contract. Establish it before comparing."""
    stored = build_memory(
        org_id="org_x", space_id="spc_x", content="a fact", metadata={label: value}
    )
    assert MemoryFilter(metadata=((label, value),)).matches(stored)


@pytest.mark.parametrize(("label", "value"), VALUES, ids=[v[0] for v in VALUES])
def test_the_reference_filter_rejects_a_different_value(label: str, value: Any) -> None:
    """And must not match everything, or agreement is meaningless."""
    other = "a value nothing else uses" if value != "a value nothing else uses" else "x"
    stored = build_memory(
        org_id="org_x", space_id="spc_x", content="a fact", metadata={label: other}
    )
    assert not MemoryFilter(metadata=((label, value),)).matches(stored)


def test_a_missing_key_does_not_match_a_none_filter() -> None:
    """The one place the reference semantics are genuinely surprising.

    `memory.metadata.get(key)` returns None for an ABSENT key, so filtering on
    `{"x": None}` matches memories that never had an `x` at all. Recorded as a
    known property rather than quietly relied upon -- JSONB containment does NOT
    behave this way, and this is the one row of the matrix where the two backends
    are permitted to differ.
    """
    without = build_memory(org_id="org_x", space_id="spc_x", content="a fact")
    assert MemoryFilter(metadata=(("x", None),)).matches(without)


def test_all_metadata_pairs_must_match() -> None:
    stored = build_memory(
        org_id="org_x",
        space_id="spc_x",
        content="a fact",
        metadata={"env": "prod", "verified": True},
    )
    assert MemoryFilter(metadata=(("env", "prod"), ("verified", True))).matches(stored)
    assert not MemoryFilter(metadata=(("env", "prod"), ("verified", False))).matches(stored)


# -- the SQL, compiled without a database ----------------------------------


def _compiled(value: Any) -> tuple[str, dict[str, Any]]:
    """The WHERE clause the Postgres store builds: SQL text plus bound params.

    Not `literal_binds` -- SQLAlchemy has no literal renderer for JSONB, which is
    itself informative: the old predicate could be rendered inline because it had
    reduced the value to a plain string first, and that reduction was the bug.
    The operator is read from the SQL and the value from the params.
    """
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql

    from mapi.store.postgres.models import MemoryRow
    from mapi.store.postgres.store import PostgresStore

    stmt = PostgresStore._apply_filters(
        None,  # type: ignore[arg-type]
        select(MemoryRow.id),
        MemoryFilter(statuses=frozenset({MemoryStatus.ACTIVE}), metadata=(("k", value),)),
    )
    compiled = stmt.compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


@pytest.mark.parametrize(("label", "value"), VALUES, ids=[v[0] for v in VALUES])
def test_the_sql_uses_containment_not_text_comparison(label: str, value: Any) -> None:
    """The fix, asserted at the operator level.

    `@>` compares JSON to JSON. `->> = '...'` compared JSON text against a Python
    `str()`, which is why five of these twelve types matched nothing at all.
    """
    sql, _params = _compiled(value)
    assert "@>" in sql, f"{label}: expected JSONB containment, got:\n{sql}"
    assert "->>" not in sql, f"{label}: still comparing JSON text:\n{sql}"


@pytest.mark.parametrize(("label", "value"), VALUES, ids=[v[0] for v in VALUES])
def test_the_bound_value_stays_a_native_json_type(label: str, value: Any) -> None:
    """The other half: the value must reach the driver un-stringified.

    A `str()` anywhere in this path is the bug, so the parameter has to still be
    the Python object the caller passed -- `True`, not `'True'`; `{"b": 1}`, not
    `"{'b': 1}"`. The JSONB type then serializes it with `json.dumps`, which
    produces `true` and `{"b": 1}`.
    """
    _sql, params = _compiled(value)
    bound = [v for v in params.values() if isinstance(v, dict) and "k" in v]
    assert bound, f"{label}: no JSONB parameter found in {params}"
    assert bound[0]["k"] == value
    assert type(bound[0]["k"]) is type(value), (
        f"{label}: value reached the driver as {type(bound[0]['k']).__name__}, "
        f"not {type(value).__name__}"
    )


def test_a_boolean_survives_as_a_boolean() -> None:
    """The specific reachable case: `{"verified": true}` from POST /search.

    The old predicate emitted `meta ->> 'k' = 'True'` against stored JSON text
    `'true'`, so this filter returned zero rows in production and the right ones
    in every in-memory dev run.
    """
    _sql, params = _compiled(True)
    bound = next(v for v in params.values() if isinstance(v, dict))
    assert bound["k"] is True
    assert bound["k"] != "True"


def test_a_none_filter_is_passed_as_json_null() -> None:
    _sql, params = _compiled(None)
    bound = next(v for v in params.values() if isinstance(v, dict) and "k" in v)
    assert bound["k"] is None


def test_a_nested_object_is_not_reduced_to_a_dict_repr() -> None:
    """Python's dict repr uses single quotes and is not JSON at all."""
    sql, params = _compiled({"region": "us-central1"})
    assert "'region':" not in sql, f"Python repr leaked into SQL:\n{sql}"
    bound = next(v for v in params.values() if isinstance(v, dict) and "k" in v)
    assert bound["k"] == {"region": "us-central1"}


def test_the_status_predicate_is_unaffected() -> None:
    """A guard that the edit was scoped to the metadata loop."""
    sql, _params = _compiled("production")
    assert "status" in sql.lower()


def test_several_metadata_pairs_produce_several_containment_predicates() -> None:
    """Each pair is its own `@>`, so all of them must hold -- matching
    `MemoryFilter.matches`, which returns False on the first mismatch."""
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql

    from mapi.store.postgres.models import MemoryRow
    from mapi.store.postgres.store import PostgresStore

    stmt = PostgresStore._apply_filters(
        None,  # type: ignore[arg-type]
        select(MemoryRow.id),
        MemoryFilter(metadata=(("env", "prod"), ("verified", True))),
    )
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert sql.count("@>") == 2
