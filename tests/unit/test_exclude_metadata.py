"""Per-request search shaping: exclude_metadata, max_per_source, snippets (B9).

"Who should I meet about X" searches an event directory in which the asker has
an entry of their own -- usually the best match there is. `MemoryFilter` had no
negation, so the caller over-fetched and dropped itself afterwards, shrinking
the window it asked for and leaking its own card through any caller that forgot
the step. `exclude_metadata` is that step, done by the store.

The other three fields move server-wide settings (or nothing at all) to the
request, because one server answers a person's own memory and a directory, and
those want opposite answers to all of them.
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.support.factories import memory as build_memory

from mapi.domain.models import MemoryStatus
from mapi.store.base import MemoryFilter

# -- the reference semantics ------------------------------------------------------


def _stored(**metadata: Any) -> Any:
    return build_memory(org_id="org_x", space_id="spc_x", content="a card", metadata=metadata)


def test_a_matching_pair_excludes() -> None:
    assert not MemoryFilter(exclude_metadata=(("user_id", "u1"),)).matches(
        _stored(user_id="u1")
    )


def test_a_different_value_does_not_exclude() -> None:
    assert MemoryFilter(exclude_metadata=(("user_id", "u1"),)).matches(_stored(user_id="u2"))


def test_every_pair_must_match_to_exclude() -> None:
    """`NOT (meta @> map)`: the whole map is contained, or nothing is excluded."""
    rule = MemoryFilter(exclude_metadata=(("user_id", "u1"), ("event", "hackgt")))
    assert not rule.matches(_stored(user_id="u1", event="hackgt"))
    assert rule.matches(_stored(user_id="u1", event="other"))
    assert rule.matches(_stored(user_id="u1"))


def test_a_missing_key_never_matches_an_exclusion() -> None:
    """Presence, like JSONB containment: `{"k": None}` does not exclude a memory without k.

    The POSITIVE filter's `.get` makes an absent key look like None; the
    negative one must not inherit that, or excluding `{"k": null}` would hide
    every memory that never had a `k`.
    """
    assert MemoryFilter(exclude_metadata=(("k", None),)).matches(_stored())
    assert not MemoryFilter(exclude_metadata=(("k", None),)).matches(_stored(k=None))


def test_exclusion_composes_with_inclusion() -> None:
    rule = MemoryFilter(metadata=(("event", "hackgt"),), exclude_metadata=(("user_id", "me"),))
    assert rule.matches(_stored(event="hackgt", user_id="you"))
    assert not rule.matches(_stored(event="hackgt", user_id="me"))
    assert not rule.matches(_stored(event="other", user_id="you"))


def test_an_empty_exclusion_excludes_nothing() -> None:
    assert MemoryFilter().matches(_stored(user_id="u1"))


# -- the SQL, compiled without a database --------------------------------------------


def test_the_sql_is_negated_containment() -> None:
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql

    from mapi.store.postgres.models import MemoryRow
    from mapi.store.postgres.store import PostgresStore

    stmt = PostgresStore._apply_filters(
        None,  # type: ignore[arg-type]
        select(MemoryRow.id),
        MemoryFilter(
            statuses=frozenset({MemoryStatus.ACTIVE}),
            exclude_metadata=(("user_id", "u1"), ("n", 3)),
        ),
    )
    compiled = stmt.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "NOT (memories.meta @>" in sql
    assert {"user_id": "u1", "n": 3} in compiled.params.values()


# -- over HTTP ---------------------------------------------------------------------------


async def _card(client: Any, space_id: str, content: str, **metadata: Any) -> str:
    response = await client.post(
        f"/v1/spaces/{space_id}/memories", json={"content": content, "metadata": metadata}
    )
    assert response.status_code in (200, 201), response.text
    return str(response.json()["memory"]["id"])


async def test_search_can_drop_the_caller(client, space_id) -> None:
    await _card(client, space_id, "rust compilers and wasm toolchains", user_id="me")
    theirs = await _card(client, space_id, "rust embedded firmware work", user_id="them")
    body = {"query": "rust", "exclude_metadata": {"user_id": "me"}}
    response = await client.post(f"/v1/spaces/{space_id}/search", json=body)
    assert response.status_code == 200
    ids = [hit["memory"]["id"] for hit in response.json()["results"]]
    assert ids == [theirs]


async def test_nested_exclusions_are_refused(client, space_id) -> None:
    body = {"query": "rust", "exclude_metadata": {"user": {"id": "me"}}}
    response = await client.post(f"/v1/spaces/{space_id}/search", json=body)
    assert response.status_code == 422


async def test_max_per_source_is_per_request(client, space_id) -> None:
    for i in range(3):
        await _card(client, space_id, f"kubernetes operator note {i}", doc_id="alice")
    await _card(client, space_id, "kubernetes helm chart note", doc_id="bob")

    capped = await client.post(
        f"/v1/spaces/{space_id}/search", json={"query": "kubernetes", "max_per_source": 1}
    )
    people = [hit["memory"]["metadata"]["doc_id"] for hit in capped.json()["results"]]
    assert sorted(people) == ["alice", "bob"]

    uncapped = await client.post(f"/v1/spaces/{space_id}/search", json={"query": "kubernetes"})
    assert len(uncapped.json()["results"]) == 4


async def test_max_per_source_unset_keeps_the_server_setting(app_context, client, space_id):
    service = app_context["app"].state.service
    app_context["app"].state.settings = service.settings.model_copy(
        update={"max_per_source": 1}
    )
    for i in range(3):
        await _card(client, space_id, f"postgres tuning note {i}", doc_id="carol")
    response = await client.post(f"/v1/spaces/{space_id}/search", json={"query": "postgres"})
    assert len(response.json()["results"]) == 1
    explicit = await client.post(
        f"/v1/spaces/{space_id}/search", json={"query": "postgres", "max_per_source": 0}
    )
    assert len(explicit.json()["results"]) == 3


async def test_content_can_be_left_out(client, space_id) -> None:
    await _card(client, space_id, "the deploy pipeline uses cloud build and artifact registry")
    response = await client.post(
        f"/v1/spaces/{space_id}/search",
        json={"query": "cloud build", "include_content": False},
    )
    hit = response.json()["results"][0]
    assert hit["memory"]["content"] == ""
    assert "cloud build" in hit["matched_text"]


async def test_snippets_are_capped(client, space_id) -> None:
    await _card(client, space_id, "grafana dashboards " + "and more words " * 40)
    response = await client.post(
        f"/v1/spaces/{space_id}/search", json={"query": "grafana", "snippet_chars": 25}
    )
    hit = response.json()["results"][0]
    assert len(hit["matched_text"]) == 25
    assert len(hit["memory"]["content"]) > 25  # content is untouched unless excluded


@pytest.mark.parametrize("bad", [0, 20_001])
async def test_snippet_bounds_are_enforced(client, space_id, bad: int) -> None:
    response = await client.post(
        f"/v1/spaces/{space_id}/search", json={"query": "x", "snippet_chars": bad}
    )
    assert response.status_code == 422


async def test_the_defaults_change_nothing(client, space_id) -> None:
    """No new field set: the response is what it always was."""
    content = "terraform state lives in a gcs bucket"
    await _card(client, space_id, content)
    hit = (
        await client.post(f"/v1/spaces/{space_id}/search", json={"query": "terraform"})
    ).json()["results"][0]
    assert hit["memory"]["content"] == content
    assert hit["matched_text"] == content
