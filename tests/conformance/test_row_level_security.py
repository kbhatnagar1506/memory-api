"""Row-level security, tested against the database rather than the code.

The store already filters by `org_id` in every query, and the conformance
suite already proves that. These tests prove something different and harder:
that the isolation survives a query which FORGETS to filter. They go around
the store entirely and issue raw SQL, because a policy that only holds when
the application is well-behaved is not a policy — it is the same application
guarantee written twice.

Postgres only. There is nothing to test on the in-memory backend: it has no
policies, and its isolation is the Python code the rest of the suite covers.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Chunk, Memory, Organization, Space
from mapi.store.base import MemoryFilter

DIMENSIONS = int(os.getenv("MAPI_TEST_DIMENSIONS", "128"))


@pytest.fixture
async def pg():
    url = os.getenv("MAPI_TEST_DATABASE_URL")
    if not url:
        pytest.skip("MAPI_TEST_DATABASE_URL is not set")
    from mapi.store.postgres.store import PostgresStore

    store = PostgresStore(url, dimensions=DIMENSIONS)
    await store.initialize()
    yield store
    await store.aclose()


async def _org_with_memory(store, label: str) -> tuple[str, str, str]:
    """An organization holding one memory. Returns (org_id, space_id, text)."""
    org = await store.create_organization(Organization(name=f"RLS {label}"))
    space = await store.create_space(
        Space(org_id=org.id, slug=f"rls{org.id[-10:]}", name="Space")
    )
    body = f"a secret belonging to {label}"
    vector = await DeterministicEmbedder(dimensions=DIMENSIONS).embed_one(body)
    memory = Memory(org_id=org.id, space_id=space.id, content=body)
    memory = memory.model_copy(
        update={"chunks": [Chunk(memory_id=memory.id, ordinal=0, text=body, embedding=vector)]}
    )
    await store.upsert_memory(memory)
    return org.id, space.id, body


async def test_policies_are_enabled_and_forced(pg) -> None:
    """FORCE matters: Postgres exempts a table's owner, and we are the owner.

    Without it the migration applies to nobody and reads as protection.
    """
    async with pg._session() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity "
                    "FROM pg_class WHERE relname = ANY(:names)"
                ),
                {
                    "names": [
                        "memories",
                        "chunks",
                        "spaces",
                        "relation_edges",
                        "memory_versions",
                    ]
                },
            )
        ).all()

    assert len(rows) == 5, "a tenant table is missing"
    for name, enabled, forced in rows:
        assert enabled, f"{name} has RLS disabled"
        assert forced, f"{name} does not FORCE RLS, so the owner bypasses it"


async def test_a_query_without_a_where_clause_sees_only_its_own_tenant(pg) -> None:
    """The whole point. `SELECT * FROM memories` must not be a data breach."""
    org_a, _, text_a = await _org_with_memory(pg, "A")
    _, _, text_b = await _org_with_memory(pg, "B")

    async with pg._session() as session:
        await pg._scope(session, org_a)
        # Deliberately unfiltered -- this is the bug the policy exists to
        # survive, written on purpose.
        visible = {r[0] for r in await session.execute(text("SELECT content FROM memories"))}

    assert text_a in visible
    assert text_b not in visible, "another tenant's memory was readable"


async def test_an_unscoped_session_sees_nothing_at_all(pg) -> None:
    """Fail-closed. Forgetting to scope must be loud and local, not permissive."""
    await _org_with_memory(pg, "C")

    async with pg._session() as session:
        rows = (await session.execute(text("SELECT count(*) FROM memories"))).scalar_one()

    assert rows == 0, "an unscoped session could read tenant data"


async def test_no_session_setting_unlocks_other_tenants(pg) -> None:
    """Any session can set a custom setting, so none may widen what it sees: the old
    `app.bypass_rls` escape hatch is gone (migration 0009)."""
    org_a, _, text_a = await _org_with_memory(pg, "J")
    _, _, text_b = await _org_with_memory(pg, "K")

    async with pg._session() as session, session.begin():
        await pg._scope(session, org_a)
        await session.execute(text("SELECT set_config('app.bypass_rls', 'on', true)"))
        visible = {r[0] for r in await session.execute(text("SELECT content FROM memories"))}
        policies = {
            r[0] for r in await session.execute(text("SELECT policyname FROM pg_policies"))
        }

    assert text_a in visible
    assert text_b not in visible
    assert not any(p.endswith("_admin_bypass") for p in policies)


async def test_a_write_cannot_be_aimed_at_another_tenant(pg) -> None:
    """WITH CHECK, not just USING.

    Without it a compromised or buggy path could INSERT into another org even
    while being unable to read one back -- invisible poisoning.
    """
    org_a, _, _ = await _org_with_memory(pg, "D")
    org_b, space_b, _ = await _org_with_memory(pg, "E")

    with pytest.raises(Exception) as caught:
        async with pg._session() as session, session.begin():
            await pg._scope(session, org_a)
            await session.execute(
                text(
                    "INSERT INTO memories (id, org_id, space_id, content, summary, kind, "
                    "meta, tags, source, status, content_sha256, occurred_at, version) "
                    "VALUES ('mem_rlsprobe0000000000000000', :org, :space, 'x', '', "
                    "'episodic', '{}', '{}', '', 'active', 'h', now(), 1)"
                ),
                {"org": org_b, "space": space_b},
            )
    assert "policy" in str(caught.value).lower()


async def test_chunks_are_isolated_too(pg) -> None:
    """A leak through the chunk table is a leak through the search index."""
    org_a, _, text_a = await _org_with_memory(pg, "F")
    _, _, text_b = await _org_with_memory(pg, "G")

    async with pg._session() as session:
        await pg._scope(session, org_a)
        visible = {r[0] for r in await session.execute(text("SELECT text FROM chunks"))}

    assert text_a in visible
    assert text_b not in visible


async def test_the_store_still_works_normally_with_policies_on(pg) -> None:
    """RLS is a backstop, not a behaviour change."""
    org_a, space_a, text_a = await _org_with_memory(pg, "H")

    page = await pg.list_memories(org_a, space_a, filters=MemoryFilter(), limit=10)
    assert [m.content for m in page.items] == [text_a]
