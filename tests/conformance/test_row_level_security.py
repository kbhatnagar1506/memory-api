"""Row-level security, tested against the database rather than the code.

The store already filters by `org_id` in every query, and the conformance
suite already proves that. These tests prove something different and harder:
that the isolation survives a query which FORGETS to filter. They go around
the store entirely and issue raw SQL, because a policy that only holds when
the application is well-behaved is not a policy — it is the same application
guarantee written twice.

Postgres only. There is nothing to test on the in-memory backend: it has no
policies, and its isolation is the Python code the rest of the suite covers.

TWO PRECONDITIONS, both checked rather than assumed, because either one makes
every test here pass or fail for reasons unrelated to the policies:

  * The database was built by ALEMBIC. The policies live in migrations 0005
    and 0007; `create_all` builds the same tables with none, and against such a
    schema this file used to fail with "memory_versions has RLS disabled" --
    true, and useless as a diagnosis.
  * The connecting role is subject to RLS: not a superuser, not BYPASSRLS.
    Either attribute skips every policy, and CI connected as the image's
    superuser. Production's role owns the tables (FORCE binds owners) and has
    neither; the lane has to run as that shape of role to test anything.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from tests.support import postgres as pg_support

from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Chunk, Memory, Organization, Space
from mapi.store.base import MemoryFilter

#: Marked so `-m postgres` selects this file with the rest of the lane. It had
#: no marker, so the documented lane command deselected every test in it.
pytestmark = pytest.mark.postgres

DIMENSIONS = pg_support.dimensions()

TENANT_TABLES = ["memories", "chunks", "spaces", "relation_edges", "memory_versions"]


@pytest.fixture
async def pg():
    from mapi.store.postgres.store import SCHEMA_REVISION

    store = pg_support.make_store(dimensions=DIMENSIONS)
    revision = await store.schema_revision()
    if revision != SCHEMA_REVISION:
        await store.aclose()
        pytest.fail(
            f"the RLS lane needs a database migrated by alembic to {SCHEMA_REVISION}, "
            f"found {revision!r}. Policies come from migrations 0005 and 0007 and a "
            "schema built by create_all has none: run `alembic upgrade head` as the "
            "lane's role first (scripts/pg_lanes.py does)."
        )
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


async def test_the_lane_runs_as_a_role_the_policies_bind(pg) -> None:
    """Superusers and BYPASSRLS roles skip every policy, FORCE or not.

    Without this check a lane connected as `postgres` reports on nothing: the
    isolation tests below fail for a reason that has nothing to do with the
    policies, and the bypass tests could pass while proving nothing.
    """
    async with pg._session() as session:
        superuser, bypass = (
            await session.execute(
                text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
            )
        ).one()
    assert not superuser, "the lane connects as a superuser, which ignores every policy"
    assert not bypass, "the lane's role has BYPASSRLS, which ignores every policy"


async def test_policies_are_enabled_and_forced(pg) -> None:
    """FORCE matters: Postgres exempts a table's owner, and we are the owner.

    Without it the migration applies to nobody and reads as protection.
    """
    async with pg._session() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity "
                    "FROM pg_class WHERE relname = ANY(:names) "
                    "AND relnamespace = to_regnamespace(current_schema())"
                ),
                {"names": TENANT_TABLES},
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
                    "INSERT INTO memories "
                    "(id, org_id, space_id, content, content_sha256, occurred_at) "
                    "VALUES ('mem_rlsprobe0000000000000000', :org, :space, 'x', 'h', now())"
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


# -- migration 0007: the bypass is gone ------------------------------------------


async def test_only_the_tenant_policy_remains_on_each_table(pg) -> None:
    """0005 added `*_admin_bypass` beside each isolation policy; 0007 drops it.

    Asserted on the catalogue, not only by behaviour, so a policy re-added
    under another name that still reads `app.bypass_rls` is caught too.
    """
    async with pg._session() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT tablename, policyname, "
                    "coalesce(qual, '') || coalesce(with_check, '') "
                    "FROM pg_policies WHERE schemaname = current_schema() "
                    "AND tablename = ANY(:names)"
                ),
                {"names": TENANT_TABLES},
            )
        ).all()

    by_table: dict[str, set[str]] = {}
    for table, policy, expression in rows:
        by_table.setdefault(table, set()).add(policy)
        assert "bypass" not in expression, f"{table}.{policy} still honours a bypass flag"
    assert by_table == {t: {f"{t}_tenant_isolation"} for t in TENANT_TABLES}


async def test_the_bypass_flag_no_longer_reads_other_tenants(pg) -> None:
    """The hole 0007 closes: any session could `SET app.bypass_rls = 'on'`.

    A custom setting needs no privilege, and the policy was permissive, so it
    OR-ed with the tenant policy into "every row". Now the flag is inert: the
    scoped session still sees only its tenant, and an unscoped one nothing.
    """
    org_a, _, text_a = await _org_with_memory(pg, "bypass A")
    _, _, text_b = await _org_with_memory(pg, "bypass B")

    async with pg._session() as session:
        await pg._scope(session, org_a)
        await session.execute(text("SELECT set_config('app.bypass_rls', 'on', true)"))
        visible = {r[0] for r in await session.execute(text("SELECT content FROM memories"))}
    assert text_a in visible
    assert text_b not in visible, "app.bypass_rls still opens other tenants' memories"

    async with pg._session() as session:
        await session.execute(text("SELECT set_config('app.bypass_rls', 'on', true)"))
        unscoped = (await session.execute(text("SELECT count(*) FROM memories"))).scalar_one()
    assert unscoped == 0, "app.bypass_rls lets an unscoped session read tenant data"


async def test_the_bypass_flag_no_longer_writes_into_other_tenants(pg) -> None:
    """WITH CHECK had the same bypass. A write aimed at another org still fails."""
    org_a, _, _ = await _org_with_memory(pg, "bypass C")
    org_b, space_b, _ = await _org_with_memory(pg, "bypass D")

    with pytest.raises(Exception) as caught:
        async with pg._session() as session, session.begin():
            await pg._scope(session, org_a)
            await session.execute(text("SELECT set_config('app.bypass_rls', 'on', true)"))
            await session.execute(
                text(
                    "INSERT INTO memories "
                    "(id, org_id, space_id, content, content_sha256, occurred_at) "
                    "VALUES ('mem_rlsbypass000000000000000', :org, :space, 'x', 'h', now())"
                ),
                {"org": org_b, "space": space_b},
            )
    assert "policy" in str(caught.value).lower()
