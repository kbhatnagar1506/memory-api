"""`mapi admin` end to end against a real Postgres, as the application role.

tests/unit/test_admin_cli.py covers every command on the in-memory store.
This file repeats the two that make promises about the database itself, on
the database: a minted key is a working credential that only ever reached a
0600 file, and a purge leaves ZERO rows for the org in every tenant table --
counted with SQL under the org's own row-level scope, `memory_versions`
included, because that table has no foreign key and is the one a naive purge
leaves full of content.

The CLI opens its own store per command, exactly as it does from a shell;
only the connection (URL, schema, width) comes from the lane's settings.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from tests.support import postgres as pg_support

from mapi import admin, cli
from mapi.config import reset_settings_cache
from mapi.core.security import hash_key
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Chunk, Memory, RelationEdge, RelationType, Space

pytestmark = pytest.mark.postgres

PEPPER = "an-operator-pepper-for-the-postgres-lane"
TENANT_TABLES = ("memory_versions", "relation_edges", "chunks", "memories", "spaces")


@pytest.fixture
async def store(monkeypatch: pytest.MonkeyPatch):
    """The lane's store for arranging and checking; the CLI opens its own."""

    def _open(_settings: object) -> Any:
        return pg_support.make_store(require_migrated=True)

    monkeypatch.setattr(admin, "open_admin_store", _open)
    monkeypatch.setenv("MAPI_API_KEY_PEPPER", PEPPER)
    monkeypatch.setenv("MAPI_ENVIRONMENT", "production")
    reset_settings_cache()
    s = pg_support.make_store()
    await s.initialize()
    yield s
    await s.aclose()
    reset_settings_cache()


async def run(capsys, *argv: str) -> tuple[int, dict[str, Any], str]:
    """`mapi admin ...` in-process, on this test's event loop."""
    parser_args = _parse(["admin", *argv])
    code = await cli._admin_async(parser_args)
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip() else {}), out + err


def _parse(argv: list[str]) -> Any:
    """The real parser, without `main`'s asyncio.run (the lane owns the loop)."""
    import argparse

    parser = argparse.ArgumentParser(prog="mapi")
    sub = parser.add_subparsers(dest="command", required=True)
    cli._add_admin(sub)
    return parser.parse_args(argv)


async def _populate(store, org_id: str, slug: str) -> Space:
    """Two memories, a supersedes edge, and a deleted memory's kept history."""
    space = await store.create_space(Space(org_id=org_id, slug=slug, name=slug))
    embedder = DeterministicEmbedder(dimensions=pg_support.dimensions())
    ids = []
    for body in (f"{slug} old", f"{slug} new", f"{slug} deleted"):
        memory = Memory(org_id=org_id, space_id=space.id, content=body)
        vector = (await embedder.embed([body])).vectors[0]
        chunk = Chunk(memory_id=memory.id, ordinal=0, text=body, embedding=vector)
        await store.upsert_memory(memory.model_copy(update={"chunks": [chunk]}))
        ids.append(memory.id)
    await store.create_relation(
        RelationEdge(
            org_id=org_id,
            space_id=space.id,
            source_id=ids[1],
            target_id=ids[0],
            type=RelationType.SUPERSEDES,
        )
    )
    await store.delete_memory(org_id, space.id, ids[2])
    return space


async def _rows(store, org_id: str, space_ids: list[str]) -> dict[str, int]:
    """Rows per tenant table for these spaces, read under the org's scope."""
    counts: dict[str, int] = {}
    async with store._session() as session:
        await store._scope(session, org_id)
        for table in TENANT_TABLES:
            column = "id" if table == "spaces" else "space_id"
            counts[table] = (
                await session.execute(
                    text(f"SELECT count(*) FROM {table} WHERE {column} = ANY(:ids)"),
                    {"ids": space_ids},
                )
            ).scalar_one()
    return counts


async def test_mint_writes_a_working_key_to_a_0600_file_only(store, capsys, tmp_path) -> None:
    code, created, _ = await run(capsys, "org", "create", "--name", "lane:mint")
    assert code == 0
    org_id = created["organization"]["id"]

    out = tmp_path / "service.key"
    code, minted, printed = await run(
        capsys, "key", "mint", "--org", org_id, "--name", "facemash-server", "--out", str(out)
    )
    assert code == 0, printed
    plaintext = out.read_text()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert plaintext not in printed and "sm_" not in printed

    # The file's key is the database's key: it resolves by hash, to this org,
    # without admin scope.
    record = await store.get_api_key_by_hash(hash_key(plaintext, PEPPER))
    assert record is not None
    assert record.id == minted["key"]["id"]
    assert record.org_id == org_id
    assert "admin" not in {s.value for s in record.scopes}

    code, _, _ = await run(capsys, "key", "revoke", "--org", org_id, record.id)
    assert code == 0
    revoked = await store.get_api_key_by_hash(hash_key(plaintext, PEPPER))
    assert revoked is None or revoked.revoked_at is not None


async def test_rename_by_name(store, capsys) -> None:
    code, created, _ = await run(capsys, "org", "create", "--name", "lane:before-rename")
    org_id = created["organization"]["id"]
    code, renamed, _ = await run(capsys, "org", "rename", org_id, "--name", "lane:renamed")
    assert code == 0
    fetched = await store.get_organization(org_id)
    assert fetched is not None and fetched.name == "lane:renamed"
    assert renamed["previous_name"] == "lane:before-rename"


async def test_purge_leaves_zero_rows_in_every_table(store, capsys, tmp_path: Path) -> None:
    code, created, _ = await run(capsys, "org", "create", "--name", "lane:purge")
    org_id = created["organization"]["id"]
    alice = await _populate(store, org_id, "alice")
    bob = await _populate(store, org_id, "bob")
    keeper = await _populate(store, org_id, "keeper")
    doomed = [alice.id, bob.id]

    before = await _rows(store, org_id, doomed)
    assert all(before[t] > 0 for t in TENANT_TABLES), before

    code, plan, _ = await run(capsys, "org", "purge", org_id, "--keep-space", keeper.id)
    assert code == 0 and plan["dry_run"] is True
    assert await _rows(store, org_id, doomed) == before, "a dry run deleted rows"

    keep_file = tmp_path / "keep"
    keep_file.write_text(f"{keeper.id}\n")
    code, done, printed = await run(
        capsys, "org", "purge", org_id, "--keep-spaces-file", str(keep_file), "--yes"
    )
    assert code == 0, printed
    assert done["purged"]["spaces"] == 2
    assert done["purged"]["versions"] == before["memory_versions"]

    assert set((await _rows(store, org_id, doomed)).values()) == {0}
    kept = await _rows(store, org_id, [keeper.id])
    assert all(kept[t] > 0 for t in TENANT_TABLES), "the kept space was touched"
    assert await store.get_organization(org_id) is not None

    code, again, _ = await run(
        capsys, "org", "purge", org_id, "--keep-space", keeper.id, "--yes"
    )
    assert code == 0 and set(again["purged"].values()) == {0}
