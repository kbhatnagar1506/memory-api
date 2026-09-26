"""`mapi admin`: organizations and keys from an operator's shell.

The two promises tested hardest: a minted key reaches a 0600 file and never
stdout, and a purge takes every table -- `memory_versions` included -- while
sparing exactly the spaces it was told to spare.

Runs the real argument parser and command functions against an in-memory
store injected where the Postgres one would be opened.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mapi import admin, cli
from mapi.config import Environment, Settings, StoreBackend, reset_settings_cache
from mapi.core.errors import ConfigurationError
from mapi.core.security import hash_key
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Chunk, Memory, Organization, RelationEdge, RelationType, Space
from mapi.store.memory import InMemoryStore

PEPPER = "an-operator-pepper-that-is-not-the-default"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> InMemoryStore:
    shared = InMemoryStore()
    monkeypatch.setattr(admin, "open_admin_store", lambda _settings: shared)
    monkeypatch.setenv("MAPI_API_KEY_PEPPER", PEPPER)
    monkeypatch.setenv("MAPI_ENVIRONMENT", "production")
    reset_settings_cache()
    yield shared
    reset_settings_cache()


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict, str]:
    code = cli.main(["admin", *argv])
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip() else {}), out + err


def _org(capsys, name: str) -> str:
    code, doc, _ = run(capsys, "org", "create", "--name", name)
    assert code == 0
    return str(doc["organization"]["id"])


# -- organizations -------------------------------------------------------------------


def test_org_create_list_and_rename_by_name(store, capsys) -> None:
    org_id = _org(capsys, "facemash:staging")
    code, listed, _ = run(capsys, "org", "list")
    assert code == 0
    assert [o["name"] for o in listed["organizations"]] == ["facemash:staging"]

    code, renamed, _ = run(capsys, "org", "rename", "facemash:staging", "--name", "fm:stage")
    assert code == 0
    assert renamed["organization"] == {**renamed["organization"], "id": org_id}
    assert renamed["organization"]["name"] == "fm:stage"
    assert renamed["previous_name"] == "facemash:staging"


def test_an_ambiguous_name_is_refused_with_both_ids(store, capsys) -> None:
    first = _org(capsys, "twin")
    second = _org(capsys, "twin")
    code, _, text = run(capsys, "org", "rename", "twin", "--name", "x")
    assert code == 1
    assert first in text and second in text
    # By id it works, and touches only that one.
    assert run(capsys, "org", "rename", first, "--name", "x")[0] == 0


def test_unknown_orgs_fail_cleanly(store, capsys) -> None:
    code, _, text = run(capsys, "org", "rename", "nobody", "--name", "x")
    assert code == 1
    assert json.loads(text)["error"] == "not_found"


def test_names_are_bounded(store, capsys) -> None:
    code, _, text = run(capsys, "org", "create", "--name", "x" * 201)
    assert code == 1
    assert "1-200" in text


# -- keys ---------------------------------------------------------------------------


def test_mint_writes_the_key_to_a_0600_file_and_nowhere_else(store, capsys, tmp_path) -> None:
    org_id = _org(capsys, "facemash:hackgt13")
    out = tmp_path / "service.key"
    code, doc, printed = run(
        capsys,
        "key", "mint", "--org", "facemash:hackgt13", "--name", "facemash-server",
        "--out", str(out),
    )  # fmt: skip
    assert code == 0

    plaintext = out.read_text()
    assert plaintext.startswith("sm_")
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    # Nothing secret on stdout or stderr: not the key, not even its prefix.
    assert plaintext not in printed
    assert plaintext[:11] not in printed
    assert "sm_" not in printed

    assert doc["written_to"] == str(out)
    assert doc["key"]["org_id"] == org_id
    assert doc["key"]["scopes"] == [
        "memories:read", "memories:write", "search", "spaces:read", "spaces:write",
    ]  # fmt: skip
    # The file holds a working credential: it hashes, under the configured
    # pepper, to the stored key.
    stored = [k for k in store._keys.values() if k.org_id == org_id]
    assert len(stored) == 1
    assert stored[0].key_hash == hash_key(plaintext, PEPPER)


def test_mint_refuses_an_existing_file_unless_forced(store, capsys, tmp_path) -> None:
    _org(capsys, "o")
    out = tmp_path / "k"
    out.write_text("previous")
    os.chmod(out, 0o644)
    code, _, _ = run(capsys, "key", "mint", "--org", "o", "--name", "n", "--out", str(out))
    assert code == 1
    assert out.read_text() == "previous"
    assert len(store._keys) == 0, "a refused mint still created a key"

    code, _, _ = run(
        capsys, "key", "mint", "--org", "o", "--name", "n", "--out", str(out), "--force"
    )
    assert code == 0
    assert out.read_text().startswith("sm_")
    # Replaced, not written through: the old 0644 mode did not survive.
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_mint_never_follows_a_symlink(store, capsys, tmp_path) -> None:
    _org(capsys, "o")
    target = tmp_path / "elsewhere"
    link = tmp_path / "k"
    link.symlink_to(target)
    code, _, _ = run(capsys, "key", "mint", "--org", "o", "--name", "n", "--out", str(link))
    assert code == 1
    assert not target.exists()


def test_a_failed_insert_leaves_no_file(store, capsys, tmp_path, monkeypatch) -> None:
    _org(capsys, "o")

    async def _boom(_key):
        raise ConfigurationError("database went away")

    monkeypatch.setattr(store, "create_api_key", _boom)
    out = tmp_path / "k"
    code, _, _ = run(capsys, "key", "mint", "--org", "o", "--name", "n", "--out", str(out))
    assert code == 1
    assert not out.exists()


def test_mint_refuses_the_development_pepper_outside_development(
    store, capsys, tmp_path, monkeypatch
) -> None:
    _org(capsys, "o")
    monkeypatch.delenv("MAPI_API_KEY_PEPPER")
    reset_settings_cache()
    out = tmp_path / "k"
    code, _, text = run(capsys, "key", "mint", "--org", "o", "--name", "n", "--out", str(out))
    assert code == 1
    assert "development default" in text
    assert not out.exists()


def test_scopes_expiry_list_and_revoke(store, capsys, tmp_path) -> None:
    _org(capsys, "o")
    out = tmp_path / "k"
    code, doc, _ = run(
        capsys,
        "key", "mint", "--org", "o", "--name", "reader", "--out", str(out),
        "--scopes", "search,memories:read", "--expires-in-days", "30",
    )  # fmt: skip
    assert code == 0
    assert doc["key"]["scopes"] == ["memories:read", "search"]
    expires = datetime.fromisoformat(doc["key"]["expires_at"])
    assert 29 <= (expires - datetime.now(UTC)).days <= 30

    key_id = doc["key"]["id"]
    code, listed, printed = run(capsys, "key", "list", "--org", "o")
    assert [k["id"] for k in listed["keys"]] == [key_id]
    assert "sm_" not in printed

    assert run(capsys, "key", "revoke", "--org", "o", key_id)[0] == 0
    code, _, _ = run(capsys, "key", "revoke", "--org", "o", key_id)
    assert code == 1, "revoking twice should say so"
    assert run(capsys, "key", "list", "--org", "o")[1]["keys"][0]["revoked"] is True


def test_unknown_scopes_are_refused(store, capsys, tmp_path) -> None:
    _org(capsys, "o")
    code, _, text = run(
        capsys, "key", "mint", "--org", "o", "--name", "n", "--out", str(tmp_path / "k"),
        "--scopes", "search,root",
    )  # fmt: skip
    assert code == 1
    assert "root" in text


# -- purge --------------------------------------------------------------------------


async def _populate(store: InMemoryStore, org_id: str, slug: str) -> tuple[str, str]:
    """A space with two memories, an edge between them, and deleted history."""
    space = await store.create_space(Space(org_id=org_id, slug=slug, name=slug))
    embedder = DeterministicEmbedder(dimensions=16)
    ids = []
    for text in (f"{slug} one", f"{slug} two", f"{slug} gone"):
        memory = Memory(org_id=org_id, space_id=space.id, content=text)
        vector = (await embedder.embed([text])).vectors[0]
        chunk = Chunk(memory_id=memory.id, ordinal=0, text=text, embedding=vector)
        memory = memory.model_copy(update={"chunks": [chunk]})
        await store.upsert_memory(memory)
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
    # delete_memory keeps history on purpose; purge must not.
    await store.delete_memory(org_id, space.id, ids[2])
    return space.id, ids[0]


def test_purge_is_a_dry_run_without_yes(store, capsys) -> None:
    import asyncio

    org_id = _org(capsys, "event")
    asyncio.run(_populate(store, org_id, "alice"))
    code, plan, _ = run(capsys, "org", "purge", "event")
    assert code == 0
    assert plan["dry_run"] is True
    assert plan["spaces_to_purge"] == 1
    assert plan["active_memories_to_purge"] == 2
    assert len(store._spaces) == 1, "a dry run purged"


def test_purge_takes_everything_but_the_kept_spaces(store, capsys, tmp_path) -> None:
    import asyncio

    org_id = _org(capsys, "event")
    other_org = asyncio.run(store.create_organization(Organization(name="bystander")))
    alice, _ = asyncio.run(_populate(store, org_id, "alice"))
    bob, _ = asyncio.run(_populate(store, org_id, "bob"))
    keeper, kept_memory = asyncio.run(_populate(store, org_id, "keeper"))
    bystander, bystander_memory = asyncio.run(_populate(store, other_org.id, "alice"))
    keep_file = tmp_path / "keep.txt"
    keep_file.write_text(f"# opted to keep their memory\n{keeper}\n\n")

    code, doc, _ = run(
        capsys, "org", "purge", org_id, "--keep-spaces-file", str(keep_file), "--yes"
    )
    assert code == 0
    assert doc["purged"] == {
        "spaces": 2, "memories": 4, "chunks": 4, "edges": 2,
        # 3 versions per space: two live memories and the deleted one's history.
        "versions": 6,
    }  # fmt: skip

    remaining = {s.id for s in store._spaces.values()}
    assert alice not in remaining and bob not in remaining
    assert {keeper, bystander} <= remaining
    assert all(v.space_id in {keeper, bystander} for vs in store._versions.values() for v in vs)
    assert asyncio.run(store.get_memory(org_id, keeper, kept_memory)) is not None
    assert asyncio.run(store.get_memory(other_org.id, bystander, bystander_memory)) is not None

    # Idempotent: a second run finds nothing left to take.
    code, again, _ = run(capsys, "org", "purge", org_id, "--keep-space", keeper, "--yes")
    assert set(again["purged"].values()) == {0}


def test_a_keep_list_from_the_wrong_event_purges_nothing(store, capsys) -> None:
    import asyncio

    org_id = _org(capsys, "event")
    other = asyncio.run(store.create_organization(Organization(name="other")))
    alice, _ = asyncio.run(_populate(store, org_id, "alice"))
    foreign, _ = asyncio.run(_populate(store, other.id, "foreign"))

    code, _, text = run(capsys, "org", "purge", org_id, "--keep-space", foreign, "--yes")
    assert code == 1
    assert "not spaces of" in text
    assert alice in store._spaces


def test_a_malformed_keep_entry_is_refused(store, capsys, tmp_path) -> None:
    _org(capsys, "event")
    bad = tmp_path / "keep.txt"
    bad.write_text("space_but_not_really\n")
    code, _, text = run(capsys, "org", "purge", "event", "--keep-spaces-file", str(bad))
    assert code == 1
    assert "not space ids" in text


# -- opening the store ----------------------------------------------------------------


def test_the_admin_store_must_be_the_database() -> None:
    """An in-memory org would vanish when the command exits, reporting success."""
    with pytest.raises(ConfigurationError, match="MAPI_STORE_BACKEND=postgres"):
        admin.open_admin_store(Settings(store_backend=StoreBackend.MEMORY))


def test_the_admin_store_never_runs_boot_ddl() -> None:
    store = admin.open_admin_store(
        Settings(
            environment=Environment.LOCAL,
            store_backend=StoreBackend.POSTGRES,
            database_url="postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none",
        )
    )
    assert store._require_migrated is True


def test_the_admin_help_lists_every_command(capsys) -> None:
    with pytest.raises(SystemExit):
        cli.main(["admin", "--help"])
    for group, commands in (("org", "create list rename purge"), ("key", "mint list revoke")):
        with pytest.raises(SystemExit):
            cli.main(["admin", group, "--help"])
        out = capsys.readouterr().out
        for command in commands.split():
            assert command in out


def test_read_keep_list_merges_flags_and_files(tmp_path: Path) -> None:
    ids = [Space(org_id="org_" + "a" * 26, slug=f"s{i}", name="s").id for i in range(3)]
    file = tmp_path / "keep"
    file.write_text(f"{ids[1]}  # a comment\n{ids[2]}\n")
    assert admin.read_keep_list([ids[0]], [file]) == frozenset(ids)
