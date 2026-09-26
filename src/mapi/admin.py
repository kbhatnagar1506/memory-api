"""Operator commands behind `mapi admin`: organizations and their keys.

Everything here used to need a running server in staging mode with a
bootstrap admin key: boot a throwaway instance, have it seed an org named
"Bootstrap", rename the row by hand in SQL, then mint service keys over HTTP
with the bootstrap string as the credential. That is three secrets in a shell
history to do what is, underneath, four inserts.

These talk to the store directly, as the operator's own database role, and
follow two rules:

  * NOTHING SECRET GOES TO STDOUT. A minted key is written to a file created
    0600 with O_EXCL, and the command prints its id and where it went. Stdout
    is where secrets leak from -- terminal scrollback, CI logs, `| tee` -- and
    a key that was never printed cannot be recovered from any of them.
  * NO SCHEMA CHANGES. The store is opened with `require_migrated`, so a
    database alembic has not brought to head is refused rather than patched
    up with boot-time DDL from an operator's laptop.

Output is one JSON object per command, so the commands compose with `jq` and
scripts without scraping prose.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import Environment, Settings, StoreBackend
from .core.errors import ConfigurationError, NotFoundError, ValidationError
from .core.ids import is_valid
from .core.security import build_api_key, default_scopes
from .domain.models import Organization, Scope, utcnow
from .store.base import MemoryStore

#: The pepper `Settings` ships with. A key minted under it only authenticates
#: against a server that also never set one, so outside local development it
#: is always a mistake -- usually a shell missing MAPI_API_KEY_PEPPER.
_DEV_PEPPER = "dev-insecure-pepper"

#: Organization names are a String(200) column.
_MAX_NAME = 200


def open_admin_store(settings: Settings) -> MemoryStore:
    """The store the admin commands run against.

    Postgres only: the in-memory backend lives and dies with this process, so
    an org created there would vanish the moment the command returned while
    reporting success.
    """
    if settings.store_backend is not StoreBackend.POSTGRES or not settings.database_url:
        raise ConfigurationError(
            "admin commands need the database: set MAPI_STORE_BACKEND=postgres and "
            "MAPI_DATABASE_URL"
        )
    from .store.postgres.store import PostgresStore

    return PostgresStore(
        settings.database_url,
        dimensions=settings.embedding_dimensions,
        pool_size=2,
        max_overflow=0,
        statement_timeout_ms=settings.db_statement_timeout_ms,
        require_migrated=True,
    )


def parse_scopes(raw: str) -> frozenset[Scope]:
    """`default`, `all`, or a comma-separated list of scope names."""
    value = raw.strip()
    if value == "default":
        return default_scopes()
    if value == "all":
        return Scope.all()
    names = [part.strip() for part in value.split(",") if part.strip()]
    known = {s.value: s for s in Scope}
    unknown = sorted(n for n in names if n not in known)
    if unknown or not names:
        raise ValidationError(
            f"unknown scopes {unknown or [raw]}; choose from default, all, or "
            f"{', '.join(sorted(known))}",
            field="scopes",
        )
    return frozenset(known[n] for n in names)


def _clean_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or len(cleaned) > _MAX_NAME:
        raise ValidationError(f"name must be 1-{_MAX_NAME} characters", field="name")
    return cleaned


async def resolve_org(store: MemoryStore, ref: str) -> Organization:
    """An organization by id, or by exact name when the name is unambiguous.

    Names are what an operator remembers (`facemash:hackgt13`); ids are what
    is unique. A name matching two orgs is refused with both ids rather than
    guessed, because the command it feeds may be a purge.
    """
    if is_valid(ref, "org"):
        org = await store.get_organization(ref)
        if org is None:
            raise NotFoundError(f"organization {ref} not found", field="org")
        return org
    matches = [o for o in await store.list_organizations() if o.name == ref]
    if not matches:
        raise NotFoundError(f"no organization is named {ref!r}", field="org")
    if len(matches) > 1:
        raise ValidationError(
            f"{len(matches)} organizations are named {ref!r}: "
            f"{', '.join(o.id for o in matches)}; pass the id",
            field="org",
        )
    return matches[0]


def _org_json(org: Organization) -> dict[str, Any]:
    return {"id": org.id, "name": org.name, "created_at": org.created_at.isoformat()}


async def create_org(store: MemoryStore, name: str) -> dict[str, Any]:
    org = await store.create_organization(Organization(name=_clean_name(name)))
    return {"organization": _org_json(org)}


async def list_orgs(store: MemoryStore) -> dict[str, Any]:
    return {"organizations": [_org_json(o) for o in await store.list_organizations()]}


async def rename_org(store: MemoryStore, ref: str, name: str) -> dict[str, Any]:
    org = await resolve_org(store, ref)
    renamed = await store.rename_organization(org.id, _clean_name(name))
    if renamed is None:  # deleted between the lookup and the update
        raise NotFoundError(f"organization {org.id} not found", field="org")
    return {"organization": _org_json(renamed), "previous_name": org.name}


def _create_secret_file(path: Path, *, force: bool) -> int:
    """Open `path` for writing as a new 0600 file, never following a symlink.

    O_EXCL is what makes the mode trustworthy: an existing file keeps
    whatever mode it already had, so "create 0600" on a pre-existing
    world-readable file would silently produce a world-readable secret.
    `force` removes the old file first rather than writing through it.
    """
    if force:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValidationError(
            f"{path} already exists; pass --force to replace it", field="out"
        ) from exc
    # The umask can only narrow 0600, but be explicit about the result.
    os.fchmod(fd, 0o600)
    return fd


def _finish_secret_file(fd: int, secret: str) -> None:
    try:
        os.write(fd, secret.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _discard_secret_file(fd: int, path: Path) -> None:
    os.close(fd)
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


async def mint_key(
    store: MemoryStore,
    settings: Settings,
    *,
    org_ref: str,
    name: str,
    scopes: frozenset[Scope],
    out: Path,
    expires_in_days: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create an API key and write its plaintext to `out`, and only there."""
    if settings.api_key_pepper == _DEV_PEPPER and settings.environment not in (
        Environment.LOCAL,
        Environment.TEST,
    ):
        raise ConfigurationError(
            "MAPI_API_KEY_PEPPER is the development default; a key minted with it "
            "will not authenticate against a real deployment"
        )
    org = await resolve_org(store, org_ref)
    expires_at = (
        utcnow() + timedelta(days=expires_in_days) if expires_in_days is not None else None
    )
    # The file first, so a path that cannot be written fails before a key
    # exists that nobody holds; removed again if the insert fails.
    fd = _create_secret_file(out, force=force)
    try:
        record, plaintext = build_api_key(
            org_id=org.id,
            name=_clean_name(name),
            pepper=settings.api_key_pepper,
            scopes=scopes,
            expires_at=expires_at,
        )
        stored = await store.create_api_key(record)
    except BaseException:
        _discard_secret_file(fd, out)
        raise
    _finish_secret_file(fd, plaintext)
    return {
        "key": {
            "id": stored.id,
            "org_id": stored.org_id,
            "name": stored.name,
            "scopes": sorted(s.value for s in stored.scopes),
            "expires_at": stored.expires_at.isoformat() if stored.expires_at else None,
        },
        "written_to": str(out),
        "mode": "0600",
    }


async def list_keys(store: MemoryStore, org_ref: str) -> dict[str, Any]:
    """Keys by id and name. No prefix: even a display prefix is key material."""
    org = await resolve_org(store, org_ref)
    return {
        "org_id": org.id,
        "keys": [
            {
                "id": k.id,
                "name": k.name,
                "scopes": sorted(s.value for s in k.scopes),
                "created_at": k.created_at.isoformat(),
                "revoked": k.revoked_at is not None,
            }
            for k in await store.list_api_keys(org.id)
        ],
    }


async def revoke_key(store: MemoryStore, org_ref: str, key_id: str) -> dict[str, Any]:
    org = await resolve_org(store, org_ref)
    if not await store.revoke_api_key(org.id, key_id):
        raise NotFoundError(
            f"key {key_id} not found in {org.id}, or already revoked", field="key_id"
        )
    return {"revoked": key_id, "org_id": org.id}


def read_keep_list(values: Iterable[str], files: Iterable[Path]) -> frozenset[str]:
    """Space ids to spare, from flags and from files of one id per line.

    Files because the list is "attendees who opted to keep their memory",
    which is data an operator exports, not something to type. `#` comments
    and blank lines are ignored; anything that is not a space id is refused
    rather than skipped, since a typo here would purge someone who asked not
    to be purged.
    """
    ids = [v.strip() for v in values if v.strip()]
    for path in files:
        for line in path.read_text().splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                ids.append(entry)
    bad = sorted({i for i in ids if not is_valid(i, "space")})
    if bad:
        raise ValidationError(f"not space ids: {bad[:5]}", field="keep_space")
    return frozenset(ids)


async def _purge_target(
    store: MemoryStore, org_ref: str, keep_space_ids: frozenset[str]
) -> tuple[Organization, set[str]]:
    org = await resolve_org(store, org_ref)
    present = {s.id for s in await store.list_spaces(org.id)}
    # A kept id that is not in this org is almost certainly a list exported
    # from the wrong event. Refuse: the safe failure is purging nothing.
    stray = sorted(keep_space_ids - present)
    if stray:
        raise ValidationError(
            f"{len(stray)} kept space ids are not spaces of {org.id}: {stray[:5]}",
            field="keep_space",
        )
    return org, present


async def plan_purge(
    store: MemoryStore, org_ref: str, *, keep_space_ids: frozenset[str]
) -> dict[str, Any]:
    """What `purge_org` would remove, without removing it. The default."""
    org, present = await _purge_target(store, org_ref, keep_space_ids)
    counts = await store.count_memories_by_space(org.id)
    doomed = present - keep_space_ids
    return {
        "dry_run": True,
        "org_id": org.id,
        "org_name": org.name,
        "spaces_to_purge": len(doomed),
        "active_memories_to_purge": sum(counts.get(s, 0) for s in doomed),
        "kept_spaces": len(keep_space_ids),
        "hint": "re-run with --yes to purge",
    }


async def purge_org(
    store: MemoryStore, org_ref: str, *, keep_space_ids: frozenset[str]
) -> dict[str, Any]:
    org, _ = await _purge_target(store, org_ref, keep_space_ids)
    report = await store.purge_org(org.id, keep_space_ids=keep_space_ids)
    return {"org_id": org.id, "kept_spaces": len(keep_space_ids), "purged": asdict(report)}


__all__ = [
    "create_org",
    "list_keys",
    "list_orgs",
    "mint_key",
    "open_admin_store",
    "parse_scopes",
    "plan_purge",
    "purge_org",
    "read_keep_list",
    "rename_org",
    "resolve_org",
    "revoke_key",
]
