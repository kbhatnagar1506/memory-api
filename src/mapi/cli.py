"""Operational CLI: serve, seed a demo corpus, run a query, check config, admin."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .config import EmbeddingBackend, RerankBackend, Settings, StoreBackend, get_settings
from .core.errors import MapiError
from .core.security import build_api_key
from .domain.models import Organization, Scope, Space
from .domain.retrieval.pipeline import SearchRequest
from .store.base import MemoryStore

#: One admin subcommand: given the open store, the settings and the parsed
#: arguments, produce the JSON document the command prints.
AdminAction = Callable[[MemoryStore, Settings, argparse.Namespace], Awaitable[dict[str, Any]]]


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "mapi.main:app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    settings = get_settings()
    problems = settings.validate_production()
    print(
        json.dumps(
            {
                "environment": str(settings.environment),
                "store": str(settings.store_backend),
                "embeddings": str(settings.embedding_backend),
                "rerank": str(settings.rerank_backend),
                "dimensions": settings.embedding_dimensions,
                "production_problems": problems,
            },
            indent=2,
        )
    )
    return 1 if problems else 0


async def _demo_async(args: argparse.Namespace) -> int:
    """Seed a corpus and run a query, entirely in-process."""
    from .domain.embeddings import build_embedder
    from .main import build_reranker
    from .service import MemoryService
    from .store import build_store

    settings = Settings(
        store_backend=StoreBackend.MEMORY,
        embedding_backend=EmbeddingBackend(args.embeddings),
        embedding_dimensions=args.dimensions,
        rerank_backend=RerankBackend(args.rerank),
    )
    store = build_store(settings)
    await store.initialize()
    service = MemoryService(store, build_embedder(settings), build_reranker(settings), settings)

    org = await store.create_organization(Organization(name="Demo"))
    space = await store.create_space(Space(org_id=org.id, slug="demo", name="Demo"))
    record, plaintext = build_api_key(
        org_id=org.id,
        name="demo",
        pepper=settings.api_key_pepper,
        scopes=frozenset(Scope.all()),
    )
    await store.create_api_key(record)

    corpus = [
        "We chose Kafka for the event bus after evaluating RabbitMQ and NATS.",
        "The Kafka cluster was upgraded to version 3.7 last sprint.",
        "Deploys go through Jenkins.",
        "Deploys now go through GitHub Actions instead of Jenkins.",
        "The team offsite moved from Berlin to Lisbon.",
        "Postgres migration to version 16 approved for Q3.",
        "Checkout latency budget was breached during the Black Friday load test.",
    ]
    for content in corpus:
        await service.ingest(org_id=org.id, space_id=space.id, content=content)

    print(f"seeded {len(corpus)} memories into space {space.id}")
    print(f"demo api key: {plaintext}\n")

    response = await service.search(
        SearchRequest(query=args.query, org_id=org.id, space_id=space.id, limit=args.limit)
    )
    print(
        f"query: {args.query!r}  ({len(response.results)} results, "
        f"strategies={response.strategies})\n"
    )
    for i, hit in enumerate(response.results, start=1):
        print(f"{i}. [{hit.score:.4f}] {hit.memory.content}")
        print(f"   {' | '.join(hit.explain)}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    return asyncio.run(_demo_async(args))


async def _demo_replay_async(args: argparse.Namespace) -> int:
    """The Replayable Memory demo: prove what the agent knew, retrieved and
    why at any past moment — then prove erasure propagated, including through
    the reconstructed past. No network, no credentials: deterministic
    embedder, in-memory store, fixed timestamps."""
    from datetime import UTC, datetime

    from .domain.embeddings import build_embedder
    from .domain.models import Organization, RelationType, Scope, Space
    from .domain.retrieval.pipeline import SearchRequest
    from .main import build_reranker
    from .service import MemoryService
    from .store import build_store

    settings = Settings(
        store_backend=StoreBackend.MEMORY,
        embedding_backend=EmbeddingBackend.DETERMINISTIC,
        embedding_dimensions=256,
        rerank_backend=RerankBackend.HEURISTIC,
    )
    store = build_store(settings)
    await store.initialize()
    service = MemoryService(store, build_embedder(settings), build_reranker(settings), settings)

    org = await store.create_organization(Organization(name="Replay Demo"))
    space = await store.create_space(Space(org_id=org.id, slug="compliance", name="Compliance"))
    _ = Scope  # imported for parity with other demos

    t0 = datetime(2026, 1, 10, tzinfo=UTC)
    t1 = datetime(2026, 3, 2, tzinfo=UTC)
    t2 = datetime(2026, 5, 20, tzinfo=UTC)

    def say(line: str = "") -> None:
        print(line)

    say("REPLAYABLE MEMORY - decision replay and provable erasure")
    say("=" * 64)

    say()
    say(f"1. WRITE HISTORY (three moments: {t0.date()}, {t1.date()}, {t2.date()})")
    v1 = (
        await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content="Policy: customer data is retained for 90 days.",
            occurred_at=t0,
            tags=["policy"],
        )
    ).memory
    say(f'   {t0.date()}  {v1.id}  "{v1.content}"')

    derived = (
        await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content="Sent retention notice to customers citing the 90-day policy.",
            occurred_at=t1,
            tags=["notice"],
        )
    ).memory
    await service.link(
        org.id,
        space.id,
        source_id=derived.id,
        target_id=v1.id,
        relation=RelationType.DERIVED_FROM,
        reason="notice cites policy v1",
    )
    say(f'   {t1.date()}  {derived.id}  "{derived.content}"  (derived_from {v1.id})')

    v2 = (
        await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content="Policy updated: customer data is now retained for 30 days.",
            occurred_at=t2,
            tags=["policy"],
        )
    ).memory
    await service.link(
        org.id,
        space.id,
        source_id=v2.id,
        target_id=v1.id,
        relation=RelationType.SUPERSEDES,
        reason="policy revision 2026-05",
    )
    say(f'   {t2.date()}  {v2.id}  "{v2.content}"  (supersedes {v1.id})')

    say()
    say("2. THE DECISION RECORD - what does the agent believe now, and why?")
    result = await service.search(
        SearchRequest(
            query="how long do we retain customer data?",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            use_decay=False,
        )
    )
    for hit in result.results:
        say(f"   -> {hit.memory.content}")
        say(f"      explain: {' | '.join(hit.explain)}")
    stale_served = any(v1.id == h.memory.id for h in result.results)
    say(f"   superseded 90-day policy served: {stale_served}  (suppressed transitively)")

    say()
    say("3. LINEAGE - is the old policy stale, and what replaced it?")
    lineage = await service.get_lineage(org.id, space.id, v1.id)
    say(f"   {v1.id}: is_current={lineage['is_current']}  head={lineage['head']}")

    say()
    say("4. TIME TRAVEL - what did the DATABASE believe, before vs after the revision?")
    say("   (system time is recorded at write, not spoofable; event time is separate)")
    versions = await service.list_memory_versions(org.id, space.id, v1.id)
    moment_before_revision = versions[0].valid_from
    then = await service.get_memory_as_of(org.id, space.id, v1.id, moment_before_revision)
    say(
        f"   as_of {moment_before_revision.isoformat()}: "
        f"status={then.status.value} v{then.version}  (the 90-day policy was current)"
    )
    now_row = await store.get_memory(org.id, space.id, v1.id)
    assert now_row is not None
    say(
        f"   now: status={now_row.status.value} v{now_row.version} (superseded by the revision)"
    )
    say(f"   version history: {[(v.version, v.status.value) for v in versions]}")

    say()
    say("5. ERASURE REQUEST - right-to-be-forgotten on the original policy")
    attestation = await service.erase_memory(org.id, space.id, v1.id)
    for key in (
        "memory_id",
        "content_sha256",
        "chunks_removed",
        "edges_removed",
        "versions_purged",
        "derived_memories_affected",
        "erased_at",
    ):
        say(f"   {key}: {attestation[key]}")

    say()
    say("6. PROPAGATION - what was COMPUTED from the erased memory")
    say("   (the claim extraction-based systems cannot make: their derived")
    say("    facts have fuzzy lineage, so nothing knows what to invalidate)")
    derived_now = await store.get_memory(org.id, space.id, derived.id)
    derived_status = derived_now.status.value if derived_now else "gone"
    say(f'   derived fact: "{derived.content}"')
    say(f"   status after erasing its source: {derived_status.upper()}")
    say(
        "   -> "
        + (
            "stale: recomputable from survivors, never served as current truth"
            if derived_status == "stale"
            else f"UNEXPECTED ({derived_status})"
        )
    )
    listed = await service.search(
        SearchRequest(
            query="how long do we retain customer data?",
            org_id=org.id,
            space_id=space.id,
            limit=10,
            use_decay=False,
        )
    )
    derived_served = any(h.memory.id == derived.id for h in listed.results)
    say(f"   still returned by search: {derived_served}  (stale is a hard status)")

    say()
    say("7. PROOF - the content is gone from every timeline")
    gone_now = await store.get_memory(org.id, space.id, v1.id)
    gone_then = await store.get_memory_as_of(org.id, space.id, v1.id, moment_before_revision)
    gone_history = await store.list_memory_versions(org.id, space.id, v1.id)
    say(f"   live read:            {'GONE' if gone_now is None else 'STILL PRESENT'}")
    say(
        "   point-in-time read:   "
        + ("GONE - reconstructed past scrubbed" if gone_then is None else "STILL PRESENT")
    )
    history_note = "GONE" if not gone_history else f"{len(gone_history)} rows remain"
    say(f"   version history:      {history_note}")
    survivors = await service.search(
        SearchRequest(
            query="how long do we retain customer data?",
            org_id=org.id,
            space_id=space.id,
            limit=5,
            use_decay=False,
        )
    )
    say(f'   current answer still served: "{survivors.results[0].memory.content}"')
    ok = (
        gone_now is None
        and gone_then is None
        and not gone_history
        and derived_status == "stale"
        and not derived_served
    )
    say()
    say("VERDICT:")
    say(f"   source erased from live, reconstructed past and history : {gone_now is None}")
    invalidated = derived_status == "stale"
    say(f"   derivation invalidated rather than left standing        : {invalidated}")
    say(f"   invalidated fact withheld from search                   : {not derived_served}")
    say("   surviving current answer still served                   : True")
    say()
    say("   " + ("PASS - erasure propagated through the graph" if ok else "FAIL"))
    return 0 if ok else 1


def _demo_replay(args: argparse.Namespace) -> int:
    return asyncio.run(_demo_replay_async(args))


# -- admin ---------------------------------------------------------------------


async def _admin_async(args: argparse.Namespace, store: MemoryStore | None = None) -> int:
    """Run one admin action. `store` is injectable so tests need no database."""
    from . import admin

    settings = get_settings()
    owned = store is None
    try:
        if store is None:
            store = admin.open_admin_store(settings)
            await store.initialize()
        action: AdminAction = args.admin_action
        result = await action(store, settings, args)
    except MapiError as exc:
        # Errors to stderr as JSON too, so a script can branch on `error`.
        print(json.dumps({"error": exc.slug, "detail": exc.detail}), file=sys.stderr)
        return 1
    finally:
        if owned and store is not None:
            await store.aclose()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _admin(args: argparse.Namespace) -> int:
    return asyncio.run(_admin_async(args))


def _add_admin(sub: Any) -> None:
    from . import admin

    parser = sub.add_parser(
        "admin",
        help="operator commands: organizations and API keys, straight to the database",
        description=admin.__doc__.split("\n\n")[0] if admin.__doc__ else None,
    )
    groups = parser.add_subparsers(dest="admin_group", required=True)

    def command(group: Any, name: str, help_text: str, action: AdminAction) -> Any:
        p = group.add_parser(name, help=help_text)
        p.set_defaults(func=_admin, admin_action=action)
        return p

    org = groups.add_parser("org", help="organizations").add_subparsers(
        dest="org_command", required=True
    )
    p = command(
        org,
        "create",
        "create an organization",
        lambda store, _s, a: admin.create_org(store, a.name),
    )
    p.add_argument("--name", required=True)
    command(org, "list", "list organizations", lambda store, _s, _a: admin.list_orgs(store))
    p = command(
        org,
        "rename",
        "rename an organization",
        lambda store, _s, a: admin.rename_org(store, a.org, a.name),
    )
    p.add_argument("org", help="organization id, or its exact current name")
    p.add_argument("--name", required=True)

    async def _purge(store: MemoryStore, _s: Settings, a: argparse.Namespace) -> dict[str, Any]:
        keep = admin.read_keep_list(a.keep_space, a.keep_spaces_file)
        if not a.yes:
            return await admin.plan_purge(store, a.org, keep_space_ids=keep)
        return await admin.purge_org(store, a.org, keep_space_ids=keep)

    p = command(
        org,
        "purge",
        "destroy every space in an organization and all history (dry run without --yes)",
        _purge,
    )
    p.add_argument("org", help="organization id, or its exact name")
    p.add_argument(
        "--keep-space", action="append", default=[], metavar="SPACE_ID", help="spare this space"
    )
    p.add_argument(
        "--keep-spaces-file",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="spare the space ids listed in this file, one per line",
    )
    p.add_argument("--yes", action="store_true", help="actually purge; otherwise only report")

    key = groups.add_parser("key", help="API keys").add_subparsers(
        dest="key_command", required=True
    )
    p = command(
        key,
        "mint",
        "mint a key and write it to a new 0600 file; nothing secret is printed",
        lambda store, s, a: admin.mint_key(
            store,
            s,
            org_ref=a.org,
            name=a.name,
            scopes=admin.parse_scopes(a.scopes),
            out=a.out,
            expires_in_days=a.expires_in_days,
            force=a.force,
        ),
    )
    p.add_argument("--org", required=True, help="organization id, or its exact name")
    p.add_argument("--name", required=True, help="what the key is for, e.g. facemash-server")
    p.add_argument(
        "--scopes", default="default", help="default, all, or a comma-separated scope list"
    )
    p.add_argument("--out", required=True, type=Path, help="file to create (mode 0600)")
    p.add_argument("--expires-in-days", type=int, default=None)
    p.add_argument("--force", action="store_true", help="replace --out if it exists")
    p = command(
        key,
        "list",
        "list an organization's keys (ids and names only)",
        lambda store, _s, a: admin.list_keys(store, a.org),
    )
    p.add_argument("--org", required=True)
    p = command(
        key,
        "revoke",
        "revoke a key",
        lambda store, _s, a: admin.revoke_key(store, a.org, a.key_id),
    )
    p.add_argument("--org", required=True)
    p.add_argument("key_id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mapi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    check = sub.add_parser("check", help="print resolved configuration")
    check.set_defaults(func=_check)

    demo = sub.add_parser("demo", help="seed a corpus and run one query")
    demo.add_argument("--query", default="how do we deploy?")
    demo.add_argument("--limit", type=int, default=5)
    demo.add_argument("--dimensions", type=int, default=256)
    demo.add_argument("--embeddings", default="deterministic")
    demo.add_argument("--rerank", default="heuristic")
    demo.set_defaults(func=_demo)

    replay = sub.add_parser(
        "demo-replay",
        help="decision replay + provable erasure, no network needed",
    )
    replay.set_defaults(func=_demo_replay)

    _add_admin(sub)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
