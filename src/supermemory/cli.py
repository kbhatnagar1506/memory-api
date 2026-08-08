"""Operational CLI: serve, seed a demo corpus, run a query, check config."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import EmbeddingBackend, RerankBackend, Settings, StoreBackend, get_settings
from .core.security import build_api_key
from .domain.models import Organization, Scope, Space
from .domain.retrieval.pipeline import SearchRequest


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "supermemory.main:app",
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
    say("6. PROOF - the content is gone from every timeline")
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
    ok = gone_now is None and gone_then is None and not gone_history
    say()
    say("VERDICT: " + ("erasure propagated to live, past and history - PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _demo_replay(args: argparse.Namespace) -> int:
    return asyncio.run(_demo_replay_async(args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supermemory", description=__doc__)
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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
