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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
