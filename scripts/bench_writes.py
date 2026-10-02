"""What one write costs as a space fills up, on a real Postgres.

    MAPI_TEST_DATABASE_URL=postgresql+asyncpg://... python scripts/bench_writes.py 50 300 1000

Ingests memories into a fresh space at 768 dimensions with the deterministic embedder (no
network) and reports per-write latency at each size: p50 and p95 over the last 20 writes
before the size is reached. Every write runs the full path: chunk, embed, neighbours,
dedupe, supersession, association, store.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

from mapi.config import Settings
from mapi.domain.embeddings import DeterministicEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.postgres.store import PostgresStore

DIMENSIONS = 768
TOPICS = ("order", "refund", "address", "payment", "flight", "baggage", "calendar", "email")


async def main(sizes: list[int]) -> None:
    store = PostgresStore(os.environ["MAPI_TEST_DATABASE_URL"], dimensions=DIMENSIONS)
    await store.initialize()
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=DIMENSIONS,
        rerank_backend="heuristic",
        api_key_pepper="x" * 32,
    )
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=DIMENSIONS), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="bench"))
    space = await store.create_space(
        Space(org_id=org.id, slug=f"bench-{int(time.time())}", name="bench")
    )
    timings: list[float] = []
    written = 0
    print(f"{'memories':>9} {'p50 ms':>8} {'p95 ms':>8}")
    for size in sorted(sizes):
        while written < size:
            topic = TOPICS[written % len(TOPICS)]
            started = time.perf_counter()
            await service.ingest(
                org_id=org.id,
                space_id=space.id,
                content=f"Function {written} handles the {topic} step {written % 37} "
                f"for customer segment {written % 11}.",
                extract=False,
            )
            timings.append((time.perf_counter() - started) * 1000)
            written += 1
        last = sorted(timings[-20:])
        p95 = last[max(0, round(len(last) * 0.95) - 1)]
        print(f"{size:>9} {statistics.median(last):>8.1f} {p95:>8.1f}")
    await store.aclose()


if __name__ == "__main__":
    asyncio.run(main([int(a) for a in sys.argv[1:]] or [50, 300, 1000]))
