"""Bulk write benchmark: a 91-section attendee memory through a real Postgres.

    MAPI_TEST_DATABASE_URL=postgresql+asyncpg://... MAPI_TEST_DIMENSIONS=768 \\
    MAPI_TEST_SCHEMA=<migrated schema> \\
        .venv/bin/python -m bench.bulk_write [--items 91] [--mode both|batched|sequential]

Wall time depends on where the database is: production runs next to it at
~2.5 ms a round trip, a laptop through the Cloud SQL proxy sees 30-45 ms. The
number that travels between the two is the ROUND-TRIP COUNT, so that is what
this reports first -- counted at the asyncpg boundary, where every prepare,
bound execute, pipelined executemany and transaction verb is one exchange
with the server -- and then the wall time it produced here.

Embeddings are the deterministic hasher, so no vendor is called and the
embedding time is near zero: what is left is the write path itself.

Three scenarios per mode, each in a fresh space: `first sync` (91 new
sections), `re-sync` (the same 91 again: all exact duplicates) and `edit`
(the same 91 with every fifth section changed).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

#: Words the synthetic sections are drawn from. Varied enough that sections
#: are not near-duplicates of each other, repetitive enough to look like prose.
_WORDS = (
    "robotics pottery sailing compilers birding chess baking climbing jazz "
    "kubernetes postgres gardening marathon photography origami violin "
    "startup fundraising hiring mentoring research thesis lab protein "
    "genome satellite telescope drone welding carpentry cycling surfing "
    "espresso ramen sourdough kimchi tacos museum archive library podcast "
    "novel poetry theatre film animation shader compiler kernel scheduler "
    "latency throughput cache index vector embedding retrieval ranking"
)
_VOCAB = _WORDS.split()


def attendee_sections(n: int = 91, *, seed: int = 7) -> list[dict[str, Any]]:
    """`n` sections of 1-4 KB, shaped like the game's attendee memory."""
    rng = random.Random(seed)
    sections = []
    for i in range(n):
        target = rng.randint(1_000, 4_000)
        words: list[str] = [f"Section {i}:"]
        size = len(words[0])
        while size < target:
            sentence = " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(6, 14)))
            sentence = f"{sentence.capitalize()} {rng.randint(1, 9999)}."
            words.append(sentence)
            size += len(sentence) + 1
        key = f"muse:s{i:03d}"
        sections.append(
            {
                "content": " ".join(words),
                "metadata": {"key": key, "section": f"s{i % 9}"},
                "tags": [f"s{i % 9}"],
                "source": key,
            }
        )
    return sections


@dataclass
class RoundTrips:
    """Server exchanges made through asyncpg while the counter was installed."""

    prepare: int = 0
    execute: int = 0
    executemany: int = 0

    @property
    def total(self) -> int:
        return self.prepare + self.execute + self.executemany


@contextlib.contextmanager
def count_round_trips() -> Iterator[RoundTrips]:
    """Count every asyncpg call that is one exchange with the server.

    Patched at the driver's public surface, which is what SQLAlchemy's
    adapter calls: `Connection.prepare` (Parse/Describe), `PreparedStatement.
    fetch` (Bind/Execute/Sync), `Connection.executemany` (pipelined: every
    Bind, one Sync) and `Connection.execute` / `fetchrow` (simple queries --
    which is also how asyncpg sends BEGIN, COMMIT and ROLLBACK, so the
    transaction verbs are counted there, once each). A statement asyncpg
    prepares for itself inside `executemany` the first time it sees it is not
    counted; that happens once per connection, not per request.
    """
    import asyncpg
    from asyncpg.prepared_stmt import PreparedStatement

    counts = RoundTrips()
    patched: list[tuple[Any, str, Any]] = []

    def wrap(owner: Any, name: str, bucket: str) -> None:
        original = getattr(owner, name)

        async def counted(*args: Any, **kwargs: Any) -> Any:
            setattr(counts, bucket, getattr(counts, bucket) + 1)
            return await original(*args, **kwargs)

        patched.append((owner, name, original))
        setattr(owner, name, counted)

    wrap(asyncpg.Connection, "prepare", "prepare")
    wrap(asyncpg.Connection, "execute", "execute")
    wrap(asyncpg.Connection, "fetchrow", "execute")
    wrap(asyncpg.Connection, "executemany", "executemany")
    wrap(PreparedStatement, "fetch", "execute")
    try:
        yield counts
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)


def _items(sections: list[dict[str, Any]]) -> list[Any]:
    from mapi.service import IngestItem

    return [IngestItem(**section) for section in sections]


async def _service(store: Any, *, batched: bool) -> Any:
    from tests.support import postgres as pg_support

    from mapi.config import Settings
    from mapi.domain.embeddings import DeterministicEmbedder
    from mapi.domain.retrieval.rerank import HeuristicReranker
    from mapi.service import MemoryService

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=pg_support.dimensions(),
        rerank_backend="none",
        api_key_pepper="bench-pepper",
        embedding_cache_size=0,
        bulk_write_batching=batched,
    )
    embedder = DeterministicEmbedder(dimensions=pg_support.dimensions(), batch_size=100)
    return MemoryService(store, embedder, HeuristicReranker(), settings)


async def run(n: int, modes: list[str]) -> list[dict[str, Any]]:
    from tests.support import postgres as pg_support

    from mapi.bulkwrite import BulkStats
    from mapi.domain.models import Organization, Space

    store = pg_support.make_store()
    await store.initialize()
    rows: list[dict[str, Any]] = []
    try:
        org = await store.create_organization(Organization(name="bulk-bench"))
        first = attendee_sections(n)
        edited = [
            {**s, "content": s["content"] + " Edited."} if i % 5 == 0 else s
            for i, s in enumerate(first)
        ]
        for mode in modes:
            service = await _service(store, batched=mode == "batched")
            space = await store.create_space(
                Space(org_id=org.id, slug=f"bench-{mode}-{time.time_ns()}", name="bench")
            )
            # Warm the pool and the space cache so connection setup is not
            # charged to the first scenario.
            await service.get_space_or_raise(org.id, space.id)
            await store.ping()
            scenarios = (("first sync", first), ("re-sync", first), ("edit", edited))
            for scenario, sections in scenarios:
                stats = BulkStats()
                start = time.perf_counter()
                with count_round_trips() as trips:
                    outcomes = await service.ingest_many(
                        org_id=org.id, space_id=space.id, items=_items(sections), stats=stats
                    )
                wall = (time.perf_counter() - start) * 1000
                created = sum(1 for o in outcomes if o.result and o.result.created)
                failed = sum(1 for o in outcomes if o.error is not None)
                rows.append(
                    {
                        "mode": mode,
                        "scenario": scenario,
                        "items": n,
                        "created": created,
                        "failed": failed,
                        "round_trips": trips.total,
                        "per_item": round(trips.total / n, 2),
                        "wall_ms": round(wall, 1),
                        "db_ms": round(stats.db_ms, 1),
                        "embed_ms": round(stats.embed_ms, 1),
                    }
                )
    finally:
        await store.aclose()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bulk_write", description=__doc__)
    parser.add_argument("--items", type=int, default=91)
    parser.add_argument("--mode", choices=["both", "batched", "sequential"], default="both")
    args = parser.parse_args(argv)
    modes = ["sequential", "batched"] if args.mode == "both" else [args.mode]
    rows = asyncio.run(run(args.items, modes))
    header = f"{'mode':<11} {'scenario':<11} {'rt':>6} {'rt/item':>8} {'wall ms':>9}"
    header += f" {'db ms':>8} {'created':>8} {'failed':>7}"
    print(header)
    for r in rows:
        print(
            f"{r['mode']:<11} {r['scenario']:<11} {r['round_trips']:>6} {r['per_item']:>8}"
            f" {r['wall_ms']:>9} {r['db_ms']:>8} {r['created']:>8} {r['failed']:>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
