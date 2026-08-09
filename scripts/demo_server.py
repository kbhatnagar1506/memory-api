"""A seeded server for looking at the memory graph.

Boots the real app with the in-memory store and writes a small but
deliberately *relational* corpus: a policy that gets revised, a fact computed
from two episodes, and two statements that contradict each other. Those are
the three edge types no extraction-built graph can draw, because extraction
links a document to its chunks and stops there.

Run:  .venv/bin/python -m scripts.demo_server
Then: open the URL it prints (space id and key are generated).
"""

from __future__ import annotations

import contextlib
import pathlib
from datetime import UTC, datetime

import uvicorn

from mapi.config import EmbeddingBackend, RerankBackend, Settings, StoreBackend
from mapi.domain.models import (
    MemoryKind,
    Organization,
    RelationType,
    Scope,
    Space,
)
from mapi.main import create_app

PORT = 8077


def _at(month: int, day: int) -> datetime:
    return datetime(2026, month, day, 9, 30, tzinfo=UTC)


async def seed(app) -> tuple[str, str]:
    from mapi.core.security import build_api_key

    store = app.state.store
    service = app.state.service
    org = await store.create_organization(Organization(name="Demo Org"))
    space = await store.create_space(
        Space(org_id=org.id, slug="engineering", name="Engineering")
    )
    record, key = build_api_key(
        org_id=org.id,
        name="demo",
        pepper="dev-insecure-pepper",
        scopes=frozenset(Scope.all()),
    )
    await store.create_api_key(record)

    async def write(text: str, when: datetime, **kw):
        return (
            await service.ingest(
                org_id=org.id, space_id=space.id, content=text, occurred_at=when, **kw
            )
        ).memory

    # -- a policy that gets revised twice: a supersession CHAIN, not a pair.
    p1 = await write("Customer data is retained for 90 days.", _at(1, 12))
    p2 = await write("Retention reduced to 60 days after the audit.", _at(3, 4))
    p3 = await write("Retention is now 30 days under the new DPA.", _at(6, 18))
    for newer, older in ((p2, p1), (p3, p2)):
        await service.link(
            org.id,
            space.id,
            source_id=newer.id,
            target_id=older.id,
            relation=RelationType.SUPERSEDES,
        )

    # -- episodes, and a fact COMPUTED from them. The derived_from edges are
    #    what make erasure propagate; without them nothing knows what a
    #    deleted source invalidates.
    e1 = await write("Shipped the auth rewrite on Tuesday.", _at(4, 7))
    e2 = await write("Shipped the billing migration on Thursday.", _at(4, 9))
    e3 = await write("Shipped the search reindex the following Monday.", _at(4, 13))
    fact = await write("Three releases went out in April.", _at(4, 30), kind=MemoryKind.DERIVED)
    for src in (e1, e2, e3):
        await service.link(
            org.id,
            space.id,
            source_id=fact.id,
            target_id=src.id,
            relation=RelationType.DERIVED_FROM,
        )

    # -- two statements that cannot both be true, neither replacing the other.
    #    A differing figure on the same subject, because this server runs the
    #    deterministic embedder: it is a hash, so it only scores lexically
    #    close pairs above the similarity gate. A real embedder would also
    #    catch "in us-east-1" against "in eu-west-2"; that hash does not.
    await write("The on-call rotation has 8 engineers.", _at(5, 2))
    await write("The on-call rotation has 12 engineers.", _at(5, 9), detect_conflicts=True)

    # -- a reference, and some islands so a sparse graph looks sparse.
    note = await write("See the incident review for the outage timeline.", _at(5, 20))
    inc = await write("Outage on May 19 caused by a bad migration.", _at(5, 19))
    await service.link(
        org.id,
        space.id,
        source_id=note.id,
        target_id=inc.id,
        relation=RelationType.REFERENCES,
    )
    for text, when in (
        ("Standup: nothing blocking today.", _at(2, 3)),
        ("Coffee machine on floor 3 is fixed.", _at(2, 14)),
        ("Team offsite booked for September.", _at(7, 1)),
    ):
        await write(text, when)

    return space.id, key


def main() -> None:
    settings = Settings(
        store_backend=StoreBackend.MEMORY,
        embedding_backend=EmbeddingBackend.DETERMINISTIC,
        embedding_dimensions=256,
        rerank_backend=RerankBackend.NONE,
    )
    app = create_app(settings)

    # `create_app` already installs a lifespan, and FastAPI silently ignores
    # on_event handlers once one exists -- so wrap it rather than add to it.
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(scope):
        async with inner(scope):
            space_id, key = await seed(app)
            url = f"http://127.0.0.1:{PORT}/graph?space={space_id}&key={key}"
            pathlib.Path(__file__).with_name(".demo_url").write_text(url)
            print(f"\n{'=' * 72}\n  MEMORY GRAPH\n  {url}\n{'=' * 72}\n", flush=True)
            yield

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_config=None)


if __name__ == "__main__":
    main()
