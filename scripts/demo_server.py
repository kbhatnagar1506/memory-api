"""A server seeded with real LongMemEval histories, one space per person.

Six profiles, one per question type the benchmark tests, each holding that
person's actual chat history: ten to sixteen sessions spread over months, the
user's own turns as individual memories. Every profile carries the benchmark's
question and gold answer in its space metadata, so the viewer can show what
this history is supposed to be able to answer.

Nothing in the graph is hand-drawn. The seeder writes memories and turns on
`auto_supersede` and `detect_conflicts`; every edge you see was proposed by
the same code path a live write goes through. That is the point of seeding
from a real corpus instead of a tidy fixture -- a fixture can be made to show
whatever the author wants, and this cannot.

Run:  .venv/bin/python -m scripts.build_demo_corpus   (once)
      .venv/bin/python -m scripts.demo_server
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import time
from datetime import UTC, datetime
from typing import Any

import uvicorn

from mapi.config import EmbeddingBackend, RerankBackend, Settings, StoreBackend
from mapi.domain.models import Organization, Scope, Space
from mapi.main import create_app

from .cached_embedder import DiskCachedEmbedder

PORT = 8077
CORPUS = pathlib.Path(__file__).with_name(".demo_corpus.json")
VECTORS = pathlib.Path("bench/data/cache/vectors.sqlite")

#: Same model and width the benchmark runs, so the cache is shared with it.
MODEL = "text-embedding-004"
DIMENSIONS = 768

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?:[^\d]+(\d{2}):(\d{2}))?")

#: One space per person, named by the benchmark's own question id. Inventing
#: display names would make two runs of the same corpus impossible to line up
#: against each other, or against a LongMemEval result file.
SPACE_PREFIX = "lme"


def _adc_project() -> str | None:
    """The project ADC was set up with, so the demo needs no env var."""
    path = pathlib.Path.home() / ".config/gcloud/application_default_credentials.json"
    if not path.exists():
        return None
    with contextlib.suppress(Exception):
        return str(json.loads(path.read_text()).get("quota_project_id")) or None
    return None


def _when(raw: str) -> datetime:
    m = _DATE_RE.search(raw or "")
    if not m:
        return datetime(2023, 1, 1, tzinfo=UTC)
    y, mo, d, hh, mm = m.groups()
    return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), tzinfo=UTC)


def _slug(question_id: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", question_id.casefold()).strip("-")
    return f"{SPACE_PREFIX}-{clean}"[:64]


async def seed(app: Any) -> tuple[str, str, list[dict[str, Any]]]:
    from mapi.core.security import build_api_key

    if not CORPUS.exists():
        raise SystemExit(
            f"{CORPUS} not found -- run: .venv/bin/python -m scripts.build_demo_corpus"
        )
    corpus = json.loads(CORPUS.read_text())

    store = app.state.store
    service = app.state.service
    org = await store.create_organization(Organization(name="LongMemEval demo"))
    record, key = build_api_key(
        org_id=org.id,
        name="demo",
        pepper="dev-insecure-pepper",
        scopes=frozenset(Scope.all()),
    )
    await store.create_api_key(record)

    started = time.perf_counter()
    profiles: list[dict[str, Any]] = []

    for p in corpus["profiles"]:
        name = _slug(p["question_id"])
        space = await store.create_space(
            Space(
                org_id=org.id,
                slug=name,
                name=name,
                description=p["question"],
                metadata={
                    "question": p["question"],
                    "answer": p["answer"],
                    "question_type": p["question_type"],
                    "question_id": p["question_id"],
                    "asked_at": p["question_date"],
                    "evidence_session_ids": p["evidence_session_ids"],
                    "sessions_shown": len(p["sessions"]),
                    "sessions_total": p["sessions_total"],
                },
            )
        )

        written = superseded = conflicts = 0
        for session in p["sessions"]:
            occurred = _when(session["date"])
            # Session only. `is_evidence` is a gold label and it stays out of
            # the tags: `propose_supersessions` scores shared tags, so tagging
            # the answer sessions would feed the benchmark's own key back into
            # the mechanism being demonstrated. It rides in metadata instead,
            # which nothing scores.
            tags = [f"session:{session['id']}"]
            for turn in session["turns"]:
                result = await service.ingest(
                    org_id=org.id,
                    space_id=space.id,
                    content=turn,
                    occurred_at=occurred,
                    source=session["id"],
                    tags=tags,
                    metadata={
                        "session_id": session["id"],
                        "session_date": session["date"],
                        "is_evidence": session["is_evidence"],
                    },
                    # Every edge in the picture comes from here.
                    auto_supersede=True,
                    detect_conflicts=True,
                )
                written += 1
                superseded += len(result.superseded)
                conflicts += len(result.contradicts)

        profiles.append(
            {
                "name": name,
                "space_id": space.id,
                "type": p["question_type"],
                "memories": written,
                "sessions": len(p["sessions"]),
                "superseded": superseded,
                "conflicts": conflicts,
            }
        )
        print(
            f"  {name:<6} {p['question_type']:<26} {len(p['sessions']):>2} sessions  "
            f"{written:>3} memories  {superseded:>2} superseded  {conflicts:>2} conflicts",
            flush=True,
        )

    print(f"  seeded in {time.perf_counter() - started:.0f}s", flush=True)
    return org.id, key, profiles


def main() -> None:
    settings = Settings(
        store_backend=StoreBackend.MEMORY,
        embedding_backend=EmbeddingBackend.GEMINI,
        embedding_model=MODEL,
        embedding_dimensions=DIMENSIONS,
        embedding_batch_size=32,
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT") or _adc_project(),
        rerank_backend=RerankBackend.HEURISTIC,
    )
    app = create_app(settings)

    # `create_app` already installs a lifespan, and FastAPI silently ignores
    # on_event handlers once one exists -- so wrap it rather than add to it.
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(scope: Any) -> Any:
        async with inner(scope):
            # Wrap the embedder `create_app` already built, rather than
            # constructing a second client against the same project.
            app.state.embedder = DiskCachedEmbedder(app.state.embedder, VECTORS)
            app.state.service.embedder = app.state.embedder
            _org, key, profiles = await seed(app)
            url = f"http://127.0.0.1:{PORT}/graph?key={key}"
            pathlib.Path(__file__).with_name(".demo_url").write_text(url)
            print(f"\n{'=' * 72}\n  MEMORY GRAPH\n  {url}\n{'=' * 72}\n", flush=True)
            _ = profiles
            yield

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_config=None)


if __name__ == "__main__":
    main()
