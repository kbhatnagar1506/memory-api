"""Run the stage trace over the lab corpus. No credentials required.

    python -m bench.lab.run_trace

The retrieval half of every verdict runs with the deterministic embedder, so
this is free and repeatable. Answer-side verdicts (SYNTHESIS vs SCORING) need a
model; when cached answers from a previous experiment exist they are used, and
otherwise the answer column is blank and the verdict is retrieval-only --
which is `sufficient@k` with the missing evidence named per question.

EXPECTATIONS are authored here in the lab's `must + bonus` form rather than in
`corpus.py`'s legacy strings, with per-question EVIDENCE markers naming which
session holds the answer. The evidence markers are what turn "this question
failed" into "this question failed because session s02 was not retrieved".
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from mapi.config import Settings
from mapi.domain.embeddings.deterministic import DeterministicEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

from .corpus import SESSIONS
from .scoring import Expected
from .tracer import report, trace

#: question -> (expected spec, evidence markers). The markers are substrings of
#: the sessions that hold the answer -- session text, not gold labels, so the
#: trace works on content the way a debugger would.
CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "What database do we use?": ("postgres", ("Postgres 16",)),
    "Who accepted the founding engineer offer?": ("priya", ("Priya accepted",)),
    "How many vendor invoices did I process in January?": ("3|three", ("INV-4471",)),
    "How many candidates did I interview for the founding engineer role?": (
        "3|three",
        ("Interviewed three candidates",),
    ),
    "Which invoice was dated earliest in January?": ("inv-4471", ("INV-4471",)),
    "What was the last thing I submitted to a venue?": (
        "persistence of vision",
        ("Submitted The Persistence of Vision",),
    ),
    "How many days before the PoPETs deadline did I submit?": ("6|six", ("six days",)),
    "Is the Lisbon hotel cheaper per night than last year?": ("yes + 180 210", ("Lisbon",)),
    "Which costs more per month, Cloud SQL or Heroku?": ("cloud sql", ("410 dollars",)),
    "Can you recommend how I should book my next long flight?": (
        "aisle; red-eye|red eye|redeye",
        ("aisle",),
    ),
    "What is our entire infrastructure?": (
        "postgres; redis; heroku; python",
        ("Postgres 16", "Redis 7"),
    ),
    "How much does Cloud SQL cost per month?": ("410", ("410 dollars",)),
    "Are we moving off Redis?": ("no|not", ("NOT moving off Redis",)),
    "What did I do in March?": ("migrat|migration", ("Migrated off Heroku Postgres",)),
    "What is invoice INV-4472 for?": ("datadog + 890", ("INV-4472",)),
}


async def main() -> None:
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=256,
        rerank_backend="heuristic",
        api_key_pepper="lab-trace-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=256), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="Lab"))
    space = await store.create_space(Space(org_id=org.id, slug="trace", name="Trace"))
    for session_id, occurred, tags, body in SESSIONS:
        await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content=body.strip(),
            occurred_at=occurred,
            tags=tuple(tags),
            metadata={"session_id": session_id},
            extract=False,
        )

    # Cached answers from the most recent format experiment, if present, so the
    # synthesis half of the classification runs without new model calls. The
    # "blocks" arm is the shipped format.
    answers: dict[str, str] = {}
    cache = Path(__file__).with_name("format_results.json")
    if cache.exists():
        answers = json.load(cache.open()).get("answers", {}).get("blocks", {})
        print(f"using cached answers for {len(answers)} questions (blocks arm)")
    else:
        print("no cached answers; running the retrieval half only")

    asked_at = datetime(2026, 8, 13, tzinfo=UTC)
    verdicts = []
    for question, (spec, evidence) in CASES.items():
        verdicts.append(
            await trace(
                service,
                org.id,
                space.id,
                question=question,
                expected=Expected.parse(spec),
                evidence=evidence,
                answer=answers.get(question, ""),
                limit=6,
                asked_at=asked_at,
            )
        )
    print(report(verdicts))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
