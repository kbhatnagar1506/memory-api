"""Does write-time extraction close the rank-6 margin on preference retrieval?

The preference lab ended on a knife's edge: with the production embedder, the
two zero-overlap bridges ("book" -> "novel on the commute", "schedule my 1:1"
-> "blocked my calendar before 10am") were retrieved at ranks 5 and 6 of a
6-wide window. One slot of margin. k=4 misses them, and any distractor growth
pushes them out.

The hypothesis: extraction at write time turns a diary sentence into an
explicit claim ("The user blocks mornings before 10am for deep work"), and the
claim sits semantically closer to the advice question than the diary prose
does -- so the evidence should move UP the ranking, not merely into it.

The counter-hypothesis, measured once on LongMemEval and worth respecting:
extraction ADDS retrieval units, and claims from one session can crowd the
window (full_recall fell 0.968 -> 0.948 while MRR rose). This corpus is small
enough that crowding should not bite, which is itself part of what the
comparison shows.

Two arms, identical corpus, identical questions, production embedder:

    off   sessions only (extract=False)          -- the measured baseline
    on    sessions + write-time claims           -- extractor: gemini-2.5-flash

Metric: per question, the best rank of any memory carrying the needed
preference (the session itself OR a claim derived from it), plus
`sufficient@4` -- whether the tighter window now contains the evidence.

    python -m bench.lab.extraction_rank        # ~10 extraction calls + embeds
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

from mapi.config import Settings
from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

from .preference import QUESTIONS, SESSIONS

#: question -> the session(s) whose preference must reach the window. For
#: `updated` questions the LATEST statement is required -- retrieving only the
#: superseded one is a failure wearing a success's rank.
NEEDS: dict[str, tuple[str, ...]] = {
    "Book me a seat for the six-hour flight to Seattle -- which seat?": ("p07",),
    "Can you recommend a hotel for my Barcelona trip?": ("p10",),
    "When should I schedule my weekly 1:1 with Priya?": ("p04",),
    "Pick a flight to New York for me: 6am red-eye arrival or midday departure?": ("p01",),
    "Recommend a keyboard for my new desk setup.": ("p03",),
    "Suggest a book for my flight.": ("p06",),
    "Window or aisle for the long-haul to Tokyo?": ("p07",),
    "What should I order at the steakhouse tomorrow?": ("p09",),
    "Recommend a birthday gift for Aaditi.": ("p08",),
    "Plan a restaurant for my birthday dinner.": ("p09", "p02"),
    "Suggest a weekend hiking trip.": ("p05",),
    "What drink should I bring for movie night?": ("p02",),
}


async def _build(extract: bool) -> tuple[MemoryService, str, str, dict[str, str]]:
    """One arm's corpus. Returns (service, org, space, memory_id -> session_id).

    The id map is what makes claims traceable: a claim carries
    `extracted_from` naming its parent memory, and the parent carries the
    session id -- so any retrieved row resolves to the session whose
    preference it expresses.
    """
    from bench.harness import build_model_client

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",  # unused; embedder injected below
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="extraction-lab-pepper",
    )
    extractor = None
    if extract:
        client = build_model_client("gemini-2.5-flash", "patchguard-reakon")

        async def extractor(prompt: str) -> str:
            raw, _tokens = await client.complete(prompt, max_tokens=2048)
            return raw

    store = InMemoryStore()
    service = MemoryService(
        store,
        GeminiEmbedder(dimensions=768),
        HeuristicReranker(),
        settings,
        extractor=extractor,
    )
    org = await store.create_organization(Organization(name="ExtractLab"))
    space = await store.create_space(Space(org_id=org.id, slug="x", name="X"))

    owner: dict[str, str] = {}
    for sid, when, text in SESSIONS:
        result = await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content=text,
            occurred_at=when,
            metadata={"session_id": sid},
            extract=extract,
        )
        owner[result.memory.id] = sid
        for claim_id in result.extracted:
            owner[claim_id] = sid
    return service, org.id, space.id, owner


async def main() -> None:
    asked = datetime(2026, 8, 13, tzinfo=UTC)
    tables: dict[str, dict[str, tuple[int | None, ...]]] = {}
    counts: dict[str, int] = {}

    for arm in ("off", "on"):
        service, org_id, space_id, owner = await _build(extract=(arm == "on"))
        counts[arm] = len(owner)
        ranks_by_question: dict[str, tuple[int | None, ...]] = {}
        for _kind, question, _spec in QUESTIONS:
            response = await service.search(
                SearchRequest(
                    query=question, org_id=org_id, space_id=space_id, limit=6, asked_at=asked
                )
            )
            returned = [owner.get(h.memory.id) for h in response.results]
            ranks = tuple(
                (returned.index(sid) + 1) if sid in returned else None
                for sid in NEEDS[question]
            )
            ranks_by_question[question] = ranks
        tables[arm] = ranks_by_question

    print(
        f"\nmemories per arm: off={counts['off']}  on={counts['on']} "
        f"(+{counts['on'] - counts['off']} extracted claims)"
    )
    print(f"\n{'question':52} {'off':>9} {'on':>9}")
    moved_up = moved_down = 0
    sufficient4 = {"off": 0, "on": 0}
    for _kind, question, _spec in QUESTIONS:
        off, on = tables["off"][question], tables["on"][question]

        def show(ranks: tuple[int | None, ...]) -> str:
            return ",".join("--" if r is None else str(r) for r in ranks)

        def worst(ranks: tuple[int | None, ...]) -> int:
            return max((r if r is not None else 99) for r in ranks)

        for arm, ranks in (("off", off), ("on", on)):
            if worst(ranks) <= 4:
                sufficient4[arm] += 1
        delta = worst(off) - worst(on)
        moved_up += delta > 0
        moved_down += delta < 0
        mark = "  ^" if delta > 0 else ("  v" if delta < 0 else "")
        print(f"  {question[:50]:52} {show(off):>7} {show(on):>7}{mark}")

    print(
        f"\nsufficient@4 (evidence inside a 4-wide window): "
        f"off {sufficient4['off']}/12   on {sufficient4['on']}/12"
    )
    print(f"questions where evidence moved up: {moved_up}, down: {moved_down}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
