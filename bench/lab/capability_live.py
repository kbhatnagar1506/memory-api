"""Every capability the memory system claims, against a live model, end to end.

Not LongMemEval and not a retrieval ablation. This ingests a corpus built so
that each capability has a question only that capability can answer, then asks
through the SHIPPING synthesis path -- `service.search` into
`synthesis.chat.answer` -- so what is measured is the product, including the
classifier that routes advice away from the decline contract and the prompts
that were changed this week.

The retrieval-only numbers elsewhere in bench/ carry no judge variance and are
exactly reproducible, which is their value and also their limit: a system can
retrieve perfectly and still answer badly, and the whole argument of the lab
work has been that those two failures need separating. This file deliberately
puts the model back in.

WHAT IS COVERED, and why each needs its own question:

    direct        the fact is in one memory; the floor
    count         a cardinality that exists in no single memory
    order         first/last over dated rows
    date_arith    a span between two events
    compare       judgement over two retrieved figures
    list_all      the table itself is the answer
    temporal      a question scoped to a window ("in March")
    multi         evidence deliberately split across memories
    supersede     a fact that CHANGED; the old value is the trap
    contradict    two memories that disagree and must not be silently picked
    unverified    a memory marked shaky; the caveat must survive
    abstain       the answer is NOT in the store; inventing it is the failure
    preference    explicit / implicit / negative / updated / composed
    tenancy       a fact that exists only in another space and must not leak

The last two are the ones a benchmark will not tell you about and a product
lives or dies on: a memory system that invents an answer is worse than one that
returns nothing, and one that leaks across spaces is a security incident rather
than a quality problem.

    python -m bench.lab.capability_live
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from mapi.config import Settings
from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.domain.synthesis.chat import answer as chat_answer
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

from .scoring import Expected, score
from .stats import wilson

MODEL = "gemini-2.5-pro"
PROJECT = "patchguard-reakon"
MAX_INFLIGHT = 8
ASKED_AT = datetime(2026, 8, 14, tzinfo=UTC)


def _d(y: int, m: int, day: int) -> datetime:
    return datetime(y, m, day, 12, 0, tzinfo=UTC)


#: (occurred_at, content, metadata). One persona, one working year, built so
#: that every capability below has evidence and every trap has bait.
MEMORIES: tuple[tuple[datetime, str, dict], ...] = (
    (
        _d(2026, 1, 12),
        "I set my freelance day rate at 60 pounds an hour when I went independent.",
        {},
    ),
    (
        _d(2026, 1, 20),
        "Signed the first contract with Halden Foods for a packaging refresh.",
        {},
    ),
    (_d(2026, 2, 3), "Bought a Prusa MK4 printer for the workshop, 899 pounds.", {}),
    (
        _d(2026, 2, 17),
        "Shipped the Halden Foods packaging refresh. They paid within a week.",
        {},
    ),
    (
        _d(2026, 3, 4),
        "Started the Bramwell Cycles project, a rear light housing in aluminium.",
        {},
    ),
    (_d(2026, 3, 9), "Flew to Milan for the design fair and came back with three leads.", {}),
    (
        _d(2026, 3, 24),
        "Shipped the Bramwell Cycles light housing. Tooling signed off on the first pass.",
        {},
    ),
    (
        _d(2026, 4, 2),
        "Turned down a vape packaging job. I don't take tobacco or vape work, ever.",
        {},
    ),
    (
        _d(2026, 4, 15),
        "Bought a Bambu X1C printer, 1299 pounds, because the Prusa could not keep up.",
        {},
    ),
    (
        _d(2026, 4, 28),
        "Shipped the Okonjo Studio retail fixture. Third project done this year.",
        {},
    ),
    (
        _d(2026, 5, 6),
        "Raised my day rate from 60 to 85 pounds an hour. Nobody pushed back.",
        {},
    ),
    (
        _d(2026, 5, 19),
        "Quoted studio A at 1400 a month and studio B at 1150 a month for the Peckham space.",
        {},
    ),
    (
        _d(2026, 6, 2),
        "Signed with Vireo Health for a device enclosure. Fourth project of the year.",
        {},
    ),
    (
        _d(2026, 6, 11),
        "I work in metric only. An imperial drawing gets sent back, no exceptions.",
        {},
    ),
    (_d(2026, 6, 23), "Shipped the Vireo Health enclosure two days early.", {}),
    (
        _d(2026, 7, 7),
        "Sarah at Bramwell said their annual tooling budget is around 40000 pounds.",
        {"confidence": "unverified"},
    ),
    (
        _d(2026, 7, 8),
        "Bramwell's finance director put the tooling budget at 52000 pounds for the year.",
        {},
    ),
    (
        _d(2026, 7, 15),
        "Third client meeting I've done at eight in the morning. I do my best work before ten.",
        {},
    ),
    (
        _d(2026, 7, 21),
        "Took the sleeper train to Glasgow for the Vireo review rather than fly.",
        {},
    ),
    (
        _d(2026, 7, 22),
        "Second time this year I've taken the train instead of flying to a client.",
        {},
    ),
    (
        _d(2026, 8, 1),
        "Moved off the Bambu to a resin printer for the fine detail work on medical parts.",
        {},
    ),
)

#: A DIFFERENT space, same org. Nothing here may ever appear in an answer about
#: the space above -- this is the tenancy probe, and it is the one failure in
#: this file that would be a security incident rather than a quality problem.
OTHER_MEMORIES: tuple[tuple[datetime, str], ...] = (
    (_d(2026, 5, 2), "The Caldwell Motors retainer is 4200 pounds a month."),
    (_d(2026, 5, 3), "Caldwell's brand colour is Pantone 3005 C."),
)

#: (capability, question, spec-or-None, note). A None spec means the check is
#: behavioural rather than lexical and is asserted separately below.
CASES: tuple[tuple[str, str, str | None], ...] = (
    ("direct", "What printer did I buy in February?", "prusa"),
    ("direct", "Which client did I sign first?", "halden"),
    ("count", "How many client projects did I ship in 2026?", "four|4"),
    ("order", "Which client did I sign most recently?", "vireo"),
    ("order", "What was the first thing I bought for the workshop?", "prusa"),
    (
        "date_arith",
        "How many days passed between starting and shipping the Bramwell light housing?",
        "20|twenty",
    ),
    (
        "compare",
        "Of the two Peckham studios I quoted, which was cheaper and by how much?",
        "b|1150; 250",
    ),
    ("compare", "Which printer cost me more, the Prusa or the Bambu?", "bambu|x1c"),
    (
        "list_all",
        "List every client I have worked with this year.",
        "halden; bramwell; okonjo; vireo",
    ),
    ("temporal", "What did I do in March?", "bramwell|milan|fair|housing"),
    ("multi", "How much did I spend on 3D printers in total?", "2198|2 198"),
    ("supersede", "What is my current day rate?", "85"),
    ("supersede", "What should I quote a new client for a day's work?", "85"),
    ("contradict", "What is Bramwell's annual tooling budget?", "40000|52000"),
    ("unverified", "What did Sarah tell me about Bramwell's budget?", "40000"),
    ("abstain", "What is my accountant's name?", None),
    ("abstain", "Which university did I study at?", None),
    ("abstain", "What did the Caldwell Motors retainer come to?", None),
    ("tenancy", "What is Caldwell's brand colour?", None),
    (
        "pref_explicit",
        "A drawing package is coming in from a new client. What should I insist on?",
        "metric|mm|millimet + imperial|inch",
    ),
    (
        "pref_implicit",
        "I need to book a recurring slot for deep design work. When should I put it?",
        "morning|mornings|early|before ten|10|nine|8",
    ),
    (
        "pref_negative",
        "A nicotine pouch brand wants a packaging refresh at double my rate. Should I take it?",
        "no|decline|turn|refuse|pass|avoid|reject|not",
    ),
    ("pref_updated", "Quote me for a four-day job for a returning client.", "340|85 + 60|240"),
    (
        "pref_composed",
        "A tobacco-adjacent startup wants a device enclosure, drawings in inches, "
        "meeting at 8am. Which parts of that should I push back on?",
        "tobacco|nicotine|vape|decline|turn|refuse; imperial|inch|metric",
    ),
    (
        "pref_composed",
        "Plan how I should get to a client review in Edinburgh next month.",
        "train|rail|sleeper + fly|flight|plane",
    ),
)


async def main() -> None:
    from bench.harness import build_model_client

    client = build_model_client(MODEL, PROJECT)
    gap = asyncio.Semaphore(MAX_INFLIGHT)

    async def complete(prompt: str) -> str:
        async with gap:
            raw, _tokens = await client.complete(prompt, max_tokens=1500)
        return raw

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="capability-live-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, GeminiEmbedder(dimensions=768), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="CapabilityLive"))
    main_space = await store.create_space(Space(org_id=org.id, slug="cl-main", name="Studio"))
    other_space = await store.create_space(Space(org_id=org.id, slug="cl-other", name="Other"))

    print(f"ingesting {len(MEMORIES)} memories (+{len(OTHER_MEMORIES)} in a second space) ...")
    for occurred, content, meta in MEMORIES:
        await service.ingest(
            org_id=org.id,
            space_id=main_space.id,
            content=content,
            occurred_at=occurred,
            metadata=meta,
            extract=False,
        )
    for occurred, content in OTHER_MEMORIES:
        await service.ingest(
            org_id=org.id,
            space_id=other_space.id,
            content=content,
            occurred_at=occurred,
            extract=False,
        )

    async def ask(capability: str, question: str, spec: str | None) -> dict:
        response = await service.search(
            SearchRequest(
                query=question,
                org_id=org.id,
                space_id=main_space.id,
                limit=10,
                asked_at=ASKED_AT,
            )
        )
        result = await chat_answer(question, response.results, complete)
        reply = result.reply
        low = reply.lower()

        declined = any(
            phrase in low
            for phrase in (
                "don't have",
                "do not have",
                "not in memory",
                "no memory",
                "nothing in",
            )
        )
        # Leakage is checked against the OTHER space's distinctive values, not
        # against a similarity -- either the string is in the reply or it is not.
        leaked = any(token in low for token in ("caldwell", "pantone", "3005", "4200"))
        if spec is None:
            #: abstain/tenancy: the ONLY correct behaviour is to decline, and a
            #: fluent invented answer is the failure this catches.
            correct = declined and not leaked
        else:
            correct = score(reply, Expected.parse(spec)).correct
        return {
            "capability": capability,
            "question": question,
            "correct": correct,
            "declined": declined,
            "leaked": leaked,
            "cited": len(result.cited),
            "used_unverified": result.used_unverified,
            "reply": reply,
        }

    print(f"asking {len(CASES)} questions through the production chat path ({MODEL}) ...\n")
    rows = await asyncio.gather(*(ask(c, q, s) for c, q, s in CASES))

    n = len(rows)
    hits = sum(r["correct"] for r in rows)
    low, high = wilson(hits, n)
    print(f"OVERALL  {hits}/{n}  {hits / n:.1%}   95% CI [{low:.1%}, {high:.1%}]\n")

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["capability"], []).append(r)
    print(f"{'capability':16} {'score':>8}   {'cited':>6}")
    for cap, items in groups.items():
        c = sum(i["correct"] for i in items)
        cited = sum(i["cited"] for i in items)
        mark = "" if c == len(items) else "   <-- "
        print(f"{cap:16} {c:4}/{len(items):<3} {cited:6}{mark}")

    leaks = [r for r in rows if r["leaked"]]
    print(f"\nTENANCY: {len(leaks)} answers leaked another space's content", end="")
    print(" -- CLEAN" if not leaks else f" -- {[r['question'] for r in leaks]}")

    abstain = [r for r in rows if r["capability"] in {"abstain", "tenancy"}]
    refused = sum(r["declined"] for r in abstain)
    print(f"ABSTENTION: declined {refused}/{len(abstain)} unanswerable")

    unver = [r for r in rows if r["capability"] == "unverified"]
    print(f"UNVERIFIED flagged on {sum(r['used_unverified'] for r in unver)}/{len(unver)}")

    print("\nFAILURES:")
    for r in rows:
        if not r["correct"]:
            print(f"\n  [{r['capability']}] {r['question']}")
            print(f"    {r['reply'][:400]}")

    out = Path(__file__).with_name("capability_live_results.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
