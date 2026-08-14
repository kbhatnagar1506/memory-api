"""Composed preferences: the one kind no prompt fixed, diagnosed then attacked.

Round two's kind table, over 37 questions and three instruction arms:

    explicit   8/8   8/8   8/8      stated preferences survive any prompt
    implicit   2/7   6/7   6/7      the advice prompt fixed these
    negative   2/7   6/7   6/7      and these
    updated    3/8   7/8   7/8      and these
    composed   2/7   4/7   3/7      <- nothing fixed these

A composed question needs TWO preferences, from two different sessions, to
BOTH shape one answer ("plan my birthday dinner" = vegetarian AND no
shellfish). It is the multi-evidence synthesis wall from LongMemEval -- where
accuracy fell ~12pp per additional evidence session while retrieval stayed
flat -- reproduced on a corpus we own and can iterate on for cents.

DIAGNOSE BEFORE FIXING, which is the discipline that has paid every time. A
composed failure has three possible stages and they need different fixes:

    RETRIEVAL     the second preference never reached the window
    ATTENTION     both were in the window; the answer used one and ignored the
                  other
    SCORING       the answer honoured both and the spec failed to say so

`score()` reports per-group hits, so the missed dimension is nameable; and a
missed group is cross-checked against the retrieved context, which separates
"never arrived" from "arrived and was ignored".

THEN THE LEVERS, tested only where the diagnosis points:

    compose   an instruction that enumerates before it recommends: list EVERY
              preference bearing on the request, then reconcile them. Targets
              ATTENTION.
    multi     one retrieval per preference dimension the question implies,
              unioned. Targets RETRIEVAL.

    python -m bench.lab.composed bench/lab/packs_v2.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mapi.config import Settings
from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

from .preference import ADVICE_PROMPT
from .preference_v2 import T0, gate
from .scoring import Expected, _contains, _normalize, score

#: The candidate. `ADVICE_PROMPT` says "make a concrete recommendation that
#: reflects those preferences" -- singular in spirit, and the model obliges by
#: finding one. This says: enumerate first, then reconcile, and name any
#: preference you had to trade off. Enumeration is the cheapest known fix for
#: "the model used the first thing it found", and it is the same shape as the
#: derive path's map-then-reduce.
COMPOSE_PROMPT = """\
You are advising a user, drawing on excerpts from your history with them. Each \
excerpt is prefixed with its date.

Excerpts:
{context}

Request: {question}

Work in three steps, all in your reply:
1. RELEVANT PREFERENCES: list EVERY preference in the excerpts that bears on \
this request, one per line, however many there are -- constraints they avoid \
count as much as things they like. If a preference changed over time, list \
only the latest.
2. CHECK: for each one, state in a few words how the recommendation must \
satisfy it.
3. RECOMMENDATION: one concrete recommendation satisfying ALL of them at once. \
If two conflict, say which you traded off and why.

Never reply NO_ANSWER. Keep every step brief."""


def _missed_groups(answer: str, expected: Expected) -> list[tuple[str, ...]]:
    """Which must-groups the answer failed to satisfy."""
    answer_norm = _normalize(answer)
    return [
        group
        for group in expected.must
        if not any(_contains(answer_norm, surface) for surface in group)
    ]


def _supported_by(context: str, group: tuple[str, ...]) -> bool:
    """Whether the retrieved context contains evidence for this group.

    Loose on purpose: a session says "Flat or rolling trails" and the spec says
    `flat|rolling|gentle`. If ANY surface of the group appears in the context,
    the evidence was there and a missing answer is an attention failure rather
    than a retrieval one. False negatives here bias toward calling a failure
    RETRIEVAL, which is the conservative direction -- it blames the system we
    can fix rather than the model we cannot.
    """
    context_norm = _normalize(context)
    return any(_contains(context_norm, surface) for surface in group)


async def main(packs: list[dict]) -> None:
    from bench.harness import build_model_client

    questions, _dropped = gate(packs)
    composed = [q for q in questions if q["kind"] == "composed"]
    print(f"composed questions: {len(composed)}")

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="composed-lab-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, GeminiEmbedder(dimensions=768), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="ComposedLab"))
    spaces: dict[str, str] = {}
    for i, pack in enumerate(packs):
        space = await store.create_space(
            Space(org_id=org.id, slug=f"cx-{i}", name=str(pack["persona"])[:40])
        )
        spaces[pack["persona"]] = space.id
        for s in pack["sessions"]:
            await service.ingest(
                org_id=org.id,
                space_id=space.id,
                content=s["text"],
                occurred_at=T0 + timedelta(weeks=s["week"]),
                metadata={"session_id": s["id"]},
                extract=False,
            )

    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    asked = datetime(2026, 8, 13, tzinfo=UTC)

    async def context_for(q: dict, *, multi: bool, limit: int = 6) -> str:
        """One search, or one per preference dimension unioned.

        The multi arm derives its extra queries from the question itself, not
        from the spec -- deriving them from the answer key would be reading the
        gold at retrieval time, which is the classic way to build a lever that
        works in the lab and nowhere else.
        """
        space_id = spaces[q["pack"]]
        queries = [q["question"]]
        if multi:
            probe, _tokens = await client.complete(
                "A user asked an assistant for a recommendation. List the 2-3 "
                "DISTINCT aspects of this person's tastes, habits or "
                "constraints an assistant would need to look up to answer well. "
                "One short search phrase per line, no numbering, no other text."
                f"\n\nRequest: {q['question']}",
                max_tokens=256,
            )
            queries += [line.strip(" -*\t") for line in probe.splitlines() if line.strip()][:3]

        seen: dict[str, object] = {}
        for query in queries:
            response = await service.search(
                SearchRequest(
                    query=query, org_id=org.id, space_id=space_id, limit=limit, asked_at=asked
                )
            )
            for hit in response.results:
                seen.setdefault(hit.memory.id, hit.memory)
        ordered = sorted(seen.values(), key=lambda m: m.occurred_at)  # type: ignore[attr-defined]
        return "\n\n---\n\n".join(
            f"[{m.occurred_at:%Y-%m-%d}]\n{m.content}"
            for m in ordered  # type: ignore[attr-defined]
        )

    #: (template, multi-query, k). `advice-wide` is the CONTROL: multi-query
    #: puts more memories in context (8.4 vs 6.0), so a win could be targeting
    #: OR width. This arm buys the width with a plain k=9 single search. If it
    #: matches multi, the extra model call is paying for nothing.
    arms = {
        "advice": (ADVICE_PROMPT, False, 6),
        "advice-wide": (ADVICE_PROMPT, False, 9),
        "compose": (COMPOSE_PROMPT, False, 6),
        "compose+multi": (COMPOSE_PROMPT, True, 6),
    }
    results: dict[str, dict[str, dict]] = {arm: {} for arm in arms}

    for arm, (template, multi, limit) in arms.items():
        for q in composed:
            context = await context_for(q, multi=multi, limit=limit)
            raw, _tokens = await client.complete(
                template.format(context=context, question=q["question"]), max_tokens=1500
            )
            answer = raw.strip()
            expected = Expected.parse(q["spec"])
            graded = score(answer, expected)
            missed = _missed_groups(answer, expected)
            stage = "-"
            if not graded.correct:
                unsupported = [g for g in missed if not _supported_by(context, g)]
                stage = "RETRIEVAL" if unsupported else "ATTENTION"
            results[arm][q["question"]] = {
                "correct": graded.correct,
                "stage": stage,
                "missed": ["|".join(g) for g in missed],
                "answer": answer,
                "n_memories": context.count("---") + 1,
            }

    print(f"\n{'ARM':16} {'correct':>9}   failure stages")
    for arm in arms:
        rows = results[arm]
        correct = sum(r["correct"] for r in rows.values())
        stages: dict[str, int] = {}
        for r in rows.values():
            if r["stage"] != "-":
                stages[r["stage"]] = stages.get(r["stage"], 0) + 1
        mem = sum(r["n_memories"] for r in rows.values()) / len(rows)
        label = ", ".join(f"{k}:{v}" for k, v in sorted(stages.items())) or "none"
        print(
            f"{arm:16} {correct:5}/{len(composed):<3} {label:>22}   "
            f"(avg {mem:.1f} memories in context)"
        )

    print("\nPER QUESTION:")
    for q in composed:
        marks = "  ".join(
            f"{arm}={'.' if results[arm][q['question']]['correct'] else 'X'}" for arm in arms
        )
        print(f"  {q['question'][:56]:58} {marks}")
        for arm in arms:
            row = results[arm][q["question"]]
            if not row["correct"]:
                print(f"      {arm:14} {row['stage']:10} missed: {row['missed']}")

    out = Path(__file__).with_name("composed_results.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    _packs = json.loads(Path(sys.argv[1]).read_text())["packs"]
    sys.exit(asyncio.run(main(_packs)))
