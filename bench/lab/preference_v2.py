"""Preference lab v2: two personas, forty questions, paired-arm statistics.

Round one's n=12 could not distinguish the advice arms -- a one-question gap IS
the noise at that size. This round scales to n=40 across two authored personas,
and scores the arms as PAIRED comparisons: same question, same evidence,
different instruction, count the disagreements. Forty paired trials resolve a
real one-directional difference that forty independent ones cannot.

The corpus is machine-authored and machine-verified (two adversarial lenses:
coherence and scoring-validity), then gated HERE in code, because mechanical
checks belong to code and semantic ones to reviewers:

    G1  pack shape: 13 sessions, 20 questions, 4 per kind
    G2  every evidence_quote is a verbatim substring of exactly one session
    G3  every spec parses
    G4  no leak: no required surface appears whole-word in its own question
    G5  session weeks strictly increase

A question failing a gate is DROPPED and reported, never silently kept -- a bad
case that stays produces a fake finding, which is the lesson the first scorer
taught twice.

Each persona lives in its OWN space, because two people's preferences in one
retrieval pool is a tenancy bug wearing a corpus's clothes.

    python -m bench.lab.preference_v2 packs.json     # ~120 flash calls
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

from .preference import ADVICE_PROMPT, FACT_PROMPT, GROUNDED_PROMPT
from .scoring import Expected, _normalize, score

ARMS = {"fact": FACT_PROMPT, "advice": ADVICE_PROMPT, "grounded": GROUNDED_PROMPT}
T0 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def gate(packs: list[dict]) -> tuple[list[dict], list[str]]:
    """Apply G1-G5. Returns (usable questions with pack refs, drop reasons)."""
    usable: list[dict] = []
    dropped: list[str] = []
    for pack in packs:
        sessions = pack["sessions"]
        label = pack.get("persona", "?")[:12]
        weeks = [s["week"] for s in sessions]
        if sorted(weeks) != weeks or len(set(weeks)) != len(weeks):
            dropped.append(f"[{label}] G5: weeks not strictly increasing; keeping order as-is")
        if len(sessions) != 13:
            dropped.append(f"[{label}] G1: {len(sessions)} sessions (expected 13)")
        for q in pack["questions"]:
            tag = f"[{label}] {q['question'][:44]}"
            try:
                expected = Expected.parse(q["spec"])
            except ValueError as exc:
                dropped.append(f"{tag} G3: {exc}")
                continue
            holders = [s["id"] for s in sessions if q["evidence_quote"] in s["text"]]
            if len(holders) != 1:
                dropped.append(f"{tag} G2: evidence_quote in {len(holders)} sessions")
                continue
            question_norm = _normalize(q["question"])
            leaks = [
                surface
                for group in expected.must
                for surface in group
                if _normalize(surface) and _normalize(surface) in question_norm
            ]
            if leaks:
                dropped.append(f"{tag} G4: answer surfaces in question: {leaks}")
                continue
            usable.append({**q, "pack": pack["persona"], "evidence_session": holders[0]})
    return usable, dropped


async def main(packs: list[dict]) -> None:
    from bench.harness import build_model_client

    questions, dropped = gate(packs)
    kinds_count: dict[str, int] = {}
    for q in questions:
        kinds_count[q["kind"]] = kinds_count.get(q["kind"], 0) + 1
    print(f"packs: {len(packs)}   usable questions: {len(questions)}   dropped: {len(dropped)}")
    for reason in dropped:
        print(f"  drop: {reason}")
    print(f"  by kind: {kinds_count}")

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="preference-v2-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, GeminiEmbedder(dimensions=768), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="PrefV2"))
    spaces: dict[str, str] = {}
    session_owner: dict[str, dict[str, str]] = {}
    for i, pack in enumerate(packs):
        space = await store.create_space(
            Space(org_id=org.id, slug=f"persona-{i}", name=str(pack["persona"])[:40])
        )
        spaces[pack["persona"]] = space.id
        owner: dict[str, str] = {}
        for s in pack["sessions"]:
            result = await service.ingest(
                org_id=org.id,
                space_id=space.id,
                content=s["text"],
                occurred_at=T0 + timedelta(weeks=s["week"]),
                metadata={"session_id": s["id"]},
                extract=False,
            )
            owner[result.memory.id] = s["id"]
        session_owner[pack["persona"]] = owner

    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    asked = datetime(2026, 8, 13, tzinfo=UTC)

    # Retrieval once per question; every arm reads identical evidence.
    contexts: dict[str, str] = {}
    evidence_rank: dict[str, int | None] = {}
    for q in questions:
        space_id = spaces[q["pack"]]
        response = await service.search(
            SearchRequest(
                query=q["question"], org_id=org.id, space_id=space_id, limit=6, asked_at=asked
            )
        )
        owner = session_owner[q["pack"]]
        returned = [owner.get(h.memory.id) for h in response.results]
        evidence_rank[q["question"]] = (
            returned.index(q["evidence_session"]) + 1
            if q["evidence_session"] in returned
            else None
        )
        ordered = sorted(response.results, key=lambda h: h.memory.occurred_at)
        contexts[q["question"]] = "\n\n---\n\n".join(
            f"[{h.memory.occurred_at:%Y-%m-%d}]\n{h.memory.content}" for h in ordered
        )

    retrieved = sum(1 for r in evidence_rank.values() if r is not None)
    print(
        f"\nretrieval: evidence in window for {retrieved}/{len(questions)}   "
        f"rank<=4 for {sum(1 for r in evidence_rank.values() if r and r <= 4)}"
    )

    async def ask(arm: str, q: dict) -> tuple[str, str, bool, float]:
        raw, _tokens = await client.complete(
            ARMS[arm].format(context=contexts[q["question"]], question=q["question"]),
            max_tokens=1024,
        )
        graded = score(raw.strip(), Expected.parse(q["spec"]))
        return arm, q["question"], graded.correct, graded.completeness

    rows = await asyncio.gather(*(ask(arm, q) for arm in ARMS for q in questions))
    verdict: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    naming: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for arm, question, correct, completeness in rows:
        verdict[arm][question] = correct
        naming[arm].append(completeness)

    n = len(questions)
    print(f"\n{'ARM':10} {'correct':>10} {'completeness':>13}")
    for arm in ARMS:
        c = sum(verdict[arm].values())
        print(f"{arm:10} {c:6}/{n:<3} {sum(naming[arm]) / n:12.2f}")

    print("\nBY KIND:")
    kinds = sorted(kinds_count)
    print(f"  {'kind':10}" + "".join(f"{arm:>10}" for arm in ARMS) + f"{'n':>5}")
    for kind in kinds:
        subset = [q["question"] for q in questions if q["kind"] == kind]
        row = f"  {kind:10}"
        for arm in ARMS:
            row += f"{sum(verdict[arm][q] for q in subset):>10}"
        print(row + f"{len(subset):>5}")

    def paired(a: str, b: str) -> None:
        only_a = [q for q in verdict[a] if verdict[a][q] and not verdict[b][q]]
        only_b = [q for q in verdict[b] if verdict[b][q] and not verdict[a][q]]
        print(f"\nPAIRED {a} vs {b}: {a}-only wins {len(only_a)}, {b}-only wins {len(only_b)}")
        for q in only_a[:6]:
            print(f"  {a} only: {q[:64]}")
        for q in only_b[:6]:
            print(f"  {b} only: {q[:64]}")

    paired("advice", "fact")
    paired("grounded", "advice")

    out = Path(__file__).with_name("preference_v2_results.json")
    out.write_text(
        json.dumps({"verdict": verdict, "evidence_rank": evidence_rank, "n": n}, indent=1)
    )
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    _packs = json.loads(Path(sys.argv[1]).read_text())["packs"]
    sys.exit(asyncio.run(main(_packs)))
