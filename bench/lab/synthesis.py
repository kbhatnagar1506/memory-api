"""Synthesis, measured with retrieval switched off.

The width sweep put every memory a persona has into the prompt and the failures
did not go away: composed 23/35, negative 29/35, while explicit, implicit and
updated came in at 35/36, 34/35 and 36/36. Three kinds were retrieval problems.
Two are not, and no amount of better fetching will touch them.

That also means four levers were declined for the wrong reason. Write-time
extraction, multi-query, the domain index and width itself are all retrieval
mechanisms, and they were being scored on a question set whose remaining
failures are synthesis. Worse, the one ATTENTION lever tried -- `COMPOSE_PROMPT`
from composed.py -- was tested at k=6, where the sweep now shows the ranker was
losing eleven points of accuracy. Its null was confounded: it could have been
fixing attention and still lost to the noise underneath it.

So every arm here runs at k=ALL. Retrieval is not a variable, the evidence is
guaranteed present, and any difference between arms is the model's use of it.

WHAT `diagnose` FOUND, and it is why nothing was built from the score alone:
of 20 failures at k=ALL, ELEVEN were the ruler, not the model. Two were a real
scorer bug -- "slip-ons" could never match a spec written "slipons", because
`_normalize` turns a hyphen into a space -- now fixed in scoring.py for every
experiment. The other nine were specs too narrow to accept a correct answer:
"lands during the day" refused by a group listing `daytime|daylight`, "Do not
take on the work" refused by a group listing decline verbs, "a paid local
hand" refused by a group that did not include the corpus's own word. Seven
were genuine, and they split two ways -- VIOLATION (recommending a Saturday to
someone whose Saturdays are pennant days) and OMISSION (ordering the decaf
right and dropping the plant-milk constraint).

WHAT THE ARMS MEASURED, at k=ALL and paired:

    advice    159/177  89.8%      the unstructured instruction
    compose   169/177  95.5%      enumerate constraints, then answer
    verify    169/177  95.5%      answer, then re-read and revise

    verify  vs advice   10-0  p=0.002 *
    compose vs advice   13-3  p=0.021 *
    verify  vs compose   3-3  p=1.000

Structure wins and the second pass does not. `verify` spends a whole extra
model call to arrive at the same score as `compose`, so what helps is being
made to lay the constraints out, not being made to check the draft -- and the
cheaper arm is the one to ship. This is also the first lever in five to
survive its own control.

WHAT SHIPPED, measured as the artifact rather than the mechanism. Production
already ran a TWO-step advice prompt, so it sits between `advice` and
`compose` and cannot inherit the whole gap. Run twice against the proposed
three-step replacement:

    current   161/177  161/177       proposed  169/177  168/177
    paired    12-4  p=0.077          rerun     12-5  p=0.143

Neither run clears p<0.05 on its own, and the k default was declined at
p=0.122, so the difference is worth stating rather than eliding: the effect
REPLICATED at nearly identical numbers, no kind regressed in either run, and
the mechanism behind it is significant twice over in the arms above. That is a
different evidential position from a single unreplicated trend.

    python -m bench.lab.synthesis bench/lab/packs_v150.json diagnose
    python -m bench.lab.synthesis bench/lab/packs_v150.json arms
    python -m bench.lab.synthesis bench/lab/packs_v150.json ship
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

from .corpus import SESSIONS as INFRA
from .preference import ADVICE_PROMPT
from .preference_v2 import T0, gate
from .scoring import Expected, _contains, _normalize, score

MAX_INFLIGHT = 12
#: Every memory a persona has: 13 authored sessions plus 10 unrelated padding.
K_ALL = 23


def missed_groups(answer: str, expected: Expected) -> list[tuple[str, ...]]:
    answer_norm = _normalize(answer)
    return [g for g in expected.must if not any(_contains(answer_norm, s) for s in g)]


def in_context(context: str, group: tuple[str, ...]) -> bool:
    context_norm = _normalize(context)
    return any(_contains(context_norm, s) for s in group)


async def build(packs: list[dict], service: MemoryService, org_id: str) -> dict[str, str]:
    """One space per persona, 13 authored sessions plus the shared padding."""
    spaces: dict[str, str] = {}
    pad = [body.strip() for _sid, _occ, _tags, body in INFRA]
    for i, pack in enumerate(packs):
        space = await service.store.create_space(  # type: ignore[attr-defined]
            Space(org_id=org_id, slug=f"sx-{i}", name=str(pack["persona"])[:40])
        )
        spaces[pack["persona"]] = space.id
        for s in pack["sessions"]:
            await service.ingest(
                org_id=org_id,
                space_id=space.id,
                content=s["text"],
                occurred_at=T0 + timedelta(weeks=s["week"]),
                metadata={"session_id": s["id"]},
                extract=False,
            )
        for j, body in enumerate(pad):
            await service.ingest(
                org_id=org_id,
                space_id=space.id,
                content=body,
                occurred_at=T0 + timedelta(weeks=21 + j),
                metadata={"session_id": f"pad-{j}"},
                extract=False,
            )
    return spaces


def make_service() -> tuple[MemoryService, InMemoryStore]:
    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="synthesis-lab-pepper",
    )
    store = InMemoryStore()
    return (
        MemoryService(store, GeminiEmbedder(dimensions=768), HeuristicReranker(), settings),
        store,
    )


async def diagnose(packs: list[dict]) -> None:
    """Dump every failure at k=ALL, with the answer, so it can be READ.

    Nothing is fixed from a score. The 12 composed and 6 negative failures could
    be attention, could be the model overriding a constraint it named, or could
    be the spec refusing a correct answer -- and those need three different
    responses. Building a lever before reading them is how the first scorer
    produced two rounds of fake findings.
    """
    from bench.harness import build_model_client

    questions, _dropped = gate(packs)
    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    gap = asyncio.Semaphore(MAX_INFLIGHT)
    service, store = make_service()
    org = await store.create_organization(Organization(name="SynthLab"))
    spaces = await build(packs, service, org.id)
    asked = datetime(2026, 8, 13, tzinfo=UTC)

    async def run(q: dict) -> dict:
        async with gap:
            response = await service.search(
                SearchRequest(
                    query=q["question"],
                    org_id=org.id,
                    space_id=spaces[q["pack"]],
                    limit=K_ALL,
                    asked_at=asked,
                )
            )
            ordered = sorted(response.results, key=lambda h: h.memory.occurred_at)
            context = "\n\n---\n\n".join(
                f"[{h.memory.occurred_at:%Y-%m-%d}]\n{h.memory.content}" for h in ordered
            )
            raw, _tokens = await client.complete(
                ADVICE_PROMPT.format(context=context, question=q["question"]),
                max_tokens=1024,
            )
        answer = raw.strip()
        expected = Expected.parse(q["spec"])
        missed = missed_groups(answer, expected)
        return {
            "kind": q["kind"],
            "question": q["question"],
            "spec": q["spec"],
            "correct": score(answer, expected).correct,
            "answer": answer,
            "missed": ["|".join(g) for g in missed],
            #: With k=ALL this should be True for every missed group. A False
            #: means the corpus never contained the evidence -- an authoring
            #: bug, not a synthesis failure, and it must not be counted as one.
            "evidence_present": [in_context(context, g) for g in missed],
            "n_memories": len(response.results),
        }

    rows = await asyncio.gather(*(run(q) for q in questions))
    failures = [r for r in rows if not r["correct"]]
    print(
        f"k=ALL: {sum(r['correct'] for r in rows)}/{len(rows)} correct, {len(failures)} failed"
    )

    orphan = [r for r in failures if not all(r["evidence_present"])]
    print(f"failures whose evidence is NOT in the corpus (authoring bugs): {len(orphan)}")
    for r in orphan:
        print(f"  [{r['kind']}] {r['question'][:60]}  missing: {r['missed']}")

    print("\n" + "=" * 96)
    for r in failures:
        if r["kind"] not in {"composed", "negative"}:
            continue
        print(f"\n[{r['kind']}] {r['question']}")
        print(f"  spec:   {r['spec']}")
        print(f"  missed: {r['missed']}  evidence_in_context={r['evidence_present']}")
        print(f"  ANSWER: {r['answer'][:700]}")

    out = Path(__file__).with_name("synthesis_diagnosis.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out.name}")


#: The candidate, and what the failures actually looked like.
#:
#: Seven genuine synthesis failures survived at k=ALL once the scorer's
#: compound bug was fixed, and they are two shapes of one defect:
#:
#:   VIOLATION   "ask for a Saturday afternoon" -- to a woman whose diary says
#:               Saturdays are spoken for by pennant from September to March.
#:               "recommend a Tiramisu" -- creamy, to someone avoiding dairy.
#:   OMISSION    decaf ordered correctly and the plant-milk constraint dropped;
#:               the train to Zurich chosen and the cycling preference dropped.
#:
#: `COMPOSE_PROMPT` already enumerates constraints BEFORE recommending, and the
#: Saturday answer is the proof that enumeration is not enough: listing "no
#: Saturdays" does not stop you writing Saturday two lines later. Nothing
#: re-reads the recommendation.
#:
#: So this pass gets the draft and is asked to falsify it. Checking an answer
#: against a constraint is a different and easier task than producing one that
#: satisfies every constraint at once, which is the same reason the review of a
#: patch catches what writing it did not.
VERIFY_PROMPT = """\
You are checking a draft recommendation against what is known about this user.

Excerpts from your history with them, each prefixed with its date:
{context}

Their request: {question}

The draft reply:
{draft}

Do this:
1. List every preference, habit or constraint in the excerpts that bears on \
this request -- things they avoid count as much as things they like. If a \
preference changed over time, use only the LATEST.
2. For each one, mark the draft SATISFIED, VIOLATED or IGNORED. Be strict: a \
draft that never mentions a constraint has IGNORED it, and one that \
recommends against it has VIOLATED it.
3. Write the final recommendation. If everything was satisfied, restate the \
draft's recommendation. Otherwise correct it so that every preference holds at \
once, and say which you changed.

End your reply with the final recommendation in full. Never reply NO_ANSWER."""


async def arms(packs: list[dict]) -> None:
    """Three arms at k=ALL: no structure, structure before, structure after.

    `compose` is the CONTROL that matters. `verify` spends a second model call,
    so a win over `advice` alone could just be more reasoning tokens on the
    problem -- `compose` buys that reasoning in ONE call, before the answer. If
    verify beats compose, the mechanism is re-reading the OUTPUT; if they tie,
    it is structure, and the cheaper arm should win on cost alone.
    """
    from bench.harness import build_model_client

    from .composed import COMPOSE_PROMPT
    from .stats import fmt_p, sign_test, wilson

    questions, _dropped = gate(packs)
    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    gap = asyncio.Semaphore(MAX_INFLIGHT)
    service, store = make_service()
    org = await store.create_organization(Organization(name="SynthArms"))
    spaces = await build(packs, service, org.id)
    asked = datetime(2026, 8, 13, tzinfo=UTC)

    async def context_for(q: dict) -> str:
        response = await service.search(
            SearchRequest(
                query=q["question"],
                org_id=org.id,
                space_id=spaces[q["pack"]],
                limit=K_ALL,
                asked_at=asked,
            )
        )
        ordered = sorted(response.results, key=lambda h: h.memory.occurred_at)
        return "\n\n---\n\n".join(
            f"[{h.memory.occurred_at:%Y-%m-%d}]\n{h.memory.content}" for h in ordered
        )

    async def run(q: dict, arm: str) -> bool:
        async with gap:
            context = await context_for(q)
            template = COMPOSE_PROMPT if arm == "compose" else ADVICE_PROMPT
            raw, _tokens = await client.complete(
                template.format(context=context, question=q["question"]), max_tokens=1500
            )
            answer = raw.strip()
            if arm == "verify":
                checked, _t = await client.complete(
                    VERIFY_PROMPT.format(context=context, question=q["question"], draft=answer),
                    max_tokens=1500,
                )
                answer = checked.strip()
        return score(answer, Expected.parse(q["spec"])).correct

    names = ("advice", "compose", "verify")
    verdicts: dict[str, list[bool]] = {}
    for arm in names:
        print(f"running {arm} over {len(questions)} questions ...")
        verdicts[arm] = list(await asyncio.gather(*(run(q, arm) for q in questions)))

    n = len(questions)
    print(f"\n{'arm':10} {'correct':>10} {'acc':>7}   {'95% CI':^14}")
    for arm in names:
        hits = sum(verdicts[arm])
        low, high = wilson(hits, n)
        print(f"{arm:10} {hits:6}/{n:<3} {hits / n:6.1%}   [{low:5.1%}, {high:5.1%}]")

    def paired(a: str, b: str, idx: list[int] | None = None) -> str:
        rows = idx if idx is not None else list(range(n))
        x = sum(1 for i in rows if verdicts[a][i] and not verdicts[b][i])
        y = sum(1 for i in rows if verdicts[b][i] and not verdicts[a][i])
        return f"{x:2}-{y:<2} {fmt_p(sign_test(x, y))}"

    print("\nPAIRED SIGN TESTS (exact McNemar; * marks p<0.05)")
    for a, b in (("verify", "advice"), ("compose", "advice"), ("verify", "compose")):
        print(f"  {a:8} vs {b:8}  {paired(a, b)}")

    kinds = sorted({q["kind"] for q in questions})
    print(f"\n{'BY KIND':12}" + "".join(f"{a:>10}" for a in names) + f"{'ver-vs-adv':>16}")
    for kind in kinds:
        idx = [i for i, q in enumerate(questions) if q["kind"] == kind]
        row = f"{kind:12}" + "".join(
            f"{sum(verdicts[a][i] for i in idx):>6}/{len(idx):<3}" for a in names
        )
        print(row + f"{paired('verify', 'advice', idx):>16}")

    out = Path(__file__).with_name("synthesis_arms_results.json")
    out.write_text(
        json.dumps(
            {
                "n": n,
                "verdicts": {a: verdicts[a] for a in names},
                "kinds": [q["kind"] for q in questions],
                "questions": [q["question"] for q in questions],
            },
            indent=1,
        )
    )
    print(f"\nwrote {out.name}")


#: The proposed production template, and why it is not just COMPOSE_PROMPT.
#:
#: `arms` measured the lab's unstructured `ADVICE_PROMPT` against a three-step
#: structure and got 10-0. Production does NOT run the unstructured prompt --
#: `ADVICE_CHAT_PROMPT` is already two-step (name the preferences and cite them,
#: then recommend), so it sits between the two arms and cannot be assumed to
#: inherit the whole gap. What ships has to be measured as what ships.
#:
#: Three elements move across, each traceable to a failure that was read:
#:
#:   EVERY, however many      the omissions -- decaf ordered and the plant-milk
#:                            constraint dropped. "the preference(s)" invites
#:                            one; "every, however many there are" does not.
#:   avoidances count         the negative kind, 30/35 -> 34/35 in the lab. A
#:                            prohibition is not a taste and the model treats it
#:                            as lower-priority unless told they rank equally.
#:   CHECK, then ALL at once  the violations -- "ask for a Saturday afternoon"
#:                            to someone whose Saturdays are pennant days.
#:                            Naming a constraint does not stop you breaking it
#:                            two lines later; checking the recommendation
#:                            against each one does.
#:
#: Everything production already guarantees is kept: numbered citations, the
#: "no stored preference applies" branch, the UNVERIFIED caveat, brevity.
PROPOSED_ADVICE_CHAT_PROMPT = """\
You are advising this user from a memory store. The numbered memories below \
are what you know about them; they will NOT contain a ready-made answer -- \
they contain what this user likes, avoids, owns and does.

Work in three steps, all in your reply:
1. PREFERENCES: list EVERY preference in the memories that bears on this \
request, one per line, however many there are -- what they AVOID counts as \
much as what they like. Quote the memory and cite it as [1], [2] inline. If a \
preference CHANGED over time, list only the latest. If NO memory bears on the \
request, say "no stored preference applies here" instead.
2. CHECK: for each one, state in a few words what the recommendation must do \
to satisfy it.
3. RECOMMENDATION: one concrete recommendation that satisfies ALL of them at \
once, following from exactly the preferences you cited -- never from general \
taste. If two genuinely conflict, say which you traded off and why. If none \
applied, still recommend, and say plainly that it is a guess.

A memory marked UNVERIFIED is not established fact; say so if you lean on it. \
Keep every step brief.

{history}Memories:
{memories}

Request: {question}
Answer:"""


async def ship(packs: list[dict]) -> None:
    """The production template as it stands, against the proposed replacement.

    Rendered through `format_memories`, the real function, so the numbering and
    the UNVERIFIED labelling are what production actually sends -- a prompt
    measured on a different context format is a prompt measured somewhere else.
    """
    from bench.harness import build_model_client
    from mapi.domain.models import ScoredMemory
    from mapi.domain.synthesis.chat import ADVICE_CHAT_PROMPT, format_memories

    from .stats import fmt_p, sign_test, wilson

    questions, _dropped = gate(packs)
    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    gap = asyncio.Semaphore(MAX_INFLIGHT)
    service, store = make_service()
    org = await store.create_organization(Organization(name="ShipArms"))
    spaces = await build(packs, service, org.id)
    asked = datetime(2026, 8, 13, tzinfo=UTC)

    templates = {"current": ADVICE_CHAT_PROMPT, "proposed": PROPOSED_ADVICE_CHAT_PROMPT}

    async def run(q: dict, arm: str) -> bool:
        async with gap:
            response = await service.search(
                SearchRequest(
                    query=q["question"],
                    org_id=org.id,
                    space_id=spaces[q["pack"]],
                    limit=K_ALL,
                    asked_at=asked,
                )
            )
            hits: list[ScoredMemory] = sorted(
                response.results, key=lambda h: h.memory.occurred_at
            )
            raw, _tokens = await client.complete(
                templates[arm].format(
                    history="", memories=format_memories(hits), question=q["question"]
                ),
                max_tokens=1500,
            )
        return score(raw.strip(), Expected.parse(q["spec"])).correct

    verdicts: dict[str, list[bool]] = {}
    for arm in templates:
        print(f"running {arm} over {len(questions)} questions ...")
        verdicts[arm] = list(await asyncio.gather(*(run(q, arm) for q in questions)))

    n = len(questions)
    print(f"\n{'template':10} {'correct':>10} {'acc':>7}   {'95% CI':^14}")
    for arm in templates:
        hits_n = sum(verdicts[arm])
        low, high = wilson(hits_n, n)
        print(f"{arm:10} {hits_n:6}/{n:<3} {hits_n / n:6.1%}   [{low:5.1%}, {high:5.1%}]")

    def paired(idx: list[int] | None = None) -> str:
        rows = idx if idx is not None else list(range(n))
        x = sum(1 for i in rows if verdicts["proposed"][i] and not verdicts["current"][i])
        y = sum(1 for i in rows if verdicts["current"][i] and not verdicts["proposed"][i])
        return f"{x:2}-{y:<2} {fmt_p(sign_test(x, y))}"

    print(f"\nPAIRED  proposed vs current:  {paired()}")
    print(f"\n{'BY KIND':12}{'current':>10}{'proposed':>10}{'paired':>16}")
    for kind in sorted({q["kind"] for q in questions}):
        idx = [i for i, q in enumerate(questions) if q["kind"] == kind]
        cur = sum(verdicts["current"][i] for i in idx)
        pro = sum(verdicts["proposed"][i] for i in idx)
        print(f"{kind:12}{cur:>6}/{len(idx):<3}{pro:>6}/{len(idx):<3}{paired(idx):>16}")

    out = Path(__file__).with_name("synthesis_ship_results.json")
    out.write_text(
        json.dumps(
            {"n": n, "verdicts": verdicts, "kinds": [q["kind"] for q in questions]}, indent=1
        )
    )
    print(f"\nwrote {out.name}")


_MODES = {"diagnose": diagnose, "arms": arms, "ship": ship}

if __name__ == "__main__":
    _packs = json.loads(Path(sys.argv[1]).read_text())["packs"]
    _mode = sys.argv[2] if len(sys.argv) > 2 else "diagnose"
    sys.exit(asyncio.run(_MODES[_mode](_packs)))
