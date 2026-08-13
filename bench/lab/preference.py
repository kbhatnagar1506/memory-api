"""The preference lab: does the model APPLY a preference, or only recall it?

Preference is the weakest measured capability -- 0.600 on LongMemEval's 30
questions, against 0.93+ for plain lookups -- and its retrieval was PERFECT
(1.000) while it failed. Pure synthesis. The format experiment then showed
context layout is a null variable. What is left to vary is the thing neither of
those touched: the INSTRUCTION -- how the model is told to use what it sees.

The known mechanism, from the bench history: an advice request under a
fact-lookup prompt sends the model hunting for a stored answer that does not
exist, and it returns NO_ANSWER on a question whose entire point is to apply
preferences. That was measured once, on LME, through a classifier. This corpus
lets us measure it directly, per preference TYPE, without LME.

THE CORPUS covers the shapes a real preference takes, because "preference" is
not one capability:

    explicit     stated outright ("I always book the aisle")
    implicit     never stated, visible only as a behaviour pattern
    negative     a standing prohibition ("never before 10am")
    updated      the preference CHANGED, and applying the old one is a failure
    composed     two preferences that must BOTH shape one recommendation

THE ARMS vary only the instruction; retrieval and evidence are identical:

    fact         the lookup prompt. Expected to fail -- the measured baseline.
    advice       the personalisation prompt: use what they like, never refuse.
    grounded     advice + the innovation candidate: QUOTE the preference you
                 are applying before recommending. Hypothesis: forcing the
                 model to bind its recommendation to a named preference stops
                 it recommending from its own priors -- the same mechanism
                 that makes chat citations work.

    python -m bench.lab.preference        # ~45 flash calls
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mapi.config import Settings
from mapi.domain.embeddings.deterministic import DeterministicEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

from .scoring import Expected, score

T0 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def _when(week: int) -> datetime:
    return T0 + timedelta(weeks=week)


#: (session_id, when, text). Preferences are embedded in ordinary diary prose --
#: not labelled as preferences -- because that is what real memory looks like and
#: extraction-from-context is part of what is being measured.
SESSIONS: list[tuple[str, datetime, str]] = [
    (
        "p01",
        _when(0),
        "Booked the Denver flight. Aisle seat as always -- I can't stand climbing over people. Paid extra to avoid the red-eye; I'm useless the whole next day after one.",
    ),
    (
        "p02",
        _when(1),
        "Dinner with the team went fine, though I had to swap plates -- the shellfish allergy strikes again. The gnocchi was excellent. Sparkling water, since I don't drink.",
    ),
    (
        "p03",
        _when(2),
        "Returned the second mechanical keyboard this month. The clicking drives me up the wall in calls. Back to the quiet low-profile one.",
    ),
    (
        "p04",
        _when(3),
        "Blocked my calendar before 10am permanently. Third time someone booked me at 8:30; mornings are for deep work, not meetings.",
    ),
    (
        "p05",
        _when(5),
        "Weekend hike was great. Note to self: my knees hate descents over 500m now. Flat or rolling trails from here on.",
    ),
    (
        "p06",
        _when(7),
        "Finished another sci-fi novel on the commute -- that's four this quarter. Tried a thriller, gave up after a chapter.",
    ),
    (
        "p07",
        _when(9),
        "I used to insist on window seats for the view, but honestly since the knee thing the aisle is non-negotiable now.",
    ),
    (
        "p08",
        _when(11),
        "Set the budget rule with Aaditi: gifts between us cap at 75 dollars, experiences over things.",
    ),
    (
        "p09",
        _when(13),
        "Vegetarian February went so well I'm keeping it. No meat since the 1st, and I feel better for it. Fish was never really my thing anyway.",
    ),
    (
        "p10",
        _when(15),
        "The Lisbon hotel had a rooftop pool, which sold me instantly. I will pick a hotel with a pool over a fancier one without, every time.",
    ),
]

#: (kind, question, expected spec). Bonus tokens after `+` measure whether the
#: recommendation NAMES the preference it applied, which is the grounded arm's
#: whole hypothesis.
QUESTIONS: list[tuple[str, str, str]] = [
    ("explicit", "Book me a seat for the six-hour flight to Seattle -- which seat?", "aisle"),
    ("explicit", "Can you recommend a hotel for my Barcelona trip?", "pool"),
    (
        "negative",
        "When should I schedule my weekly 1:1 with Priya?",
        "10|afternoon|later + morning",
    ),
    (
        "negative",
        "Pick a flight to New York for me: 6am red-eye arrival or midday departure?",
        "midday|noon + red-eye",
    ),
    (
        "implicit",
        "Recommend a keyboard for my new desk setup.",
        "quiet|silent|low-profile|low profile + click",
    ),
    ("implicit", "Suggest a book for my flight.", "sci-fi|science fiction"),
    ("updated", "Window or aisle for the long-haul to Tokyo?", "aisle + knee"),
    (
        "updated",
        "What should I order at the steakhouse tomorrow?",
        "vegetarian|vegetable|meatless|salad|pasta + meat",
    ),
    (
        "composed",
        "Recommend a birthday gift for Aaditi.",
        "experience|class|tickets|dinner|show + 75",
    ),
    (
        "composed",
        "Plan a restaurant for my birthday dinner.",
        "vegetarian|vegetable + shellfish",
    ),
    ("composed", "Suggest a weekend hiking trip.", "flat|rolling|gentle + descent"),
    (
        "explicit",
        "What drink should I bring for movie night?",
        "sparkling|non-alcoholic|soda|juice|water",
    ),
]

FACT_PROMPT = """\
You are answering a question using excerpts retrieved from a long history of \
conversations. Each excerpt is prefixed with its date.

Excerpts:
{context}

Question: {question}

Answer with the specific stored value. If the excerpts do not contain the \
answer, reply NO_ANSWER."""

ADVICE_PROMPT = """\
You are advising a user, drawing on excerpts from your history with them. Each \
excerpt is prefixed with its date.

Excerpts:
{context}

Request: {question}

The excerpts will NOT contain a ready-made answer -- they contain what this \
user likes, avoids, owns and does. Make a concrete recommendation that reflects \
those preferences. If a preference CHANGED over time, honour the latest one. \
Never reply NO_ANSWER; a recommendation is always possible. Keep it brief."""

GROUNDED_PROMPT = """\
You are advising a user, drawing on excerpts from your history with them. Each \
excerpt is prefixed with its date.

Excerpts:
{context}

Request: {question}

Work in two steps, both in your reply:
1. PREFERENCE: quote, word for word, the excerpt fragment(s) stating the \
preference(s) that bear on this request. If one changed over time, quote the \
LATEST. If nothing bears on it, write "none stated".
2. RECOMMENDATION: one concrete recommendation that follows from exactly the \
preferences you quoted -- never from general taste. Keep it brief.

Never reply NO_ANSWER; if no preference applies, recommend anyway and say it \
is a guess."""

ARMS = {"fact": FACT_PROMPT, "advice": ADVICE_PROMPT, "grounded": GROUNDED_PROMPT}


async def main() -> None:
    from bench.harness import build_model_client

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=256,
        rerank_backend="heuristic",
        api_key_pepper="preference-lab-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, DeterministicEmbedder(dimensions=256), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="PrefLab"))
    space = await store.create_space(Space(org_id=org.id, slug="pref", name="Pref"))
    for sid, when, text in SESSIONS:
        await service.ingest(
            org_id=org.id,
            space_id=space.id,
            content=text,
            occurred_at=when,
            metadata={"session_id": sid},
            extract=False,
        )

    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    asked = datetime(2026, 8, 13, tzinfo=UTC)

    # Retrieval ONCE per question; every arm reads identical evidence, so a
    # difference between arms is the instruction and nothing else.
    contexts: dict[str, str] = {}
    for _kind, question, _spec in QUESTIONS:
        response = await service.search(
            SearchRequest(
                query=question, org_id=org.id, space_id=space.id, limit=6, asked_at=asked
            )
        )
        ordered = sorted(response.results, key=lambda h: h.memory.occurred_at)
        contexts[question] = "\n\n---\n\n".join(
            f"[{h.memory.occurred_at:%Y-%m-%d}]\n{h.memory.content}" for h in ordered
        )

    results: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    scores: dict[str, dict[str, tuple[bool, float]]] = {arm: {} for arm in ARMS}
    for arm, template in ARMS.items():
        for _kind, question, spec in QUESTIONS:
            raw, _tokens = await client.complete(
                template.format(context=contexts[question], question=question),
                max_tokens=1024,
            )
            answer = raw.strip()
            graded = score(answer, Expected.parse(spec))
            results[arm][question] = answer
            scores[arm][question] = (graded.correct, graded.completeness)

    print(f"\n{'ARM':10} {'correct':>8} {'names the preference':>21}")
    for arm in ARMS:
        correct = sum(c for c, _p in scores[arm].values())
        naming = sum(p for _c, p in scores[arm].values()) / len(QUESTIONS)
        print(f"{arm:10} {correct:5}/12 {naming:20.2f}")

    print("\nBY PREFERENCE TYPE (correct per arm):")
    kinds = sorted({k for k, _q, _s in QUESTIONS})
    print(f"  {'type':10}" + "".join(f"{arm:>10}" for arm in ARMS))
    for kind in kinds:
        row = f"  {kind:10}"
        for arm in ARMS:
            n = sum(1 for k, q, _s in QUESTIONS if k == kind)
            c = sum(scores[arm][q][0] for k, q, _s in QUESTIONS if k == kind)
            row += f"{c}/{n:>7}"
        print(row)

    print("\nDISAGREEMENTS (questions where arms differ):")
    for _kind, question, _spec in QUESTIONS:
        verdicts = {arm: scores[arm][question][0] for arm in ARMS}
        if len(set(verdicts.values())) > 1:
            print(f"  {question[:58]}")
            for arm in ARMS:
                mark = "." if verdicts[arm] else "X"
                print(f"    {mark} {arm:9} {results[arm][question][:90]!r}")

    out = Path(__file__).with_name("preference_results.json")
    json.dump(
        {
            "results": results,
            "scores": {a: {q: list(v) for q, v in s.items()} for a, s in scores.items()},
        },
        out.open("w"),
        indent=1,
    )
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
