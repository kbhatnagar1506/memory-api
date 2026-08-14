"""How many memories should reach the prompt? The only lever four rounds left.

Every mechanism tried against composed preferences -- write-time extraction,
multi-query retrieval, a compose instruction, a write-time domain index -- has
either done nothing or decomposed into WIDTH under a control. The domain index
made that explicit at n=177: it beat the baseline 20-7 (p=0.019), and a plain
k=8 matched per question took +9 of its +13, leaving routing itself at 15-11.

So the question stops being "what clever thing retrieves better" and becomes
"how many rows, and where does it stop paying". That is a config default, it
costs nothing, and it has never actually been measured here -- k=6 was picked
once and inherited by every experiment since.

THE TOP ARM IS THE POINT. Each persona's space holds exactly 23 memories, so
`k=23` is not a retrieval setting at all: it is the whole corpus, retrieval
switched off, the model handed everything. That arm answers the question the
sweep exists to ask:

  * if k=23 is the best arm, RANKING IS NOT THE PROBLEM and never was -- the
    system's job is to fit the budget, not to choose.
  * if k=23 is worse than some middle k, dilution is real and measurable, and
    the peak is the number to ship.

Paired sign tests throughout, versus k=6 and versus the neighbouring k, because
the arms differ by one setting on identical questions and a bare accuracy
column cannot tell a two-question wobble from a finding.

RESULT AT n=177, and it is the uncomfortable branch:

       k   correct    acc          vs k=6
       4   115/177   65.0%    8-19  p=0.052
       6   126/177   71.2%      --
       8   135/177   76.3%   15-6   p=0.078
      10   142/177   80.2%   19-3   p<0.001 *
      12   141/177   79.7%   25-10  p=0.017 *
      16   152/177   85.9%   32-6   p<0.001 *
      23   157/177   88.7%   36-5   p<0.001 *      <- retrieval switched off

Monotonic to the top. There is no dilution peak to ship: on this corpus the
best thing to do with the ranker is not use it, and every k below 23 is worse
than handing the model the whole space. The ranker is not sorting, it is
LOSING -- k=6 leaves eleven points of accuracy on the floor.

WHAT THE CEILING ARM DECOMPOSES. With every memory in context, retrieval
removed as a variable entirely:

    explicit   35/36     implicit  34/35     updated  36/36
    negative   29/35     composed  23/35

Three kinds are solved -- they were retrieval problems and nothing else.
Composed is 23/35, which is EXACTLY its k=8 score, paired 4-4, p=1.000. Give
the model every preference the persona has ever recorded and it still fails
twelve of thirty-five composed questions. Negative keeps six.

So the wall finally has a name, and it is not retrieval. Four levers were
declined against composed preferences -- write-time extraction, multi-query,
the domain index, and width itself -- and every one of them was a RETRIEVAL
lever aimed at a SYNTHESIS failure. They could not have worked. The one
attention lever tried (COMPOSE_PROMPT) failed too, but it was at least
pointed at the right thing.

THE CAVEAT THAT KEEPS THIS HONEST: 23 memories per persona is small enough
that "send everything" fits in a prompt, and a production space holds
thousands where that is not an option. So this is not "delete the ranker". It
is two narrower claims -- that a 23-memory corpus cannot evaluate a ranker,
because the ceiling arm is always available and always wins; and that composed
and negative failures are synthesis, PROVEN, because they do not move when
retrieval is made perfect.

    python -m bench.lab.width_sweep bench/lab/packs_v150.json
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
from .scoring import Expected, score
from .stats import fmt_p, sign_test, wilson

#: 23 is every memory a persona has: 13 authored sessions plus 10 unrelated
#: padding. Included deliberately as the no-retrieval reference.
WIDTHS = (4, 6, 8, 10, 12, 16, 23)
MAX_INFLIGHT = 12


async def main(packs: list[dict]) -> None:
    from bench.harness import build_model_client

    questions, dropped = gate(packs)
    print(f"packs: {len(packs)}   usable questions: {len(questions)}   dropped: {len(dropped)}")

    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    gap = asyncio.Semaphore(MAX_INFLIGHT)

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="width-sweep-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, GeminiEmbedder(dimensions=768), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="WidthLab"))

    spaces: dict[str, str] = {}
    pad_text = [body.strip() for _sid, _occ, _tags, body in INFRA]
    for i, pack in enumerate(packs):
        space = await store.create_space(
            Space(org_id=org.id, slug=f"wx-{i}", name=str(pack["persona"])[:40])
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
        for j, body in enumerate(pad_text):
            await service.ingest(
                org_id=org.id,
                space_id=space.id,
                content=body,
                occurred_at=T0 + timedelta(weeks=21 + j),
                metadata={"session_id": f"pad-{j}"},
                extract=False,
            )

    corpus_size = len(packs[0]["sessions"]) + len(pad_text)
    print(f"corpus: {corpus_size} memories/persona ({len(pad_text)} unrelated)\n")

    asked = datetime(2026, 8, 13, tzinfo=UTC)

    async def run(q: dict, k: int) -> tuple[bool, int]:
        async with gap:
            response = await service.search(
                SearchRequest(
                    query=q["question"],
                    org_id=org.id,
                    space_id=spaces[q["pack"]],
                    limit=k,
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
        return score(raw.strip(), Expected.parse(q["spec"])).correct, len(response.results)

    verdicts: dict[int, list[bool]] = {}
    got: dict[int, float] = {}
    for k in WIDTHS:
        print(f"running k={k} ...")
        rows = await asyncio.gather(*(run(q, k) for q in questions))
        verdicts[k] = [r[0] for r in rows]
        got[k] = sum(r[1] for r in rows) / len(rows)

    n = len(questions)
    print(
        f"\n{'k':>4} {'correct':>10} {'acc':>7}   {'95% CI':^14} {'mem':>6}"
        f"   {'vs k=6':>14}   {'vs previous':>14}"
    )
    for i, k in enumerate(WIDTHS):
        hits = sum(verdicts[k])
        low, high = wilson(hits, n)
        ci = f"[{low:5.1%}, {high:5.1%}]"
        row = f"{k:>4} {hits:6}/{n:<3} {hits / n:6.1%}   {ci} {got[k]:6.1f}"

        def paired(a: int, b: int) -> str:
            wins = sum(1 for x, y in zip(verdicts[a], verdicts[b], strict=True) if x and not y)
            loss = sum(1 for x, y in zip(verdicts[a], verdicts[b], strict=True) if y and not x)
            return f"{wins:2}-{loss:<2} {fmt_p(sign_test(wins, loss))}"

        row += f"   {'--' if k == 6 else paired(k, 6):>14}"
        row += f"   {'--' if i == 0 else paired(k, WIDTHS[i - 1]):>14}"
        print(row)

    print(f"\n{'BY KIND':12}" + "".join(f"{f'k={k}':>7}" for k in WIDTHS))
    kinds = sorted({q["kind"] for q in questions})
    for kind in kinds:
        idx = [i for i, q in enumerate(questions) if q["kind"] == kind]
        row = f"{kind:12}"
        for k in WIDTHS:
            row += f"{sum(verdicts[k][i] for i in idx):>4}/{len(idx):<2}"
        print(row)

    full = WIDTHS[-1]
    best = max(WIDTHS, key=lambda k: sum(verdicts[k]))
    #: Compared against the best arm that actually RETRIEVES. If `full` wins,
    #: the ranker is worse than not ranking, and reporting it against itself
    #: would hide that behind a 0-0.
    rival = max((k for k in WIDTHS if k != full), key=lambda k: sum(verdicts[k]))
    print(f"\nbest arm: k={best} at {sum(verdicts[best])}/{n}")
    wins = sum(1 for x, y in zip(verdicts[full], verdicts[rival], strict=True) if x and not y)
    loss = sum(1 for x, y in zip(verdicts[full], verdicts[rival], strict=True) if y and not x)
    print(
        f"k={full} (whole corpus, retrieval off) vs best retrieving arm k={rival}: "
        f"{wins}-{loss} {fmt_p(sign_test(wins, loss))}"
    )

    print("\nCEILING -- still wrong with every memory in context:")
    for kind in kinds:
        idx = [i for i, q in enumerate(questions) if q["kind"] == kind]
        print(f"  {kind:12} {len(idx) - sum(verdicts[full][i] for i in idx):2} of {len(idx)}")

    out = Path(__file__).with_name("width_sweep_results.json")
    out.write_text(
        json.dumps(
            {
                "n": n,
                "widths": list(WIDTHS),
                "verdicts": {str(k): v for k, v in verdicts.items()},
                "kinds": [q["kind"] for q in questions],
                "questions": [q["question"] for q in questions],
            },
            indent=1,
        )
    )
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    _packs = json.loads(Path(sys.argv[1]).read_text())["packs"]
    sys.exit(asyncio.run(main(_packs)))
