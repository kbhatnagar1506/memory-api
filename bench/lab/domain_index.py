"""Index preferences by the DOMAIN they govern, and retrieve by key not cosine.

The one hypothesis left standing after three declines. Composed preferences
fail because the second preference is semantically distant from the question --
"plan Saturday breakfast with the grandkids" against a session about an allergy
panel and enchiladas -- so rank, not window size, is the binding constraint.
Widening k only worked on a corpus small enough to swallow whole; multi-query
made it worse under padding; write-time claim extraction moved nothing.

All three of those competed in the same arena: cosine. This one changes the
arena. If a session carrying "peanut-free zone" is TAGGED `pref:food` at write
time, and "plan Saturday breakfast" is classified `food` at query time, the
session is reachable by a filter that does not care how far apart the two
sentences sit in embedding space.

    write   session -> {preference, domain} pairs -> domain tags on the memory
    query   question -> domains -> one tag-filtered search per domain
    answer  union of semantic top-k and the domain hits

A FIXED VOCABULARY is what makes it work at all: if the writer says "food" and
the reader says "dining", nothing matches, and the whole mechanism silently
degrades to the semantic baseline. Both prompts get the same closed list.

THE CONTROL, because last round's win was width and not mechanism. The domain
arm adds memories to the context, so a plain wider search that adds the SAME
number is run alongside. If they tie, the domain machinery is buying nothing
that a bigger k would not.

Run on the PADDED corpus (23 sessions, 10 of them unrelated) -- the regime where
the previous finding evaporated, and therefore the only one worth testing in.

AND AT A SIZE THAT CAN ANSWER. The first run of this file scored 30/37 against
a 28/37 baseline and a 27/37 control, and the honest reading was written into
the commit message: n is too small. It was not a modest claim, it was an
arithmetic one -- a paired sign test needs a 6-0 split among disagreements
before it reaches p<0.05, and n=37 does not produce six of anything. So this
runs over nine authored personas, ~160 questions, and reports exact McNemar
p-values for domain-vs-baseline AND domain-vs-its-own-width-control. The second
is the one that decides: beating the baseline shows the arm helps, beating the
control shows the help is the ROUTING and not the extra rows routing drags in.

A null here is a real result and gets shipped as one. Three levers have already
been declined on this corpus; a fourth costs nothing extra to decline honestly.

VERDICT AT n=177 (9 personas, 35 composed): DECLINED, and this is the fourth.

    advice k=6 (baseline)        127/177  71.8%   [64.7, 77.9]   6.1 memories
    advice k=6 + domain index    140/177  79.1%   [72.5, 84.4]   7.9
    advice k=8 (WIDTH CONTROL)   136/177  76.8%   [70.1, 82.4]   7.9

    domain vs baseline   20-7    p=0.019 *
    domain vs control    15-11   p=0.557
    control vs baseline  18-9    p=0.122

The arm beats the baseline and the effect is real. It is also NOT THE
MECHANISM. Routing decomposes into the rows it adds and the choice of which
rows, and the width control -- fed the same count per question -- takes +9 of
the +13. What is left for routing itself is 15-11, a coin flip, and it is a
coin flip inside every kind separately, including the composed questions this
was built for (5-2, p=0.453). Two model calls per write and one per query buy
four questions that cannot be told apart from noise.

So the finding is the same shape as the multi-query result it was designed to
escape: a lever that looks like targeting and measures as width. Rank is still
the binding constraint on composed preferences, and nothing tried in four
attempts moves it. What DID move is k, which is a config default and free --
though at 18-9 (p=0.122) that is a direction, not yet a result, and a plain
width sweep is the cheap way to settle it.

    python -m bench.lab.domain_index bench/lab/packs_v150.json
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
from mapi.store.base import MemoryFilter
from mapi.store.memory import InMemoryStore

from .corpus import SESSIONS as INFRA
from .preference import ADVICE_PROMPT
from .preference_v2 import T0, gate
from .scoring import Expected, score
from .stats import fmt_p, sign_test, wilson

#: The closed vocabulary. Both the writer and the reader see exactly this list,
#: because a writer saying "food" and a reader saying "dining" is a silent
#: degradation to the baseline -- the mechanism would appear to run and do
#: nothing, which is the hardest kind of null to notice.
DOMAINS = (
    "food",
    "travel",
    "scheduling",
    "shopping",
    "exercise",
    "health",
    "work",
    "media",
    "home",
    "social",
)

TAG_PROMPT = """\
Below is one diary entry. Which of these life domains does it record a lasting \
PREFERENCE, habit or constraint about -- something that should shape a future \
recommendation?

Domains: {domains}

Reply with only the matching domain words, comma-separated, lowercase. Reply \
with the single word `none` if the entry records no lasting preference, only \
events. Most entries match one or two domains; never invent a domain outside \
the list.

Entry:
{text}"""

#: SELECTIVITY IS THE MECHANISM, and the first version threw it away.
#:
#: The instruction was "include every domain that could carry a relevant
#: constraint". The router obliged: 4 to 6 domains of 10 per question. A filter
#: that admits 60% of the vocabulary is not a filter, and the arm scored exactly
#: what the width control scored (4/7 both) because that is all it had become.
#:
#: At most two, and the second only when the request genuinely spans two areas
#: of life. A composed question needs precisely that -- two domains, not six.
QUERY_PROMPT = """\
Below is a request someone made to an assistant that knows their preferences.

Domains: {domains}

Name AT MOST TWO domains whose stored preferences would most change the answer. \
Pick one if the request touches one area of life; pick two only when it \
genuinely spans two (a dinner plan may span `food` and `health` if a dietary \
constraint could apply). Never pick three.

Reply with only the domain words, comma-separated, lowercase.

Request:
{question}"""


def _parse_domains(raw: str) -> list[str]:
    """Only words from the closed vocabulary survive."""
    words = {w.strip().lower() for w in raw.replace("\n", ",").split(",")}
    return [d for d in DOMAINS if d in words]


#: Ceiling on concurrent model calls. At n=180 the four arms are ~750
#: completions, and an unbounded `gather` fires every one of them at once --
#: which stops being a measurement of the memory system and becomes a
#: measurement of somebody's rate limiter. The retries that provokes would land
#: as unexplained variance between arms, which is the failure mode that makes a
#: result unreproducible rather than merely slow.
MAX_INFLIGHT = 12


async def main(packs: list[dict]) -> None:
    from bench.harness import build_model_client

    questions, dropped = gate(packs)
    composed = questions  # all kinds: n=7 was too small, and routing can regress others
    print(f"packs: {len(packs)}   usable questions: {len(composed)}   dropped: {len(dropped)}")
    for reason in dropped[:20]:
        print(f"  drop: {reason}")
    if len(dropped) > 20:
        print(f"  ... and {len(dropped) - 20} more")

    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")

    #: Gated at the JOB, not at the completion. One question's arm is a router
    #: call, one or two searches and an answer call, and only the two
    #: completions would be covered if the gate sat on `complete` -- leaving the
    #: EMBEDDING request behind every search ungated, and `gather` over 180
    #: questions would fire nine hundred of those at once. Holding the slot for
    #: the whole job costs a little throughput and bounds every outbound call.
    gap = asyncio.Semaphore(MAX_INFLIGHT)

    async def complete(prompt: str, *, max_tokens: int) -> str:
        raw, _tokens = await client.complete(prompt, max_tokens=max_tokens)
        return raw

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="domain-index-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, GeminiEmbedder(dimensions=768), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="DomainLab"))

    # -- write path: tag every session with the domains it governs -----------
    #
    # Tag each DISTINCT text once. The ten padding entries are byte-identical
    # across every persona, so tagging per (persona, session) would spend nine
    # calls answering the same question nine times -- and worse, sampling could
    # land the same padding in different domains for different personas, which
    # is variance injected straight into the surface the control measures.
    pad_text = [body.strip() for _sid, _occ, _tags, body in INFRA]
    distinct = sorted({s["text"] for pack in packs for s in pack["sessions"]} | set(pad_text))

    async def tag(text: str) -> list[str]:
        async with gap:
            raw = await complete(
                TAG_PROMPT.format(domains=", ".join(DOMAINS), text=text), max_tokens=128
            )
        return _parse_domains(raw)

    print(f"tagging {len(distinct)} distinct sessions ...")
    tags_for = dict(zip(distinct, await asyncio.gather(*map(tag, distinct)), strict=True))

    spaces: dict[str, str] = {}
    tagged = 0
    for i, pack in enumerate(packs):
        space = await store.create_space(
            Space(org_id=org.id, slug=f"dx-{i}", name=str(pack["persona"])[:40])
        )
        spaces[pack["persona"]] = space.id

        async def write(text: str, when: datetime, sid: str, space_id: str = space.id) -> int:
            domains = tags_for[text]
            await service.ingest(
                org_id=org.id,
                space_id=space_id,
                content=text,
                occurred_at=when,
                tags=tuple(f"pref:{d}" for d in domains),
                metadata={"session_id": sid},
                extract=False,
            )
            return len(domains)

        for s in pack["sessions"]:
            tagged += await write(s["text"], T0 + timedelta(weeks=s["week"]), s["id"])
        # The same padding as the previous round: unrelated work/infra diary.
        # Tagged by the same writer, so the padding gets whatever domains it
        # genuinely carries -- no thumb on the scale.
        for j, body in enumerate(pad_text):
            tagged += await write(body, T0 + timedelta(weeks=21 + j), f"pad-{j}")

    sizes = {len(pack["sessions"]) + len(INFRA) for pack in packs}
    print(f"corpus: {sorted(sizes)} sessions/persona ({len(INFRA)} unrelated)")
    print(f"domain tags written: {tagged}\n")

    asked = datetime(2026, 8, 13, tzinfo=UTC)

    async def semantic(q: dict, k: int) -> dict[str, object]:
        response = await service.search(
            SearchRequest(
                query=q["question"],
                org_id=org.id,
                space_id=spaces[q["pack"]],
                limit=k,
                asked_at=asked,
            )
        )
        return {h.memory.id: h.memory for h in response.results}

    async def domain_hits(q: dict, per_domain: int) -> tuple[dict[str, object], list[str]]:
        raw = await complete(
            QUERY_PROMPT.format(domains=", ".join(DOMAINS), question=q["question"]),
            max_tokens=128,
        )
        domains = _parse_domains(raw)
        found: dict[str, object] = {}
        for domain in domains:
            response = await service.search(
                SearchRequest(
                    query=q["question"],
                    org_id=org.id,
                    space_id=spaces[q["pack"]],
                    limit=per_domain,
                    asked_at=asked,
                    filters=MemoryFilter(tags=(f"pref:{domain}",)),
                )
            )
            for h in response.results:
                found.setdefault(h.memory.id, h.memory)
        return found, domains

    def render(memories: dict[str, object]) -> str:
        ordered = sorted(memories.values(), key=lambda m: m.occurred_at)  # type: ignore[attr-defined]
        return "\n\n---\n\n".join(
            f"[{m.occurred_at:%Y-%m-%d}]\n{m.content}"
            for m in ordered  # type: ignore[attr-defined]
        )

    async def run_domain(q: dict) -> tuple[bool, int, list[str]]:
        async with gap:
            base = await semantic(q, 6)
            extra, domains = await domain_hits(q, 3)
            merged = {**base, **extra}
            raw = await complete(
                ADVICE_PROMPT.format(context=render(merged), question=q["question"]),
                max_tokens=1024,
            )
        return score(raw.strip(), Expected.parse(q["spec"])).correct, len(merged), domains

    print(f"running domain arm over {len(composed)} questions ...")
    domain_rows = await asyncio.gather(*(run_domain(q) for q in composed))
    width = round(sum(r[1] for r in domain_rows) / len(domain_rows))

    async def run_plain(q: dict, k: int) -> tuple[bool, int]:
        async with gap:
            memories = await semantic(q, k)
            raw = await complete(
                ADVICE_PROMPT.format(context=render(memories), question=q["question"]),
                max_tokens=1024,
            )
        return score(raw.strip(), Expected.parse(q["spec"])).correct, len(memories)

    #: The control is matched PER QUESTION, not on the average. The domain arm
    #: returns a different number of memories for each question -- 6 when the
    #: router adds nothing new, 9 when two domains each contribute -- so a
    #: single averaged k over-feeds the easy ones and starves the hard ones,
    #: and it is precisely the hard ones the mechanism claims to fix. Matching
    #: row-for-row means any remaining difference cannot be width.
    print(f"running baseline (k=6) and per-question width control (avg k={width}) ...")
    base_rows, control_rows = await asyncio.gather(
        asyncio.gather(*(run_plain(q, 6) for q in composed)),
        asyncio.gather(
            *(run_plain(q, row[1]) for q, row in zip(composed, domain_rows, strict=True))
        ),
    )

    n = len(composed)
    verdicts = {
        "baseline": [r[0] for r in base_rows],
        "domain": [r[0] for r in domain_rows],
        "control": [r[0] for r in control_rows],
    }
    memories = {
        "baseline": sum(r[1] for r in base_rows) / n,
        "domain": sum(r[1] for r in domain_rows) / n,
        "control": sum(r[1] for r in control_rows) / n,
    }
    labels = {
        "baseline": "advice k=6 (baseline)",
        "domain": "advice k=6 + domain index",
        "control": f"advice k={width} (WIDTH CONTROL)",
    }

    print(f"\n{'arm':30} {'correct':>9} {'acc':>7}   {'95% CI':^14} {'memories':>9}")
    for arm in ("baseline", "domain", "control"):
        hits = sum(verdicts[arm])
        low, high = wilson(hits, n)
        print(
            f"{labels[arm]:30} {hits:5}/{n:<3} {hits / n:6.1%}   "
            f"[{low:5.1%}, {high:5.1%}] {memories[arm]:9.1f}"
        )

    #: The decision. Paired on the question, so the enormous variance in
    #: question difficulty cancels; only disagreements carry signal.
    #:
    #: Two comparisons, and the SECOND is the one that matters. Beating the
    #: baseline only shows the arm helps. Beating its own width control shows
    #: the help is the domain routing rather than the extra rows the routing
    #: happens to drag in -- which is exactly the confound that killed the
    #: multi-query result last round, where a plain wider k matched it.
    def paired(a: str, b: str) -> tuple[int, int, float]:
        a_only = sum(1 for x, y in zip(verdicts[a], verdicts[b], strict=True) if x and not y)
        b_only = sum(1 for x, y in zip(verdicts[a], verdicts[b], strict=True) if y and not x)
        return a_only, b_only, sign_test(a_only, b_only)

    print("\nPAIRED SIGN TESTS (exact McNemar; * marks p<0.05)")
    for a, b in (("domain", "baseline"), ("domain", "control"), ("control", "baseline")):
        a_only, b_only, p = paired(a, b)
        print(
            f"  {a:8} vs {b:8}  {a_only:3} - {b_only:<3} "
            f"({a_only + b_only} discordant of {n})   {fmt_p(p)}"
        )

    print(f"\n{'BY KIND':12} {'base':>8} {'domain':>8} {'control':>8}   dom-vs-base")
    kinds: dict[str, list[int]] = {}
    index: dict[str, list[int]] = {}
    for i, (q, dom, base, ctrl) in enumerate(
        zip(composed, domain_rows, base_rows, control_rows, strict=True)
    ):
        row = kinds.setdefault(q["kind"], [0, 0, 0, 0])
        row[0] += base[0]
        row[1] += dom[0]
        row[2] += ctrl[0]
        row[3] += 1
        index.setdefault(q["kind"], []).append(i)
    for kind, (b, d, c, total) in sorted(kinds.items()):
        rows = index[kind]
        d_only = sum(1 for i in rows if verdicts["domain"][i] and not verdicts["baseline"][i])
        b_only = sum(1 for i in rows if verdicts["baseline"][i] and not verdicts["domain"][i])
        p = sign_test(d_only, b_only)
        print(
            f"{kind:12} {b:5}/{total:<2} {d:5}/{total:<2} {c:5}/{total:<2}   "
            f"{d_only:2} - {b_only:<2}  {fmt_p(p)}"
        )

    routed = [len(r[2]) for r in domain_rows]
    picked: dict[str, int] = {}
    for r in domain_rows:
        for domain in r[2]:
            picked[domain] = picked.get(domain, 0) + 1
    print(
        f"\nrouter selectivity: {sum(routed) / len(routed):.1f} domains per question "
        f"(of {len(DOMAINS)}); max {max(routed)}; "
        f"{sum(1 for x in routed if x == 0)} routed nowhere"
    )
    spread = ", ".join(f"{k}:{v}" for k, v in sorted(picked.items(), key=lambda kv: -kv[1]))
    print(f"router spread: {spread}")

    print("\nDISAGREEMENTS (domain vs baseline), first 25:")
    shown = 0
    for q, dom, base in zip(composed, domain_rows, base_rows, strict=True):
        if dom[0] != base[0] and shown < 25:
            shown += 1
            winner = "domain" if dom[0] else "base"
            print(f"  {winner:7} only: [{q['kind']:9}] {q['question'][:48]}")
            print(f"           routed: {dom[2]}")

    out = Path(__file__).with_name("domain_index_results.json")
    out.write_text(
        json.dumps(
            {
                "n": n,
                "packs": len(packs),
                **verdicts,
                "control_k": width,
                "kinds": [q["kind"] for q in composed],
                "questions": [q["question"] for q in composed],
                "domains_per_question": [r[2] for r in domain_rows],
            },
            indent=1,
        )
    )
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    _packs = json.loads(Path(sys.argv[1]).read_text())["packs"]
    sys.exit(asyncio.run(main(_packs)))
