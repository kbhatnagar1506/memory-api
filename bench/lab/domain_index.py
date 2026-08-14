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

    python -m bench.lab.domain_index bench/lab/packs_v2.json
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


async def main(packs: list[dict]) -> None:
    from bench.harness import build_model_client

    questions, _dropped = gate(packs)
    composed = questions  # all kinds: n=7 was too small, and routing can regress others
    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")

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
    spaces: dict[str, str] = {}
    tagged = 0
    for i, pack in enumerate(packs):
        space = await store.create_space(
            Space(org_id=org.id, slug=f"dx-{i}", name=str(pack["persona"])[:40])
        )
        spaces[pack["persona"]] = space.id

        async def write(text: str, when: datetime, sid: str, space_id: str = space.id) -> int:
            raw, _tokens = await client.complete(
                TAG_PROMPT.format(domains=", ".join(DOMAINS), text=text), max_tokens=128
            )
            domains = _parse_domains(raw)
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
        for j, (sid, _occ, _tags, body) in enumerate(INFRA):
            tagged += await write(body.strip(), T0 + timedelta(weeks=21 + j), f"pad-{sid}")

    total_sessions = len(packs[0]["sessions"]) + len(INFRA)
    print(f"corpus: {total_sessions} sessions/persona ({len(INFRA)} unrelated)")
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
        raw, _tokens = await client.complete(
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
        base = await semantic(q, 6)
        extra, domains = await domain_hits(q, 3)
        merged = {**base, **extra}
        raw, _tokens = await client.complete(
            ADVICE_PROMPT.format(context=render(merged), question=q["question"]),
            max_tokens=1024,
        )
        return score(raw.strip(), Expected.parse(q["spec"])).correct, len(merged), domains

    print("running domain arm ...")
    domain_rows = await asyncio.gather(*(run_domain(q) for q in composed))
    width = round(sum(r[1] for r in domain_rows) / len(domain_rows))

    async def run_plain(q: dict, k: int) -> tuple[bool, int]:
        memories = await semantic(q, k)
        raw, _tokens = await client.complete(
            ADVICE_PROMPT.format(context=render(memories), question=q["question"]),
            max_tokens=1024,
        )
        return score(raw.strip(), Expected.parse(q["spec"])).correct, len(memories)

    base_rows = await asyncio.gather(*(run_plain(q, 6) for q in composed))
    control_rows = await asyncio.gather(*(run_plain(q, width) for q in composed))

    n = len(composed)
    print(f"\n{'arm':28} {'correct':>9}  {'avg memories':>13}")
    print(
        f"{'advice k=6 (baseline)':28} {sum(r[0] for r in base_rows):5}/{n:<3} "
        f"{sum(r[1] for r in base_rows) / n:12.1f}"
    )
    print(
        f"{'advice k=6 + domain index':28} {sum(r[0] for r in domain_rows):5}/{n:<3} "
        f"{sum(r[1] for r in domain_rows) / n:12.1f}"
    )
    print(
        f"{f'advice k={width} (WIDTH CONTROL)':28} {sum(r[0] for r in control_rows):5}/{n:<3} "
        f"{sum(r[1] for r in control_rows) / n:12.1f}"
    )

    print(f"\n{'BY KIND':12} {'base':>8} {'domain':>8} {'control':>8}")
    kinds: dict[str, list[int]] = {}
    for q, dom, base, ctrl in zip(composed, domain_rows, base_rows, control_rows, strict=True):
        row = kinds.setdefault(q["kind"], [0, 0, 0, 0])
        row[0] += base[0]
        row[1] += dom[0]
        row[2] += ctrl[0]
        row[3] += 1
    for kind, (b, d, c, total) in sorted(kinds.items()):
        print(f"{kind:12} {b:5}/{total:<2} {d:5}/{total:<2} {c:5}/{total:<2}")

    routed = [len(r[2]) for r in domain_rows]
    print(
        f"\nrouter selectivity: {sum(routed) / len(routed):.1f} domains per question "
        f"(of {len(DOMAINS)}); max {max(routed)}"
    )

    print("\nDISAGREEMENTS (domain vs baseline):")
    for q, dom, base in zip(composed, domain_rows, base_rows, strict=True):
        if dom[0] != base[0]:
            winner = "domain" if dom[0] else "base"
            print(f"  {winner:7} only: [{q['kind']}] {q['question'][:52]}")
            print(f"           routed: {dom[2]}")

    out = Path(__file__).with_name("domain_index_results.json")
    out.write_text(
        json.dumps(
            {
                "baseline": [r[0] for r in base_rows],
                "domain": [r[0] for r in domain_rows],
                "control": [r[0] for r in control_rows],
                "control_k": width,
                "domains_per_question": [r[2] for r in domain_rows],
            },
            indent=1,
        )
    )
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    _packs = json.loads(Path(sys.argv[1]).read_text())["packs"]
    sys.exit(asyncio.run(main(_packs)))
