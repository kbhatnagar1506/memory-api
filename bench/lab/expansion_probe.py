"""Does `use_expansion` earn its model call? Measured on LoCoMo, not LongMemEval.

`use_expansion` has shipped defaulting False since it was written, with no
evidence either way. That is the same profile the reranker had, and the
reranker turned out to be the worst of five configs on every retrieval metric
at once -- so an untested flag is not a neutral thing to leave lying around.

WHY LoCoMo AND NOT LME. Expansion can only help where retrieval currently
fails, and on LongMemEval it barely fails: full_recall@10 is 0.968, leaving
3.2% of headroom for a mechanism that costs a model call per query. LoCoMo
labels individual dialogue TURNS rather than whole sessions, which is a much
harder target -- recall@10 is 0.644 -- so it has room to show an effect in
either direction. Measuring a lever where the metric is saturated is how you
conclude "no effect" about a corpus rather than about the lever.

WHAT EXPANSION DOES. `CompletionExpander` is HyDE: ask the model for a
hypothetical answer passage, embed it with the DOCUMENT-side embedding, and
average it into the query vector (`_mean_unit_vector`). The bet is that a
hypothetical answer sits nearer the real evidence in embedding space than the
question does. The risk, which is why this needs measuring rather than
assuming, is that a wrong hypothesis drags the query vector away from evidence
it would otherwise have found.

Scoring is against labelled evidence, so no judge and no judge variance. The
expansion itself needs a model; the measurement does not.

    python -m bench.lab.expansion_probe [n_corpora]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from mapi.config import Settings
from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

from .stats import fmt_p, sign_test

K = 10
MAX_INFLIGHT = 8


async def main(n_corpora: int) -> None:
    from bench.datasets.locomo import LoCoMo
    from bench.harness import build_model_client

    corpora = LoCoMo(Path("bench/data/locomo10.json")).load(limit_corpora=n_corpora)
    PER_CORPUS = 60  # sampled evenly; 1,978 x 2 arms is more than the effect needs
    scored = [(c, [q for q in c.questions if q.evidence_ids][:PER_CORPUS]) for c in corpora]
    total_q = sum(len(qs) for _, qs in scored)
    print(f"LoCoMo: {len(corpora)} corpora, {total_q} questions with labelled evidence")

    client = build_model_client("gemini-2.5-flash", "patchguard-reakon")
    gap = asyncio.Semaphore(MAX_INFLIGHT)

    async def complete(prompt: str) -> str:
        async with gap:
            raw, _tokens = await client.complete(prompt, max_tokens=256)
        return raw

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="expansion-probe",
    )
    embedder = GeminiEmbedder(dimensions=768)
    store = InMemoryStore()
    # MemoryService builds the expander itself from the completer -- that is
    # the wiring under test, so use it rather than injecting one directly.
    service = MemoryService(store, embedder, HeuristicReranker(), settings, completer=complete)
    org = await store.create_organization(Organization(name="ExpansionProbe"))

    rows: list[dict] = []
    for ci, (corpus, questions) in enumerate(scored, start=1):
        space = await store.create_space(
            Space(org_id=org.id, slug=f"lc-{ci}", name=corpus.corpus_id[:40])
        )
        doc_to_memory: dict[str, str] = {}
        for doc in corpus.documents:
            result = await service.ingest(
                org_id=org.id,
                space_id=space.id,
                content=doc.text,
                occurred_at=doc.occurred_at,
                metadata={"doc_id": doc.id},
                extract=False,
            )
            doc_to_memory[doc.id] = result.memory.id
        print(
            f"  [{ci}/{len(scored)}] {corpus.corpus_id}: "
            f"{len(corpus.documents)} docs, {len(questions)} questions"
        )

        async def ask(
            q,
            *,
            expand: bool,
            space_id: str = space.id,
            docmap: dict[str, str] = doc_to_memory,
        ) -> dict:
            async with gap:
                t0 = time.perf_counter()
                r = await service.search(
                    SearchRequest(
                        query=q.text,
                        org_id=org.id,
                        space_id=space_id,
                        limit=K,
                        use_expansion=expand,
                    )
                )
                ms = (time.perf_counter() - t0) * 1000
            got = {h.memory.id for h in r.results}
            want = {docmap[d] for d in q.evidence_ids if d in docmap}
            if not want:
                return {}
            hit = len(got & want)
            return {
                "qid": q.qid,
                "category": q.category,
                "expand": expand,
                "recall": hit / len(want),
                "full": hit == len(want),
                "ms": ms,
            }

        for q in questions:
            for expand in (False, True):
                row = await ask(q, expand=expand)
                if row:
                    rows.append(row)

    base = {r["qid"]: r for r in rows if not r["expand"]}
    exp = {r["qid"]: r for r in rows if r["expand"]}
    common = sorted(set(base) & set(exp))
    n = len(common)
    print(f"\nscored {n} questions, k={K}\n")

    def agg(d, key):
        return sum(d[q][key] for q in common) / n

    print(f"{'arm':16}{'recall@10':>11}{'full_recall':>13}{'median ms':>11}")
    for name, d in (("no expansion", base), ("expansion (HyDE)", exp)):
        ms = sorted(d[q]["ms"] for q in common)
        print(
            f"{name:16}{agg(d, 'recall'):>10.3f}{agg(d, 'full'):>13.3f}"
            f"{ms[len(ms) // 2]:>11.0f}"
        )

    win = sum(1 for q in common if exp[q]["recall"] > base[q]["recall"])
    lose = sum(1 for q in common if exp[q]["recall"] < base[q]["recall"])
    print(
        f"\nPAIRED  expansion better on {win}, worse on {lose}, "
        f"tied on {n - win - lose}   {fmt_p(sign_test(win, lose))}"
    )

    cats: dict[str, list[str]] = {}
    for q in common:
        cats.setdefault(base[q]["category"], []).append(q)
    print(f"\n{'category':22}{'base':>8}{'expand':>9}{'n':>6}")
    for cat, qs in sorted(cats.items()):
        b = sum(base[q]["recall"] for q in qs) / len(qs)
        e = sum(exp[q]["recall"] for q in qs) / len(qs)
        print(f"{cat!s:22}{b:>8.3f}{e:>9.3f}{len(qs):>6}")

    out = Path(__file__).with_name("expansion_probe_results.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)))
