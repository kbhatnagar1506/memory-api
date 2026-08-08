# Roadmap — from retrieval engine to memory system

The thesis, one line: **store everything losslessly, derive meaning at read
time, prove everything you claim.** Every competitor either extracts at write
time (and loses what the next question needed) or retrieves blobs and hopes
the model synthesizes. We do neither: episodes are ground truth forever, and
answers that exist in no episode are computed — with provenance — the moment
a question makes them matter.

Every phase below carries entry gates, exit criteria, and kill criteria.
A roadmap without kill criteria is a wishlist.

---

## Operating principles (each one was paid for)

1. **Measure before building.** Entity bridging was obviously right — 97% of
   multi-hop evidence sits in another session, entities bridge sessions. It
   measured 0.0 gain at 2.6× latency, twice, and is now a documented negative.
   The failure table, not the architecture diagram, picks what gets built.
2. **The harness is production code.** Seven harness bugs in this project
   fabricated or destroyed results (cache-key aliasing, silently-failed
   patches, API errors scored as answers, a thread-pool deadlock, context
   truncation to 9%, a judge grading rubrics as facts, packed evidence ids).
   Benchmark code gets the same tests, review, and typing as the service.
3. **Lossless write path, always.** The one independent cost study
   (arXiv 2603.04814) found extraction-based memory loses recall to raw
   replay. Write-time extraction decides what matters before the question
   exists. Nothing in any phase below deletes or compresses an episode.
4. **Extraction at write time is guessing; at read time it is answering.**
   The question tells you exactly what to extract. This inversion is the
   architectural bet of the whole system.
5. **Code does arithmetic; models find instances.** "How many bikes do I
   own?" was answered "Multiple" with all three bikes in context. 45% of
   failures-with-complete-evidence were counts. `len()` cannot miscount.
6. **Provenance or it didn't happen.** Every derived fact carries
   `derived_from` edges. Every extraction row carries a verbatim quote that
   is substring-checked against its source. Every erasure returns an
   attestation of what was destroyed — including the edges it had to bridge.
7. **Grade with the benchmark's own judge, never yourself.** Our homegrown
   judge was stricter than LongMemEval's on three of our four weakest
   capabilities; self-grading is the first thing a reader should distrust
   (largest published gap: 94.4% self-reported vs 49.0% independent). The
   judge is cross-family (Claude grades Gemini) and the self-preference
   delta is itself measured and published.
8. **Fail open.** The derive path's worst case is the status quo. Any
   subsystem whose failure mode is worse than its absence is misdesigned.

---

## Phase ledger

| phase | name | state | gate to proceed |
|---|---|---|---|
| 0 | Foundation | **done** | — |
| 1 | Validation run (4 stacked fixes) | **in flight** | results decompose per fix |
| 2 | Materialized derivation = derived facts + profiles | designed | derive path validated by P1 |
| 3 | Consolidation (proactive derivation) | designed | P2 invalidation proven |
| 4 | Distribution surface (MCP, mass-forget, keys) | ready | independent of P1 — build anytime |
| 5 | Prospective memory (intentions) | sketched | P2 shipped (same memory-kind machinery) |
| 6 | Procedural memory | assessed | explicit product decision, not drift |
| 7 | Scale, hardening, the replay flagship | partial | P2 shipped |

---

## Phase 0 — Foundation (done)

What exists, with the number that proves it:

- Hybrid retrieval (BM25 + pgvector HNSW, RRF): **full_recall@10 = 0.968**,
  hit@10 = 0.994, MRR = 0.939 on LongMemEval-S n=500. Retrieval is not the
  bottleneck; this number is retired as the thing to optimize.
- Delivery: **full_recall@4 = 0.972** — complete evidence reaches the
  answerer for 97.2% of questions.
- Bitemporal store: event time vs system time, `?as_of=` reads, version
  history, in-place-correction rule. Dual-backend conformance (in-memory
  reference + Postgres) — the suite is what makes two backends safe to have.
- Belief revision: typed edges (supersedes / contradicts / derived_from /
  references), transitive supersession suppression, **bridge-on-removal**
  (deleting the middle of a revision chain no longer resurrects the oldest
  fact — found live, fixed in both backends, 8 conformance tests).
- Erasure with attestation: live row + chunks + edges + full version history;
  attests to edges bridged as well as destroyed; erased neighbors are omitted
  from surviving contexts, never stubbed.
- Derive path (synthesis): classify → map → ground → reduce. Code counts,
  code sorts dates, code subtracts; grounding is a substring check; dedup is
  content-word Jaccard (stopwords excluded *before* comparison — raw Jaccard
  collapsed "three bikes" to one on scaffold words). 25 unit tests.
- Context endpoint: one read returns a memory with its entire relation
  neighborhood resolved — currency, replacement head, provenance,
  derivatives, references, contradictions.
- Judge protocol: LongMemEval's official per-type prompts verbatim;
  cross-family default (claude-sonnet-4-5 on Vertex grades gemini-2.5-flash);
  self-preference delta measured per run; predictions stored untruncated so
  any run can be re-graded in seconds.
- Benchmarks: LoCoMo (turn-level) + LongMemEval (session-level) under one
  harness, per-capability reporting, deliberately no blended score.
  184 tests, ruff, mypy --strict.

## Phase 1 — Validation run (in flight)

One run measures four stacked fixes: full-session context, official judge,
cross-family grading, derive path. Forecast, stated before the run:
central **72%**, range 62–80%; P(≥60%) ≈ 85%, P(≥70%) ≈ 55%, P(≥80%) ≈ 15%.

Decomposition comes from the per-question records:
`derived: true` isolates the derive increment; `judge_bias` isolates
self-grading; verdict deltas per capability isolate the judge protocol.

Decision tree on the headline:

- **< 62%** → the derive path is flipping right answers wrong. Audit the
  `derived: true` subset first; suspect the compute-preferred policy and map
  extraction recall. Fix before any phase builds on derive.
- **62–70%** → derive added little over context alone. Keep it (it can only
  add), demote its priority in P3, proceed to P2 on the provenance value.
- **70–80%** → forecast confirmed; proceed to P2 at full speed.
- **> 82%** → check the judge before celebrating. Sample 30 CORRECT verdicts
  by hand; a grader this generous is a bug until proven otherwise.

Also delivered by this run: the LoCoMo re-run with fixed evidence parsing,
and the first judge-bias number on real data.

## Phase 2 — Materialized derivation: derived facts and profiles are ONE mechanism

The market gap and our roadmap converge here. Every competitor ships "user
profiles"; none can say *why* a profile fact is true. A profile bucket
("preferences", "dietary", "work context") is nothing but a **standing derive
query, materialized**:

```
POST /v1/spaces/{space}/derive        # one-shot: question -> DerivedAnswer
POST /v1/spaces/{space}/profiles      # standing: bucket definition (a query)
GET  /v1/spaces/{space}/profiles/{bucket}
```

Mechanism — entirely existing machinery composed:

1. Run the derive path over the space (map → ground → reduce).
2. Store the result as a memory with `kind=derived`, content = the answer,
   metadata = the table, plus `derived_from` edges to every source episode.
3. Search returns derived facts like any memory — a repeated aggregate
   question becomes an indexed lookup. This is a **demand-driven index**:
   we materialize exactly the facts users actually ask about, and a wrong
   guess costs nothing because raw episodes remain.
4. **Invalidation is the differentiator.** Erase or supersede a source →
   walk incoming `derived_from` edges → mark derivations STALE → re-derive
   lazily on next read (or eagerly for profiles). Erasure that provably
   propagates into the profile. Extraction-based competitors cannot do this:
   their derived facts have fuzzy lineage by construction.
5. Profile buckets re-derive from scratch when the definition changes —
   possible only because episodes were never thrown away.

Staleness semantics: a derived memory is valid while its source set and the
bucket definition are unchanged. Store `source_set_hash`; the context
endpoint already exposes `derived_from`, so staleness is inspectable, not
hidden.

- Entry gate: P1 confirms derive answers are not net harmful (≥ 62% branch).
- Exit criteria: e2e test — ask an aggregate question twice (second hit is
  the materialized fact, measurably faster); erase a source; verify the
  profile fact goes stale and re-derives without the erased content; the
  attestation lists affected derivations.
- Kill criteria: if invalidation cannot be made airtight (a stale derived
  fact surviving its sources is a *lie with provenance*), ship one-shot
  `/derive` only and hold materialization.
- Effort: ~1 week. New code is one service method + staleness walk; the
  storage, edges, and derive engine all exist.

## Phase 3 — Consolidation: the same derivation, run proactively

Sleep-time compute (Letta's AutoDream, EverMemOS's engram lifecycle) frames
background consolidation as biology. Strip the metaphor and it is: **run
standing derive queries when the system is idle instead of when the user is
waiting.** Same map/ground/reduce, same provenance, same invalidation — only
the trigger differs.

- Scheduler: a background worker per space, budgeted (N derivations/day),
  triggered by write volume and staleness, never blocking reads.
- Targets, in priority order: stale profile buckets; **entity timelines**
  (dated event lists per recurring entity — Zep's graph, but demand-grown
  and lossless underneath); frequent-query aggregates (mined from search
  logs).
- Reconsolidation-lite: `last_accessed` + access count on memories (cheap
  now, a ranking signal later; ships with P3, evaluated on the harness
  before it touches ranking).
- Cost model published per derivation (map calls × docs + one reduce),
  because "background LLM calls" is exactly how competitors' token bills
  became marketing liabilities.
- Entry gate: P2 invalidation proven end-to-end.
- Exit criteria: a corpus ingested, idle consolidation runs, next-day
  aggregate questions hit materialized facts at lookup latency with correct
  invalidation under supersession.
- Kill criteria: if consolidated facts' hit rate on real queries is < ~20%
  (we materialized things nobody asks), consolidation stays lazy-only.

## Phase 4 — Distribution surface (independent of P1–P3; build anytime)

Not memory science — the reason anyone sees the memory science.

- **MCP server** (highest leverage per hour): expose search, ingest, derive,
  context, profiles as MCP tools. The portfolio stops being a repo and
  becomes something a Claude user experiences in five minutes.
- **Mass-forget with dryrun** (compliance UX Supermemory ships; ours can be
  stronger): `POST /forget {query, dryrun}` → preview the memory set →
  confirm → per-memory erasure with a **single combined attestation**,
  including every derivation invalidated. Machinery: search + erase + edge
  walk, all existing.
- **Container-scoped API keys**: keys bound to a space, not just org+scope.
- Explicitly deferred: connectors (pipeline plumbing, zero research value),
  infinite-chat proxy (distribution, revisit post-P5), multimodal ingestion
  (real scope cut — stated, not hidden).

## Phase 5 — Prospective memory: the whitespace

Every system in the market answers "what did the user say?" None answers
"what did the user ask me to remember to do?" Intentions are memory with a
**trigger**, and nobody ships them well.

```
{content: "user wants to be reminded to book flights",
 kind: "intention",
 trigger: {type: "topic", match: "travel planning"},
 status: "armed" | "fired" | "expired"}
```

- Evaluation at retrieval time: every search response carries a
  `fired_intentions` block — intentions whose trigger matched the query
  context. No scheduler, no push infrastructure; the agent surfaces it.
  (Time-based triggers = a trivial index scan on `not_after`.)
- Trigger matching is the existing lexical/vector machinery over trigger
  specs — an intention is retrieved by its trigger, not its content.
- One-shot semantics: fired intentions supersede themselves (existing
  machinery), so a reminder does not fire forever; `?as_of=` shows when it
  was armed and when it fired — auditable intentions for free.
- Entry gate: P2 (same memory-kind + lifecycle machinery).
- Kill criteria: if trigger precision on a held-out set is poor (constant
  false fires), narrow to explicit-topic triggers only. A reminder that
  fires wrongly is worse than no reminder.
- Why this is the differentiating phase: it is cheap (a memory kind + a
  retrieval-time filter), user-visible, and genuinely absent from Mem0,
  Zep, Letta, and Supermemory. First-mover on a mechanism beats
  fifth-mover on profiles.

## Phase 6 — Procedural memory (honest assessment)

"How we do things" — workflows, conventions, tool habits (Letta, LangMem
territory). Real gap, different product: it serves agent *execution*, not
recall QA, and none of our benchmarks measure it. The trap is scope drift
into an agent framework.

Minimal viable slice if entered: `kind=procedure` memories with structured
steps, retrieved by task similarity, updated by outcome feedback
(success/failure counts on edges — reconsolidation applied to skills).
Decision point after P5, and only with a benchmark for it; unmeasured
features are how this repo would rot.

## Phase 7 — Scale, hardening, and the replay flagship

- Scale: pgvector HNSW at 1M/10M chunks — measure recall/latency vs
  `max_scan_tuples`, iterative-scan relaxed ordering effects, publish curves.
  Latency SLOs stated as p50/p95/p99 at corpus size (49ms means nothing
  without them; nobody in the market states theirs either).
- Postgres bridge/conformance parity in local dev (docker), not CI-only.
- **The replay demo as the flagship artifact** (from the market synthesis —
  the demo no competitor can run): replay query Q as of time T → exact
  retrieved set, ranks, explain traces. Then erase a source episode →
  re-run the replay → cryptographically attest that the episode and every
  derivation transitively reachable from it is gone from reconstructed
  history too. Bitemporal store + supersession bridge + derived-fact
  invalidation + attestation, composed into one script. This is Thesis A
  made runnable.
- Compliance-shaped ops: exportable audit events, retention policies,
  legal hold. Certification is a company decision; the *code* being
  certifiable is ours.

---

## Standing rules for benchmarks and publishing

- Official judges only, named and versioned in every report; judge protocol
  in the payload (`judge_protocol`, `judge_bias`).
- Cross-family judge by default; self-grading delta published when measured.
- Per-capability tables always; blended scores never.
- Every report carries: commit hash, n, CIs (Wilson), error counts
  (excluded, never substituted, always disclosed), context tokens, latency.
- Negative results stay in the codebase as documentation (entity bridging,
  MMR, heuristic rerank) — a measured negative is worth more than an
  unmeasured feature.
- Any number that cannot be regenerated by `python -m bench.run` from a
  clean checkout does not get published.

## Risk register

| risk | exposure | mitigation |
|---|---|---|
| Derive flips right→wrong (compute-preferred policy) | P1 headline | per-question `derived` flag; decision tree branch < 62% |
| Map-stage extraction misses instances (under-count) | P2/P3 quality | grounding already strict; add extraction-recall probe set in P2 |
| Stale derived facts surviving invalidation | trust in P2 | kill criterion: one-shot-only fallback |
| Judge variance dominates above ~75% | interpretation | bias arm + hand-audit sample on any result > 82% |
| Dataset label noise (measured 0.5% LoCoMo; LongMemEval unknown) | ceiling claims | report as measured bound, never round up |
| Cost of consolidation LLM calls | P3 economics | published per-derivation cost model; daily budget cap |
| Scope drift into agent framework | P6 | benchmark-first entry rule |

## Non-goals (each killed by a measurement or a study)

- **Write-time fact extraction** — 2603.04814; the lossless write path is
  the moat.
- **Destructive forgetting / lossy consolidation** — same study; decay is a
  ranking signal, never a deletion policy.
- **Entity-bridging retrieval** — measured 0.0 twice at 2.6× latency.
- **MMR / heuristic rerank by default** — measured losses on this workload.
- **Blended benchmark scores** — a system excellent at recall and dangerous
  at abstention must not be able to hide either number.
- **Benchmark marketing** — the category's credibility crater is the
  competitor's weakness; inheriting it is optional.
