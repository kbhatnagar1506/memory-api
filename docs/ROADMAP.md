# Roadmap — the best memory system, and nothing else

**Thesis:** store everything losslessly, derive meaning at read time, prove
everything you claim.

This document covers memory quality only. No connectors, no MCP server, no
proxy, no ingestion pipelines for third-party SaaS. Those are distribution;
they make a product, not a memory system, and every hour spent on them is an
hour not spent on the thing being judged. If the memory is the best in the
world, surfacing it is a week's work later.

---

## What "best" means — six measurable axes

A memory system is not one number. These are the axes, each with the
question it answers and where we stand today (all measured, LongMemEval-S,
n=500, unless noted):

| # | axis | the question | where we are |
|---|---|---|---|
| 1 | **Recall** | can it find the evidence? | `full_recall@10 = 0.968`, delivery to the answerer `0.972` — **solved; retired as a target** |
| 2 | **Synthesis** | can it answer what no single memory contains? | derive path shipped, unmeasured; multi-session measured ceiling **54%** — **the frontier** |
| 3 | **Currency** | does it know which belief is current? | bitemporal + transitive supersession + bridge-on-removal — **strongest in market, one gap: no automatic contradiction detection** |
| 4 | **Calibration** | does it know when it does *not* know? | abstention 86.7%; 27.8% of answerable questions declined *while holding the evidence* — **both directions need work** |
| 5 | **Provenance** | can it prove why it believes something? | `derived_from` edges, verbatim-quote grounding, erase attestation, `?as_of=` replay — **nothing in the market matches this** |
| 6 | **Durability** | does removal remove, and does it hold at scale? | erase purges history + bridges chains + invalidates derivations; **scale untested past 374k chunks** |

Ranking these by remaining headroom is what orders the phases below. Axis 1
is done. Axis 2 is where the points are. Axis 4 is where trust is. Axes 3
and 5 are where we already beat everyone and must not regress.

---

## The measurement that drives everything

From the completed 500-corpus run, of the **454 questions that received
complete evidence**:

| outcome | n | share |
|---|---|---|
| correct | 190 | 41.9% |
| **declined while holding the evidence** | 126 | 27.8% |
| answered, judged wrong | 138 | 30.4% |

Retrieval contributed **2.8%** of total loss. Everything else happened after
the right memories were already in hand. And of the 264 failures with
complete evidence: **45% COUNT**, 12% ORDER, 6% DATE-ARITH, 3% COMPARE.

That table is the whole argument for read-time derivation, and it is the
reason no phase below is about retrieval.

---

## Phase ledger

| phase | axis | name | state |
|---|---|---|---|
| 1 | — | Validation run (4 stacked fixes) | **in flight** |
| 2 | 2, 5 | Materialized derivation + profiles | **done** |
| 3 | 2 | Compositional derivation (multi-hop) | next |
| 4 | 2 | Consolidation + entity timelines | after 3 |
| 5 | 3 | Contradiction detection | after 3 |
| 6 | 4 | Calibration | after 1 |
| 7 | 2, 4 | Extraction quality | after 3 |
| 8 | 3, 4 | Forgetting as ranking | after 5 |
| 9 | 3 | Prospective memory | after 5 |
| 10 | 1–6 | Evaluation completeness | continuous |
| 11 | 6 | Scale, durability, replay flagship | last |

---

## Phase 1 — Validation (in flight)

One run measures four stacked fixes: full-session context, official judge,
cross-family grading, derive path. Pre-registered forecast: central **72%**,
range 62–80%.

Decomposition is already instrumented — `derived: true` isolates the derive
increment, `judge_bias` isolates self-grading, per-capability verdicts
isolate the judge protocol.

**Decision tree (pre-registered, not to be rationalized after the fact):**

- **< 62%** → derive is flipping right answers wrong. Audit the `derived`
  subset first; suspect the compute-preferred policy and map recall.
  **Phase 7 jumps to the front of the queue.**
- **62–70%** → derive added little over the context fix alone. Keep it
  (it can only add), and Phase 3 becomes the test of whether composition
  earns its cost.
- **70–80%** → confirmed; proceed in listed order.
- **> 82%** → hand-audit 30 CORRECT verdicts before believing it. A grader
  this generous is a bug until proven otherwise.

---

## Phase 3 — Compositional derivation (axis 2, the frontier)

**Why.** `multi-session` is 133 questions — 28% of the benchmark — and its
measured potential is only **54%**, because 16% of its failures are
*not-literal*: the answer exists in no single episode and no single
aggregate. "How has my exercise routine changed this year" is not a count,
not a max, not a span. It is a derivation over derivations.

Today `classify()` picks exactly one kind and runs one reduce. That
architecture cannot express a two-step question.

**Steps.**

1. **Plan instead of classify.** Replace the single-label router with a
   plan: an ordered list of 1–3 derive steps, each with its own kind and
   sub-question. Regex handles the single-step majority (unchanged, still
   deterministic); a plan is only requested from the model when the question
   carries multi-hop markers (`and then`, `compared to`, superlative +
   filter, `which of the ... that I`).
2. **Derive over derived.** A step may take the previous step's *table* as
   its input instead of retrieved episodes. `Extraction` rows already carry
   date + fact + source_id, so a reduce over a table is the same code path.
3. **Provenance composes.** Step 2's derived memory gets `derived_from`
   edges to step 1's derived memory, not to the episodes. Invalidation is
   already transitive over those edges (Phase 2, tested two hops deep), so
   erasing a source episode still reaches the final answer. **This is why
   Phase 2 had to land first.**
4. **Filter-then-reduce.** The most common real shape is a filter followed
   by an aggregate ("restaurants I visited *in March* → cheapest"). Add a
   FILTER step kind that narrows a table by date range or predicate in code.
5. **Bounded.** Max 3 steps, max one model call per step per document, hard
   fail-open to single-step derivation on any planning error.

**Exit criteria.** On the multi-session slice specifically: measurable gain
over Phase 2 with the ablation flag isolating it, and no regression on
single-session capabilities.

**Kill criteria.** If plans are wrong more often than single-step
classification is, ship FILTER-then-reduce only (the deterministic half) and
abandon model-generated plans. A wrong plan wastes calls and produces a
confidently wrong table — worse than declining.

---

## Phase 4 — Consolidation and entity timelines (axis 2)

**Why.** Two reasons, one measured and one structural. Measured: repeated
aggregate questions currently pay full derivation cost every time.
Structural: the single most useful derived object for multi-session
questions is a **dated event timeline per entity** — and it is exactly what
graph-first competitors build at write time by guessing, which we can build
from real query demand without guessing.

**Steps.**

1. **Idle worker per space**, budgeted (N derivations/day), triggered by
   write volume and staleness, never blocking a read.
2. **Targets, in priority order:** stale profile buckets (Phase 2 marks
   them, nothing re-derives them yet — this closes that loop); entity
   timelines for recurring entities; aggregates mined from repeated queries.
3. **Entity timelines** are a derived memory of kind DERIVED whose table is
   the dated event list. Every subsequent COUNT/ORDER/ARITH question about
   that entity reduces over the timeline instead of re-reading episodes —
   fewer model calls, and Phase 3 composition gets a pre-built input.
4. **Published cost model** per derivation (map calls × docs + one reduce).
   Background LLM spend is how competitors' token bills became marketing
   liabilities; ours is stated, capped, and attributable.
5. **Nothing destructive.** No merging away originals, no `forgetAfter`
   deletion, no timestamp-forced conflict resolution. Consolidation only
   *adds* derived facts; episodes are untouched. This is the deliberate
   inversion of the "dream cycle" pattern.

**Exit criteria.** Ingest a corpus, let it idle, verify next-day aggregate
questions hit materialized facts at lookup latency, and that supersession of
a source still invalidates correctly.

**Kill criteria.** If materialized facts are hit by < ~20% of real queries,
we are guessing what matters — revert to lazy-only derivation.

---

## Phase 5 — Contradiction detection (axis 3, the one real gap)

**Why.** We have a `contradicts` edge type, a symmetric-edge writer, and an
explicit link endpoint — but **nothing ever detects a contradiction
automatically**. Supersession handles *revision* (new replaces old, ordered
by time). It cannot handle *disagreement*: two ACTIVE memories, neither
superseding the other, that cannot both be true. Real belief revision needs
both, and this is the one axis-3 hole in an otherwise market-leading story.

**Steps.**

1. **Detection at ingest, proposal not application** — same posture as
   supersession, for the same reason: silently hiding a user's data on a
   similarity score is a bad default with an invisible failure mode.
   High-similarity + negation/antonym + *non-orderable* timestamps ⇒ a
   `ContradictionProposal` with confidence and reason.
2. **Retrieval surfaces conflict rather than hiding it.** When two results
   in one response are joined by a `contradicts` edge, the response says so
   explicitly. An agent told "these two disagree" behaves better than one
   silently handed the winner. This is a genuine differentiator: every
   competitor resolves conflicts invisibly (Supermemory by newest
   timestamp, Mem0 at read time) — none *reports* them.
3. **Derivation refuses to average over contradictions.** If the grounded
   table contains contradicting rows, the derived answer says so rather than
   counting both. A count computed over mutually exclusive facts is a
   confidently wrong number.
4. **Contradictions are not invalidations.** A contradicted memory stays
   ACTIVE (nothing replaced it and it may be the true one) — this is why
   the status is not reused.

**Exit criteria.** Conformance tests for detect → propose → link → surface;
`knowledge-update` capability measured before and after.

**Kill criteria.** If precision is poor (paraphrases flagged as conflicts),
narrow to explicit negation patterns only. A false contradiction erodes
trust faster than a missed one.

---

## Phase 6 — Calibration (axis 4, where trust lives)

**Why.** Benchmarks reward answering; users punish confident wrongness. We
have both failure directions measured and neither optimized: **126 questions
declined while holding complete evidence** (under-confidence) and an
abstention rate of 86.7% on genuinely unanswerable questions, meaning ~13%
were answered when they should not have been (over-confidence).

**Steps.**

1. **Re-measure both directions post-Phase-1** — the context fix plausibly
   erased most under-confident declines; optimizing before measuring would
   be guessing.
2. **Grounded refusal.** A decline should carry *why*: no relevant memory
   retrieved, versus retrieved-but-insufficient. These are different
   failures and only the second is a memory-system failure.
3. **Confidence from the table, not the model.** Derivation already knows
   how much evidence it had (row count, source count, grounding rejection
   rate). Report it. A count from one grounded row is not a count from six.
4. **Abstention as a first-class metric, permanently reported.** Already
   judge-free (a sentinel check), so it carries no judge variance — the
   cleanest number in the whole report.

**Exit criteria.** Both directions improve without trading off against each
other; the report shows them side by side, never blended.

---

## Phase 7 — Extraction quality (axes 2, 4)

**Why.** Everything downstream of the map stage inherits its recall. If
extraction misses one of three bikes, code counts two — confidently, with
provenance, and wrong. This is the single most dangerous failure mode we
have introduced, and it is currently unmeasured.

**Steps.**

1. **Extraction-recall probe set.** Hand-label instances in ~50 real
   sessions; measure what the map stage finds. This is the number that
   bounds every derived answer's correctness, and it does not exist yet.
2. **Relative-date resolution audit.** The map prompt asks the model to
   resolve "last Tuesday" against the document date. Never verified. On a
   benchmark where `temporal-reasoning` is 133 questions, an unverified date
   resolver is a load-bearing assumption.
3. **Grounding-rejection telemetry.** Log how many extracted rows fail the
   substring check. A rising rejection rate is an early warning that the
   extractor is drifting — and is also a free hallucination-rate metric,
   which nobody else in this market publishes.
4. **Tune the dedup threshold on real data.** Jaccard ≥ 0.6 over
   content words is currently a reasoned guess validated on constructed
   cases (three bikes vs one gym visit). It should be fit to labelled
   retellings.

**Kill criteria.** If extraction recall is poor and cannot be improved by
prompt or by chunk-size changes, derivation stays advisory: report the table
and the computed value, but do not prefer it over the direct answer.

---

## Phase 8 — Forgetting done right (axes 3, 4)

**Why.** Every competitor forgets destructively (`forgetAfter`, dream-cycle
pruning, interference-based decay). The independent cost study is the reason
we will not: lossy consolidation measurably loses recall. But "never forget"
must not mean "rank everything equally forever."

**Steps.**

1. **Access signals.** `last_accessed` + access count per memory (cheap,
   ships as pure telemetry first).
2. **Reconsolidation as ranking only.** Frequently-retrieved memories rank
   higher; nothing is ever deleted or hidden by an access counter.
   Evaluated on the harness before it touches ranking — decay tuning is
   exactly the kind of change that feels right and measures negative.
3. **Decay tuning.** The current exponential-with-floor is a reasoned
   default; fit it per capability (temporal questions want recency,
   knowledge-update wants the current value, not the recent one).
4. **Explicit archival, never automatic deletion.** Archiving stays a
   caller decision. The only thing that removes content is erasure, and
   erasure attests.

---

## Phase 9 — Prospective memory (axis 3, the whitespace)

**Why.** Every system in this market answers "what did the user say?" None
answers "what did the user ask me to remember to *do*?" Intentions are
memory with a trigger, and nobody ships them.

**Steps.**

1. `kind=intention` memories with `{trigger: {type, match}, not_after}`.
2. **Evaluated at retrieval time** — every search response carries a
   `fired_intentions` block. No scheduler, no push infrastructure; triggers
   are matched by the existing lexical/vector machinery over trigger specs.
   Time triggers are an index scan on `not_after`.
3. **One-shot via self-supersession** — a fired intention supersedes itself,
   so reminders do not fire forever, and `?as_of=` shows when it was armed
   and when it fired. Auditable intentions, entirely from existing
   machinery.

**Kill criteria.** Poor trigger precision → narrow to explicit-topic
triggers. A reminder that fires wrongly is worse than no reminder.

---

## Phase 10 — Evaluation completeness (continuous, all axes)

A memory system is exactly as good as its measurement is honest. Standing
obligations:

1. **LoCoMo re-run** with the fixed evidence parser — current numbers
   predate that fix.
2. **A third benchmark.** LoCoMo (turn-level) and LongMemEval
   (session-level) test different things; a third with different structure
   again (ConvoMem-class) closes the "tuned to one benchmark" objection.
   Each stays separately reported — never blended.
3. **A derivation-specific suite.** No public benchmark isolates
   computed-answer correctness. Build one: counts, orderings, spans, and
   filter-then-aggregate over synthetic corpora with known ground truth.
   This is the benchmark our architecture implies, and publishing it with
   the harness is a stronger claim than any score.
4. **An invalidation suite.** Erase a source mid-run, verify no derived fact
   built on it is ever served again. Nobody in this market tests this
   because nobody else can pass it.
5. **Judge discipline, permanently:** official prompts, cross-family judge,
   self-preference delta published, per-capability tables, Wilson CIs,
   errors disclosed and never substituted, commit hash in every report.
6. **Negative results stay in the tree** with their numbers (entity
   bridging 0.0 at 2.6× latency, MMR, heuristic rerank). A measured
   negative is worth more than an unmeasured feature.

---

## Phase 11 — Scale and durability (axis 6)

Last on purpose: correctness first, then the numbers that prove it holds.

1. **pgvector HNSW at 1M and 10M chunks** — recall/latency curves against
   `max_scan_tuples`, iterative-scan relaxed-ordering effects, published.
2. **Latency SLOs as p50/p95/p99 at stated corpus size.** "49ms" means
   nothing without them — and notably, no competitor states theirs either,
   so stating ours is itself a differentiator.
3. **Postgres conformance in local dev**, not CI-only. The supersession
   bridge and derived-lifecycle tests currently prove one backend locally.
4. **Non-blocking ingest.** Accept → queue → embed, with processing status.
   Safe for us precisely because our write path has no extractor: the only
   deferred work is embedding, which cannot lose meaning. (Competitors defer
   *extraction*, which can.)
5. **The replay flagship.** Reconstruct a past query as of time T — exact
   retrieved set, ranks, explain traces. Then erase a source episode, re-run
   the replay, and attest that the episode and every fact transitively
   derived from it is gone from reconstructed history too. Every component
   exists; this composes them into the one demo no competitor can run.
6. **Adversarial review of the erasure path** — attempted three times,
   killed by session limits each time. Erasure is the one operation where a
   bug is unrecoverable.

---

## Standing engineering rules

1. Measure before building; a plausible mechanism that measures zero is a
   documented negative, not a feature.
2. The harness is production code — seven harness bugs in this project
   fabricated or destroyed results.
3. No LLM in the write path, ever. It is the moat.
4. No LLM in the default read path. Search stays deterministic and fast.
5. Code does arithmetic; models only find instances.
6. Every derived claim carries provenance; every removal attests.
7. Fail open — a subsystem whose failure is worse than its absence is
   misdesigned.
8. Every published number regenerable by `python -m bench.run` from a clean
   checkout.

## Non-goals

- **Connectors, MCP, proxies, SDK surface** — distribution, not memory.
  Explicitly excluded from this roadmap.
- **Write-time fact extraction** — the measured trap and the thing we are
  the alternative to.
- **Destructive forgetting / lossy consolidation** — same study; decay is a
  ranking signal, never a deletion policy.
- **Automatic conflict resolution by timestamp** — the market default; it
  silently picks a winner and hides the disagreement.
- **Entity-bridging retrieval, MMR, heuristic rerank by default** — measured
  losses, retained as documented negatives.
- **Working-memory / session-state management** — an agent-framework
  concern, orthogonal to a memory API.
- **Multimodal ingestion** — a real scope cut, stated rather than hidden.
- **Blended benchmark scores** — a system excellent at recall and dangerous
  at abstention must not be able to hide either number.
