# Architecture

How the system is put together, why each part is shaped the way it is, and
what was measured to decide it. This is the internal document — the public
`/docs` page deliberately describes behaviour rather than mechanism.

Conventions used throughout: **measured** means a number produced by our own
benchmark runs on LongMemEval (n=500) or LoCoMo (n=1,986), and the number is
quoted. **Rejected** means it was built, measured, and removed or defaulted
off — those sections are as important as the ones describing what shipped.

---

## 1. The shape of the thing

A memory API for agents. Two verbs matter — write something, ask something —
and everything below exists to make the second one answerable months after
the first.

```
                        ┌─────────────────────────────────────┐
   agent / SDK ────────▶│  FastAPI                            │
   browser ────────────▶│    /v1/*      API-key auth          │
                        │    /app/*     session-cookie auth   │
                        └────────────────┬────────────────────┘
                                         │
                                 ┌───────▼────────┐
                                 │ MemoryService  │  transactional boundary
                                 └───┬────────┬───┘
                        write path   │        │   read path
                     ┌───────────────▼──┐  ┌──▼──────────────────┐
                     │ chunk → embed →  │  │ RetrievalPipeline   │
                     │ dedupe →         │  │  fuse → rerank →    │
                     │ consolidate      │  │  decay → suppress   │
                     └───────┬──────────┘  └──────────┬──────────┘
                             │                        │
                        ┌────▼────────────────────────▼────┐
                        │  MemoryStore  (abstract)         │
                        │    InMemoryStore | PostgresStore │
                        └──────────────────────────────────┘
```

Three layers, and the boundaries are enforced rather than conventional:

| layer | may import | never imports |
|---|---|---|
| `api/` | service, schemas | domain internals |
| `service.py` | domain, store | FastAPI |
| `domain/` | nothing outward | store, api, vendors |
| `store/` | domain models | service, api |

`domain/` is pure. It contains no I/O, no vendor SDK, and no framework — which
is why every algorithm in it (fusion, decay, MMR, consolidation, grounding)
is unit-testable without a database or a network. Anything needing I/O is
injected as a callable: `CompleteFn = async (str) -> str` is the entire
surface between the domain and any LLM vendor.

---

## 2. Data model

Seven tables. Ids are prefixed ULIDs (`mem_01J...`, `spc_...`, `org_...`) —
sortable by creation time, and the prefix makes a mis-routed id a validation
error instead of a silent empty result.

### 2.1 Tenancy

```
organizations ──┬── spaces ──── memories ──── chunks
                ├── api_keys                     │
                └── memberships ──── users       └── embedding vector(768)
                                                     search_vector tsvector
```

**`organizations`** — the billing and isolation boundary.
**`spaces`** — one subject's memory. Usually one end user. Unique on
`(org_id, slug)` so a readable name is addressable.
**`users` / `memberships`** — identity, separate from authorisation. A user is
keyed on `google_sub`, never email: email is mutable and reassignable inside a
Workspace domain, and matching on it means a departed employee's replacement
inherits their memories. Membership is many-to-many because a person belongs
to several organizations by design.
**`api_keys`** — `key_hash` only. The plaintext key exists once, in the
response that created it.

### 2.2 Memories

| column | type | why it exists |
|---|---|---|
| `content` | text | the memory, stored losslessly |
| `summary` | text | optional, caller-supplied |
| `meta` | jsonb | GIN-indexed; carries `extracted_from`, `doc_id`, `subject` |
| `tags` | varchar[] | GIN-indexed |
| `status` | varchar | `active` · `superseded` · `archived` · `stale` |
| `kind` | varchar | `episodic` (what happened) · `derived` (what is true) |
| `occurred_at` | timestamptz | **event** time — when the thing happened |
| `created_at` | timestamptz | **system** time — when we learned it |
| `content_sha256` | varchar | exact-duplicate check before any embedding |
| `version` | integer | monotonic, drives `memory_versions` |

The two timestamps are the bitemporal core. "What did I believe about Krishna's
job in March, given what we know now" and "what did I believe in March, using
only what we knew in March" are different questions, and a single timestamp
can answer only one of them.

Indexes on `memories`:

```
ix_memories_tenant             (org_id, space_id)
ix_memories_tenant_status_id   (org_id, space_id, status, id)   -- keyset pagination
ix_memories_tenant_occurred    (org_id, space_id, occurred_at)  -- time-scoped queries
ix_memories_content_hash       (org_id, space_id, content_sha256)
ix_memories_tags               GIN (tags)
ix_memories_metadata           GIN (meta)
```

`ix_memories_tenant_status_id` exists for cursor pagination specifically: the
cursor is `(status, id)`, and without the composite the keyset scan degrades
to a sort.

### 2.3 Chunks — the retrieval unit

A memory is stored whole and retrieved in pieces. `chunks` carries both search
representations side by side:

```
embedding      vector(768)   HNSW (m=16, ef_construction=64), vector_cosine_ops
search_vector  tsvector      GENERATED ALWAYS AS to_tsvector('english', text) STORED
               GIN index
```

The tsvector is a **generated column**, not computed at query time — Postgres
maintains it on write, so the index can never disagree with the text. Cosine
ops rather than L2 because embeddings are L2-normalised on write, which makes
cosine similarity a dot product.

Unique on `(memory_id, ordinal)`: chunk order is part of the data, not an
artefact of insertion.

### 2.4 Relations — belief revision as a graph

```sql
relation_edges (source_id, target_id, type, reason, confidence)
UNIQUE (space_id, source_id, target_id, type)
```

Four types, and the distinction between the first two is the whole design:

- **`supersedes`** — revision. A newer memory replaces an older one, and time
  orders them. The old one flips to `superseded` and default retrieval hides
  it while keeping it addressable.
- **`contradicts`** — disagreement. Two memories conflict and nothing about
  their timestamps says which wins. **Neither is hidden.** Every competitor
  resolves this invisibly by taking the newest, which is indistinguishable
  from there being no conflict at all. An agent told "these two disagree" can
  ask the user; an agent handed the winner cannot.
- **`derived_from`** — provenance. Links an extracted claim to the passage it
  came from.
- **`references`** — a soft mention.

### 2.5 Versions — the bitemporal history

```sql
memory_versions (memory_id, version, valid_from, valid_to, ...)
ix_versions_current  btree (memory_id) WHERE valid_to IS NULL   -- partial
```

Every mutation closes the current row (`valid_to = now()`) and opens a new
one. `get_memory_as_of(t)` selects the row where `valid_from <= t < valid_to`.
The partial index makes "current version" a single-row lookup rather than a
`MAX(version)` aggregate.

---

## 3. The write path

```
validate → normalize → exact-dup (hash) → quota → chunk → embed
        → near-dup (cosine) → [extract] → [supersede] → [contradict] → persist
```

Ordering is the design. Each step is placed where it costs least:

1. **Exact duplicate before embedding.** A content hash is free; an embedding
   call is not, and re-ingesting an unchanged document is the single most
   common write a memory system sees.
2. **Quota before embedding.** Checking after would let a tenant over its
   limit still spend the vendor call the limit exists to prevent.
3. **Near-duplicate after embedding**, because it needs the vector.
4. **Consolidation last**, because it needs the memory to exist for edges to
   point at.

### 3.1 Chunking

Target 320 tokens, overlap 8, splitting on paragraph → sentence → hard cut.
`_SENTENCE_RE` handles both Latin `.!?` and CJK `。！？`.

### 3.2 Consolidation candidates

Every consolidation check asks the same question — *which existing memories
are close enough to this one to be about the same thing* — and it is answered
with `store.neighbours()`, an ANN lookup over the HNSW index bounded by
`consolidation_candidates` (64).

> **Fixed.** This was previously `list_memories(limit=256)`: the 256 **newest**
> memories with every embedding pulled across the wire, roughly 1.5MB per
> write at 768 dimensions, almost all of it immediately discarded by a
> similarity filter. It was also a silent scale ceiling — past 256 memories in
> a space, an older fact could never again be superseded or contradicted,
> because it was never among the rows compared.

### 3.3 Supersession

`propose_supersessions` scores candidates; the service applies only those
above `supersede_min_confidence`.

> **Measured.** Applying every proposal fired 94 supersessions on one corpus at
> a median confidence of 0.43. At the proposer's own floor, "same subject"
> means cosine 0.72 — two chat turns about the same hobby clear that easily
> without either replacing the other. Applying flips the older memory to
> `superseded`, which default retrieval hides, so a topical coincidence
> silently deletes a true memory from every future answer. Declined proposals
> are reported in the response rather than dropped.

### 3.4 Contradiction detection

Two passes. The lexical one is free and runs always: flipped negation,
opposing terms, differing figure or weekday — over pairs inside the
similarity band `[0.82, 0.97)`.

The second pass exists because the first was gated on seven strings
(`not`, `never`, `cannot`, `can't`, `won't`, `will not`, `stopped`). "I gave
up eating meat" and "I no longer eat meat" read as agreement. So
`unexplained_pairs()` hands the model exactly the pairs that cleared the
similarity gate and that no lexical signal could explain.

**This one fails closed** — uniquely in the system. Everywhere else a vendor
failure degrades a ranking. Here a failure that invented conflicts would write
`contradicts` edges between memories that agree, and a false contradiction
erodes trust faster than a missed one. Timeout, parse failure, or no backend:
zero proposals.

### 3.5 Write-time extraction

`extract.py` decomposes a passage into atomic, grounded claims — each carrying
a verbatim quote from its source, checked in code rather than by a judge.
Claims are **added** beside the parent with a `derived_from` edge; the parent
is untouched and still retrievable.

Input is windowed at `MAX_CHARS = 12,000` on paragraph boundaries.

> **Fixed.** There was no upper bound at all, and `max_content_bytes` is 1MB.
> A 129,000-character document went into a single prompt against an
> 8,192-token output budget: the model read the opening, the completion hit
> MAX_TOKENS, the salvage recovered whatever objects were complete, and the
> result was a handful of claims about page one presented as the facts of the
> whole document. Silent, plausible under-extraction is the worst failure
> shape available.

> **Measured, and the reason extraction is opt-in.** On LongMemEval:
>
> | arm | accuracy | Δ questions |
> |---|---|---|
> | baseline | 0.8255 | — |
> | add (parent + claims) | 0.7809 | −21 |
> | only (claims replace) | 0.7617 | −30 |
> | add + per-source cap | 0.7106 | −54 |
> | add + cap + kind routing | 0.7100 | −54 |
>
> The sign split cleanly by memory kind, six capabilities for six: every
> **semantic** capability held or improved, every **episodic** one lost. And
> the `only` arm retrieved *better* (full-recall 0.950 vs 0.948) while
> answering *worse* — so the cost lands at retrieval-time selection, not at
> storage. Claims answer "what is true"; episodes answer "what happened".
>
> This is why extraction defaults off, why `route_by_kind` exists, and why
> the `kinds` filter is exposed on the API.

---

## 4. The read path

```
query
  ├─▶ embed (query-side task type) ─┐
  └─▶ classify intent (small LLM) ──┘  concurrent
        ↓
  vector search ─┐
  lexical search ┘  concurrent
        ↓
  reciprocal rank fusion (k=60)
        ↓
  [entity bridging]        ← off by default, see §4.6
        ↓
  hydrate → rerank → recency decay → supersession suppression → [MMR] → top-k
```

Every stage appends to `explain`, so a result can always answer "why is this
here, and why here rather than three places up".

### 4.1 Stage order

- **Fusion before reranking**, so the reranker sees candidates *either*
  strategy liked, not just the vector winner.
- **Reranking before decay**, because a reranker judges topical relevance and
  has no idea how old anything is. Feeding it decayed scores would let age
  leak into a judgement that should be about meaning.
- **Decay before MMR**, so diversification trades off the final relevance.
- **Suppression last among the filters**, because it needs the full candidate
  set to know whether a superseding memory is *also* in the results — the only
  case where hiding the old one is safe.

### 4.2 Hybrid retrieval and RRF

Vector and lexical run concurrently and fuse by reciprocal rank
(`1/(k + rank)`, k=60). Rank-based rather than score-based because the two
scores are not commensurable: cosine similarity and `ts_rank` do not share a
scale, and normalising them invents a relationship that does not exist.

Losing the embedding provider degrades to lexical-only rather than failing the
request; the response records which strategies actually ran.

### 4.3 Query understanding — the one model call on the read path

The read path is otherwise LLM-free. `understand.py` classifies each question
into one of seven shapes with the smallest available model
(`gemini-2.5-flash-lite`, thinking off, 16 output tokens).

Three properties make it safe there:

- **Fails open.** Timeout, vendor error, unparseable reply, no backend
  configured — every one falls through to the regex classifier. The degraded
  mode is exactly the prior behaviour, so a vendor outage costs ranking
  quality and never availability.
- **Bounded.** Hard `asyncio.wait_for`. A slow vendor makes search dumber,
  never slower.
- **Cached and circuit-broken.** LRU by question; after 3 consecutive failures
  it stops calling for 30s, so an outage costs one timeout rather than one per
  search.

It runs **concurrently with the query embedding**, which is what makes it
affordable: both are one round trip, so understanding costs the difference
between them rather than its own latency.

Disagreements with the regex are logged with both labels — the regexes were
validated against 264 real failures and 1,986 LoCoMo questions with zero false
positives, so replacing them on faith would trade a measured thing for an
unmeasured one.

### 4.4 Coverage — comprehensive questions

`list_all` questions widen the window instead of taking top-k.

> **Measured.** A space holding 25 infrastructure facts, asked "what is our
> entire infrastructure", returned 10 with every score inside a 2% band
> (0.0143–0.0164) — so *which* ten was arbitrary. Postgres, Redis and pgvector
> were not among them. Ranking answers "which is most relevant"; this question
> asks "what is everything", and relevance has no opinion on completeness.

The decision is made at **stage 0**, before the candidate fetch — widening
after the fetch widens a set already cut to size. It is detected from the
question's **shape**, never from the score distribution: a score-cliff rule
was measured and falsified (`budget.py`) — similarity scores are flat enough
that "at least 80% of the top score" admits nearly everything, a fixed number
wearing an adaptive costume.

### 4.5 Recency decay and confidence

Exponential decay on `occurred_at` with a 180-day half-life, applied after
reranking. `confidence.py` computes an assertion-support level from the score
distribution rather than asking a model.

> **Measured.** On 500 questions the system failed in **both** calibration
> directions at once — 24 declines while holding complete evidence, 8 answers
> to unanswerable questions. A model's own certainty is a property of its
> tone, not of the data.

### 4.6 Rejected: entity bridging

Retrieve seeds, mine entities, look those entities up directly. The
motivating diagnosis was correct — on LoCoMo, multi-hop supporting turns sit a
median **204 turns apart** and **97%** are in different sessions, so only 2.2%
fall inside any window you could widen to.

The remedy was not.

> **Measured.** LoCoMo: 0.10 usable entities per turn, 91% of turns have none,
> 63 distinct entities across a 5,882-turn corpus — nothing to bridge with,
> 0 results attributed. LongMemEval: ~50 entities per session, so density was
> not the blocker, and it still produced **identical** full recall (0.888) at
> **2.6×** the latency (6,666ms vs 2,537ms) on n=500.

Kept as a documented negative and disabled by default.

### 4.7 Rejected as a default: MMR

Measured twice on conversational corpora: changed no retrieval metric while
costing 2.5× latency (61ms → 110ms on LoCoMo). Still the right tool for
corpora that genuinely accumulate restatements; not free enough to be a
default.

---

## 5. Read-time synthesis

Some answers exist in no single memory.

> **Measured motivation.** Retrieval delivers complete evidence for **97.2%**
> of LongMemEval questions, yet 45% of the remaining failures are COUNT
> questions — "how many bikes do I own?" answered "Multiple", with all three
> bikes in context. The model was never missing memory. It was being asked to
> be a calculator over 2,400 tokens of prose.

```
map     one extraction call per retrieved memory, in parallel
ground  drop any row whose quote is not in its source   (code, not a judge)
reduce  count / order / span computed in code; only COMPARE composes via the model
```

Grounding is the load-bearing part: a row survives only if its quote appears
verbatim in the source it claims to come from. That is a string operation, not
a judgement, and it is why a derived answer can be labelled `verified`.

### 5.1 Question shapes

`direct` · `count` · `order` · `date_arith` · `compare` · `advice` ·
`list_all`. Every kind fails open to the direct path, so a misclassification
is cheap.

`advice` exists because advice requests were failing for a structural reason:
a fact-lookup prompt sends the model hunting for a stored answer that never
existed, and it returns NO_ANSWER. The diagnosis came from reading our own
wrong outputs.

> **Methodology caveat, stated because it matters.** The ADVICE regexes were
> iterated against LongMemEval's 30 preference questions (16/30 → 29/30). No
> answers are encoded and no dataset label is read at runtime, but the tuning
> loop saw the test data, so *sensitivity* on that capability is optimistically
> biased. *Specificity* is independently validated: 0 false positives across
> 1,986 LoCoMo questions, a benchmark not consulted while writing the pattern.

---

## 6. Multi-tenancy and security

### 6.1 Three layers, and none of them trusts the others

| layer | mechanism | catches |
|---|---|---|
| authentication | `Bearer sm_...` → SHA-256 + pepper → `api_keys.key_hash` | forged credentials |
| authorization | scopes on the key; `require_scope` per route | over-broad keys |
| isolation | `org_id` in every query **and** Postgres RLS | a forgotten WHERE clause |

**Row-level security** (migration `0005`) is the backstop. Policies on
`spaces`, `memories`, `chunks`, `relation_edges`, `memory_versions` compare
`org_id` against a per-transaction GUC:

```sql
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE  ROW LEVEL SECURITY;   -- the app IS the owner
CREATE POLICY memories_tenant_isolation ON memories
  USING      (org_id = current_setting('app.org_id', true))
  WITH CHECK (org_id = current_setting('app.org_id', true));
```

`FORCE` matters: Postgres exempts a table's owner from its own policies, and
the application connects as the owner. Without it the migration would apply to
nobody and read as protection.

The GUC is set with `set_config(..., true)` — **transaction**-local. Sessions
come from a pool, and a session-level setting would outlive the request and
greet whichever tenant checked the connection out next. Unset means
`current_setting` returns NULL, every comparison is NULL, and every row is
filtered out — **fail-closed**. A store method that forgets to scope sees an
empty database, which is loud and local.

`WITH CHECK` matters as much as `USING`: without it a buggy path could INSERT
into another org while being unable to read one back.

Deliberately **not** under RLS: `api_keys` (authentication looks a key up by
hash *in order to discover* the org — there is no org context yet, and its
isolation is the unguessable secret), and `users` / `memberships` /
`organizations` (identity, which spans orgs by design).

### 6.2 404, never 403

A space in another organization is indistinguishable from one that does not
exist. `require_member` raises NotFound rather than Forbidden for the same
reason: the difference between the two status codes is an existence oracle.

### 6.3 Quotas

Rate limiting caps how *fast* a tenant calls. Quotas cap how much they
accumulate: `max_memories_per_org`, `max_bytes_per_org`, `max_writes_per_day`.
All default to 0 (unlimited) — a memory API shipping opinions about how much
its users may remember would be wrong for almost everyone.

Counted per **organization**, not per space, because a per-space limit is
escaped by creating another space. `402`, not `429`: the request was
well-formed and not too fast, and a client that retries a quota failure on a
backoff loop hammers the endpoint forever.

The usage query runs only when a limit is configured.

### 6.4 Rate limiting

Token bucket. In-memory for a single process; Redis with a Lua script for
multi-dyno, so the check-and-decrement is atomic. Startup validation warns
when `redis_url` is unset in production, because per-process limiting across
N dynos is an N× limit nobody asked for.

---

## 7. Storage backends

One abstract `MemoryStore`; two implementations.

- **`InMemoryStore`** — the reference. Brute force, no approximation, no
  index. It computes the exact answer and lets the ANN backend be the thing
  that approximates.
- **`PostgresStore`** — SQLAlchemy 2.0 async + asyncpg, pgvector for ANN,
  generated tsvector for full text.

They are kept honest by one parametrized conformance suite
(`tests/conformance/test_store_contract.py`) that runs against both. Without
it the backends drift — a filter meaning one thing in Python and another in
SQL, a cursor stable in one and not the other — and the difference surfaces as
a production bug no test reproduces.

Postgres specifics worth knowing:

- `websearch_to_tsquery`, not `plainto_` or `to_tsquery` — the latter two
  raise on characters users absolutely type.
- `SET LOCAL hnsw.iterative_scan = 'relaxed_order'` so filtered ANN searches
  do not return short. Filters ride *inside* the ANN statement; filtering
  after a top-k scan lets a restrictive filter return nothing.
- Keyset pagination on `(status, id)`, never OFFSET.
- Erasure bridges supersession chains before deleting. A supersedes B
  supersedes C: removing B severs the only path from A to C, and C — a fact
  the user explicitly replaced — resurfaces as current. Found live, on a
  Portland → Austin → Seattle chain that brought Portland back.

---

## 8. Deployment

```
Heroku dyno
  ├─ bin/with-cloudsql ──▶ cloud-sql-proxy ══TLS/IAM══▶ Cloud SQL  mapi-db
  │    web:     uvicorn                                  Postgres 16.14
  │    release: alembic upgrade head                     pgvector 0.8.1
  │         (both wrapped — a release dyno is                db-custom-2-7680
  │          a separate container)                          20GB SSD, auto-grow
  │                                                         PITR · backup 07:00
  ├──── REDIS_URL ────▶ rate limiter                        us-central1
  └──── Vertex AI (ADC) ────▶ embeddings · extraction · understanding
```

**Why a proxy rather than an authorized network.** Heroku dynos have no stable
outbound address, so reaching Cloud SQL over its public IP means authorizing
`0.0.0.0/0` and relying on TLS plus the password alone. The Auth Proxy
authenticates with the service account's IAM identity and listens on
localhost, so the application needs no network authorization at all: reaching
the database requires an IAM principal holding `roles/cloudsql.client`, and a
leaked password alone is not sufficient. That is a property an
authorized-network setup cannot provide.

The authorized-networks list is not empty — it holds developer `/32`s so the
conformance suite can run from a laptop — but it never contains `0.0.0.0/0`,
and nothing in the deployed path depends on it.

`bin/with-cloudsql` waits for the listener rather than sleeping a fixed
interval — the release phase runs `alembic upgrade head` immediately, and a
migration that starts one second early fails the deploy with a connection
error that reads like a configuration problem. It also exits with the proxy's
status if the proxy dies, instead of polling a port nothing will ever bind.

The proxy binary is fetched at **build** time by `bin/post_compile`, not at
boot: a 30MB download on every dyno start adds latency to every restart and
makes boot depend on GitHub being reachable.

Credentials are Application Default Credentials. `core/gcp.py` materialises
`GOOGLE_APPLICATION_CREDENTIALS_JSON` into a 0600 file at boot and points
`GOOGLE_APPLICATION_CREDENTIALS` at it — ADC is a *resolution order*, not a
single mechanism, so a service-account file works where `gcloud` does not
exist.

`config.py` maps platform-injected `DATABASE_URL` / `REDIS_URL` and rewrites
the `postgres://` scheme to `postgresql+asyncpg://`. It is a **before**
validator: an after-validator cannot return a rebuilt model when the object is
constructed through `__init__`, and pydantic warns rather than raising — so
the adoption silently did nothing.

Production readiness is validated at startup rather than assumed: default
pepper, `debug_errors`, non-Postgres store, deterministic embeddings,
bootstrap key set, missing Redis.

---

## 9. The SDK

`sdk/` is a separate distribution (`mapi-sdk` on PyPI). httpx and the standard
library, nothing else — every additional dependency is a version conflict in
somebody else's application.

It **never imports the server**, and that is a test rather than a convention,
because the failure is silent: someone reaches for one shared helper and the
next release carries the retrieval pipeline, the prompts and the benchmark
harness onto PyPI. Four guarantees are asserted by parsing the import graph
with `ast` (not regex — a docstring beginning "from it, since…" reads as an
import of a module named `it`), including one that inspects the **built wheel
and sdist**, because packaging config is its own failure surface.

Sync and async clients share one set of resource classes handed a request
callable. Two hand-written implementations drift, and the async one always
drifts last and silently.

---

## 10. Failure modes

The system is designed to degrade rather than fail. What happens when each
dependency breaks:

| broken | result |
|---|---|
| embedding provider | lexical-only search; `strategies` records it |
| reranker | first-stage order; `rerank_degraded: true` |
| query understanding | regex classification (the prior behaviour) |
| contradiction adjudication | lexical signals only, no invented conflicts |
| extraction | the original text is stored exactly as before |
| Redis | per-process rate limiting, with a startup warning |
| Postgres | the request fails — this one has no degraded mode |

The asymmetry is deliberate. Everything on the enrichment path fails open,
because a worse answer beats no answer. The one thing that fails **closed** is
contradiction adjudication, because its failure would write false claims into
the graph.
