# Mapi

A production-grade memory API for AI agents. Hybrid retrieval, belief revision,
and LLM reranking behind a typed, multi-tenant HTTP service.

[![ci](https://github.com/kbhatnagar1506/mapi/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.13-blue.svg)](pyproject.toml)
[![mypy](https://img.shields.io/badge/mypy-strict-brightgreen.svg)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

```bash
make install && make demo     # no Docker, no database, no API keys
```

**82.4%** on LongMemEval's 500 questions (independent judge, official protocol) ·
**0.968** full-recall@10 · 1320 tests · strict mypy · [how it was measured](#measurement)

---

## Why this is not another RAG wrapper

Most "memory" APIs are a vector index with a REST façade. That design fails on
contact with real usage, in three specific ways this service addresses.

**1. Facts go stale, and appending does not fix that.**
Ask a document index "how do we deploy?" after a team migrates Jenkins →
CircleCI → GitHub Actions, and it returns all three with no signal about which
is current. Memories here form a typed graph — `supersedes`, `contradicts`,
`derived_from`, `references` — stored as indexed edges, and retrieval suppresses
a memory when something that **transitively** supersedes it is present in the
results. Transitively matters: with a one-hop check, the two-hops-stale
"Jenkins" answer survives as though it were current. That condition — *present
in the results* — matters too: hiding the old fact when the new one did not
match the query would answer with silence, which is worse than answering with
something stale.

Because the graph is indexed in both directions, you can also just ask:
`GET /memories/{id}/lineage` → `{"is_current": false, "head": "mem_…"}`.

**1b. And you can ask what you believed last week.**
Every write records a version snapshot, with system time (`valid_from`/
`valid_to`) kept separate from event time (`occurred_at`). So
`GET /memories/{id}?as_of=2026-03-01T00:00:00Z` returns the memory as the
database understood it then — not what is true now.

**2. Dense vectors alone miss exactly what you search for most.**
Embeddings are poor at exact tokens — an order number, `error TS2345`, a
surname. Lexical search is poor at paraphrase. This fuses both with Reciprocal
Rank Fusion, which combines *ranks* rather than scores and so avoids the
unsolvable problem of calibrating BM25 against cosine similarity.

**3. Top-k returns the same fact five times.**
A corpus that accumulates over months is full of near-duplicates. Maximal
Marginal Relevance trades relevance against redundancy so the agent's context
window is not spent on restatements of one thing.

Every result carries its full score provenance:

```json
{
  "score": 0.8022,
  "vector_score": 0.71, "lexical_score": 2.14,
  "fusion_score": 0.0328, "rerank_score": 0.80, "recency_factor": 1.0,
  "explain": ["lexical rank 1", "vector rank 1", "fused 0.03279",
              "heuristic rerank rank 1", "recency x1.000"]
}
```

---

## The retrieval pipeline

```
query
  ├─ embed (query-side task type)
  ├─ vector search  ─┐
  ├─ lexical search ─┘  concurrent
  ├─ reciprocal rank fusion
  ├─ hydrate
  ├─ rerank            heuristic (offline) or LLM (listwise)
  ├─ recency decay     exponential, half-life configurable, with a floor
  ├─ supersession      suppress facts a present result replaced
  ├─ MMR               diversify against redundancy
  └─ top-k
```

Stage order is deliberate and documented in
[`pipeline.py`](src/mapi/domain/retrieval/pipeline.py). Two examples:
fusion runs *before* reranking so the reranker sees candidates either strategy
liked; reranking runs *before* decay because a reranker judges topical
relevance and has no idea how old anything is.

| Component | Choice | Why |
|---|---|---|
| Fusion | Reciprocal Rank Fusion (k=60) | Combines ranks, so BM25 and cosine need no shared scale |
| Lexical | BM25 / Postgres `ts_rank_cd` | Saturating term frequency, length normalization |
| Vector | pgvector HNSW, cosine | No training step, unlike IVFFlat, so it is correct on an empty table |
| Rerank | LLM listwise, or offline heuristic | Precision at the top; degrades to first-stage order on any failure |
| Decay | Exponential with a floor | Old-but-unique memories stay findable |
| Diversity | MMR (λ = 0.7) | Suppresses near-duplicate results |
| Staleness | Transitive graph walk | One hop is not enough; see ARCHITECTURE §4 |

### Indexing

| Data | Postgres | In-memory reference |
|---|---|---|
| Vectors | HNSW (`vector_cosine_ops`, m=16, ef=64) | Exact cosine scan |
| Full text | Generated `tsvector` + GIN | BM25 (k1=1.2, b=0.75) |
| Graph edges | btree on `(org, space, source, type)` **and** `(…, target, type)` | Dicts keyed both directions |
| Versions | `(org, space, memory, valid_from)` + partial index on the open row | Per-memory list |
| Tags / metadata | GIN on `tags`, GIN on `meta` JSONB | Linear filter |
| Listing | `(org, space, status, id)` for cursor pagination | Sorted ids |

---

## Architecture

```
src/mapi/
  api/            HTTP: schemas, dependencies, middleware, v1 routes
  core/           ids, errors, logging, metrics, security, rate limiting
  domain/         entities, chunking, text analysis, embeddings, retrieval,
                  consolidation      ← no HTTP, no SQL, no framework
  store/          the storage port + two implementations
  service.py      workflow layer between API and domain
```

**Storage is a port with two real implementations.** In-memory runs exact cosine
k-NN and a real BM25 index; PostgreSQL runs HNSW and `tsvector`. One
[conformance suite](tests/conformance/test_store_contract.py) runs against both,
which is what makes having two safe — without it they drift, and the difference
surfaces as a production bug no test reproduces.

That design also means the whole service runs with **zero infrastructure** for
development and CI, while production uses Postgres through the same code path.

**Tenancy is enforced in the storage port**, not in route handlers. Every method
takes `org_id` and `space_id` and filters on both, so a leak has to be written
deliberately rather than by forgetting a `WHERE` clause in one handler.

---

## Quick start

### No infrastructure

```bash
make install
make demo         # seeds a corpus, runs a query, prints score provenance
make run          # http://localhost:8000/docs
```

### With Postgres and Redis

```bash
docker compose up --build
curl -H "Authorization: Bearer sm_local_dev_key_do_not_use_in_prod" \
     http://localhost:8000/v1/spaces
```

### Real embeddings

Vertex AI via Application Default Credentials — no key material in the
environment:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project
export MAPI_EMBEDDING_BACKEND=gemini
export MAPI_RERANK_BACKEND=llm
```

Or `GEMINI_API_KEY` / `OPENAI_API_KEY` for the key-based APIs.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/spaces` | Create a namespace |
| `POST` | `/v1/spaces/{id}/memories` | Ingest (chunk, embed, dedupe) |
| `POST` | `/v1/spaces/{id}/memories/bulk` | Up to 100 at once |
| `POST` | `/v1/spaces/{id}/search` | Hybrid search |
| `POST` | `/v1/multi-search` | One question, up to 4 spaces, one embedding |
| `POST` | `/v1/spaces/{id}/similar` | Neighbours of a stored memory, no embedding |
| `GET` | `/v1/spaces/{id}/memories` | Cursor-paginated listing |
| `GET` | `/v1/spaces/{id}/memories/{id}?as_of=…` | Point-in-time read |
| `POST` | `/v1/spaces/{id}/memories/{id}/relations` | Supersede, contradict, link |
| `GET` | `/v1/spaces/{id}/memories/{id}/relations?direction=in\|out` | Graph edges, either direction |
| `GET` | `/v1/spaces/{id}/memories/{id}/lineage` | Is this stale, and what replaced it |
| `GET` | `/v1/spaces/{id}/memories/{id}/versions` | Full version history |
| `POST` | `/v1/keys` | Mint a scoped API key |
| `GET` | `/health`, `/ready`, `/metrics` | Liveness, readiness, Prometheus |

Asking whether a fact is still current:

```bash
curl localhost:8000/v1/spaces/$SPACE/memories/$ID/lineage -H "Authorization: Bearer $KEY"
# {"is_current": false, "successors": ["mem_…", "mem_…"], "head": "mem_…"}
```

```bash
curl -X POST localhost:8000/v1/spaces/$SPACE/search \
  -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"query": "how do we deploy?", "limit": 5, "explain": true}'
```

Full OpenAPI at `/docs`.

---

## Production concerns

- **Auth** — API keys are never stored; only a peppered SHA-256 is, and lookup
  is a single indexed read. Scopes are enforced per route. Unknown, revoked and
  expired keys return identical responses, because the difference is
  information an attacker can use. A resolved key is reused for
  `MAPI_AUTH_CACHE_TTL_S` (30 s) and `last_used_at` is written at most once a
  minute; a revoke through the API evicts at once, one made by another process
  is honoured within the TTL.
- **Errors** — RFC 9457 `application/problem+json` with a stable machine slug,
  the offending field, and a request id. Internal detail never reaches a client.
- **Rate limiting** — token bucket, Redis-backed and atomic via Lua so replicas
  share one budget. Fails *open*: a limiter outage must not become a service
  outage.
- **Observability** — structured logs with request correlation and automatic
  secret scrubbing; Prometheus metrics labelled by route *template*, never by
  resolved path, so a memory id cannot explode cardinality.
- **Config safety** — `validate_production()` refuses to boot production with a
  default pepper, debug errors, non-semantic embeddings, or a bootstrap key.
- **Liveness vs readiness** are separate endpoints. Conflating them turns a
  brief database blip into a cluster-wide restart loop.

---

## Measurement

Retrieval claims are cheap to make and cheap to check, so this repo checks them.
`bench/` runs the service against public benchmarks; `bench/lab/` is a
purpose-built experiment harness for the questions the public sets cannot answer.

### LongMemEval — 500 questions, 23,867 documents

Retrieval is scored with **no LLM in the loop**, so the numbers carry no judge
variance and are exactly reproducible.

| config | full_recall@10 | hit@10 | MRR |
|---|---|---|---|
| `vector_only` | **0.968** | 0.994 | 0.938 |
| `hybrid_rrf` | 0.952 | 0.996 | 0.945 |
| `lexical_only` | 0.916 | 0.984 | 0.909 |

`full_recall@k` is the headline, not `hit@k`: it is 1.0 only when **every**
evidence session is retrieved. 324 of the 500 questions need two or more
sessions, so answering a three-hop question with one of three facts is a wrong
answer that `hit@k` reports as a success.

End-to-end (LongMemEval's published metric), answering with `gemini-2.5-flash`:

**QA accuracy 82.4%** (95% CI 78.7–85.6, n=467) — **86.4%** with `gemini-2.5-pro`.

> The judge is deliberately **not** the answering model. Grading your own output
> is a known self-preference bias, and the largest published gap on this
> benchmark (94.4% self-reported vs 49.0% independently measured) is attributed
> to judge configuration rather than to the memory engine. Judge and protocol
> are recorded in every result file.

### The lab — measuring what benchmarks cannot

Public sets are pass/fail on a fixed corpus; they cannot tell you *why* a
capability fails or whether a proposed fix works. `bench/lab/` adds nine
authored personas, 177 gated questions across five preference kinds, and
**paired significance testing** (exact McNemar) on every arm.

Two results worth the space, both of which changed the system:

**Retrieval was the bottleneck, until it wasn't.** Sweeping the context budget
from k=4 to the whole corpus:

```
k=4  115/177     k=8  135/177     k=16 152/177
k=6  126/177     k=12 141/177     k=23 157/177   ← retrieval switched off
```

Monotonic to the top — on a 23-memory space the ranker is *losing* evidence, not
sorting it. The useful part is the ceiling arm: with every memory in the window,
`explicit` 35/36, `implicit` 34/35 and `updated` 36/36 are solved outright, while
`composed` sits at 23/35. Three kinds were retrieval problems. Two were not.

**So the remaining failures got a synthesis fix, with a control.** An
instruction that enumerates every constraint, checks the draft against each, then
commits to one recommendation:

| arm | correct | paired vs baseline |
|---|---|---|
| baseline | 159/177 | — |
| enumerate-then-answer | **169/177** | 13-3, p=0.021 |
| answer-then-verify (2 calls) | 169/177 | 10-0, p=0.002 |

The second model call buys nothing over the first (3-3, p=1.000), so the
one-call version shipped. Four earlier candidates — write-time claim extraction,
multi-query retrieval, a write-time domain index, and width itself — were tested
the same way and **declined**, each because a plain width control matched it.
Those nulls are committed with their numbers; a lab that only records wins is
not measuring, it is advertising.

---

## Testing

```bash
make check     # lint + strict mypy + full suite
```

- **Unit** — algorithms in isolation, heavy on edge cases: empty input, CJK and
  emoji, NUL bytes, 500KB documents, malformed LLM output, dimension
  mismatches, zero vectors, clock skew.
- **Conformance** — one suite over both storage backends. Postgres runs in CI.
- **End-to-end** — the real ASGI app: auth, scopes, cross-tenant isolation,
  pagination, concurrency, the error contract.

Strict `mypy` passes across the package with no escape hatches beyond three
documented third-party stub gaps.

A few things the tests caught during development, kept here because they are the
interesting part: chunking silently dropped content when a document repeated
itself; `parse_authorization("Bearer   ")` returned the literal string `Bearer`
as the key; the in-memory BM25 index indexed stopwords and did not stem, so
"how do we **deploy**?" ranked an unrelated memory first *and* missed the one
document that answered it — a real divergence from Postgres's `english` text
search configuration, now pinned by [`test_text.py`](tests/unit/test_text.py);
and supersession suppression only ever looked one hop, so a fact two revisions
out of date was returned as current
([`test_multi_hop_supersession_hides_every_stale_revision`](tests/e2e/test_api.py)).

---

## Design notes

Longer rationale in [ARCHITECTURE.md](ARCHITECTURE.md): the ordering of pipeline
stages, why supersession is opt-in rather than automatic, the HNSW-over-IVFFlat
decision, and the prompt-injection handling in the LLM reranker.

## License

MIT
