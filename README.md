# Mapi

A production-grade memory API for AI agents. Hybrid retrieval, belief revision,
and LLM reranking behind a typed, multi-tenant HTTP service.

[![ci](https://github.com/krishnabhatnagar/mapi/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

```bash
make install && make demo     # no Docker, no database, no API keys
```

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
  information an attacker can use.
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
