# HNSW recall: the gap every other number in `bench/` hides

`bench/run.py` ingests into `InMemoryStore` and searches it by brute force, so
every retrieval figure this repo reports — `full_recall@k = 0.968` at k=10,
`0.998` at k=25, the whole five-config ablation — is **exact** nearest
neighbour. Production runs Postgres with an HNSW index. Exact search cannot
express an approximate index's error, so those numbers are an upper bound on
production rather than a measurement of it.

This measures the gap directly.

## What it runs against

pgvector compiled to WASM (`@electric-sql/pglite` + `@electric-sql/pglite-pgvector`),
because this machine has no Docker, no Homebrew and no system Postgres. It is
the same pgvector C code and the same index parameters the migration creates
(`m = 16, ef_construction = 64`), so **recall** transfers; **latency does not**,
and the millisecond figures below are useful only as a ratio.

Vectors are 20,000 real `text-embedding-004` embeddings pulled from
`bench/data/cache/vectors.sqlite`, not random noise — HNSW recall depends
heavily on how the data is distributed, and random vectors would flatter it.

## Result

Asking for k=60, which is what the pipeline actually requests
(`limit=10 × candidate_multiplier=6`):

    ef_search  iterative_scan   recall@60   full_recall@60   median ms
       40         off             66.67%         0.0%           2.5
       40      relaxed_order      99.43%        82.0%           2.5
      120 (2x)  relaxed_order     99.82%        94.0%           2.6
      180 (3x)  relaxed_order     99.96%        98.0%           3.1
      240 (4x)  relaxed_order     99.98%        99.0%           3.5

Three things fall out of that table.

**`ef_search` was never set, so it was 40.** pgvector cannot return more rows
than its search list holds: at ef_search=40 a request for 60 gets exactly 40,
which is the 66.67% and the 0% full recall on the first row. That is a hard
structural limit, not a tuning preference.

**`iterative_scan` was already rescuing it.** Production sets
`relaxed_order`, which keeps pulling from the index until the limit is
satisfied, and it recovers the row count and 99.43% recall. So the damage was
never the 66.67% the first row suggests — it was `full_recall` sitting at 82%,
one query in five carrying an incomplete candidate set.

**Conjunctive metrics are why that matters.** `full_recall@k` is 1.0 only when
every evidence item survives, so a candidate set missing one member of a pair
loses the whole question. Sizing `ef_search` to 3× the request moves that from
82% to 98% for half a millisecond, on a pipeline that already spends over a
second.

## Caveats

* WASM, not server Postgres. Recall should be identical; latency is not.
* **Unfiltered.** Every real query here is scoped to an org and a space, and
  filtered ANN is HNSW's hard case — it is the reason `iterative_scan` is set
  at all. The filtered gap is likely worse than these numbers and is still
  unmeasured.
* 20,000 rows. Production is larger, and HNSW recall degrades slowly with
  corpus size.

## Running it

    cd bench/hnsw
    npm install @electric-sql/pglite @electric-sql/pglite-pgvector
    # export vectors from the embedding cache first (see recall.mjs header)
    node recall.mjs
    node recall_iterative.mjs
