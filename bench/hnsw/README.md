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

## The filtered case

Every real query is scoped to an org and a space, so the numbers above are the
easy half. Adding `WHERE space_id = $1` and a btree index on it, same corpus,
same k=60:

    space share   ef    iterative        recall@60   full_recall   rows
    1%            any   any               100.00%      100.0%      60/60
    5%            any   any               100.00%      100.0%      60/60
    25%           any   any               100.00%      100.0%      60/60
    50%           40    off                32.77%        0.0%      19.7/60
    50%           40    relaxed_order      99.25%       71.0%      60/60
    50%           180   relaxed_order      99.47%       78.0%      60/60

**Postgres will not use HNSW when the filter is selective**, and that is the
whole story. `EXPLAIN` at 1% shows `Limit → Sort` over a btree hit on
`space_id`: it pulls the space's rows and orders them exactly, so recall is
100% and the index is never consulted. Only at 50% — where the filter stops
paying for itself — does the planner switch to `Index Scan using
chunks_embedding_idx`, and there the filtered case is WORSE than the unfiltered
one: 78% full recall at ef=180 against 98% without a filter.

So the exposure is not "filters break HNSW". It is a DOMINANT SPACE: a
single-tenant deployment, or one space holding most of the table, where the
scope predicate no longer discriminates. Multi-tenant traffic with many spaces
lands on the exact path and never sees an approximation.

This is also where `iterative_scan` is load-bearing rather than merely helpful:
without it, a filtered query at ef=40 returned 19.7 rows of a requested 60 and
32.77% recall.

Caveat on the crossover: these are 20,000 rows, and the planner chooses on
COST, not on a fixed share. The 25%/50% boundary here is not a constant to rely
on — a much larger table moves it.
