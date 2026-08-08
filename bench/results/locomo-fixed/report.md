# LoCoMo benchmark — locomo-fixed

Corpus: **5882 dialogue turns** across 10 conversations, one memory per turn, one space per conversation.  
Questions: **150 sampled** (stratified by category) from 1986.  
Embeddings: `text-embedding-004` (768d, gemini).  
Retrieval depth k = 10. Seed 20260808.

Latency below is **retrieval only**: the query-embedding cache is warmed before timing, so every config is measured on equal footing. Query embedding costs a further 150 ms per query uncached, and is reported separately because it is identical across configs and would otherwise swamp the comparison.

Retrieval metrics are scored against LoCoMo's labelled evidence turns — no LLM, so they carry no judge variance and are exactly reproducible. Adversarial (unanswerable) questions are excluded from retrieval metrics, since they have no evidence to retrieve by construction.

## Retrieval ablation

| config | recall@10 | hit@1 | hit@10 | MRR | nDCG@10 | latency ms |
|---|---|---|---|---|---|---|
| `vector_only` | 0.644 | 0.345 | 0.723 | 0.456 | 0.478 | 27.1 |
| `lexical_only` | 0.575 | 0.338 | 0.655 | 0.435 | 0.443 | 26.3 |
| `hybrid_rrf` | 0.637 | 0.338 | 0.730 | 0.464 | 0.484 | 30.1 |
| `hybrid_rerank_heuristic` | 0.619 | 0.399 | 0.716 | 0.488 | 0.485 | 28.1 |
| `hybrid_mmr` | 0.622 | 0.338 | 0.716 | 0.463 | 0.478 | 61.7 |
| `hybrid_decay` | 0.639 | 0.338 | 0.730 | 0.462 | 0.483 | 27.3 |
| `full_no_llm` | 0.598 | 0.345 | 0.710 | 0.454 | 0.460 | 64.1 |
| `hybrid_hyde` | 0.648 | 0.358 | 0.737 | 0.472 | 0.494 | 2496.0 |
| `hybrid_hyde_rerank` | 0.625 | 0.392 | 0.716 | 0.484 | 0.485 | 1945.1 |

### Against the hybrid baseline

| config | Δ MRR | Δ recall | verdict |
|---|---|---|---|
| `vector_only` | -0.008 | +0.006 | no effect |
| `lexical_only` | -0.029 | -0.062 | hurts |
| `hybrid_rerank_heuristic` | +0.024 | -0.019 | helps |
| `hybrid_mmr` | -0.002 | -0.016 | no effect |
| `hybrid_decay` | -0.002 | +0.001 | no effect |
| `full_no_llm` | -0.010 | -0.039 | hurts |
| `hybrid_hyde` | +0.008 | +0.011 | no effect |
| `hybrid_hyde_rerank` | +0.020 | -0.012 | helps |

### By question category

| config | adversarial | multi_hop | open_domain | single_hop | temporal |
|---|---|---|---|---|---|
| `vector_only` | 0.314 | 0.532 | 0.301 | 0.559 | 0.566 |
| `lexical_only` | 0.426 | 0.405 | 0.239 | 0.529 | 0.561 |
| `hybrid_rrf` | 0.382 | 0.457 | 0.309 | 0.566 | 0.596 |
| `hybrid_rerank_heuristic` | 0.413 | 0.561 | 0.257 | 0.569 | 0.624 |
| `hybrid_mmr` | 0.372 | 0.467 | 0.304 | 0.561 | 0.596 |
| `hybrid_decay` | 0.397 | 0.452 | 0.309 | 0.568 | 0.575 |
| `full_no_llm` | 0.379 | 0.440 | 0.244 | 0.551 | 0.640 |
| `hybrid_hyde` | 0.422 | 0.467 | 0.234 | 0.574 | 0.649 |
| `hybrid_hyde_rerank` | 0.412 | 0.554 | 0.248 | 0.566 | 0.622 |

Cells are MRR. Category counts: adversarial=30, multi_hop=30, open_domain=28, single_hop=30, temporal=30

## End-to-end

Config `hybrid_rerank_heuristic` (best MRR). Answering model `gemini-2.5-flash`, judge `gemini-2.5-flash`.

**MemScore: 49% / 54ms / 710tok**

| metric | value |
|---|---|
| accuracy (answerable) | 49.2% (n=120) |
| adversarial decline rate | 0.0 (n=30) |
| retrieval hit@10 | 0.700 |
| mean search latency | 53.7 ms |
| mean context tokens | 710 |
| errors | 0 |

Accuracy by category:

| category | n | accuracy |
|---|---|---|
| adversarial | 30 | 0.0% |
| multi_hop | 30 | 36.7% |
| open_domain | 30 | 36.7% |
| single_hop | 30 | 63.3% |
| temporal | 30 | 60.0% |

The adversarial rate is a *decline* rate: the correct behaviour on an unanswerable question is to answer `NO_ANSWER`. It is scored without a judge and kept out of the headline accuracy, because a system that confidently answers the unanswerable should not be rewarded for it.

## What this is and is not

This is **not** a run of the `supermemoryai/memorybench` repository. That harness requires `bun` and hosted provider API keys (`SUPERMEMORY_API_KEY`, `MEM0_API_KEY`, `ZEP_API_KEY`), none of which were available. This re-implements its pipeline shape — ingest, index, search, answer, evaluate, report — against this service, with a different judge model.

So these numbers are comparable **in kind** to a MemoryBench leaderboard, not head-to-head with it. A different judge, a different answering model, a question subset and a re-implemented prompt all move absolute accuracy.

Every number here came from a real API response. Failed calls are counted in `n_errors` and excluded rather than substituted.
