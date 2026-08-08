# LoCoMo benchmark — locomo-full

Corpus: **5882 dialogue turns** across 10 conversations, one memory per turn, one space per conversation.  
Questions: **200 sampled** (stratified by category) from 1986.  
Embeddings: `text-embedding-004` (768d, gemini).  
Retrieval depth k = 10. Seed 20260808.

Latency below is **retrieval only**: the query-embedding cache is warmed before timing, so every config is measured on equal footing. Query embedding costs a further 88 ms per query uncached, and is reported separately because it is identical across configs and would otherwise swamp the comparison.

Retrieval metrics are scored against LoCoMo's labelled evidence turns — no LLM, so they carry no judge variance and are exactly reproducible. Adversarial (unanswerable) questions are excluded from retrieval metrics, since they have no evidence to retrieve by construction.

## Retrieval ablation

| config | recall@10 | hit@1 | hit@10 | MRR | nDCG@10 | latency ms |
|---|---|---|---|---|---|---|
| `vector_only` | 0.630 | 0.355 | 0.716 | 0.460 | 0.475 | 53.2 |
| `lexical_only` | 0.583 | 0.345 | 0.670 | 0.443 | 0.448 | 47.8 |
| `hybrid_rrf` | 0.629 | 0.340 | 0.721 | 0.465 | 0.480 | 48.4 |
| `hybrid_rerank_heuristic` | 0.626 | 0.381 | 0.721 | 0.482 | 0.485 | 50.2 |
| `hybrid_mmr` | 0.620 | 0.340 | 0.711 | 0.464 | 0.477 | 110.5 |
| `hybrid_decay` | 0.636 | 0.345 | 0.721 | 0.466 | 0.483 | 55.6 |
| `full_no_llm` | 0.606 | 0.335 | 0.706 | 0.452 | 0.463 | 109.6 |

### Against the hybrid baseline

| config | Δ MRR | Δ recall | verdict |
|---|---|---|---|
| `vector_only` | -0.005 | +0.001 | no effect |
| `lexical_only` | -0.022 | -0.046 | hurts |
| `hybrid_rerank_heuristic` | +0.017 | -0.003 | helps |
| `hybrid_mmr` | -0.001 | -0.009 | no effect |
| `hybrid_decay` | +0.001 | +0.006 | no effect |
| `full_no_llm` | -0.014 | -0.024 | hurts |

### By question category

| config | adversarial | multi_hop | open_domain | single_hop | temporal |
|---|---|---|---|---|---|
| `vector_only` | 0.259 | 0.547 | 0.331 | 0.586 | 0.568 |
| `lexical_only` | 0.490 | 0.414 | 0.251 | 0.502 | 0.541 |
| `hybrid_rrf` | 0.376 | 0.513 | 0.306 | 0.565 | 0.553 |
| `hybrid_rerank_heuristic` | 0.472 | 0.560 | 0.252 | 0.542 | 0.568 |
| `hybrid_mmr` | 0.369 | 0.521 | 0.302 | 0.566 | 0.549 |
| `hybrid_decay` | 0.387 | 0.510 | 0.319 | 0.566 | 0.537 |
| `full_no_llm` | 0.443 | 0.453 | 0.242 | 0.525 | 0.579 |

Cells are MRR. Category counts: adversarial=40, multi_hop=40, open_domain=37, single_hop=40, temporal=40

## End-to-end

Config `hybrid_rerank_heuristic` (best MRR). Answering model `gemini-2.5-flash`, judge `gemini-2.5-flash`.

**MemScore: 39% / 80ms / 594tok**

| metric | value |
|---|---|
| accuracy (answerable) | 38.8% (n=160) |
| adversarial decline rate | 0.975 (n=40) |
| retrieval hit@10 | 0.694 |
| mean search latency | 79.7 ms |
| mean context tokens | 594 |
| errors | 0 |

Accuracy by category:

| category | n | accuracy |
|---|---|---|
| adversarial | 40 | 97.5% |
| multi_hop | 40 | 30.0% |
| open_domain | 40 | 20.0% |
| single_hop | 40 | 75.0% |
| temporal | 40 | 30.0% |

The adversarial rate is a *decline* rate: the correct behaviour on an unanswerable question is to answer `NO_ANSWER`. It is scored without a judge and kept out of the headline accuracy, because a system that confidently answers the unanswerable should not be rewarded for it.

## What this is and is not

This is **not** a run of the `supermemoryai/memorybench` repository. That harness requires `bun` and hosted provider API keys (`SUPERMEMORY_API_KEY`, `MEM0_API_KEY`, `ZEP_API_KEY`), none of which were available. This re-implements its pipeline shape — ingest, index, search, answer, evaluate, report — against this service, with a different judge model.

So these numbers are comparable **in kind** to a MemoryBench leaderboard, not head-to-head with it. A different judge, a different answering model, a question subset and a re-implemented prompt all move absolute accuracy.

Every number here came from a real API response. Failed calls are counted in `n_errors` and excluded rather than substituted.
