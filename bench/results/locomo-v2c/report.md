# LoCoMo benchmark — locomo-v2c

Corpus: **5882 dialogue turns** across 10 conversations, one memory per turn, one space per conversation.  
Questions: **496 sampled** (stratified by category) from 1986.  
Embeddings: `text-embedding-004` (768d, gemini).  
Retrieval depth k = 10. Seed 20260808.

Latency below is **retrieval only**: the query-embedding cache is warmed before timing, so every config is measured on equal footing. Query embedding costs a further 66 ms per query uncached, and is reported separately because it is identical across configs and would otherwise swamp the comparison.

Retrieval metrics are scored against LoCoMo's labelled evidence turns — no LLM, so they carry no judge variance and are exactly reproducible. Adversarial (unanswerable) questions are excluded from retrieval metrics, since they have no evidence to retrieve by construction.

## Retrieval ablation

| config | recall@10 | hit@1 | hit@10 | MRR | nDCG@10 | latency ms |
|---|---|---|---|---|---|---|
| `vector_only` | 0.607 | 0.323 | 0.691 | 0.434 | 0.450 | 27.6 |
| `lexical_only` | 0.554 | 0.323 | 0.634 | 0.415 | 0.426 | 30.4 |
| `hybrid_rrf` | 0.589 | 0.323 | 0.679 | 0.443 | 0.454 | 28.5 |
| `hybrid_rerank_heuristic` | 0.595 | 0.364 | 0.685 | 0.465 | 0.468 | 29.9 |
| `hybrid_mmr` | 0.593 | 0.323 | 0.683 | 0.444 | 0.455 | 62.9 |
| `hybrid_decay` | 0.592 | 0.317 | 0.681 | 0.439 | 0.452 | 28.4 |
| `full_no_llm` | 0.585 | 0.337 | 0.673 | 0.441 | 0.451 | 63.2 |

### Against the hybrid baseline

| config | Δ MRR | Δ recall | verdict |
|---|---|---|---|
| `vector_only` | -0.010 | +0.018 | no effect |
| `lexical_only` | -0.029 | -0.035 | hurts |
| `hybrid_rerank_heuristic` | +0.022 | +0.006 | helps |
| `hybrid_mmr` | +0.000 | +0.004 | no effect |
| `hybrid_decay` | -0.005 | +0.003 | no effect |
| `full_no_llm` | -0.002 | -0.004 | no effect |

### By question category

| config | adversarial | multi_hop | open_domain | single_hop | temporal |
|---|---|---|---|---|---|
| `vector_only` | 0.179 | 0.520 | 0.309 | 0.618 | 0.532 |
| `lexical_only` | 0.475 | 0.309 | 0.230 | 0.548 | 0.496 |
| `hybrid_rrf` | 0.304 | 0.479 | 0.266 | 0.607 | 0.547 |
| `hybrid_rerank_heuristic` | 0.479 | 0.457 | 0.262 | 0.570 | 0.540 |
| `hybrid_mmr` | 0.302 | 0.482 | 0.265 | 0.607 | 0.548 |
| `hybrid_decay` | 0.307 | 0.461 | 0.271 | 0.605 | 0.535 |
| `full_no_llm` | 0.462 | 0.382 | 0.266 | 0.546 | 0.535 |

Cells are MRR. Category counts: adversarial=100, multi_hop=100, open_domain=92, single_hop=100, temporal=100

## End-to-end

Config `hybrid_rerank_heuristic` (best MRR). Answering model `gemini-2.5-flash`, judge `gemini-2.5-flash`.

**MemScore: 60% / 49ms / 1832tok**

| metric | value |
|---|---|
| accuracy (answerable) | 59.9% (n=396) |
| adversarial decline rate | 0.87 (n=100) |
| retrieval hit@10 | 0.669 |
| mean search latency | 49.4 ms |
| mean context tokens | 1832 |
| errors | 0 |

Accuracy by category:

| category | n | accuracy |
|---|---|---|
| adversarial | 100 | 87.0% |
| multi_hop | 100 | 38.0% |
| open_domain | 96 | 51.0% |
| single_hop | 100 | 85.0% |
| temporal | 100 | 65.0% |

The adversarial rate is a *decline* rate: the correct behaviour on an unanswerable question is to answer `NO_ANSWER`. It is scored without a judge and kept out of the headline accuracy, because a system that confidently answers the unanswerable should not be rewarded for it.

## What this is and is not

This is **not** a run of the `supermemoryai/memorybench` repository. That harness requires `bun` and hosted provider API keys (`SUPERMEMORY_API_KEY`, `MEM0_API_KEY`, `ZEP_API_KEY`), none of which were available. This re-implements its pipeline shape — ingest, index, search, answer, evaluate, report — against this service, with a different judge model.

So these numbers are comparable **in kind** to a MemoryBench leaderboard, not head-to-head with it. A different judge, a different answering model, a question subset and a re-implemented prompt all move absolute accuracy.

Every number here came from a real API response. Failed calls are counted in `n_errors` and excluded rather than substituted.
