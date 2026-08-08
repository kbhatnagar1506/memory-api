# LoCoMo benchmark — locomo-v1

Corpus: **5882 dialogue turns** across 10 conversations, one memory per turn, one space per conversation.  
Questions: **496 sampled** (stratified by category) from 1986.  
Embeddings: `text-embedding-004` (768d, gemini).  
Retrieval depth k = 10. Seed 20260808.

Latency below is **retrieval only**: the query-embedding cache is warmed before timing, so every config is measured on equal footing. Query embedding costs a further 74 ms per query uncached, and is reported separately because it is identical across configs and would otherwise swamp the comparison.

Retrieval metrics are scored against LoCoMo's labelled evidence turns — no LLM, so they carry no judge variance and are exactly reproducible. Adversarial (unanswerable) questions are excluded from retrieval metrics, since they have no evidence to retrieve by construction.

## Retrieval ablation

| config | recall@10 | hit@1 | hit@10 | MRR | nDCG@10 | latency ms |
|---|---|---|---|---|---|---|
| `vector_only` | 0.607 | 0.323 | 0.691 | 0.434 | 0.450 | 28.4 |
| `lexical_only` | 0.554 | 0.323 | 0.634 | 0.415 | 0.426 | 29.0 |
| `hybrid_rrf` | 0.589 | 0.325 | 0.679 | 0.444 | 0.455 | 29.5 |
| `hybrid_rerank_heuristic` | 0.593 | 0.366 | 0.683 | 0.466 | 0.469 | 28.6 |
| `hybrid_mmr` | 0.593 | 0.325 | 0.683 | 0.445 | 0.456 | 70.1 |
| `hybrid_decay` | 0.592 | 0.319 | 0.681 | 0.440 | 0.453 | 30.4 |
| `full_no_llm` | 0.583 | 0.339 | 0.671 | 0.442 | 0.451 | 68.4 |

### Against the hybrid baseline

| config | Δ MRR | Δ recall | verdict |
|---|---|---|---|
| `vector_only` | -0.011 | +0.018 | hurts |
| `lexical_only` | -0.029 | -0.035 | hurts |
| `hybrid_rerank_heuristic` | +0.022 | +0.004 | helps |
| `hybrid_mmr` | +0.000 | +0.004 | no effect |
| `hybrid_decay` | -0.005 | +0.003 | no effect |
| `full_no_llm` | -0.002 | -0.006 | no effect |

### By question category

| config | adversarial | multi_hop | open_domain | single_hop | temporal |
|---|---|---|---|---|---|
| `vector_only` | 0.179 | 0.520 | 0.309 | 0.618 | 0.532 |
| `lexical_only` | 0.475 | 0.309 | 0.230 | 0.548 | 0.496 |
| `hybrid_rrf` | 0.304 | 0.479 | 0.266 | 0.611 | 0.547 |
| `hybrid_rerank_heuristic` | 0.479 | 0.457 | 0.262 | 0.574 | 0.540 |
| `hybrid_mmr` | 0.302 | 0.482 | 0.265 | 0.612 | 0.547 |
| `hybrid_decay` | 0.307 | 0.461 | 0.271 | 0.610 | 0.535 |
| `full_no_llm` | 0.462 | 0.382 | 0.265 | 0.551 | 0.535 |

Cells are MRR. Category counts: adversarial=100, multi_hop=100, open_domain=92, single_hop=100, temporal=100

## End-to-end

Config `hybrid_rerank_heuristic` (best MRR). Answering model `gemini-2.5-flash`, judge `gemini-2.5-flash`.

**MemScore: 59% / 49ms / 2530tok**

| metric | value |
|---|---|
| accuracy (answerable) | 58.6% (n=396) |
| adversarial decline rate | 0.91 (n=100) |
| retrieval hit@10 | 0.667 |
| mean search latency | 49.5 ms |
| mean context tokens | 2530 |
| errors | 0 |

Accuracy by category:

| category | n | accuracy |
|---|---|---|
| adversarial | 100 | 91.0% |
| multi_hop | 100 | 35.0% |
| open_domain | 96 | 45.8% |
| single_hop | 100 | 88.0% |
| temporal | 100 | 65.0% |

The adversarial rate is a *decline* rate: the correct behaviour on an unanswerable question is to answer `NO_ANSWER`. It is scored without a judge and kept out of the headline accuracy, because a system that confidently answers the unanswerable should not be rewarded for it.

## What this is and is not

This is **not** a run of the `supermemoryai/memorybench` repository. That harness requires `bun` and hosted provider API keys (`SUPERMEMORY_API_KEY`, `MEM0_API_KEY`, `ZEP_API_KEY`), none of which were available. This re-implements its pipeline shape — ingest, index, search, answer, evaluate, report — against this service, with a different judge model.

So these numbers are comparable **in kind** to a MemoryBench leaderboard, not head-to-head with it. A different judge, a different answering model, a question subset and a re-implemented prompt all move absolute accuracy.

Every number here came from a real API response. Failed calls are counted in `n_errors` and excluded rather than substituted.
