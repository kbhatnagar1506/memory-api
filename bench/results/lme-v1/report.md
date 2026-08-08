# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## longmemeval

60 corpora · 2900 documents · 60 questions · k=10 · evidence at **session** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.950** | 1.000 | 0.946 | 0.930 | 1104 |
| `lexical_only` | **0.933** | 0.983 | 0.913 | 0.910 | 766 |
| `hybrid_rrf` | **0.967** | 1.000 | 0.956 | 0.949 | 737 |
| `hybrid_rerank` | **0.900** | 0.983 | 0.853 | 0.845 | 715 |
| `hybrid_rerank_entity` | **0.900** | 0.983 | 0.853 | 0.845 | 3187 |

### Per capability — `hybrid_rrf` (best full recall)

| capability | n | avg evidence | full_recall@k | 95% CI | hit@k |
|---|---|---|---|---|---|
| knowledge-update | 10 | 2.0 | **1.000** | 0.72-1.00 | 1.000 |
| multi-session | 10 | 3.1 | **0.900** | 0.60-0.98 | 1.000 |
| single-session-assistant | 10 | 1.0 | **1.000** | 0.72-1.00 | 1.000 |
| single-session-preference | 10 | 1.0 | **1.000** | 0.72-1.00 | 1.000 |
| single-session-user | 10 | 1.0 | **1.000** | 0.72-1.00 | 1.000 |
| temporal-reasoning | 10 | 2.1 | **0.900** | 0.60-0.98 | 1.000 |

Failed searches: 0 (excluded, never substituted).

