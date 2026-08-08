# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## longmemeval

500 corpora · 23867 documents · 500 questions · k=10 · evidence at **session** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.968** | 0.994 | 0.939 | 0.937 | 2537 |
| `lexical_only` | **0.916** | 0.984 | 0.910 | 0.898 | 2083 |
| `hybrid_rrf` | **0.952** | 0.994 | 0.944 | 0.937 | 2317 |
| `hybrid_rerank` | **0.888** | 0.976 | 0.864 | 0.850 | 2395 |
| `hybrid_rerank_entity` | **0.888** | 0.976 | 0.864 | 0.850 | 6666 |

### Per capability — `vector_only` (best full recall)

| capability | n | avg evidence | full_recall@k | 95% CI | hit@k |
|---|---|---|---|---|---|
| knowledge-update | 78 | 2.0 | **1.000** | 0.95-1.00 | 1.000 |
| multi-session | 133 | 2.59 | **0.970** | 0.93-0.99 | 1.000 |
| single-session-assistant | 56 | 1.0 | **1.000** | 0.94-1.00 | 1.000 |
| single-session-preference | 30 | 1.0 | **1.000** | 0.89-1.00 | 1.000 |
| single-session-user | 70 | 1.0 | **1.000** | 0.95-1.00 | 1.000 |
| temporal-reasoning | 133 | 2.2 | **0.910** | 0.85-0.95 | 0.977 |

### End-to-end (LongMemEval's published metric) — `vector_only`

Answering model `gemini-2.5-flash`, judge `gemini-2.5-flash`.

**QA accuracy: 40.9%** (95% CI 36.5%-45.4%, n=467)  
Abstention: 0.8667 (n=30) · context 2388 tok · search 1430 ms · errors 3

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| knowledge-update | 78 | 51.3% | 0.40-0.62 |
| multi-session | 133 | 33.1% | 0.26-0.41 |
| single-session-assistant | 56 | 76.8% | 0.64-0.86 |
| single-session-preference | 30 | 16.7% | 0.07-0.34 |
| single-session-user | 67 | 64.2% | 0.52-0.75 |
| temporal-reasoning | 133 | 31.6% | 0.24-0.40 |

Failed searches: 0 (excluded, never substituted).

