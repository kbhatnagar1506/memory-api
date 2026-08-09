# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## longmemeval

500 corpora · 23867 documents · 500 questions · k=10 · evidence at **session** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.968** | 0.994 | 0.939 | 0.937 | 1568 |
| `lexical_only` | **0.916** | 0.984 | 0.910 | 0.898 | 1608 |
| `hybrid_rrf` | **0.952** | 0.994 | 0.944 | 0.937 | 1448 |
| `hybrid_rerank` | **0.888** | 0.976 | 0.864 | 0.850 | 1464 |
| `hybrid_rerank_entity` | **0.888** | 0.976 | 0.864 | 0.850 | 3990 |

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

Answering model `gemini-2.5-flash`, judge `claude-sonnet-4-5@20250929`, judge protocol `longmemeval-official`.

**QA accuracy: 45.1%** (95% CI 40.7%-49.6%, n=470)  
Abstention: 0.5 (n=30) · context 9696 tok · search 213 ms · errors 0

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| knowledge-update | 78 | 35.9% | 0.26-0.47 |
| multi-session | 133 | 30.8% | 0.24-0.39 |
| single-session-assistant | 56 | 91.1% | 0.81-0.96 |
| single-session-preference | 30 | 26.7% | 0.14-0.44 |
| single-session-user | 70 | 80.0% | 0.69-0.88 |
| temporal-reasoning | 133 | 32.3% | 0.25-0.41 |

### Judge independence

The answering model does not grade itself. Both arms below grade the **same** predictions with the **same** prompt; the only variable is whether the grader produced them.

| arm | grader | accuracy |
|---|---|---|
| independent (headline) | `claude-sonnet-4-5@20250929` | 48.0% |
| self-graded (control) | `gemini-2.5-flash` | 48.2% |

Self-preference: **+0.2%** (more generous to itself), judge agreement 96.2% on n=442. The headline uses the independent grader; this delta is what a self-graded number would have silently added.

### Completeness

Failed searches: **0** (excluded from retrieval metrics, never substituted).
Failed answer/judge calls: **0** of 500 attempted; 500 scored. An API failure is missing data, not a wrong answer, so it is excluded from accuracy — but the gap is reported here because an unreported gap silently inflates the headline.

