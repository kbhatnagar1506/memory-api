# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## longmemeval

500 corpora · 23867 documents · 500 questions · k=10 · evidence at **session** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.968** | 0.994 | 0.939 | 0.937 | 1411 |
| `lexical_only` | **0.916** | 0.984 | 0.910 | 0.898 | 1477 |
| `hybrid_rrf` | **0.952** | 0.994 | 0.944 | 0.937 | 1542 |
| `hybrid_rerank` | **0.888** | 0.976 | 0.864 | 0.850 | 1443 |
| `hybrid_rerank_entity` | **0.888** | 0.976 | 0.864 | 0.850 | 3998 |

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

**QA accuracy: 81.1%** (95% CI 77.3%-84.4%, n=470)  
Abstention: 0.8333 (n=30) · context 52764 tok · search 235 ms · errors 0

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| knowledge-update | 78 | 80.8% | 0.71-0.88 |
| multi-session | 133 | 77.4% | 0.70-0.84 |
| single-session-assistant | 56 | 98.2% | 0.91-1.00 |
| single-session-preference | 30 | 50.0% | 0.33-0.67 |
| single-session-user | 70 | 94.3% | 0.86-0.98 |
| temporal-reasoning | 133 | 78.2% | 0.70-0.84 |

### Judge independence

The answering model does not grade itself. Both arms below grade the **same** predictions with the **same** prompt; the only variable is whether the grader produced them.

| arm | grader | accuracy |
|---|---|---|
| independent (headline) | `claude-sonnet-4-5@20250929` | 83.5% |
| self-graded (control) | `gemini-2.5-flash` | 83.1% |

Self-preference: **-0.4%** (stricter on itself), judge agreement 95.6% on n=456. The headline uses the independent grader; this delta is what a self-graded number would have silently added.

### Methodology caveats

* The advice/preference router that selects the personalisation prompt was **tuned against this benchmark's own `single-session-preference` questions**. Its specificity is independently validated (0 false positives on 1,986 LoCoMo questions), but its sensitivity is fitted, so the `single-session-preference` figure above is optimistically biased. Every other capability is untouched by it.
* `question_date` is a benchmark-provided input, not a label: it is what "how many weeks ago" is measured from, and every system evaluated here receives it.

### Completeness

Failed searches: **0** (excluded from retrieval metrics, never substituted).
Failed answer/judge calls: **0** of 500 attempted; 500 scored. An API failure is missing data, not a wrong answer, so it is excluded from accuracy — but the gap is reported here because an unreported gap silently inflates the headline.

