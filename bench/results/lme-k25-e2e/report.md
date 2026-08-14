# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## longmemeval

500 corpora · 23867 documents · 500 questions · k=25 · evidence at **session** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.998** | 1.000 | 0.939 | 0.942 | 1956 |
| `lexical_only` | **0.970** | 0.998 | 0.910 | 0.907 | 2183 |
| `hybrid_rrf` | **0.992** | 1.000 | 0.945 | 0.945 | 2016 |
| `hybrid_rerank` | **0.992** | 1.000 | 0.865 | 0.869 | 2180 |
| `hybrid_rerank_entity` | **0.992** | 1.000 | 0.865 | 0.869 | 5085 |

### Per capability — `vector_only` (best full recall)

| capability | n | avg evidence | full_recall@k | 95% CI | hit@k |
|---|---|---|---|---|---|
| knowledge-update | 78 | 2.0 | **1.000** | 0.95-1.00 | 1.000 |
| multi-session | 133 | 2.59 | **1.000** | 0.97-1.00 | 1.000 |
| single-session-assistant | 56 | 1.0 | **1.000** | 0.94-1.00 | 1.000 |
| single-session-preference | 30 | 1.0 | **1.000** | 0.89-1.00 | 1.000 |
| single-session-user | 70 | 1.0 | **1.000** | 0.95-1.00 | 1.000 |
| temporal-reasoning | 133 | 2.2 | **0.993** | 0.96-1.00 | 1.000 |

### End-to-end (LongMemEval's published metric) — `vector_only`

Answering model `gemini-2.5-flash`, judge `claude-sonnet-4-5@20250929`, judge protocol `longmemeval-official`.

**QA accuracy: 82.0%** (95% CI 78.3%-85.2%, n=467)  
Abstention: 0.6 (n=30) · context 65337 tok · search 283 ms · errors 3

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| knowledge-update | 78 | 85.9% | 0.76-0.92 |
| multi-session | 131 | 74.8% | 0.67-0.81 |
| single-session-assistant | 56 | 96.4% | 0.88-0.99 |
| single-session-preference | 30 | 43.3% | 0.27-0.61 |
| single-session-user | 69 | 91.3% | 0.82-0.96 |
| temporal-reasoning | 133 | 79.7% | 0.72-0.86 |

### Judge independence

The answering model does not grade itself. Both arms below grade the **same** predictions with the **same** prompt; the only variable is whether the grader produced them.

| arm | grader | accuracy |
|---|---|---|
| independent (headline) | `claude-sonnet-4-5@20250929` | 82.0% |
| self-graded (control) | `gemini-2.5-flash` | 80.7% |

Self-preference: **-1.3%** (stricter on itself), judge agreement 97.9% on n=467. The headline uses the independent grader; this delta is what a self-graded number would have silently added.

### Methodology caveats

* The advice/preference router that selects the personalisation prompt was **tuned against this benchmark's own `single-session-preference` questions**. Its specificity is independently validated (0 false positives on 1,986 LoCoMo questions), but its sensitivity is fitted, so the `single-session-preference` figure above is optimistically biased. Every other capability is untouched by it.
* `question_date` is a benchmark-provided input, not a label: it is what "how many weeks ago" is measured from, and every system evaluated here receives it.

### Completeness

Failed searches: **0** (excluded from retrieval metrics, never substituted).
Failed answer/judge calls: **3** of 500 attempted; 497 scored. An API failure is missing data, not a wrong answer, so it is excluded from accuracy — but the gap is reported here because an unreported gap silently inflates the headline.

