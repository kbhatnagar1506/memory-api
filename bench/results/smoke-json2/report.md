# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## longmemeval

5 corpora · 235 documents · 5 questions · k=10 · evidence at **session** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **1.000** | 1.000 | 0.900 | 0.891 | 712 |
| `lexical_only` | **1.000** | 1.000 | 1.000 | 1.000 | 695 |
| `hybrid_rrf` | **1.000** | 1.000 | 1.000 | 1.000 | 701 |
| `hybrid_rerank` | **1.000** | 1.000 | 1.000 | 0.961 | 716 |
| `hybrid_rerank_entity` | **1.000** | 1.000 | 1.000 | 0.961 | 3620 |

### Per capability — `vector_only` (best full recall)

| capability | n | avg evidence | full_recall@k | 95% CI | hit@k |
|---|---|---|---|---|---|
| knowledge-update | 1 | 2.0 | **1.000** | 0.21-1.00 | 1.000 |
| multi-session | 1 | 3.0 | **1.000** | 0.21-1.00 | 1.000 |
| single-session-assistant | 1 | 1.0 | **1.000** | 0.21-1.00 | 1.000 |
| single-session-preference | 1 | 1.0 | **1.000** | 0.21-1.00 | 1.000 |
| single-session-user | 1 | 1.0 | **1.000** | 0.21-1.00 | 1.000 |

### End-to-end (LongMemEval's published metric) — `vector_only`

Answering model `gemini-2.5-flash`, judge `claude-sonnet-4-5@20250929`, judge protocol `longmemeval-official`.

**QA accuracy: 100.0%** (95% CI 56.5%-100.0%, n=5)  
Abstention: None (n=0) · context 28060 tok · search 701 ms · errors 0

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| knowledge-update | 1 | 100.0% | 0.21-1.00 |
| multi-session | 1 | 100.0% | 0.21-1.00 |
| single-session-assistant | 1 | 100.0% | 0.21-1.00 |
| single-session-preference | 1 | 100.0% | 0.21-1.00 |
| single-session-user | 1 | 100.0% | 0.21-1.00 |

### Judge independence

The answering model does not grade itself. Both arms below grade the **same** predictions with the **same** prompt; the only variable is whether the grader produced them.

| arm | grader | accuracy |
|---|---|---|
| independent (headline) | `claude-sonnet-4-5@20250929` | 100.0% |
| self-graded (control) | `gemini-2.5-flash` | 100.0% |

Self-preference: **+0.0%** (stricter on itself), judge agreement 100.0% on n=5. The headline uses the independent grader; this delta is what a self-graded number would have silently added.

### Methodology caveats

* The advice/preference router that selects the personalisation prompt was **tuned against this benchmark's own `single-session-preference` questions**. Its specificity is independently validated (0 false positives on 1,986 LoCoMo questions), but its sensitivity is fitted, so the `single-session-preference` figure above is optimistically biased. Every other capability is untouched by it.
* `question_date` is a benchmark-provided input, not a label: it is what "how many weeks ago" is measured from, and every system evaluated here receives it.

### Completeness

Failed searches: **0** (excluded from retrieval metrics, never substituted).
Failed answer/judge calls: **0** of 5 attempted; 5 scored. An API failure is missing data, not a wrong answer, so it is excluded from accuracy — but the gap is reported here because an unreported gap silently inflates the headline.

