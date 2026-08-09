# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## locomo

10 corpora · 5882 documents · 1986 questions · k=10 · evidence at **turn** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.625** | 0.734 | 0.480 | 0.510 | 261 |
| `lexical_only` | **0.573** | 0.671 | 0.453 | 0.477 | 328 |
| `hybrid_rrf` | **0.624** | 0.736 | 0.499 | 0.526 | 292 |
| `hybrid_rerank` | **0.609** | 0.719 | 0.500 | 0.522 | 318 |
| `hybrid_rerank_entity` | **0.609** | 0.719 | 0.500 | 0.522 | 268 |

### Per capability — `vector_only` (best full recall)

| capability | n | avg evidence | full_recall@k | 95% CI | hit@k |
|---|---|---|---|---|---|
| adversarial | 446 | 1.03 | **0.478** | 0.43-0.52 | 0.491 |
| multi_hop | 282 | 3.12 | **0.277** | 0.23-0.33 | 0.819 |
| open_domain | 89 | 2.21 | **0.337** | 0.25-0.44 | 0.584 |
| single_hop | 841 | 1.06 | **0.804** | 0.78-0.83 | 0.818 |
| temporal | 320 | 1.17 | **0.747** | 0.70-0.79 | 0.819 |

### End-to-end (LongMemEval's published metric) — `vector_only`

Answering model `gemini-2.5-flash`, judge `claude-sonnet-4-5@20250929`, judge protocol `longmemeval-official`.

**QA accuracy: 61.3%** (95% CI 58.8%-63.7%, n=1540)  
Abstention: 0.8296 (n=446) · context 954 tok · search 40 ms · errors 0

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| adversarial | 446 | 83.0% | 0.79-0.86 |
| multi_hop | 282 | 43.6% | 0.38-0.49 |
| open_domain | 96 | 31.2% | 0.23-0.41 |
| single_hop | 841 | 71.2% | 0.68-0.74 |
| temporal | 321 | 59.8% | 0.54-0.65 |

### Judge independence

The answering model does not grade itself. Both arms below grade the **same** predictions with the **same** prompt; the only variable is whether the grader produced them.

| arm | grader | accuracy |
|---|---|---|
| independent (headline) | `claude-sonnet-4-5@20250929` | 72.7% |
| self-graded (control) | `gemini-2.5-flash` | 69.5% |

Self-preference: **-3.2%** (stricter on itself), judge agreement 93.5% on n=1298. The headline uses the independent grader; this delta is what a self-graded number would have silently added.

### Methodology caveats

* The advice/preference router that selects the personalisation prompt was **tuned against this benchmark's own `single-session-preference` questions**. Its specificity is independently validated (0 false positives on 1,986 LoCoMo questions), but its sensitivity is fitted, so the `single-session-preference` figure above is optimistically biased. Every other capability is untouched by it.
* `question_date` is a benchmark-provided input, not a label: it is what "how many weeks ago" is measured from, and every system evaluated here receives it.

### Completeness

Failed searches: **0** (excluded from retrieval metrics, never substituted).
Failed answer/judge calls: **0** of 1986 attempted; 1986 scored. An API failure is missing data, not a wrong answer, so it is excluded from accuracy — but the gap is reported here because an unreported gap silently inflates the headline.

