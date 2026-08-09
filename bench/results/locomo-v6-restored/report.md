# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## locomo

10 corpora · 5882 documents · 1986 questions · k=10 · evidence at **turn** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.625** | 0.734 | 0.480 | 0.510 | 221 |
| `lexical_only` | **0.573** | 0.671 | 0.453 | 0.477 | 219 |
| `hybrid_rrf` | **0.625** | 0.737 | 0.500 | 0.526 | 218 |
| `hybrid_rerank` | **0.609** | 0.720 | 0.500 | 0.522 | 219 |
| `hybrid_rerank_entity` | **0.609** | 0.720 | 0.500 | 0.522 | 220 |

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

**QA accuracy: 72.6%** (95% CI 70.3%-74.8%, n=1540)  
Abstention: 0.8072 (n=446) · context 3440 tok · search 37 ms · errors 0

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| adversarial | 446 | 80.7% | 0.77-0.84 |
| multi_hop | 282 | 59.9% | 0.54-0.65 |
| open_domain | 96 | 43.8% | 0.34-0.54 |
| single_hop | 841 | 82.2% | 0.79-0.85 |
| temporal | 321 | 67.3% | 0.62-0.72 |

### Judge independence

The answering model does not grade itself. Both arms below grade the **same** predictions with the **same** prompt; the only variable is whether the grader produced them.

| arm | grader | accuracy |
|---|---|---|
| independent (headline) | `claude-sonnet-4-5@20250929` | 78.3% |
| self-graded (control) | `gemini-2.5-flash` | 74.3% |

Self-preference: **-4.1%** (stricter on itself), judge agreement 93.1% on n=1427. The headline uses the independent grader; this delta is what a self-graded number would have silently added.

### Methodology caveats

* The advice/preference router that selects the personalisation prompt was **tuned against this benchmark's own `single-session-preference` questions**. Its specificity is independently validated (0 false positives on 1,986 LoCoMo questions), but its sensitivity is fitted, so the `single-session-preference` figure above is optimistically biased. Every other capability is untouched by it.
* `question_date` is a benchmark-provided input, not a label: it is what "how many weeks ago" is measured from, and every system evaluated here receives it.

### Completeness

Failed searches: **0** (excluded from retrieval metrics, never substituted).
Failed answer/judge calls: **0** of 1986 attempted; 1986 scored. An API failure is missing data, not a wrong answer, so it is excluded from accuracy — but the gap is reported here because an unreported gap silently inflates the headline.

