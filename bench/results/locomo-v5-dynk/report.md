# Multi-benchmark retrieval report

Retrieval scored against each benchmark's labelled evidence — no LLM, so these numbers carry no judge variance and are exactly reproducible.

**`full_recall@k` is the headline**: 1.0 only when EVERY evidence item is retrieved. `hit@k` (any evidence found) is shown beside it because it is what most published numbers report, and the gap between the two is the point — on multi-evidence questions hit@k overstates readiness badly. Answering a three-hop question with one of three facts is a wrong answer.

Evidence granularity differs by benchmark and the numbers are **not comparable across benchmarks**: LoCoMo labels individual dialogue turns, LongMemEval labels whole sessions, and finding the right session is a far easier target than finding the right turn.

## locomo

10 corpora · 5882 documents · 1986 questions · k=10 · evidence at **turn** level · embedder `text-embedding-004`

| config | full_recall@k | hit@k | MRR | nDCG | latency ms |
|---|---|---|---|---|---|
| `vector_only` | **0.625** | 0.734 | 0.480 | 0.510 | 1688 |
| `lexical_only` | **0.573** | 0.671 | 0.453 | 0.477 | 434 |
| `hybrid_rrf` | **0.625** | 0.737 | 0.500 | 0.526 | 409 |
| `hybrid_rerank` | **0.609** | 0.719 | 0.500 | 0.522 | 378 |
| `hybrid_rerank_entity` | **0.609** | 0.719 | 0.500 | 0.522 | 393 |

### Per capability — `hybrid_rrf` (best full recall)

| capability | n | avg evidence | full_recall@k | 95% CI | hit@k |
|---|---|---|---|---|---|
| adversarial | 446 | 1.03 | **0.630** | 0.58-0.67 | 0.646 |
| multi_hop | 282 | 3.12 | **0.202** | 0.16-0.25 | 0.738 |
| open_domain | 89 | 2.21 | **0.270** | 0.19-0.37 | 0.472 |
| single_hop | 841 | 1.06 | **0.767** | 0.74-0.79 | 0.794 |
| temporal | 320 | 1.17 | **0.719** | 0.67-0.77 | 0.787 |

### End-to-end (LongMemEval's published metric) — `hybrid_rrf`

Answering model `gemini-2.5-flash`, judge `claude-sonnet-4-5@20250929`, judge protocol `longmemeval-official`.

**QA accuracy: 50.1%** (95% CI 47.6%-52.6%, n=1540)  
Abstention: 0.8677 (n=446) · context 829 tok · search 55 ms · errors 0

| capability | n | accuracy | 95% CI |
|---|---|---|---|
| adversarial | 446 | 86.8% | 0.83-0.90 |
| multi_hop | 282 | 27.7% | 0.23-0.33 |
| open_domain | 96 | 31.2% | 0.23-0.41 |
| single_hop | 841 | 59.9% | 0.57-0.63 |
| temporal | 321 | 49.8% | 0.44-0.55 |

### Judge independence

The answering model does not grade itself. Both arms below grade the **same** predictions with the **same** prompt; the only variable is whether the grader produced them.

| arm | grader | accuracy |
|---|---|---|
| independent (headline) | `claude-sonnet-4-5@20250929` | 63.4% |
| self-graded (control) | `gemini-2.5-flash` | 60.5% |

Self-preference: **-2.9%** (stricter on itself), judge agreement 95.0% on n=1218. The headline uses the independent grader; this delta is what a self-graded number would have silently added.

### Methodology caveats

* The advice/preference router that selects the personalisation prompt was **tuned against this benchmark's own `single-session-preference` questions**. Its specificity is independently validated (0 false positives on 1,986 LoCoMo questions), but its sensitivity is fitted, so the `single-session-preference` figure above is optimistically biased. Every other capability is untouched by it.
* `question_date` is a benchmark-provided input, not a label: it is what "how many weeks ago" is measured from, and every system evaluated here receives it.

### Completeness

Failed searches: **0** (excluded from retrieval metrics, never substituted).
Failed answer/judge calls: **0** of 1986 attempted; 1986 scored. An API failure is missing data, not a wrong answer, so it is excluded from accuracy — but the gap is reported here because an unreported gap silently inflates the headline.

