# Decisions

One line per real tradeoff: what was chosen, and what the alternative was.
Anything touching **arm parity**, **ground truth**, or **layout stability** is
marked ⚑ and was raised rather than decided silently.

## Graph and queries

- ⚑ **Node IDs are decorrelated from time order** (|Spearman rho| ≤ 0.35, asserted
  and retried on failure). If N01 were reliably the oldest node, `temporal_pair`
  and `oldest_in_domain` would be answerable from the ID string with the payload
  ignored entirely, and all four arms would score identically for a reason that
  has nothing to do with representation. Alternative: assign `t` in ID order —
  simpler, and silently invalidates two of the six query types.
- ⚑ **`n_per_type` is a target, not a guarantee.** Rule (a) (no two instances of
  a type share an answer node) caps `oldest_in_domain` at 4 (one per domain) and
  `root_cause` at one per chain (6 at 20 nodes, 8 at 40/80). Actual totals are
  44 queries per graph at 40 nodes, not 48. Alternative: relax rule (a) for
  those types, which would let one node's legibility dominate a type's score.
- ⚑ **`supersede` is asked in both directions** ("which supersedes X" / "which
  does X supersede"). 4–6 node-disjoint pairs cannot supply 8 single-direction
  questions. The second phrasing also tests whether the model reads edge
  *direction* rather than mere adjacency. Alternative: cap the type at 6.
- **Labels come from a hand-written per-domain vocabulary**, not a generator.
  Reads like a real memory record; costs 96 lines of vocab.py.
- **`random.Random` with an explicit integer seed**, not numpy. Stable across
  Python versions for the operations used, and one less determinism surface.

## Renderer

- ⚑ **Geometry is absolute authored pixels, not fractions of canvas width.** The
  first version scaled every constant by `width/1540`, which made the
  authored-width sweep a *no-op*: scaling content and canvas together preserves
  the aspect ratio, so a long-edge-capped provider receives a pixel-identical
  image at 1540 and 2560. With absolute geometry a wider canvas genuinely fits
  more nodes per row. `tests/test_legibility.py::test_uniform_geometry_scaling_
  does_not_change_postresize_size` pins the trap so it cannot come back.
- **Node size and fonts come from `scripts/search_geometry.py`**, which maximises
  post-resize label cap-height subject to zero node collisions across 10 seeds.
  Chosen values: `node_w=164, node_h=80, font_id=19, font_label=12`. The search
  output, under the assumed 1568px long-edge cap:

  | nodes | width | subrows | canvas | scale | post-resize cap (id / label) |
  |---|---|---|---|---|---|
  | 20 | 1540 | 3 | 1540×1466 | 1.000 | 14.0 / 9.0 |
  | 40 | 1540 | 4 | 1540×1826 | 0.859 | 12.0 / 7.7 |
  | 80 | 1540 | 8 | 1540×3266 | 0.480 | 6.7 / 4.3 |

  Alternative: pick sizes by eye, and discover at analysis time that the 80-node
  arm was illegible for a reason unrelated to the hypothesis.
- **Sub-row counts are a per-(nodes, width) constant, not data-dependent.** Band
  heights must not depend on how crowded a band happens to be, or adding a node
  to Platform would move every node in Finance. Measured table in
  `layout.SUBROWS_MEASURED`; the fallback heuristic is flagged as unverified.
- ⚑ **Sub-row assignment is first-fit over x-overlapping neighbours in time
  order.** This is the one place a cascade is still possible: deleting a node can
  free a slot that a later overlapping node moves into. A pure `hash(id) % rows`
  rule would be perfectly stable but would overlap boxes. §7 measures the
  residual cascade rather than assuming it away.
- **`contradicts` gets end bars, not double arrowheads.** The brief calls it a
  "double-ended connector", but arrowheads are reserved for causality, so a
  double arrowhead would collide with the one convention the grammar most needs
  to keep clean.
- **Border weights are 2 / 5 / 8 / 11 px.** 2/4/7/10 made `entity` and `event`
  visually indistinguishable in the legend swatch. Border weight is the most
  fragile primitive under downscaling; the pretest measures it directly rather
  than assuming it survives.
- **The date sits on the ID line, right-aligned**, not on its own row. Saves a
  text row per node, which is what keeps the 40-node canvas short enough to
  matter. Costs a slightly denser box.
- **Labels wrap to 2 lines and are never truncated.** Truncation would drop
  information the text arms still carry, i.e. break arm parity. `render_report()`
  asserts no overflow; `test_arm_parity` asserts the depicted label equals the
  graph label.
- **Legend width is measured from its own text**, not estimated from a character
  count. The estimate clipped the longest connector caption — precisely the line
  a cold-reading model needs.
- **Edges are straight lines clipped to box borders.** Orthogonal routing would
  read better at 80 nodes; it is also a large amount of code whose bugs would be
  hard to distinguish from grammar failures. Revisit if edge-tracing shows up as
  a failure cause.
- **Fonts are matplotlib's bundled DejaVu Sans**, not a system font, so renders
  are byte-identical across machines.

## Providers and harness

- **Model IDs are resolved against each provider's live model list**, never
  hardcoded. Config supplies candidates; the manifest records the resolved ID
  *and how it was resolved*. Vertex has no Claude model-list endpoint, so that
  path records "from config, not list-verified" rather than implying otherwise.
- **Gemini defaults to Vertex AI + ADC**; Anthropic can use either an API key or
  Vertex. OpenAI has no ADC path and refuses to construct without a key rather
  than silently dropping itself from the matrix.
- **Errors are never cached.** A 429 must not become a permanent result.
- **`unparseable` is a first-class outcome**, distinct from `wrong`.
- **Provider resize rules are labelled `assumed` until measured.** Every
  cap-height row carries its provenance. `pretest --verify-resize` measures the
  real scaling from reported `input_tokens` and writes `legibility.md`.

## Stage 0 findings (runs/gate-01)

- ⚑ **Gemini's measured image scaling contradicts the assumed rule.** Token cost
  saturates at 3354 (= 13 x 258, i.e. a 13-tile cap): 1540x1568 and 1540x1826
  cost identically, as do 1540x2400, 2560x1600 and 2560x2786. The assumed
  "no downscale below 3072px long edge" was wrong, so `ASSUMED_RULES["gemini"]`
  must not be used for any reported number. Measured table: runs/gate-01/legibility.md.
- ⚑ **`border_weight` fails the gate (40% cold / 20% primed) and a targeted fix
  did not help.** Rendering the legend swatches at true node size changed
  nothing (identical error pattern), so the swatch-size hypothesis is rejected.
  Controlled sweeps show the border *is* read but resolves only ~1 bit
  (thin vs thick) across 4 authored levels, and label semantics fill the gap.
  Full write-up: runs/gate-01/diagnosis.md. No query type reads node type, so
  this is off the critical path -- flagged rather than fixed unilaterally,
  because the brief specifies border weight as the type encoding.
- **Gemini 2.5+ spend the answer budget on reasoning tokens.** At max_tokens=24
  gemini-2.5-pro returned a truncated `'N'` (19 thought tokens), which would have
  scored `unparseable` across the board and looked exactly like an unreadable
  grammar. The adapter now reserves headroom on top of the caller's answer budget
  and reports truncation as an error rather than letting it score as a wrong
  answer. This is the class of bug that manufactures a false negative result.
- **Five probes per primitive per condition cannot measure a 95% gate.** The 95%
  CI on 4/5 spans roughly 28%-99%. The 40-canvas budget in the brief buys 5 per
  primitive across 8 primitives; gating at 95% needs ~30+ per primitive.

## Stage 0, higher-powered re-run (runs/gate-03)

- ⚑ **The pretest cache key omitted the payload hash, and it corrupted a run.**
  Probe ids like `band_label-00` are stable across runs, but gate-02's probe
  content differed from gate-01's, so the cache served gate-01's answers for
  gate-02's canvases. It manufactured a fake 87% on `band_label` and a fake 20%
  on `date_read` -- both would have read as grammar failures. The key now binds
  to the PNG bytes and the expected answer. `runner.Call.cache_key()` always had
  `payload_hash`; the pretest path did not. Regression test:
  `test_pretest_cache_key_includes_payload`.
- ⚑ **Two `edge_type` probes were degenerate.** Two same-band nodes one date-step
  apart put their boxes in contact, leaving the dashed `contradicts` connector
  with ZERO visible span. The model answered `caused` on exactly those two, and
  nothing else -- `edge_type` went from 93% to 100% once they were fixed. Probes
  now enforce a minimum same-band separation, `Probe.png()` asserts no
  collisions, and a test asserts every connector has >40px of visible span.
- **The gate is three-state, judged on the Wilson interval.** A two-state
  interval gate is unpassable at small n (30/30 has a lower bound of 88.6%).
  FAIL = interval entirely below threshold, PASS = entirely at or above,
  INCONCLUSIVE = this sample cannot tell. A decisive per-primitive PASS at 95%
  needs n>=75 with zero errors, or n>=150 to tolerate one.
- **Result: critical path 358/360 = 99.4%, CI 98.0%-99.8%, clears 95% pooled.**
  Per-primitive rows are individually INCONCLUSIVE at n=30, which is a sample
  size statement, not a grammar statement. Zero unparseable, zero errors.
