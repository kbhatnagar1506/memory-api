"""Stage 0 pretest -- the gate.

Tiny canvases, each testing exactly ONE grammar primitive in isolation with no
distractors. If a frontier model cannot read a single primitive off a clean
canvas at >=95%, the grammar itself is broken and the full ablation would only
be measuring that.

Sample size is not uniform across primitives, and that is deliberate. The
original 40-canvas budget bought 5 probes per primitive, which cannot measure a
95% threshold at all: the 95% CI on a 4/5 result runs from about 28% to 99%.
Primitives on the critical path for the six query types therefore get 30 each;
the two that no query exercises stay at 5. See CRITICAL_PATH.

These canvases go through the *same* renderer as the real ones. A mock would
test a drawing that does not exist.

Answers here are not always node IDs (a date, a domain name, a relation word),
so each primitive carries its own scorer. That is the one place the pretest
deliberately differs from the main harness, which is single-format by design.
"""

from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass, field

from .graph import Edge, Graph, Node
from .prompts import _PRIMER
from .render.draw import render, render_report
from .render.grammar import TYPE_ORDER
from .render.layout import RenderConfig

PRETEST_PROMPT_VERSION = "pretest-v1"

#: The bar. Below this on any primitive, the grammar is the problem.
GATE_THRESHOLD = 0.95

_DOMAINS4 = ("Finance", "Growth", "People", "Platform")


def _node(nid: str, label: str, ntype: str, domain: str, t: int, date: str) -> Node:
    return Node(id=nid, label=label, type=ntype, domain=domain, t=t, date=date)


def _tiny(nodes: list[Node], edges: list[Edge]) -> Graph:
    """A hand-built graph.

    Deliberately bypasses generate_graph: the structural invariants exist to
    make *queries* single-answered on a full graph, and a 2-node probe cannot
    and need not satisfy them.
    """
    return Graph(seed=-1, n_nodes=len(nodes), nodes=nodes, edges=edges, chains=[])


@dataclass
class Probe:
    primitive: str
    index: int
    graph: Graph
    question: str
    answer: str
    #: How to check a free-text response for this primitive.
    scorer: str
    accepted: tuple[str, ...] = field(default_factory=tuple)
    #: Probe-level covariates (date separation, same-domain, relation) so
    #: the report can check whether errors cluster rather than being uniform.
    meta: dict = field(default_factory=dict)

    @property
    def pid(self) -> str:
        return f"{self.primitive}-{self.index:02d}"

    def png(self, width: int = 1540) -> bytes:
        cfg = RenderConfig(width=width, subrows=1)
        res = render(self.graph, cfg)
        render_report(res)  # assert nothing was truncated
        # Overlapping boxes hide the very thing a probe is meant to isolate.
        assert not res.collisions, (
            f"{self.pid}: node boxes overlap ({res.collisions}); "
            "this probe cannot test its primitive"
        )
        buf = io.BytesIO()
        res.image.save(buf, format="PNG", optimize=False)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# probe builders -- 5 instances per primitive, 8 primitives, 40 canvases
# ---------------------------------------------------------------------------

_LABELS = (
    "Cache layer added", "Trial length shortened", "Board budget approved",
    "Team reorg completed", "Search index rebuilt", "Ad budget reallocated",
    "Vendor contract renewed", "Oncall rotation expanded", "Free tier introduced",
    "Region failover tested",
)
_DATES = (
    "2025-02-10", "2025-04-22", "2025-06-30", "2025-08-14", "2025-10-05",
    "2025-11-19", "2026-01-08", "2026-03-17", "2026-05-02", "2026-06-21",
)


#: Primitives that the six query types in queries.py actually depend on.
#:
#:   temporal_pair       -> time_order
#:   oldest_in_domain    -> time_order, band_label, same_band
#:   root_cause          -> causal_direction, edge_type
#:   cross_domain_effect -> causal_direction, band_label
#:   contradiction       -> edge_type
#:   supersede           -> supersede_direction, edge_type
#:
#: These get the sample size needed to actually measure a 95% gate. The rest
#: stay small: `date_read` is redundant with x-position, and `border_weight`
#: is a known defect that no query exercises (runs/gate-01/diagnosis.md).
CRITICAL_PATH = (
    "time_order", "band_label", "same_band",
    "causal_direction", "edge_type", "supersede_direction",
)
NON_CRITICAL = ("date_read", "border_weight")

#: n=5 gives a 95% CI of roughly 28%-99% on a 4/5 result -- useless against a
#: 95% threshold. n=30 narrows a 29/30 result to about 83%-100%.
#: Minimum date-step separation for two nodes placed in the SAME band. Below
#: this their boxes touch and any connector between them is fully occluded.
_MIN_SAME_BAND_GAP = 3

DEFAULT_CRITICAL_N = 30
DEFAULT_NON_CRITICAL_N = 5


def default_counts() -> dict[str, int]:
    counts = {p: DEFAULT_CRITICAL_N for p in CRITICAL_PATH}
    counts.update({p: DEFAULT_NON_CRITICAL_N for p in NON_CRITICAL})
    return counts


def _pair(rng: random.Random, i: int, same_domain: bool = False
          ) -> tuple[Node, Node, dict]:
    """Two nodes, varied in domain, date separation, type and label.

    Date separation is varied deliberately: it is the pretest analogue of
    `t_gap`, and it is the axis along which a spatial encoding is most likely
    to degrade. Recorded in meta so the report can check whether errors cluster
    at small separations rather than being uniform.
    """
    d1 = _DOMAINS4[rng.randrange(4)]
    if same_domain:
        d2 = d1
    else:
        d2 = _DOMAINS4[(_DOMAINS4.index(d1) + 1 + rng.randrange(3)) % 4]

    lo = rng.randrange(0, len(_DATES) - 1)
    hi = rng.randrange(lo + 1, len(_DATES))
    # Two same-band nodes one date-step apart sit close enough that their boxes
    # touch and the connector between them has ZERO visible span -- there is no
    # line left to read, so the probe tests nothing. Clamping (rather than
    # redrawing) keeps the RNG stream stable so unaffected probes stay cached.
    if same_domain and hi - lo < _MIN_SAME_BAND_GAP:
        hi = lo + _MIN_SAME_BAND_GAP
        if hi >= len(_DATES):
            hi = len(_DATES) - 1
            lo = hi - _MIN_SAME_BAND_GAP
    a = _node(f"N{2 * (i % 45) + 1:02d}", _LABELS[rng.randrange(len(_LABELS))],
              TYPE_ORDER[rng.randrange(4)], d1, 1, _DATES[lo])
    b = _node(f"N{2 * (i % 45) + 2:02d}", _LABELS[rng.randrange(len(_LABELS))],
              TYPE_ORDER[rng.randrange(4)], d2, 2, _DATES[hi])
    if a.label == b.label:  # keep the two boxes textually distinguishable
        b = _node(b.id, _LABELS[(_LABELS.index(b.label) + 1) % len(_LABELS)],
                  b.type, b.domain, b.t, b.date)
    return a, b, {"date_gap_steps": hi - lo, "same_domain": d1 == d2}


def _build_probes(counts: dict[str, int] | None = None, seed: int = 20260805
                  ) -> list[Probe]:
    counts = counts or default_counts()
    probes: list[Probe] = []
    rng = random.Random(seed)

    for i in range(counts.get("time_order", 0)):
        a, b, meta = _pair(rng, i)
        probes.append(Probe(
            "time_order", i, _tiny([a, b], []),
            f"Which of {a.id} and {b.id} is older?", a.id, "node_id", meta=meta,
        ))

    for i in range(counts.get("causal_direction", 0)):
        a, b, meta = _pair(rng, i)
        probes.append(Probe(
            "causal_direction", i,
            _tiny([a, b], [Edge(a.id, b.id, "caused")]),
            f"One of {a.id} and {b.id} caused the other. Which one is the cause?",
            a.id, "node_id", meta=meta,
        ))

    for i in range(counts.get("same_band", 0)):
        a, b, meta = _pair(rng, i, same_domain=True)
        third = _node(
            f"N{2 * (i % 45) + 62:02d}", _LABELS[rng.randrange(len(_LABELS))],
            TYPE_ORDER[rng.randrange(4)],
            _DOMAINS4[(_DOMAINS4.index(a.domain) + 1 + rng.randrange(3)) % 4],
            3, _DATES[rng.randrange(len(_DATES))],
        )
        probes.append(Probe(
            "same_band", i, _tiny([a, b, third], []),
            "Which two of the three nodes are in the same domain band? "
            "Answer with their two node IDs.",
            f"{a.id},{b.id}", "id_pair", meta=meta,
        ))

    rels = ("caused", "contradicts", "supersedes")
    for i in range(counts.get("edge_type", 0)):
        rel = rels[i % 3]  # balanced across the three relations
        a, b, meta = _pair(rng, i, same_domain=(rel == "contradicts"))
        src, dst = (b, a) if rel == "supersedes" else (a, b)
        probes.append(Probe(
            "edge_type", i, _tiny([a, b], [Edge(src.id, dst.id, rel)]),
            f"What is the relationship drawn between {a.id} and {b.id}? "
            "Answer with one word: caused, contradicts, or supersedes.",
            rel, "word", accepted=rels, meta={**meta, "rel": rel},
        ))

    for i in range(counts.get("border_weight", 0)):
        a, b, meta = _pair(rng, i)
        probes.append(Probe(
            "border_weight", i, _tiny([a, b], []),
            f"Using the border weight legend, what is the type of node {a.id}? "
            "Answer with one word.",
            a.type, "word", accepted=TYPE_ORDER, meta=meta,
        ))

    for i in range(counts.get("date_read", 0)):
        a, b, meta = _pair(rng, i)
        probes.append(Probe(
            "date_read", i, _tiny([a, b], []),
            f"What date is shown on node {b.id}? Answer in YYYY-MM-DD form.",
            b.date, "date", meta=meta,
        ))

    for i in range(counts.get("band_label", 0)):
        a, b, meta = _pair(rng, i)
        probes.append(Probe(
            "band_label", i, _tiny([a, b], []),
            f"Which domain band is node {b.id} in? Answer with the band name.",
            b.domain, "word", accepted=_DOMAINS4, meta=meta,
        ))

    for i in range(counts.get("supersede_direction", 0)):
        a, b, meta = _pair(rng, i)
        probes.append(Probe(
            "supersede_direction", i,
            _tiny([a, b], [Edge(b.id, a.id, "supersedes")]),
            f"One of {a.id} and {b.id} supersedes the other. "
            "Which node is the one doing the superseding?",
            b.id, "node_id", meta=meta,
        ))

    return probes


def build_probes(counts: dict[str, int] | None = None, seed: int = 20260805
                 ) -> list[Probe]:
    """Public builder. Deterministic for a given (counts, seed)."""
    return _build_probes(counts, seed)


PROBES = _build_probes()
PRIMITIVES = tuple(dict.fromkeys(p.primitive for p in PROBES))


# ---------------------------------------------------------------------------
# prompt + scoring
# ---------------------------------------------------------------------------

_FORMAT_HINT = {
    "node_id": "Answer with a single node ID such as N07 and nothing else.",
    "id_pair": "Answer with exactly two node IDs separated by a comma, "
               "such as N07,N12, and nothing else.",
    "word": "Answer with a single lowercase word and nothing else.",
    "date": "Answer with a single date in YYYY-MM-DD form and nothing else.",
}


def build_pretest_prompt(probe: Probe, condition: str) -> str:
    parts = [
        "You are reading a single small panel from a project memory map.",
        "",
        "The record below is an image: a rendered 2D map of the project memory.",
    ]
    if condition == "primed":
        parts += ["", _PRIMER]
    parts += [
        "",
        "--- BEGIN RECORD ---",
        "[image above]",
        "--- END RECORD ---",
        "",
        f"Question: {probe.question}",
        "",
        _FORMAT_HINT[probe.scorer],
    ]
    return "\n".join(parts)


_ID_RE = re.compile(r"\bN(\d{2})\b")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def score_probe(probe: Probe, text: str) -> tuple[str | None, str]:
    """Returns (parsed, outcome). `unparseable` stays distinct from `wrong`."""
    t = (text or "").strip()
    if probe.scorer == "node_id":
        m = _ID_RE.search(t)
        if not m:
            return None, "unparseable"
        got = f"N{m.group(1)}"
        return got, ("correct" if got == probe.answer else "wrong")
    if probe.scorer == "id_pair":
        ids = [f"N{m}" for m in _ID_RE.findall(t)]
        if len(ids) < 2:
            return None, "unparseable"
        got = ",".join(sorted(ids[:2]))
        want = ",".join(sorted(probe.answer.split(",")))
        return got, ("correct" if got == want else "wrong")
    if probe.scorer == "date":
        m = _DATE_RE.search(t)
        if not m:
            return None, "unparseable"
        return m.group(1), ("correct" if m.group(1) == probe.answer else "wrong")
    if probe.scorer == "word":
        low = t.lower()
        hits = [w for w in probe.accepted if re.search(rf"\b{w.lower()}\b", low)]
        if len(hits) != 1:
            return (None, "unparseable") if not hits else (",".join(hits), "wrong")
        return hits[0], ("correct" if hits[0].lower() == probe.answer.lower() else "wrong")
    raise ValueError(f"unknown scorer {probe.scorer}")
