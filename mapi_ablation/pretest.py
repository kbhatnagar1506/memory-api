"""Stage 0 pretest -- the gate.

40 tiny canvases, each testing exactly ONE grammar primitive in isolation with
no distractors. If a frontier model cannot read a single primitive off a clean
canvas at >=95%, the grammar itself is broken and the full ablation would only
be measuring that.

These canvases go through the *same* renderer as the real ones. A mock would
test a drawing that does not exist.

Answers here are not always node IDs (a date, a domain name, a relation word),
so each primitive carries its own scorer. That is the one place the pretest
deliberately differs from the main harness, which is single-format by design.
"""

from __future__ import annotations

import io
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

    @property
    def pid(self) -> str:
        return f"{self.primitive}-{self.index:02d}"

    def png(self, width: int = 1540) -> bytes:
        cfg = RenderConfig(width=width, subrows=1)
        res = render(self.graph, cfg)
        render_report(res)  # assert nothing was truncated
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


def _pair(i: int, same_domain: bool = False) -> tuple[Node, Node]:
    """Two nodes with a deterministic, well-separated position on the canvas."""
    d1 = _DOMAINS4[i % 4]
    d2 = d1 if same_domain else _DOMAINS4[(i + 1 + (i % 2)) % 4]
    older_date = _DATES[i % 5]
    newer_date = _DATES[5 + (i % 5)]
    a = _node(f"N{2 * i + 1:02d}", _LABELS[i % len(_LABELS)],
              TYPE_ORDER[i % 4], d1, 1, older_date)
    b = _node(f"N{2 * i + 2:02d}", _LABELS[(i + 3) % len(_LABELS)],
              TYPE_ORDER[(i + 2) % 4], d2, 2, newer_date)
    return a, b


def _build_probes() -> list[Probe]:
    probes: list[Probe] = []

    for i in range(5):  # 1. time order
        a, b = _pair(i)
        probes.append(Probe(
            "time_order", i, _tiny([a, b], []),
            f"Which of {a.id} and {b.id} is older?", a.id, "node_id",
        ))

    for i in range(5):  # 2. causal direction
        a, b = _pair(i)
        probes.append(Probe(
            "causal_direction", i,
            _tiny([a, b], [Edge(a.id, b.id, "caused")]),
            f"One of {a.id} and {b.id} caused the other. Which one is the cause?",
            a.id, "node_id",
        ))

    for i in range(5):  # 3. same band
        a, b = _pair(i, same_domain=True)
        third = _node(f"N{2 * i + 30:02d}", _LABELS[(i + 5) % len(_LABELS)],
                      TYPE_ORDER[(i + 1) % 4],
                      _DOMAINS4[(_DOMAINS4.index(a.domain) + 2) % 4], 3, _DATES[i % 5])
        probes.append(Probe(
            "same_band", i, _tiny([a, b, third], []),
            "Which two of the three nodes are in the same domain band? "
            "Answer with their two node IDs.",
            f"{a.id},{b.id}", "id_pair",
        ))

    for i, rel in enumerate(  # 4. edge type
        ["caused", "contradicts", "supersedes", "caused", "contradicts"]
    ):
        a, b = _pair(i, same_domain=(rel == "contradicts"))
        src, dst = (b, a) if rel == "supersedes" else (a, b)
        probes.append(Probe(
            "edge_type", i, _tiny([a, b], [Edge(src.id, dst.id, rel)]),
            f"What is the relationship drawn between {a.id} and {b.id}? "
            "Answer with one word: caused, contradicts, or supersedes.",
            rel, "word",
            accepted=("caused", "contradicts", "supersedes"),
        ))

    for i in range(5):  # 5. border weight -> type
        a, b = _pair(i)
        probes.append(Probe(
            "border_weight", i, _tiny([a, b], []),
            f"Using the border weight legend, what is the type of node {a.id}? "
            "Answer with one word.",
            a.type, "word", accepted=TYPE_ORDER,
        ))

    for i in range(5):  # 6. date read
        a, b = _pair(i)
        probes.append(Probe(
            "date_read", i, _tiny([a, b], []),
            f"What date is shown on node {b.id}? Answer in YYYY-MM-DD form.",
            b.date, "date",
        ))

    for i in range(5):  # 7. band label
        a, b = _pair(i)
        probes.append(Probe(
            "band_label", i, _tiny([a, b], []),
            f"Which domain band is node {b.id} in? Answer with the band name.",
            b.domain, "word", accepted=_DOMAINS4,
        ))

    for i in range(5):  # 8. supersede direction
        a, b = _pair(i)
        probes.append(Probe(
            "supersede_direction", i,
            _tiny([a, b], [Edge(b.id, a.id, "supersedes")]),
            f"One of {a.id} and {b.id} supersedes the other. "
            "Which node is the one doing the superseding?",
            b.id, "node_id",
        ))

    return probes


PROBES = _build_probes()
PRIMITIVES = tuple(dict.fromkeys(p.primitive for p in PROBES))

assert len(PROBES) == 40, f"expected 40 probes, built {len(PROBES)}"


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
