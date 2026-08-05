"""Arm C -- spatial organization, in text.

This arm is the one that can kill the product without killing the finding. It
applies the canvas's organizing principles -- group by domain, order by time,
separate relations by type -- using nothing but text layout. No raster pixels.

If C is roughly as good as D, then the invention is the *layout*, not the image,
and the right product is a text formatter, which is far cheaper to build and to
adopt. That is a legitimate outcome and the report must be able to detect it, so
this arm gets the same care as the canvas rather than being a strawman.
"""

from __future__ import annotations

import re

from ..graph import DOMAINS, Graph
from . import Payload, normalize_edge

NAME = "layout_text"
DESCRIPTOR = "a structured outline of the project memory"

_NODE_RE = re.compile(
    r"^(N\d{2}) · (\d{4}-\d{2}-\d{2}) · (.+?) \[(\w+)\]$", re.M
)
_EDGE_RE = re.compile(r"^(N\d{2}) -> (N\d{2})$", re.M)
_DOMAIN_RE = re.compile(r"^## (\w+)$", re.M)
_REL_RE = re.compile(r"^### (\w+)$", re.M)


def build(g: Graph, **kw) -> Payload:
    lines: list[str] = ["# Nodes, grouped by domain, oldest first"]
    for d in DOMAINS:
        lines.append("")
        lines.append(f"## {d}")
        for n in sorted(g.nodes_in_domain(d), key=lambda x: x.t):
            lines.append(f"{n.id} · {n.date} · {n.label} [{n.type}]")

    lines.append("")
    lines.append("# Edges, grouped by relation")
    for rel in ("caused", "contradicts", "supersedes"):
        lines.append("")
        lines.append(f"### {rel}")
        for e in sorted(g.rel_edges(rel), key=lambda e: (e.src, e.dst)):
            lines.append(f"{e.src} -> {e.dst}")

    return Payload(
        arm=NAME,
        kind="text",
        descriptor=DESCRIPTOR,
        text="\n".join(lines),
        meta={"n_lines": len(lines)},
    )


def recover(p: Payload) -> tuple[set, set]:
    text = p.text or ""

    # A node's domain comes from the "## <domain>" header above it, which is
    # exactly the information the arm encodes spatially.
    domain_at: list[tuple[int, str]] = [
        (m.start(), m.group(1)) for m in _DOMAIN_RE.finditer(text)
    ]

    def domain_for(pos: int) -> str:
        current = ""
        for start, name in domain_at:
            if start < pos:
                current = name
            else:
                break
        return current

    nodes = set()
    for m in _NODE_RE.finditer(text):
        nid, date, label, ntype = m.groups()
        nodes.add((nid, label, ntype, domain_for(m.start()), date))

    rel_at: list[tuple[int, str]] = [
        (m.start(), m.group(1)) for m in _REL_RE.finditer(text)
    ]

    def rel_for(pos: int) -> str:
        current = ""
        for start, name in rel_at:
            if start < pos:
                current = name
            else:
                break
        return current

    edges = set()
    for m in _EDGE_RE.finditer(text):
        a, b = m.groups()
        edges.add(normalize_edge(a, b, rel_for(m.start())))
    return nodes, edges
