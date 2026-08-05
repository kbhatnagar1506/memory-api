"""Arm A -- prose.

A flat natural-language summary in time order: no headers, no grouping, no
visual structure. This is the naive baseline that most memory systems
effectively produce when they concatenate retrieved facts.

Edge sentences are interleaved at the point of their source node rather than
collected at the end, because collecting them would be a form of structure and
this arm is defined by its absence.
"""

from __future__ import annotations

import re

from ..graph import Graph
from . import Payload, normalize_edge

NAME = "prose"
DESCRIPTOR = "a written summary of the project memory"

_NODE_RE = re.compile(
    r"\b(N\d{2}) is a (\w+) (\w+) recorded on (\d{4}-\d{2}-\d{2}): ([^.]+)\."
)
_CAUSED_RE = re.compile(r"\b(N\d{2}) caused (N\d{2})\.")
_CONTRA_RE = re.compile(r"\b(N\d{2}) contradicts (N\d{2})\.")
_SUPER_RE = re.compile(r"\b(N\d{2}) supersedes (N\d{2})\.")


def build(g: Graph, **kw) -> Payload:
    caused = {e.src: e.dst for e in g.rel_edges("caused")}
    supersedes = {e.src: e.dst for e in g.rel_edges("supersedes")}
    # Emit a contradiction once, at whichever endpoint comes first in time.
    contra_at: dict[str, str] = {}
    for e in g.rel_edges("contradicts"):
        first, second = sorted((e.src, e.dst), key=lambda n: g.node(n).t)
        contra_at[first] = second

    parts: list[str] = []
    for n in sorted(g.nodes, key=lambda x: x.t):
        parts.append(
            f"{n.id} is a {n.domain} {n.type} recorded on {n.date}: {n.label}."
        )
        if n.id in caused:
            parts.append(f"{n.id} caused {caused[n.id]}.")
        if n.id in contra_at:
            parts.append(f"{n.id} contradicts {contra_at[n.id]}.")
        if n.id in supersedes:
            parts.append(f"{n.id} supersedes {supersedes[n.id]}.")

    return Payload(
        arm=NAME,
        kind="text",
        descriptor=DESCRIPTOR,
        text=" ".join(parts),
        meta={"n_sentences": len(parts)},
    )


def recover(p: Payload) -> tuple[set, set]:
    text = p.text or ""
    nodes = {
        (nid, label.strip(), ntype, domain, date)
        for nid, domain, ntype, date, label in _NODE_RE.findall(text)
    }
    edges = set()
    for a, b in _CAUSED_RE.findall(text):
        edges.add(normalize_edge(a, b, "caused"))
    for a, b in _CONTRA_RE.findall(text):
        edges.add(normalize_edge(a, b, "contradicts"))
    for a, b in _SUPER_RE.findall(text):
        edges.add(normalize_edge(a, b, "supersedes"))
    return nodes, edges
