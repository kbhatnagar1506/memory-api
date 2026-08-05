"""Arm B -- compact JSON.

The strong text baseline. Machine-readable, no wasted tokens, every field
explicit. If the canvas cannot beat this, it is not beating "text".
"""

from __future__ import annotations

import json

from ..graph import Graph
from . import Payload, normalize_edge

NAME = "json"
DESCRIPTOR = "a JSON record of the project memory"


def build(g: Graph, **kw) -> Payload:
    doc = {
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "type": n.type,
                "domain": n.domain,
                "date": n.date,
            }
            for n in sorted(g.nodes, key=lambda x: x.t)
        ],
        "edges": [
            {"from": e.src, "to": e.dst, "rel": e.rel}
            for e in sorted(g.edges, key=lambda e: (e.rel, e.src, e.dst))
        ],
    }
    return Payload(
        arm=NAME,
        kind="text",
        descriptor=DESCRIPTOR,
        text=json.dumps(doc, separators=(",", ":")),
        meta={"n_nodes": len(doc["nodes"]), "n_edges": len(doc["edges"])},
    )


def recover(p: Payload) -> tuple[set, set]:
    doc = json.loads(p.text or "{}")
    nodes = {
        (n["id"], n["label"], n["type"], n["domain"], n["date"])
        for n in doc.get("nodes", [])
    }
    edges = {
        normalize_edge(e["from"], e["to"], e["rel"]) for e in doc.get("edges", [])
    }
    return nodes, edges
