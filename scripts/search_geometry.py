"""Search the density/legibility frontier. Output is quoted in DECISIONS.md.

The constraint that drives everything: providers downscale by the LONG edge. So
the authored canvas competes against itself -- more sub-rows means fewer
overlapping nodes but a taller canvas, and a taller canvas is downscaled harder,
and post-resize glyph cap-height falls one-for-one with that.

The objective here is therefore: among geometries that render all seeds with
ZERO collisions and ZERO label overflow, maximise post-resize label cap-height.

Run:  PYTHONPATH=. .venv/bin/python scripts/search_geometry.py
"""

from __future__ import annotations

import itertools
from dataclasses import replace

from mapi_ablation.graph import generate_graph
from mapi_ablation.render import grammar as G
from mapi_ablation.render.draw import cap_height, render
from mapi_ablation.render.layout import RenderConfig

SEEDS = range(10)
CAP = G.ASSUMED_LONG_EDGE_CAP


def evaluate(geom: G.Geometry, width: int, n_nodes: int, subrows: int) -> dict:
    collisions = 0
    overflow = 0
    h = w = 0
    for s in SEEDS:
        g = generate_graph(s, n_nodes)
        r = render(g, RenderConfig(width=width, subrows=subrows, geom=geom))
        collisions += len(r.collisions)
        overflow += len(r.overflowed_labels)
        w, h = r.layout.width, r.layout.height
    scale = min(1.0, CAP / max(w, h))
    return {
        "collisions": collisions,
        "overflow": overflow,
        "w": w,
        "h": h,
        "scale": scale,
        "id_cap": cap_height(geom.font_id, bold=True) * scale,
        "label_cap": cap_height(geom.font_label) * scale,
    }


def search(n_nodes: int, width: int) -> None:
    print(f"\n=== n_nodes={n_nodes} authored_width={width} ===")
    best = None
    rows = []
    for node_w, font_label in itertools.product(
        (124, 136, 150, 164, 178, 196), (11, 12, 13, 14, 16)
    ):
        font_id = round(font_label * 1.62)
        node_h = int(round(font_id * 1.18 + 2 * font_label * 1.2 + 2 * (7 + 10) * 0.85))
        geom = replace(
            G.DEFAULT_GEOMETRY,
            node_w=node_w,
            node_h=node_h,
            font_label=font_label,
            font_id=font_id,
            font_date=max(10, font_label - 2),
        )
        for subrows in range(2, 10):
            r = evaluate(geom, width, n_nodes, subrows)
            if r["overflow"]:
                continue
            rows.append((node_w, font_label, node_h, subrows, r))
            if r["collisions"] == 0:
                if best is None or r["label_cap"] > best[4]["label_cap"]:
                    best = (node_w, font_label, node_h, subrows, r)
                break  # smallest collision-free subrow count for this geometry
    def fmt(node_w, fl, nh, sub, r) -> str:
        return (
            f"node_w={node_w:3d} font_label={fl:2d} node_h={nh:3d} subrows={sub} "
            f"canvas={r['w']}x{r['h']:4d} scale={r['scale']:.3f} "
            f"collisions={r['collisions']:3d} "
            f"id_cap={r['id_cap']:4.1f} label_cap={r['label_cap']:4.1f}"
        )

    clean = [r for r in rows if r[4]["collisions"] == 0]
    if not clean:
        print("  NO collision-free geometry in the search space. Least-bad:")
        for row in sorted(rows, key=lambda x: x[4]["collisions"])[:3]:
            print("   ", fmt(*row))
        return
    print(f"  {len(clean)} collision-free geometries. Top by post-resize label cap:")
    for row in sorted(clean, key=lambda x: -x[4]["label_cap"])[:4]:
        print("   ", fmt(*row))


if __name__ == "__main__":
    for n in (20, 40, 80):
        for w in (1540, 2560):
            search(n, w)
