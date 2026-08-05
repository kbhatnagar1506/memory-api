"""Measure the subrow / canvas-height tradeoff. Output quoted in DECISIONS.md.

More sub-rows -> fewer node collisions, but a taller canvas, and every provider
downscales by the LONG edge, so height is paid for directly in glyph pixels.
"""
from mapi_ablation.graph import generate_graph
from mapi_ablation.render import render, RenderConfig
from mapi_ablation.render.draw import font
from mapi_ablation.render import grammar as G

SEEDS = range(10)
LONG_EDGE_CAP = 1568  # Anthropic; verified empirically later by legibility.py

for n in (20, 40, 80):
    for sub in range(2, 8):
        tot = worst = 0
        h = w = 0
        for s in SEEDS:
            g = generate_graph(s, n)
            r = render(g, RenderConfig(width=1540, subrows=sub))
            tot += len(r.collisions)
            worst = max(worst, len(r.collisions))
            h, w = r.layout.height, r.layout.width
        scale = min(1.0, LONG_EDGE_CAP / max(h, w))
        sc = G.scale_for_width(1540)
        cap_id = font(sc(G.FONT_ID), bold=True).getbbox("H")
        cap_lab = font(sc(G.FONT_LABEL)).getbbox("H")
        id_cap = (cap_id[3] - cap_id[1]) * scale
        lab_cap = (cap_lab[3] - cap_lab[1]) * scale
        print(f"n={n:3d} subrows={sub}  collisions tot={tot:3d} worst={worst:2d}  "
              f"canvas={w}x{h}  postresize_scale={scale:.3f}  "
              f"id_cap={id_cap:.1f}px label_cap={lab_cap:.1f}px")
    print()
