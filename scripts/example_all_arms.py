"""Render one graph in all four arms plus its canvas PNG, for eyeballing."""
from pathlib import Path
import json
from mapi_ablation.graph import generate_graph
from mapi_ablation.queries import generate_queries, annotate_with_layout
from mapi_ablation.arms import ARMS, build
from mapi_ablation.render import render, render_report, save, RenderConfig
from mapi_ablation.render.legibility import cap_heights_for
from mapi_ablation.prompts import build_prompt

SEED, N = 3, 40
out = Path("runs/_example"); out.mkdir(parents=True, exist_ok=True)
g = generate_graph(SEED, N)
qs = generate_queries(g)

payloads = {name: build(name, g) for name in ARMS}
for name, p in payloads.items():
    if p.kind == "text":
        (out / f"arm_{name}.txt").write_text(p.text)
    else:
        (out / f"arm_{name}.png").write_bytes(p.image_png)

cfg = RenderConfig.for_graph(N)
res = render(g, cfg); save(res, out / "canvas.png")
save(render(g, cfg, debug=True), out / "canvas_debug.png")
rep = render_report(res)
qs = annotate_with_layout(qs, res.layout.positions())

print(f"graph seed={SEED} n={N} fingerprint={g.fingerprint()}")
print("render:", json.dumps({k: v for k, v in rep.items() if k != 'positions'}))
for prov in ("anthropic", "gemini", "openai"):
    ch = cap_heights_for(cfg.geom, res.layout.width, res.layout.height, prov)
    print(f"  {prov:10s} scale={ch.scale:.3f} ({ch.provenance})  "
          f"post-resize cap px: id={ch.postresize_id} label={ch.postresize_label} "
          f"date={ch.postresize_date}")
print()
for name, p in payloads.items():
    print(f"arm {name:12s} {p.kind:5s} {p.size_note():>16s}  hash={p.payload_hash()}")
print()
print("queries:", len(qs), {k: v['sampled'] for k, v in qs.capacity.items()})
q = qs.by_type("root_cause")[0]
bp = build_prompt(payloads["canvas"], q.question, "cold")
(out / "prompt_canvas_cold.txt").write_text(bp.text)
(out / "prompt_json_primed.txt").write_text(
    build_prompt(payloads["json"], q.question, "primed").text)
print("\nexample question:", q.question, "-> answer", q.answer,
      f"(hops={q.hops}, pixel_dist={q.pixel_dist})")
