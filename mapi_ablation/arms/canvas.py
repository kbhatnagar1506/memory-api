"""Arm D -- the rendered 2D canvas. The artifact actually under test.

The payload is a PNG. Because the parity test cannot OCR a PNG, `recover()`
reads what the renderer recorded itself as having drawn (`depicted_nodes` /
`depicted_edges`), which comes from the same code path that emits the text. A
truncation or drop bug therefore shows up in parity rather than hiding.
"""

from __future__ import annotations

import io

from ..graph import Graph
from ..render.draw import cap_height, render, render_report
from ..render.layout import RenderConfig
from . import Payload, normalize_edge

NAME = "canvas"
DESCRIPTOR = "an image: a rendered 2D map of the project memory"


def build(g: Graph, width: int = 1540, config: RenderConfig | None = None,
          debug: bool = False, **kw) -> Payload:
    cfg = config or RenderConfig.for_graph(g.n_nodes, width=width)
    res = render(g, cfg, debug=debug)
    report = render_report(res)  # asserts no label was truncated

    buf = io.BytesIO()
    res.image.save(buf, format="PNG", optimize=False)

    geom = cfg.geom
    return Payload(
        arm=NAME,
        kind="image",
        descriptor=DESCRIPTOR,
        image_png=buf.getvalue(),
        meta={
            **report,
            "authored_width": res.layout.width,
            "authored_height": res.layout.height,
            "subrows": cfg.subrows,
            "positions": res.layout.positions(),
            "authored_cap_px": {
                "id": cap_height(geom.font_id, bold=True),
                "label": cap_height(geom.font_label),
                "date": cap_height(geom.font_date),
            },
            # recovery sets, kept out of the image itself
            "_nodes": [
                (d["id"], d["label"], d["type"], d["domain"], d["date"])
                for d in res.depicted_nodes.values()
            ],
            "_edges": list(res.depicted_edges),
        },
    )


def recover(p: Payload) -> tuple[set, set]:
    nodes = {tuple(t) for t in p.meta.get("_nodes", [])}
    edges = {normalize_edge(*t) for t in p.meta.get("_edges", [])}
    return nodes, edges
