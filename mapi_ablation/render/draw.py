"""Coordinates -> PIL image. Deterministic: same graph in, same PNG bytes out.

Fonts come from matplotlib's bundled DejaVu Sans rather than a system font, so a
render on this laptop is byte-identical to a render on CI. See DECISIONS.md.

Node labels are wrapped, never truncated. Truncation would silently drop
information that the text arms still carry, which would break arm parity -- the
one property the whole experiment rests on. `render_report()` asserts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..graph import DOMAINS, Graph
from . import grammar as G
from .layout import Layout, RenderConfig, compute_layout


@lru_cache(maxsize=None)
def _font_path(bold: bool = False) -> str:
    import matplotlib

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    if not p.exists():  # pragma: no cover - would mean a broken matplotlib
        raise FileNotFoundError(f"bundled font missing: {p}")
    return str(p)


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_font_path(bold), max(1, int(size)))


def cap_height(size: int, bold: bool = False) -> int:
    """Rendered cap-height in px of the font actually used, from its metrics."""
    box = font(size, bold).getbbox("H")
    return box[3] - box[1]


@dataclass
class RenderResult:
    image: Image.Image
    layout: Layout
    #: What the render actually depicts, per node. The arm-parity test cannot
    #: OCR a PNG, so it checks this instead -- it is produced by the same code
    #: path that draws the text, so a truncation bug shows up here too.
    depicted_nodes: dict[str, dict] = field(default_factory=dict)
    depicted_edges: list[tuple[str, str, str]] = field(default_factory=list)
    #: Labels that would not fit in label_max_lines. Must stay empty.
    overflowed_labels: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _wrap(text: str, f: ImageFont.FreeTypeFont, max_w: float, max_lines: int
          ) -> tuple[list[str], bool]:
    """Greedy word wrap. Returns (lines, overflowed). Never truncates."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if f.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    overflow = len(lines) > max_lines or any(f.getlength(l) > max_w for l in lines)
    return lines, overflow


def _clip_to_box(cx: float, cy: float, w: float, h: float,
                 tx: float, ty: float) -> tuple[float, float]:
    """Point where the ray (cx,cy)->(tx,ty) leaves the box centred at (cx,cy)."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2) / abs(dx) if dx else math.inf
    sy = (h / 2) / abs(dy) if dy else math.inf
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def _dashed_line(d: ImageDraw.ImageDraw, p0, p1, colour, width, on, off) -> None:
    x0, y0 = p0
    x1, y1 = p1
    total = math.hypot(x1 - x0, y1 - y0)
    if total == 0:
        return
    ux, uy = (x1 - x0) / total, (y1 - y0) / total
    pos = 0.0
    while pos < total:
        seg = min(on, total - pos)
        d.line(
            [x0 + ux * pos, y0 + uy * pos,
             x0 + ux * (pos + seg), y0 + uy * (pos + seg)],
            fill=colour,
            width=width,
        )
        pos += on + off


def _arrowhead(d: ImageDraw.ImageDraw, tip, frm, colour, size) -> None:
    ang = math.atan2(tip[1] - frm[1], tip[0] - frm[0])
    spread = math.radians(26)
    p1 = (tip[0] - size * math.cos(ang - spread), tip[1] - size * math.sin(ang - spread))
    p2 = (tip[0] - size * math.cos(ang + spread), tip[1] - size * math.sin(ang + spread))
    d.polygon([tip, p1, p2], fill=colour)


def _end_bar(d: ImageDraw.ImageDraw, at, frm, colour, size, width) -> None:
    """Perpendicular tick: marks a connector end without implying direction."""
    ang = math.atan2(at[1] - frm[1], at[0] - frm[0]) + math.pi / 2
    dx, dy = math.cos(ang) * size / 2, math.sin(ang) * size / 2
    d.line([at[0] - dx, at[1] - dy, at[0] + dx, at[1] + dy], fill=colour, width=width)


def _chevrons(d: ImageDraw.ImageDraw, at, frm, colour, size, width) -> None:
    """Two nested chevrons pointing at `at`. Marks the newer node."""
    ang = math.atan2(at[1] - frm[1], at[0] - frm[0])
    spread = math.radians(38)
    for k in (0, 1):
        tip = (at[0] - math.cos(ang) * size * 0.9 * k,
               at[1] - math.sin(ang) * size * 0.9 * k)
        for sgn in (-1, 1):
            a = ang + sgn * spread
            d.line(
                [tip[0], tip[1],
                 tip[0] - size * math.cos(a), tip[1] - size * math.sin(a)],
                fill=colour,
                width=width,
            )


# ---------------------------------------------------------------------------
# the render
# ---------------------------------------------------------------------------


def render(g: Graph, config: RenderConfig | None = None, debug: bool = False
           ) -> RenderResult:
    lay = compute_layout(g, config)
    geom = lay.geom
    img = Image.new("RGB", (lay.width, lay.height), G.BG)
    d = ImageDraw.Draw(img)

    f_id = font(geom.font_id, bold=True)
    f_label = font(geom.font_label)
    f_date = font(geom.font_date)
    f_band = font(geom.font_band, bold=True)
    f_axis = font(geom.font_axis)
    f_legend = font(geom.font_legend)
    f_title = font(geom.font_title, bold=True)

    # -- bands -------------------------------------------------------------
    for i, dom in enumerate(DOMAINS):
        y0, y1 = lay.bands[dom]
        d.rectangle([0, y0, lay.width, y1],
                    fill=G.BAND_FILL_A if i % 2 == 0 else G.BAND_FILL_B)
        d.line([0, y0, lay.width, y0], fill=G.BAND_RULE, width=2)
        d.text((12, (y0 + y1) / 2), dom, font=f_band, fill=G.BAND_LABEL, anchor="lm")
    d.line([0, lay.axis_y, lay.width, lay.axis_y], fill=G.BAND_RULE, width=2)

    # -- title + axis meaning ---------------------------------------------
    d.text((12, 14), lay.config.title, font=f_title, fill=G.LEGEND_TEXT)
    d.text(
        (lay.plot_x1, 18),
        "HORIZONTAL AXIS = TIME.   older <--- left      right ---> newer",
        font=f_axis,
        fill=G.AXIS_TEXT,
        anchor="ra",
    )
    d.text((lay.plot_x1, 18 + geom.font_axis + 4),
           "HORIZONTAL BANDS = DOMAIN.   legend at bottom right",
           font=f_axis, fill=G.AXIS_TEXT, anchor="ra")

    # -- time ruler --------------------------------------------------------
    ruler_y = lay.axis_y + 22
    d.line([lay.plot_x0, ruler_y, lay.plot_x1, ruler_y], fill=G.AXIS, width=2)
    for x, label in lay.ticks:
        d.line([x, ruler_y - 6, x, ruler_y + 6], fill=G.AXIS, width=2)
        d.text((x, ruler_y + 9), label, font=f_axis, fill=G.AXIS_TEXT, anchor="ma")
    d.text((12, ruler_y), "TIME", font=f_axis, fill=G.AXIS_TEXT, anchor="lm")

    # -- edges, under the nodes -------------------------------------------
    depicted_edges: list[tuple[str, str, str]] = []
    ew = geom.edge_width
    for e in g.edges:
        a, b = lay.placements[e.src], lay.placements[e.dst]
        p0 = _clip_to_box(a.cx, a.cy, a.w, a.h, b.cx, b.cy)
        p1 = _clip_to_box(b.cx, b.cy, b.w, b.h, a.cx, a.cy)
        colour = G.RELATION_COLOUR[e.rel]
        if e.rel == "caused":
            d.line([p0, p1], fill=colour, width=ew)
            _arrowhead(d, p1, p0, colour, geom.arrow_head)
        elif e.rel == "contradicts":
            _dashed_line(d, p0, p1, colour, ew, geom.dash_on, geom.dash_off)
            _end_bar(d, p0, p1, colour, geom.arrow_head, ew)
            _end_bar(d, p1, p0, colour, geom.arrow_head, ew)
        else:  # supersedes: src is the newer node, chevrons mark that end
            d.line([p0, p1], fill=colour, width=ew)
            _chevrons(d, p0, p1, colour, geom.chevron, ew)
        depicted_edges.append((e.src, e.dst, e.rel))

    # -- nodes -------------------------------------------------------------
    depicted: dict[str, dict] = {}
    overflowed: list[str] = []
    # Text is inset by the widest possible border so a 2-line label stays 2
    # lines regardless of the node's type.
    inset = geom.node_pad + geom.max_border_weight
    for n in g.nodes:
        p = lay.placements[n.id]
        x0, y0, x1, y1 = p.box
        bw = geom.weight_for(n.type)
        d.rounded_rectangle([x0, y0, x1, y1], radius=geom.node_radius,
                            fill=G.NODE_FILL, outline=G.NODE_BORDER, width=bw)
        tx = x0 + inset
        ty = y0 + inset * 0.7
        # ID left, date right-aligned on the SAME line: one text row instead of
        # two, which is what keeps the canvas short enough to stay inside the
        # provider's long-edge cap and avoid downscaling entirely.
        d.text((tx, ty), n.id, font=f_id, fill=G.ID_TEXT)
        d.text((x1 - inset, ty + geom.font_id * 0.3), n.date,
               font=f_date, fill=G.DATE_TEXT, anchor="ra")
        ty += geom.font_id * 1.18

        max_w = (x1 - x0) - 2 * inset
        lines, over = _wrap(n.label, f_label, max_w, geom.label_max_lines)
        if over:
            overflowed.append(n.id)
        for line in lines:
            d.text((tx, ty), line, font=f_label, fill=G.LABEL_TEXT)
            ty += geom.font_label * 1.2

        depicted[n.id] = {
            "id": n.id,
            "label": " ".join(lines),
            "date": n.date,
            "type": n.type,
            "domain": n.domain,
        }

    _draw_legend(d, lay, f_legend)

    if debug:
        _draw_debug(d, lay)

    return RenderResult(
        image=img,
        layout=lay,
        depicted_nodes=depicted,
        depicted_edges=depicted_edges,
        overflowed_labels=overflowed,
        collisions=lay.collisions,
    )


def _draw_legend(d: ImageDraw.ImageDraw, lay: Layout, f) -> None:
    """Two columns, fixed bottom-right corner, identical in every render."""
    geom = lay.geom
    x0, y0, x1, y1 = lay.legend_rect
    d.rectangle([x0, y0, x1, y1], fill=G.LEGEND_BG, outline=G.LEGEND_BORDER, width=2)
    pad, line_h = geom.legend_pad, geom.legend_line
    swatch, gap, ew = geom.legend_swatch_w, 12, geom.edge_width

    left = x0 + pad
    d.text((left, y0 + pad), "LEGEND", font=f, fill=G.LEGEND_TEXT)
    d.text((left + f.getlength("LEGEND   "), y0 + pad),
           "horizontal = time, older at left;  band = domain",
           font=f, fill=G.LEGEND_TEXT)

    col_a = left
    type_col_w = max(
        [f.getlength(t) for t in G.TYPE_ORDER] + [f.getlength("border weight = type")]
    )
    col_b = left + swatch + gap + type_col_w + geom.legend_col_gap
    d.text((col_a, y0 + pad + line_h), "border weight = type",
           font=f, fill=G.LEGEND_TEXT)
    d.text((col_b, y0 + pad + line_h), "connectors", font=f, fill=G.LEGEND_TEXT)

    for i, t in enumerate(G.TYPE_ORDER):
        y = y0 + pad + (i + 2) * line_h
        d.rounded_rectangle([col_a, y + 3, col_a + swatch, y + line_h - 3],
                            radius=4, fill=G.NODE_FILL, outline=G.NODE_BORDER,
                            width=geom.weight_for(t))
        d.text((col_a + swatch + gap, y), t, font=f, fill=G.LEGEND_TEXT)

    for i, (rel, desc) in enumerate(G.CONNECTOR_LEGEND):
        y = y0 + pad + (i + 2) * line_h
        cy = y + line_h / 2 - 2
        a, b = (col_b, cy), (col_b + swatch, cy)
        colour = G.RELATION_COLOUR[rel]
        if rel == "contradicts":
            _dashed_line(d, a, b, colour, ew, geom.dash_on, geom.dash_off)
            _end_bar(d, a, b, colour, geom.arrow_head, ew)
            _end_bar(d, b, a, colour, geom.arrow_head, ew)
        else:
            d.line([a, b], fill=colour, width=ew)
            if rel == "caused":
                _arrowhead(d, b, a, colour, geom.arrow_head)
            else:
                _chevrons(d, b, a, colour, geom.chevron, ew)
        d.text((col_b + swatch + gap, y), f"{rel}: {desc}", font=f, fill=G.LEGEND_TEXT)


def _draw_debug(d: ImageDraw.ImageDraw, lay: Layout) -> None:
    for x, _ in lay.ticks:
        d.line([x, 0, x, lay.axis_y], fill=G.DEBUG_GRID, width=1)
    for y0, y1 in lay.bands.values():
        d.line([0, y0, lay.width, y0], fill=G.DEBUG_GRID, width=2)
        d.line([0, y1, lay.width, y1], fill=G.DEBUG_GRID, width=2)
    for p in lay.placements.values():
        d.rectangle(list(p.box), outline=G.DEBUG_BOX, width=2)
        d.line([p.cx - 5, p.cy, p.cx + 5, p.cy], fill=G.DEBUG_BOX, width=2)


def render_report(res: RenderResult) -> dict:
    """Machine-checkable summary of what the render actually depicts."""
    assert not res.overflowed_labels, (
        f"labels overflowed their node box: {res.overflowed_labels}. "
        "Truncating them would break arm parity."
    )
    geom = res.layout.geom
    return {
        "width": res.layout.width,
        "height": res.layout.height,
        "long_edge": res.layout.long_edge,
        "subrows": res.layout.config.subrows,
        "n_nodes": len(res.depicted_nodes),
        "n_edges": len(res.depicted_edges),
        "collisions": res.collisions,
        "authored_cap_px": {
            "id": cap_height(geom.font_id, bold=True),
            "label": cap_height(geom.font_label),
            "date": cap_height(geom.font_date),
        },
    }


def save(res: RenderResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    res.image.save(p, format="PNG", optimize=False)
    return p
