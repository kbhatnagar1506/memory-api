"""Deterministic graph -> coordinates.

Stability is a hard requirement here, not a nice-to-have, so read the two rules
before changing anything:

  * x is a FIXED global time->pixel map (EPOCH_START, EPOCH_SPAN_DAYS). It is
    never normalized to the graph's own min/max date, because that would make
    every node's x depend on every other node's date, so adding one node would
    shift the entire canvas.

  * y is a fixed band (domain, sorted by name) plus a sub-row. Band heights come
    from `subrows`, a *configuration* constant, not from how crowded a band
    happens to be -- so adding a node to Platform cannot shift Finance.

The only thing that can cascade is sub-row assignment inside one band, and only
among nodes whose x-intervals actually overlap. That residual cascade is
measured honestly by stability.py rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ..graph import DOMAINS, EPOCH_SPAN_DAYS, EPOCH_START, Graph
from . import grammar as G


#: Smallest sub-row count that renders all 10 seeds with zero collisions at
#: DEFAULT_GEOMETRY, measured by scripts/search_geometry.py. Keyed by
#: (n_nodes, authored width). A wrong value here costs either overlapping nodes
#: or wasted glyph pixels, so it is measured rather than guessed.
SUBROWS_MEASURED: dict[tuple[int, int], int] = {
    (20, 1540): 3, (20, 2560): 2,
    (40, 1540): 4, (40, 2560): 4,
    (80, 1540): 8, (80, 2560): 5,
}


@dataclass(frozen=True)
class RenderConfig:
    width: int = G.REFERENCE_WIDTH
    #: Sub-rows per band. Pinned by the caller so a mutated graph renders with
    #: exactly the same band geometry as its base graph.
    subrows: int = 4
    geom: G.Geometry = G.DEFAULT_GEOMETRY
    title: str = "Project memory graph"

    @staticmethod
    def for_graph(n_nodes: int, width: int = G.REFERENCE_WIDTH,
                  geom: G.Geometry | None = None, **kw) -> "RenderConfig":
        g = geom or G.DEFAULT_GEOMETRY
        measured = SUBROWS_MEASURED.get((n_nodes, width))
        if measured is not None and g == G.DEFAULT_GEOMETRY:
            return RenderConfig(width=width, subrows=measured, geom=g, **kw)
        # Fallback for configurations outside the measured table: mean row
        # density, inflated for clustering. Verify with search_geometry.py
        # before trusting it for a real run.
        slots = max(1, int((width - g.margin_left - g.margin_right)
                           // (g.node_w + g.node_x_clearance)))
        per_band = (n_nodes + len(DOMAINS) - 1) // len(DOMAINS)
        subrows = max(2, min(10, int(round(2.2 * per_band / slots)) + 1))
        return RenderConfig(width=width, subrows=subrows, geom=g, **kw)


@dataclass
class Placement:
    node_id: str
    cx: float
    cy: float
    w: float
    h: float
    subrow: int

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)


@dataclass
class Layout:
    config: RenderConfig
    width: int
    height: int
    placements: dict[str, Placement]
    bands: dict[str, tuple[float, float]]  # domain -> (y_top, y_bottom)
    axis_y: float
    plot_x0: float
    plot_x1: float
    ticks: list[tuple[float, str]] = field(default_factory=list)
    legend_rect: tuple[float, float, float, float] = (0, 0, 0, 0)
    #: Nodes that could not be placed without overlapping another node.
    collisions: list[str] = field(default_factory=list)

    @property
    def geom(self) -> G.Geometry:
        return self.config.geom

    def positions(self) -> dict[str, tuple[float, float]]:
        return {k: (p.cx, p.cy) for k, p in self.placements.items()}

    @property
    def long_edge(self) -> int:
        return max(self.width, self.height)


def _x_for_date(iso: str, geom: G.Geometry, width: int) -> float:
    """Fixed global time -> x. Never derived from the graph's own date range."""
    d = date.fromisoformat(iso)
    frac = (d - EPOCH_START).days / EPOCH_SPAN_DAYS
    frac = min(max(frac, 0.0), 1.0)
    x0 = geom.margin_left
    x1 = width - geom.margin_right
    return x0 + geom.node_w / 2 + frac * ((x1 - x0) - geom.node_w)


def _month_ticks(geom: G.Geometry, width: int, every: int = 2
                 ) -> list[tuple[float, str]]:
    ticks: list[tuple[float, str]] = []
    y, m = EPOCH_START.year, EPOCH_START.month
    end = EPOCH_START + timedelta(days=EPOCH_SPAN_DAYS)
    x0, x1 = geom.margin_left, width - geom.margin_right
    while True:
        cur = date(y, m, 1)
        if cur > end:
            break
        if cur >= EPOCH_START:
            frac = (cur - EPOCH_START).days / EPOCH_SPAN_DAYS
            ticks.append((x0 + frac * (x1 - x0), cur.strftime("%Y-%m")))
        m += every
        while m > 12:
            m -= 12
            y += 1
    return ticks


def _legend_width(geom: G.Geometry) -> int:
    """Measure the legend from its real text, so it can never clip itself.

    Guessing this from a character-count heuristic clipped the longest
    connector caption, which is exactly the line a cold-reading model needs.
    """
    from .draw import font  # local import: draw imports layout

    f = font(geom.font_legend)
    header = f.getlength("LEGEND") + f.getlength(
        "   horizontal = time, older at left;  band = domain"
    )
    col_a = geom.legend_swatch_w + 12 + max(
        [f.getlength(t) for t in G.TYPE_ORDER] + [f.getlength("border weight = type")]
    )
    col_b = geom.legend_swatch_w + 12 + max(
        [f.getlength(f"{rel}: {desc}") for rel, desc in G.CONNECTOR_LEGEND]
        + [f.getlength("connectors")]
    )
    body = col_a + geom.legend_col_gap + col_b
    return int(round(max(header, body) + 2 * geom.legend_pad))


def compute_layout(g: Graph, config: RenderConfig | None = None) -> Layout:
    cfg = config or RenderConfig.for_graph(g.n_nodes)
    geom = cfg.geom
    width = cfg.width

    row_pitch = geom.node_h + geom.subrow_gap
    band_h = cfg.subrows * row_pitch - geom.subrow_gap + 2 * geom.band_pad

    bands: dict[str, tuple[float, float]] = {}
    for i, d in enumerate(DOMAINS):
        y0 = geom.margin_top + i * band_h
        bands[d] = (y0, y0 + band_h)

    axis_y = geom.margin_top + len(DOMAINS) * band_h
    legend_h = geom.legend_rows * geom.legend_line + 2 * geom.legend_pad
    height = int(round(axis_y + geom.axis_height + legend_h + 2 * geom.legend_margin))

    legend_w = _legend_width(geom)
    lx1 = width - geom.legend_margin
    ly1 = height - geom.legend_margin
    legend_rect = (lx1 - legend_w, ly1 - legend_h, lx1, ly1)

    # -- sub-row packing, per band, in intrinsic (time) order ---------------
    placements: dict[str, Placement] = {}
    collisions: list[str] = []

    for d in DOMAINS:
        members = sorted(g.nodes_in_domain(d), key=lambda n: (n.t, n.id))
        occupied: list[list[tuple[float, float]]] = [[] for _ in range(cfg.subrows)]
        y0, _ = bands[d]
        for n in members:
            cx = _x_for_date(n.date, geom, width)
            half = geom.node_w / 2 + geom.node_x_clearance
            lo, hi = cx - half, cx + half
            chosen = None
            for r in range(cfg.subrows):
                if all(hi <= a or lo >= b for a, b in occupied[r]):
                    chosen = r
                    break
            if chosen is None:
                # Every sub-row conflicts. Pick the least-bad one and record it,
                # rather than silently stacking unreadable boxes.
                def worst_overlap(r: int) -> float:
                    return max(
                        (min(hi, b) - max(lo, a) for a, b in occupied[r]), default=0.0
                    )

                chosen = min(range(cfg.subrows), key=worst_overlap)
                collisions.append(n.id)
            occupied[chosen].append((lo, hi))
            cy = y0 + geom.band_pad + chosen * row_pitch + geom.node_h / 2
            placements[n.id] = Placement(
                n.id, cx, cy, geom.node_w, geom.node_h, chosen
            )

    return Layout(
        config=cfg,
        width=width,
        height=height,
        placements=placements,
        bands=bands,
        axis_y=axis_y,
        plot_x0=geom.margin_left,
        plot_x1=width - geom.margin_right,
        ticks=_month_ticks(geom, width),
        legend_rect=legend_rect,
        collisions=collisions,
    )
