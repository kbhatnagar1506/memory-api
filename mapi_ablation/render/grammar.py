"""The visual grammar: fixed conventions a model should be able to cold-read.

Two things here are load-bearing and easy to get wrong.

**Geometry is in absolute authored pixels, not fractions of canvas width.**
An earlier version scaled every constant by width/1540, which quietly made the
authored-width sweep a no-op: scaling the content and the canvas together
produces the same aspect ratio, so the provider's downscale produces a
pixel-identical final image. With absolute geometry, a wider canvas genuinely
fits more nodes per row, which is a real mechanism with a real effect.

**Providers downscale by the LONG edge.** So vertical space is not free: every
extra sub-row makes the canvas taller, and a taller canvas is downscaled harder,
and glyph cap-height falls one-for-one. The consequence is that the useful
design target is to keep the authored long edge at or below the provider cap so
that no downscaling happens at all -- shrinking content to fit inside the cap
beats authoring big and letting the provider throw the pixels away.
`scripts/search_geometry.py` searches that frontier; DECISIONS.md quotes it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

REFERENCE_WIDTH = 1540

#: Working assumption, replaced by the measured value once `pretest --verify-resize`
#: has run against each provider. Only used to *design* the canvas, never to
#: report a result: reported cap-heights come from measured scaling.
ASSUMED_LONG_EDGE_CAP = 1568

# -- palette ---------------------------------------------------------------
# Colour is redundant with the stroke patterns below, never load-bearing on its
# own: every relation stays distinguishable in greyscale.
BG = (255, 255, 255)
BAND_FILL_A = (248, 249, 251)
BAND_FILL_B = (238, 241, 246)
BAND_RULE = (188, 196, 208)
BAND_LABEL = (60, 70, 88)
AXIS = (110, 120, 138)
AXIS_TEXT = (60, 70, 88)
NODE_FILL = (255, 255, 255)
NODE_BORDER = (24, 30, 42)
ID_TEXT = (12, 16, 24)
LABEL_TEXT = (44, 52, 66)
DATE_TEXT = (104, 114, 130)
CAUSED = (26, 66, 168)
CONTRADICTS = (188, 36, 48)
SUPERSEDES = (18, 118, 84)
LEGEND_BG = (255, 255, 255)
LEGEND_BORDER = (140, 150, 166)
LEGEND_TEXT = (24, 30, 42)
DEBUG_GRID = (255, 0, 255)
DEBUG_BOX = (0, 190, 190)

TYPE_ORDER = ("entity", "event", "decision", "preference")

RELATION_COLOUR = {
    "caused": CAUSED,
    "contradicts": CONTRADICTS,
    "supersedes": SUPERSEDES,
}

#: The three connector styles, exactly as the legend states them.
#: Arrowheads mean causality and nothing else, so `contradicts` gets end bars
#: rather than the double arrowhead "double-ended connector" might suggest.
CONNECTOR_LEGEND = (
    ("caused", "arrowhead points at the effect"),
    ("contradicts", "dashed, bar at both ends"),
    ("supersedes", "chevrons point at newer node"),
)


@dataclass(frozen=True)
class Geometry:
    """Every tunable number in the render, in absolute authored pixels."""

    # canvas frame
    margin_left: int = 128
    margin_right: int = 28
    margin_top: int = 54
    axis_height: int = 64

    # node box -- picked by scripts/search_geometry.py, not by taste
    node_w: int = 164
    node_h: int = 80
    node_radius: int = 10
    node_pad: int = 7
    label_max_lines: int = 2
    node_x_clearance: int = 8

    # band packing
    subrow_gap: int = 10
    band_pad: int = 14

    # type -> border weight. Widely separated because thin strokes are the
    # first thing to vanish under downscaling. Listed in the legend.
    border_weights: tuple[int, int, int, int] = (2, 4, 7, 10)

    # typography. font_id is the largest because the ID is what the model has
    # to read off and emit; labels are secondary and degrade first.
    font_id: int = 19
    font_label: int = 12
    font_date: int = 10
    font_band: int = 20
    font_axis: int = 13
    font_legend: int = 14
    font_title: int = 17

    # connectors
    arrow_head: int = 12
    dash_on: int = 9
    dash_off: int = 6
    chevron: int = 10
    edge_width: int = 2

    # legend block
    legend_pad: int = 12
    legend_line: int = 34
    legend_margin: int = 14
    legend_rows: int = 6
    legend_col_gap: int = 22
    legend_swatch_w: int = 52
    legend_type_col_w: int = 96

    def weight_for(self, node_type: str) -> int:
        return self.border_weights[TYPE_ORDER.index(node_type)]

    @property
    def max_border_weight(self) -> int:
        return max(self.border_weights)

    def scaled(self, k: float) -> "Geometry":
        """Uniformly scale the whole grammar (used by the resolution sweep)."""
        def s(v: int) -> int:
            return max(1, round(v * k))

        return replace(
            self,
            margin_left=s(self.margin_left),
            margin_right=s(self.margin_right),
            margin_top=s(self.margin_top),
            axis_height=s(self.axis_height),
            node_w=s(self.node_w),
            node_h=s(self.node_h),
            node_radius=s(self.node_radius),
            node_pad=s(self.node_pad),
            node_x_clearance=s(self.node_x_clearance),
            subrow_gap=s(self.subrow_gap),
            band_pad=s(self.band_pad),
            border_weights=tuple(s(w) for w in self.border_weights),  # type: ignore[arg-type]
            font_id=s(self.font_id),
            font_label=s(self.font_label),
            font_date=s(self.font_date),
            font_band=s(self.font_band),
            font_axis=s(self.font_axis),
            font_legend=s(self.font_legend),
            font_title=s(self.font_title),
            arrow_head=s(self.arrow_head),
            dash_on=s(self.dash_on),
            dash_off=s(self.dash_off),
            chevron=s(self.chevron),
            edge_width=s(self.edge_width),
            legend_pad=s(self.legend_pad),
            legend_line=s(self.legend_line),
            legend_margin=s(self.legend_margin),
            legend_col_gap=s(self.legend_col_gap),
            legend_swatch_w=s(self.legend_swatch_w),
            legend_type_col_w=s(self.legend_type_col_w),
        )


DEFAULT_GEOMETRY = Geometry()
