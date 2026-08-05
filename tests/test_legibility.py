"""Legibility measurement.

The point of these tests is that no reported cap-height is ever an authored
cap-height wearing a disguise. Every results row must carry a post-resize number
and say whether the scaling behind it was assumed or measured.
"""

from __future__ import annotations

import pytest

from mapi_ablation.arms import build
from mapi_ablation.graph import generate_graph
from mapi_ablation.render import RenderConfig, render
from mapi_ablation.render.draw import cap_height
from mapi_ablation.render.grammar import DEFAULT_GEOMETRY, Geometry
from mapi_ablation.render.legibility import (
    ASSUMED_RULES,
    ResizeRule,
    authored_caps,
    cap_heights_for,
    infer_scaling,
)


def test_cap_height_comes_from_real_font_metrics():
    """Not a fraction-of-font-size approximation."""
    small, large = cap_height(12), cap_height(24)
    assert 0 < small < large
    assert large >= 2 * small - 2  # roughly linear in point size


def test_cap_height_is_monotone_in_font_size():
    heights = [cap_height(s) for s in range(8, 40, 2)]
    assert heights == sorted(heights)


def test_authored_caps_ordered_id_label_date():
    """The ID is the thing the model must emit, so it must be the largest."""
    a_id, a_label, a_date = authored_caps(DEFAULT_GEOMETRY)
    assert a_id > a_label >= a_date


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "openai"])
def test_postresize_is_never_larger_than_authored(provider):
    ch = cap_heights_for(DEFAULT_GEOMETRY, 1540, 1826, provider)
    assert 0 < ch.scale <= 1.0
    assert ch.postresize_id <= ch.authored_id
    assert ch.postresize_label <= ch.authored_label


def test_assumed_scaling_is_labelled_assumed():
    ch = cap_heights_for(DEFAULT_GEOMETRY, 1540, 1826, "anthropic")
    assert ch.provenance == "assumed"


def test_measured_scaling_overrides_and_is_labelled_measured():
    ch = cap_heights_for(DEFAULT_GEOMETRY, 1540, 1826, "anthropic",
                         measured_scale=0.5)
    assert ch.provenance == "measured"
    assert ch.scale == 0.5
    assert ch.postresize_id == round(ch.authored_id * 0.5, 2)


def test_long_edge_cap_shrinks_a_tall_canvas():
    rule = ResizeRule("t", long_edge_cap=1000, megapixel_cap=None, tile_px=None, note="")
    assert rule.scale_for(500, 2000) == pytest.approx(0.5)
    assert rule.scale_for(500, 800) == 1.0


def test_megapixel_cap_applies_after_long_edge_cap():
    rule = ResizeRule("t", long_edge_cap=1000, megapixel_cap=0.25, tile_px=None, note="")
    s = rule.scale_for(1000, 1000)
    assert (1000 * s) * (1000 * s) == pytest.approx(250_000, rel=1e-3)


def test_canvas_arm_payload_carries_authored_caps():
    """Every canvas payload must expose what a post-resize column needs."""
    p = build("canvas", generate_graph(0, 40))
    caps = p.meta["authored_cap_px"]
    assert set(caps) == {"id", "label", "date"}
    assert all(v > 0 for v in caps.values())
    assert p.meta["long_edge"] == max(p.meta["width"], p.meta["height"])


def test_bigger_graphs_lose_glyph_pixels():
    """The core tension, asserted: more nodes -> taller canvas -> smaller glyphs.

    If this ever stops holding, the legibility curve in the report is measuring
    nothing and someone has changed the layout in a way that needs re-checking.
    """
    caps = []
    for n in (20, 40, 80):
        cfg = RenderConfig.for_graph(n, width=1540)
        res = render(generate_graph(0, n), cfg)
        ch = cap_heights_for(cfg.geom, res.layout.width, res.layout.height, "anthropic")
        caps.append(ch.postresize_id)
    assert caps[0] > caps[1] > caps[2]


def test_uniform_geometry_scaling_does_not_change_postresize_size():
    """Why authored width alone is not a real variable.

    Scaling the whole grammar and the canvas together gives the same aspect
    ratio, so a long-edge-capped provider produces the same final glyph size.
    This is the trap that made an earlier version's width sweep a no-op.
    """
    base = DEFAULT_GEOMETRY
    doubled = base.scaled(2.0)
    a = cap_heights_for(base, 1540, 1826, "anthropic")
    b = cap_heights_for(doubled, 3080, 3652, "anthropic")
    assert b.postresize_id == pytest.approx(a.postresize_id, abs=1.0)


def test_infer_scaling_reports_no_scale_without_a_known_formula():
    measurement = {
        "provider": "p", "model": "m", "text_only_input_tokens": 10,
        "observations": [
            {"authored_w": 512, "authored_h": 512, "authored_px": 262144,
             "input_tokens": 110, "png_bytes": 1},
            {"authored_w": 1024, "authored_h": 1024, "authored_px": 1048576,
             "input_tokens": 410, "png_bytes": 1},
        ],
    }
    out = infer_scaling(measurement, tokens_per_px=None)
    assert all("effective_scale" not in r for r in out["rows"])
    assert "no tokens-per-pixel constant" in out["method"]
    assert out["rows"][0]["image_tokens"] == 100


def test_infer_scaling_detects_the_saturation_point():
    """Two authored sizes costing the same tokens IS the measured cap."""
    measurement = {
        "provider": "p", "model": "m", "text_only_input_tokens": 0,
        "observations": [
            {"authored_w": 512, "authored_h": 512, "authored_px": 262144,
             "input_tokens": 100, "png_bytes": 1},
            {"authored_w": 2048, "authored_h": 2048, "authored_px": 4194304,
             "input_tokens": 100, "png_bytes": 1},
        ],
    }
    out = infer_scaling(measurement, tokens_per_px=None)
    assert out["saturation_point"]["authored_w"] == 2048


def test_every_provider_has_a_documented_assumption():
    for name, rule in ASSUMED_RULES.items():
        assert rule.note.startswith("assumed"), name
