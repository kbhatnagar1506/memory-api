"""Glyph cap-height measurement and provider resize behaviour.

This is a headline output of the harness, not a footnote. Every provider resizes
images before the vision encoder sees them, so an authored cap-height of 14px
tells you nothing about what the model can actually read. All reported
cap-heights are POST-RESIZE.

The provider scaling rules below are *assumptions until measured*. The measured
path is `measure_provider_scaling()`, which sends the same canvas at several
authored sizes, reads back the `input_tokens` each provider reports, and infers
the effective scaling from the provider's documented token formula. Nothing in
a results table may cite an assumed value once a measured one exists -- the
column carries its provenance ("assumed" / "measured") so a reader can tell.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, asdict

from PIL import Image

from .draw import cap_height
from .grammar import Geometry


@dataclass(frozen=True)
class ResizeRule:
    """How a provider is believed to preprocess an image before encoding."""

    provider: str
    #: Long-edge cap in px, or None if the provider does not cap that way.
    long_edge_cap: int | None
    #: Megapixel cap applied after the long-edge cap, or None.
    megapixel_cap: float | None
    #: Square tile edge used by the provider's token formula, or None.
    tile_px: int | None
    note: str

    def scale_for(self, w: int, h: int) -> float:
        s = 1.0
        if self.long_edge_cap:
            s = min(s, self.long_edge_cap / max(w, h))
        if self.megapixel_cap:
            mp = (w * s) * (h * s) / 1_000_000
            if mp > self.megapixel_cap:
                s *= math.sqrt(self.megapixel_cap / mp)
        return s


#: Working assumptions only. Replaced per-run by measured values.
ASSUMED_RULES: dict[str, ResizeRule] = {
    "anthropic": ResizeRule(
        provider="anthropic",
        long_edge_cap=1568,
        megapixel_cap=1.15,
        tile_px=None,
        note="assumed: long edge <=1568px and <=~1.15MP; tokens ~= w*h/750",
    ),
    "gemini": ResizeRule(
        provider="gemini",
        long_edge_cap=3072,
        megapixel_cap=None,
        tile_px=768,
        note="assumed: tiled at 768px; small images cost a flat token count",
    ),
    "openai": ResizeRule(
        provider="openai",
        long_edge_cap=2048,
        megapixel_cap=None,
        tile_px=512,
        note="assumed: fit to 2048 long edge, short edge to 768, 512px tiles",
    ),
}


@dataclass
class CapHeights:
    """Authored and post-resize cap-heights for the three text roles."""

    provider: str
    scale: float
    provenance: str  # "assumed" | "measured"
    authored_id: float
    authored_label: float
    authored_date: float

    @property
    def postresize_id(self) -> float:
        return round(self.authored_id * self.scale, 2)

    @property
    def postresize_label(self) -> float:
        return round(self.authored_label * self.scale, 2)

    @property
    def postresize_date(self) -> float:
        return round(self.authored_date * self.scale, 2)

    def as_row(self) -> dict:
        d = asdict(self)
        d.update(
            postresize_id=self.postresize_id,
            postresize_label=self.postresize_label,
            postresize_date=self.postresize_date,
        )
        return d


def authored_caps(geom: Geometry) -> tuple[float, float, float]:
    """Cap-height in px of the ID, label and date text, from real font metrics."""
    return (
        float(cap_height(geom.font_id, bold=True)),
        float(cap_height(geom.font_label)),
        float(cap_height(geom.font_date)),
    )


def cap_heights_for(
    geom: Geometry,
    width: int,
    height: int,
    provider: str,
    measured_scale: float | None = None,
) -> CapHeights:
    a_id, a_label, a_date = authored_caps(geom)
    if measured_scale is not None:
        scale, prov = measured_scale, "measured"
    else:
        rule = ASSUMED_RULES.get(provider)
        scale = rule.scale_for(width, height) if rule else 1.0
        prov = "assumed"
    return CapHeights(
        provider=provider,
        scale=round(scale, 4),
        provenance=prov,
        authored_id=a_id,
        authored_label=a_label,
        authored_date=a_date,
    )


# ---------------------------------------------------------------------------
# Empirical measurement
# ---------------------------------------------------------------------------


def _solid_png(w: int, h: int) -> bytes:
    """A trivial image of a given size -- token cost depends on size, not content."""
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


#: Probe sizes: below, at, and above every cap we might be dealing with.
PROBE_SIZES: tuple[tuple[int, int], ...] = (
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1540, 1024),
    (1540, 1568),
    (1540, 1826),
    (1540, 2400),
    (2560, 1600),
    (2560, 2786),
)


def measure_provider_scaling(adapter, sizes=PROBE_SIZES, baseline_prompt="ok") -> dict:
    """Infer a provider's effective scaling from reported input_tokens.

    Sends a blank image at each probe size and records the provider's own
    `input_tokens`. Two images that the provider resizes to the same dimensions
    cost the same number of tokens, so the point at which token count stops
    growing with authored size IS the cap -- measured, not documented.

    Returns raw observations. Interpretation happens in `infer_scaling()` so the
    raw numbers stay in the run directory for anyone who wants to re-derive it.
    """
    obs = []
    for (w, h) in sizes:
        png = _solid_png(w, h)
        resp = adapter.complete(
            text=baseline_prompt,
            image_png=png,
            system="Reply with the single word: ok",
        )
        obs.append(
            {
                "authored_w": w,
                "authored_h": h,
                "authored_px": w * h,
                "input_tokens": resp.input_tokens,
                "png_bytes": len(png),
            }
        )
    # Cost of the text-only prompt, so image tokens can be isolated.
    base = adapter.complete(
        text=baseline_prompt, image_png=None, system="Reply with the single word: ok"
    )
    return {
        "provider": adapter.name,
        "model": adapter.model,
        "text_only_input_tokens": base.input_tokens,
        "observations": obs,
    }


def infer_scaling(measurement: dict, tokens_per_px: float | None = None) -> dict:
    """Turn raw token observations into an effective scale per probe size.

    If `tokens_per_px` is supplied (from the provider's documented formula) the
    effective encoded pixel count is tokens/tokens_per_px, and the scale is
    sqrt(encoded_px / authored_px). Otherwise we report the token counts and the
    saturation point only, and say so, rather than inventing a conversion.
    """
    base = measurement["text_only_input_tokens"]
    rows = []
    for o in measurement["observations"]:
        image_tokens = o["input_tokens"] - base
        row = {**o, "image_tokens": image_tokens}
        if tokens_per_px:
            encoded_px = image_tokens / tokens_per_px
            row["encoded_px"] = round(encoded_px)
            row["effective_scale"] = round(
                math.sqrt(max(encoded_px, 1) / o["authored_px"]), 4
            )
        rows.append(row)

    saturation = None
    for a, b in zip(rows, rows[1:]):
        if b["authored_px"] > a["authored_px"] and b["image_tokens"] <= a["image_tokens"]:
            saturation = b
            break

    return {
        "provider": measurement["provider"],
        "model": measurement["model"],
        "rows": rows,
        "saturation_point": saturation,
        "method": (
            "scale = sqrt(encoded_px / authored_px), encoded_px = image_tokens / "
            f"{tokens_per_px}"
            if tokens_per_px
            else "token counts only; no tokens-per-pixel constant supplied, so no "
                 "scale was inferred"
        ),
    }
