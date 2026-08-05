"""CLI entrypoints.

`pretest` is the gate. `run`, `stability`, `report` and `sweep` are deliberately
not implemented yet: the build order in the brief puts the pretest first and
says not to skip past it. They exit with a clear message rather than pretending
to work.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .pretest import (
    GATE_THRESHOLD,
    PRETEST_PROMPT_VERSION,
    PRIMITIVES,
    PROBES,
    build_pretest_prompt,
    score_probe,
)
from .providers import make_adapter
from .providers.base import ProviderError
from .render.legibility import infer_scaling, measure_provider_scaling
from .runner import ResponseCache, git_commit, write_manifest

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_CONFIG).read_text())


def _run_dir(run_id: str | None, prefix: str) -> Path:
    rid = run_id or f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    d = REPO / "runs" / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_adapters(cfg: dict, model_keys: list[str]) -> dict:
    adapters = {}
    for key in model_keys:
        spec = cfg["models"][key]
        adapters[key] = make_adapter(
            spec["provider"], list(spec["candidates"]),
            temperature=spec.get("temperature", 0.0),
        )
        console.print(
            f"  [green]resolved[/] {key}: {adapters[key].model} "
            f"({getattr(adapters[key], 'model_resolution', 'n/a')})"
        )
    return adapters


# ---------------------------------------------------------------------------
# pretest -- the gate
# ---------------------------------------------------------------------------


@app.command()
def pretest(
    models: str = typer.Option(None, help="comma-separated model keys from config"),
    conditions: str = typer.Option(None, help="cold,primed"),
    config: Path = typer.Option(None, help="config yaml"),
    run_id: str = typer.Option(None),
    width: int = typer.Option(1540, help="authored canvas width for the probes"),
    dry_run: bool = typer.Option(False, "--dry-run", help="build canvases, no API calls"),
    verify_resize: bool = typer.Option(
        False, help="also measure each provider's real image scaling from token counts"
    ),
) -> None:
    """Stage 0: 40 one-primitive canvases. Gate before the full ablation."""
    cfg = load_config(config)
    model_keys = (models.split(",") if models else cfg["pretest"]["models"])
    conds = (conditions.split(",") if conditions else cfg["pretest"]["conditions"])
    threshold = cfg["pretest"].get("gate_threshold", GATE_THRESHOLD)

    out = _run_dir(run_id, "pretest")
    canvases = out / "canvases"
    canvases.mkdir(exist_ok=True)

    console.print(f"[bold]Stage 0 pretest[/] -> {out}")
    console.print(f"  {len(PROBES)} probes x {len(conds)} conditions x "
                  f"{len(model_keys)} models = "
                  f"{len(PROBES) * len(conds) * len(model_keys)} calls")

    pngs = {}
    for p in PROBES:
        png = p.png(width=width)
        (canvases / f"{p.pid}.png").write_bytes(png)
        pngs[p.pid] = png
    console.print(f"  wrote {len(pngs)} canvases to {canvases}")

    if dry_run:
        table = Table("primitive", "n", "example question", "answer")
        for prim in PRIMITIVES:
            ex = next(p for p in PROBES if p.primitive == prim)
            table.add_row(prim, str(sum(1 for p in PROBES if p.primitive == prim)),
                          ex.question[:64], ex.answer)
        console.print(table)
        console.print("[yellow]--dry-run: no API calls made[/]")
        return

    try:
        adapters = _build_adapters(cfg, model_keys)
    except ProviderError as exc:
        console.print(f"[red]provider setup failed:[/] {exc}")
        raise typer.Exit(code=2)

    cache = ResponseCache(REPO / cfg["runner"]["cache_path"])
    rows: list[dict] = []
    responses_path = out / "responses.jsonl"

    for key, adapter in adapters.items():
        for cond in conds:
            for p in PROBES:
                prompt = build_pretest_prompt(p, cond)
                ck = f"{adapter.model}|{p.pid}|{cond}|{PRETEST_PROMPT_VERSION}|{width}"
                import hashlib

                ckey = hashlib.sha256(ck.encode()).hexdigest()
                resp = cache.get(ckey)
                cached = resp is not None
                if resp is None:
                    resp = adapter.complete(
                        text=prompt, image_png=pngs[p.pid], max_tokens=24
                    )
                    cache.put(ckey, resp)
                if resp.ok:
                    parsed, outcome = score_probe(p, resp.text)
                else:
                    parsed, outcome = None, "error"
                row = {
                    "model_key": key, "model": adapter.model, "condition": cond,
                    "primitive": p.primitive, "pid": p.pid, "question": p.question,
                    "expected": p.answer, "parsed": parsed, "outcome": outcome,
                    "raw_text": resp.text, "input_tokens": resp.input_tokens,
                    "latency_ms": round(resp.latency_ms, 1), "error": resp.error,
                    "cached": cached,
                }
                rows.append(row)
                with open(responses_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                mark = {"correct": "[green].[/]", "wrong": "[red]x[/]",
                        "unparseable": "[yellow]?[/]", "error": "[red]![/]"}[outcome]
                console.print(mark, end="")
            console.print(f"  {key}/{cond}")

    _pretest_report(rows, out, threshold, width)

    if verify_resize:
        _verify_resize(adapters, out)

    write_manifest(
        out / "manifest.json",
        kind="pretest",
        git_commit=git_commit(),
        prompt_version=PRETEST_PROMPT_VERSION,
        width=width,
        models={k: {"model": a.model,
                    "resolution": getattr(a, "model_resolution", None)}
                for k, a in adapters.items()},
        conditions=conds,
        n_probes=len(PROBES),
        gate_threshold=threshold,
    )


def _pretest_report(rows: list[dict], out: Path, threshold: float, width: int) -> None:
    if not rows:
        console.print("[red]no results[/]")
        return
    agg: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        agg[(r["model_key"], r["condition"], r["primitive"])].append(r["outcome"])

    lines = [
        "# Stage 0 pretest",
        "",
        f"Authored canvas width: {width}px. Gate: every primitive >= "
        f"{threshold:.0%} for a frontier model.",
        "",
        "| model | condition | primitive | n | correct | wrong | unparseable | "
        "error | accuracy | gate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    table = Table("model", "condition", "primitive", "n", "acc", "gate")
    failures: list[str] = []
    for (mk, cond, prim), outcomes in sorted(agg.items()):
        n = len(outcomes)
        c = outcomes.count("correct")
        acc = c / n if n else 0.0
        passed = acc >= threshold
        if not passed:
            failures.append(f"{mk}/{cond}/{prim} = {acc:.0%}")
        lines.append(
            f"| {mk} | {cond} | {prim} | {n} | {c} | {outcomes.count('wrong')} | "
            f"{outcomes.count('unparseable')} | {outcomes.count('error')} | "
            f"{acc:.0%} | {'PASS' if passed else 'FAIL'} |"
        )
        table.add_row(mk, cond, prim, str(n), f"{acc:.0%}",
                      "[green]PASS[/]" if passed else "[red]FAIL[/]")

    lines += ["", "## Verdict", ""]
    if failures:
        lines.append(
            "GATE NOT PASSED. The following primitives fell below the threshold, "
            "so at least part of the visual grammar is not cold-readable and the "
            "full ablation would partly be measuring that rather than the "
            "hypothesis:"
        )
        lines += [f"- {f}" for f in failures]
    else:
        lines.append(
            "GATE PASSED. Every primitive is readable at or above the threshold, "
            "so a failure in the full ablation would not be attributable to an "
            "unreadable grammar."
        )
    (out / "pretest.md").write_text("\n".join(lines) + "\n")
    console.print(table)
    console.print(f"\nwrote {out / 'pretest.md'}")
    if failures:
        console.print(f"[red]GATE NOT PASSED[/]: {', '.join(failures)}")
    else:
        console.print("[green]GATE PASSED[/]")


def _verify_resize(adapters: dict, out: Path) -> None:
    """Measure real provider scaling instead of trusting documented numbers."""
    console.print("\n[bold]measuring provider image scaling[/]")
    doc = []
    for key, adapter in adapters.items():
        console.print(f"  probing {key} ({adapter.model}) ...")
        raw = measure_provider_scaling(adapter)
        # tokens-per-pixel is only applied when we actually know the formula;
        # otherwise the saturation point is reported without an inferred scale.
        tpp = {"anthropic": 1 / 750}.get(adapter.name)
        doc.append(infer_scaling(raw, tokens_per_px=tpp))
        (out / f"resize_raw_{key}.json").write_text(json.dumps(raw, indent=2))

    lines = ["# Provider image preprocessing, measured", ""]
    lines.append(
        "Authored pixels are not encoded pixels. Each row below sends a blank "
        "PNG at a known authored size and records the provider's own reported "
        "input_tokens. Where two authored sizes cost the same tokens, the "
        "provider has resized both to the same thing -- that saturation point "
        "is the real cap, measured rather than quoted."
    )
    for d in doc:
        lines += ["", f"## {d['provider']} / {d['model']}", "", f"method: {d['method']}",
                  "", "| authored | authored px | image tokens | encoded px | "
                  "effective scale |", "|---|---|---|---|---|"]
        for r in d["rows"]:
            lines.append(
                f"| {r['authored_w']}x{r['authored_h']} | {r['authored_px']:,} | "
                f"{r['image_tokens']} | {r.get('encoded_px', '-')} | "
                f"{r.get('effective_scale', '-')} |"
            )
        sat = d["saturation_point"]
        lines.append("")
        lines.append(
            f"saturation first seen at {sat['authored_w']}x{sat['authored_h']}"
            if sat else "no saturation point observed in the probed range"
        )
    (out / "legibility.md").write_text("\n".join(lines) + "\n")
    console.print(f"wrote {out / 'legibility.md'}")


# ---------------------------------------------------------------------------
# gated commands
# ---------------------------------------------------------------------------

_GATED = (
    "Gated on Stage 0. The build order puts the pretest first: if the grammar "
    "is not cold-readable, the full ablation measures that instead of the "
    "hypothesis. Run `pretest` and review runs/<id>/pretest.md first."
)


@app.command()
def run(seeds: str = typer.Option(None), models: str = typer.Option(None),
        arms: str = typer.Option(None), dry_run: bool = typer.Option(False, "--dry-run")
        ) -> None:
    """Full ablation. Not built yet -- see the pretest gate."""
    console.print(f"[yellow]{_GATED}[/]")
    raise typer.Exit(code=3)


@app.command()
def stability(seeds: str = typer.Option(None), models: str = typer.Option(None),
              arms: str = typer.Option(None),
              dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Layout stability under mutation. Not built yet -- see the pretest gate."""
    console.print(f"[yellow]{_GATED}[/]")
    raise typer.Exit(code=3)


@app.command()
def report(seeds: str = typer.Option(None), models: str = typer.Option(None),
           arms: str = typer.Option(None),
           dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Analysis and plots. Not built yet -- see the pretest gate."""
    console.print(f"[yellow]{_GATED}[/]")
    raise typer.Exit(code=3)


@app.command()
def sweep(seeds: str = typer.Option(None), models: str = typer.Option(None),
          arms: str = typer.Option(None),
          dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Staged sweep. Not built yet -- see the pretest gate."""
    console.print(f"[yellow]{_GATED}[/]")
    raise typer.Exit(code=3)


@app.command()
def example(seed: int = 3, n_nodes: int = 40, width: int = 1540,
            out_dir: Path = typer.Option(None)) -> None:
    """Serialize one graph in all four arms and render its canvas."""
    from .arms import ARMS, build
    from .graph import generate_graph
    from .queries import annotate_with_layout, generate_queries
    from .render import RenderConfig, render, render_report, save
    from .render.legibility import cap_heights_for

    out = out_dir or (REPO / "runs" / "_example")
    out.mkdir(parents=True, exist_ok=True)
    g = generate_graph(seed, n_nodes)
    qs = generate_queries(g)
    for name in ARMS:
        p = build(name, g, width=width)
        if p.kind == "text":
            (out / f"arm_{name}.txt").write_text(p.text)
        else:
            (out / f"arm_{name}.png").write_bytes(p.image_png)

    cfg = RenderConfig.for_graph(n_nodes, width=width)
    res = render(g, cfg)
    save(res, out / "canvas.png")
    save(render(g, cfg, debug=True), out / "canvas_debug.png")
    rep = render_report(res)
    annotate_with_layout(qs, res.layout.positions())

    t = Table("provider", "assumed scale", "post-resize cap px (id/label/date)")
    for prov in ("anthropic", "gemini", "openai"):
        ch = cap_heights_for(cfg.geom, res.layout.width, res.layout.height, prov)
        t.add_row(prov, f"{ch.scale:.3f}",
                  f"{ch.postresize_id} / {ch.postresize_label} / {ch.postresize_date}")
    console.print(f"[bold]seed={seed} n={n_nodes}[/] fingerprint={g.fingerprint()}")
    console.print(json.dumps(rep))
    console.print(t)
    console.print(f"queries: {len(qs)} "
                  f"{ {k: v['sampled'] for k, v in qs.capacity.items()} }")
    console.print(f"wrote {out}")


if __name__ == "__main__":
    app()
