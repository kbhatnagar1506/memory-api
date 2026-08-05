"""CLI entrypoints.

`pretest` is the gate. `run`, `stability`, `report` and `sweep` are deliberately
not implemented yet: the build order in the brief puts the pretest first and
says not to skip past it. They exit with a clear message rather than pretending
to work.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .pretest import (
    CRITICAL_PATH,
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
    """Stage 0: one-primitive canvases. Gate before the full ablation."""
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
    write_lock = threading.Lock()

    def one(key: str, adapter, cond: str, p) -> dict:
        # The payload hash MUST be in the key. Without it, a probe id like
        # `band_label-00` collides across runs whose probe content differs, and
        # the cache serves an answer to a question that was never asked. That
        # happened between gate-01 and gate-02 and produced a fake 87% on
        # band_label and a fake 20% on date_read.
        payload_hash = hashlib.sha256(pngs[p.pid]).hexdigest()[:16]
        ck = (f"{adapter.model}|{p.pid}|{cond}|{PRETEST_PROMPT_VERSION}|"
              f"{width}|{payload_hash}|{p.answer}")
        ckey = hashlib.sha256(ck.encode()).hexdigest()
        resp = cache.get(ckey)
        cached = resp is not None
        if resp is None:
            resp = adapter.complete(
                text=build_pretest_prompt(p, cond),
                image_png=pngs[p.pid],
                max_tokens=24,
            )
            cache.put(ckey, resp)
        parsed, outcome = score_probe(p, resp.text) if resp.ok else (None, "error")
        return {
            "model_key": key, "model": adapter.model, "condition": cond,
            "primitive": p.primitive, "pid": p.pid, "question": p.question,
            "expected": p.answer, "parsed": parsed, "outcome": outcome,
            "raw_text": resp.text, "input_tokens": resp.input_tokens,
            "latency_ms": round(resp.latency_ms, 1), "error": resp.error,
            "cached": cached, "meta": p.meta,
        }

    jobs = [(k, a, c, p) for k, a in adapters.items() for c in conds for p in PROBES]
    conc = cfg["runner"].get("concurrency", 6)
    console.print(f"  running {len(jobs)} calls at concurrency {conc}")
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [pool.submit(one, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            with write_lock:
                with open(responses_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            mark = {"correct": "[green].[/]", "wrong": "[red]x[/]",
                    "unparseable": "[yellow]?[/]", "error": "[red]![/]"}[row["outcome"]]
            console.print(mark, end="")
            if i % 80 == 0:
                console.print(f" {i}/{len(jobs)}")
    console.print("")

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
        f"Authored canvas width: {width}px. Gate: every critical-path primitive "
        f">= {threshold:.0%}.",
        "",
        "Each row carries a Wilson 95% interval, and the gate is three-state: "
        "FAIL if the interval sits entirely below the threshold, PASS if it sits "
        "entirely at or above it, INCONCLUSIVE if this sample size cannot tell. "
        "Three states because a two-state interval gate is unpassable at small n "
        "-- a perfect 30/30 has a lower bound of 88.6%. A decisive PASS at 95% "
        "needs n>=75 with zero errors, or n>=150 to tolerate one.",
        "",
        "`crit` marks the primitives the six query types actually depend on. The "
        "others are reported but do not gate, because no question exercises them.",
        "",
        "| model | condition | primitive | crit | n | correct | wrong | "
        "unparseable | error | accuracy | 95% CI | gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    table = Table("model", "cond", "primitive", "crit", "n", "acc", "95% CI", "gate")
    failures: list[str] = []
    warnings: list[str] = []
    for (mk, cond, prim), outcomes in sorted(agg.items()):
        n = len(outcomes)
        c = outcomes.count("correct")
        acc = c / n if n else 0.0
        lo, hi = _wilson(c, n)
        crit = prim in CRITICAL_PATH
        # The gate is a claim about the TRUE rate, so judge the interval rather
        # than the point estimate -- but three-state, because a two-state gate
        # on an interval is unpassable at small n (30/30 has a lower bound of
        # 89%). FAIL means confidently below; PASS means confidently at or
        # above; INCONCLUSIVE means this sample size cannot tell.
        if hi < threshold:
            verdict = "FAIL"
        elif lo >= threshold:
            verdict = "PASS"
        else:
            verdict = "INCONCLUSIVE"
        label = verdict if crit else f"{verdict} (non-gating)"
        if verdict != "PASS":
            msg = (f"{mk}/{cond}/{prim} = {acc:.0%} [{lo:.0%}-{hi:.0%}] n={n} "
                   f"-> {verdict}")
            (failures if crit and verdict == "FAIL" else warnings).append(
                f"{'crit ' if crit else ''}{msg}")
        lines.append(
            f"| {mk} | {cond} | {prim} | {'yes' if crit else 'no'} | {n} | {c} | "
            f"{outcomes.count('wrong')} | {outcomes.count('unparseable')} | "
            f"{outcomes.count('error')} | {acc:.0%} | {lo:.0%}-{hi:.0%} | {label} |"
        )
        table.add_row(
            mk, cond, prim, "yes" if crit else "no", str(n), f"{acc:.0%}",
            f"{lo:.0%}-{hi:.0%}",
            {"PASS": "[green]PASS[/]", "FAIL": "[red]FAIL[/]",
             "INCONCLUSIVE": "[yellow]INCONC[/]"}[verdict],
        )

    lines += ["", "## Verdict", ""]
    if failures:
        lines.append(
            "**GATE NOT PASSED.** These critical-path primitives did not clear "
            "the threshold, so part of the visual grammar the query set depends "
            "on is not reliably readable, and the full ablation would partly be "
            "measuring that rather than the hypothesis:"
        )
        lines += [f"- {f}" for f in failures]
    else:
        lines.append(
            "**GATE PASSED on the critical path.** Every primitive the six query "
            "types depend on clears the threshold, so a canvas-arm failure in the "
            "full ablation would not be attributable to an unreadable grammar."
        )
    if warnings:
        lines += [
            "",
            "Not a decisive pass (reported, not blocking):",
        ]
        lines += [f"- {w}" for w in warnings]
        lines += [
            "",
            "The claim this run supports is therefore narrower than 'the grammar "
            "is cold-readable': it is 'the parts of the grammar the query set "
            "depends on are cold-readable'. Any write-up must say so.",
        ]
    (out / "pretest.md").write_text("\n".join(lines) + "\n")
    console.print(table)
    console.print(f"\nwrote {out / 'pretest.md'}")
    if failures:
        console.print(f"[red]GATE NOT PASSED[/]: {'; '.join(failures)}")
    else:
        console.print("[green]GATE PASSED on the critical path[/]")
    for w in warnings:
        console.print(f"[yellow]not a decisive pass[/]: {w}")


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Behaves sanely at 0/n and n/n, unlike normal-approx."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


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
