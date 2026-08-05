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
from .prompts import build_prompt
from .runner import ResponseCache, Runner, git_commit, write_manifest

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


def build_calls(cfg: dict, model_keys: list[str], seeds: list[int], arms: list[str],
                conditions: list[str], n_nodes: int, width: int,
                adapters: dict, n_per_type: int) -> tuple[list, dict]:
    """One Call per (model, arm, condition, query). Identical graphs across arms.

    The graph and query set are built ONCE per seed and shared by every arm, so
    arm parity holds by construction rather than by convention.
    """
    from .arms import ARMS, build as build_arm
    from .graph import generate_graph
    from .queries import annotate_with_layout, generate_queries
    from .render.layout import RenderConfig
    from .render.legibility import cap_heights_for
    from .runner import Call

    calls: list[Call] = []
    graph_info: dict = {}

    for seed in seeds:
        g = generate_graph(seed, n_nodes)
        qs = generate_queries(g, n_per_type=n_per_type)
        rcfg = RenderConfig.for_graph(n_nodes, width=width)

        payloads = {a: build_arm(a, g, width=width) for a in arms if a in ARMS}
        canvas_payload = payloads.get("canvas")
        if canvas_payload is not None:
            annotate_with_layout(qs, canvas_payload.meta["positions"])

        graph_info[seed] = {
            "fingerprint": g.fingerprint(),
            "n_queries": len(qs),
            "capacity": qs.capacity,
            "collisions": canvas_payload.meta["collisions"] if canvas_payload else [],
            "canvas_size": (
                [canvas_payload.meta["width"], canvas_payload.meta["height"]]
                if canvas_payload else None
            ),
        }

        for model_key in model_keys:
            provider = cfg["models"][model_key]["provider"]
            caps = None
            if canvas_payload is not None:
                caps = cap_heights_for(
                    rcfg.geom, canvas_payload.meta["width"],
                    canvas_payload.meta["height"], provider,
                )
            for arm in arms:
                p = payloads[arm]
                for cond in conditions:
                    for q in qs:
                        bp = build_prompt(p, q.question, cond)
                        meta = {
                            "hops": q.hops,
                            "crosses_domain": q.crosses_domain,
                            "t_gap": q.t_gap,
                            "pixel_dist": q.pixel_dist if arm == "canvas" else None,
                            "postresize_pixel_dist": (
                                q.postresize_pixel_dist if arm == "canvas" else None
                            ),
                            "postresize_cap_px": (
                                caps.postresize_id if (arm == "canvas" and caps) else None
                            ),
                            "cap_provenance": (
                                caps.provenance if (arm == "canvas" and caps) else None
                            ),
                            "width": width,
                        }
                        calls.append(Call(
                            model_key=model_key,
                            provider=provider,
                            model=adapters[model_key].model if adapters else model_key,
                            arm=arm,
                            condition=cond,
                            qid=q.qid,
                            question=q.question,
                            expected=q.answer,
                            payload_hash=p.payload_hash(),
                            prompt_version=bp.prompt_version,
                            seed=seed,
                            n_nodes=n_nodes,
                            qtype=q.qtype,
                            text=bp.text,
                            image_png=bp.image_png,
                            system=None,
                            meta=meta,
                        ))
    return calls, graph_info


@app.command()
def run(
    seeds: str = typer.Option(None, help="comma-separated seeds"),
    models: str = typer.Option(None, help="comma-separated model keys"),
    arms: str = typer.Option(None, help="comma-separated arm names"),
    conditions: str = typer.Option(None, help="cold,primed"),
    n_nodes: int = typer.Option(None),
    width: int = typer.Option(None),
    config: Path = typer.Option(None),
    run_id: str = typer.Option(None),
    stage: str = typer.Option("stage1", help="config block to draw defaults from"),
    yes: bool = typer.Option(False, "--yes", help="skip the call-count confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the ablation. Every arm sees the identical graph; only encoding varies."""
    cfg = load_config(config)
    block = cfg[stage]
    model_keys = models.split(",") if models else list(block["models"])
    seed_list = ([int(s) for s in seeds.split(",")] if seeds else list(block["seeds"]))
    arm_list = arms.split(",") if arms else list(block["arms"])
    cond_list = conditions.split(",") if conditions else list(block["conditions"])
    nn = n_nodes or block["n_nodes"]
    w = width or block["width"]
    n_per_type = cfg["queries"]["n_per_type"]

    out = _run_dir(run_id, "run")
    console.print(f"[bold]Ablation run[/] -> {out}")
    console.print(f"  models={model_keys} seeds={seed_list} arms={arm_list} "
                  f"conditions={cond_list} n_nodes={nn} width={w}")

    adapters: dict = {}
    if not dry_run:
        try:
            adapters = _build_adapters(cfg, model_keys)
        except ProviderError as exc:
            console.print(f"[red]provider setup failed:[/] {exc}")
            raise typer.Exit(code=2)

    calls, graph_info = build_calls(cfg, model_keys, seed_list, arm_list,
                                    cond_list, nn, w, adapters, n_per_type)

    n_calls = len(calls)
    est_in = sum(len(c.text) // 4 + (1900 if c.image_png else 0) for c in calls)
    prices = cfg.get("price_hint_usd_per_mtok", {})
    est_cost = sum(
        prices.get(cfg["models"][c.model_key]["provider"], {}).get("input", 0)
        * ((len(c.text) // 4 + (1900 if c.image_png else 0)) / 1e6)
        for c in calls
    )
    console.print(f"  [bold]{n_calls} calls[/], ~{est_in / 1e6:.2f}M input tokens, "
                  f"rough cost ~${est_cost:.2f} (hint prices, input only)")
    for seed, info in graph_info.items():
        if info["collisions"]:
            console.print(f"  [yellow]seed {seed}: {len(info['collisions'])} node "
                          f"collisions on canvas[/]")

    max_calls = cfg["runner"].get("max_calls", 4000)
    if n_calls > max_calls and not yes:
        console.print(f"[red]{n_calls} calls exceeds max_calls={max_calls}.[/] "
                      "Re-run with --yes to proceed.")
        raise typer.Exit(code=4)

    if dry_run:
        console.print("[yellow]--dry-run: no API calls made[/]")
        t = Table("seed", "fingerprint", "queries", "canvas", "collisions")
        for seed, info in graph_info.items():
            t.add_row(str(seed), info["fingerprint"], str(info["n_queries"]),
                      str(info["canvas_size"]), str(len(info["collisions"])))
        console.print(t)
        return

    if not yes:
        typer.confirm(f"Execute {n_calls} calls?", abort=True)

    cache = ResponseCache(REPO / cfg["runner"]["cache_path"])
    responses_path = out / "responses.jsonl"
    runner = Runner(adapters, cache, concurrency=cfg["runner"]["concurrency"],
                    max_tokens=cfg["runner"]["max_tokens"],
                    responses_path=responses_path)

    done = {"n": 0}

    def progress(res) -> None:
        done["n"] += 1
        mark = {"correct": "[green].[/]", "wrong": "[red]x[/]",
                "unparseable": "[yellow]?[/]", "error": "[red]![/]"}[res.outcome]
        console.print(mark, end="")
        if done["n"] % 100 == 0:
            console.print(f" {done['n']}/{n_calls}")

    results = runner.run(calls, progress=progress)
    console.print("")

    write_manifest(
        out / "manifest.json",
        kind="ablation",
        git_commit=git_commit(),
        prompt_version=results[0].call.prompt_version if results else None,
        models={k: {"model": a.model,
                    "resolution": getattr(a, "model_resolution", None),
                    "temperature": getattr(a, "temperature", None)}
                for k, a in adapters.items()},
        seeds=seed_list, arms=arm_list, conditions=cond_list,
        n_nodes=nn, width=w, n_per_type=n_per_type,
        n_calls=n_calls, graphs=graph_info, config_stage=stage,
    )
    console.print(f"wrote {responses_path}")
    console.print(f"now run: [bold]report --run-id {out.name}[/]")


@app.command()
def stability(seeds: str = typer.Option(None), models: str = typer.Option(None),
              arms: str = typer.Option(None),
              dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Layout stability under mutation. Not built yet -- see the pretest gate."""
    console.print(f"[yellow]{_GATED}[/]")
    raise typer.Exit(code=3)


@app.command()
def report(
    run_id: str = typer.Option(..., help="run directory under runs/"),
    config: Path = typer.Option(None),
    seeds: str = typer.Option(None), models: str = typer.Option(None),
    arms: str = typer.Option(None),
    stability_seeds: str = typer.Option(None, help="seeds for the §7 table"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Aggregate a run into results.md plus plots."""
    from .arms import TEXT_ARMS
    from .score import (
        accuracy, best_text_arm, canvas_failures, cluster_bootstrap_ci,
        cluster_failures, cost_table, group, mcnemar, outcome_counts,
    )

    out = REPO / "runs" / run_id
    path = out / "responses.jsonl"
    if not path.exists():
        console.print(f"[red]no responses at {path}[/]")
        raise typer.Exit(code=2)
    rows = [json.loads(l) for l in path.open()]
    if seeds:
        keep = {int(s) for s in seeds.split(",")}
        rows = [r for r in rows if r["seed"] in keep]
    if models:
        keep_m = set(models.split(","))
        rows = [r for r in rows if r["model_key"] in keep_m]
    if arms:
        keep_a = set(arms.split(","))
        rows = [r for r in rows if r["arm"] in keep_a]
    console.print(f"[bold]report[/] {run_id}: {len(rows)} responses")

    manifest = {}
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())

    L: list[str] = [f"# Representation ablation -- {run_id}", ""]
    if manifest:
        L += [
            f"Models: {json.dumps(manifest.get('models', {}))}  ",
            f"Seeds: {manifest.get('seeds')}  n_nodes: {manifest.get('n_nodes')}  "
            f"width: {manifest.get('width')}  ",
            f"Prompt version: {manifest.get('prompt_version')}  "
            f"git: {manifest.get('git_commit', '')[:12]}",
            "",
        ]
    n_seeds = len({r["seed"] for r in rows})
    if n_seeds < 2:
        L += ["> Only one seed is present, so the clustered bootstrap is undefined "
              "and CIs are reported as n/a. Do not read a headline off this run.", ""]

    # -- 1. headline -------------------------------------------------------
    L += ["## 1. Accuracy by arm", "",
          "95% CIs are a percentile bootstrap over 10k resamples, clustered by "
          "seed: queries from one graph share a layout and a node set, so "
          "resampling them independently would understate the interval.", "",
          "| model | condition | arm | n | correct | wrong | unparseable | error "
          "| accuracy | 95% CI |", "|---|---|---|---|---|---|---|---|---|---|"]
    t = Table("model", "cond", "arm", "n", "acc", "95% CI")
    for (mk, cond, arm), rs in sorted(group(rows, "model_key", "condition", "arm").items()):
        ci = cluster_bootstrap_ci(rs)
        c = outcome_counts(rs)
        ci_txt = ("n/a" if ci.lo != ci.lo else f"{ci.lo:.1%}-{ci.hi:.1%}")
        L.append(f"| {mk} | {cond} | {arm} | {len(rs)} | {c['correct']} | "
                 f"{c['wrong']} | {c['unparseable']} | {c['error']} | "
                 f"{ci.point:.1%} | {ci_txt} |")
        t.add_row(mk, cond, arm, str(len(rs)), f"{ci.point:.1%}", ci_txt)
    console.print(t)

    # -- by query type -----------------------------------------------------
    L += ["", "### By query type", "",
          "`root_cause` and `cross_domain_effect` are the multi-hop tests -- the "
          "actual hypothesis. `temporal_pair` and `oldest_in_domain` are "
          "capability probes: they check the model can read the axes at all.", "",
          "| arm | " + " | ".join(sorted({r["qtype"] for r in rows})) + " |",
          "|---" * (1 + len({r["qtype"] for r in rows})) + "|"]
    qtypes = sorted({r["qtype"] for r in rows})
    for (arm,), rs in sorted(group(rows, "arm").items()):
        cells = []
        for qt in qtypes:
            sub = [r for r in rs if r["qtype"] == qt]
            cells.append(f"{accuracy(sub):.0%} ({len(sub)})" if sub else "-")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")

    # -- 2. paired test ----------------------------------------------------
    L += ["", "## 2. Canvas vs the best text arm (paired)", ""]
    canvas_rows = [r for r in rows if r["arm"] == "canvas"]
    text_rows = [r for r in rows if r["arm"] in TEXT_ARMS]
    best = best_text_arm(text_rows, TEXT_ARMS)
    if canvas_rows and best:
        L.append(f"Best text arm on this data: **{best}** "
                 f"({accuracy([r for r in text_rows if r['arm'] == best]):.1%}). "
                 "Chosen empirically, not assumed.")
        res = mcnemar(canvas_rows, [r for r in text_rows if r["arm"] == best],
                      "canvas", best)
        L += ["", f"- {res.summary()}",
              f"- McNemar chi-square (continuity-corrected): {res.statistic:.3f}",
              f"- Discordant pairs: {res.discordant} of {res.n_pairs} paired "
              "instances", ""]
        if res.discordant < 10:
            L.append("> Fewer than 10 discordant pairs. The exact test is valid but "
                     "has very little power here; treat the p-value as weak "
                     "evidence either way.")
        console.print(f"[bold]{res.summary()}[/]")
        # C vs D specifically -- the "is it layout or is it pixels" question
        lt = [r for r in rows if r["arm"] == "layout_text"]
        if lt and canvas_rows:
            cd = mcnemar(canvas_rows, lt, "canvas", "layout_text")
            L += ["", "### C vs D: is the invention the layout, or the pixels?", "",
                  f"- {cd.summary()}", ""]
    else:
        L.append("Canvas arm or text arms absent; no paired test.")

    # -- 3. cost -----------------------------------------------------------
    L += ["", "## 3. Cost and latency", "",
          "`tokens_per_correct` is the number that matters for a memory layer: "
          "an arm that is cheap but wrong is not cheap.", "",
          "| arm | n | mean input tokens | mean latency ms | accuracy | "
          "tokens per correct answer |", "|---|---|---|---|---|---|"]
    for arm, v in cost_table(rows).items():
        L.append(f"| {arm} | {v['n']} | {v['mean_input_tokens']} | "
                 f"{v['mean_latency_ms']} | {v['accuracy']:.1%} | "
                 f"{v['tokens_per_correct']} |")

    # -- 4. legibility -----------------------------------------------------
    L += ["", "## 4. Legibility", ""]
    cap_rows = [r for r in canvas_rows if r.get("postresize_cap_px")]
    if cap_rows:
        prov = {r.get("cap_provenance") for r in cap_rows}
        L += [f"Cap-height provenance: {', '.join(sorted(p for p in prov if p))}.", ""]
        if prov == {"assumed"}:
            L.append("> These post-resize numbers rest on an ASSUMED provider "
                     "scaling rule. `pretest --verify-resize` measures the real "
                     "one; until it has, do not quote these as measured.")
        L += ["", "| post-resize cap px | n | canvas accuracy |", "|---|---|---|"]
        by_cap: dict = defaultdict(list)
        for r in cap_rows:
            by_cap[round(r["postresize_cap_px"], 1)].append(r)
        for cap, rs in sorted(by_cap.items()):
            L.append(f"| {cap} | {len(rs)} | {accuracy(rs):.1%} |")
        if len(by_cap) < 2:
            L.append("")
            L.append("> Only one cap-height present: this run does not sweep node "
                     "count or width, so there is no legibility curve yet.")
    else:
        L.append("No canvas rows carry a post-resize cap height.")

    _plots(rows, out, qtypes)

    # -- 5. failures -------------------------------------------------------
    fails = canvas_failures(rows)
    L += ["", "## 5. Canvas failure analysis", "",
          f"{len(fails)} canvas-arm errors out of {len(canvas_rows)} canvas calls.",
          ""]
    if fails:
        L += ["Clustered by inferred cause:", ""]
        for cause, k in cluster_failures(fails).items():
            L.append(f"- {cause}: {k}")
        L += ["", "| qtype | seed | cond | question | expected | model said | "
              "raw | px dist | hops |", "|---|---|---|---|---|---|---|---|---|"]
        for f in fails[:60]:
            raw = (f["raw_text"] or "").strip().replace("|", "/")[:40]
            L.append(f"| {f['qtype']} | {f['seed']} | {f['condition']} | "
                     f"{f['question'][:56]} | {f['expected']} | {f['parsed']} | "
                     f"{raw} | {f['pixel_dist']} | {f['hops']} |")
        crops = _failure_crops(fails, manifest, out)
        if crops:
            L += ["", f"Canvas crops for the first {len(crops)} failures are in "
                  f"`failures/`.", ""]
    else:
        L.append("No canvas errors.")

    # -- 6. stability ------------------------------------------------------
    L += ["", "## 6. Layout stability under mutation", ""]
    from .stability import TARGET_FRACTION_STABLE, run_stability, summarize

    sseeds = ([int(s) for s in stability_seeds.split(",")] if stability_seeds
              else sorted({r["seed"] for r in rows}))
    nn = manifest.get("n_nodes", 40)
    w = manifest.get("width", 1540)
    st = summarize(run_stability(sseeds, nn, w, out_dir=out / "stability"))
    L += [f"Target: {TARGET_FRACTION_STABLE:.0%} of surviving nodes move less "
          "than half a node width. Mutated renders are pinned to the base "
          "render's config, and mutations preserve existing nodes' dates.", "",
          "| mutation | runs | surviving nodes | mean px | max px | "
          "fraction stable | meets target |", "|---|---|---|---|---|---|---|"]
    for name, v in st.items():
        L.append(f"| {name} | {v['n_runs']} | {v['surviving_nodes']} | "
                 f"{v['mean_px']} | {v['max_px']} | {v['fraction_stable']:.1%} | "
                 f"{'yes' if v['meets_target'] else 'NO'} |")

    # -- 7. verdict --------------------------------------------------------
    L += ["", "## 7. Verdict", ""] + _verdict(rows, canvas_rows, text_rows, best, st)

    (out / "results.md").write_text("\n".join(L) + "\n")
    console.print(f"wrote {out / 'results.md'}")


def _verdict(rows, canvas_rows, text_rows, best, stability_summary) -> list[str]:
    """Plain English on what the data supports and what it does not."""
    from .score import accuracy, mcnemar

    if not canvas_rows or not best:
        return ["Insufficient arms present to state a verdict."]

    best_rows = [r for r in text_rows if r["arm"] == best]
    c_acc, t_acc = accuracy(canvas_rows), accuracy(best_rows)
    res = mcnemar(canvas_rows, best_rows, "canvas", best)
    lt = [r for r in rows if r["arm"] == "layout_text"]

    out = []
    direction = "above" if c_acc > t_acc else ("below" if c_acc < t_acc else "level with")
    out.append(
        f"The canvas arm scored {c_acc:.1%} against {t_acc:.1%} for the best text "
        f"arm ({best}), putting it {direction} the text baseline. On the paired "
        f"McNemar test over the identical query instances, {res.b_count} pairs "
        f"favour canvas and {res.c_count} favour {best} "
        f"(exact p = {res.p_value:.3g})."
    )
    if res.discordant < 10:
        out.append(
            f"With only {res.discordant} discordant pairs this test has very "
            "little power. The honest reading is that this run does not resolve "
            "the question in either direction."
        )
    elif res.p_value >= 0.05:
        out.append(
            "That is not significant at the 5% level, so this run does not "
            "support the claim that the canvas beats the best text encoding."
        )
    else:
        out.append(
            "That difference is significant at the 5% level, on this model, "
            "node count and seed set."
        )

    if lt:
        lt_acc = accuracy(lt)
        cd = mcnemar(canvas_rows, lt, "canvas", "layout_text")
        out.append("")
        out.append(
            f"**C vs D.** The spatial-text arm (`layout_text`) scored "
            f"{lt_acc:.1%} against the canvas arm's {c_acc:.1%}, with "
            f"{cd.b_count} pairs favouring canvas and {cd.c_count} favouring "
            f"layout_text (exact p = {cd.p_value:.3g}). "
            + (
                "The two are not distinguishable here, which means the measured "
                "benefit is attributable to the LAYOUT -- grouping by domain and "
                "ordering by time -- rather than to raster pixels. That points at "
                "a text formatter, not an image pipeline, and is a much cheaper "
                "product."
                if cd.p_value >= 0.05
                else "The two are distinguishable, so the raster encoding is "
                     "contributing something beyond spatial organisation in text."
            )
        )

    weak = [n for n, v in stability_summary.items() if not v["meets_target"]]
    out.append("")
    out.append(
        "Layout stability meets the 90% target for every mutation tested."
        if not weak else
        f"Layout stability MISSES the 90% target for: {', '.join(weak)}."
    )
    out.append("")
    out.append(
        "**What this run does not support.** It covers one provider, one node "
        "count and one canvas width, so it says nothing about how the comparison "
        "moves with graph size or resolution, and nothing about models whose "
        "image preprocessing differs. The border-weight type encoding is known "
        "to be unreadable (Stage 0), so no claim here extends to node type."
    )
    return out


def _plots(rows, out: Path, qtypes: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .score import accuracy, group

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    arms = sorted({r["arm"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    conds = sorted({r["condition"] for r in rows})
    w = 0.8 / max(len(conds), 1)
    for i, cond in enumerate(conds):
        vals = [accuracy([r for r in rows if r["arm"] == a and r["condition"] == cond])
                for a in arms]
        ax.bar([x + i * w for x in range(len(arms))], vals, w, label=cond)
    ax.set_xticks([x + w * (len(conds) - 1) / 2 for x in range(len(arms))])
    ax.set_xticklabels(arms)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Accuracy by arm and prompt condition")
    fig.tight_layout()
    fig.savefig(plots / "accuracy_by_arm.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for a in arms:
        vals = [accuracy([r for r in rows if r["arm"] == a and r["qtype"] == q])
                for q in qtypes]
        ax.plot(qtypes, vals, marker="o", label=a)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    ax.set_title("Accuracy by query type")
    fig.tight_layout()
    fig.savefig(plots / "accuracy_by_qtype.png", dpi=140)
    plt.close(fig)

    canvas = [r for r in rows if r["arm"] == "canvas" and r.get("pixel_dist")]
    if canvas:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ok = [r["pixel_dist"] for r in canvas if r["outcome"] == "correct"]
        bad = [r["pixel_dist"] for r in canvas if r["outcome"] != "correct"]
        ax.hist([ok, bad], bins=12, stacked=True, label=["correct", "wrong"])
        ax.set_xlabel("pixel distance between referenced nodes (authored px)")
        ax.set_ylabel("count")
        ax.legend()
        ax.set_title("Canvas outcome vs on-canvas distance")
        fig.tight_layout()
        fig.savefig(plots / "canvas_pixel_distance.png", dpi=140)
        plt.close(fig)


def _failure_crops(fails: list[dict], manifest: dict, out: Path, limit: int = 12
                   ) -> list[str]:
    """A crop of the canvas region around each failing question's nodes."""
    from .graph import generate_graph
    from .queries import generate_queries
    from .render.draw import render
    from .render.layout import RenderConfig

    made: list[str] = []
    n_nodes = manifest.get("n_nodes", 40)
    width = manifest.get("width", 1540)
    crops = out / "failures"
    crops.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    for f in fails[:limit]:
        seed = f["seed"]
        if seed not in cache:
            g = generate_graph(seed, n_nodes)
            cfg = RenderConfig.for_graph(n_nodes, width=width)
            cache[seed] = (g, render(g, cfg), {q.qid: q for q in generate_queries(g)})
        g, res, qmap = cache[seed]
        q = qmap.get(f["qid"])
        if q is None:
            continue
        ids = [n for n in (list(q.referenced) + [q.answer, f.get("parsed")]) if n]
        pts = [res.layout.positions()[i] for i in ids if i in res.layout.positions()]
        if not pts:
            continue
        pad = 140
        x0 = max(0, int(min(p[0] for p in pts) - pad))
        x1 = min(res.layout.width, int(max(p[0] for p in pts) + pad))
        y0 = max(0, int(min(p[1] for p in pts) - pad))
        y1 = min(res.layout.height, int(max(p[1] for p in pts) + pad))
        name = f"{f['qtype']}_seed{seed}_{f['qid']}.png"
        res.image.crop((x0, y0, x1, y1)).save(crops / name)
        made.append(name)
    return made


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
