"""Layout stability under graph mutation (§7).

Positional priors only help a model if a node stays put when the graph changes,
and human auditability depends on it too. So this measures the thing rather than
asserting it.

Two details matter for the measurement to be honest:

  * Mutations preserve existing nodes' DATES. `generate_graph` derives dates
    from time rank and total node count, so naively regenerating with N+3 would
    move every node horizontally -- and that would be an artefact of the
    generator, not of the layout rule. Real memory graphs gain nodes without
    rewriting the timestamps of old ones, so the mutations here do the same.

  * The mutated render is pinned to the BASE render's RenderConfig. Band
    heights come from `subrows`, so letting a mutation change the node count
    into a different measured subrow value would move every band and swamp the
    signal we are trying to isolate.

What remains free to move is sub-row assignment inside a band, which is exactly
the residual cascade the first-fit packing rule can produce. That is the number
this file reports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from PIL import Image

from .graph import (
    DOMAINS,
    EPOCH_SPAN_DAYS,
    EPOCH_START,
    Edge,
    Graph,
    Node,
)
from .render.draw import render, save
from .render.layout import RenderConfig

#: The bar from the brief: 90% of surviving nodes move less than half a node
#: width. Stated here so a failure is reported rather than tuned away.
TARGET_FRACTION_STABLE = 0.90

MUTATIONS = ("add_3", "supersede_2", "delete_2")


@dataclass
class MutationResult:
    mutation: str
    seed: int
    n_nodes: int
    n_surviving: int
    mean_displacement_px: float
    max_displacement_px: float
    #: Fraction of surviving nodes that moved LESS than half a node width.
    fraction_stable: float
    threshold_px: float
    moved_nodes: list[tuple[str, float]] = field(default_factory=list)
    image_pair: str | None = None

    @property
    def meets_target(self) -> bool:
        return self.fraction_stable >= TARGET_FRACTION_STABLE

    def as_row(self) -> dict:
        return {
            "mutation": self.mutation,
            "seed": self.seed,
            "n_nodes": self.n_nodes,
            "surviving": self.n_surviving,
            "mean_px": round(self.mean_displacement_px, 1),
            "max_px": round(self.max_displacement_px, 1),
            "fraction_stable": round(self.fraction_stable, 4),
            "threshold_px": round(self.threshold_px, 1),
            "meets_target": self.meets_target,
        }


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------


def _date_between(a: str, b: str) -> str:
    """A date strictly between two ISO dates, or adjacent if they touch."""
    da, db = date.fromisoformat(a), date.fromisoformat(b)
    if (db - da).days <= 1:
        return (da + timedelta(days=1)).isoformat()
    return (da + timedelta(days=(db - da).days // 2)).isoformat()


def _fresh_id(g: Graph, offset: int) -> str:
    used = {n.id for n in g.nodes}
    i = 1
    while True:
        cand = f"N{i:02d}" if i < 100 else f"N{i}"
        if cand not in used and i > offset:
            return cand
        i += 1


def _rebuild(nodes: list[Node], edges: list[Edge], base: Graph) -> Graph:
    """A Graph carrying mutated content. Chains are recomputed for traversal."""
    ids = {n.id for n in nodes}
    kept_edges = [e for e in edges if e.src in ids and e.dst in ids]
    chains = [[m for m in ch if m in ids] for ch in base.chains]
    chains = [ch for ch in chains if len(ch) >= 2]
    return Graph(seed=base.seed, n_nodes=len(nodes), nodes=sorted(nodes, key=lambda n: n.id),
                 edges=sorted(kept_edges, key=lambda e: (e.rel, e.src, e.dst)),
                 chains=chains)


def mutate_add(g: Graph, rng: random.Random, k: int = 3) -> Graph:
    """Add k nodes at fresh dates. Existing nodes keep their dates exactly."""
    nodes = list(g.nodes)
    by_date = sorted(nodes, key=lambda n: n.date)
    max_t = max(n.t for n in nodes)
    for j in range(k):
        i = rng.randrange(len(by_date) - 1)
        new_date = _date_between(by_date[i].date, by_date[i + 1].date)
        nid = _fresh_id(_rebuild(nodes, [], g), 0)
        nodes.append(Node(
            id=nid,
            label=f"Added node {j + 1}",
            type="entity",
            domain=DOMAINS[rng.randrange(len(DOMAINS))],
            t=max_t + 1 + j,
            date=new_date,
        ))
        by_date = sorted(nodes, key=lambda n: n.date)
    return _rebuild(nodes, list(g.edges), g)


def mutate_supersede(g: Graph, rng: random.Random, k: int = 2) -> Graph:
    """Add k newer nodes, each superseding an existing one.

    This is the mutation a real memory layer performs most often, so it is the
    one where positional drift would hurt most.
    """
    nodes = list(g.nodes)
    edges = list(g.edges)
    max_t = max(n.t for n in nodes)
    last_date = max(n.date for n in nodes)
    targets = rng.sample([n for n in g.nodes], k)
    for j, target in enumerate(targets):
        nid = _fresh_id(_rebuild(nodes, [], g), 0)
        new_date = (date.fromisoformat(last_date) + timedelta(days=3 * (j + 1)))
        capped = min(new_date, EPOCH_START + timedelta(days=EPOCH_SPAN_DAYS))
        nodes.append(Node(
            id=nid,
            label=f"Supersedes {target.id}",
            type="decision",
            domain=target.domain,
            t=max_t + 1 + j,
            date=capped.isoformat(),
        ))
        edges.append(Edge(src=nid, dst=target.id, rel="supersedes"))
    return _rebuild(nodes, edges, g)


def mutate_delete(g: Graph, rng: random.Random, k: int = 2) -> Graph:
    """Delete k nodes and every edge touching them."""
    doomed = {n.id for n in rng.sample(list(g.nodes), k)}
    nodes = [n for n in g.nodes if n.id not in doomed]
    edges = [e for e in g.edges if e.src not in doomed and e.dst not in doomed]
    return _rebuild(nodes, edges, g)


MUTATORS = {
    "add_3": lambda g, rng: mutate_add(g, rng, 3),
    "supersede_2": lambda g, rng: mutate_supersede(g, rng, 2),
    "delete_2": lambda g, rng: mutate_delete(g, rng, 2),
}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def compare(base: Graph, mutated: Graph, cfg: RenderConfig,
            out_dir: Path | None = None, tag: str = "") -> MutationResult:
    base_res = render(base, cfg)
    # Pinned to the SAME config: band geometry must not move underneath us.
    mut_res = render(mutated, cfg)

    p0 = base_res.layout.positions()
    p1 = mut_res.layout.positions()
    surviving = sorted(set(p0) & set(p1))

    threshold = cfg.geom.node_w / 2
    displacements: list[tuple[str, float]] = []
    for nid in surviving:
        (x0, y0), (x1, y1) = p0[nid], p1[nid]
        displacements.append((nid, ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5))

    dists = [d for _, d in displacements] or [0.0]
    stable = sum(1 for d in dists if d < threshold)

    pair_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pair = _side_by_side(base_res.image, mut_res.image)
        pair_path = str(out_dir / f"stability_{tag}.png")
        pair.save(pair_path, format="PNG")
        save(base_res, out_dir / f"stability_{tag}_before.png")
        save(mut_res, out_dir / f"stability_{tag}_after.png")

    return MutationResult(
        mutation=tag.split("_seed")[0] if "_seed" in tag else tag,
        seed=base.seed,
        n_nodes=base.n_nodes,
        n_surviving=len(surviving),
        mean_displacement_px=sum(dists) / len(dists),
        max_displacement_px=max(dists),
        fraction_stable=stable / len(dists),
        threshold_px=threshold,
        moved_nodes=sorted(
            [(n, round(d, 1)) for n, d in displacements if d >= threshold],
            key=lambda t: -t[1],
        ),
        image_pair=pair_path,
    )


def _side_by_side(a: Image.Image, b: Image.Image) -> Image.Image:
    gap = 16
    w = a.width + b.width + gap
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (w, h), (210, 214, 222))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    return canvas


def run_stability(seeds: list[int], n_nodes: int = 40, width: int = 1540,
                  out_dir: Path | None = None) -> list[MutationResult]:
    from .graph import generate_graph

    results: list[MutationResult] = []
    for seed in seeds:
        g = generate_graph(seed, n_nodes)
        # Pin the config from the BASE graph and reuse it for every mutation.
        cfg = RenderConfig.for_graph(n_nodes, width=width)
        for name, mutator in MUTATORS.items():
            rng = random.Random(seed * 31 + hash(name) % 1000)
            mutated = mutator(g, rng)
            res = compare(g, mutated, cfg, out_dir, tag=f"{name}_seed{seed}")
            res.mutation = name
            results.append(res)
    return results


def summarize(results: list[MutationResult]) -> dict:
    by_mutation: dict[str, list[MutationResult]] = {}
    for r in results:
        by_mutation.setdefault(r.mutation, []).append(r)

    summary = {}
    for name, rs in by_mutation.items():
        total_nodes = sum(r.n_surviving for r in rs)
        total_stable = sum(round(r.fraction_stable * r.n_surviving) for r in rs)
        summary[name] = {
            "n_runs": len(rs),
            "surviving_nodes": total_nodes,
            "mean_px": round(sum(r.mean_displacement_px for r in rs) / len(rs), 1),
            "max_px": round(max(r.max_displacement_px for r in rs), 1),
            "fraction_stable": round(total_stable / total_nodes, 4) if total_nodes else 0,
            "meets_target": (total_stable / total_nodes >= TARGET_FRACTION_STABLE)
            if total_nodes else False,
        }
    return summary
