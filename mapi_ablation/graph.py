"""Seeded synthetic knowledge-graph generator.

Everything here is deterministic from an integer seed: the same seed produces a
byte-identical serialization on any machine. The structural invariants below are
not stylistic -- they are what guarantee that every generated query has exactly
one defensible answer. If generation cannot satisfy an invariant, it retries
with a bumped internal counter; it never relaxes the invariant.

One non-obvious invariant deserves calling out: node IDs are deliberately
*decorrelated* from time order. If N01 were always the oldest node, the temporal
queries would be answerable from the ID string alone, without reading the
payload at all, and every arm would score identically for the wrong reason.
See ID_TIME_RHO_MAX.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from .vocab import DOMAIN_VOCAB

# ---------------------------------------------------------------------------
# Fixed grammar of the graph
# ---------------------------------------------------------------------------

NODE_TYPES: tuple[str, ...] = ("entity", "decision", "event", "preference")

#: Band order on the canvas is this tuple's order, which is sorted by name, so
#: a domain sits at the same vertical position in every render ever produced.
DOMAINS: tuple[str, ...] = ("Finance", "Growth", "People", "Platform")

#: Global time origin and span. Fixed constants, NOT derived per-graph -- the
#: renderer maps dates to x with this same fixed scale so that adding a node
#: cannot shift every other node horizontally.
EPOCH_START = date(2025, 1, 1)
EPOCH_SPAN_DAYS = 548  # ~18 months

RELATIONS: tuple[str, ...] = ("caused", "contradicts", "supersedes")

#: Max |Spearman rho| permitted between ID order and time order.
ID_TIME_RHO_MAX = 0.35

MAX_ATTEMPTS = 400


class GraphGenerationError(RuntimeError):
    """Raised when the generator cannot satisfy its invariants."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Node:
    id: str
    label: str
    type: str
    domain: str
    t: int  # unique rank, 1..N; smaller == older
    date: str  # ISO date, strictly increasing in t

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "domain": self.domain,
            "t": self.t,
            "date": self.date,
        }


@dataclass(frozen=True, order=True)
class Edge:
    src: str
    dst: str
    rel: str

    def as_dict(self) -> dict:
        return {"from": self.src, "to": self.dst, "rel": self.rel}


@dataclass
class Graph:
    seed: int
    n_nodes: int
    nodes: list[Node]
    edges: list[Edge]
    chains: list[list[str]] = field(default_factory=list)

    # -- lookups ----------------------------------------------------------

    def __post_init__(self) -> None:
        self._by_id = {n.id: n for n in self.nodes}

    def node(self, nid: str) -> Node:
        return self._by_id[nid]

    @property
    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def rel_edges(self, rel: str) -> list[Edge]:
        return [e for e in self.edges if e.rel == rel]

    def nodes_in_domain(self, domain: str) -> list[Node]:
        return [n for n in self.nodes if n.domain == domain]

    def caused_out(self, nid: str) -> list[str]:
        return [e.dst for e in self.edges if e.rel == "caused" and e.src == nid]

    def caused_in(self, nid: str) -> list[str]:
        return [e.src for e in self.edges if e.rel == "caused" and e.dst == nid]

    def root_cause(self, nid: str) -> str:
        """Walk `caused` edges backward to the chain head.

        Chains are node-disjoint and linear, so this terminates at exactly one
        node and needs no cycle guard beyond the assertion.
        """
        seen = {nid}
        cur = nid
        while True:
            parents = self.caused_in(cur)
            if not parents:
                return cur
            assert len(parents) == 1, f"{cur} has {len(parents)} causal parents"
            cur = parents[0]
            assert cur not in seen, "cycle in caused edges"
            seen.add(cur)

    def contradiction_of(self, nid: str) -> str | None:
        for e in self.edges:
            if e.rel != "contradicts":
                continue
            if e.src == nid:
                return e.dst
            if e.dst == nid:
                return e.src
        return None

    def superseded_by(self, nid: str) -> str | None:
        """The newer node that supersedes `nid`, if any."""
        for e in self.edges:
            if e.rel == "supersedes" and e.dst == nid:
                return e.src
        return None

    def supersedes_what(self, nid: str) -> str | None:
        """The older node that `nid` supersedes, if any."""
        for e in self.edges:
            if e.rel == "supersedes" and e.src == nid:
                return e.dst
        return None

    def chain_of(self, nid: str) -> list[str] | None:
        for chain in self.chains:
            if nid in chain:
                return chain
        return None

    # -- serialization ----------------------------------------------------

    def to_canonical_json(self) -> str:
        """Stable serialization. Byte-identical for a given seed + node count."""
        payload = {
            "seed": self.seed,
            "n_nodes": self.n_nodes,
            "nodes": [n.as_dict() for n in sorted(self.nodes, key=lambda x: x.id)],
            "edges": [
                e.as_dict() for e in sorted(self.edges, key=lambda x: (x.rel, x.src, x.dst))
            ],
            "chains": self.chains,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. Inputs here are already ranks, so this is Pearson."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _date_for_rank(t: int, n: int) -> str:
    """Map time rank -> ISO date, strictly increasing, spread over the epoch."""
    if n == 1:
        return EPOCH_START.isoformat()
    offset = ((t - 1) * EPOCH_SPAN_DAYS) // (n - 1)
    return (EPOCH_START + timedelta(days=offset)).isoformat()


def _plan_chains(n_nodes: int, rng: random.Random) -> list[int]:
    """Chain lengths: 6-8 chains, each 3-5 nodes, total within the node budget."""
    n_chains = 6 if n_nodes < 30 else 8
    lengths = [3] * n_chains
    budget = max(3 * n_chains, min(int(0.75 * n_nodes), 5 * n_chains))
    extra = budget - sum(lengths)
    order = list(range(n_chains))
    while extra > 0:
        rng.shuffle(order)
        placed = False
        for i in order:
            if extra == 0:
                break
            if lengths[i] < 5:
                lengths[i] += 1
                extra -= 1
                placed = True
        if not placed:
            break
    return lengths


def _plan_pair_counts(n_nodes: int, rng: random.Random) -> tuple[int, int]:
    """contradicts / supersedes pair counts, each 4-6, node-disjoint overall."""
    options = [
        (k, m)
        for k in (4, 5, 6)
        for m in (4, 5, 6)
        if 2 * (k + m) <= n_nodes
    ]
    if not options:
        raise GraphGenerationError(
            f"n_nodes={n_nodes} too small for 4 contradicts + 4 supersedes pairs"
        )
    return rng.choice(options)


def generate_graph(seed: int, n_nodes: int = 40) -> Graph:
    """Deterministic graph for (seed, n_nodes). Retries internally on failure."""
    if n_nodes < 16:
        raise GraphGenerationError("n_nodes must be >= 16 to satisfy the invariants")

    last_err: str = ""
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(seed * 100_003 + attempt)
        try:
            g = _try_generate(seed, n_nodes, rng)
            _assert_invariants(g)
            return g
        except GraphGenerationError as exc:  # bump the counter, try again
            last_err = str(exc)
            continue
    raise GraphGenerationError(
        f"could not generate graph for seed={seed} n_nodes={n_nodes} "
        f"in {MAX_ATTEMPTS} attempts; last failure: {last_err}"
    )


def _try_generate(seed: int, n_nodes: int, rng: random.Random) -> Graph:
    ids = [f"N{i:02d}" for i in range(1, n_nodes + 1)]

    # -- domains: balanced, >= 3 per domain -------------------------------
    per = n_nodes // len(DOMAINS)
    domain_slots: list[str] = []
    for d in DOMAINS:
        domain_slots.extend([d] * per)
    i = 0
    while len(domain_slots) < n_nodes:
        domain_slots.append(DOMAINS[i % len(DOMAINS)])
        i += 1
    rng.shuffle(domain_slots)
    domain_of = dict(zip(ids, domain_slots))

    # -- time ranks: a shuffled permutation, decorrelated from ID order ----
    ranks = list(range(1, n_nodes + 1))
    rng.shuffle(ranks)
    t_of = dict(zip(ids, ranks))
    rho = _spearman_rho(
        [float(k) for k in range(n_nodes)], [float(t_of[i_]) for i_ in ids]
    )
    if abs(rho) > ID_TIME_RHO_MAX:
        raise GraphGenerationError(f"ID order correlates with time (rho={rho:.2f})")

    # -- labels: unique within a domain -----------------------------------
    label_pools = {d: list(DOMAIN_VOCAB[d]) for d in DOMAINS}
    for d in DOMAINS:
        rng.shuffle(label_pools[d])
    label_of: dict[str, str] = {}
    for nid in ids:
        pool = label_pools[domain_of[nid]]
        if not pool:
            raise GraphGenerationError(f"vocabulary exhausted for domain {domain_of[nid]}")
        label_of[nid] = pool.pop()

    type_of = {nid: NODE_TYPES[rng.randrange(len(NODE_TYPES))] for nid in ids}

    nodes = [
        Node(
            id=nid,
            label=label_of[nid],
            type=type_of[nid],
            domain=domain_of[nid],
            t=t_of[nid],
            date=_date_for_rank(t_of[nid], n_nodes),
        )
        for nid in ids
    ]

    # -- caused: node-disjoint linear chains ------------------------------
    lengths = _plan_chains(n_nodes, rng)
    pool = list(ids)
    rng.shuffle(pool)
    if sum(lengths) > len(pool):
        raise GraphGenerationError("chain budget exceeds node count")

    chains: list[list[str]] = []
    cursor = 0
    for ln in lengths:
        members = pool[cursor : cursor + ln]
        cursor += ln
        members.sort(key=lambda nid: t_of[nid])  # edges then always run old -> new
        chains.append(members)

    crossing = sum(
        1
        for ch in chains
        if any(domain_of[a] != domain_of[b] for a, b in zip(ch, ch[1:]))
    )
    if crossing < (len(chains) + 1) // 2:
        raise GraphGenerationError(
            f"only {crossing}/{len(chains)} chains cross a domain boundary"
        )

    edges: list[Edge] = []
    for ch in chains:
        for a, b in zip(ch, ch[1:]):
            edges.append(Edge(src=a, dst=b, rel="caused"))

    # -- contradicts: same-domain, node-disjoint pairs ---------------------
    k_contra, m_super = _plan_pair_counts(n_nodes, rng)

    by_domain: dict[str, list[str]] = {d: [] for d in DOMAINS}
    for nid in ids:
        by_domain[domain_of[nid]].append(nid)
    for d in DOMAINS:
        if len(by_domain[d]) < 3:
            raise GraphGenerationError(f"domain {d} has < 3 nodes")
        rng.shuffle(by_domain[d])

    used: set[str] = set()
    contra_pairs: list[tuple[str, str]] = []
    domain_cycle = list(DOMAINS)
    rng.shuffle(domain_cycle)
    di = 0
    while len(contra_pairs) < k_contra:
        if di > 4 * len(DOMAINS) * k_contra:
            raise GraphGenerationError("could not place contradicts pairs")
        d = domain_cycle[di % len(domain_cycle)]
        di += 1
        free = [n for n in by_domain[d] if n not in used]
        if len(free) < 2:
            continue
        a, b = free[0], free[1]
        used.update({a, b})
        contra_pairs.append(tuple(sorted((a, b))))  # canonical order; symmetric
    for a, b in contra_pairs:
        edges.append(Edge(src=a, dst=b, rel="contradicts"))

    # -- supersedes: node-disjoint from contradicts and from each other ----
    remaining = [n for n in ids if n not in used]
    rng.shuffle(remaining)
    if len(remaining) < 2 * m_super:
        raise GraphGenerationError("not enough free nodes for supersedes pairs")
    super_pairs: list[tuple[str, str]] = []
    for j in range(m_super):
        a, b = remaining[2 * j], remaining[2 * j + 1]
        newer, older = (a, b) if t_of[a] > t_of[b] else (b, a)
        super_pairs.append((newer, older))
        used.update({a, b})
    for newer, older in super_pairs:
        edges.append(Edge(src=newer, dst=older, rel="supersedes"))

    return Graph(
        seed=seed,
        n_nodes=n_nodes,
        nodes=sorted(nodes, key=lambda n: n.id),
        edges=sorted(edges, key=lambda e: (e.rel, e.src, e.dst)),
        chains=chains,
    )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def _assert_invariants(g: Graph) -> None:
    """The five invariants from the spec, plus the ID/time decorrelation one.

    Raises GraphGenerationError so the caller retries with a bumped counter.
    """

    def check(cond: bool, msg: str) -> None:
        if not cond:
            raise GraphGenerationError(msg)

    ts = [n.t for n in g.nodes]
    check(len(set(ts)) == len(ts), "t values are not unique")
    check(sorted(ts) == list(range(1, g.n_nodes + 1)), "t is not a 1..N permutation")

    # dates strictly increase with t
    by_t = sorted(g.nodes, key=lambda n: n.t)
    for a, b in zip(by_t, by_t[1:]):
        check(a.date < b.date, f"dates not strictly increasing at {a.id}->{b.id}")

    t = {n.id: n.t for n in g.nodes}
    dom = {n.id: n.domain for n in g.nodes}

    # (1) temporal direction of caused / supersedes
    for e in g.edges:
        if e.rel == "caused":
            check(t[e.src] < t[e.dst], f"caused edge {e.src}->{e.dst} runs backward")
        elif e.rel == "supersedes":
            check(t[e.src] > t[e.dst], f"supersedes edge {e.src}->{e.dst} runs backward")
        elif e.rel == "contradicts":
            check(dom[e.src] == dom[e.dst], f"contradicts {e.src}/{e.dst} cross domains")

    # (2) each node in <= 1 contradicts pair and <= 1 supersedes pair,
    #     and the two relations are node-disjoint from each other
    contra_nodes: list[str] = []
    super_src: list[str] = []
    super_dst: list[str] = []
    for e in g.edges:
        if e.rel == "contradicts":
            contra_nodes.extend([e.src, e.dst])
        elif e.rel == "supersedes":
            super_src.append(e.src)
            super_dst.append(e.dst)
    check(len(set(contra_nodes)) == len(contra_nodes), "node in >1 contradicts pair")
    check(len(set(super_src)) == len(super_src), "node is source of >1 supersedes")
    check(len(set(super_dst)) == len(super_dst), "node is target of >1 supersedes")
    check(
        not (set(contra_nodes) & (set(super_src) | set(super_dst))),
        "contradicts and supersedes pairs share a node",
    )

    # (3) chains node-disjoint and linear -> unique root from any member
    flat = [n for ch in g.chains for n in ch]
    check(len(set(flat)) == len(flat), "chains share a node")
    check(all(3 <= len(ch) <= 5 for ch in g.chains), "chain length outside 3..5")
    check(6 <= len(g.chains) <= 8, "chain count outside 6..8")
    for nid in g.ids:
        check(len(g.caused_in(nid)) <= 1, f"{nid} has >1 causal parent")
        check(len(g.caused_out(nid)) <= 1, f"{nid} has >1 causal child")
    for ch in g.chains:
        check(g.caused_in(ch[0]) == [], f"chain head {ch[0]} has an incoming caused edge")
        for member in ch:
            check(g.root_cause(member) == ch[0], f"root of {member} is not {ch[0]}")

    crossing = sum(
        1 for ch in g.chains if any(dom[a] != dom[b] for a, b in zip(ch, ch[1:]))
    )
    check(
        crossing >= (len(g.chains) + 1) // 2,
        f"only {crossing}/{len(g.chains)} chains cross a domain boundary",
    )

    # (4) every domain has >= 3 nodes and a unique earliest node
    for d in DOMAINS:
        members = g.nodes_in_domain(d)
        check(len(members) >= 3, f"domain {d} has {len(members)} nodes")
        earliest = min(m.t for m in members)
        check(
            sum(1 for m in members if m.t == earliest) == 1,
            f"domain {d} has a tie for earliest node",
        )

    # (6, ours) IDs must not encode time order
    rho = _spearman_rho(
        [float(i) for i in range(len(g.nodes))],
        [float(n.t) for n in sorted(g.nodes, key=lambda x: x.id)],
    )
    check(abs(rho) <= ID_TIME_RHO_MAX, f"ID order correlates with time (rho={rho:.2f})")


def id_time_rho(g: Graph) -> float:
    """Exposed for tests and the run manifest."""
    return _spearman_rho(
        [float(i) for i in range(len(g.nodes))],
        [float(n.t) for n in sorted(g.nodes, key=lambda x: x.id)],
    )
