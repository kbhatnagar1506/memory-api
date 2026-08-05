"""Query generation and ground truth.

Ground truth is computed by traversing the graph -- never hand-written, never
inferred from a payload. Every answer is a single node ID, so scoring is exact
string match and all four arms are scored by identical code.

Capacity note (this touches ground truth, so it is stated loudly rather than
buried): rule (a) below requires that no two instances of the same type share an
answer node. That caps some types below the requested `n_per_type`:

  * oldest_in_domain  -> at most 4 instances (one per domain, by construction)
  * root_cause        -> at most one per chain (6 at 20 nodes, 8 at 40/80)

`generate_queries` therefore treats n_per_type as a *target*, samples
min(target, capacity), and records both numbers in QuerySet.capacity so the
report can state exactly how many questions of each type were actually asked.
Nothing is padded to hit a round number.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, asdict

from .graph import DOMAINS, Graph

QUERY_TYPES: tuple[str, ...] = (
    "temporal_pair",
    "oldest_in_domain",
    "root_cause",
    "cross_domain_effect",
    "contradiction",
    "supersede",
)

#: The types that actually test the hypothesis (multi-hop relational reads).
MULTIHOP_TYPES = ("root_cause", "cross_domain_effect")
#: The types that test whether the model can read the axes at all.
CAPABILITY_TYPES = ("temporal_pair", "oldest_in_domain")


@dataclass
class QueryInstance:
    qid: str
    qtype: str
    question: str
    answer: str
    args: tuple[str, ...]
    #: Nodes named in the question text (never includes the answer, except for
    #: temporal_pair where the answer is one of the two named nodes).
    referenced: tuple[str, ...]
    hops: int
    crosses_domain: bool
    t_gap: int | None = None
    #: Filled in by annotate_with_layout(); center-to-center distance in
    #: authored pixels between the anchor node and the answer node.
    pixel_dist: float | None = None
    postresize_pixel_dist: float | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["args"] = list(self.args)
        d["referenced"] = list(self.referenced)
        return d


@dataclass
class QuerySet:
    seed: int
    n_nodes: int
    items: list[QueryInstance]
    #: qtype -> {"requested": int, "available": int, "sampled": int}
    capacity: dict[str, dict[str, int]] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def by_type(self, qtype: str) -> list[QueryInstance]:
        return [q for q in self.items if q.qtype == qtype]


def _qid(seed: int, n_nodes: int, qtype: str, args: tuple[str, ...]) -> str:
    raw = "|".join([str(seed), str(n_nodes), qtype, *args])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-type candidate enumeration. Each returns (args, answer, meta) triples.
# Enumeration is exhaustive and deterministic; sampling happens afterwards.
# ---------------------------------------------------------------------------


def _cand_temporal_pair(g: Graph) -> list[tuple[tuple[str, ...], str, dict]]:
    out = []
    ids = sorted(g.ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            na, nb = g.node(a), g.node(b)
            older = a if na.t < nb.t else b
            out.append(
                (
                    (a, b),
                    older,
                    {
                        "hops": 0,
                        "crosses_domain": na.domain != nb.domain,
                        "t_gap": abs(na.t - nb.t),
                    },
                )
            )
    return out


def _cand_oldest_in_domain(g: Graph) -> list[tuple[tuple[str, ...], str, dict]]:
    out = []
    for d in DOMAINS:
        members = g.nodes_in_domain(d)
        if len(members) < 2:
            continue
        oldest = min(members, key=lambda n: n.t)
        out.append(((d,), oldest.id, {"hops": 0, "crosses_domain": False, "t_gap": None}))
    return out


def _cand_root_cause(g: Graph) -> list[tuple[tuple[str, ...], str, dict]]:
    out = []
    for chain in g.chains:
        head = chain[0]
        for pos, member in enumerate(chain):
            if pos == 0:
                continue  # asking the head about itself is degenerate
            path = chain[: pos + 1]
            crosses = any(
                g.node(x).domain != g.node(y).domain for x, y in zip(path, path[1:])
            )
            out.append(
                (
                    (member,),
                    head,
                    {"hops": pos, "crosses_domain": crosses, "t_gap": None},
                )
            )
    return out


def _cand_cross_domain_effect(g: Graph) -> list[tuple[tuple[str, ...], str, dict]]:
    """Only (source, domain) pairs with exactly one caused target in that domain.

    We additionally require the target domain to differ from the source domain;
    otherwise the question is not a cross-domain question at all.
    """
    out = []
    for src in sorted(g.ids):
        targets = g.caused_out(src)
        if not targets:
            continue
        src_domain = g.node(src).domain
        for d in DOMAINS:
            in_d = [x for x in targets if g.node(x).domain == d]
            if len(in_d) != 1:
                continue  # rule (b)
            if d == src_domain:
                continue
            out.append(
                (
                    (src, d),
                    in_d[0],
                    {"hops": 1, "crosses_domain": True, "t_gap": None},
                )
            )
    return out


def _cand_contradiction(g: Graph) -> list[tuple[tuple[str, ...], str, dict]]:
    out = []
    for e in g.rel_edges("contradicts"):
        for anchor, answer in ((e.src, e.dst), (e.dst, e.src)):
            out.append(
                (
                    (anchor,),
                    answer,
                    {"hops": 1, "crosses_domain": False, "t_gap": None},
                )
            )
    return out


def _cand_supersede(g: Graph) -> list[tuple[tuple[str, ...], str, dict]]:
    """Two phrasings per pair.

    A supersedes pair yields only one question under a single phrasing, so 4-6
    pairs could never supply 8 instances. Asking both directions doubles the
    supply and, usefully, also tests whether the model reads edge *direction*
    rather than just edge presence. Recorded in DECISIONS.md.
    """
    out = []
    for e in g.rel_edges("supersedes"):
        newer, older = e.src, e.dst
        # "Which node supersedes <older>?" -> the newer one
        out.append(
            (
                (older, "forward"),
                newer,
                {"hops": 1, "crosses_domain": g.node(newer).domain != g.node(older).domain},
            )
        )
        # "Which node does <newer> supersede?" -> the older one
        out.append(
            (
                (newer, "reverse"),
                older,
                {"hops": 1, "crosses_domain": g.node(newer).domain != g.node(older).domain},
            )
        )
    return out


_CANDIDATE_FNS = {
    "temporal_pair": _cand_temporal_pair,
    "oldest_in_domain": _cand_oldest_in_domain,
    "root_cause": _cand_root_cause,
    "cross_domain_effect": _cand_cross_domain_effect,
    "contradiction": _cand_contradiction,
    "supersede": _cand_supersede,
}


# ---------------------------------------------------------------------------
# Question text. Identical across arms -- the question lives outside the payload.
# ---------------------------------------------------------------------------


def _question_text(qtype: str, args: tuple[str, ...]) -> str:
    if qtype == "temporal_pair":
        return f"Which of {args[0]} and {args[1]} is older?"
    if qtype == "oldest_in_domain":
        return f"Which node in the {args[0]} domain is the oldest?"
    if qtype == "root_cause":
        return (
            f"Following causal edges backward from {args[0]}, "
            "which node is the original cause?"
        )
    if qtype == "cross_domain_effect":
        return f"Which node in the {args[1]} domain was directly caused by {args[0]}?"
    if qtype == "contradiction":
        return f"Which node contradicts {args[0]}?"
    if qtype == "supersede":
        if args[1] == "forward":
            return f"Which node supersedes {args[0]}?"
        return f"Which node does {args[0]} supersede?"
    raise ValueError(f"unknown query type {qtype}")


def _referenced_nodes(qtype: str, args: tuple[str, ...]) -> tuple[str, ...]:
    if qtype == "temporal_pair":
        return (args[0], args[1])
    if qtype == "oldest_in_domain":
        return ()
    return (args[0],)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_distinct_answers(
    candidates: list[tuple[tuple[str, ...], str, dict]],
    k: int,
    rng: random.Random,
) -> list[tuple[tuple[str, ...], str, dict]]:
    """Rule (a): at most one instance per answer node."""
    by_answer: dict[str, list] = {}
    for cand in candidates:
        by_answer.setdefault(cand[1], []).append(cand)
    answers = sorted(by_answer)
    rng.shuffle(answers)
    chosen = []
    for ans in answers[:k]:
        group = sorted(by_answer[ans], key=lambda c: c[0])
        chosen.append(group[rng.randrange(len(group))])
    return chosen


def _sample_temporal_pairs(
    candidates: list[tuple[tuple[str, ...], str, dict]],
    k: int,
    rng: random.Random,
    n_nodes: int,
) -> list[tuple[tuple[str, ...], str, dict]]:
    """Rule (c): stratify by t_gap so accuracy-vs-adjacency is measurable.

    Strata are k equal-width bands over [1, n_nodes-1]. Within a stratum we take
    the first candidate whose answer node is not already used.
    """
    if k == 0:
        return []
    width = max(1, (n_nodes - 1) // k)
    used_answers: set[str] = set()
    chosen: list = []
    for s in range(k):
        lo = 1 + s * width
        hi = n_nodes - 1 if s == k - 1 else lo + width - 1
        bucket = [c for c in candidates if lo <= c[2]["t_gap"] <= hi]
        rng.shuffle(bucket)
        pick = next((c for c in bucket if c[1] not in used_answers), None)
        if pick is None:
            continue
        used_answers.add(pick[1])
        chosen.append(pick)
    # Backfill if some strata were empty, still honouring rule (a).
    if len(chosen) < k:
        rest = [c for c in candidates if c[1] not in used_answers]
        rng.shuffle(rest)
        for c in rest:
            if len(chosen) >= k:
                break
            if c[1] in used_answers:
                continue
            used_answers.add(c[1])
            chosen.append(c)
    return chosen


def generate_queries(g: Graph, n_per_type: int = 8) -> QuerySet:
    """Deterministic query set for a graph."""
    rng = random.Random(g.seed * 7919 + g.n_nodes)
    items: list[QueryInstance] = []
    capacity: dict[str, dict[str, int]] = {}

    for qtype in QUERY_TYPES:
        candidates = _CANDIDATE_FNS[qtype](g)
        available = len({c[1] for c in candidates})  # distinct answer nodes
        k = min(n_per_type, available)
        if qtype == "temporal_pair":
            chosen = _sample_temporal_pairs(candidates, k, rng, g.n_nodes)
        else:
            chosen = _sample_distinct_answers(candidates, k, rng)

        capacity[qtype] = {
            "requested": n_per_type,
            "available": available,
            "sampled": len(chosen),
        }

        for args, answer, meta in sorted(chosen, key=lambda c: c[0]):
            items.append(
                QueryInstance(
                    qid=_qid(g.seed, g.n_nodes, qtype, args),
                    qtype=qtype,
                    question=_question_text(qtype, args),
                    answer=answer,
                    args=args,
                    referenced=_referenced_nodes(qtype, args),
                    hops=meta["hops"],
                    crosses_domain=meta["crosses_domain"],
                    t_gap=meta.get("t_gap"),
                )
            )

    _assert_query_invariants(g, items)
    return QuerySet(seed=g.seed, n_nodes=g.n_nodes, items=items, capacity=capacity)


def _assert_query_invariants(g: Graph, items: list[QueryInstance]) -> None:
    seen_qids = set()
    per_type_answers: dict[str, set[str]] = {}
    for q in items:
        assert q.qid not in seen_qids, f"duplicate qid {q.qid}"
        seen_qids.add(q.qid)
        assert q.answer in g.ids, f"answer {q.answer} not in graph"
        answers = per_type_answers.setdefault(q.qtype, set())
        assert q.answer not in answers, f"duplicate answer {q.answer} for {q.qtype}"
        answers.add(q.answer)
        # The question must never name its own answer, except temporal_pair
        # where choosing between two named nodes is the whole task.
        if q.qtype != "temporal_pair":
            assert q.answer not in q.referenced, f"{q.qid} names its own answer"
        assert _verify_against_graph(g, q), f"ground truth mismatch for {q.qid}"


def _verify_against_graph(g: Graph, q: QueryInstance) -> bool:
    """Independent recomputation of the answer. Belt and braces."""
    if q.qtype == "temporal_pair":
        a, b = q.args
        return q.answer == (a if g.node(a).t < g.node(b).t else b)
    if q.qtype == "oldest_in_domain":
        members = g.nodes_in_domain(q.args[0])
        return q.answer == min(members, key=lambda n: n.t).id
    if q.qtype == "root_cause":
        return q.answer == g.root_cause(q.args[0])
    if q.qtype == "cross_domain_effect":
        src, dom = q.args
        hits = [x for x in g.caused_out(src) if g.node(x).domain == dom]
        return len(hits) == 1 and hits[0] == q.answer
    if q.qtype == "contradiction":
        return q.answer == g.contradiction_of(q.args[0])
    if q.qtype == "supersede":
        anchor, direction = q.args
        if direction == "forward":
            return q.answer == g.superseded_by(anchor)
        return q.answer == g.supersedes_what(anchor)
    return False


def annotate_with_layout(qs: QuerySet, positions: dict[str, tuple[float, float]],
                         resize_scale: float = 1.0) -> QuerySet:
    """Record the on-canvas distance between the anchor node and the answer.

    This is how we find out *why* the canvas arm wins or loses, not just whether
    it does: if canvas errors cluster at large pixel distances, the failure is
    about visual search range, not about the representation.
    """
    for q in qs.items:
        if q.qtype == "temporal_pair":
            a, b = q.args
        elif q.qtype == "oldest_in_domain":
            q.pixel_dist = None
            q.postresize_pixel_dist = None
            continue
        else:
            a, b = q.args[0], q.answer
        if a not in positions or b not in positions:
            continue
        (x1, y1), (x2, y2) = positions[a], positions[b]
        d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        q.pixel_dist = round(d, 1)
        q.postresize_pixel_dist = round(d * resize_scale, 1)
    return qs
