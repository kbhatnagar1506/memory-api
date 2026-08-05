"""Parse, score, aggregate, bootstrap, McNemar.

Two statistical choices are load-bearing.

**The bootstrap is clustered by seed.** Queries from one graph are not
independent -- they share a node set, a layout and a set of collisions -- so
resampling individual queries would understate the interval. We resample
*seeds* with replacement and recompute accuracy over whichever queries come
with them.

**The arm comparison is paired.** Canvas and the best text arm answer the exact
same query instances, so McNemar on the discordant pairs is the right test.
Comparing two unpaired accuracies over ~44 questions per graph would mostly
measure which questions happened to land in which bucket.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260806


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.lo:.1%}-{self.hi:.1%}] n={self.n}"


def answered(rows: list[dict]) -> list[dict]:
    """Rows where we actually got a response from the model.

    `error` means the call failed -- a 429, a timeout, a laptop going to sleep
    mid-run. That is MISSING DATA, not a wrong answer, and counting it as
    incorrect would let a network blip masquerade as a weak arm. It did exactly
    that once: a dropped connection turned 160 canvas calls into apparent
    failures and moved the arm's headline by 20 points.

    `unparseable` is NOT excluded. There the model did answer, just not in the
    required format, and that is a real failure of the arm.
    """
    return [r for r in rows if r["outcome"] != "error"]


def accuracy(rows: list[dict]) -> float:
    """Accuracy over answered calls. Errors are excluded, never counted wrong."""
    ok = answered(rows)
    if not ok:
        return 0.0
    return sum(1 for r in ok if r["outcome"] == "correct") / len(ok)


def completeness(rows: list[dict]) -> tuple[int, int]:
    """(answered, attempted). Any gap must be shown, never silently averaged."""
    return len(answered(rows)), len(rows)


def cluster_bootstrap_ci(
    rows: list[dict],
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Percentile bootstrap over SEEDS, not over individual queries.

    Queries drawn from the same graph share a layout and a node set, so they are
    correlated. Resampling them independently would produce an interval that is
    too narrow and an overconfident headline.
    """
    if not rows:
        return Interval(0.0, 0.0, 0.0, 0)

    rows = answered(rows)
    if not rows:
        return Interval(0.0, 0.0, 0.0, 0)
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    seeds = sorted(by_seed)
    point = accuracy(rows)

    if len(seeds) < 2:
        # One cluster: a cluster bootstrap is undefined. Say so by returning a
        # degenerate interval rather than silently reporting a fake one.
        return Interval(point, float("nan"), float("nan"), len(rows))

    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(resamples):
        picked = [by_seed[rng.choice(seeds)] for _ in seeds]
        flat = [r for group in picked for r in group]
        stats.append(accuracy(flat))
    stats.sort()
    lo = stats[int((alpha / 2) * resamples)]
    hi = stats[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(point, lo, hi, len(rows))


# ---------------------------------------------------------------------------
# paired test
# ---------------------------------------------------------------------------


@dataclass
class McNemarResult:
    arm_a: str
    arm_b: str
    #: a correct, b wrong
    b_count: int
    #: a wrong, b correct
    c_count: int
    n_pairs: int
    p_value: float
    statistic: float

    @property
    def discordant(self) -> int:
        return self.b_count + self.c_count

    def summary(self) -> str:
        return (
            f"{self.arm_a} vs {self.arm_b}: {self.b_count} pairs favour "
            f"{self.arm_a}, {self.c_count} favour {self.arm_b} "
            f"({self.discordant} discordant of {self.n_pairs}), "
            f"exact p = {self.p_value:.4g}"
        )


def mcnemar(rows_a: list[dict], rows_b: list[dict], arm_a: str, arm_b: str
            ) -> McNemarResult:
    """Exact (binomial) McNemar on paired query instances.

    Pairing is on (model, condition, seed, n_nodes, qid) so we only ever compare
    answers to the identical question under identical conditions.
    """
    def key(r: dict):
        return (r["model_key"], r["condition"], r["seed"], r["n_nodes"], r["qid"])

    # A pair is only informative if BOTH arms actually answered it.
    a = {key(r): r for r in answered(rows_a)}
    b = {key(r): r for r in answered(rows_b)}
    shared = sorted(set(a) & set(b))

    b_count = c_count = 0
    for k in shared:
        ac = a[k]["outcome"] == "correct"
        bc = b[k]["outcome"] == "correct"
        if ac and not bc:
            b_count += 1
        elif bc and not ac:
            c_count += 1

    p = _binom_two_sided(b_count, b_count + c_count)
    n = b_count + c_count
    stat = ((abs(b_count - c_count) - 1) ** 2 / n) if n else 0.0
    return McNemarResult(arm_a, arm_b, b_count, c_count, len(shared), p, stat)


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value. Exact, not the chi-square approximation.

    With ~44 questions per graph the discordant count is small, and the
    chi-square approximation is unreliable exactly where it matters.
    """
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n, p).pvalue)
    except Exception:  # pragma: no cover - scipy is a hard dep, this is belt
        def pmf(i: int) -> float:
            return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

        target = pmf(k) * (1 + 1e-9)
        return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= target))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return dict(out)


def outcome_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"correct": 0, "wrong": 0, "unparseable": 0, "error": 0}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return counts


def cost_table(rows: list[dict]) -> dict[str, dict]:
    """Mean input tokens, mean latency, and tokens per CORRECT answer.

    Tokens-per-correct is the number that matters for a memory layer: an arm
    that is cheap but wrong is not cheap.
    """
    out: dict[str, dict] = {}
    for (arm,), rs_all in sorted(group(rows, "arm").items()):
        rs = answered(rs_all)
        toks = [r["input_tokens"] for r in rs if r.get("input_tokens")]
        lats = [r["latency_ms"] for r in rs if r.get("latency_ms")]
        n_correct = sum(1 for r in rs if r["outcome"] == "correct")
        total_tokens = sum(toks)
        out[arm] = {
            "n": len(rs),
            "attempted": len(rs_all),
            "errors": len(rs_all) - len(rs),
            "mean_input_tokens": round(sum(toks) / len(toks), 1) if toks else None,
            "mean_latency_ms": round(sum(lats) / len(lats), 1) if lats else None,
            "accuracy": round(accuracy(rs), 4),
            "tokens_per_correct": (
                round(total_tokens / n_correct, 1) if n_correct else None
            ),
        }
    return out


def best_text_arm(rows: list[dict], text_arms: tuple[str, ...]) -> str | None:
    """The strongest text baseline, chosen by accuracy on this data.

    Picked empirically rather than assumed to be `json`, so the canvas arm is
    always compared against the best competitor it actually faced.
    """
    scored = [
        (accuracy([r for r in rows if r["arm"] == a]), a)
        for a in text_arms
        if any(r["arm"] == a for r in rows)
    ]
    return max(scored)[1] if scored else None


def canvas_failures(rows: list[dict]) -> list[dict]:
    """Every canvas-arm error, with the covariates needed to explain it."""
    out = []
    for r in rows:
        if r["arm"] != "canvas" or r["outcome"] in ("correct", "error"):
            continue
        out.append({
            "qid": r["qid"],
            "qtype": r["qtype"],
            "seed": r["seed"],
            "n_nodes": r["n_nodes"],
            "condition": r["condition"],
            "model_key": r["model_key"],
            "question": r["question"],
            "expected": r["expected"],
            "parsed": r["parsed"],
            "outcome": r["outcome"],
            "raw_text": r.get("raw_text", ""),
            "pixel_dist": r.get("pixel_dist"),
            "postresize_pixel_dist": r.get("postresize_pixel_dist"),
            "hops": r.get("hops"),
            "crosses_domain": r.get("crosses_domain"),
            "t_gap": r.get("t_gap"),
            "postresize_cap_px": r.get("postresize_cap_px"),
        })
    return sorted(out, key=lambda d: (d["qtype"], d["seed"], d["qid"]))


def cluster_failures(failures: list[dict]) -> dict[str, int]:
    """Group canvas errors by an inferred cause, where one can be inferred."""
    buckets: dict[str, int] = defaultdict(int)
    for f in failures:
        if f["outcome"] == "unparseable":
            buckets["answered in prose rather than an ID"] += 1
        elif f["outcome"] == "error":
            buckets["API error"] += 1
        elif f.get("pixel_dist") is not None and f["pixel_dist"] > 900:
            buckets["referenced nodes far apart on canvas (>900px)"] += 1
        elif (f.get("hops") or 0) >= 3:
            buckets["long causal chain (>=3 hops)"] += 1
        elif f.get("t_gap") is not None and f["t_gap"] <= 3:
            buckets["temporally adjacent nodes (t_gap<=3)"] += 1
        elif f.get("crosses_domain"):
            buckets["path crosses a domain band"] += 1
        else:
            buckets["no cause inferred"] += 1
    return dict(sorted(buckets.items(), key=lambda kv: -kv[1]))
