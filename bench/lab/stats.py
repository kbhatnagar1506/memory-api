"""Paired significance for arm comparisons. No scipy, no approximations.

Every lab result so far has been a bare fraction -- 30/37 against 28/37 -- and
the honest reading of those was "n is too small", written into the commit
message because the numbers could not say otherwise. This module is what makes
n=150 worth paying for: it turns two fractions into a claim with a p-value, or
into an explicit failure to reject.

TWO TESTS, and picking the wrong one is the classic way to manufacture a win.

`sign_test` is for PAIRED arms: the same question, the same evidence, a
different mechanism. Questions differ enormously in difficulty, and that
variance is shared by both arms, so pairing removes it. Only the DISCORDANT
pairs carry information -- questions both arms got right, or both got wrong,
say nothing about which is better. This is McNemar's test in its exact
binomial form, which is correct at any n; the chi-square version is not, and
the discordant counts here are small enough for that to matter.

`wilson` is for reporting one arm's accuracy honestly. The textbook
normal-approximation interval puts its bound above 1.0 near the top of the
range and produces an interval of width zero at exactly 1.0, which is how a
7/7 gets reported as certainty. Wilson does neither.
"""

from __future__ import annotations

from math import comb, sqrt

__all__ = ["fmt_p", "sign_test", "wilson"]


def sign_test(a_only: int, b_only: int) -> float:
    """Two-sided exact p-value for `a_only` vs `b_only` discordant pairs.

    Under the null -- the two arms are equally good -- each discordant pair is
    an independent coin flip, so the count of a-wins is Binomial(n, 0.5). The
    p-value is the two-sided tail beyond the observed split.

    Returns 1.0 when there are no disagreements at all, which is the correct
    answer: two arms that never differ have produced no evidence of difference.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = max(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Default z is 95%."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_p(p: float) -> str:
    """`p<0.001` rather than `p=0.0000`, and a plain marker for significance."""
    mark = "*" if p < 0.05 else " "
    return f"p<0.001{mark}" if p < 0.001 else f"p={p:.3f}{mark}"
