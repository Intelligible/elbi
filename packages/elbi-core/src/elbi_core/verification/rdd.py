"""Regression-discontinuity gate: was the running variable manipulated at the cutoff?

A regression-discontinuity design identifies a causal effect from units that fall just
either side of a threshold, assuming they are otherwise comparable. That assumption
breaks if units can *manipulate* their score to land on the favourable side, which
shows up as a discontinuity (bunching) in the density of the running variable at the
cutoff. The McCrary test detects it; the stdlib-feasible equivalent is an exact
binomial test that, in a symmetric window around the cutoff, points are no more likely
to sit just above it than just below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _mean_var
from ._report import Check, VerificationReport


def verify_rdd(
    rows: Sequence[dict[str, Any]], running: str, cutoff: float
) -> VerificationReport:
    """Verify the ``running`` variable was not manipulated at ``cutoff`` (RDD validity).

    Counts points in a symmetric window around the cutoff and runs an exact binomial
    test of equal mass on each side. The verdict is ``unsound`` (significant bunching:
    units appear to manipulate their score to cross the threshold, so the design is
    invalid), ``sound`` (no bunching, the density is continuous), or ``inconclusive``
    (too few points near the cutoff).
    """
    vals = [v for row in rows if (v := _as_float(row.get(running))) is not None]
    if len(vals) < 50:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("an RDD manipulation test needs at least 50 points",),
        )
    _, var = _mean_var(vals)
    bandwidth = 0.5 * math.sqrt(var)  # a window of half a standard deviation each side
    if bandwidth <= 0:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("no spread in the running variable",)
        )
    left = sum(1 for v in vals if cutoff - bandwidth <= v < cutoff)
    right = sum(1 for v in vals if cutoff <= v <= cutoff + bandwidth)
    total = left + right
    if total < 20:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("too few points within the window around the cutoff",),
        )
    p = _binomial_two_sided(right, total)
    bunched = p < 0.05
    checks = (
        Check(
            "no-manipulation",
            not bunched,
            f"{left} below vs {right} above the cutoff (p = {p:.2g})",
        ),
    )
    pivotal = (
        (
            f"the density jumps at the cutoff ({right} just above vs {left} below, "
            f"p = {p:.2g}): the running variable appears manipulated, so the "
            "discontinuity design is invalid"
        )
        if bunched
        else None
    )
    verdict = "unsound" if bunched else "sound"
    return VerificationReport(verdict, right - left, True, checks, pivotal, ())


def _binomial_two_sided(k: int, n: int, prob: float = 0.5) -> float:
    """Exact two-sided binomial p-value for ``k`` successes in ``n`` trials."""

    def pmf(i: int) -> float:
        return math.comb(n, i) * prob**i * (1 - prob) ** (n - i)

    observed = pmf(k)
    # sum the probability of every outcome no more likely than the observed one
    p = math.fsum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-12)
    return min(p, 1.0)
