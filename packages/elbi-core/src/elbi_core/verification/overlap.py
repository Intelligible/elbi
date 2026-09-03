"""Overlap / positivity gate: can two groups actually be compared?

Propensity-score matching and weighting assume positivity: that every kind of unit could
have been in either group (the treated and control propensity distributions overlap).
Where they do not, the effect for the non-overlapping units is not identified from data
and the estimate leans entirely on model extrapolation. This gate estimates the
propensity with a logistic model and measures overlap three ways: the share of units at
the extremes of the propensity (near 0 or 1), the overlap coefficient between the two
propensity distributions, and per-covariate standardized mean differences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _frame, _inv, _mean_var
from ._report import Check, VerificationReport

#: Propensities beyond [_EXTREME, 1 - _EXTREME] are near-violations of positivity.
_EXTREME = 0.1
#: A standardized mean difference beyond this is meaningful covariate imbalance.
_SMD = 0.1
#: Below this overlap coefficient the propensity distributions barely intersect.
_MIN_OVERLAP = 0.5


def verify_overlap(
    rows: Sequence[dict[str, Any]], treatment: str, covariates: Sequence[str]
) -> VerificationReport:
    """Verify the treated and control groups overlap enough to be compared.

    Fits a logistic propensity model of ``treatment`` on ``covariates`` and checks the
    overlap of the resulting propensity distributions. The verdict is ``unsound`` (poor
    overlap: many units at the propensity extremes or a low overlap coefficient, so
    positivity is near-violated and a matched/weighted estimate extrapolates), ``sound``
    (the groups overlap), or ``inconclusive`` (too few rows, not a binary treatment, or
    no covariates).
    """
    covs = list(covariates)
    data, n = _frame(rows, [treatment, *covs])
    if n < 40 or not covs:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("an overlap check needs at least 40 rows and a covariate",),
        )
    t = [round(v) for v in data[treatment]]
    if set(t) != {0, 1}:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (f"'{treatment}' must be a binary 0/1 treatment indicator",),
        )
    e = _propensity(data, covs, t)
    treated = [e[i] for i in range(n) if t[i] == 1]
    control = [e[i] for i in range(n) if t[i] == 0]
    if len(treated) < 10 or len(control) < 10:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("a group has too few units",)
        )
    extreme = sum(1 for p in e if p < _EXTREME or p > 1 - _EXTREME) / n
    ovl = _overlap_coefficient(treated, control)
    imbalanced = [c for c in covs if abs(_smd(data, c, t)) > _SMD]
    poor = extreme > 0.2 or ovl < _MIN_OVERLAP
    checks = (
        Check(
            "overlap",
            not poor,
            f"overlap coefficient = {ovl:.2f}, {extreme:.0%} extreme",
        ),
    )
    pivotal = (
        (
            f"the groups barely overlap (overlap {ovl:.2f}, {extreme:.0%} of units "
            "at the propensity extremes): positivity is near-violated, so a "
            "matched or weighted estimate extrapolates rather than compares"
        )
        if poor
        else None
    )
    verdict = "unsound" if poor else "sound"
    caveats = (
        f"covariates still imbalanced (|SMD| > 0.1): {imbalanced}"
        if imbalanced
        else "covariates are balanced (|SMD| < 0.1)",
    )
    return VerificationReport(verdict, ovl, True, checks, pivotal, caveats)


def _propensity(
    data: dict[str, list[float]], covs: Sequence[str], t: Sequence[int]
) -> list[float]:
    """Fitted propensity scores from a logistic regression of ``t`` on covariates."""
    n, p = len(t), len(covs) + 1
    design = [[1.0] * n] + [data[c] for c in covs]
    beta = [0.0] * p
    for _ in range(25):
        eta = [math.fsum(beta[j] * design[j][k] for j in range(p)) for k in range(n)]
        mu = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, e)))) for e in eta]
        w = [max(mu[k] * (1 - mu[k]), 1e-9) for k in range(n)]
        z = [eta[k] + (t[k] - mu[k]) / w[k] for k in range(n)]
        xtwx = [
            [
                math.fsum(design[i][k] * w[k] * design[j][k] for k in range(n))
                for j in range(p)
            ]
            for i in range(p)
        ]
        inv = _inv(xtwx)
        if inv is None:
            break
        xtwz = [
            math.fsum(design[i][k] * w[k] * z[k] for k in range(n)) for i in range(p)
        ]
        new = [math.fsum(inv[i][j] * xtwz[j] for j in range(p)) for i in range(p)]
        if max(abs(new[i] - beta[i]) for i in range(p)) < 1e-8:
            beta = new
            break
        beta = new
    eta = [math.fsum(beta[j] * design[j][k] for j in range(p)) for k in range(n)]
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, e)))) for e in eta]


def _overlap_coefficient(a: Sequence[float], b: Sequence[float]) -> float:
    """Histogram overlap of two [0,1] distributions (0 disjoint, 1 identical)."""
    bins = 20
    pa = [0.0] * bins
    pb = [0.0] * bins
    for v in a:
        pa[min(int(v * bins), bins - 1)] += 1.0 / len(a)
    for v in b:
        pb[min(int(v * bins), bins - 1)] += 1.0 / len(b)
    return math.fsum(min(pa[i], pb[i]) for i in range(bins))


def _smd(data: dict[str, list[float]], col: str, t: Sequence[int]) -> float:
    """Standardized mean difference of a covariate between the two groups."""
    treated = [data[col][i] for i in range(len(t)) if t[i] == 1]
    control = [data[col][i] for i in range(len(t)) if t[i] == 0]
    mt, vt = _mean_var(treated)
    mc, vc = _mean_var(control)
    pooled = math.sqrt((vt + vc) / 2.0)
    return (mt - mc) / pooled if pooled > 0 else 0.0
