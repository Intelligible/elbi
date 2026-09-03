"""Count-data gate: is a Poisson model appropriate, or is the count overdispersed?

Poisson regression assumes the variance of a count equals its mean. Real counts are
usually *overdispersed* (the variance is larger) because of unmodelled heterogeneity,
and fitting Poisson anyway gives standard errors that are too small and significance
that is overstated. This gate fits the Poisson model, measures the Pearson dispersion
(variance-to-mean ratio adjusted for the fit), and tests it against the equidispersion
null; overdispersion means a negative-binomial or quasi-Poisson model should be used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _chi2_sf, _frame, _inv
from ._report import Check, VerificationReport


def verify_counts(
    rows: Sequence[dict[str, Any]],
    count: str,
    predictors: Sequence[str] | None = None,
) -> VerificationReport:
    """Verify a Poisson model fits the ``count`` outcome (no overdispersion).

    Fits a Poisson regression of ``count`` on ``predictors`` (intercept-only if none),
    then tests the Pearson dispersion against the equidispersion null. The verdict is
    ``unsound`` (overdispersed: the variance exceeds the mean, so Poisson standard
    errors are too small; use negative-binomial or quasi-Poisson), ``sound`` (the
    Poisson assumption holds), or ``inconclusive`` (too few rows or the outcome is not
    counts).
    """
    preds = list(predictors or [])
    data, n = _frame(rows, [count, *preds])
    if n < 30:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("a count-model check needs at least 30 rows",),
        )
    y = data[count]
    if any(v < 0 or abs(v - round(v)) > 1e-9 for v in y):
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (f"'{count}' is not a non-negative integer count",),
        )
    design = [[1.0] * n] + [data[p] for p in preds]
    mu = _poisson_fit(design, y)
    p_params = len(design)
    pearson = math.fsum((y[k] - mu[k]) ** 2 / mu[k] for k in range(n) if mu[k] > 0)
    dispersion = pearson / max(n - p_params, 1)
    p_value = _chi2_sf(pearson, n - p_params)
    overdispersed = dispersion > 1.0 and p_value < 0.05
    checks = (
        Check(
            "equidispersion",
            not overdispersed,
            f"Pearson dispersion = {dispersion:.2f} (p = {p_value:.2g})",
        ),
    )
    pivotal = (
        (
            f"the counts are overdispersed (dispersion {dispersion:.2f} > 1, p = "
            f"{p_value:.2g}): a Poisson model understates the standard errors; use a "
            "negative-binomial or quasi-Poisson model"
        )
        if overdispersed
        else None
    )
    verdict = "unsound" if overdispersed else "sound"
    return VerificationReport(verdict, dispersion, True, checks, pivotal, ())


def _poisson_fit(design: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    """Fitted means of a Poisson regression (log link) by IRLS / Fisher scoring."""
    n, p = len(y), len(design)
    beta = [0.0] * p
    mu = [yi + 0.1 for yi in y]
    for _ in range(25):
        eta = [math.log(m) for m in mu]
        z = [eta[k] + (y[k] - mu[k]) / mu[k] for k in range(n)]
        w = mu
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
        mu = [math.exp(min(30.0, max(-30.0, e))) for e in eta]
    return mu
