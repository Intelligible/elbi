"""Time-series stationarity gate: a unit root invites spurious regression.

Regressing one non-stationary series on another routinely produces a large, significant,
and entirely spurious relationship (Granger & Newbold 1974): the single most common
time-series error. Before a series is correlated, regressed, or trend- tested, it should
be checked for a unit root. This gate uses the Lo-MacKinlay variance-ratio test, whose
statistic is asymptotically standard normal (no special critical-value table): a
random-walk level has a variance ratio of one, a mean- reverting (stationary) level
below one, and a trending or positively autocorrelated level above one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _corr
from ._report import Check, VerificationReport

#: Aggregation horizons for the variance-ratio test; the strongest signal is used.
_HORIZONS = (2, 4, 8)
#: A deterministic trend explaining at least this much variance is non-stationary
#: even though the variance-ratio test (which removes the drift) would miss it.
_TREND_R2 = 0.5


def verify_stationarity(
    rows: Sequence[dict[str, Any]], value: str, time: str | None = None
) -> VerificationReport:
    """Verify that the ``value`` series is stationary (no unit root).

    Orders by ``time`` when given (else by row order) and runs the variance-ratio test
    across several horizons. The verdict is ``sound`` (mean-reverting / stationary, so
    standard correlation and regression are safe), ``unsound`` (a random walk or a
    trending series: non-stationary, so any regression on another series risks being
    spurious), or ``inconclusive`` (too short to tell). Difference the series or model
    the trend before regressing when this is unsound.
    """
    series = _ordered(rows, value, time)
    if len(series) < 50:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("a stationarity test needs at least 50 ordered points",),
        )
    pivotal: str | None = None
    # a strong deterministic trend is non-stationary, but the variance-ratio test
    # subtracts the drift and so cannot see it: catch it directly first
    trend_r2 = _corr(series, [float(i) for i in range(len(series))]) ** 2
    if trend_r2 > _TREND_R2:
        pivotal = (
            f"the series follows a deterministic trend (R² = {trend_r2:.2f} on time), "
            "so it is non-stationary: detrend it before regressing on another series"
        )
        checks = (Check("stationarity", False, f"trend R² = {trend_r2:.2f}"),)
        return VerificationReport("unsound", trend_r2, True, checks, pivotal, ())

    # the horizon with the most extreme standardised statistic
    best_vr, best_m = 1.0, 0.0
    for q in _HORIZONS:
        if len(series) > 2 * q:
            vr, m = _variance_ratio(series, q)
            if abs(m) > abs(best_m):
                best_vr, best_m = vr, m
    significant = abs(best_m) > 1.96
    stationary = significant and best_vr < 1.0
    unit_root = not significant
    if stationary:
        verdict = "sound"
    elif unit_root:
        verdict = "unsound"
        pivotal = (
            "the series has a unit root (variance ratio ≈ 1): it is a random walk, so "
            "regressing it on another risks a spurious link; difference it first"
        )
    else:  # significant and vr > 1
        verdict = "unsound"
        pivotal = (
            "the series is trending / strongly persistent (variance ratio > 1), so it "
            "is non-stationary: model or remove the trend before regressing"
        )
    checks = (
        Check(
            "stationarity",
            verdict == "sound",
            f"variance ratio = {best_vr:.2f} (z = {best_m:+.2f})",
        ),
    )
    return VerificationReport(verdict, best_vr, True, checks, pivotal, ())


def _ordered(
    rows: Sequence[dict[str, Any]], value: str, time: str | None
) -> list[float]:
    """The value series in time order (or row order when no time column is given)."""
    pairs: list[tuple[float, float]] = []
    for i, row in enumerate(rows):
        v = _as_float(row.get(value))
        if v is None:
            continue
        key = _as_float(row.get(time)) if time else float(i)
        if key is not None:
            pairs.append((key, v))
    pairs.sort(key=lambda p: p[0])
    return [v for _, v in pairs]


def _variance_ratio(series: Sequence[float], q: int) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio at horizon ``q`` and its N(0,1) statistic."""
    n = len(series)
    diffs = [series[i] - series[i - 1] for i in range(1, n)]
    mu = math.fsum(diffs) / len(diffs)
    var1 = math.fsum((d - mu) ** 2 for d in diffs) / (n - 1)
    if var1 <= 0:
        return 1.0, 0.0
    q_diffs = [series[i] - series[i - q] for i in range(q, n)]
    var_q = math.fsum((d - q * mu) ** 2 for d in q_diffs) / (q * (n - q))
    vr = var_q / var1
    phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n)
    m = (vr - 1.0) / math.sqrt(phi) if phi > 0 else 0.0
    return vr, m
