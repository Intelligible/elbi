"""Extrapolation gate: is a prediction being made outside the training support?

A model is only trustworthy within the region its training data covered; a prediction
for a point outside that region is extrapolation, and the model's behaviour there is
unconstrained. This gate checks a query point two ways: per-feature, whether each value
lies within the observed range, and jointly, whether its Mahalanobis distance from the
training distribution exceeds the chi-square bound (which also catches a point that is
in-range on every feature but in an unobserved *combination*).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _chi2_sf, _dot, _frame, _inv
from ._report import Check, VerificationReport


def verify_extrapolation(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    query: dict[str, float],
) -> VerificationReport:
    """Verify that ``query`` lies within the training support of ``features``.

    The verdict is ``unsound`` (the point is outside the observed range of a feature, or
    its Mahalanobis distance exceeds the 0.975 chi-square bound; the model is
    extrapolating and its output there is not supported by data), ``sound`` (within
    support), or ``inconclusive`` (too few rows or missing query values).
    """
    feats = [f for f in features if f in query]
    data, n = _frame(rows, feats)
    if n < 30 or len(feats) < 1:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("need at least 30 rows and one queried feature in support",),
        )
    outside = []
    for f in feats:
        col = data[f]
        if query[f] < min(col) or query[f] > max(col):
            outside.append(f)
    d2 = _mahalanobis2(data, feats, query)
    # Mahalanobis^2 is chi-square with len(feats) df under approximate normality
    joint_out = d2 is not None and _chi2_sf(d2, len(feats)) < 0.025
    extrapolating = bool(outside) or joint_out
    detail = (
        f"outside the observed range on {outside}"
        if outside
        else (f"Mahalanobis D² = {d2:.1f}" if d2 is not None else "within range")
    )
    checks = (Check("in-support", not extrapolating, detail),)
    if outside:
        pivotal: str | None = (
            f"the query is outside the training range on {outside}: the model is "
            "extrapolating and its prediction there is unsupported"
        )
    elif joint_out:
        pivotal = (
            "the query is within range on each feature but in a combination far from "
            "the training data (Mahalanobis): still extrapolation"
        )
    else:
        pivotal = None
    verdict = "unsound" if extrapolating else "sound"
    return VerificationReport(verdict, d2 or 0.0, True, checks, pivotal, ())


def _mahalanobis2(
    data: dict[str, list[float]], feats: Sequence[str], query: dict[str, float]
) -> float | None:
    """Squared Mahalanobis distance of ``query`` from the training distribution."""
    k = len(feats)
    n = len(data[feats[0]])
    means = {f: math.fsum(data[f]) / n for f in feats}
    cov = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            fi, fj = feats[i], feats[j]
            c = math.fsum(
                (data[fi][r] - means[fi]) * (data[fj][r] - means[fj]) for r in range(n)
            ) / (n - 1)
            cov[i][j] = cov[j][i] = c
    for i in range(k):  # ridge for numerical stability / singular covariance
        cov[i][i] += 1e-9
    inv = _inv(cov)
    if inv is None:
        return None
    delta = [query[f] - means[f] for f in feats]
    return _dot(delta, [_dot(inv[i], delta) for i in range(k)])
