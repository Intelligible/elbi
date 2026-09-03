"""Logistic-regression (GLM) coefficient gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _frame
from ._report import _T, Check, VerificationReport


def verify_logistic(
    rows: Sequence[dict[str, Any]],
    x: str,
    y: str,
    *,
    controls: Sequence[str] = (),
) -> VerificationReport:
    """Verify a logistic-regression effect (odds ratio) of ``x`` on a binary ``y``.

    The headline failure is separation: when a predictor perfectly (or nearly) splits
    the outcome, the maximum-likelihood odds ratio diverges to infinity and any finite
    number reported is an artifact of regularization, not an estimate. This detects
    perfect/quasi separation, then, when the fit is well-posed, checks the coefficient
    is significant. The verdict is ``unsound`` (separation: the odds ratio is not
    identifiable), ``sound`` (a significant, well-posed effect), or ``inconclusive`` (no
    significant effect, or too little data).
    """
    cols = [x, *controls]
    data, n = _frame(rows, [y, *cols])
    cols = [c for c in cols if c in data]
    if n < 40 or y not in data or x not in data:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few numeric rows",)
        )
    yv = [round(v) for v in data[y]]
    if set(yv) - {0, 1} or len(set(yv)) < 2:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("the outcome must be binary 0/1",)
        )

    # perfect / quasi separation by x: a threshold that (almost) splits the classes
    sep = _separation(data[x], yv)
    if sep is not None:
        return VerificationReport(
            "unsound",
            sep,
            False,
            (
                Check(
                    "identifiability",
                    False,
                    f"x separates the outcome ({sep:.0%} by a threshold)",
                ),
            ),
            "separation: a threshold on x predicts the outcome almost perfectly, so "
            "the maximum-likelihood odds ratio is infinite and not identifiable, any "
            "finite value reported is an artifact of regularization",
            ("with separation, report the threshold rule, not an odds ratio",),
        )

    beta, t = _logit_coef(data, yv, x, cols)
    if abs(t) <= _T:
        return VerificationReport(
            "inconclusive",
            math.exp(beta),
            False,
            (),
            None,
            (f"the effect of '{x}' on '{y}' is not significant",),
        )
    odds = math.exp(beta)
    checks = [
        Check("significance", True, f"odds ratio {odds:.2g} (|t| = {abs(t):.1f})")
    ]
    caveats = (
        f"a one-unit increase in '{x}' multiplies the odds of '{y}' by {odds:.2g}; "
        "an odds ratio is not a risk ratio when the outcome is common",
    )
    return VerificationReport("sound", odds, True, tuple(checks), None, caveats)


def _separation(xs: Sequence[float], ys: Sequence[int]) -> float | None:
    """Best single-threshold accuracy if it (almost) perfectly splits y, else None."""
    pairs = sorted(zip(xs, ys, strict=True))
    n = len(pairs)
    ones = sum(ys)
    best = max(ones, n - ones) / n
    left_ones = 0
    for i in range(1, n):
        left_ones += pairs[i - 1][1]
        if pairs[i][0] == pairs[i - 1][0]:
            continue
        correct = max(left_ones, i - left_ones) + max(
            ones - left_ones, (n - i) - (ones - left_ones)
        )
        best = max(best, correct / n)
    return best if best >= 0.99 else None


def _logit_coef(
    data: dict[str, list[float]], yv: Sequence[int], x: str, cols: Sequence[str]
) -> tuple[float, float]:
    """Coefficient of ``x`` and its Wald t from logistic regression (Newton-IRLS)."""
    n = len(yv)
    design = [[1.0] * n] + [data[c] for c in cols]
    p = len(design)
    beta = [0.0] * p
    last = None
    for _ in range(50):
        eta = [math.fsum(beta[j] * design[j][k] for j in range(p)) for k in range(n)]
        mu = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, e)))) for e in eta]
        w = [max(mu[k] * (1 - mu[k]), 1e-9) for k in range(n)]
        xtwx = [
            [
                math.fsum(design[i][k] * w[k] * design[j][k] for k in range(n))
                for j in range(p)
            ]
            for i in range(p)
        ]
        grad = [
            math.fsum(design[i][k] * (yv[k] - mu[k]) for k in range(n))
            for i in range(p)
        ]
        inv = _invert(xtwx)
        if inv is None:
            break
        step = [math.fsum(inv[i][j] * grad[j] for j in range(p)) for i in range(p)]
        beta = [beta[i] + step[i] for i in range(p)]
        if last is not None and max(abs(s) for s in step) < 1e-8:
            break
        last = beta
    inv = _invert(
        [
            [
                math.fsum(
                    design[i][k]
                    * max(_p(beta, design, k) * (1 - _p(beta, design, k)), 1e-9)
                    * design[j][k]
                    for k in range(n)
                )
                for j in range(p)
            ]
            for i in range(p)
        ]
    )
    se = math.sqrt(inv[1][1]) if inv and inv[1][1] > 0 else math.inf
    return beta[1], (beta[1] / se if se > 0 else 0.0)


def _p(beta: Sequence[float], design: Sequence[Sequence[float]], k: int) -> float:
    eta = math.fsum(beta[j] * design[j][k] for j in range(len(beta)))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, eta))))


def _invert(matrix: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan inverse of a small square matrix; None if singular."""
    nn = len(matrix)
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(nn)]
        for i, row in enumerate(matrix)
    ]
    for col in range(nn):
        pivot = max(range(col, nn), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(nn):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [row[nn:] for row in aug]
