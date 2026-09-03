"""Forecasting gate (skill versus a naive baseline)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float
from ._report import Check, VerificationReport


def verify_forecast(
    rows: Sequence[dict[str, Any]],
    time: str,
    actual: str,
    forecast: str,
    *,
    seasonal_period: int = 1,
) -> VerificationReport:
    """Verify that a forecast actually beats the naive baseline.

    A low-looking error is worthless if it does not beat the (seasonal-)naive forecast:
    copying the last value, or the value one season ago. This computes the mean absolute
    scaled error (MASE): the model's error divided by the in-sample naive error. The
    verdict is ``sound`` (MASE < 1, the model beats naive), ``inconclusive`` (MASE >= 1,
    no better than copying the last season), or ``inconclusive`` with a note when there
    is too little data.
    """
    triples = []
    for r in rows:
        t = _as_float(r.get(time))
        a = _as_float(r.get(actual))
        f = _as_float(r.get(forecast))
        if t is not None and a is not None and f is not None:
            triples.append((t, a, f))
    if len(triples) < max(2 * seasonal_period + 2, 20):
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("too few points to score a forecast",),
        )
    triples.sort()
    acts = [a for _, a, _ in triples]
    fcs = [f for _, _, f in triples]
    m = max(1, seasonal_period)
    naive_errors = [abs(acts[i] - acts[i - m]) for i in range(m, len(acts))]
    naive_mae = math.fsum(naive_errors) / len(naive_errors) if naive_errors else 0.0
    model_mae = math.fsum(abs(a - f) for a, f in zip(acts, fcs, strict=True)) / len(
        acts
    )
    if naive_mae <= 0:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("the series is constant; MASE is undefined",),
        )
    mase = model_mae / naive_mae
    held = mase < 1.0
    label = "seasonal-naive" if m > 1 else "naive (last value)"
    checks = [
        Check(
            "baseline",
            held,
            f"MASE = {mase:.2f} versus the {label} forecast"
            + ("" if held else ": no better than the naive baseline"),
        )
    ]
    if held:
        verdict, pivotal = "sound", None
    else:
        verdict, pivotal = (
            "inconclusive",
            (
                f"the forecast does not beat the {label} baseline "
                f"(MASE {mase:.2f} >= 1); "
                "it adds no value over copying the last season"
            ),
        )
    caveats = (
        "skill is measured on this holdout; confirm the split was temporal (no future "
        "leakage) and that the horizon matches the claim",
    )
    return VerificationReport(verdict, mase, held, tuple(checks), pivotal, caveats)
