"""Difference-in-differences gate: do the pre-treatment trends look parallel?

A difference-in-differences estimate is only causal if, absent treatment, the treated
and control groups would have moved in parallel: the parallel-trends assumption. That
counterfactual is untestable, but it has a testable implication: the two groups should
already have been trending together *before* treatment. This gate runs the standard
pre-trends falsification (regressing the pre-period outcome on the group-by-time
interaction and testing it) and flags a significant differential pre-trend. As Roth
(2022) stresses, passing this does not *prove* parallel trends (the test has low power),
so a pass is reported as "not contradicted", never as proof.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _ols, _t_sf
from ._report import Check, VerificationReport


def verify_did(
    rows: Sequence[dict[str, Any]],
    group: str,
    time: str,
    value: str,
    treatment_start: float,
) -> VerificationReport:
    """Verify the pre-treatment trends are parallel (a DiD parallel-trends check).

    Restricts to the rows before ``treatment_start`` and tests the group-by-time
    interaction on ``value``. The verdict is ``unsound`` (a significant differential
    pre-trend: the groups were already diverging before treatment, so the parallel-
    trends assumption is violated and the DiD estimate is biased), ``sound`` (no
    differential pre-trend, parallel trends are *not contradicted*, not proven), or
    ``inconclusive`` (too few pre-treatment periods or not two groups).
    """
    labels = sorted(
        {str(r.get(group, "")).strip() for r in rows if str(r.get(group, "")).strip()}
    )
    if len(labels) != 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("a DiD pre-trends check needs exactly two groups",),
        )
    a, b = labels
    cols: dict[str, list[float]] = {"value": [], "g": [], "t": [], "gt": []}
    for row in rows:
        lab = str(row.get(group, "")).strip()
        tv = _as_float(row.get(time))
        yv = _as_float(row.get(value))
        if lab not in (a, b) or tv is None or yv is None or tv >= treatment_start:
            continue
        g = 1.0 if lab == b else 0.0
        cols["value"].append(yv)
        cols["g"].append(g)
        cols["t"].append(tv)
        cols["gt"].append(g * tv)
    n = len(cols["value"])
    pre_periods = len(set(cols["t"]))
    if n < 30 or pre_periods < 3:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("need at least 30 rows over 3+ pre-treatment periods",),
        )
    # the group-by-time interaction is the differential pre-trend
    coef, t = _ols(cols, "value", "gt", ["g", "t"])
    df = n - 4
    p = _t_sf(t, df)
    parallel = p >= 0.05
    checks = (
        Check(
            "parallel-pre-trends",
            parallel,
            f"differential pre-trend slope = {coef:+.3g} (p = {p:.2g})",
        ),
    )
    pivotal = (
        None
        if parallel
        else (
            f"the groups '{a}' and '{b}' were already diverging before treatment "
            f"(differential pre-trend p = {p:.2g}): parallel trends fail, so the "
            "difference-in-differences estimate is biased"
        )
    )
    verdict = "sound" if parallel else "unsound"
    caveats = (
        (
            "parallel pre-trends are not contradicted, but this does not prove "
            "post-period parallel trends (the test has low power; Roth 2022)",
        )
        if parallel
        else ()
    )
    return VerificationReport(verdict, coef, True, checks, pivotal, caveats)
