"""Trend-over-time gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _as_float,
    _autocorr1,
    _iqr_mask,
    _ols,
    _residual_on,
    _t_sf,
)
from ._report import (
    _MIN_KEEP,
    Check,
    VerificationReport,
)

#: Periods a trend needs. A point here is a *period* -- a cohort, a month -- not an
#: observation, so the 30-row floor the row-level gates use would refuse two and a half
#: years of monthly cohorts. Ten is where the trend literature puts the recommended
#: minimum (Mann-Kendall runs from four and is advised from eight to ten, which is also
#: where its normal approximation takes over), and the autocorrelation, endpoint and
#: outlier checks below still have to pass, so the bar is more than significance.
_MIN_PERIODS = 10


def verify_trend(
    rows: Sequence[dict[str, Any]], time: str, value: str
) -> VerificationReport:
    """Verify a real trend in ``value`` over ``time`` (not noise or an artifact).

    Fits a linear trend and checks it is significant, not inflated by autocorrelated
    residuals (which overstate significance), not an artifact of the first/last
    points, and not driven by outliers.

    Significance is a Student-t tail on the fitted slope's own degrees of freedom, not a
    fixed 1.96: at ten periods that critical value is 2.31, and after trimming endpoints
    2.45, so a normal approximation would call trends significant that are not.
    """
    pts: list[tuple[float, float]] = []
    for row in rows:
        tv, vv = _as_float(row.get(time)), _as_float(row.get(value))
        if tv is not None and vv is not None:
            pts.append((tv, vv))
    if len(pts) < _MIN_PERIODS:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (f"too few points for a trend: {len(pts)}, and {_MIN_PERIODS} are needed",),
        )
    pts.sort()
    n = len(pts)
    data = {"t": [p[0] for p in pts], "v": [p[1] for p in pts]}
    slope, t = _ols(data, "v", "t", ())
    if _t_sf(t, n - 2) >= 0.05:
        return VerificationReport(
            "inconclusive",
            slope,
            False,
            (),
            None,
            (f"no significant trend in '{value}' over '{time}'",),
        )
    positive = slope > 0
    checks: list[Check] = []
    pivotal: str | None = None

    ac = _autocorr1(_residual_on(data, "v", "t"))
    held = abs(ac) < 0.5
    checks.append(
        Check(
            "autocorrelation",
            held,
            f"residual lag-1 autocorrelation = {ac:+.2f}"
            + ("" if held else " (significance overstated)"),
        )
    )
    if not held:
        pivotal = "the trend's significance is inflated by autocorrelated residuals"

    k = max(1, n // 10)
    trimmed = {"t": data["t"][k : n - k], "v": data["v"][k : n - k]}
    s2, t2 = _ols(trimmed, "v", "t", ())
    held2 = _t_sf(t2, len(trimmed["t"]) - 2) < 0.05 and (s2 > 0) == positive
    checks.append(
        Check(
            "endpoints",
            held2,
            "holds after trimming endpoints"
            if held2
            else "driven by the first/last points",
        )
    )
    if not held2:
        pivotal = pivotal or "the trend is an artifact of the endpoints"

    keep = _iqr_mask(data, ("v",))
    if sum(keep) >= _MIN_KEEP * n:
        s3, t3 = _ols(data, "v", "t", (), mask=keep)
        held3 = _t_sf(t3, sum(keep) - 2) < 0.05 and (s3 > 0) == positive
        checks.append(
            Check(
                "outliers",
                held3,
                "survives IQR outlier removal" if held3 else "driven by outliers",
            )
        )
        if not held3:
            pivotal = pivotal or "the trend is driven by outliers"

    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    caveats = ("a trend over time is description, not a forecast or a cause",)
    return VerificationReport(verdict, slope, True, tuple(checks), pivotal, caveats)
