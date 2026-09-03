"""Sensitivity gate: how robust an effect is to unmeasured confounding (E-value).

Whether an observed association is confounded by something unmeasured cannot be tested
from the data; it needs the causal graph. The taught response (VanderWeele & Ding 2017)
is not to test it but to quantify it: the E-value is the minimum strength of
association, on the risk-ratio scale, that an unmeasured confounder would need with both
the exposure and the outcome to fully explain away the observed effect. A large E-value
means only an implausibly strong hidden confounder could overturn the result; an E-value
near 1 means a weak one suffices. This gate reports the E-value for the estimate and for
the confidence limit nearest the null, so a claim ships with its robustness to the thing
that cannot be checked.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _corr
from ._report import Check, VerificationReport

#: An effect whose confidence-limit E-value clears this needs a non-trivial hidden
#: confounder (risk-ratio association above it on both arms) to explain away.
_ROBUST = 1.25


def verify_sensitivity(
    rows: Sequence[dict[str, Any]], x: str, y: str
) -> VerificationReport:
    """Report how robust the effect of ``x`` on ``y`` is to unmeasured confounding.

    Estimates the association, converts it to an approximate risk ratio, and computes
    the E-value for the point estimate and for the confidence limit nearest the null.
    The verdict is ``sound`` (the confidence-limit E-value clears 1.25, so only a
    non-trivial hidden confounder could explain the effect away), ``inconclusive`` (the
    effect is fragile to a weak confounder, or not significant), never ``unsound``,
    because this quantifies robustness rather than refuting the claim.
    """
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        xv, yv = _as_float(row.get(x)), _as_float(row.get(y))
        if xv is not None and yv is not None:
            xs.append(xv)
            ys.append(yv)
    n = len(xs)
    if n < 20:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("too few rows to assess sensitivity",),
        )
    r = _corr(xs, ys)
    if abs(r) >= 1.0:
        return VerificationReport(
            "inconclusive",
            r,
            False,
            (),
            None,
            ("a perfect correlation suggests an identity or leakage, not an effect",),
        )
    # Fisher-z confidence interval for the correlation; take the limit nearest 0
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    if lo <= 0 <= hi:
        return VerificationReport(
            "inconclusive",
            r,
            False,
            (),
            None,
            (f"the association (r = {r:.2f}) is not significant to stress-test",),
        )
    near_null = lo if abs(lo) < abs(hi) else hi

    e_point = _e_value(_corr_to_rr(r))
    e_bound = _e_value(_corr_to_rr(near_null))
    robust = e_bound >= _ROBUST
    checks = (
        Check(
            "confounding-robustness",
            robust,
            f"confidence-limit E-value = {e_bound:.2f}",
        ),
    )
    caveats = (
        f"E-value = {e_point:.2f} (estimate), {e_bound:.2f} (confidence limit): a "
        f"hidden confounder would need a risk-ratio link above {e_bound:.2f} with "
        "both x and y to explain the effect away",
    )
    verdict = "sound" if robust else "inconclusive"
    return VerificationReport(verdict, e_point, True, checks, None, caveats)


def _corr_to_rr(r: float) -> float:
    """Approximate the risk ratio implied by a correlation (VanderWeele & Ding 2017)."""
    r = min(max(r, -0.999), 0.999)
    d = 2.0 * r / math.sqrt(1.0 - r * r)  # correlation -> standardized mean difference
    return math.exp(0.91 * abs(d))  # d -> approximate risk ratio


def _e_value(rr: float) -> float:
    """The E-value for a risk ratio (symmetric in RR and 1/RR; >= 1)."""
    rr = max(rr, 1.0 / rr) if rr > 0 else 1.0
    if rr <= 1.0:
        return 1.0
    return rr + math.sqrt(rr * (rr - 1.0))
