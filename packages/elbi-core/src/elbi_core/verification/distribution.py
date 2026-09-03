"""Goodness-of-fit gate: whether a column can be treated as normally distributed.

Parametric tests, control limits, and many models assume a column is normal; using them
on a skewed or heavy-tailed column gives wrong p-values and intervals. This gate runs
two complementary normality tests, D'Agostino-Pearson K² (an omnibus on skewness and
kurtosis, with a clean p-value) and Anderson-Darling (the most powerful tail- sensitive
EDF test), and treats the assumption as sound only when neither rejects it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _mean_var, _normal_p
from ._report import Check, VerificationReport

#: Anderson-Darling critical value for normality at alpha = 0.05 (Stephens 1974).
_AD_CRIT = 0.752


def verify_normality(rows: Sequence[dict[str, Any]], column: str) -> VerificationReport:
    """Verify that ``column`` can be treated as normally distributed.

    Runs D'Agostino-Pearson K² and Anderson-Darling. The verdict is ``sound`` (neither
    test rejects normality, so parametric methods are safe), ``unsound`` (a test rejects
    it; the column is skewed or heavy-tailed, so transform it or use a rank- based
    method), or ``inconclusive`` (fewer than 20 values). Skewness and kurtosis are
    reported either way.
    """
    xs = [v for row in rows if (v := _as_float(row.get(column))) is not None]
    if len(xs) < 20:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("a normality test needs at least 20 values",),
        )
    p_k2 = _normal_p(xs)
    a2 = _anderson_darling(xs)
    checks: list[Check] = []
    pivotal: str | None = None
    if p_k2 is not None:
        held = p_k2 >= 0.05
        checks.append(Check("dagostino", held, f"D'Agostino-Pearson p = {p_k2:.3g}"))
        if not held:
            pivotal = "skewness/kurtosis depart from normal (D'Agostino-Pearson)"
    ad_held = a2 < _AD_CRIT
    checks.append(Check("anderson-darling", ad_held, f"A² = {a2:.2f} (reject > 0.75)"))
    if not ad_held and pivotal is None:
        pivotal = "the distribution's tails depart from normal (Anderson-Darling)"

    m, var = _mean_var(xs)
    sd = math.sqrt(var)
    skew = (math.fsum(((x - m) / sd) ** 3 for x in xs) / len(xs)) if sd > 0 else 0.0
    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    if pivotal is not None:
        pivotal += "; transform the column or use a rank-based method"
    return VerificationReport(
        verdict,
        skew,
        True,
        tuple(checks),
        pivotal,
        (f"skewness = {skew:+.2f}",),
    )


def _anderson_darling(xs: Sequence[float]) -> float:
    """Anderson-Darling A²* statistic for normality (Stephens small-sample form)."""
    n = len(xs)
    m, var = _mean_var(xs)
    sd = math.sqrt(var)
    if sd <= 0:
        return 0.0
    z = sorted((x - m) / sd for x in xs)
    eps = 1e-12

    def phi(v: float) -> float:
        return min(max(0.5 * math.erfc(-v / math.sqrt(2.0)), eps), 1.0 - eps)

    s = math.fsum(
        (2 * (i + 1) - 1) * (math.log(phi(z[i])) + math.log(1.0 - phi(z[n - 1 - i])))
        for i in range(n)
    )
    a2 = -n - s / n
    return a2 * (1.0 + 4.0 / n - 25.0 / (n * n))
