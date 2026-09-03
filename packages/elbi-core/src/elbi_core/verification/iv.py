"""Instrumental-variables gate: is the instrument strong enough to trust?

An instrumental-variables estimate has two assumptions. The exclusion restriction (the
instrument affects the outcome only through the treatment) is untestable and must be
argued substantively: this gate states that. But instrument *strength* (relevance) is
testable: a weak instrument, one only loosely correlated with the treatment, makes the
2SLS estimate badly biased and its standard errors unreliable. The first-stage F
statistic measures it, and the Staiger-Stock rule of thumb flags F below 10 as weak.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import _frame, _ols
from ._report import Check, VerificationReport

#: First-stage F below this is a weak instrument (Staiger-Stock 1997).
_WEAK_F = 10.0


def verify_iv(
    rows: Sequence[dict[str, Any]], instrument: str, treatment: str, outcome: str
) -> VerificationReport:
    """Verify the ``instrument`` is strong enough for an IV estimate of ``treatment``.

    Runs the first-stage regression of ``treatment`` on ``instrument`` and reports the F
    statistic. The verdict is ``sound`` (first-stage F at or above 10, a strong
    instrument: though the exclusion restriction still must be argued), ``unsound`` (F
    below 10, a weak instrument, so the 2SLS estimate is biased and imprecise), or
    ``inconclusive`` (too few rows). ``outcome`` is named only to frame the design; its
    relation to the instrument is the untestable part.
    """
    data, n = _frame(rows, [instrument, treatment, outcome])
    if n < 30:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("an instrument-strength test needs at least 30 rows",),
        )
    _, t = _ols(data, treatment, instrument, [])
    f_stat = t * t  # single instrument: first-stage F is the squared t
    strong = f_stat >= _WEAK_F
    checks = (Check("instrument-strength", strong, f"first-stage F = {f_stat:.1f}"),)
    pivotal = (
        None
        if strong
        else (
            f"first-stage F = {f_stat:.1f} is below 10: '{instrument}' is a weak "
            f"instrument for '{treatment}', so the IV estimate is biased and imprecise"
        )
    )
    verdict = "sound" if strong else "unsound"
    caveats = (
        f"strength is testable and {'adequate' if strong else 'inadequate'}; the "
        f"exclusion restriction, that '{instrument}' affects '{outcome}' only through "
        f"'{treatment}', is untestable and must be argued substantively",
    )
    return VerificationReport(verdict, f_stat, True, checks, pivotal, caveats)
