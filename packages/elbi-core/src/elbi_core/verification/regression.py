"""OLS-coefficient gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _breusch_pagan,
    _coef_t,
    _fragility,
    _frame,
    _low_influence,
    _numeric_columns,
    _ols_full,
    _reset_t,
    _robust_t,
    _vif,
)
from ._report import (
    _MIN_KEEP,
    _T,
    _T_STRICT,
    Check,
    VerificationReport,
)


def verify_regression(
    rows: Sequence[dict[str, Any]],
    y: str,
    x: str,
    *,
    controls: Sequence[str] = (),
) -> VerificationReport:
    """Verify an OLS coefficient claim: the slope of ``x`` on ``y`` holding controls.

    Re-fits the regression and checks the assumptions whose violation invalidates
    the coefficient or its reported significance: multicollinearity (the variance
    inflation factor), heteroskedastic errors (Breusch-Pagan, re-tested under
    heteroskedasticity-robust standard errors), reliance on a few influential points
    (Cook's distance), and functional-form misspecification (a Ramsey RESET term).
    The verdict is ``sound`` (every check held), ``unsound`` (one broke it), or
    ``inconclusive`` (the coefficient is not significant, or too little data to fit).
    """
    # a non-numeric control cannot enter a linear fit; drop it rather than let it void
    # every row (the effect gate does the same), so the coefficient is still checked
    numeric = set(_numeric_columns(rows))
    cols = [x, *(c for c in controls if c in numeric)]
    data, n = _frame(rows, [y, *cols])
    cols = [c for c in cols if c in data]
    if n < 30 or y not in data or x not in data:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few numeric rows",)
        )
    p = len(cols) + 1
    if n < 10 * p:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (f"too few rows ({n}) to fit {p} parameters reliably",),
        )
    fit = _ols_full(data, y, cols)
    if fit is None:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("the design is singular (perfectly collinear predictors)",),
        )
    bx, tx = _coef_t(fit, 1)  # index 0 is the intercept, 1 is x
    if abs(tx) <= _T:
        return VerificationReport(
            "inconclusive",
            bx,
            False,
            (),
            None,
            (f"the coefficient of '{x}' is not significant",),
        )
    positive = bx > 0
    checks: list[Check] = []
    pivotal: str | None = None

    # multicollinearity matters only when it actually destabilizes the coefficient:
    # a precisely estimated coefficient can be trustworthy despite a high VIF, so the
    # test is whether the estimate survives resampling, with the VIF for context.
    others = [c for c in cols if c != x]
    vif = _vif(data, x, others)
    frag = _fragility(data, x, y, others, base_positive=positive)
    held = frag <= 0.30
    checks.append(
        Check(
            "stability",
            held,
            f"VIF = {vif:.1f}; stable under resampling ({frag:.0%} of subsets break it)"
            if held
            else f"VIF = {vif:.1f}; unstable ({frag:.0%} of subsets break it)",
        )
    )
    if not held:
        pivotal = "the coefficient is unstable under resampling" + (
            f" (collinearity, VIF = {vif:.1f})" if vif > 5 else ""
        )

    # heteroskedastic errors make the ordinary standard error wrong (usually too
    # small); require the coefficient to stay significant under robust errors.
    bp_p = _breusch_pagan(fit, len(cols))
    if bp_p < 0.05:
        t_rob = _robust_t(fit, 1)
        held = abs(t_rob) > _T
        checks.append(
            Check(
                "heteroskedasticity",
                held,
                f"heteroskedastic (BP p = {bp_p:.2g}); robust |t| = {abs(t_rob):.2f}"
                + ("" if held else ": significance not robust"),
            )
        )
        if not held:
            pivotal = pivotal or (
                "the significance disappears under heteroskedasticity-robust "
                "standard errors"
            )
    else:
        checks.append(
            Check("heteroskedasticity", True, f"homoskedastic (BP p = {bp_p:.2g})")
        )

    # a coefficient should not rest on a few high-leverage points
    keep = _low_influence(fit)
    if _MIN_KEEP * n <= sum(keep) < n:
        refit = _ols_full(data, y, cols, mask=keep)
        if refit is not None:
            b2, t2 = _coef_t(refit, 1)
            held = abs(t2) > _T and (b2 > 0) == positive
            checks.append(
                Check(
                    "influence",
                    held,
                    "survives dropping influential points"
                    if held
                    else "driven by a few influential points",
                )
            )
            if not held:
                pivotal = pivotal or "the coefficient is driven by influential points"

    # a significant power of the fitted values signals the linear form is wrong,
    # which biases the coefficient (Ramsey's regression specification error test).
    reset_t = _reset_t(data, y, cols, fit)
    held = abs(reset_t) <= _T_STRICT
    checks.append(
        Check(
            "specification",
            held,
            f"linear form adequate (RESET |t| = {abs(reset_t):.2f})"
            if held
            else f"nonlinear misspecification (RESET |t| = {abs(reset_t):.2f})",
        )
    )
    if not held:
        pivotal = pivotal or (
            "the linear specification is misspecified, biasing the coefficient: try a "
            "transform (e.g. log of a skewed variable) or a nonlinear term, then re-run"
        )

    # A failed specification check alone means the *linear* form is the wrong tool: the
    # linear coefficient cannot be certified, but that is not evidence against a
    # relationship (a real nonlinear effect fails RESET too), so it must not veto the
    # effect. Report it inconclusive, not unsound. Any other failure (an unstable,
    # heteroskedastic, or influence-driven coefficient) is a genuine unsound.
    hard_failure = any(not c.survived and c.name != "specification" for c in checks)
    spec_ok = all(c.survived for c in checks if c.name == "specification")
    if hard_failure:
        verdict = "unsound"
    elif not spec_ok:
        verdict = "inconclusive"
    else:
        verdict = "sound"
    caveats = (
        "the coefficient is associational unless an identification strategy holds",
    )
    return VerificationReport(verdict, bx, True, tuple(checks), pivotal, caveats)
