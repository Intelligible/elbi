"""Causal-effect soundness gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _autocorr1,
    _design,
    _fragility,
    _frame,
    _independent,
    _iqr_mask,
    _is_unshielded_collider,
    _ols,
    _role,
    _spanned,
    _untested_caveats,
)
from ._report import (
    _MIN_KEEP,
    _RANDOM_WALK,
    _T,
    Check,
    VerificationReport,
)


def verify_effect(
    rows: Sequence[dict[str, Any]],
    x: str,
    y: str,
    *,
    controls: Sequence[str] = (),
    negative_controls: Sequence[str] = (),
) -> VerificationReport:
    """Verify that the effect of ``x`` on ``y`` (holding ``controls``) is sound.

    ``negative_controls`` name variables ``x`` cannot plausibly cause but that share
    any confounder; an association between ``x`` and one exposes latent confounding.
    Every other numeric column, and every text column with 2 to 10 values each on at
    least 10 rows, is a candidate confounder or collider to test against. A text
    control is held fixed as 0/1 columns, a control that cannot be used is named in an
    ``inconclusive`` report, never left out.
    """
    d = _design(rows, x, y, controls, exclude=negative_controls)
    unusable = d.refusal or next(
        (
            f"negative control '{w}' is not a numeric column"
            for w in negative_controls
            if w not in d.groups
        ),
        None,
    )
    if unusable:
        return VerificationReport("inconclusive", 0.0, False, (), None, (unusable,))
    held = [c for z in controls for c in d.groups[z]]
    tested = [c for z in d.candidates for c in d.groups[z]]
    data, n = _frame(d.rows, [x, y, *tested, *held, *negative_controls])
    if n < 30 or x not in data or y not in data:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few numeric rows",)
        )
    held = _independent(data, held)
    if held and _spanned(data, x, held):
        named = ", ".join(f"'{c}'" for c in controls)
        msg = f"'{x}' is determined by {named}, so its effect cannot be separated"
        return VerificationReport("inconclusive", 0.0, False, (), None, (msg,))

    base_b, base_t = _ols(data, y, x, held)
    if abs(base_t) <= _T:
        return VerificationReport(
            "inconclusive",
            base_b,
            False,
            (),
            None,
            ("the claimed effect is not significant under this specification",),
        )

    # The claim is that x has this signed effect on y; a perturbation breaks it by
    # turning the coefficient non-significant or flipping its sign.
    def holds(coef_t: tuple[float, float]) -> bool:
        b, t = coef_t
        return abs(t) > _T and (b > 0) == (base_b > 0)

    checks: list[Check] = []
    pivotal: str | None = None
    contingent: str | None = None
    copies: list[str] = []

    # an uncontrolled confounder that, held fixed, overturns the effect
    for z in d.candidates:
        zc = d.groups[z]
        fixed = _independent(data, [*held, *zc])
        if len(fixed) == len(held):
            continue
        pinned = next((v for v in (x, y) if _spanned(data, v, fixed)), None)
        if pinned is not None:
            if all(_spanned(data, c, [*held, pinned]) for c in zc):
                copies.append(z)
                continue
            contingent = contingent or (
                f"'{pinned}' is determined by '{z}', so the effect cannot be "
                f"separated from '{z}'"
            )
            continue
        if _role(data, x, y, zc) != "confounder":
            continue
        if not holds(_ols(data, y, x, fixed)):
            # controlling z overturns the effect: z is a confounder (the effect is
            # spurious) or a mediator on the x->y path (the effect is real but
            # indirect). The data cannot tell them apart, so the simple effect is not
            # certified either way; the distinction needs causal-structure knowledge.
            pivotal = (
                f"controlling for '{z}' overturns the effect: '{z}' is a confounder "
                f"(spurious) or a mediator on the x->y path (real but indirect)"
            )
            checks.append(
                Check("confounding", False, f"'{z}' is a confounder or a mediator")
            )
            break
    else:
        checks.append(Check("confounding", True, "no observed confounder overturns it"))

    # a control the effect hinges on: dropping it overturns the coefficient. If that
    # control is an identifiable collider, controlling it manufactured the effect: a
    # data artifact, so unsound. Otherwise the control is shielded and its causal role
    # is not identifiable from the data (a confounder or suppressor revealing a real
    # effect and a collider or mediator fabricating one look identical), so the effect
    # is real-or-artifact on an assumption the data cannot settle: not certifiable.
    for z in controls:
        rest = [c for w in controls if w != z for c in d.groups[w]]
        if holds(_ols(data, y, x, _independent(data, rest))):
            continue
        if _is_unshielded_collider(data, x, y, d.groups[z]):
            pivotal = (
                pivotal or f"the effect only appears when controlling collider '{z}'"
            )
            checks.append(Check("collider-control", False, f"'{z}' is a collider"))
            break
        contingent = contingent or (
            f"the effect reverses or vanishes when '{z}' is not controlled; whether "
            f"adjusting for '{z}' is valid turns on its causal role (a pre-{x} common "
            f"cause to control for, or a collider/mediator not to), which the data "
            "alone cannot determine"
        )
        break
    else:
        checks.append(
            Check("collider-control", True, "the effect survives dropping each control")
        )

    # reliance on a few extreme points: only continuous columns have IQR outliers; a
    # binary or few-valued predictor (e.g. a 0/1 treatment) does not, and masking it
    # would drop an entire level rather than an outlier
    continuous = [c for c in (x, y) if len(set(data[c])) > 10]
    keep = _iqr_mask(data, continuous) if continuous else [True] * n
    if sum(keep) >= _MIN_KEEP * n:
        survived = holds(_ols(data, y, x, held, mask=keep))
        checks.append(
            Check(
                "outliers",
                survived,
                "survives IQR outlier removal"
                if survived
                else "effect vanishes without outliers",
            )
        )
        if not survived:
            pivotal = pivotal or "the effect is driven by outliers"

    # signal vs noise: how often random half-size subsets break the effect
    rate = _fragility(data, x, y, held, base_positive=base_b > 0)
    survived = rate <= 0.30
    checks.append(
        Check(
            "fragility",
            survived,
            f"stable under resampling ({rate:.0%} of subsets break it)"
            if survived
            else f"fragile ({rate:.0%} of subsets break it)",
        )
    )
    if not survived:
        pivotal = pivotal or "the effect is fragile to resampling (likely noise)"

    # spurious regression: a coefficient between two non-stationary (random-walk-like)
    # series can come from shared drift alone; require it to survive differencing
    if _autocorr1(data[x]) > _RANDOM_WALK and _autocorr1(data[y]) > _RANDOM_WALK:
        diff = {
            "_dx": [data[x][i] - data[x][i - 1] for i in range(1, n)],
            "_dy": [data[y][i] - data[y][i - 1] for i in range(1, n)],
        }
        db, dt = _ols(diff, "_dy", "_dx", ())
        survived = abs(dt) > _T and (db > 0) == (base_b > 0)
        checks.append(
            Check(
                "stationarity",
                survived,
                "the effect survives differencing (not a shared trend)"
                if survived
                else "spurious: x and y are non-stationary and the effect vanishes "
                "after differencing",
            )
        )
        if not survived:
            pivotal = pivotal or (
                "spurious regression: x and y are non-stationary and the effect "
                "disappears once differenced"
            )

    # latent confounding: x predicts a variable it cannot cause
    for w in negative_controls:
        if w not in data:
            continue
        if abs(_ols(data, w, x, ())[1]) > _T:
            pivotal = pivotal or (
                f"x is associated with negative control '{w}' it cannot cause"
                ": unmeasured confounding"
            )
            checks.append(
                Check(
                    "negative-control",
                    False,
                    f"x predicts negative control '{w}' (latent confounding)",
                )
            )
            break
    else:
        if negative_controls:
            checks.append(
                Check(
                    "negative-control",
                    True,
                    "x is independent of the negative controls",
                )
            )

    if not all(c.survived for c in checks):
        verdict = "unsound"
    elif contingent is not None:
        # every robustness check passed, but the effect stands only on a control whose
        # validity the data cannot establish: sound if the assumption holds, artifact if
        # not; the oracle reports what it can prove, which here is neither.
        verdict = "inconclusive"
        pivotal = pivotal or contingent
    else:
        verdict = "sound"
    # The verdict and every check above stand on the linear specification; the
    # reported magnitude, though, uses the estimator suited to the treatment and
    # covariate overlap (doubly-robust or double-ML), falling back to the linear
    # coefficient when that is unavailable. Deterministic, so the certified
    # estimate stays reproducible.
    estimate = base_b
    method_caveat: tuple[str, ...] = ()
    try:
        from . import _causal

        bp = _causal.best_practice_effect(data, x, y, held)
    except ImportError:
        bp = None
    if bp is not None and (bp["estimate"] > 0) == (base_b > 0):
        estimate = bp["estimate"]
        overlap = f", overlap {bp['overlap']:.2f}" if bp["overlap"] is not None else ""
        method_caveat = (
            f"estimate is the {bp['method']} adjustment{overlap}; "
            f"the linear coefficient was {base_b:+.3g}",
        )
    caveats = (
        "causal direction (x→y vs y→x) is not verified from data; "
        "confirm with domain knowledge",
        "soundness is conditional on no unmeasured confounder beyond those tested",
        *method_caveat,
        *_untested_caveats(copies, d.untested, x, y),
    )
    return VerificationReport(verdict, estimate, True, tuple(checks), pivotal, caveats)
