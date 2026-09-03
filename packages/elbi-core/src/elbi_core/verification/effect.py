"""Causal-effect soundness gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _autocorr1,
    _fragility,
    _frame,
    _iqr_mask,
    _is_unshielded_collider,
    _numeric_columns,
    _ols,
    _role,
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
    Every other column with numeric, complete values is treated as a candidate
    confounder or collider to test against.
    """
    pool = [c for c in _numeric_columns(rows) if c not in (x, y, *negative_controls)]
    data, n = _frame(rows, [x, y, *pool, *negative_controls])
    if n < 30 or x not in data or y not in data:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few numeric rows",)
        )

    pool = [c for c in pool if c in data]
    controls = [c for c in controls if c in data]
    base_b, base_t = _ols(data, y, x, controls)
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

    # an uncontrolled confounder that, held fixed, overturns the effect
    for z in pool:
        if z in controls or _role(data, x, y, z) != "confounder":
            continue
        if not holds(_ols(data, y, x, [*controls, z])):
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
    contingent: str | None = None
    for z in controls:
        if holds(_ols(data, y, x, [c for c in controls if c != z])):
            continue
        if _is_unshielded_collider(data, x, y, z):
            pivotal = (
                pivotal or f"the effect only appears when controlling collider '{z}'"
            )
            checks.append(Check("collider-control", False, f"'{z}' is a collider"))
            break
        contingent = (
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
        survived = holds(_ols(data, y, x, controls, mask=keep))
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
    rate = _fragility(data, x, y, controls, base_positive=base_b > 0)
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

        bp = _causal.best_practice_effect(data, x, y, controls)
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
    )
    return VerificationReport(verdict, estimate, True, tuple(checks), pivotal, caveats)
