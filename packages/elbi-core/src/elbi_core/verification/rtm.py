"""Regression-to-the-mean gate: a pre/post change conditioned on the baseline (ANCOVA).

When a group is chosen because its baseline was extreme (the lowest scorers, the sickest
patients, last quarter's worst branches) it moves toward the mean on its own when
measured again, with no treatment at all (Galton 1886). The principled way to tell that
artefact from a real effect is the analysis of covariance: regress the outcome on the
baseline and the group, so the group coefficient is the change *beyond* what the
baseline already predicts (Barnett 2005). This gate runs that adjusted effect through
the full causal-effect gate (inheriting its outlier and fragility checks, so a chance
coefficient does not slip through) and reframes the verdict as regression to the mean.
Conditioning on the baseline absorbs the reversion at any selection strength; a control
group is still what turns the surviving coefficient into a causal claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _mean_var
from ._report import Check, VerificationReport
from .effect import verify_effect


def verify_rtm(
    rows: Sequence[dict[str, Any]],
    before: str,
    after: str,
    group: str,
) -> VerificationReport:
    """Verify a group's ``before``->``after`` change survives conditioning on baseline.

    Runs the analysis of covariance ``after ~ before + group`` through the effect gate:
    the group coefficient is the change the baseline does not already explain, so
    regression to the mean (which lives entirely in the baseline) is absorbed however
    the group was selected, and the effect gate's robustness checks keep a spurious
    coefficient from certifying. The verdict is ``sound`` (a change beyond reversion
    that survives those checks), ``unsound`` (the naive gain is consistent with
    regression to the mean), or ``inconclusive`` (fewer than 40 rows, or no variation).
    """
    trip = [
        (bv, av, g)
        for r in rows
        if (bv := _as_float(r.get(before))) is not None
        and (av := _as_float(r.get(after))) is not None
        and (g := str(r.get(group, "")).strip())
    ]
    labels = sorted({g for _, _, g in trip})
    if len(labels) != 2 or len(trip) < 40:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("regression to the mean needs before, after and a two-level group",),
        )
    mu_b, var_b = _mean_var([b for b, _, _ in trip])
    if var_b == 0:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("no variation in the baseline",)
        )
    sd_b = var_b**0.5

    def before_mean(lab: str) -> float:
        vals = [b for b, _, g in trip if g == lab]
        return sum(vals) / len(vals)

    # the selected level (more extreme baseline) is encoded 1, so the coefficient reads
    # in the direction of the treated group; this labelling does not change the verdict
    selected = max(labels, key=lambda lab: abs(before_mean(lab) - mu_b))
    baseline_gap = (before_mean(selected) - mu_b) / sd_b
    side = "low" if baseline_gap < 0 else "high"

    # ANCOVA with robustness: the effect of group membership on the outcome, holding
    # the baseline fixed, checked by the full effect gate (significance, outliers, …)
    encoded = [
        {"__grp": 1.0 if g == selected else 0.0, "__before": b, "__after": a}
        for b, a, g in trip
    ]
    adjusted = verify_effect(encoded, "__grp", "__after", controls=["__before"])
    coef = adjusted.effect
    detail = (
        f"baseline-adjusted group effect {coef:+.2f} ({adjusted.verdict}); the group's "
        f"baseline sits {baseline_gap:+.1f} SD from the mean"
    )

    if adjusted.verdict == "sound":
        return VerificationReport(
            "sound",
            coef,
            True,
            (Check("beyond-regression-to-mean", True, detail),),
            None,
            (
                f"holding the baseline fixed (which absorbs regression to the mean), "
                f"the group effect is {coef:+.2f} and survives the effect gate's "
                "robustness checks: more than reversion, though causal only under a "
                "controlled design",
            ),
        )
    return VerificationReport(
        "unsound",
        coef,
        True,
        (Check("beyond-regression-to-mean", False, detail),),
        f"the {before}->{after} gain is consistent with regression to the mean: "
        f"holding the baseline fixed, the group effect does not stand as a sound "
        f"effect ({adjusted.verdict}): a group selected on {side} '{before}' reverts "
        "toward the mean on its own, so compare against a control group",
        (f"baseline-adjusted group coefficient {coef:+.2f}",),
    )
