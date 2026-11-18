"""Pooled-effect (random-effects meta-analysis) gate.

Pooling several past studies, one effect and standard error each, is a different claim
from any one of them: the studies can disagree by more than their standard errors allow
(heterogeneity), and with few studies that spread is itself barely estimable. The
DerSimonian-Laird estimator is the classic method of moments for it, so the gate reports
it alongside what a statistician reviewing it asks for: the Hartung-Knapp interval (a t
on k-1 df with the variance rescaled to the studies' actual scatter, which keeps its
coverage with few studies where the DL Wald interval does not).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from ._numerics import _as_float, _chi2_sf, _p_two_sided, _t_inv
from ._report import Check, VerificationReport

#: Below this many studies tau² is not reliably estimable: it rests on a handful of
#: residuals, so DL often returns zero by chance and its Wald interval is too narrow.
_MIN_STUDIES = 5
#: I² at or above this is "considerable" heterogeneity (the Cochrane Handbook's 75-100%
#: band), where how the studies are weighted can decide the pooled conclusion.
_CONSIDERABLE_I2 = 0.75


@dataclass(frozen=True)
class _Fit:
    mu_fe: float
    se_fe: float
    q: float
    tau2: float
    i2: float
    mu: float
    se: float
    se_hk: float


def _dl(effects: Sequence[float], ses: Sequence[float]) -> _Fit:
    """Inverse-variance fixed effect, then DerSimonian-Laird random effects."""
    k = len(effects)
    w = [1.0 / s**2 for s in ses]
    sw = math.fsum(w)
    mu_fe = math.fsum(wi * e for wi, e in zip(w, effects, strict=True)) / sw
    q = math.fsum(wi * (e - mu_fe) ** 2 for wi, e in zip(w, effects, strict=True))
    c = sw - math.fsum(wi * wi for wi in w) / sw
    # a Q below its k-1 expectation means no detectable spread, not a negative variance
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    ws = [1.0 / (s**2 + tau2) for s in ses]
    sws = math.fsum(ws)
    mu = math.fsum(wi * e for wi, e in zip(ws, effects, strict=True)) / sws
    scatter = math.fsum(wi * (e - mu) ** 2 for wi, e in zip(ws, effects, strict=True))
    se_hk = math.sqrt(scatter / ((k - 1) * sws)) if k > 1 else math.inf
    return _Fit(mu_fe, math.sqrt(1 / sw), q, tau2, i2, mu, math.sqrt(1 / sws), se_hk)


def _sign(lo: float, hi: float) -> int:
    return 1 if lo > 0 else -1 if hi < 0 else 0


def _iv(lo: float, hi: float) -> str:
    return f"[{lo:+.3g}, {hi:+.3g}]"


def verify_pooled_effect(
    rows: Sequence[dict[str, Any]],
    effect: str,
    se: str,
    *,
    study: str | None = None,
    new_effect: float | None = None,
    new_se: float | None = None,
    alpha: float = 0.05,
) -> VerificationReport:
    """Verify an effect pooled across several studies before trusting it.

    Each row is one study: its ``effect`` and that effect's standard error ``se``,
    optionally labelled by ``study``. Intervals are all at level ``1 - alpha``. The
    checks then try to break the pooled claim: fewer than five studies (tau² not
    reliably estimable), considerable heterogeneity under which the fixed- and
    random-effects intervals disagree on sign, and one study whose omission flips the
    sign or leaves the Hartung-Knapp interval. Given ``new_effect`` and ``new_se``, the
    pooled estimate is also tested as a prior for that new study: a prior-data conflict
    means it should be down-weighted, not borrowed at full weight. The verdict is
    ``sound`` (every check survives), ``unsound`` (any fails), or ``inconclusive``
    (fewer than two studies, or a standard error that is not positive).
    """
    labels: list[str] = []
    effects: list[float] = []
    ses: list[float] = []
    for i, row in enumerate(rows):
        e, s = _as_float(row.get(effect)), _as_float(row.get(se))
        if e is None or s is None or not (math.isfinite(e) and math.isfinite(s)):
            continue
        name = str(row.get(study) or "").strip() if study else ""
        labels.append(name or f"study {i + 1}")
        effects.append(e)
        ses.append(s)

    def _inconclusive(why: str) -> VerificationReport:
        return VerificationReport("inconclusive", 0.0, False, (), None, (why,))

    k = len(effects)
    if k < 2:
        return _inconclusive(
            f"pooling needs two or more studies with '{effect}' and '{se}', found {k}"
        )
    if (new_effect is None) != (new_se is None):
        return _inconclusive("a new study needs both its effect and its standard error")
    if new_effect is not None and not math.isfinite(new_effect):
        return _inconclusive("a new study's effect must be a finite number")
    if any(not s > 0 for s in [*ses, *([] if new_se is None else [new_se])]):
        return _inconclusive(
            "every standard error must be positive to weight its study"
        )

    fit = _dl(effects, ses)
    level = f"{1 - alpha:.0%}"
    z = NormalDist().inv_cdf(1 - alpha / 2)
    ci = (fit.mu - z * fit.se, fit.mu + z * fit.se)
    fe_ci = (fit.mu_fe - z * fit.se_fe, fit.mu_fe + z * fit.se_fe)
    t_hk = _t_inv(alpha, k - 1)
    hk = (fit.mu - t_hk * fit.se_hk, fit.mu + t_hk * fit.se_hk)
    checks: list[Check] = []
    pivotal: str | None = None

    if k >= _MIN_STUDIES:
        checks.append(
            Check("enough-studies", True, f"{k} studies, enough to estimate tau²")
        )
    else:
        checks.append(
            Check(
                "enough-studies",
                False,
                f"tau² is not reliably estimable from {k} studies; the Hartung-Knapp "
                "interval is reported",
            )
        )
        pivotal = (
            f"too few studies: with {k}, the between-study spread is barely "
            f"estimable, so read the Hartung-Knapp interval {_iv(*hk)} and add "
            "studies before relying on the pooled estimate"
        )

    # Fixed effect trusts the most precise studies and random effects spreads weight to
    # the rest, so under considerable heterogeneity the weighting can flip the sign.
    contains_fe = ci[0] <= fit.mu_fe <= ci[1]
    disagree = _sign(*fe_ci) != _sign(*ci)
    if fit.i2 < _CONSIDERABLE_I2 or contains_fe or not disagree:
        if contains_fe:
            why = "lies inside the random-effects interval"
        elif not disagree:
            why = "agrees in sign with the random-effects interval"
        else:
            why = "differs, but the heterogeneity is not considerable"
        checks.append(
            Check(
                "heterogeneity-quantified",
                True,
                f"I² = {fit.i2:.0%}; fixed effect {fit.mu_fe:+.3g} {why}",
            )
        )
    else:
        checks.append(
            Check(
                "heterogeneity-quantified",
                False,
                f"I² = {fit.i2:.0%}; fixed effect {fit.mu_fe:+.3g} {_iv(*fe_ci)} and "
                f"random effects {fit.mu:+.3g} {_iv(*ci)} disagree on sign",
            )
        )
        pivotal = pivotal or (
            f"considerable heterogeneity (I² = {fit.i2:.0%}): the fixed- and "
            "random-effects estimates disagree on sign, so the pooled mean is not a "
            "usable summary; explain the spread (a moderator) instead of averaging it"
        )

    tol = 1e-9 * max(1.0, abs(fit.mu))
    t_drop = _t_inv(alpha, k - 2)
    drops = []
    for i in range(k):
        f = _dl(effects[:i] + effects[i + 1 :], ses[:i] + ses[i + 1 :])
        half_drop = t_drop * f.se_hk
        drops.append((labels[i], f.mu, _sign(f.mu - half_drop, f.mu + half_drop) != 0))
    definite = _sign(*hk) != 0

    # Crossing zero reverses the sign only when one side of it is significant: a
    # null pool wobbling from +0.001 to -0.001 is noise, but one study hiding a
    # significant effect of the other sign is not.
    def flips(m: float, drop_definite: bool) -> bool:
        return (definite or drop_definite) and (m > 0) != (fit.mu > 0)

    breaks = [
        (abs(m - fit.mu), name, m, d)
        for name, m, d in drops
        if flips(m, d) or not hk[0] - tol <= m <= hk[1] + tol
    ]
    if not breaks:
        lo, hi = min(m for _, m, _ in drops), max(m for _, m, _ in drops)
        same_sign = "the same sign and " if definite else ""
        checks.append(
            Check(
                "leave-one-out-stable",
                True,
                f"dropping any one study keeps the estimate in {_iv(lo, hi)}, "
                f"{same_sign}inside the Hartung-Knapp interval",
            )
        )
    else:
        _, name, m, d = max(breaks)
        how = (
            "reversing its sign"
            if flips(m, d)
            else f"outside the Hartung-Knapp interval {_iv(*hk)}"
        )
        others = f" ({len(breaks) - 1} others also do)" if len(breaks) > 1 else ""
        checks.append(
            Check(
                "leave-one-out-stable",
                False,
                f"dropping '{name}' moves the estimate to {m:+.3g}, {how}{others}",
            )
        )
        pivotal = pivotal or (
            f"one study carries the result: without '{name}' the pooled estimate "
            f"is {m:+.3g} rather than {fit.mu:+.3g}; investigate that study and "
            "report the estimate both ways"
        )

    if new_effect is not None and new_se is not None:
        # the new study's distance from the pooled predictive: its own sampling error,
        # the between-study spread, and the uncertainty in the pooled mean
        zc = (new_effect - fit.mu) / math.sqrt(new_se**2 + fit.tau2 + fit.se**2)
        p = _p_two_sided(zc)
        if p >= alpha:
            checks.append(
                Check(
                    "no-prior-data-conflict",
                    True,
                    f"the new study's {new_effect:+.3g} (se {new_se:.3g}) is "
                    f"consistent with the pooled prior (z = {zc:+.2f}, p = {p:.2g})",
                )
            )
        else:
            checks.append(
                Check(
                    "no-prior-data-conflict",
                    False,
                    f"the new study's {new_effect:+.3g} (se {new_se:.3g}) conflicts "
                    f"with the pooled prior (z = {zc:+.2f}, p = {p:.2g}); down-weight "
                    "the prior (a robust MAP mixture) rather than borrow it at full "
                    "weight",
                )
            )
            pivotal = pivotal or (
                f"prior-data conflict (p = {p:.2g}): the new study does not resemble "
                "the pooled ones, so borrowing their estimate at full weight would "
                "bias it; down-weight the prior (a robust MAP mixture with a vague "
                "component)"
            )

    # Higgins, Thompson & Spiegelhalter (2009): t on k-2 df, since both the mean and
    # tau² are estimated from the k studies
    if k >= 3:
        half = _t_inv(alpha, k - 2) * math.sqrt(fit.tau2 + fit.se**2)
        pi = (
            f"{level} prediction interval for a new study's true effect "
            f"{_iv(fit.mu - half, fit.mu + half)} (t on {k - 2} df): the range to "
            "expect in a new setting, not the CI"
        )
    else:
        pi = "a prediction interval for a new study needs at least three studies"
    caveats = [
        f"random-effects (DerSimonian-Laird) pooled '{effect}' = {fit.mu:+.3g} across "
        f"{k} studies, {level} CI {_iv(*ci)}, Hartung-Knapp {_iv(*hk)} (t on {k - 1} "
        f"df); tau² = {fit.tau2:.3g}, I² = {fit.i2:.0%}",
        f"fixed-effect estimate {fit.mu_fe:+.3g} {_iv(*fe_ci)}; Cochran's Q = "
        f"{fit.q:.2f} on {k - 1} df (p = {_chi2_sf(fit.q, k - 1):.2g})",
        pi,
        "DerSimonian-Laird underestimates tau² when studies are few, narrowing its "
        "interval; REML or Paule-Mandel estimate tau² better, and the Hartung-Knapp "
        "interval keeps its coverage",
    ]
    if len(rows) > k:
        caveats.append(
            f"{len(rows) - k} rows without a numeric '{effect}' and '{se}' were "
            "left out"
        )
    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    significant = _sign(*hk) != 0
    return VerificationReport(
        verdict, fit.mu, significant, tuple(checks), pivotal, tuple(caveats)
    )
