"""A second, independent estimate of an average effect, combined with the agent's own.

An agent that estimates an effect alone commits to one specification and reports its
sampling uncertainty only, so a wrong modelling choice yields a confident wrong answer.
elbi re-estimates the same effect from the rows with several outcome models and a
doubly robust estimator, then combines that analysis with the agent's, equally
believed. Where the two agree the combined answer is as sharp as either; where they
disagree it moves between them (a point) or spans them (an interval). Nothing is fitted
to make the combination work: the weights are equal by construction.

- ``estimate_effect``: per-model g-computation estimates with bootstrap standard
  errors, plus a cross-fitted AIPW estimate when the treatment takes two values. The
  estimand is the average effect over everyone (``ate``), the treated (``att``,
  rows at ``x1``), or the untreated (``atc``, rows at ``x0``).
- ``combine_interval``: an interval answer. The agent's interval and each elbi model
  become normal distributions; the agent holds half the weight and elbi's models share
  the other half; the answer is the central ``level`` interval of that mixture. Under
  a proper interval score the best interval is the central quantiles of one's belief,
  so a disagreement widens the answer exactly as far as the belief warrants.
- ``combine_point``: a point answer. The equal-weight mean of the agent's estimate
  and elbi's doubly robust estimate.

Both combinations are for a question that asks for the effect itself. When a question
asks for the output of a named procedure, the procedure defines the answer and
combining would move it off target; check it instead.

Models: additive splines on quantile ranks (with treatment and covariates fitted
separately per arm for a two-valued treatment, so the penalty never shrinks the effect
being estimated), a linear model, and gradient boosting. Binary outcomes use the
classifier counterparts, cross-validated on log-loss. Everything is seeded, so an
opinion is a pure function of the rows.

scikit-learn is an optional dependency (the ``ml`` extra), as for ``_causal``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

from ._numerics import _frame

Floats: TypeAlias = NDArray[np.float64]
Estimand: TypeAlias = Literal["ate", "att", "atc"]

MODELS = ("spline", "linear", "boosted")
DOUBLY_ROBUST = "doubly_robust"
_MIN_ROWS = 20
_MIN_ARM = 5
_PROPENSITY_CLIP = 0.01


@dataclass(frozen=True)
class EffectOpinion:
    """elbi's estimate of one average effect.

    ``estimates`` and ``standard_errors`` are keyed by model name (``MODELS``, plus
    ``DOUBLY_ROBUST`` for a two-valued treatment). ``caveats`` say what could not be
    estimated and why; an opinion with no estimates is one elbi declines to give.
    """

    x: str
    y: str
    adjust: tuple[str, ...]
    x0: float
    x1: float
    estimand: Estimand
    n: int
    estimates: dict[str, float] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()

    @property
    def doubly_robust(self) -> float | None:
        """The AIPW estimate, when the treatment takes two values."""
        return self.estimates.get(DOUBLY_ROBUST)

    def range(self, width: float = 2.0) -> tuple[float, float] | None:
        """The span of every estimate +/- ``width`` standard errors.

        An answer outside it disagrees with every elbi model by more than its sampling
        error: worth a second look before it is reported.
        """
        keys = [k for k in self.estimates if math.isfinite(self.estimates[k])]
        if not keys:
            return None
        lo = min(self.estimates[k] - width * self.standard_errors[k] for k in keys)
        hi = max(self.estimates[k] + width * self.standard_errors[k] for k in keys)
        return lo, hi


def _binary(v: Floats) -> bool:
    return set(np.unique(v).tolist()) <= {0.0, 1.0}


def _model(kind: str, binary: bool, n: int) -> Any:
    if kind == "spline":
        basis = make_pipeline(
            QuantileTransformer(n_quantiles=min(1000, n)),
            SplineTransformer(n_knots=8, degree=3, extrapolation="linear"),
        )
        head: Any = (
            LogisticRegressionCV(Cs=6, max_iter=3000, scoring="neg_log_loss")
            if binary
            else RidgeCV(alphas=np.logspace(-3, 3, 13))
        )
        return make_pipeline(basis, head)
    if kind == "linear":
        return LogisticRegression(max_iter=3000) if binary else LinearRegression()
    if binary:
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)
    return HistGradientBoostingRegressor(max_iter=200, random_state=0)


def _fit_predict(kind: str, X: Floats, y: Floats, new: Floats) -> Floats:
    """Expected outcome at ``new`` from a model of ``y`` on ``X``.

    A constant outcome needs no model (and a classifier cannot fit one class).
    """
    if np.unique(y).size < 2:
        return np.full(len(new), float(y[0]) if len(y) else math.nan)
    binary = _binary(y)
    m = _model(kind, binary, len(X)).fit(X, y)
    out = m.predict_proba(new)[:, 1] if binary else m.predict(new)
    return np.asarray(out, dtype=np.float64)


def _population(
    t: Floats, x0: float, x1: float, estimand: Estimand
) -> NDArray[np.bool_]:
    if estimand == "ate":
        return np.ones(len(t), dtype=bool)
    return np.isclose(t, x1 if estimand == "att" else x0)


def _g_effect(
    kind: str,
    T: Floats,
    Z: Floats,
    Y: Floats,
    x0: float,
    x1: float,
    estimand: Estimand,
    two_valued: bool,
) -> float:
    """g-computation: mean of E[Y | x1, Z] - E[Y | x0, Z] over the estimand's rows."""
    pop = _population(T, x0, x1, estimand)
    if not pop.any():
        return math.nan
    if kind == "spline" and two_valued and Z.shape[1] > 0:
        arm = np.isclose(T, x1)
        if arm.sum() < _MIN_ARM or (~arm).sum() < _MIN_ARM:
            return math.nan
        f1 = _fit_predict(kind, Z[arm], Y[arm], Z[pop])
        f0 = _fit_predict(kind, Z[~arm], Y[~arm], Z[pop])
        return float(np.mean(f1 - f0))
    X = np.column_stack([T, Z])
    P = X[pop]
    X1, X0 = P.copy(), P.copy()
    X1[:, 0], X0[:, 0] = x1, x0
    both = _fit_predict(kind, X, Y, np.vstack([X1, X0]))
    return float(np.mean(both[: len(P)] - both[len(P) :]))


def aipw_effect(
    T: Floats,
    Z: Floats,
    Y: Floats,
    estimand: Estimand = "ate",
    folds: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Cross-fitted AIPW estimate and influence-function standard error.

    ``T`` is 0/1. Outcome models per arm are the mean of elbi's model kinds, so no
    choice among them is made; the propensity is a cross-validated logistic model,
    clipped away from 0 and 1.
    """
    n = len(Y)
    m1, m0, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Z):
        for arm, out in ((1.0, m1), (0.0, m0)):
            s = tr[T[tr] == arm]
            out[te] = np.mean(
                [_fit_predict(k, Z[s], Y[s], Z[te]) for k in MODELS], axis=0
            )
        ps = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=6, max_iter=3000, scoring="neg_log_loss"),
        )
        e[te] = ps.fit(Z[tr], T[tr]).predict_proba(Z[te])[:, 1]
    e = np.clip(e, _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP)
    if estimand == "ate":
        psi = m1 - m0 + T * (Y - m1) / e - (1 - T) * (Y - m0) / (1 - e)
        tau = float(psi.mean())
        phi = psi - tau
    elif estimand == "att":
        num = T * (Y - m0) - (1 - T) * e / (1 - e) * (Y - m0)
        tau = float(num.sum() / T.sum())
        phi = (num - T * tau) / T.mean()
    else:
        num = (1 - T) * (m1 - Y) + T * (1 - e) / e * (Y - m1)
        tau = float(num.sum() / (1 - T).sum())
        phi = (num - (1 - T) * tau) / (1 - T).mean()
    return tau, float(np.std(phi, ddof=1) / math.sqrt(n))


def estimate_effect(
    rows: Sequence[dict[str, Any]],
    x: str,
    y: str,
    adjust: Sequence[str] = (),
    *,
    x0: float = 0.0,
    x1: float = 1.0,
    estimand: Estimand = "ate",
    bootstrap: int = 40,
    seed: int = 0,
) -> EffectOpinion:
    """Independently estimate the effect on ``y`` of moving ``x`` from ``x0`` to ``x1``.

    ``adjust`` is the adjustment set the caller's causal reasoning chose; this does not
    decide identification, only estimates given it. Rows with a missing or non-numeric
    value in any named column are dropped.
    """
    if estimand not in ("ate", "att", "atc"):
        raise ValueError(f"estimand must be 'ate', 'att' or 'atc', not {estimand!r}")
    adjust = tuple(dict.fromkeys(adjust))
    data, n = _frame(rows, [x, y, *adjust])
    base = EffectOpinion(x, y, adjust, float(x0), float(x1), estimand, n)
    if n < _MIN_ROWS:
        return replace(base, caveats=(f"too few complete rows ({n})",))
    T = np.asarray(data[x], dtype=np.float64)
    Y = np.asarray(data[y], dtype=np.float64)
    Z = (
        np.column_stack([np.asarray(data[c], dtype=np.float64) for c in adjust])
        if adjust
        else np.zeros((n, 0))
    )
    two_valued = bool(np.all(np.isclose(T, x0) | np.isclose(T, x1)))
    rng = np.random.default_rng(seed)
    draws = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(bootstrap)]
    estimates: dict[str, float] = {}
    errors: dict[str, float] = {}
    for kind in MODELS:
        vals = [
            _g_effect(kind, T[ix], Z[ix], Y[ix], base.x0, base.x1, estimand, two_valued)
            for ix in draws
        ]
        boot = np.asarray(vals[1:], dtype=np.float64)
        boot = boot[np.isfinite(boot)]
        if math.isfinite(vals[0]) and boot.size >= 2:
            estimates[kind], errors[kind] = vals[0], float(np.std(boot, ddof=1))
    caveats: list[str] = []
    arms_ok = (
        min(int(np.isclose(T, x1).sum()), int(np.isclose(T, x0).sum())) >= _MIN_ARM
    )
    if two_valued and arms_ok:
        T01: Floats = np.isclose(T, x1).astype(np.float64)
        Zd = Z if adjust else np.zeros((n, 1))
        estimates[DOUBLY_ROBUST], errors[DOUBLY_ROBUST] = aipw_effect(
            T01, Zd, Y, estimand, seed=seed
        )
    elif two_valued:
        caveats.append(
            f"fewer than {_MIN_ARM} rows at x0 or x1: no doubly robust estimate"
        )
    else:
        caveats.append(
            "treatment takes more than two values: no doubly robust estimate"
        )
    return replace(
        base, estimates=estimates, standard_errors=errors, caveats=tuple(caveats)
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_interval(
    components: Sequence[tuple[float, float, float]], level: float = 0.95
) -> tuple[float, float]:
    """Central ``level`` interval of a mixture of normals ``(weight, mean, sd)``."""
    w = [c[0] for c in components]
    total = math.fsum(w)
    parts = [
        (wi / total, m, max(sd, 1e-12))
        for wi, (_, m, sd) in zip(w, components, strict=True)
    ]

    def cdf(v: float) -> float:
        return math.fsum(wi * _normal_cdf((v - m) / sd) for wi, m, sd in parts)

    lo_all = min(m - 10 * sd for _, m, sd in parts)
    hi_all = max(m + 10 * sd for _, m, sd in parts)

    def quantile(p: float) -> float:
        a, b = lo_all, hi_all
        for _ in range(200):
            mid = (a + b) / 2
            if cdf(mid) < p:
                a = mid
            else:
                b = mid
        return (a + b) / 2

    tail = (1 - level) / 2
    return quantile(tail), quantile(1 - tail)


def combine_interval(
    lower: float, upper: float, opinion: EffectOpinion, level: float = 0.95
) -> tuple[float, float]:
    """The agent's ``level`` interval combined with elbi's g-computation models.

    The agent's interval is read as a normal (centre, half-width / z); it holds half
    the weight and elbi's models share the other half, each a normal at its estimate
    with its bootstrap standard error. Returns the agent's interval unchanged when
    elbi has no estimate.
    """
    if upper < lower:
        raise ValueError("upper must not be below lower")
    z = _z(level)
    models = [k for k in MODELS if k in opinion.estimates]
    if not models:
        return lower, upper
    agent = (0.5, (lower + upper) / 2, (upper - lower) / (2 * z))
    elbi = [
        (0.5 / len(models), opinion.estimates[k], opinion.standard_errors[k])
        for k in models
    ]
    return mixture_interval([agent, *elbi], level)


def combine_point(estimate: float, opinion: EffectOpinion) -> float:
    """The equal-weight mean of the agent's estimate and elbi's doubly robust one.

    Returns the agent's estimate unchanged when elbi has no doubly robust estimate.
    """
    dr = opinion.doubly_robust
    return estimate if dr is None else (estimate + dr) / 2


def _z(level: float) -> float:
    """Two-sided normal quantile for ``level`` (bisection on the CDF)."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    target = 1 - (1 - level) / 2
    a, b = 0.0, 40.0
    for _ in range(200):
        mid = (a + b) / 2
        if _normal_cdf(mid) < target:
            a = mid
        else:
            b = mid
    return (a + b) / 2
