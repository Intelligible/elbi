"""Best-practice effect estimators for the certified estimate.

The soundness gates decide whether an effect *holds*; these decide *how large* it is,
using estimators appropriate to the treatment and the covariate overlap rather than a
single linear coefficient. Everything is seeded (fixed model state, fixed cross-fitting
folds) so the certified estimate is a pure function of the rows: a re-run reproduces it
exactly, which is what a repeatable derivation requires.

scikit-learn is an optional dependency (shared with the prediction gate); if it is
absent these are unavailable and the effect gate keeps its linear estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold

Floats: TypeAlias = NDArray[np.float64]


class EffectEstimate(TypedDict):
    """An ATE, the estimator that produced it, and the overlap that chose it.

    ``overlap`` is None where the estimator does not rest on a propensity overlap
    diagnostic (double ML on a continuous treatment).
    """

    estimate: float
    method: str
    overlap: float | None


def _reg(X: Floats, y: Floats) -> Any:
    return GradientBoostingRegressor(random_state=0).fit(X, y)


def _folds(n: int) -> KFold:
    return KFold(n_splits=min(5, max(2, n // 20)), shuffle=True, random_state=0)


def overlap_fraction(X: Floats, T: Floats) -> float:
    """Cross-fitted share of units with propensity in [0.05, 0.95].

    A gold-free positivity diagnostic: a low share means weak overlap, where
    IPW/AIPW is unstable.
    """
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    e = np.zeros(len(T))
    for tr, te in _folds(len(T)).split(X):
        e[te] = (
            GradientBoostingClassifier(random_state=0)
            .fit(X[tr], T[tr])
            .predict_proba(X[te])[:, 1]
        )
    return float(np.mean((e > 0.05) & (e < 0.95)))


def aipw(X: Floats, T: Floats, Y: Floats, trim: float = 0.02) -> float:
    """Cross-fitted doubly-robust AIPW ATE.

    Consistent if either the outcome or the propensity model is correct
    (Robins; AJE 2021).
    """
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    Y = np.asarray(Y, float)
    n = len(Y)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    e = np.zeros(n)
    for tr, te in _folds(n).split(X):
        Xtr, Ttr, Ytr = X[tr], T[tr], Y[tr]
        mu1[te] = _reg(Xtr[Ttr == 1], Ytr[Ttr == 1]).predict(X[te])
        mu0[te] = _reg(Xtr[Ttr == 0], Ytr[Ttr == 0]).predict(X[te])
        e[te] = (
            GradientBoostingClassifier(random_state=0)
            .fit(Xtr, Ttr)
            .predict_proba(X[te])[:, 1]
        )
    e = np.clip(e, trim, 1 - trim)
    psi = (mu1 - mu0) + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    return float(psi.mean())


def t_learner(X: Floats, T: Floats, Y: Floats) -> float:
    """Outcome-model ATE (meta-learner).

    Robust when overlap is weak and IPW explodes.
    """
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    Y = np.asarray(Y, float)
    m1 = _reg(X[T == 1], Y[T == 1])
    m0 = _reg(X[T == 0], Y[T == 0])
    return float((m1.predict(X) - m0.predict(X)).mean())


def robust_ate(
    X: Floats, T: Floats, Y: Floats, overlap_min: float = 0.9
) -> EffectEstimate:
    """Overlap-aware ATE selection.

    Doubly-robust AIPW where overlap is adequate, outcome-model where it is weak.
    The rule follows causal-inference practice, not any score.
    """
    ov = overlap_fraction(X, T)
    method = "aipw" if ov >= overlap_min else "outcome-model"
    est = aipw(X, T, Y) if method == "aipw" else t_learner(X, T, Y)
    return {"estimate": est, "method": method, "overlap": ov}


def dml_partial_linear(X: Floats, T: Floats, Y: Floats) -> float:
    """Double/debiased ML for a continuous treatment.

    Robinson partial-linear model (Chernozhukov et al. 2018): residualize T and Y
    on the covariates with cross-fitted ML, then regress residual-Y on residual-T.
    """
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    Y = np.asarray(Y, float)
    Tr = np.zeros(len(T))
    Yr = np.zeros(len(Y))
    for tr, te in _folds(len(T)).split(X):
        Tr[te] = T[te] - _reg(X[tr], T[tr]).predict(X[te])
        Yr[te] = Y[te] - _reg(X[tr], Y[tr]).predict(X[te])
    return float(np.sum(Tr * Yr) / np.sum(Tr * Tr))


def best_practice_effect(
    data: Mapping[str, Sequence[float]], x: str, y: str, controls: Sequence[str]
) -> EffectEstimate | None:
    """The certified effect estimate given the claimed adjustment set.

    Binary treatment -> overlap-aware doubly-robust ATE; continuous treatment ->
    double ML. Returns None when there is no adjustment set (the unconditional
    coefficient is already the answer) or the sample is too small to cross-fit.
    """
    controls = [c for c in controls if c in data]
    if not controls:
        return None
    T = np.asarray(data[x], float)
    Y = np.asarray(data[y], float)
    X = np.column_stack([np.asarray(data[c], float) for c in controls])
    if len(Y) < 40:
        return None
    levels = np.unique(T)
    if levels.size == 2:
        T01 = (levels.max() == T).astype(float)
        r = robust_ate(X, T01, Y)
        return {
            "estimate": r["estimate"],
            "method": r["method"],
            "overlap": r["overlap"],
        }
    if levels.size > 2:
        return {
            "estimate": dml_partial_linear(X, T, Y),
            "method": "double-ML",
            "overlap": None,
        }
    return None
