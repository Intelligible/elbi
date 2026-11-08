"""elbi's effect estimates, v2: the g-computation models of effect_claim with the binary-outcome
spline fixed, plus a cross-fitted doubly robust (AIPW) estimate for a binary treatment.

Spline fix: the penalised spline model shrank the treatment term along with ~180 covariate
basis terms; with a weak outcome signal cross-validation picks strong shrinkage and the
effect came out ~0 (jobs data: C = 0.004, effect 0.0003). Regularisation must not shrink the
quantity being estimated, so for a two-valued treatment the spline is fitted separately in
each arm (on the covariates only) and the treatment term is never penalised. Probability
models are also cross-validated on log-loss, not accuracy.

AIPW: outcome models per treatment arm (the mean of elbi's three model kinds, so no choice
among them), logistic propensity clipped to [0.01, 0.99], 5-fold cross-fitting, standard
error from the influence function. Estimands: ate, att (over x == x1), atc (over x == x0).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

KINDS = ("spline", "linear", "boosted")


def _binary(y):
    u = np.unique(y[~np.isnan(y)])
    return len(u) <= 2 and set(u) <= {0, 1}


def _model(kind, binary, n):
    if kind == "spline":
        f = make_pipeline(QuantileTransformer(n_quantiles=min(1000, n)),
                          SplineTransformer(n_knots=8, degree=3, extrapolation="linear"))
        return make_pipeline(f, LogisticRegressionCV(Cs=6, max_iter=3000, scoring="neg_log_loss")) if binary else \
               make_pipeline(f, RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if kind == "linear":
        return LogisticRegression(max_iter=3000) if binary else LinearRegression()
    return HistGradientBoostingClassifier(max_iter=200, random_state=0) if binary else \
           HistGradientBoostingRegressor(max_iter=200, random_state=0)


def _fit_predict(kind, X, y, Xnew):
    b = _binary(y)
    m = _model(kind, b, len(X)).fit(X, y)
    return m.predict_proba(Xnew)[:, 1] if b else m.predict(Xnew)


def _pop(t, x0, x1, estimand):
    return np.ones(len(t), bool) if estimand == "ate" else np.isclose(t, x1 if estimand == "att" else x0)


def g_estimates(rows, x, y, given, x0, x1, estimand="ate", B=40, seed=0):
    """Per-model g-computation estimate and bootstrap SE (outcome model on treatment + given)."""
    rng = np.random.default_rng(seed)
    idx = [np.arange(len(rows))] + [rng.integers(0, len(rows), len(rows)) for _ in range(B)]
    est, se = {}, {}
    two = set(np.round(rows[x].unique(), 9)) <= {round(x0, 9), round(x1, 9)}
    for kind in KINDS:
        vals = []
        for ix in idx:
            r = rows.iloc[ix]
            X = r[[x, *given]].to_numpy(float)
            yy = r[y].to_numpy(float)
            P = X[_pop(X[:, 0], x0, x1, estimand)]
            if len(P) == 0:
                vals.append(np.nan)
                continue
            if kind == "spline" and two and given:
                # per-arm fits: the treatment term is not penalised
                arm = np.isclose(X[:, 0], x1)
                if arm.sum() < 5 or (~arm).sum() < 5:
                    vals.append(np.nan)
                    continue
                f1 = _fit_predict(kind, X[arm, 1:], yy[arm], P[:, 1:])
                f0 = _fit_predict(kind, X[~arm, 1:], yy[~arm], P[:, 1:])
                vals.append(float(np.mean(f1 - f0)))
                continue
            X1, X0 = P.copy(), P.copy()
            X1[:, 0], X0[:, 0] = x1, x0
            both = _fit_predict(kind, X, yy, np.vstack([X1, X0]))
            vals.append(float(np.mean(both[: len(P)] - both[len(P):])))
        est[kind], se[kind] = vals[0], float(np.nanstd(vals[1:], ddof=1))
    return est, se


def aipw(rows, x, y, given, x0, x1, estimand="ate", folds=5, seed=0):
    """Cross-fitted doubly robust estimate and SE, for a treatment taking values x0 and x1."""
    r = rows[np.isclose(rows[x], x0) | np.isclose(rows[x], x1)]
    T = np.isclose(r[x].to_numpy(float), x1).astype(float)
    Y = r[y].to_numpy(float)
    Z = r[list(given)].to_numpy(float) if given else np.zeros((len(r), 1))
    n = len(r)
    m1, m0, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Z):
        for arm, out in ((1, m1), (0, m0)):
            s = tr[T[tr] == arm]
            out[te] = np.mean([_fit_predict(k, Z[s], Y[s], Z[te]) for k in KINDS], axis=0)
        ps = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=6, max_iter=3000, scoring="neg_log_loss"))
        e[te] = ps.fit(Z[tr], T[tr]).predict_proba(Z[te])[:, 1]
    e = np.clip(e, 0.01, 0.99)
    if estimand == "ate":
        psi = m1 - m0 + T * (Y - m1) / e - (1 - T) * (Y - m0) / (1 - e)
        tau = psi.mean()
        phi = psi - tau
    elif estimand == "att":
        p = T.mean()
        num = T * (Y - m0) - (1 - T) * e / (1 - e) * (Y - m0)
        tau = num.sum() / T.sum()
        phi = (num - T * tau) / p
    else:  # atc
        p = (1 - T).mean()
        num = (1 - T) * (m1 - Y) + T * (1 - e) / e * (Y - m1)
        tau = num.sum() / (1 - T).sum()
        phi = (num - (1 - T) * tau) / p
    return float(tau), float(np.std(phi, ddof=1) / np.sqrt(n))
