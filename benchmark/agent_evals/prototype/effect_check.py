"""Average-effect check with self-assessment: several g-computation estimators, a bootstrap
interval, and a confidence verdict saying whether the check itself should be trusted.

The estimators differ in their outcome model only (additive splines on ranks, linear,
gradient boosting). When they agree, the effect is stable to modeling choices and the
spline interval is a fair second opinion. When they disagree, or the variables are known
only through noisy proxies, the check says so instead of reporting a confident number.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

import latent_effect as le


def _g(model, X, y, x0, x1, binary):
    model.fit(X, y)
    f = (lambda Z: model.predict_proba(Z)[:, 1]) if binary else model.predict
    X1, X0 = X.copy(), X.copy()
    X1[:, 0], X0[:, 0] = x1, x0
    return float(np.mean(f(X1) - f(X0)))


def linear_effect(rows, treat, outcome, adjust, x0, x1):
    X = rows[[treat] + list(adjust)].to_numpy(float)
    y = rows[outcome].to_numpy(float)
    b = le.is_binary(y)
    return _g(LogisticRegression(max_iter=3000) if b else LinearRegression(), X, y, x0, x1, b)


def boosted_effect(rows, treat, outcome, adjust, x0, x1):
    X = rows[[treat] + list(adjust)].to_numpy(float)
    y = rows[outcome].to_numpy(float)
    b = le.is_binary(y)
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0) if b else HistGradientBoostingRegressor(max_iter=200, random_state=0)
    return _g(m, X, y, x0, x1, b)


def spline_effect(rows, treat, outcome, adjust, x0, x1):
    return le.g_effect(rows, np.full(len(rows), 1 / len(rows)), treat, outcome, adjust, x0, x1)


MODELS = {"spline": spline_effect, "linear": linear_effect, "boosted": boosted_effect}


def assess(rows, treat, outcome, adjust, x0, x1, proxy=False, B=60, seed=0):
    """Each outcome model's point estimate and its bootstrap standard error (same resamples
    for all models)."""
    rng = np.random.default_rng(seed)
    args = (treat, outcome, list(adjust), float(x0), float(x1))
    est = {k: f(rows, *args) for k, f in MODELS.items()}
    draws = {k: [] for k in MODELS}
    for _ in range(B):
        r = rows.iloc[rng.integers(0, len(rows), len(rows))].reset_index(drop=True)
        for k, f in MODELS.items():
            draws[k].append(f(r, *args))
    se = {k: float(np.std(v, ddof=1)) for k, v in draws.items()}
    return {"estimates": est, "se": se, "n_rows": len(rows), "proxy": bool(proxy)}


def mixture_interval(components, level=0.95):
    """Central interval of an equal-uncertainty mixture of normals [(weight, mean, sd), ...].

    Under the interval score (a proper scoring rule) the best interval is the central
    quantiles of one's uncertainty about the effect. Each candidate analysis contributes its
    sampling uncertainty; the weights say how much each candidate is believed. Where the
    candidates agree the interval is as narrow as theirs; where they disagree it spans them."""
    from scipy.optimize import brentq
    from scipy.stats import norm
    w = np.array([c[0] for c in components], float); w /= w.sum()
    m = np.array([c[1] for c in components], float)
    sd = np.maximum(np.array([c[2] for c in components], float), 1e-12)
    cdf = lambda x: float(np.sum(w * norm.cdf((x - m) / sd)))
    a, b = (m - 10 * sd).min(), (m + 10 * sd).max()
    q = lambda p: brentq(lambda x: cdf(x) - p, a, b)
    return q((1 - level) / 2), q((1 + level) / 2)


def elbi_components(r, weight=1.0):
    """elbi's candidates: its models share `weight` equally."""
    k = len(r["estimates"])
    return [(weight / k, r["estimates"][m], r["se"][m]) for m in r["estimates"]]
