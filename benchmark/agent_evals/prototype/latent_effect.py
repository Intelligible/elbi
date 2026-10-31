"""Effect estimation with noisy measurements and a non-random gold-standard subsample.

Setting: some variables are latent, observed only through several noisy measurement
columns (Name1, Name2, ...). A calibration subset also holds their true values, but its
rows were chosen by those true values (not at random), so statistics computed on it --
or a model of truth-given-measurements fit on it -- are biased.

Method ("reweight the gold standard, using the measurements as a bridge"):
1. Measurement model, fit on the calibration rows in the direction that selection on the
   truth leaves unchanged: each measurement given its own latent's true value. Continuous
   measurements: flexible mean (splines) plus Student-t noise; binary: logistic on splines.
2. Population weights for the calibration rows: each calibration row is a candidate
   "true state" (all latents, plus the error-free columns). A nonparametric maximum-
   likelihood EM finds weights under which these states, pushed through the measurement
   model (and a kernel on the error-free columns), best explain the full data.
3. g-computation on the weighted calibration rows: a flexible outcome model of the true
   values, averaged over the weighted adjustment variables at the two treatment levels.
4. Bootstrap the whole pipeline (calibration rows and data rows) for the 95% interval.
"""

import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


def proxies_of(cols, concept):
    pat = re.compile(rf"^{re.escape(concept)}\d+$")
    return [c for c in cols if pat.match(c)]


def is_binary(x):
    v = np.unique(x[~np.isnan(x)])
    return len(v) <= 2 and set(v) <= {0.0, 1.0}


def _basis(l):
    return SplineTransformer(n_knots=6, degree=3, extrapolation="linear")


class MeasurementModel:
    """log p(w_j | latent value) for each measurement column, fit on calibration rows."""

    def __init__(self, calib, latents, data_cols):
        self.parts = []
        for lat in latents:
            L = calib[lat].to_numpy(float)
            lbin = is_binary(L)
            for col in proxies_of(data_cols, lat):
                w = calib[col].to_numpy(float)
                ok = ~np.isnan(w)
                if is_binary(w):
                    if lbin:
                        p1 = (w[ok & (L == 1)].sum() + 1) / ((ok & (L == 1)).sum() + 2)
                        p0 = (w[ok & (L == 0)].sum() + 1) / ((ok & (L == 0)).sum() + 2)
                        self.parts.append(("bin_bin", lat, col, (p0, p1)))
                    else:
                        m = make_pipeline(_basis(L), LogisticRegression(max_iter=2000, C=1.0))
                        m.fit(L[ok, None], w[ok])
                        self.parts.append(("bin_cont", lat, col, m))
                else:
                    if lbin:
                        mu = (w[ok & (L == 0)].mean(), w[ok & (L == 1)].mean())
                        resid = np.where(L[ok] == 1, w[ok] - mu[1], w[ok] - mu[0])
                        self.parts.append(("cont_bin", lat, col, (mu, stats.t.fit(resid))))
                    else:
                        m = make_pipeline(_basis(L), Ridge(alpha=1e-3))
                        m.fit(L[ok, None], w[ok])
                        resid = w[ok] - m.predict(L[ok, None])
                        self.parts.append(("cont_cont", lat, col, (m, stats.t.fit(resid))))

    def loglik(self, rows, states):
        """Matrix [n_rows, n_states] of sum_j log p(w_ij | state_k)."""
        out = np.zeros((len(rows), len(states)))
        for kind, lat, col, par in self.parts:
            w = rows[col].to_numpy(float)
            L = states[lat].to_numpy(float)
            miss = np.isnan(w)
            if kind == "bin_bin":
                p = np.where(L == 1, par[1], par[0])
                ll = np.where(w[:, None] == 1, np.log(p)[None, :], np.log1p(-p)[None, :])
            elif kind == "bin_cont":
                p = np.clip(par.predict_proba(L[:, None])[:, 1], 1e-6, 1 - 1e-6)
                ll = np.where(w[:, None] == 1, np.log(p)[None, :], np.log1p(-p)[None, :])
            elif kind == "cont_bin":
                (mu0, mu1), (df, loc, sc) = par
                mu = np.where(L == 1, mu1, mu0)
                ll = stats.t.logpdf(w[:, None] - mu[None, :], df, loc, sc)
            else:
                m, (df, loc, sc) = par
                mu = m.predict(L[:, None])
                ll = stats.t.logpdf(w[:, None] - mu[None, :], df, loc, sc)
            ll[miss, :] = 0.0
            out += ll
        return out


def exact_kernel(rows, states, exact):
    """log Gaussian product kernel on the error-free columns (Silverman bandwidths)."""
    out = np.zeros((len(rows), len(states)))
    for c in exact:
        a, b = rows[c].to_numpy(float), states[c].to_numpy(float)
        if is_binary(np.concatenate([a, b])):
            out += np.where(a[:, None] == b[None, :], 0.0, np.log(1e-3))
            continue
        h = 1.06 * np.nanstd(a) * len(a) ** -0.2 + 1e-9
        out += -0.5 * ((a[:, None] - b[None, :]) / h) ** 2
    return out


def population_weights(data, calib, latents, exact, n_rows=2000, iters=300, rng=None):
    rng = rng or np.random.default_rng(0)
    rows = data.iloc[rng.choice(len(data), min(n_rows, len(data)), replace=False)]
    mm = MeasurementModel(calib, latents, list(data.columns))
    LL = mm.loglik(rows, calib) + exact_kernel(rows, calib, exact)
    logpi = np.full(len(calib), -np.log(len(calib)))
    for _ in range(iters):
        post = LL + logpi[None, :]
        post -= logsumexp(post, axis=1, keepdims=True)
        new = logsumexp(post, axis=0) - np.log(len(rows))
        if np.max(np.abs(np.exp(new) - np.exp(logpi))) < 1e-7:
            logpi = new
            break
        logpi = new
    return np.exp(logpi)


def tilt_features(states, latents, exact):
    """Smooth features of each calibration row's true state: splines of each latent (or
    the latent itself if binary) and of each error-free column. Weights are restricted to
    exp(features @ theta), so they cannot collapse onto a handful of rows."""
    cols = []
    for c in list(latents) + list(exact):
        x = states[c].to_numpy(float)[:, None]
        if is_binary(x.ravel()):
            cols.append(x)
        else:
            z = (x - x.mean()) / (x.std() + 1e-9)
            cols.append(SplineTransformer(n_knots=5, degree=3, extrapolation="linear").fit_transform(z))
    F = np.hstack(cols)
    return (F - F.mean(0)) / (F.std(0) + 1e-9)


def tilted_weights(data, calib, latents, exact, n_rows=2000, lam=0.05, rng=None):
    from scipy.optimize import minimize
    rng = rng or np.random.default_rng(0)
    rows = data.iloc[rng.choice(len(data), min(n_rows, len(data)), replace=False)]
    mm = MeasurementModel(calib, latents, list(data.columns))
    LL = mm.loglik(rows, calib) + exact_kernel(rows, calib, exact)
    F = tilt_features(calib, latents, exact)

    def f(theta):
        s = F @ theta
        logw = s - logsumexp(s)
        post = LL + logw[None, :]
        lse = logsumexp(post, axis=1, keepdims=True)
        r = np.exp(post - lse)
        rbar = r.mean(0)
        w = np.exp(logw)
        obj = -lse.mean() + lam * theta @ theta
        grad = -F.T @ (rbar - w) + 2 * lam * theta
        return obj, grad

    theta = minimize(f, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B").x
    s = F @ theta
    return np.exp(s - logsumexp(s))


def g_effect(states, w, treat, outcome, adjust, x0, x1):
    """g-computation with an additive spline model on rank-transformed features (logistic
    for a binary outcome). Ranks keep heavy-tailed variables from flattening the curve;
    additive splines are stable where boosting was not."""
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    X = states[[treat] + list(adjust)].to_numpy(float)
    y = states[outcome].to_numpy(float)
    feats = make_pipeline(QuantileTransformer(n_quantiles=min(1000, len(X))),
                          SplineTransformer(n_knots=8, degree=3, extrapolation="linear"))
    if is_binary(y):
        m = make_pipeline(feats, LogisticRegressionCV(Cs=6, max_iter=3000))
        m.fit(X, y, logisticregressioncv__sample_weight=w * len(w))
        f = lambda Z: m.predict_proba(Z)[:, 1]
    else:
        m = make_pipeline(feats, RidgeCV(alphas=np.logspace(-3, 3, 13)))
        m.fit(X, y, ridgecv__sample_weight=w * len(w))
        f = m.predict
    X1, X0 = X.copy(), X.copy()
    X1[:, 0], X0[:, 0] = x1, x0
    return float(np.sum(w * (f(X1) - f(X0))) / np.sum(w))


def estimate(data, calib, treat, outcome, adjust, x0, x1, rng=None):
    latents = [c for c in calib.columns if c not in data.columns]
    exact = [c for c in data.columns if c in calib.columns and not any(
        re.match(rf"^{re.escape(l)}\d+$", c) for l in latents)]
    w = tilted_weights(data, calib, latents, exact, rng=rng)
    return g_effect(calib, w, treat, outcome, adjust, x0, x1), w


def run(data, calib, treat, outcome, adjust, x0, x1, B=20, seed=0):
    rng = np.random.default_rng(seed)
    point, w = estimate(data, calib, treat, outcome, adjust, x0, x1, rng=rng)
    draws = []
    for _ in range(B):
        d = data.iloc[rng.integers(0, len(data), len(data))]
        c = calib.iloc[rng.integers(0, len(calib), len(calib))].reset_index(drop=True)
        draws.append(estimate(d, c, treat, outcome, adjust, x0, x1, rng=rng)[0])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    ess = 1.0 / np.sum(w ** 2)
    return {"estimate": point, "ci": [float(lo), float(hi)], "effective_calibration_rows": float(ess)}


def run_calibration_spline(calib, treat, outcome, adjust, x0, x1, B=20, seed=0):
    """Calibration rows only, unweighted, same spline g-computation: isolates the
    contribution of the reweighting step."""
    rng = np.random.default_rng(seed)
    u = lambda c: g_effect(c, np.full(len(c), 1 / len(c)), treat, outcome, adjust, x0, x1)
    point = u(calib)
    draws = [u(calib.iloc[rng.integers(0, len(calib), len(calib))].reset_index(drop=True)) for _ in range(B)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"estimate": point, "ci": [float(lo), float(hi)]}
