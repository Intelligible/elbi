"""A typed effect claim: what an analysis estimated, in which population, by which
identification strategy, under which assumed graph. Plus the operations that need it:

  check_identification  is the declared strategy valid in the declared graph?
  estimate              run the estimator the strategy implies (not always backdoor)
  comparable            may two results be combined (same estimand and population)?
  retarget              reuse a fitted analysis for a different estimand, or say why not

An estimate and interval alone support none of these: they do not say which functional
of the data they are, so a checker can only guess (and guessed backdoor adjustment when
the agent had used the front-door formula).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer

STRATEGIES = ("backdoor", "frontdoor", "not_identifiable")
ESTIMANDS = ("ate", "ett")


@dataclass(frozen=True)
class EffectClaim:
    treatment: str
    outcome: str
    estimand: str              # "ate" | "ett" (ett: averaged over rows with treatment == x1)
    x0: float
    x1: float
    strategy: str              # "backdoor" | "frontdoor" | "not_identifiable"
    adjust: tuple = ()         # backdoor set
    mediators: tuple = ()      # front-door set
    graph: tuple = ()          # assumed DAG as (cause, effect) pairs, latent nodes included
    latent: tuple = ()         # nodes in the graph with no data
    data_hash: str = ""        # the rows the analysis was fitted on

    def key(self):
        """What makes two results estimates of the same quantity."""
        return (self.treatment, self.outcome, self.estimand, round(self.x0, 9), round(self.x1, 9), self.data_hash)


def data_hash(rows: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(rows, index=False).values.tobytes()).hexdigest()[:16]


# ---------- identification against the declared graph ----------

def _dag(claim):
    g = nx.DiGraph()
    g.add_edges_from(claim.graph)
    g.add_nodes_from([claim.treatment, claim.outcome, *claim.adjust, *claim.mediators])
    return g


def _dsep(g, xs, ys, zs):
    f = getattr(nx, "is_d_separator", None) or nx.d_separated
    return f(g, set(xs), set(ys), set(zs))


def _backdoor_ok(g, x, y, z):
    if set(z) & (nx.descendants(g, x) | {x}):
        return False, f"adjustment set contains a descendant of {x}"
    h = g.copy()
    h.remove_edges_from(list(g.out_edges(x)))
    if not _dsep(h, [x], [y], z):
        return False, "adjustment set leaves a backdoor path open"
    return True, "backdoor criterion holds"


def _frontdoor_ok(g, x, y, m):
    h = g.copy()
    h.remove_nodes_from(m)
    if nx.has_path(h, x, y):
        return False, "mediators do not intercept every directed path from treatment to outcome"
    for mi in m:
        ok, _ = _backdoor_ok(g, x, mi, [])
        if not ok:
            return False, f"open backdoor path from treatment to mediator {mi}"
        ok, _ = _backdoor_ok(g, mi, y, [x])
        if not ok:
            return False, f"backdoor path from mediator {mi} to outcome not blocked by the treatment"
    return True, "front-door criterion holds"


def check_identification(claim: EffectClaim) -> tuple[bool, str]:
    """Whether the declared strategy identifies the declared estimand in the declared graph.
    (Validity is relative to the graph the analyst assumed; the graph itself is not tested.)"""
    if not claim.graph:
        return False, "no graph declared: identification cannot be checked"
    g = _dag(claim)
    if not nx.is_directed_acyclic_graph(g):
        return False, "declared graph has a cycle"
    unobserved = set(claim.latent) & {*claim.adjust, *claim.mediators, claim.treatment, claim.outcome}
    if unobserved:
        return False, f"strategy uses unobserved variables {sorted(unobserved)}"
    if claim.strategy == "backdoor":
        return _backdoor_ok(g, claim.treatment, claim.outcome, claim.adjust)
    if claim.strategy == "frontdoor":
        if claim.estimand != "ate":
            return False, "front-door estimation is implemented for the ATE only"
        return _frontdoor_ok(g, claim.treatment, claim.outcome, claim.mediators)
    return True, "declared not identifiable"


# ---------- outcome models ----------

def _binary(y):
    u = np.unique(y[~np.isnan(y)])
    return len(u) <= 2 and set(u) <= {0, 1}


def _model(kind, binary):
    if kind == "spline":
        feats = lambda n: make_pipeline(QuantileTransformer(n_quantiles=min(1000, n)),
                                        SplineTransformer(n_knots=8, degree=3, extrapolation="linear"))
        return (lambda n: make_pipeline(feats(n), LogisticRegressionCV(Cs=6, max_iter=3000))) if binary else \
               (lambda n: make_pipeline(feats(n), RidgeCV(alphas=np.logspace(-3, 3, 13))))
    if kind == "linear":
        return (lambda n: LogisticRegression(max_iter=3000)) if binary else (lambda n: LinearRegression())
    return (lambda n: HistGradientBoostingClassifier(max_iter=200, random_state=0)) if binary else \
           (lambda n: HistGradientBoostingRegressor(max_iter=200, random_state=0))


class Fitted:
    """A fitted outcome model as a picklable callable (expected outcome given features)."""

    def __init__(self, model, binary):
        self.model, self.binary = model, binary

    def __call__(self, Z):
        return self.model.predict_proba(Z)[:, 1] if self.binary else self.model.predict(Z)


def _fit(rows, features, target, kind):
    X = rows[features].to_numpy(float)
    y = rows[target].to_numpy(float)
    b = _binary(y)
    return Fitted(_model(kind, b)(len(X)).fit(X, y), b)


KINDS = ("spline", "linear", "boosted")


@dataclass
class FittedAnalysis:
    """A claim plus the fitted outcome models (point fit and bootstrap refits), so a
    related question can be answered by re-averaging instead of refitting."""
    claim: EffectClaim
    fits: dict = field(default_factory=dict)        # kind -> [point fit, boot fits...]
    boot_rows: list = field(default_factory=list)   # row indices of each bootstrap sample
    rows: pd.DataFrame | None = None
    seconds_fit: float = 0.0


def _contrast(f, rows, claim):
    """Average of f(x1, Z) - f(x0, Z) over the claim's population within `rows`."""
    pop = rows if claim.estimand == "ate" else rows[np.isclose(rows[claim.treatment], claim.x1)]
    if len(pop) == 0:
        return np.nan
    X = pop[[claim.treatment, *claim.adjust]].to_numpy(float)
    X1, X0 = X.copy(), X.copy()
    X1[:, 0], X0[:, 0] = claim.x1, claim.x0
    return float(np.mean(f(X1) - f(X0)))


def _frontdoor(rows, claim, kind, rng, n_sub=400):
    """E[Y|do(x)] = sum_m P(m|x) E_x'[E[Y|m,x']]; continuous mediators via residual draws."""
    x, y, m = claim.treatment, claim.outcome, list(claim.mediators)
    sub = rows.iloc[rng.choice(len(rows), min(n_sub, len(rows)), replace=False)]
    fy = _fit(rows, m + [x], y, kind)
    med = {}
    for mi in m:
        Xm = rows[[x]].to_numpy(float)
        ym = rows[mi].to_numpy(float)
        lin = LinearRegression().fit(Xm, ym)
        med[mi] = (lin, ym - lin.predict(Xm))

    def mean_y(xv):
        draws = np.column_stack([med[mi][0].predict(np.full((len(sub), 1), xv)) + rng.choice(med[mi][1], len(sub))
                                 for mi in m])
        xs = sub[x].to_numpy(float)
        Z = np.column_stack([np.repeat(draws, len(xs), axis=0), np.tile(xs, len(draws))])
        return float(np.mean(fy(Z)))
    return mean_y(claim.x1) - mean_y(claim.x0)


def estimate(claim: EffectClaim, rows: pd.DataFrame, B: int = 40, seed: int = 0) -> tuple[dict, FittedAnalysis]:
    """Per-model estimate and bootstrap SE for the claim's estimand by the claim's strategy."""
    ok, why = check_identification(claim) if claim.graph else (True, "graph not declared")
    if claim.strategy == "not_identifiable" or not ok:
        return {"checked": False, "reason": why}, FittedAnalysis(claim)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    fa = FittedAnalysis(claim, rows=rows)
    idx = [np.arange(len(rows))] + [rng.integers(0, len(rows), len(rows)) for _ in range(B)]
    fa.boot_rows = idx
    est, se = {}, {}
    for kind in KINDS:
        vals = []
        if claim.strategy == "backdoor":
            fa.fits[kind] = []
            for ix in idx:
                r = rows.iloc[ix].reset_index(drop=True)
                f = _fit(r, [claim.treatment, *claim.adjust], claim.outcome, kind)
                fa.fits[kind].append(f)
                vals.append(_contrast(f, r, claim))
        else:
            for ix in idx[: B // 2 + 1]:  # front-door is costlier
                vals.append(_frontdoor(rows.iloc[ix].reset_index(drop=True), claim, kind, rng))
        est[kind], se[kind] = vals[0], float(np.nanstd(vals[1:], ddof=1))
    fa.seconds_fit = time.time() - t0
    return {"checked": True, "estimates": est, "se": se, "reason": why}, fa


def comparable(a: EffectClaim, b: EffectClaim) -> tuple[bool, str]:
    """Two results may be pooled only if they estimate the same quantity on the same rows."""
    diff = [n for n, u, v in zip(("treatment", "outcome", "estimand", "x0", "x1", "data"), a.key(), b.key()) if u != v]
    return (not diff, "same estimand and population" if not diff else f"differ in {diff}")


def retarget(fa: FittedAnalysis, estimand: str, rows: pd.DataFrame | None = None) -> tuple[dict, str]:
    """Answer a different estimand from a saved analysis without refitting, when valid.

    Valid when: the strategy identifies the new estimand under the same graph (backdoor:
    the same set identifies ATE and ETT), the data are the same (hash matches), and the
    fitted models exist. Otherwise it says why and returns nothing."""
    c = replace(fa.claim, estimand=estimand)
    if rows is not None and data_hash(rows) != fa.claim.data_hash:
        return {}, "data changed since the analysis was fitted: refit, do not reuse"
    if fa.claim.strategy != "backdoor":
        return {}, f"reuse for {estimand} not supported for strategy {fa.claim.strategy}"
    ok, why = check_identification(c) if c.graph else (True, "")
    if not ok:
        return {}, why
    t0 = time.time()
    est, se = {}, {}
    for kind, fits in fa.fits.items():
        vals = [_contrast(f, fa.rows.iloc[ix].reset_index(drop=True), c) for f, ix in zip(fits, fa.boot_rows)]
        est[kind], se[kind] = vals[0], float(np.nanstd(vals[1:], ddof=1))
    return {"checked": True, "estimates": est, "se": se, "seconds": time.time() - t0}, "reused fitted models"
