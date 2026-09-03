"""Best-practice effect estimators and their use in the effect gate.

Validates the estimators on synthetic data with a KNOWN true effect (not on any
benchmark's answers) and pins the determinism the certified estimate depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from elbi_core.verification import verify_effect

# _causal needs scikit-learn (the ml extra); skip the whole module if it is absent.
_causal = pytest.importorskip("elbi_core.verification._causal")


def _rows(**cols):
    keys = list(cols)
    return [{k: cols[k][i] for k in keys} for i in range(len(cols[keys[0]]))]


def test_robust_ate_uses_aipw_under_good_overlap() -> None:
    rng = np.random.default_rng(0)
    n = 1500
    X = rng.normal(0, 1, (n, 4))
    T = (rng.random(n) < 0.5).astype(float)  # randomized -> good overlap
    Y = 2.0 * T + X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)
    r = _causal.robust_ate(X, T, Y)
    assert r["method"] == "aipw"
    assert abs(r["estimate"] - 2.0) < 0.2


def test_robust_ate_falls_back_to_outcome_model_under_weak_overlap() -> None:
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.normal(0, 1, (n, 4))
    e = 1 / (
        1 + np.exp(-(2.5 * X[:, 0] + 2.0 * X[:, 1]))
    )  # strong confounding -> weak overlap
    T = (rng.random(n) < e).astype(float)
    Y = 2.0 * T + X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)
    r = _causal.robust_ate(X, T, Y)
    assert r["method"] == "outcome-model"
    assert abs(r["estimate"] - 2.0) < 0.3


def test_dml_recovers_continuous_treatment_effect() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    X = rng.normal(0, 1, (n, 3))
    T = X[:, 0] + rng.normal(0, 1, n)  # continuous, confounded by X0
    Y = 1.5 * T + 2.0 * X[:, 0] + rng.normal(0, 1, n)
    assert abs(_causal.dml_partial_linear(X, T, Y) - 1.5) < 0.15


def test_effect_gate_reports_best_practice_and_beats_naive() -> None:
    rng = np.random.default_rng(3)
    n = 800
    z = rng.normal(0, 1, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-1.5 * z))).astype(float)  # confounded by z
    y = 3.0 * t + 2.0 * z + rng.normal(0, 0.5, n)  # true effect +3
    rows = _rows(t=t.tolist(), y=y.tolist(), z=z.tolist())
    rep = verify_effect(rows, "t", "y", controls=["z"])
    naive = y[t == 1].mean() - y[t == 0].mean()  # biased upward by z
    assert rep.verdict == "sound"
    assert abs(rep.effect - 3.0) < abs(
        naive - 3.0
    )  # closer to truth than the naive diff
    assert abs(rep.effect - 3.0) < 0.4


def test_certified_estimate_is_deterministic() -> None:
    """Same rows -> identical estimate, bit for bit. This is what lets the certified
    estimate be a repeatable derivation (the cache stays valid across re-runs)."""
    rng = np.random.default_rng(4)
    n = 600
    z = rng.normal(0, 1, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 2.0 * t + z + rng.normal(0, 0.5, n)
    rows = _rows(t=t.tolist(), y=y.tolist(), z=z.tolist())
    a = verify_effect(rows, "t", "y", controls=["z"]).effect
    b = verify_effect(rows, "t", "y", controls=["z"]).effect
    assert a == b
