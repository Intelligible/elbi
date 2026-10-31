"""The second-opinion effect estimate and the rules that combine it with an agent's.

Estimators are validated on synthetic data with a KNOWN effect; the combination rules
are held to invariants (translation and reflection equivariance, agreement leaves the
answer alone, disagreement moves it between the two) with Hypothesis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

so = pytest.importorskip("elbi_core.verification.second_opinion")


def _rows(**cols: np.ndarray) -> list[dict[str, float]]:
    keys = list(cols)
    return [{k: float(cols[k][i]) for k in keys} for i in range(len(cols[keys[0]]))]


def _confounded(n: int = 800, seed: int = 0) -> list[dict[str, float]]:
    """Binary treatment confounded by z; effect 1 + 2 z, so ATT > ATE > ATC."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-1.2 * z))).astype(float)
    y = (1 + 2 * z) * t + 3 * z + rng.normal(0, 1, n)
    return _rows(t=t, y=y, z=z)


def _truth(rows: list[dict[str, float]], estimand: str) -> float:
    eff = [1 + 2 * r["z"] for r in rows]
    pick = {
        "ate": [True] * len(rows),
        "att": [r["t"] == 1 for r in rows],
        "atc": [r["t"] == 0 for r in rows],
    }[estimand]
    return float(np.mean([e for e, keep in zip(eff, pick, strict=True) if keep]))


@pytest.fixture(scope="module")
def confounded() -> tuple[list[dict[str, float]], dict[str, object]]:
    """One dataset and its opinion per estimand, shared: each fit is the slow part."""
    rows = _confounded()
    return rows, {
        e: so.estimate_effect(rows, "t", "y", ["z"], estimand=e, bootstrap=3)
        for e in ("ate", "att", "atc")
    }


@pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
def test_recovers_each_estimand_under_confounding(
    confounded: tuple[list[dict[str, float]], dict[str, object]], estimand: str
) -> None:
    rows, opinions = confounded
    op = opinions[estimand]
    truth = _truth(rows, estimand)
    assert op.doubly_robust is not None
    assert abs(op.doubly_robust - truth) < 0.25
    lo, hi = op.range()
    assert lo < truth < hi
    naive = np.mean([r["y"] for r in rows if r["t"] == 1]) - np.mean(
        [r["y"] for r in rows if r["t"] == 0]
    )
    assert (
        abs(naive - truth) > 1.0
    )  # the confounding is real, so recovery means something


def test_estimands_are_distinguished(
    confounded: tuple[list[dict[str, float]], dict[str, object]],
) -> None:
    dr = {e: op.doubly_robust for e, op in confounded[1].items()}
    assert dr["att"] > dr["ate"] + 0.3 > dr["atc"] + 0.6


def test_spline_does_not_shrink_a_weak_binary_effect_to_zero() -> None:
    """A rare binary outcome, a small treated arm, and many covariates: a penalised
    model of treatment and covariates together shrinks the effect toward 0; fitted per
    arm, the treatment term is not penalised."""
    rng = np.random.default_rng(5)
    n, k = 2500, 15
    Z = rng.normal(0, 1, (n, k))
    t = (rng.random(n) < 0.1).astype(float)
    p = np.clip(0.85 - 0.07 + 0.14 * t + 0.01 * Z[:, 0], 0, 1)
    y = (rng.random(n) < p).astype(float)
    cols = {f"z{i}": Z[:, i] for i in range(k)}
    rows = _rows(t=t, y=y, **cols)
    op = so.estimate_effect(rows, "t", "y", list(cols), estimand="att", bootstrap=2)
    assert abs(op.estimates["spline"] - 0.14) < 0.06


def test_continuous_treatment_gets_g_computation_but_no_doubly_robust() -> None:
    rng = np.random.default_rng(2)
    n = 600
    z = rng.normal(0, 1, n)
    x = z + rng.normal(0, 1, n)
    y = 1.5 * x + 2 * z + rng.normal(0, 1, n)
    op = so.estimate_effect(
        _rows(x=x, y=y, z=z), "x", "y", ["z"], x0=0, x1=2, bootstrap=2
    )
    assert op.doubly_robust is None
    assert any("more than two values" in c for c in op.caveats)
    assert abs(op.estimates["linear"] - 3.0) < 0.2


def test_declines_with_too_few_rows_and_skips_incomplete_ones() -> None:
    rows = [{"t": 1.0, "y": 2.0, "z": None}] * 50 + [{"t": 0.0, "y": 1.0, "z": 0.0}] * 5
    op = so.estimate_effect(rows, "t", "y", ["z"])
    assert op.n == 5
    assert op.estimates == {}
    assert op.range() is None
    assert "too few" in op.caveats[0]


def test_small_arm_gets_no_doubly_robust() -> None:
    rows = _confounded(200)
    treated = [r for r in rows if r["t"] == 1][:3]
    op = so.estimate_effect(
        [r for r in rows if r["t"] == 0] + treated, "t", "y", ["z"], bootstrap=4
    )
    assert op.doubly_robust is None
    assert any("fewer than" in c for c in op.caveats)


def test_rejects_unknown_estimand() -> None:
    with pytest.raises(ValueError, match="estimand"):
        so.estimate_effect(_confounded(50), "t", "y", ["z"], estimand="late")  # type: ignore[arg-type]


def test_opinion_is_a_pure_function_of_the_rows() -> None:
    rows = _confounded(120)
    a = so.estimate_effect(rows, "t", "y", ["z"], bootstrap=2)
    b = so.estimate_effect(rows, "t", "y", ["z"], bootstrap=2)
    assert a == b


def test_no_adjustment_set_still_estimates() -> None:
    rng = np.random.default_rng(3)
    t = (rng.random(400) < 0.5).astype(float)
    y = 2 * t + rng.normal(0, 1, 400)
    op = so.estimate_effect(_rows(t=t, y=y), "t", "y", bootstrap=4)
    assert op.doubly_robust is not None
    assert abs(op.doubly_robust - 2) < 0.3


# ---------- combination rules ----------


def _opinion(estimates: dict[str, float], ses: dict[str, float]) -> object:
    return so.EffectOpinion("t", "y", (), 0.0, 1.0, "ate", 100, estimates, ses)


def test_combine_point_is_the_mean_with_the_doubly_robust_estimate() -> None:
    op = _opinion(
        {"linear": 9.0, so.DOUBLY_ROBUST: 4.0}, {"linear": 1, so.DOUBLY_ROBUST: 1}
    )
    assert so.combine_point(2.0, op) == 3.0
    assert so.combine_point(2.0, _opinion({"linear": 9.0}, {"linear": 1.0})) == 2.0


def test_combine_interval_without_estimates_returns_the_agents() -> None:
    assert so.combine_interval(1.0, 2.0, _opinion({}, {})) == (1.0, 2.0)
    with pytest.raises(ValueError, match="upper"):
        so.combine_interval(2.0, 1.0, _opinion({}, {}))


def test_single_normal_interval_matches_the_normal_quantiles() -> None:
    for level, z in ((0.95, 1.959964), (0.9, 1.644854)):
        lo, hi = so.mixture_interval([(1.0, 3.0, 2.0)], level)
        assert math.isclose(lo, 3 - 2 * z, abs_tol=1e-5)
        assert math.isclose(hi, 3 + 2 * z, abs_tol=1e-5)
    with pytest.raises(ValueError, match="level"):
        so.combine_interval(
            0.0, 1.0, _opinion({"linear": 0.5}, {"linear": 0.1}), level=1.0
        )


finite = st.floats(-50, 50, allow_nan=False)
positive = st.floats(0.01, 10, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(center=finite, half=positive, m=finite, se=positive, shift=finite)
def test_combine_interval_is_translation_and_reflection_equivariant(
    center: float, half: float, m: float, se: float, shift: float
) -> None:
    def comb(c: float, mm: float) -> tuple[float, float]:
        op = _opinion({"linear": mm, "spline": mm}, {"linear": se, "spline": se})
        return so.combine_interval(c - half, c + half, op)

    lo, hi = comb(center, m)
    slo, shi = comb(center + shift, m + shift)
    assert math.isclose(slo, lo + shift, abs_tol=1e-6)
    assert math.isclose(shi, hi + shift, abs_tol=1e-6)
    rlo, rhi = comb(-center, -m)
    assert math.isclose(rlo, -hi, abs_tol=1e-6)
    assert math.isclose(rhi, -lo, abs_tol=1e-6)


@settings(max_examples=60, deadline=None)
@given(center=finite, half=positive)
def test_agreement_leaves_the_interval_alone(center: float, half: float) -> None:
    sd = half / 1.959964
    op = _opinion({"linear": center, "boosted": center}, {"linear": sd, "boosted": sd})
    lo, hi = so.combine_interval(center - half, center + half, op)
    assert math.isclose(lo, center - half, abs_tol=1e-5)
    assert math.isclose(hi, center + half, abs_tol=1e-5)


@settings(max_examples=60, deadline=None)
@given(center=finite, half=positive, m=finite, se=positive)
def test_combined_interval_stays_within_the_two_intervals(
    center: float, half: float, m: float, se: float
) -> None:
    op = _opinion({"linear": m}, {"linear": se})
    lo, hi = so.combine_interval(center - half, center + half, op)
    assert lo >= min(center - half, m - 1.959964 * se) - 1e-6
    assert hi <= max(center + half, m + 1.959964 * se) + 1e-6
    assert lo <= hi


@settings(max_examples=60, deadline=None)
@given(center=finite, half=positive, gap=st.floats(0.01, 100), se=positive)
def test_disjoint_disagreement_widens_toward_elbi(
    center: float, half: float, gap: float, se: float
) -> None:
    m = (
        center + half + gap + 1.959964 * se
    )  # elbi's interval lies wholly above the agent's
    op = _opinion({"linear": m}, {"linear": se})
    lo, hi = so.combine_interval(center - half, center + half, op)
    assert hi > center + half
    assert lo > center - half - 1e-9
