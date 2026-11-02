"""The second-opinion effect estimate and the rules that combine it with an agent's.

Estimators are validated on synthetic data with a KNOWN effect; the combination rules
are held to invariants (translation and reflection equivariance, agreement leaves the
answer alone, disagreement moves it between the two) with Hypothesis.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

so = pytest.importorskip("elbi_core.verification.second_opinion")


@pytest.fixture(autouse=True, scope="module")
def _few_threads() -> Iterator[None]:
    """Cap native threads: under xdist, every worker's model fits otherwise take every
    core at once and the suite slows by orders of magnitude."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        yield


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


def test_sparse_binary_outcome_degrades_instead_of_raising() -> None:
    """A rare outcome and a tiny treated arm leave cross-validation folds with one
    class; the models that cannot be fitted drop out rather than failing the call."""
    rng = np.random.default_rng(0)
    n = 300
    z = rng.normal(0, 1, n)
    t = np.zeros(n)
    t[:6] = 1
    y = (rng.random(n) < 0.03).astype(float)
    y[0] = 1
    op = so.estimate_effect(
        _rows(t=t, y=y, z=z), "t", "y", ["z"], estimand="att", bootstrap=3
    )
    assert set(op.estimates) < {*so.MODELS, so.DOUBLY_ROBUST}
    assert all(math.isfinite(v) for v in op.estimates.values())


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


@pytest.mark.parametrize("treated", [0, 3])
def test_no_contrast_in_the_treatment_gives_no_estimate(treated: int) -> None:
    """Too few (or no) rows in one arm: every model would report an effect of exactly 0
    with no spread, a confident answer the data cannot support, so elbi declines."""
    rows = _confounded(200)
    kept = [r for r in rows if r["t"] == 0] + [r for r in rows if r["t"] == 1][:treated]
    op = so.estimate_effect(kept, "t", "y", ["z"], bootstrap=4)
    assert op.estimates == {}
    assert op.range() is None
    assert "barely varies" in op.caveats[0]


def test_rejects_equal_levels() -> None:
    with pytest.raises(ValueError, match="differ"):
        so.estimate_effect(_confounded(50), "t", "y", ["z"], x0=1.0, x1=1.0)


def test_rejects_levels_outside_the_observed_treatment() -> None:
    """A 1/2-coded treatment with the default 0/1 levels: the spline and boosted models
    cannot extrapolate, so each would report an effect of exactly 0."""
    rows = [{**r, "t": r["t"] + 1} for r in _confounded(200)]
    with pytest.raises(ValueError, match="observed"):
        so.estimate_effect(rows, "t", "y", ["z"], bootstrap=4)


def test_a_non_finite_doubly_robust_estimate_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(so, "aipw_effect", lambda *a, **k: (math.nan, math.nan))
    op = so.estimate_effect(_confounded(120), "t", "y", ["z"], bootstrap=2)
    assert op.doubly_robust is None
    assert "not finite" in op.caveats[-1]
    assert so.combine_point(1.5, op) == 1.5


def test_repeated_cross_fits_take_the_median_and_widen_by_their_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One split into folds can move the estimate enough to flip a graded answer, so
    the doubly robust estimate is the median over splits, and its standard error
    counts the disagreement between them."""
    splits = {0: (1.0, 0.1), 1: (5.0, 0.1), 2: (2.0, 0.3)}
    seen: list[int] = []

    def one_split(*args: object) -> tuple[float, float]:
        seed = args[-1]
        assert isinstance(seed, int)
        seen.append(seed)
        return splits[seed]

    monkeypatch.setattr(so, "_aipw_split", one_split)
    tau, se = so.aipw_effect(np.zeros(3), np.zeros((3, 1)), np.zeros(3), repeats=3)
    assert seen == [0, 1, 2]
    assert tau == 2.0
    # Per split: se^2 + (tau_k - median)^2 = 1.01, 9.01, 0.09; their median is 1.01.
    assert se == pytest.approx(math.sqrt(1.01))
    with pytest.raises(ValueError, match="repeats"):
        so.aipw_effect(np.zeros(3), np.zeros((3, 1)), np.zeros(3), repeats=0)


def test_binary_outcome_linear_model_is_not_shrunk() -> None:
    """The logistic outcome model is unpenalised, like the linear one: a penalty on
    unscaled inputs shrinks a strong effect in a small sample toward 0. Averaged over
    seeds, so the bias shows through the sampling noise."""
    bias = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 80
        z = rng.normal(0, 30, n)  # unscaled covariate, as data arrive
        t = (rng.random(n) < 0.5).astype(float)
        p = 1 / (1 + np.exp(-(-2.0 + 4.0 * t + 0.01 * z)))
        y = (rng.random(n) < p).astype(float)
        op = so.estimate_effect(_rows(t=t, y=y, z=z), "t", "y", ["z"], bootstrap=2)
        truth = np.mean(
            1 / (1 + np.exp(-(2.0 + 0.01 * z))) - 1 / (1 + np.exp(-(-2.0 + 0.01 * z)))
        )
        bias.append(op.estimates["linear"] - truth)
    assert abs(np.mean(bias)) < 0.06


def test_rejects_unknown_estimand() -> None:
    with pytest.raises(ValueError, match="estimand"):
        so.estimate_effect(_confounded(50), "t", "y", ["z"], estimand="late")  # type: ignore[arg-type]


def test_opinion_is_a_pure_function_of_the_rows() -> None:
    rows = _confounded(120)
    a = so.estimate_effect(rows, "t", "y", ["z"], bootstrap=2)
    b = so.estimate_effect(rows, "t", "y", ["z"], bootstrap=2)
    assert a == b


def test_spline_ignores_global_random_state_above_the_quantile_subsample() -> None:
    """Above 10,000 rows per fit the quantile transform subsamples; unseeded, the
    estimate would follow numpy's global random state."""
    rng = np.random.default_rng(0)
    n = 25_000
    z = rng.normal(0, 1, (n, 1))
    t = (rng.random(n) < 0.5).astype(float)
    y = t + z[:, 0] + rng.normal(0, 1, n)
    runs = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        runs.append(so._g_effect("spline", t, z, y, 0.0, 1.0, "ate", True))
    assert runs[0] == runs[1]


def test_model_fits_run_on_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenMP thread per core made fits this small about 20x slower."""
    from threadpoolctl import threadpool_info

    seen: set[int] = set()
    model = so._model

    def spy(*args: object) -> object:
        seen.update(
            i["num_threads"] for i in threadpool_info() if i["user_api"] == "openmp"
        )
        return model(*args)

    monkeypatch.setattr(so, "_model", spy)
    so.estimate_effect(_confounded(120), "t", "y", ["z"], bootstrap=2)
    assert seen == {1}


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
