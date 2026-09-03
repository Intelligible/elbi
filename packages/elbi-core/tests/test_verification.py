"""Validation of the deterministic verification oracle, at the behavior boundary.

Each test builds data with a known soundness and asserts the verdict: genuinely
sound effects pass, each unsound class is caught (confounding, collider control,
outliers, fragility, latent confounding via a negative control), and the verdicts
hold across seeds rather than on one lucky draw.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from elbi_core import verify_all
from elbi_core.verification import (
    verify_comparison,
    verify_correlation,
    verify_effect,
    verify_prediction,
    verify_regression,
    verify_trend,
)

N = 700


def _rows(**columns: list[float]) -> list[dict[str, str]]:
    names = list(columns)
    n = len(columns[names[0]])
    return [{c: str(columns[c][i]) for c in names} for i in range(n)]


def _world(kind: str, seed: int) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    rng = random.Random(seed)

    def noise(scale: float = 1.0) -> list[float]:
        return [rng.gauss(0, scale) for _ in range(N)]

    if kind == "robust":
        x = noise()
        return _rows(x=x, y=[2 * xi + ni for xi, ni in zip(x, noise())]), {}
    if kind == "moderate":
        x = noise()
        return _rows(x=x, y=[0.7 * xi + ni for xi, ni in zip(x, noise())]), {}
    if kind == "confounded":
        z = noise()
        x = [zi + ni for zi, ni in zip(z, noise(0.5))]
        y = [-0.2 * xi + 1.5 * zi + ni for xi, zi, ni in zip(x, z, noise(0.5))]
        return _rows(x=x, y=y, z=z), {}
    if kind == "collider":
        x, y = noise(), noise()
        c = [xi + yi + ni for xi, yi, ni in zip(x, y, noise(0.4))]
        return _rows(x=x, y=y, c=c), {"controls": ["c"]}
    if kind == "suppressor":
        # z is shielded (x and y are marginally dependent), and the effect only appears,
        # with a flipped sign, once z is controlled. A confounder masking a real effect
        # and a collider fabricating one are indistinguishable here, so the controlled
        # effect is not certifiable from the data alone.
        z = noise()
        x = [zi + ni for zi, ni in zip(z, noise())]
        y = [0.6 * xi - 1.5 * zi + ni for xi, zi, ni in zip(x, z, noise(0.5))]
        return _rows(x=x, y=y, z=z), {"controls": ["z"]}
    if kind == "outlier":
        x, y = noise(), noise()
        for i in rng.sample(range(N), 7):
            x[i] += 6
            y[i] += 6
        return _rows(x=x, y=y), {}
    if kind == "noise":
        return _rows(x=noise(), y=noise()), {}
    if kind == "latent":
        z = noise()
        x = [zi + ni for zi, ni in zip(z, noise(0.6))]
        y = [1.2 * zi + ni for zi, ni in zip(z, noise(0.6))]  # true x effect is zero
        w = [zi + ni for zi, ni in zip(z, noise(0.6))]
        return _rows(x=x, y=y, nc_w=w), {"negative_controls": ["nc_w"]}
    raise ValueError(kind)


def _verdict(kind: str, seed: int) -> str:
    rows, kw = _world(kind, seed)
    return verify_effect(rows, "x", "y", **kw).verdict


def test_sound_effect_passes() -> None:
    assert _verdict("robust", 0) == "sound"
    assert _verdict("moderate", 0) == "sound"


def test_confounding_is_caught_and_named() -> None:
    rows, kw = _world("confounded", 0)
    report = verify_effect(rows, "x", "y", **kw)
    assert report.verdict == "unsound"
    assert "'z'" in (report.pivotal or "")


def test_binary_treatment_effect_is_not_removed_as_outliers() -> None:
    # a 0/1 treatment's minority level must not be dropped by the IQR outlier check: a
    # binary predictor has no outliers, and removing a whole level would fake a "driven
    # by outliers" failure. The check applies only to continuous columns.
    rng = random.Random(0)
    rows = [
        {"x": str(t := rng.randint(0, 1)), "y": str(2.0 * t + rng.gauss(0, 1))}
        for _ in range(300)
    ]
    report = verify_effect(rows, "x", "y")
    assert report.verdict == "sound" and report.effect > 0


def test_collider_control_is_caught() -> None:
    rows, kw = _world("collider", 0)
    report = verify_effect(rows, "x", "y", **kw)
    assert report.verdict == "unsound"
    assert "collider" in (report.pivotal or "")


def test_collider_and_suppressor_are_distinguished_by_identifiability() -> None:
    # Both flip the sign on adjustment, but only the collider is identifiable from the
    # data (an unshielded v-structure): it is caught as unsound, while the shielded
    # suppressor cannot be classified and is honestly reported inconclusive, never a
    # false "collider" and never wrongly certified.
    rows, kw = _world("suppressor", 0)
    report = verify_effect(rows, "x", "y", **kw)
    assert report.verdict == "inconclusive"
    assert "'z'" in (report.pivotal or "") and "role" in (report.pivotal or "")
    # it presents the ambiguity, never asserts z *is* a collider (the old false claim)
    assert "controlling collider" not in (report.pivotal or "")

    # the identifiable collider, by contrast, is caught outright as unsound
    crows, ckw = _world("collider", 0)
    creport = verify_effect(crows, "x", "y", **ckw)
    assert creport.verdict == "unsound" and "collider" in (creport.pivotal or "")


def test_suppressor_is_never_certified_across_seeds() -> None:
    # The guarantee: an effect contingent on a non-identifiable adjustment is never
    # passed as sound, whatever the sample.
    assert all(_verdict("suppressor", s) == "inconclusive" for s in range(8))


def test_outlier_driven_is_caught() -> None:
    assert _verdict("outlier", 0) == "unsound"


def test_noise_is_inconclusive() -> None:
    # A pure-noise effect is not significant, so the claim does not hold.
    assert _verdict("noise", 0) == "inconclusive"


def test_latent_confounding_caught_by_negative_control() -> None:
    rows, kw = _world("latent", 0)
    report = verify_effect(rows, "x", "y", **kw)
    assert report.verdict == "unsound"
    assert "negative control" in (report.pivotal or "")


def test_stable_across_seeds() -> None:
    assert all(_verdict("robust", s) == "sound" for s in range(8))
    assert all(_verdict("confounded", s) == "unsound" for s in range(8))
    assert all(_verdict("collider", s) == "unsound" for s in range(8))


def test_direction_is_flagged_not_assumed() -> None:
    rows, _ = _world("robust", 0)
    report = verify_effect(rows, "x", "y")
    assert any("direction" in c for c in report.caveats)


def test_reproducible() -> None:
    rows, kw = _world("confounded", 3)
    first = verify_effect(rows, "x", "y", **kw)
    second = verify_effect(rows, "x", "y", **kw)
    assert first.verdict == second.verdict and first.effect == second.effect


# --- comparison (group differences + Simpson's paradox) ------------------------


def _grouped(seed: int, *, gap: float, reversal: bool = False) -> list[dict[str, str]]:
    """Rows of group/value(/segment): a real gap, or a Simpson's reversal by segment."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    if not reversal:
        for _ in range(N):
            rows.append({"group": "a", "value": str(rng.gauss(gap, 1.0))})
            rows.append({"group": "b", "value": str(rng.gauss(0.0, 1.0))})
        return rows
    # a beats b within each segment, but b dominates the easy segment -> a loses overall
    for seg, base in (("easy", 10.0), ("hard", 0.0)):
        na, nb = (40, 260) if seg == "easy" else (260, 40)
        for _ in range(na):
            rows.append(
                {"group": "a", "value": str(rng.gauss(base + 1, 1)), "seg": seg}
            )
        for _ in range(nb):
            rows.append({"group": "b", "value": str(rng.gauss(base, 1)), "seg": seg})
    return rows


def test_real_difference_is_sound() -> None:
    report = verify_comparison(_grouped(0, gap=0.8), "group", "value")
    assert report.verdict == "sound" and report.effect > 0


def test_no_difference_is_inconclusive() -> None:
    report = verify_comparison(_grouped(0, gap=0.0), "group", "value")
    assert report.verdict == "inconclusive"


def test_simpsons_reversal_is_caught() -> None:
    rows = _grouped(0, gap=0.0, reversal=True)
    report = verify_comparison(rows, "group", "value", subgroup="seg")
    assert report.verdict == "unsound"
    assert "Simpson" in (report.pivotal or "") and "seg" in (report.pivotal or "")


def test_simpsons_reversal_caught_without_naming_the_subgroup() -> None:
    # the gate auto-scans categorical columns, so it must find the reversal even when
    # the caller does not point at the offending column (the realistic failure mode)
    rows = _grouped(0, gap=0.0, reversal=True)
    report = verify_comparison(rows, "group", "value")
    assert report.verdict == "unsound"
    assert "Simpson" in (report.pivotal or "") and "seg" in (report.pivotal or "")


# --- spurious regression / non-stationarity (time-indexed data) ----------------


def _random_walks(seed: int, *, related: bool) -> list[dict[str, str]]:
    """Two random walks in row order: independent (spurious) or genuinely related."""
    rng = random.Random(seed)
    x = 0.0
    y = 0.0
    rows: list[dict[str, str]] = []
    for _ in range(500):
        x += rng.gauss(0, 1)
        if related:
            y = 2.0 * x + rng.gauss(0, 1)  # cointegrated: holds after differencing
        else:
            y += rng.gauss(0, 1)  # independent walk: only shared drift
        rows.append({"x": str(round(x, 4)), "y": str(round(y, 4))})
    return rows


def test_spurious_random_walk_correlation_is_caught() -> None:
    # two independent random walks correlate by shared drift; the guard must not
    # certify it (the classic Granger-Newbold spurious regression)
    assert all(
        verify_correlation(_random_walks(s, related=False), "x", "y").verdict != "sound"
        for s in range(8)
    )


def test_spurious_random_walk_effect_is_caught() -> None:
    assert all(
        verify_effect(_random_walks(s, related=False), "x", "y").verdict != "sound"
        for s in range(8)
    )


def test_genuinely_related_trending_series_is_sound() -> None:
    # a real relationship that survives differencing must still pass
    assert all(
        verify_correlation(_random_walks(s, related=True), "x", "y").verdict == "sound"
        for s in range(8)
    )


def test_prediction_refuses_to_certify_time_series_without_time_order() -> None:
    # an autocorrelated target on a cross-sectional split would leak the future, so
    # the prediction gate must refuse to certify it rather than report spurious skill
    def walks(seed: int) -> list[dict[str, str]]:
        rng = random.Random(seed)
        x = 0.0
        y = 0.0
        rows = []
        for t in range(500):
            x += rng.gauss(0, 1)
            y += rng.gauss(0, 1)
            rows.append({"t": str(t), "x": str(round(x, 4)), "y": str(round(y, 4))})
        return rows

    assert all(
        verify_prediction(walks(s), ["t", "x"], "y").verdict != "sound"
        for s in range(6)
    )


# --- correlation and trend -----------------------------------------------------


def test_real_correlation_is_sound() -> None:
    rng = random.Random(0)
    rows = []
    for _ in range(N):
        x = rng.gauss(0, 1)
        rows.append({"x": str(x), "y": str(1.5 * x + rng.gauss(0, 1))})
    report = verify_correlation(rows, "x", "y")
    assert report.verdict == "sound" and report.effect > 0


def test_confounded_correlation_is_flagged() -> None:
    rng = random.Random(0)
    rows = []
    for _ in range(N):
        z = rng.gauss(0, 1)
        rows.append(
            {
                "x": str(z + rng.gauss(0, 0.3)),
                "y": str(z + rng.gauss(0, 0.3)),
                "z": str(z),
            }
        )
    report = verify_correlation(rows, "x", "y")
    assert report.verdict == "unsound" and "z" in (report.pivotal or "")


def test_real_trend_is_sound() -> None:
    rng = random.Random(0)
    rows = [{"t": str(i), "v": str(0.5 * i + rng.gauss(0, 3))} for i in range(N)]
    report = verify_trend(rows, "t", "v")
    assert report.verdict == "sound" and report.effect > 0


def test_flat_series_has_no_trend() -> None:
    rng = random.Random(0)
    rows = [{"t": str(i), "v": str(rng.gauss(0, 1))} for i in range(N)]
    report = verify_trend(rows, "t", "v")
    assert report.verdict == "inconclusive"


def test_a_trend_over_ten_periods_is_answerable() -> None:
    """A point is a period, not an observation.

    "Is retention degrading cohort over cohort?" arrives as one number per cohort, so a
    30-point floor would refuse two and a half years of monthly cohorts however plain
    the decline. These are ten real month-3 retention figures, sliding from 53% to 26%.
    """
    real = [0.5275, 0.4725, 0.425, 0.375, 0.3675, 0.345, 0.2975, 0.25, 0.27, 0.255]
    rows = [{"t": str(i), "v": str(v)} for i, v in enumerate(real)]
    report = verify_trend(rows, "t", "v")
    assert report.verdict == "sound" and report.effect < 0
    assert all(c.survived for c in report.checks), [c.detail for c in report.checks]
    # and nine is still too few, with the count said out loud
    fewer = verify_trend(rows[:9], "t", "v")
    assert fewer.verdict == "inconclusive"
    assert "9" in " ".join(fewer.caveats)


def test_ten_periods_of_noise_are_not_a_trend() -> None:
    """The floor moved, so the guard against a short lucky run has to be explicit.

    Significance uses the slope's own degrees of freedom rather than a fixed 1.96 -- at
    ten periods the critical value is 2.31 -- and the autocorrelation, endpoint and
    outlier checks all still have to pass.
    """
    certified = 0
    for seed in range(40):
        rng = random.Random(seed)
        rows = [{"t": str(i), "v": str(rng.gauss(0.4, 0.05))} for i in range(10)]
        certified += verify_trend(rows, "t", "v").verdict == "sound"
    assert certified <= 2, f"{certified}/40 noise series certified as a trend"


# --- stability batteries: each gate holds across seeds, not on one lucky draw ---


def _assoc_rows(seed: int, *, confounded: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    if not confounded:
        x = [rng.gauss(0, 1) for _ in range(N)]
        return _rows(x=x, y=[1.5 * xi + rng.gauss(0, 1) for xi in x])
    z = [rng.gauss(0, 1) for _ in range(N)]  # z drives both -> spurious x~y
    return _rows(
        x=[zi + rng.gauss(0, 0.3) for zi in z],
        y=[zi + rng.gauss(0, 0.3) for zi in z],
        z=z,
    )


def _series_rows(seed: int, *, trending: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    slope = 0.5 if trending else 0.0
    return _rows(
        t=[float(i) for i in range(N)],
        v=[slope * i + rng.gauss(0, 3) for i in range(N)],
    )


def test_comparison_stable_across_seeds() -> None:
    assert all(
        verify_comparison(_grouped(s, gap=0.8), "group", "value").verdict == "sound"
        for s in range(8)
    )
    assert all(
        verify_comparison(
            _grouped(s, gap=0.0, reversal=True), "group", "value", subgroup="seg"
        ).verdict
        == "unsound"
        for s in range(8)
    )


def test_correlation_stable_across_seeds() -> None:
    # Real association is reliably sound; a confounded one is never passed as sound (it
    # comes back unsound, or inconclusive at the margin, never a false "sound").
    assert all(
        verify_correlation(_assoc_rows(s, confounded=False), "x", "y").verdict
        == "sound"
        for s in range(8)
    )
    assert all(
        verify_correlation(_assoc_rows(s, confounded=True), "x", "y").verdict != "sound"
        for s in range(8)
    )


def test_trend_stable_across_seeds() -> None:
    # Real trend is reliably sound; pure noise is never passed as a sound trend.
    assert all(
        verify_trend(_series_rows(s, trending=True), "t", "v").verdict == "sound"
        for s in range(8)
    )
    assert all(
        verify_trend(_series_rows(s, trending=False), "t", "v").verdict != "sound"
        for s in range(8)
    )


# --- regression (OLS coefficient diagnostics) ----------------------------------


def _reg_clean(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(400):
        x, c = rng.gauss(0, 1), rng.gauss(0, 1)
        rows.append(
            {"x": str(x), "c": str(c), "y": str(0.8 * x + 0.3 * c + rng.gauss(0, 1))}
        )
    return rows


def _reg_nonlinear(seed: int) -> list[dict[str, str]]:
    # a real positive slope, but the true form is curved -> the linear coefficient
    # is a biased summary, which the RESET specification check must catch
    rng = random.Random(seed)
    rows = []
    for _ in range(400):
        x = rng.uniform(0, 3)
        rows.append({"x": str(x), "y": str(x + 0.8 * x * x + rng.gauss(0, 1))})
    return rows


def _reg_recoverable_collinear(seed: int) -> list[dict[str, str]]:
    # x1 carries the real effect and x2 is a noisy copy: collinear (high VIF) but the
    # coefficient is still recoverable, so it should be certified sound, not rejected
    rng = random.Random(seed)
    rows = []
    for _ in range(400):
        x1 = rng.gauss(0, 1)
        x2 = x1 + rng.gauss(0, 0.25)
        rows.append(
            {"x1": str(x1), "x2": str(x2), "y": str(3.0 * x1 + rng.gauss(0, 1))}
        )
    return rows


def _reg_collinear_proxy(seed: int) -> list[dict[str, str]]:
    # x1 is a near-duplicate of x2 with no effect of its own (y depends on x2): the
    # coefficient is not separable, so the gate must not certify it as sound
    rng = random.Random(seed)
    rows = []
    for _ in range(400):
        x2 = rng.gauss(0, 1)
        x1 = x2 + rng.gauss(0, 0.1)
        rows.append(
            {"x1": str(x1), "x2": str(x2), "y": str(2.0 * x2 + rng.gauss(0, 1))}
        )
    return rows


def test_clean_regression_is_sound() -> None:
    report = verify_regression(_reg_clean(0), "y", "x", controls=["c"])
    assert report.verdict == "sound" and report.effect > 0


def test_nonlinear_regression_is_flagged() -> None:
    # a misspecified linear form must not certify (the coefficient is biased), but it is
    # not evidence against a relationship, so it is inconclusive rather than unsound: a
    # nonlinear effect must not be vetoed by the linear coefficient being wrong
    report = verify_regression(_reg_nonlinear(0), "y", "x")
    assert report.verdict == "inconclusive"
    assert "specifica" in (report.pivotal or "").lower()


def test_recoverable_collinear_coefficient_is_sound() -> None:
    # multicollinearity inflates variance but does not bias a recoverable estimate,
    # so a precisely estimated coefficient is certified despite a high VIF
    report = verify_regression(
        _reg_recoverable_collinear(0), "y", "x1", controls=["x2"]
    )
    assert report.verdict == "sound"


def test_collinear_proxy_coefficient_is_not_certified() -> None:
    # a coefficient that collinearity makes inseparable must never come back sound
    report = verify_regression(_reg_collinear_proxy(0), "y", "x1", controls=["x2"])
    assert report.verdict != "sound"


def test_regression_stable_across_seeds() -> None:
    assert all(
        verify_regression(_reg_clean(s), "y", "x", controls=["c"]).verdict == "sound"
        for s in range(8)
    )
    assert all(
        verify_regression(_reg_nonlinear(s), "y", "x").verdict == "inconclusive"
        for s in range(8)
    )


# --- prediction (leakage-free re-evaluation) -----------------------------------


def _pred_clean(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(400):
        f1, f2 = rng.gauss(0, 1), rng.gauss(0, 1)
        y = 1 if (0.9 * f1 + 0.5 * f2 + rng.gauss(0, 1)) > 0 else 0
        rows.append({"f1": str(f1), "f2": str(f2), "y": str(y)})
    return rows


def _pred_noise(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return [
        {
            "f1": str(rng.gauss(0, 1)),
            "f2": str(rng.gauss(0, 1)),
            "y": str(rng.randint(0, 1)),
        }
        for _ in range(400)
    ]


def _pred_leak(seed: int) -> list[dict[str, str]]:
    # 'leak' is the target plus a whisker of noise: a single feature that predicts
    # the target almost perfectly, the signature of a leaked proxy
    rng = random.Random(seed)
    rows = []
    for _ in range(400):
        f1 = rng.gauss(0, 1)
        y = 1 if (f1 + rng.gauss(0, 1)) > 0 else 0
        rows.append(
            {"f1": str(f1), "leak": str(float(y) + rng.gauss(0, 0.001)), "y": str(y)}
        )
    return rows


def test_real_prediction_is_sound() -> None:
    report = verify_prediction(_pred_clean(0), ["f1", "f2"], "y")
    assert report.verdict == "sound" and report.effect > 0


def test_leaked_proxy_is_caught() -> None:
    report = verify_prediction(_pred_leak(0), ["f1", "leak"], "y")
    assert report.verdict == "unsound" and "leak" in (report.pivotal or "").lower()


def test_prediction_stable_across_seeds() -> None:
    # Real signal is reliably sound; pure noise is never passed as predictive skill.
    assert all(
        verify_prediction(_pred_clean(s), ["f1", "f2"], "y").verdict == "sound"
        for s in range(8)
    )
    assert all(
        verify_prediction(_pred_noise(s), ["f1", "f2"], "y").verdict != "sound"
        for s in range(12)
    )


def test_prediction_robust_to_collinear_features() -> None:
    # Features that include a part and its exact sum are a rank-deficient design. A bare
    # OLS explodes out of sample and reports "no skill" on data with plenty; the ridge-
    # stabilized fit recovers the honest held-out R², so real signal is not rejected.
    rng = random.Random(0)
    rows = []
    for _ in range(400):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        y = 3 * a + 2 * b + rng.gauss(0, 0.5)
        rows.append(
            {"a": str(a), "b": str(b), "ab_sum": str(a + b), "y": str(round(y, 4))}
        )
    report = verify_prediction(rows, ["a", "b", "ab_sum"], "y")  # ab_sum == a + b
    assert report.verdict == "sound"
    assert report.effect > 0.5  # an honest R², not a negative explosion


def test_multi_feature_reconstruction_is_flagged() -> None:
    # target = a - b: neither feature alone predicts it (so the single-feature leakage
    # screen passes), yet the two together reconstruct it and the held-out fit is
    # near-perfect. The verdict is sound (it is predictive on this data), but carries a
    # caveat that near-perfect skill may be multi-feature leakage the screen cannot see.
    rng = random.Random(0)
    rows = []
    for _ in range(200):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        rows.append({"a": str(a), "b": str(b), "y": str(a - b)})
    report = verify_prediction(rows, ["a", "b"], "y")
    assert report.verdict == "sound"
    assert any("near-perfect" in c for c in report.caveats)


def test_predictive_claim_not_rescued_by_a_screen() -> None:
    # For a {target, features} claim the prediction gate IS the claim. When it finds
    # no held-out skill (noise), a sound leakage screen must NOT rescue the whole to
    # "sound": the prediction gate is certifying, so its inconclusive result blocks it.
    rng = random.Random(1)
    rows = [
        {
            "f1": str(rng.gauss(0, 1)),
            "f2": str(rng.gauss(0, 1)),
            "y": str(rng.gauss(0, 1)),
        }
        for _ in range(400)
    ]
    assert verify_prediction(rows, ["f1", "f2"], "y").verdict == "inconclusive"
    composite = verify_all(rows, target="y", features=["f1", "f2"])
    assert composite.verdict != "sound"


def _pred_panel(seed: int = 7) -> tuple[list[dict[str, str]], list[str]]:
    # A panel: each entity has a random feature centroid and a random target level that
    # is uncorrelated with the centroid, so there is no genuine cross-entity skill. With
    # more features than entities the ridge fit uses them as entity indicators, so a
    # cross-sectional split (an entity's rows on both sides) memorizes each entity's
    # level and scores high, while holding out whole entities exposes that there is
    # nothing to generalize.
    rng = random.Random(seed)
    entities, k, per, noise = 15, 12, 30, 0.15
    centroids = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(entities)]
    level = [rng.gauss(0, 1) for _ in range(entities)]
    feats = [f"f{j}" for j in range(k)]
    rows: list[dict[str, str]] = []
    for e in range(entities):
        for _ in range(per):
            r = {"entity": f"ent_{e}", "y": str(level[e] + rng.gauss(0, noise))}
            for j in range(k):
                r[f"f{j}"] = str(centroids[e][j] + rng.gauss(0, noise))
            rows.append(r)
    rng.shuffle(rows)
    return rows, feats


def test_entity_split_rejects_panel_memorization() -> None:
    # A cross-sectional split lets the same entity sit on both sides, so the model
    # memorizes each entity's level and looks skillful. Holding out whole entities (as
    # the caller asks by naming a group) shows there is no cross-entity skill.
    rows, feats = _pred_panel()
    assert verify_prediction(rows, feats, "y").verdict == "sound"
    assert verify_prediction(rows, feats, "y", group="entity").verdict != "sound"


def test_verify_all_forwards_entity_role_to_prediction() -> None:
    # The certifying prediction gate must honor a declared entity role; otherwise a
    # panel with no cross-entity skill certifies "sound" on a leaky cross-sectional
    # split. Reverting the catalog's group/time forwarding makes `honest` "sound" too,
    # so this asserts the wiring, not just the gate.
    rows, feats = _pred_panel()
    assert verify_all(rows, target="y", features=feats).verdict == "sound"
    honest = verify_all(rows, target="y", features=feats, group="entity")
    assert honest.verdict != "sound"


def test_verify_all_survives_a_gate_that_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A gate that raises on degenerate or adversarial data must not crash the composite;
    # it becomes an inconclusive result. Without the guard the exception would propagate
    # out of verify_all, so this asserts the composite is robust, not just the verdict.
    from elbi_core.verification import selection

    def boom(rows: Any, claim: Any) -> Any:
        raise ZeroDivisionError("degenerate")

    raising = selection._Check("boom", lambda rows, claim: "applies", boom)
    monkeypatch.setattr(selection, "_CATALOG", (raising, *selection._CATALOG))
    rows = [{"x": str(i), "y": str(2 * i)} for i in range(60)]
    report = verify_all(rows, x="x", y="y")  # must not raise
    verdicts = {g.name: g.verdict for g in report.ran}
    assert verdicts.get("boom") == "inconclusive"


# --- hypothesis-test rigor folded into the comparison gate ---------------------


def test_small_nonnormal_difference_needs_the_rank_test() -> None:
    # heavy-tailed groups with the same center; a few extreme draws can make the
    # parametric t significant, but the assumption-free rank test must not be fooled
    rng = random.Random(0)
    a = [rng.lognormvariate(0, 1) for _ in range(20)]
    b = [rng.lognormvariate(0, 1) for _ in range(20)]
    a[0] += 50  # an extreme point that drags the parametric mean
    report = verify_comparison(_grouped_raw(a, b), "g", "v")
    assert report.verdict != "sound"  # never certified on this evidence


def test_difference_must_survive_multiple_comparisons() -> None:
    # a barely-significant difference that is one of fifty tests should not pass
    rng = random.Random(2)
    a = [rng.gauss(0.25, 1.0) for _ in range(60)]
    b = [rng.gauss(0.0, 1.0) for _ in range(60)]
    lone = verify_comparison(_grouped_raw(a, b), "g", "v")
    corrected = verify_comparison(_grouped_raw(a, b), "g", "v", family_size=50)
    # whatever the lone verdict, correcting for 50 tests cannot leave it sound
    assert not (lone.verdict == "sound" and corrected.verdict == "sound")
    assert corrected.verdict != "sound"


def _grouped_raw(a: list[float], b: list[float]) -> list[dict[str, str]]:
    return [{"g": "a", "v": str(v)} for v in a] + [{"g": "b", "v": str(v)} for v in b]


# --- real trap datasets from the literature (ground-truth conclusions) ---------


def test_kidney_stone_simpsons_paradox() -> None:
    # Charig 1986: treatment A wins within both stone-size strata, yet loses overall
    # because it took the harder cases. The naive overall comparison is confounded.
    rows: list[dict[str, str]] = []
    # the published Charig proportions, scaled up so the (real) overall gap clears
    # significance and the within-stratum reversal can be exercised
    cells = {
        ("A", "small"): (81, 87),
        ("B", "small"): (234, 270),
        ("A", "large"): (192, 263),
        ("B", "large"): (55, 80),
    }
    for (treat, size), (success, total) in cells.items():
        for i in range(total * 4):
            cured = 1 if i % total < success else 0
            rows.append({"treat": treat, "size": size, "cured": str(cured)})
    report = verify_comparison(rows, "treat", "cured", subgroup="size")
    assert report.verdict == "unsound"
    assert "Simpson" in (report.pivotal or "")


def test_credit_card_target_leakage() -> None:
    # AER CreditCard: 'expenditure' and 'share' are consequences of holding the card,
    # not predictors of approval; either alone predicts the target almost perfectly.
    rng = random.Random(0)
    rows = []
    for _ in range(500):
        approved = 1 if rng.random() < 0.8 else 0
        income = rng.gauss(3.0, 1.0)
        # spending exists only once you hold the card: a consequence of the target,
        # exactly zero for everyone who was not approved (the real leakage)
        expenditure = rng.expovariate(1.0) if approved else 0.0
        rows.append(
            {
                "income": str(income),
                "expenditure": str(expenditure),
                "approved": str(approved),
            }
        )
    report = verify_prediction(rows, ["income", "expenditure"], "approved")
    assert report.verdict == "unsound"
    assert "expenditure" in (report.pivotal or "")
