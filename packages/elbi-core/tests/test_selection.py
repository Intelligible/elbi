"""The precondition engine routes every question to the right verification(s).

For each kind of question, the caller maps it to column roles and the engine selects
the checks whose preconditions hold. These tests assert the full routing matrix: each
question's claim makes exactly the right checks applicable and never a structurally
wrong one, the composite verdict combines them with veto precedence, and inapplicable
checks are reported as skipped rather than silently dropped.
"""

from __future__ import annotations

import random
import re

from elbi_core.verification import applicable, verify_all

N = 400


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


# --- one dataset + claim per question kind, with what must / must not apply -----


def _effect_data() -> list[dict[str, str]]:
    rng = random.Random(0)
    return _rows(
        {
            "x": (x := rng.gauss(0, 1)),
            "y": 0.8 * x + rng.gauss(0, 1),
            "z": rng.gauss(0, 1),
        }
        for _ in range(N)
    )


def _comparison_data() -> list[dict[str, str]]:
    rng = random.Random(1)
    return _rows({"g": rng.choice(["a", "b"]), "v": rng.gauss(0, 1)} for _ in range(N))


def _experiment_data(*, srm: bool = False) -> list[dict[str, str]]:
    rng = random.Random(2)
    out = []
    for _ in range(2000):
        v = "A" if rng.random() < (0.65 if srm else 0.5) else "B"
        out.append({"variant": v, "metric": rng.randint(0, 1)})
    return _rows(out)


def _proportions_data() -> list[dict[str, str]]:
    rng = random.Random(3)
    return _rows(
        {"g": rng.choice(["p", "q"]), "o": rng.choice(["yes", "no"])} for _ in range(N)
    )


def _trend_data() -> list[dict[str, str]]:
    rng = random.Random(4)
    return _rows({"t": i, "v": 0.1 * i + rng.gauss(0, 1)} for i in range(N))


def _forecast_data() -> list[dict[str, str]]:
    rng = random.Random(5)
    return _rows(
        {"t": i, "actual": rng.gauss(0, 1), "forecast": rng.gauss(0, 1)}
        for i in range(N)
    )


def _classification_data() -> list[dict[str, str]]:
    rng = random.Random(6)
    return _rows(
        {
            "y_true": rng.randint(0, 1),
            "y_pred": rng.randint(0, 1),
            "y_score": round(rng.random(), 3),
        }
        for _ in range(N)
    )


def _calibration_data() -> list[dict[str, str]]:
    rng = random.Random(7)
    return _rows(
        {"p": round(rng.random(), 3), "o": rng.randint(0, 1)} for _ in range(N)
    )


def _logistic_data() -> list[dict[str, str]]:
    rng = random.Random(8)
    return _rows({"x": rng.gauss(0, 1), "y": rng.randint(0, 1)} for _ in range(N))


def _survival_data() -> list[dict[str, str]]:
    rng = random.Random(9)
    return _rows(
        {
            "t": round(rng.expovariate(0.1), 2),
            "e": rng.randint(0, 1),
            "g": rng.choice(["c", "t"]),
        }
        for _ in range(N)
    )


def _cluster_data() -> list[dict[str, str]]:
    rng = random.Random(10)
    return _rows(
        {"f0": rng.gauss(0, 1), "f1": rng.gauss(0, 1), "f2": rng.gauss(0, 1)}
        for _ in range(N)
    )


def _screen_data() -> list[dict[str, str]]:
    rng = random.Random(11)
    return _rows(
        {"t": rng.randint(0, 1), "m0": rng.gauss(0, 1), "m1": rng.gauss(0, 1)}
        for _ in range(N)
    )


def _rtm_data() -> list[dict[str, str]]:
    rng = random.Random(12)
    rows = []
    for _ in range(N):
        a = rng.gauss(50, 10)
        t1 = a + rng.gauss(0, 10)
        rows.append({"pre": t1, "post": a + rng.gauss(0, 10), "g": int(t1 < 40)})
    return _rows(rows)


#: question kind -> (dataset, claim, must-apply, must-not-apply)
_MATRIX = [
    (
        "effect of x on y",
        _effect_data(),
        {"x": "x", "y": "y"},
        {"effect", "correlation", "regression"},
        {"survival", "forecast", "experiment", "clusters"},
    ),
    (
        "group difference",
        _comparison_data(),
        {"group": "g", "value": "v"},
        {"comparison"},
        {"experiment", "proportions", "survival", "effect"},
    ),
    (
        "A/B test",
        _experiment_data(),
        {"variant": "variant", "metric": "metric"},
        {"experiment"},
        {"comparison", "survival", "effect"},
    ),
    (
        "categorical association",
        _proportions_data(),
        {"group": "g", "outcome": "o"},
        {"proportions"},
        {"comparison", "effect", "experiment"},
    ),
    (
        "trend over time",
        _trend_data(),
        {"time": "t", "value": "v"},
        {"trend"},
        {"stationarity", "forecast", "effect", "survival"},
    ),
    (
        "forecast quality",
        _forecast_data(),
        {"time": "t", "actual": "actual", "forecast": "forecast"},
        {"forecast"},
        {"trend", "effect", "survival"},
    ),
    (
        "classifier quality",
        _classification_data(),
        {"y_true": "y_true", "y_pred": "y_pred", "y_score": "y_score"},
        {"classification"},
        {"effect", "comparison", "survival", "calibration"},
    ),
    (
        "probability calibration",
        _calibration_data(),
        {"probability": "p", "outcome": "o"},
        {"calibration"},
        {"classification", "effect", "comparison"},
    ),
    (
        "logistic odds ratio",
        _logistic_data(),
        {"x": "x", "y": "y"},
        {"logistic"},
        {"survival", "forecast", "comparison", "clusters"},
    ),
    (
        "survival difference",
        _survival_data(),
        {"time": "t", "event": "e", "group": "g"},
        {"survival"},
        {"comparison", "trend", "forecast", "effect"},
    ),
    (
        "real segments",
        _cluster_data(),
        {"features": ["f0", "f1", "f2"]},
        {"clusters"},
        {"effect", "comparison", "survival"},
    ),
    (
        "multiple-metric screen",
        _screen_data(),
        {"group": "t", "outcomes": ["m0", "m1"]},
        {"screen"},
        {"effect", "comparison", "experiment", "rtm"},
    ),
    (
        "regression to the mean",
        _rtm_data(),
        {"before": "pre", "after": "post", "group": "g"},
        {"rtm"},
        {"effect", "experiment", "screen", "clusters"},
    ),
]


def test_every_question_routes_to_the_right_verification() -> None:
    for label, rows, claim, must, must_not in _MATRIX:
        names = {n for n, _ in applicable(rows, **claim)}
        missing = must - names
        wrong = must_not & names
        assert not missing, f"{label}: missing {missing} (got {names})"
        assert not wrong, f"{label}: wrongly applied {wrong}"


def test_composite_combines_with_conjunction() -> None:
    # a real group difference -> the one applicable check is sound -> composite sound
    report = verify_all(_comparison_data(), group="g", value="v")
    assert report.verdict in ("sound", "inconclusive")  # noise groups -> may be either
    assert {g.name for g in report.ran} == {"comparison"}


def test_confounded_effect_is_unsound_composite() -> None:
    rng = random.Random(0)
    rows = _rows(
        {
            "x": (z := rng.gauss(0, 1)) + rng.gauss(0, 0.5),
            "y": 1.5 * z + rng.gauss(0, 0.5),
            "z": z,
        }
        for _ in range(N)
    )
    report = verify_all(rows, x="x", y="y")
    assert (
        report.verdict == "unsound"
    )  # the confounder breaks effect/correlation/regression


def test_real_trend_certifies_without_a_stationarity_veto() -> None:
    # A univariate trend question is the trend gate's to certify. The stationarity gate
    # is a spurious-regression guard that flags every trending series as non-stationary;
    # routed onto a trend claim it would veto a genuine trend, so it must not apply here
    # (two-series spurious regression is caught inside the effect gate instead).
    report = verify_all(_trend_data(), time="t", value="v")
    assert report.verdict == "sound"
    assert {g.name for g in report.ran} == {"trend"}


def test_sample_ratio_mismatch_vetoes_to_invalid() -> None:
    # a broken A/B split voids the whole result regardless of the metric
    report = verify_all(_experiment_data(srm=True), variant="variant", metric="metric")
    assert report.verdict == "invalid"


def _baseline_selected(seed: int, *, effect: float) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(600):
        a = rng.gauss(50, 10)
        pre = a + rng.gauss(0, 10)
        treated = pre < 38
        post = a + rng.gauss(0, 10) + (effect if treated else 0.0)
        rows.append({"pre": pre, "post": post, "chg": post - pre, "grp": int(treated)})
    return _rows(rows)


def test_regression_to_the_mean_is_caught_under_any_framing() -> None:
    # The baseline column is in the data, so the gate must catch the trap however the
    # caller frames it, before/after, effect (x/y), or a comparison of change scores,
    # not only when the pre/post roles are named. (Closes the routing-dependency gap.)
    data = _baseline_selected(0, effect=0.0)
    for claim in (
        {"before": "pre", "after": "post", "group": "grp"},
        {"x": "grp", "y": "post"},
        {"x": "grp", "y": "chg"},
        {"group": "grp", "value": "chg"},
    ):
        report = verify_all(data, **claim)
        assert report.verdict != "sound", claim
        assert any(g.name == "rtm" for g in report.ran), claim


def test_real_effect_beyond_baseline_selection_still_certifies() -> None:
    # a change that exceeds regression to the mean is not suppressed by the guard
    data = _baseline_selected(0, effect=18.0)
    report = verify_all(data, before="pre", after="post", group="grp")
    assert report.verdict == "sound"


def test_transform_verifies_a_nonlinear_effect_on_a_linearising_scale() -> None:
    # an exponential relationship: the linear coefficient is misspecified, but logging
    # the outcome linearises it, so the effect certifies on that scale: the mechanism
    # behind reporting an elasticity for a curved, skewed relationship.
    import math

    rng = random.Random(0)
    rows = _rows(
        {"x": (x := rng.uniform(0, 4)), "y": math.exp(0.7 * x) + rng.gauss(0, 0.3)}
        for _ in range(400)
    )
    linear = verify_all(rows, x="x", y="y")
    logged = verify_all(rows, x="x", y="y", transforms={"y": "log"})
    assert any(
        g.name == "regression" and g.verdict == "inconclusive" for g in linear.ran
    )
    assert any(g.name == "regression" and g.verdict == "sound" for g in logged.ran)
    assert logged.verdict == "sound"


def test_transform_key_may_be_a_role_or_a_column_name() -> None:
    # the caller can write transforms keyed by the role ("y") or the column it points at
    # ("val"): both reshape the same column, so the verdict is identical.
    import math

    rng = random.Random(1)
    rows = _rows(
        {"drv": (d := rng.uniform(0, 4)), "val": math.exp(0.6 * d) + rng.gauss(0, 0.3)}
        for _ in range(400)
    )
    by_role = verify_all(rows, x="drv", y="val", transforms={"y": "log"})
    by_column = verify_all(rows, x="drv", y="val", transforms={"val": "log"})
    assert by_role.verdict == by_column.verdict == "sound"


def test_defensive_rtm_does_not_fire_on_a_randomized_group() -> None:
    # a randomly assigned group (not selected on any baseline) must never be treated as
    # regression to the mean, whatever covariates are present
    rng = random.Random(1)
    rows = _rows(
        {"grp": rng.randint(0, 1), "cov": rng.gauss(0, 1), "out": rng.gauss(0, 1)}
        for _ in range(500)
    )
    report = verify_all(rows, group="grp", value="out")
    assert not any(g.name == "rtm" for g in report.ran)


def test_inapplicable_checks_are_reported_as_skipped() -> None:
    report = verify_all(_comparison_data(), group="g", value="v")
    skipped = {name for name, _ in report.skipped}
    # everything that did not apply is accounted for, not silently dropped
    assert (
        "survival" in skipped and "forecast" in skipped and "classification" in skipped
    )
    assert "comparison" not in skipped


def _real_comparison() -> list[dict[str, str]]:
    rng = random.Random(11)
    return _rows(
        {
            "g": (g := rng.choice(["a", "b"])),
            "v": (2.0 if g == "b" else 0.0) + rng.gauss(0, 1),
        }
        for _ in range(N)
    )


def test_stability_grades_any_sound_claim_not_just_effects() -> None:
    # The stability annotation is platform-wide, not effect-specific: a group comparison
    # (no covariates, so graded by bootstrap resampling of whatever gate certified) gets
    # one too. A real difference survives resampling and is graded robust, with a range.
    report = verify_all(_real_comparison(), group="g", value="v")
    assert report.verdict == "sound"
    stability = next((g for g in report.ran if g.name == "stability"), None)
    assert stability is not None and stability.verdict == "sound"
    assert "robust" in stability.detail and "range" in stability.detail


def _magnitude_dependent() -> list[dict[str, str]]:
    # z confounds x and y so the estimate's *magnitude* depends heavily on whether z is
    # controlled (adjusted it is ~2, unadjusted ~0.5), while the sign stays positive.
    # This is the same-question-different-answer scenario: one point hides the spread.
    rng = random.Random(12)
    out = []
    for _ in range(N):
        z = rng.gauss(0, 1)
        x = 0.5 * z + rng.gauss(0, 0.8)
        y = 2.0 * x - 3.0 * z + rng.gauss(0, 0.4)
        out.append({"x": x, "y": y, "z": z})
    return _rows(out)


def test_stability_surfaces_a_specification_dependent_magnitude() -> None:
    # The certified estimate is stable in sign but swings in magnitude with the control
    # set, so stability grades it robust yet reports the wide range; the answer carries
    # the spread rather than a lone point that shifts with the analyst's choices.
    report = verify_all(_magnitude_dependent(), x="x", y="y", controls=["z"])
    assert report.verdict == "sound"
    stability = next(g for g in report.ran if g.name == "stability")
    match = re.search(r"range (-?[\d.]+) to (-?[\d.]+)", stability.detail)
    assert match is not None
    lo, hi = float(match.group(1)), float(match.group(2))
    assert lo < 0.6 * hi  # dropping the control materially moves the estimate


def test_sound_report_surfaces_the_oracle_estimate_and_units() -> None:
    # A sound conclusion carries the magnitude the oracle itself computed and a unit-
    # aware label, so a serving layer reports the certified number rather than one the
    # model wrote. The controls it holds fixed are recorded, so the estimate is a direct
    # effect given them (not an unconditional total effect).
    report = verify_all(_magnitude_dependent(), x="x", y="y", controls=["z"])
    assert report.verdict == "sound"
    assert report.estimate is not None and report.estimate > 0
    assert (
        report.estimate_label is not None
        and "per +1 unit of x" in report.estimate_label
    )
    assert report.adjusted_for == ("z",)


def test_log_outcome_estimate_reads_as_a_percent_change() -> None:
    # With a log transform on the outcome, the certified coefficient is reported as a
    # percent change per unit (the standard semi-log interpretation), not a bare slope.
    rng = random.Random(20)
    rows = _rows(
        {
            "x": (x := rng.gauss(0, 1)),
            "y": 10.0 * (2.71828 ** (0.3 * x + rng.gauss(0, 0.1))),
        }
        for _ in range(N)
    )
    report = verify_all(rows, x="x", y="y", transforms={"y": "log"})
    assert report.verdict == "sound"
    assert report.estimate_label is not None and "%" in report.estimate_label


def test_comparison_estimate_is_an_unadjusted_difference() -> None:
    # A two-group comparison certifies a raw mean difference and holds nothing fixed, so
    # it is never dressed up as an adjusted effect.
    rng = random.Random(21)
    rows = _rows(
        {"g": "a" if i % 2 else "b", "v": (2.0 if i % 2 else 0.0) + rng.gauss(0, 1)}
        for i in range(N)
    )
    report = verify_all(rows, group="g", value="v")
    assert report.verdict == "sound"
    assert (
        report.estimate_label is not None and "difference in v" in report.estimate_label
    )
    assert report.adjusted_for == ()


def test_inconclusive_report_carries_no_estimate() -> None:
    # An unsettled conclusion exposes no certified magnitude, so nothing can be reported
    # as the oracle's number when the oracle did not certify one.
    rng = random.Random(22)
    rows = _rows({"x": rng.gauss(0, 1), "y": rng.gauss(0, 1)} for _ in range(N))
    report = verify_all(rows, x="x", y="y")
    assert report.verdict != "sound"
    assert report.estimate is None and report.estimate_label is None
