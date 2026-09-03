"""Validation of the advanced verification gates at the behaviour boundary.

Each gate is checked both ways: it certifies a genuinely sound case and catches its
characteristic silent failure, across seeds rather than on one lucky draw.
"""

from __future__ import annotations

import math
import random

from elbi_core.verification import (
    verify_calibration,
    verify_classification,
    verify_clusters,
    verify_experiment,
    verify_forecast,
    verify_logistic,
    verify_proportions,
    verify_rtm,
    verify_screen,
    verify_survival,
)


def _rows(**cols: list[object]) -> list[dict[str, str]]:
    n = len(next(iter(cols.values())))
    return [{k: str(cols[k][i]) for k in cols} for i in range(n)]


# --- A/B experiment (sample-ratio mismatch) ------------------------------------


def _ab(seed: int, *, srm: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(4000):
        v = "A" if rng.random() < (0.65 if srm else 0.5) else "B"
        conv = 1 if rng.random() < (0.20 if v == "A" else 0.30) else 0
        rows.append({"variant": v, "converted": str(conv)})
    return rows


def test_clean_ab_test_is_sound() -> None:
    report = verify_experiment(_ab(0, srm=False), "variant", "converted")
    assert report.verdict == "sound"


def test_sample_ratio_mismatch_is_caught() -> None:
    report = verify_experiment(_ab(0, srm=True), "variant", "converted")
    assert report.verdict == "unsound" and "sample-ratio" in (report.pivotal or "")


def _ab_simpson(seed: int) -> list[dict[str, str]]:
    # B wins the pooled rate but loses within every segment: A converts better in each
    # of 'easy' and 'hard', yet B is concentrated in the high-converting 'easy' segment.
    rng = random.Random(seed)
    rows = []
    for _ in range(6000):
        if rng.random() < 0.5:
            v, seg = "A", ("hard" if rng.random() < 0.7 else "easy")
        else:
            v, seg = "B", ("easy" if rng.random() < 0.7 else "hard")
        base = 0.5 if seg == "easy" else 0.1
        p = base + (0.10 if v == "A" else 0.0)  # A clearly better within each segment
        conv = int(rng.random() < p)
        rows.append({"variant": v, "segment": seg, "converted": str(conv)})
    return rows


def test_simpsons_paradox_in_ab_is_caught_without_being_named() -> None:
    # The flagship A/B trap: a pooled lift that reverses inside a segment must not
    # certify, and the guard must find the segment on its own (it is not passed in).
    report = verify_experiment(_ab_simpson(0), "variant", "converted")
    assert report.verdict == "unsound"
    assert "Simpson" in (report.pivotal or "") and "segment" in (report.pivotal or "")


# --- novelty / time-heterogeneity ----------------------------------------------


def _ab_over_time(seed: int, *, decaying: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for week in range(1, 9):
        lift = max(0.0, 0.20 * (1 - (week - 1) / 5)) if decaying else 0.08
        for _ in range(500):
            v = "A" if rng.random() < 0.5 else "B"
            p = 0.30 + (lift if v == "B" else 0.0)
            rows.append(
                {"week": str(week), "variant": v, "engaged": str(int(rng.random() < p))}
            )
    return rows


def test_novelty_effect_is_not_certified_as_durable() -> None:
    # A front-loaded lift that fades is a novelty effect, not a durable win.
    report = verify_experiment(
        _ab_over_time(0, decaying=True), "variant", "engaged", period="week"
    )
    assert report.verdict == "inconclusive" and "novelty" in (report.pivotal or "")


def test_durable_lift_over_time_is_sound() -> None:
    report = verify_experiment(
        _ab_over_time(0, decaying=False), "variant", "engaged", period="week"
    )
    assert report.verdict == "sound"


def test_novelty_is_caught_whichever_arm_sorts_first() -> None:
    # the decaying arm must be caught whether it sorts before or after the other; the
    # decay is judged toward zero, not by the sign of the pooled lift (a real miss the
    # fresh-endpoint test surfaced when the decaying arm was named 'new' < 'old')
    base = _ab_over_time(0, decaying=True)  # 'B' is the decaying superior arm
    swapped = [{**r, "variant": {"A": "B", "B": "A"}[r["variant"]]} for r in base]
    for data in (base, swapped):
        report = verify_experiment(data, "variant", "engaged", period="week")
        assert report.verdict == "inconclusive" and "novelty" in (report.pivotal or "")


# --- multiple comparisons (false-discovery-rate) -------------------------------


def _screen_family(seed: int, *, real: int) -> list[dict[str, str]]:
    # a treatment tested against 20 outcomes; the first ``real`` carry a true effect.
    rng = random.Random(seed)
    rows = []
    for _ in range(500):
        t = rng.randint(0, 1)
        row = {"treatment": str(t)}
        for j in range(20):
            shift = 0.5 * t if j < real else 0.0
            row[f"m{j:02d}"] = str(round(rng.gauss(0, 1) + shift, 3))
        rows.append(row)
    return rows


def test_screen_rejects_pure_noise_family() -> None:
    # 20 null metrics: no matter which looks significant, none survive FDR control.
    report = verify_screen(
        _screen_family(0, real=0), "treatment", [f"m{j:02d}" for j in range(20)]
    )
    assert report.verdict == "inconclusive"
    assert "false-discovery-rate" in (report.pivotal or "") or "survives" in (
        report.pivotal or ""
    )


def test_screen_finds_the_real_effects() -> None:
    report = verify_screen(
        _screen_family(0, real=3), "treatment", [f"m{j:02d}" for j in range(20)]
    )
    assert report.verdict == "sound"
    assert report.checks[0].survived


# --- regression to the mean ----------------------------------------------------


def _pre_post(seed: int, *, effect: float, high: bool = False) -> list[dict[str, str]]:
    # a latent ability with two noisy measurements; the extreme scorers are "treated".
    rng = random.Random(seed)
    rows = []
    for _ in range(600):
        a = rng.gauss(50, 10)
        t1, t2 = a + rng.gauss(0, 10), a + rng.gauss(0, 10)
        treated = t1 > 62 if high else t1 < 38
        rows.append(
            {
                "pre": str(round(t1, 1)),
                "post": str(round(t2 + (effect if treated else 0.0), 1)),
                "grp": str(int(treated)),
            }
        )
    return rows


def test_regression_to_the_mean_is_not_a_real_effect() -> None:
    # bottom-selected group improves purely by reverting: must not certify an effect.
    report = verify_rtm(_pre_post(0, effect=0.0), "pre", "post", "grp")
    assert report.verdict == "unsound" and "regression to the mean" in (
        report.pivotal or ""
    )


def test_effect_beyond_regression_to_the_mean_is_sound() -> None:
    report = verify_rtm(_pre_post(0, effect=15.0), "pre", "post", "grp")
    assert report.verdict == "sound"


def test_rtm_is_symmetric_in_selection_direction() -> None:
    # A group selected on a HIGH baseline reverts downward; the gate must handle it the
    # same as low selection: pure reversion is unsound, an effect beyond it is sound, in
    # either direction. (Guards the low-selection-only blind spot of the fixtures.)
    assert verify_rtm(
        _pre_post(0, effect=0.0, high=True), "pre", "post", "grp"
    ).verdict == ("unsound")
    assert verify_rtm(
        _pre_post(0, effect=18.0, high=True), "pre", "post", "grp"
    ).verdict == ("sound")


# --- proportions / contingency -------------------------------------------------


def _table_rows(table: list[list[int]]) -> list[dict[str, str]]:
    rows = []
    for r, row in enumerate(table):
        for c, k in enumerate(row):
            rows += [{"group": f"g{r}", "outcome": f"o{c}"}] * k
    return rows


def test_real_association_is_sound() -> None:
    report = verify_proportions(_table_rows([[40, 5], [5, 40]]), "group", "outcome")
    assert report.verdict == "sound"


def test_sparse_chi_square_inflation_is_caught() -> None:
    # chi-square reads significant only because expected counts are tiny; the exact
    # test does not confirm it, so the association is not established
    report = verify_proportions(_table_rows([[9, 0, 1], [3, 3, 3]]), "group", "outcome")
    assert report.verdict == "unsound"


# --- logistic (separation) -----------------------------------------------------


def _logit(seed: int, *, separated: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(300):
        x = rng.gauss(0, 1)
        if separated:
            y = 1 if x > 0 else 0
        else:
            y = 1 if rng.random() < 1 / (1 + math.exp(-0.9 * x)) else 0
        rows.append({"x": str(round(x, 4)), "y": str(y)})
    return rows


def test_real_logistic_effect_is_sound() -> None:
    report = verify_logistic(_logit(0, separated=False), "x", "y")
    assert report.verdict == "sound"


def test_logistic_separation_is_caught() -> None:
    report = verify_logistic(_logit(0, separated=True), "x", "y")
    assert report.verdict == "unsound" and "separation" in (report.pivotal or "")


# --- classification (accuracy paradox) -----------------------------------------


def test_real_classifier_skill_is_sound() -> None:
    rng = random.Random(0)
    rows = []
    for _ in range(800):
        f = rng.gauss(0, 1)
        y = 1 if rng.random() < 1 / (1 + math.exp(-1.5 * f)) else 0
        score = 1 / (1 + math.exp(-1.5 * f))
        rows.append(
            {
                "y": str(y),
                "pred": str(1 if score > 0.5 else 0),
                "score": str(round(score, 4)),
            }
        )
    report = verify_classification(rows, "y", y_pred="pred", y_score="score")
    assert report.verdict == "sound"


def test_accuracy_paradox_is_caught() -> None:
    rng = random.Random(0)
    rows = []
    for _ in range(2000):
        y = 1 if rng.random() < 0.05 else 0
        rows.append(
            {"y": str(y), "pred": "0", "score": str(round(rng.uniform(0, 0.4), 4))}
        )
    report = verify_classification(rows, "y", y_pred="pred", y_score="score")
    assert report.verdict != "sound" and "majority" in (report.pivotal or "")


def test_a_recall_threshold_on_rare_positives_is_still_skill() -> None:
    """A threshold at the prevalence is the right call, and must not read as no skill.

    At 5% positives the majority baseline is 0.95, so a model that trades accuracy for
    recall scores *below* it while separating the classes well. Judged on raw accuracy
    it is failed for being useful, and -- the tell -- it passes on ``y_score`` alone, so
    reporting the decisions it actually makes is what condemns it.
    """
    rng = random.Random(4)
    rows = []
    for _ in range(2000):
        f = rng.gauss(0, 1)
        y = 1 if rng.random() < 1 / (1 + math.exp(-(f - 3.0))) else 0
        risk = 1 / (1 + math.exp(-(f - 3.0)))
        rows.append(
            {
                "y": str(y),
                # cut at the base rate, not 0.5: the decision a 5% rate asks for
                "pred": str(int(risk >= 0.05)),
                "score": str(round(risk, 5)),
            }
        )
    prevalence = sum(int(r["y"]) for r in rows) / len(rows)
    assert 0.02 < prevalence < 0.10, prevalence
    with_decisions = verify_classification(rows, "y", y_pred="pred", y_score="score")
    scores_only = verify_classification(rows, "y", y_score="score")
    assert scores_only.verdict == "sound"
    assert with_decisions.verdict == "sound", with_decisions.pivotal
    skill = next(c for c in with_decisions.checks if c.name == "skill")
    assert skill.survived and "balanced accuracy" in skill.detail


# --- calibration ---------------------------------------------------------------


def test_calibrated_probabilities_are_sound() -> None:
    rng = random.Random(0)
    rows = []
    for _ in range(2000):
        p = rng.random()
        rows.append({"score": str(round(p, 4)), "y": str(1 if rng.random() < p else 0)})
    assert verify_calibration(rows, "score", "y").verdict == "sound"


def test_miscalibrated_probabilities_are_caught() -> None:
    rng = random.Random(0)
    rows = []
    for _ in range(2000):
        p = rng.random()
        y = 1 if rng.random() < p else 0
        score = min(0.999, max(0.001, 0.5 + (p - 0.5) * 1.8))  # overconfident
        rows.append({"score": str(round(score, 4)), "y": str(y)})
    report = verify_calibration(rows, "score", "y")
    assert report.verdict == "unsound" and "miscalibrated" in (report.pivotal or "")


# --- forecast (MASE) -----------------------------------------------------------


def _series(seed: int, *, skilful: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    base = [
        10 + 5 * math.sin(2 * math.pi * t / 7) + rng.gauss(0, 1) for t in range(140)
    ]
    for t in range(7, 140):
        actual = base[t]
        fc = actual + rng.gauss(0, 0.4) if skilful else base[t - 7] + rng.gauss(0, 3)
        rows.append(
            {
                "t": str(t),
                "actual": str(round(actual, 4)),
                "forecast": str(round(fc, 4)),
            }
        )
    return rows


def test_skilful_forecast_is_sound() -> None:
    report = verify_forecast(
        _series(0, skilful=True), "t", "actual", "forecast", seasonal_period=7
    )
    assert report.verdict == "sound"


def test_forecast_that_loses_to_naive_is_caught() -> None:
    report = verify_forecast(
        _series(0, skilful=False), "t", "actual", "forecast", seasonal_period=7
    )
    assert report.verdict != "sound" and "naive" in (report.pivotal or "")


# --- survival ------------------------------------------------------------------


def _surv(seed: int, *, real_difference: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(600):
        g = rng.choice(["ctrl", "treat"])
        rate = 1 / 14.0 if (g == "treat" and real_difference) else 1 / 8.0
        true_t = rng.expovariate(rate)
        cens = rng.expovariate(1 / 30.0)
        t = min(true_t, cens)
        rows.append(
            {
                "group": g,
                "time": str(round(t, 3)),
                "event": str(1 if true_t <= cens else 0),
            }
        )
    return rows


def test_real_survival_difference_is_sound() -> None:
    report = verify_survival(_surv(0, real_difference=True), "time", "event", "group")
    assert report.verdict == "sound"


def test_no_survival_difference_is_inconclusive() -> None:
    report = verify_survival(_surv(0, real_difference=False), "time", "event", "group")
    assert report.verdict != "sound"


# --- clustering ----------------------------------------------------------------


def _cluster_data(seed: int, *, real: bool) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    centers = [(-6, -6), (6, 6), (-6, 6)] if real else [(0, 0)]
    for _ in range(500):
        cx, cy = rng.choice(centers)
        rows.append(
            {
                "f0": str(round(cx + rng.gauss(0, 1), 4)),
                "f1": str(round(cy + rng.gauss(0, 1), 4)),
                "f2": str(round(rng.gauss(0, 1), 4)),
            }
        )
    return rows


def test_real_clusters_are_sound() -> None:
    report = verify_clusters(_cluster_data(0, real=True), ["f0", "f1", "f2"], k=3)
    assert report.verdict == "sound"


def test_imposed_clusters_are_caught() -> None:
    report = verify_clusters(_cluster_data(0, real=False), ["f0", "f1", "f2"], k=4)
    assert report.verdict == "unsound" and "no real cluster structure" in (
        report.pivotal or ""
    )
