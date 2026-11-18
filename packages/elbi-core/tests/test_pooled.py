"""Validation of the pooled-effect gate against statsmodels and hand-computed
formulas."""

from __future__ import annotations

import math

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from elbi_core.verification import verify_all, verify_pooled_effect
from elbi_core.verification.pooled import _dl

MARKETS = [
    ("Austin", 4.1, 1.24),
    ("Berlin", 1.9, 1.32),
    ("Cairo", 4.3, 1.19),
    ("Denver", 2.2, 1.28),
    ("Edinburgh", -0.3, 1.36),
    ("Fukuoka", 5.1, 1.21),
]
EFFECTS = [e for _, e, _ in MARKETS]
SES = [s for _, _, s in MARKETS]
#: Five small negative effects and one large positive one: random-effects weights are
#: near-equal under the large tau², so the mean is about (6 - 2) / 6 > 0, and without
#: the outlier it is about -0.4.
OUTLIER_STUDIES = [
    *[(f"m{i}", e, 0.3) for i, e in enumerate([-0.5, -0.3, -0.6, -0.4, -0.2])],
    ("Outlier", 6.0, 0.3),
]
#: t_{0.975} on 5 df, scipy.stats.t.isf(0.025, 5) (tables give 2.5706).
T_975_5 = 2.570581835636315
#: t_{0.975} on 4 df, scipy.stats.t.isf(0.025, 4) (tables give 2.7764).
T_975_4 = 2.7764451051977996


def _rows(studies) -> list[dict[str, str]]:
    return [{"market": m, "lift": str(e), "se": str(s)} for m, e, s in studies]


def _check(report, name: str):
    return next(c for c in report.checks if c.name == name)


@pytest.fixture
def ref():
    np = pytest.importorskip("numpy")
    meta = pytest.importorskip("statsmodels.stats.meta_analysis")
    return meta.combine_effects(np.array(EFFECTS), np.array(SES) ** 2, method_re="dl")


def test_dl_matches_statsmodels(ref) -> None:
    # statsmodels does not floor tau² at zero, so the table has Q above its k-1
    # expectation and the two agree only because no floor applies
    assert ref.q > len(MARKETS) - 1
    fit = _dl(EFFECTS, SES)
    assert fit.mu_fe == pytest.approx(ref.mean_effect_fe, abs=1e-9)
    assert fit.se_fe == pytest.approx(ref.sd_eff_w_fe, abs=1e-9)
    assert fit.q == pytest.approx(ref.q, abs=1e-9)
    assert fit.tau2 == pytest.approx(ref.tau2, abs=1e-9)
    assert fit.i2 == pytest.approx(ref.i2, abs=1e-9)
    assert fit.mu == pytest.approx(ref.mean_effect_re, abs=1e-9)
    assert fit.se == pytest.approx(ref.sd_eff_w_re, abs=1e-9)
    report = verify_pooled_effect(_rows(MARKETS), "lift", "se", study="market")
    assert report.effect == pytest.approx(ref.mean_effect_re, abs=1e-9)


def test_hartung_knapp_interval_matches_hand_computation() -> None:
    effects, ses = EFFECTS, SES
    k = len(effects)
    # Hartung & Knapp (2001), DerSimonian & Laird (1986), written out independently
    w = [1 / s**2 for s in ses]
    fe = sum(wi * e for wi, e in zip(w, effects)) / sum(w)
    q = sum(wi * (e - fe) ** 2 for wi, e in zip(w, effects))
    c = sum(w) - sum(wi**2 for wi in w) / sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = [1 / (s**2 + tau2) for s in ses]
    mu = sum(wi * e for wi, e in zip(ws, effects)) / sum(ws)
    var_hk = sum(wi * (e - mu) ** 2 for wi, e in zip(ws, effects)) / ((k - 1) * sum(ws))
    lo, hi = mu - T_975_5 * math.sqrt(var_hk), mu + T_975_5 * math.sqrt(var_hk)

    assert _dl(effects, ses).se_hk == pytest.approx(math.sqrt(var_hk), rel=1e-12)
    report = verify_pooled_effect(_rows(MARKETS), "lift", "se", study="market")
    assert f"Hartung-Knapp [{lo:+.3g}, {hi:+.3g}] (t on 5 df)" in report.caveats[0]
    # the HK interval is wider than the DL Wald one here, as it should be with 6 studies
    assert hi - lo > 2 * 1.959964 * _dl(effects, ses).se


def test_hartung_knapp_interval_matches_statsmodels_t_interval(ref) -> None:
    # statsmodels' "random effect wls" row on a t is the Hartung-Knapp interval
    lo, hi = ref.conf_int(alpha=0.05, use_t=True)[3]
    report = verify_pooled_effect(_rows(MARKETS), "lift", "se", study="market")
    assert f"Hartung-Knapp [{lo:+.3g}, {hi:+.3g}]" in report.caveats[0]


def test_reported_intervals_match_statsmodels_and_hand_formulas(ref) -> None:
    fe_ci, re_ci = ref.conf_int(alpha=0.05, use_t=False)[:2]
    report = verify_pooled_effect(_rows(MARKETS), "lift", "se", study="market")
    assert f"95% CI [{re_ci[0]:+.3g}, {re_ci[1]:+.3g}]" in report.caveats[0]
    assert report.caveats[1] == (
        f"fixed-effect estimate {ref.mean_effect_fe:+.3g} "
        f"[{fe_ci[0]:+.3g}, {fe_ci[1]:+.3g}]; Cochran's Q = {ref.q:.2f} on 5 df "
        f"(p = {ref.test_homogeneity().pvalue:.2g})"
    )
    # Higgins, Thompson & Spiegelhalter (2009): mu +- t_{k-2} sqrt(tau² + se(mu)²)
    half = T_975_4 * math.sqrt(ref.tau2 + ref.sd_eff_w_re**2)
    lo, hi = ref.mean_effect_re - half, ref.mean_effect_re + half
    assert report.caveats[2].startswith(
        f"95% prediction interval for a new study's true effect "
        f"[{lo:+.3g}, {hi:+.3g}] (t on 4 df)"
    )


def test_six_market_pool_is_sound() -> None:
    report = verify_pooled_effect(
        _rows(MARKETS), "lift", "se", study="market", new_effect=8.0, new_se=4.9
    )
    assert report.verdict == "sound"
    assert [c.name for c in report.checks] == [
        "enough-studies",
        "heterogeneity-quantified",
        "leave-one-out-stable",
        "no-prior-data-conflict",
    ]
    assert report.pivotal is None
    assert report.significant  # the HK interval excludes zero


def test_three_studies_fail_enough_studies_but_still_report() -> None:
    report = verify_pooled_effect(_rows(MARKETS[:3]), "lift", "se", study="market")
    check = _check(report, "enough-studies")
    assert not check.survived
    assert "not reliably estimable from 3 studies" in check.detail
    assert report.verdict == "unsound"
    assert "Hartung-Knapp" in report.pivotal
    assert "Hartung-Knapp [" in report.caveats[0]
    assert "prediction interval" in report.caveats[2]


def test_five_studies_are_enough() -> None:
    report = verify_pooled_effect(_rows(MARKETS[:5]), "lift", "se", study="market")
    assert _check(report, "enough-studies").survived


def test_outlier_that_carries_the_sign_fails_leave_one_out() -> None:
    report = verify_pooled_effect(_rows(OUTLIER_STUDIES), "lift", "se", study="market")
    check = _check(report, "leave-one-out-stable")
    assert not check.survived
    assert "'Outlier'" in check.detail
    assert "reversing its sign" in check.detail
    assert report.effect > 0
    assert report.verdict == "unsound"


def test_dominant_precise_study_fails_leave_one_out_on_the_interval() -> None:
    # one precise study (se 0.1, weight 100) against five imprecise ones: Q is below
    # its expectation so tau² = 0, the pooled mean is about (100 + 1) / 105 = 0.96 with
    # a narrow HK interval, and dropping the precise study leaves 0.2, the same sign
    studies = [("Precise", 1.0, 0.1)] + [(f"m{i}", 0.2, 1.0) for i in range(5)]
    report = verify_pooled_effect(_rows(studies), "lift", "se", study="market")
    check = _check(report, "leave-one-out-stable")
    assert not check.survived
    assert "'Precise'" in check.detail
    assert "outside the Hartung-Knapp interval" in check.detail


def test_conflicting_new_study_fails_and_consistent_one_survives(ref) -> None:
    def p_value(new: float, se: float) -> float:
        z = (new - ref.mean_effect_re) / math.sqrt(
            se**2 + ref.tau2 + ref.sd_eff_w_re**2
        )
        return math.erfc(abs(z) / math.sqrt(2))

    consistent = verify_pooled_effect(
        _rows(MARKETS), "lift", "se", study="market", new_effect=8.0, new_se=4.9
    )
    check = _check(consistent, "no-prior-data-conflict")
    assert check.survived
    assert f"p = {p_value(8.0, 4.9):.2g}" in check.detail

    conflicting = verify_pooled_effect(
        _rows(MARKETS), "lift", "se", study="market", new_effect=-10.0, new_se=2.0
    )
    check = _check(conflicting, "no-prior-data-conflict")
    assert not check.survived
    assert p_value(-10.0, 2.0) < 0.05
    assert f"p = {p_value(-10.0, 2.0):.2g}" in check.detail
    assert "robust MAP" in check.detail
    assert conflicting.verdict == "unsound"
    assert "prior-data conflict" in conflicting.pivotal


def test_no_conflict_check_without_a_new_study() -> None:
    report = verify_pooled_effect(_rows(MARKETS), "lift", "se", study="market")
    assert "no-prior-data-conflict" not in {c.name for c in report.checks}


def test_alpha_sets_the_conflict_threshold() -> None:
    # +8.0 (se 4.9) sits at p ~ 0.33: consistent at alpha 0.05, a conflict at 0.5
    loose = verify_pooled_effect(
        _rows(MARKETS), "lift", "se", new_effect=8.0, new_se=4.9, alpha=0.5
    )
    assert not _check(loose, "no-prior-data-conflict").survived
    assert "50% CI" in loose.caveats[0]


@pytest.mark.parametrize("sign", [1, -1])
def test_considerable_heterogeneity_with_sign_disagreement_fails(sign: int) -> None:
    # a precise large effect against five imprecise null-ish ones: fixed effect follows
    # the precise study (about +9.5, clearly positive) while random effects spreads the
    # weight (about +1, spanning zero), and I² is near 1; mirrored, the fixed-effect
    # interval is negative, so both directions of the sign comparison are exercised
    studies = [("Big", sign * 10.0, 0.1)] + [
        (f"m{i}", sign * e, 1.0) for i, e in enumerate([-1.0, -1.5, -0.5, -1.0, 0.0])
    ]
    report = verify_pooled_effect(_rows(studies), "lift", "se", study="market")
    check = _check(report, "heterogeneity-quantified")
    assert not check.survived
    assert "disagree on sign" in check.detail
    assert report.verdict == "unsound"
    assert "not a usable summary" in report.pivotal


def test_considerable_heterogeneity_alone_does_not_fail() -> None:
    # equal standard errors give fixed and random effects the same mean (both are the
    # plain average, 1.0), so however large I² is, the two models agree
    studies = [(f"m{i}", e, 0.5) for i, e in enumerate([-4, -2, 0, 2, 4, 6])]
    report = verify_pooled_effect(_rows(studies), "lift", "se", study="market")
    # Q = sum((e - 1)² / 0.25) = 280 on 5 df, so I² = (280 - 5) / 280 = 98%
    assert "I² = 98%" in _check(report, "heterogeneity-quantified").detail
    assert _check(report, "heterogeneity-quantified").survived
    assert report.effect == pytest.approx(1.0)


def test_considerable_heterogeneity_agreeing_in_sign_survives() -> None:
    # the precise study against five imprecise ones on its side of zero: I² is 97% and
    # the fixed effect (9.73) lies outside the random-effects interval (statsmodels:
    # [2.18, 8.59]), but both intervals are positive, so the weighting cannot flip the
    # conclusion
    studies = [("Big", 10.0, 0.1)] + [
        (f"m{i}", e, 1.0) for i, e in enumerate([4.0, 5.0, 3.0, 4.5, 5.5])
    ]
    check = _check(
        verify_pooled_effect(_rows(studies), "lift", "se", study="market"),
        "heterogeneity-quantified",
    )
    assert check.survived
    assert "I² = 97%" in check.detail
    assert "agrees in sign with the random-effects interval" in check.detail


def test_identical_studies_are_leave_one_out_stable() -> None:
    # with no scatter the Hartung-Knapp interval collapses to the estimate itself, and
    # dropping any one identical study leaves that same estimate: stable, not broken
    studies = [(f"m{i}", 0.1, 0.3) for i in range(6)]
    report = verify_pooled_effect(_rows(studies), "lift", "se", study="market")
    assert _check(report, "leave-one-out-stable").survived
    assert report.verdict == "sound"


def test_inconclusive_on_unusable_input() -> None:
    one = verify_pooled_effect(_rows(MARKETS[:1]), "lift", "se")
    assert one.verdict == "inconclusive"
    zero_se = verify_pooled_effect(_rows([*MARKETS, ("Z", 1.0, 0.0)]), "lift", "se")
    assert zero_se.verdict == "inconclusive"
    bad_new_se = verify_pooled_effect(
        _rows(MARKETS), "lift", "se", new_effect=1.0, new_se=0.0
    )
    assert bad_new_se.verdict == "inconclusive"
    half_new = verify_pooled_effect(_rows(MARKETS), "lift", "se", new_effect=1.0)
    assert half_new.verdict == "inconclusive"


def test_rows_missing_a_value_are_left_out_and_said_so() -> None:
    rows = [*_rows(MARKETS), {"market": "Gdansk", "lift": "3.0", "se": ""}]
    report = verify_pooled_effect(rows, "lift", "se", study="market")
    full = verify_pooled_effect(_rows(MARKETS), "lift", "se", study="market")
    assert report.effect == full.effect
    assert "1 rows without a numeric 'lift' and 'se' were left out" in report.caveats


def test_unlabelled_studies_are_named_by_row() -> None:
    report = verify_pooled_effect(_rows(OUTLIER_STUDIES), "lift", "se")
    assert "'study 6'" in _check(report, "leave-one-out-stable").detail


def test_a_missing_study_label_falls_back_to_its_row() -> None:
    studies = [*OUTLIER_STUDIES[:-1], (None, 6.0, 0.3)]
    report = verify_pooled_effect(_rows(studies), "lift", "se", study="market")
    detail = _check(report, "leave-one-out-stable").detail
    assert "'study 6'" in detail
    assert "None" not in detail


def test_a_null_pool_is_not_broken_by_a_sign_change() -> None:
    # tau² = 0, so the mean is 0.05 / 6 ≈ +0.008 and Hartung-Knapp is about
    # ±2.571 * sqrt(3.772 / 30) = ±0.91. Each drop lands in [-0.19, +0.21]: it
    # crosses zero, but the full result never had a sign to reverse.
    studies = [(f"m{i}", e, 1.0) for i, e in enumerate([1, -1, 0.9, -0.9, 0.3, -0.25])]
    report = verify_pooled_effect(_rows(studies), "lift", "se", study="market")
    assert not report.significant
    assert _check(report, "leave-one-out-stable").survived
    assert report.verdict == "sound"


@pytest.mark.parametrize("new_effect", [math.nan, math.inf, -math.inf])
def test_a_non_finite_new_study_is_inconclusive(new_effect: float) -> None:
    report = verify_pooled_effect(
        _rows(MARKETS), "lift", "se", new_effect=new_effect, new_se=1.0
    )
    assert report.verdict == "inconclusive"


def test_verify_all_dispatches_on_effect_and_se_roles(ref) -> None:
    report = verify_all(
        _rows(MARKETS),
        effect="lift",
        se="se",
        study="market",
        new_effect=8.0,
        new_se=4.9,
    )
    assert report.verdict == "sound"
    assert report.ran[0].name == "pooled-effect"
    assert {g.name for g in report.ran} <= {"pooled-effect", "stability"}
    assert report.estimate == pytest.approx(ref.mean_effect_re, abs=1e-9)
    assert (
        report.estimate_label
        == f"{ref.mean_effect_re:+.3g} in lift, pooled by random effects"
    )

    conflict = verify_all(
        _rows(MARKETS), effect="lift", se="se", new_effect=-10.0, new_se=2.0
    )
    assert conflict.verdict == "unsound"

    too_few = verify_all(_rows(MARKETS[:3]), effect="lift", se="se")
    assert too_few.verdict == "unsound"


_study = st.tuples(
    st.floats(min_value=-50, max_value=50, allow_nan=False),
    st.floats(min_value=0.05, max_value=20, allow_nan=False),
)


@given(
    studies=st.lists(_study, min_size=2, max_size=12),
    shift=st.floats(min_value=-100, max_value=100, allow_nan=False),
    scale=st.floats(min_value=0.1, max_value=10, allow_nan=False),
)
@settings(deadline=None, max_examples=200)
def test_fit_invariants(studies, shift, scale) -> None:
    effects = [e for e, _ in studies]
    ses = [s for _, s in studies]
    fit = _dl(effects, ses)
    assert fit.tau2 >= 0
    assert 0 <= fit.i2 < 1
    assert min(effects) - 1e-9 <= fit.mu <= max(effects) + 1e-9
    rev = _dl(effects[::-1], ses[::-1])
    assert rev.mu == pytest.approx(fit.mu, rel=1e-9, abs=1e-9)
    assert rev.tau2 == pytest.approx(fit.tau2, rel=1e-9, abs=1e-9)
    moved = _dl([e + shift for e in effects], ses)
    assert moved.mu == pytest.approx(fit.mu + shift, rel=1e-9, abs=1e-6)
    assert moved.tau2 == pytest.approx(fit.tau2, rel=1e-6, abs=1e-6)
    scaled = _dl([e * scale for e in effects], [s * scale for s in ses])
    assert scaled.mu == pytest.approx(fit.mu * scale, rel=1e-9, abs=1e-6)
    assert scaled.i2 == pytest.approx(fit.i2, rel=1e-6, abs=1e-9)
