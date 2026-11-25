"""Text columns as controls and as candidate confounders."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

import pytest

from elbi_core.verification import (
    verify_correlation,
    verify_effect,
    verify_logistic,
    verify_regression,
)


def _plans() -> list[dict[str, Any]]:
    # the reproduction in GitHub issue 210, verbatim.
    rng = random.Random(7)
    rows: list[dict[str, Any]] = []
    for i in range(300):
        if i < 150:
            discount = rng.uniform(15, 30)
            spend = 900 - 2 * discount + rng.uniform(-60, 60)
            rows.append({"plan": "enterprise", "discount": discount, "spend": spend})
        else:
            discount = rng.uniform(0, 12)
            spend = 300 + rng.uniform(-60, 60)
            rows.append({"plan": "team", "discount": discount, "spend": spend})
    return rows


def _flagged(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "is_enterprise": int(r["plan"] == "enterprise"),
            **{k: v for k, v in r.items() if k != "plan"},
        }
        for r in rows
    ]


def _regions(names: tuple[str, str, str] = ("north", "south", "west")):
    rng = random.Random(1)
    counts = {names[0]: 300, names[1]: 220, names[2]: 180}
    shift = {names[0]: 0.0, names[1]: 2.0, names[2]: -1.5}
    rows = []
    for region, k in counts.items():
        for _ in range(k):
            x = shift[region] + rng.gauss(0, 1)
            y = 1.5 * x + 3 * shift[region] + rng.gauss(0, 1)
            rows.append({"region": region, "x": x, "y": y})
    return rows


def _real_effect(n: int = 400) -> list[dict[str, Any]]:
    rng = random.Random(2)
    rows = []
    for _ in range(n):
        x = rng.uniform(0, 30)
        rows.append({"x": x, "y": 100 + 5 * x + rng.uniform(-20, 20)})
    return rows


def test_the_reported_confounder_is_caught() -> None:
    report = verify_effect(_plans(), "discount", "spend")
    assert report.verdict == "unsound" and "plan" in (report.pivotal or "")


def test_holding_a_two_value_text_control_matches_a_0_1_column() -> None:
    # a two-value category is one 0/1 column (patsy treatment coding).
    rows = _plans()
    text = verify_effect(rows, "discount", "spend", controls=["plan"])
    flag = verify_effect(
        _flagged(rows), "discount", "spend", controls=["is_enterprise"]
    )
    assert text.verdict == flag.verdict != "sound"
    assert text.effect == pytest.approx(flag.effect)


def test_holding_a_three_value_text_control_matches_two_0_1_columns() -> None:
    # K values are K-1 0/1 columns, the most common value as the baseline.
    rows = _regions()
    hand = [
        {
            "is_south": int(r["region"] == "south"),
            "is_west": int(r["region"] == "west"),
            "x": r["x"],
            "y": r["y"],
        }
        for r in rows
    ]
    text = verify_effect(rows, "x", "y", controls=["region"])
    flag = verify_effect(hand, "x", "y", controls=["is_south", "is_west"])
    assert text.verdict == flag.verdict == "sound"
    assert text.effect == pytest.approx(flag.effect)


def test_renaming_the_values_changes_nothing() -> None:
    a = verify_effect(_regions(), "x", "y", controls=["region"])
    b = verify_effect(_regions(("zeta", "alpha", "mu")), "x", "y", controls=["region"])
    assert (a.verdict, a.effect) == (b.verdict, pytest.approx(b.effect))


def test_a_control_that_is_not_a_column_is_named() -> None:
    # GitHub issue 210 asks for an unusable control to be named, not ignored.
    report = verify_effect(_plans(), "discount", "spend", controls=["plann"])
    assert report.verdict == "inconclusive"
    assert any("plann" in c for c in report.caveats)


def test_a_text_control_with_too_many_values_is_named() -> None:
    rows = _real_effect(600)
    for i, r in enumerate(rows):
        r["store"] = f"store-{i % 60}"
    report = verify_effect(rows, "x", "y", controls=["store"])
    assert report.verdict == "inconclusive"
    assert any("store" in c for c in report.caveats)


def test_a_text_negative_control_is_named() -> None:
    report = verify_effect(_plans(), "discount", "spend", negative_controls=["plan"])
    assert report.verdict == "inconclusive"
    assert any("plan" in c for c in report.caveats)
    assert not any("too few numeric rows" in c for c in report.caveats)


def test_text_x_points_to_the_group_comparison() -> None:
    report = verify_effect(_plans(), "plan", "spend")
    assert report.verdict == "inconclusive"
    assert any("verify_comparison" in c for c in report.caveats)


def test_a_copy_of_x_in_other_units_does_not_overturn_a_real_effect() -> None:
    # a linear function of x has no variation of its own, so it cannot confound.
    rows = _real_effect()
    for r in rows:
        r["x_pct"] = r["x"] / 100
    report = verify_effect(rows, "x", "y")
    assert report.verdict == "sound"
    assert any("x_pct" in c for c in report.caveats)


def test_a_copy_of_y_in_other_units_does_not_overturn_a_real_effect() -> None:
    rows = _real_effect()
    for r in rows:
        r["y_cents"] = r["y"] * 100
    report = verify_effect(rows, "x", "y")
    assert report.verdict == "sound"
    assert any("y_cents" in c for c in report.caveats)


def _set_by_region() -> list[dict[str, Any]]:
    rng = random.Random(3)
    level = {"north": 5.0, "south": 5.0, "west": 20.0}
    lift = {"north": 0.0, "south": 150.0, "west": 300.0}
    return [
        {"region": g, "x": level[g], "y": lift[g] + rng.gauss(0, 20)}
        for g in level
        for _ in range(150)
    ]


def test_an_effect_set_entirely_by_a_category_is_not_certified() -> None:
    # x is fixed within each region, so its effect is inseparable from region's.
    report = verify_effect(_set_by_region(), "x", "y")
    assert report.verdict == "inconclusive"
    assert "region" in (report.pivotal or "")


def test_a_control_that_sets_x_is_named() -> None:
    report = verify_effect(_set_by_region(), "x", "y", controls=["region"])
    assert report.verdict == "inconclusive"
    assert any("region" in c for c in report.caveats)


def test_a_value_on_one_row_does_not_make_the_effect_fragile() -> None:
    # a column constant on a fit's rows adds nothing beyond the intercept.
    rows = _real_effect()
    for i, r in enumerate(rows):
        r["promo"] = int(i == 0)
    report = verify_effect(rows, "x", "y", controls=["promo"])
    assert report.verdict == "sound"


def test_blank_text_counts_as_a_value() -> None:
    rows = [
        {"coupon": "SAVE10" if r["plan"] == "enterprise" else "", **r} for r in _plans()
    ]
    for r in rows:
        del r["plan"]
    report = verify_effect(rows, "discount", "spend")
    assert report.verdict == "unsound" and "coupon" in (report.pivotal or "")


def test_text_columns_left_untested_are_named() -> None:
    rows = _real_effect(300)
    for i, r in enumerate(rows):
        r["customer"] = f"cust-{i}"
        r["code"] = "X" if i < 3 else ""
    report = verify_effect(rows, "x", "y")
    assert report.verdict == "sound"
    assert any("customer" in c and "code" in c for c in report.caveats)


def test_regression_holds_a_text_control_fixed() -> None:
    rows = _plans()
    text = verify_regression(rows, "spend", "discount", controls=["plan"])
    flag = verify_regression(
        _flagged(rows), "spend", "discount", controls=["is_enterprise"]
    )
    assert text.verdict == flag.verdict
    assert text.effect == pytest.approx(flag.effect)


def test_logistic_holds_a_text_control_fixed() -> None:
    rng = random.Random(4)
    rows = []
    for i in range(600):
        enterprise = i % 3 == 0
        discount = rng.uniform(10, 30) if enterprise else rng.uniform(0, 15)
        p = 1 / (
            1 + 2.718281828 ** -(-2 + 0.08 * discount + (1.5 if enterprise else 0))
        )
        rows.append(
            {
                "plan": "enterprise" if enterprise else "team",
                "discount": discount,
                "renewed": int(rng.random() < p),
            }
        )
    text = verify_logistic(rows, "discount", "renewed", controls=["plan"])
    flag = verify_logistic(
        _flagged(rows), "discount", "renewed", controls=["is_enterprise"]
    )
    assert text.verdict == flag.verdict
    assert text.effect == pytest.approx(flag.effect)


def test_correlation_ignores_a_copy_of_x() -> None:
    rows = _real_effect()
    for r in rows:
        r["x_pct"] = r["x"] / 100
    report = verify_correlation(rows, "x", "y")
    assert report.verdict == "sound"
    assert any("x_pct" in c for c in report.caveats)


def test_correlation_catches_a_text_confounder() -> None:
    rows = _plans()
    flag = verify_correlation(_flagged(rows), "discount", "spend")
    text = verify_correlation(rows, "discount", "spend")
    assert flag.verdict == "unsound"
    assert text.verdict == "unsound" and "plan" in (text.pivotal or "")


_REGION_OF = {
    "CA": "west",
    "OR": "west",
    "NY": "east",
    "MA": "east",
    "TX": "south",
    "FL": "south",
}


def _states() -> list[dict[str, Any]]:
    rng = random.Random(11)
    shift = {"west": 0.0, "east": 2.0, "south": -1.5}
    rows = []
    for i in range(900):
        state = list(_REGION_OF)[i % 6]
        region = _REGION_OF[state]
        x = shift[region] + rng.gauss(0, 1)
        y = 1.5 * x + 3 * shift[region] + rng.gauss(0, 1)
        rows.append({"state": state, "region": region, "x": x, "y": y})
    return rows


def _without(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != column} for r in rows]


def _regression(rows: list[dict[str, Any]], x: str, y: str, **kw: Any) -> Any:
    return verify_regression(rows, y, x, **kw)


def test_a_column_set_by_a_held_control_is_not_a_new_confounder() -> None:
    # holding state fixed holds region fixed, so the region column adds nothing.
    rows = _states()
    held = verify_effect(rows, "x", "y", controls=["state"])
    alone = verify_effect(_without(rows, "region"), "x", "y", controls=["state"])
    assert held.verdict == alone.verdict == "sound"
    assert held.effect == pytest.approx(alone.effect)


@pytest.mark.parametrize("check", [verify_effect, _regression])
def test_a_control_nested_in_another_changes_nothing(check: Callable[..., Any]) -> None:
    # holding state fixed holds region fixed, so both fits are the same fit.
    rows = _states()
    both = check(rows, "x", "y", controls=["state", "region"])
    one = check(rows, "x", "y", controls=["state"])
    assert both.verdict == one.verdict == "sound"
    assert both.effect == pytest.approx(one.effect)


def test_logistic_treats_a_nested_control_as_already_held() -> None:
    rng = random.Random(13)
    rows = [
        {**r, "won": int(rng.random() < 1 / (1 + math.exp(-0.9 * r["x"])))}
        for r in _states()
    ]
    both = verify_logistic(rows, "x", "won", controls=["state", "region"])
    one = verify_logistic(rows, "x", "won", controls=["state"])
    assert both.verdict == one.verdict == "sound"
    assert both.effect == pytest.approx(one.effect)


def test_two_text_controls_blank_on_the_same_rows_are_both_held() -> None:
    # the two blank 0/1 columns are one column, so the hand fit has it once.
    rng = random.Random(6)
    rows, hand = [], []
    for i in range(600):
        x = rng.gauss(0, 1)
        y = 2 * x + rng.gauss(0, 1)
        tier = "" if i % 5 == 0 else rng.choice(["gold", "silver", "bronze"])
        channel = "" if i % 5 == 0 else rng.choice(["web", "store", "phone"])
        rows.append({"x": x, "y": y, "tier": tier, "channel": channel})
        flags = {v: int(v in (tier, channel)) for v in ("silver", "bronze", "store")}
        hand.append({"x": x, "y": y, "blank": int(tier == ""), **flags})
        hand[-1]["phone"] = int(channel == "phone")
    flag_names = ["blank", "silver", "bronze", "store", "phone"]
    text = verify_regression(rows, "y", "x", controls=["tier", "channel"])
    flag = verify_regression(hand, "y", "x", controls=flag_names)
    assert text.verdict == flag.verdict == "sound"
    assert text.effect == pytest.approx(flag.effect)
    assert (
        verify_effect(rows, "x", "y", controls=["tier", "channel"]).verdict == "sound"
    )


def test_logistic_names_a_control_that_sets_x() -> None:
    rng = random.Random(5)
    rows = [{**r, "y": int(rng.random() < 0.3 + r["x"] / 50)} for r in _set_by_region()]
    report = verify_logistic(rows, "x", "y", controls=["region"])
    assert report.verdict == "inconclusive"
    assert any("region" in c for c in report.caveats)


def test_correlation_does_not_blame_a_category_that_sets_x() -> None:
    # is_weekend is fixed within each day, so no partial correlation exists.
    rng = random.Random(8)
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    rows = []
    for i in range(700):
        weekend = int(days[i % 7] in ("sat", "sun"))
        sales = 100 + 40 * weekend + rng.gauss(0, 10)
        rows.append({"day": days[i % 7], "is_weekend": weekend, "sales": sales})
    report = verify_correlation(rows, "is_weekend", "sales")
    effect = verify_effect(rows, "is_weekend", "sales")
    assert report.verdict == effect.verdict == "inconclusive"
    assert "day" in (report.pivotal or "")


@pytest.mark.parametrize("check", [verify_effect, verify_correlation, _regression])
@pytest.mark.parametrize("token", ["N/A", "NULL"])
def test_a_missing_value_token_in_x_drops_only_its_rows(
    check: Callable[..., Any], token: str
) -> None:
    # pandas.read_csv reads "N/A" and "NULL" as missing by default.
    rows = _real_effect(500)
    marked = [{**r, "x": token} if i % 100 == 0 else r for i, r in enumerate(rows)]
    kept = [r for i, r in enumerate(rows) if i % 100]
    a, b = check(marked, "x", "y"), check(kept, "x", "y")
    assert a.verdict == b.verdict == "sound"
    assert a.effect == pytest.approx(b.effect)


def test_a_confounder_is_found_whichever_column_comes_first() -> None:
    # the verdict is a property of the rows, not of the column order.
    rng = random.Random(21)
    rows = []
    for i in range(900):
        band = ["low", "mid", "high"][i % 3]
        x = float(band == "high")
        conf = 2 * x + rng.gauss(0, 1)
        rows.append(
            {"band": band, "conf": conf, "x": x, "y": 4 * conf + rng.gauss(0, 1)}
        )
    swapped = [{"conf": r["conf"], **r} for r in rows]
    for report in (verify_effect(rows, "x", "y"), verify_effect(swapped, "x", "y")):
        assert report.verdict == "unsound" and "conf" in (report.pivotal or "")


def test_a_resample_missing_a_level_is_not_a_break() -> None:
    # full-data |t| is about 40, so no half-size resample (|t| about 28) breaks
    # it; a resample that lacks a store is refit without that store, not scored broken.
    rng = random.Random(3)
    rows = []
    for s in range(30):
        for _ in range(3):
            x = rng.gauss(0, 1)
            rows.append(
                {"x": x, "y": 2 * x + s / 10 + rng.gauss(0, 1), "store": f"s{s}"}
            )
    report = verify_effect(rows, "x", "y", controls=["store"])
    fragility = next(c for c in report.checks if c.name == "fragility")
    assert "(0% of subsets" in fragility.detail
