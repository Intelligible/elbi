"""Tests for checking an answer's numbers and citations against served values.

Expected values are worked out by hand (2/7 is 0.2857..., written 28.6%) or taken
from Python's own number formatting, never read back from the module under test.
"""

from __future__ import annotations

import math
from decimal import Decimal

import hypothesis.strategies as st
from hypothesis import given, settings

from elbi_core.citations import (
    Leaf,
    check,
    extract_citations,
    extract_numbers,
    numeric_leaves,
)

SHARE = 2 / 7  # 0.285714...


def _values(text: str) -> list[float]:
    return [mention.value for mention in extract_numbers(text)]


def _verdict(text: str, value: object) -> str:
    return check(text, {"d": {"v": value}}, {}).verdict


# --- extraction -------------------------------------------------------------------


def test_extracts_signs_separators_units_and_written_precision() -> None:
    mentions = extract_numbers("Churn was 97.7% of 1,847 users, +8.0 pts, p = 2.3e-5.")
    assert [(m.value, m.precision, m.percent, m.signed) for m in mentions] == [
        (97.7, 1, True, False),
        (1847.0, 0, False, False),
        (8.0, 1, True, True),
        (2.3e-5, 6, False, False),
    ]
    assert [m.text.strip() for m in mentions] == [
        "97.7%",
        "1,847",
        "+8.0 pts",
        "2.3e-5",
    ]


def test_trailing_zeros_are_written_precision() -> None:
    assert [m.precision for m in extract_numbers("28.60 and 30 and .5")] == [2, 0, 1]


def test_intervals_yield_both_bounds() -> None:
    assert _values("range [30, 3,664]") == [30.0, 3664.0]
    assert _values("difference -1.6 to 17.6") == [-1.6, 17.6]
    assert _values("95% CI [\u22121.6, 17.6]") == [-1.6, 17.6]
    assert _values("1.6\u201317.6") == [1.6, 17.6]


def test_a_hyphen_between_numbers_is_not_a_sign() -> None:
    mentions = extract_numbers("ages 10-20")
    assert [(m.value, m.signed) for m in mentions] == [(10.0, False), (20.0, False)]


def test_labels_are_not_claims() -> None:
    text = (
        "1. As of 2026-08-07T09:30 (risk_overlay@1a2b3c4d), August 7, 2026, "
        "7 Aug 2026 and Aug 2026, Q3 and COVID-19 [2] on 0.2.0, the 95% "
        "confidence interval, a 95% credible interval, a 99.9% CI, a 2nd look at "
        "5k rows."
    )
    assert extract_numbers(text) == []


def test_a_bare_year_is_a_date_not_a_claim() -> None:
    assert _values("the 2026 Austin re-run lifted retention 4.1 pts") == [4.1]
    assert _values("2,026 customers, -2026, 2026.5 and 2026%") == [
        2026.0,
        -2026.0,
        2026.5,
        2026.0,
    ]


def test_a_list_marker_is_masked_only_at_line_start() -> None:
    assert _values("1. First\n  2) second has 3.5") == [3.5]


def test_citations_are_name_at_hash_tokens() -> None:
    cited = extract_citations(
        "per risk_overlay@1a2b3c4d. and cov_matrix@29992dcba7d6, "
        "not sam@deadbeef.com, not x@1234567 (7 hex), not v@NOTHEX12."
    )
    assert [(c.name, c.version) for c in cited] == [
        ("risk_overlay", "1a2b3c4d"),
        ("cov_matrix", "29992dcba7d6"),
    ]


# --- flattening served values -----------------------------------------------------


def test_numeric_leaves_walks_rows_and_component_evidence() -> None:
    served = {
        "rows": [{"region": "EMEA", "share": SHARE, "flag": True}],
        "row_count": 1,
        "components": [
            {
                "statement": "Churn is 31% above a 20% discount.",
                "evidence": {"odds_ratio": 2.4, "missing": math.nan},
            }
        ],
    }
    assert numeric_leaves(served) == [
        Leaf("$.rows[0].share", SHARE),
        Leaf("$.row_count", 1.0),
        Leaf("$.components[0].statement", 31.0),
        Leaf("$.components[0].statement", 20.0),
        Leaf("$.components[0].evidence.odds_ratio", 2.4),
    ]


def test_numeric_leaves_reads_numbers_written_in_strings_but_not_hashes() -> None:
    served = {"cut": "-0.0912", "as_of": "2026-08-07", "source": "cov@1a2b3c4d"}
    assert numeric_leaves(served) == [Leaf("$.cut", -0.0912)]
    assert numeric_leaves(None) == []


# --- matching ---------------------------------------------------------------------


def test_a_number_must_round_from_a_served_value_at_its_written_precision() -> None:
    assert _verdict("28.6%", SHARE) == "faithful"
    assert _verdict("28.57%", SHARE) == "faithful"
    assert _verdict("29%", SHARE) == "faithful"
    assert _verdict("0.29", SHARE) == "faithful"
    assert _verdict("28.5%", SHARE) == "unfaithful"
    assert _verdict("28.58%", SHARE) == "unfaithful"
    assert _verdict("30%", SHARE) == "unfaithful"


def test_just_past_half_a_unit_in_the_last_place_is_unsupported() -> None:
    assert _verdict("1.2", 1.2499) == "faithful"
    assert _verdict("1.2", 1.2501) == "unfaithful"
    assert _verdict("12", 12.4999) == "faithful"
    assert _verdict("12", 12.5001) == "unfaithful"


def test_a_tie_may_round_either_way() -> None:
    assert _verdict("0.12", 0.125) == "faithful"
    assert _verdict("0.13", 0.125) == "faithful"
    assert _verdict("0.11", 0.125) == "unfaithful"
    assert _verdict("0.14", 0.125) == "unfaithful"


def test_percent_and_fraction_convert_only_in_the_direction_the_unit_implies() -> None:
    assert _verdict("97.7%", 0.977) == "faithful"
    assert _verdict("97.7 percent", 0.977) == "faithful"
    assert _verdict("97.7 pct", 0.977) == "faithful"
    assert _verdict("97.7%", 97.7) == "faithful"
    assert _verdict("0.977", 97.7) == "unfaithful"
    assert _verdict("0.977%", 97.7) == "unfaithful"
    assert _verdict("9770", 97.7) == "unfaithful"


def test_the_reported_support_names_the_value_and_conversion() -> None:
    report = check("97.7% and 1,847", {"d": {"rate": 0.977, "n": 1847}}, {})
    rate, n = report.numbers
    assert (rate.derivation, rate.path, rate.value, rate.scale) == (
        "d",
        "$.rate",
        0.977,
        100.0,
    )
    assert (n.path, n.scale) == ("$.n", 1.0)


def test_a_percent_prefers_the_fraction_over_a_count_that_rounds_the_same() -> None:
    (support,) = check("5%", {"d": {"n": 5, "rate": 0.05}}, {}).numbers
    assert support.path == "$.rate"


def test_points_convert_like_a_percent() -> None:
    assert _verdict("+8.0 pts", 0.08) == "faithful"
    assert _verdict("+8.0 pts", -0.08) == "unfaithful"
    assert _verdict("8.0 percentage points", 0.08) == "faithful"
    assert _verdict("8.0 pp", 0.08) == "faithful"
    assert _verdict("8.0 points", 0.08) == "faithful"


def test_only_an_explicit_sign_is_checked() -> None:
    assert _verdict("TTD fell 21.9%", -0.219) == "faithful"
    assert _verdict("-21.9%", -0.219) == "faithful"
    assert _verdict("+21.9%", -0.219) == "unfaithful"
    assert _verdict("-21.9%", 0.219) == "unfaithful"


def test_an_interval_is_supported_bound_by_bound() -> None:
    served = {"d": {"min": 30, "max": 3664, "ci_low": -1.62, "ci_high": 17.58}}
    assert check("[30, 3,664]; -1.6 to 17.6", served, {}).verdict == "faithful"
    assert check("[30, 3,665]", served, {}).verdict == "unfaithful"


def test_a_component_statement_is_checked_against_its_evidence() -> None:
    component = {
        "statement": "Discounts above 20% churn more (31% vs. 12%).",
        "structure": {"threshold": 0.2},
        "evidence": {"churn_rate_above": 0.312, "churn_rate_at_or_below": 0.118},
    }
    served = {"facts": {"components": [component], "count": 1}}
    text = "Above a 20% discount churn is 31.2% against 11.8%."
    assert check(text, served, {}).verdict == "faithful"


# --- verdicts and citations -------------------------------------------------------


def test_a_fabricated_number_is_named() -> None:
    report = check("Churn was 21.4%, up from 28.6%.", {"d": {"v": SHARE}}, {})
    assert report.verdict == "unfaithful"
    assert [s.mention.text for s in report.numbers if not s.supported] == ["21.4%"]
    assert "- [FAIL] 21.4%: no served value rounds to it" in report.render()


def test_no_numbers_is_inconclusive() -> None:
    report = check("Churn rose after the price change.", {"d": {"v": SHARE}}, {})
    assert report.verdict == "inconclusive"
    assert report.numbers == ()


def test_a_stale_citation_is_unfaithful_even_without_numbers() -> None:
    report = check("See d@deadbeef.", {"d": {"v": SHARE}}, {"d": "1a2b3c4d" * 8})
    assert report.verdict == "unfaithful"
    (status,) = report.citations
    assert (status.status, status.current) == ("stale", "1a2b3c4d" * 8)
    assert "- [FAIL] d@deadbeef: stale, now d@1a2b3c4d" in report.render().splitlines()


def test_a_citation_matches_by_prefix_either_way() -> None:
    full = "1a2b3c4d" * 8
    assert check("28.6% (d@1a2b3c4d)", {"d": SHARE}, {"d": full}).verdict == "faithful"
    assert check(f"28.6% (d@{full})", {"d": SHARE}, {"d": full[:8]}).verdict == (
        "faithful"
    )


def test_an_upstream_version_is_read_from_a_served_source_string() -> None:
    served = {"overlay": {"cut": -0.0912, "source": "cov_matrix@1a2b3c4d"}}
    current = check("Cut 9.1% (cov_matrix@1a2b3c4d).", served, {})
    assert current.verdict == "faithful"
    stale = check("Cut 9.1% (cov_matrix@deadbeef).", served, {})
    assert [c.status for c in stale.citations] == ["stale"]


def test_an_explicit_current_version_wins_over_a_source_string() -> None:
    served = {"overlay": {"source": "overlay@deadbeef"}}
    report = check("overlay@1a2b3c4d", served, {"overlay": "1a2b3c4d"})
    assert [c.status for c in report.citations] == ["current"]


def test_a_citation_nothing_served_names_is_unknown() -> None:
    report = check("28.6% (ghost@1a2b3c4d)", {"d": SHARE}, {})
    assert report.verdict == "unfaithful"
    assert [c.status for c in report.citations] == ["unknown"]
    assert "unknown, no served output names ghost" in report.render()


def test_a_repeated_citation_is_reported_once() -> None:
    report = check("d@1a2b3c4d and again d@1a2b3c4d", {}, {"d": "1a2b3c4d"})
    assert len(report.citations) == 1


def test_faithful_report_renders_every_check() -> None:
    report = check(
        "Churn was 28.6% of 1,847 (d@1a2b3c4d).",
        {"d": {"share": SHARE, "n": 1847}},
        {"d": "1a2b3c4d"},
    )
    assert report.render() == (
        "# Citations: **FAITHFUL**\n"
        "\n"
        "checked against: d\n"
        "2 of 2 number(s) supported\n"
        "\n"
        "numbers:\n"
        "- [ok] 28.6%: d $.share = 0.2857142857 (x100)\n"
        "- [ok] 1,847: d $.n = 1847\n"
        "\n"
        "citations:\n"
        "- [ok] d@1a2b3c4d: current"
    )


def test_a_plain_number_is_never_a_hundredth_of_a_served_value() -> None:
    report = check("3 outages, a ratio of 0.3", {"d": {"n": 312.0, "pct": 28.6}}, {})
    assert report.verdict == "unfaithful"
    assert not any(s.supported for s in report.numbers)


def test_a_huge_exponent_is_unsupported_not_a_crash() -> None:
    assert _verdict("1e309", 5) == "unfaithful"
    assert _verdict("0.0e400", 5) == "unfaithful"


def test_nothing_to_check_against_says_so() -> None:
    assert "checked against: (no served output)" in check("5", {}, {}).render()


# --- properties: Python's own rounding is the oracle ------------------------------


@given(
    x=st.floats(min_value=0, max_value=1e6, allow_nan=False), places=st.integers(0, 4)
)
@settings(deadline=None)
def test_a_value_supports_itself_as_python_formats_it(x: float, places: int) -> None:
    for text in (f"{x:,.{places}f}", f"{x * 100:.{places}f}%", f"{-x:.{places}f}"):
        expected = -x if text.startswith("-") else x
        assert _verdict(text, expected) == "faithful", text


@given(
    x=st.floats(min_value=0, max_value=1e6, allow_nan=False), places=st.integers(0, 4)
)
@settings(deadline=None)
def test_two_units_off_in_the_last_place_is_unsupported(x: float, places: int) -> None:
    written = Decimal(f"{x:.{places}f}") + 2 * Decimal(1).scaleb(-places)
    assert _verdict(f"{written:f}", x) == "unfaithful"
