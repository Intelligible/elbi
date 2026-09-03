"""Tests for the anomaly detector (:mod:`elbi.monitoring`)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elbi_core import AnomalyVerdict, detect_anomaly

STEADY = [100.0, 101.0, 99.0, 100.0, 102.0, 98.0, 100.0, 101.0]


def test_value_in_line_with_history_is_normal() -> None:
    verdict = detect_anomaly(STEADY, 100.5)
    assert isinstance(verdict, AnomalyVerdict)
    assert verdict.anomalous is False
    assert verdict.baseline is not None


def test_a_spike_is_flagged() -> None:
    verdict = detect_anomaly(STEADY, 500.0)
    assert verdict.anomalous is True
    assert verdict.score is not None and verdict.score > 3
    assert "above the baseline" in verdict.reason


def test_mad_is_robust_to_a_past_spike() -> None:
    # One historical outlier would inflate a stdev band and hide the next spike; the
    # median-based band shrugs it off and still flags the new one.
    history = [*STEADY, 900.0]
    assert detect_anomaly(history, 500.0, method="mad").anomalous is True


def test_static_bounds_apply_before_any_history() -> None:
    below = detect_anomaly([], -1.0, min_value=0.0)
    assert below.anomalous is True and "below the floor" in below.reason
    above = detect_anomaly([], 5.0, max_value=1.0)
    assert above.anomalous is True and "above the ceiling" in above.reason


def test_insufficient_history_only_checks_bounds() -> None:
    verdict = detect_anomaly([1.0, 2.0], 1000.0)  # fewer than min_history points
    assert verdict.anomalous is False
    assert "needed to learn a baseline" in verdict.reason


def test_flat_history_flags_any_departure() -> None:
    flat = [7.0] * 6
    assert detect_anomaly(flat, 7.0).anomalous is False
    assert detect_anomaly(flat, 7.5).anomalous is True


def test_sensitivity_widens_the_band() -> None:
    # A moderate deviation flags at a tight sensitivity but not a loose one.
    assert detect_anomaly(STEADY, 108.0, sensitivity=1.5).anomalous is True
    assert detect_anomaly(STEADY, 108.0, sensitivity=6.0).anomalous is False


def test_zscore_method_also_flags_a_spike() -> None:
    assert detect_anomaly(STEADY, 500.0, method="zscore").anomalous is True


def test_unknown_method_and_bad_sensitivity_raise() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        detect_anomaly(STEADY, 1.0, method="wizardry")
    with pytest.raises(ValueError, match="sensitivity must be positive"):
        detect_anomaly(STEADY, 1.0, sensitivity=0)


@given(
    history=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=100,
    ),
    method=st.sampled_from(["mad", "zscore"]),
)
def test_a_value_equal_to_the_center_is_never_anomalous(
    history: list[float], method: str
) -> None:
    # The baseline center (median or mean) is by construction inside its own band, so a
    # value equal to it is never flagged: a property that must hold for any history.
    verdict = detect_anomaly(history, history[len(history) // 2], method=method)
    center = verdict.baseline
    if center is not None:
        assert detect_anomaly(history, center, method=method).anomalous is False


@given(
    center=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    spread=st.floats(min_value=1.0, max_value=100.0),
)
def test_far_enough_values_are_always_anomalous(center: float, spread: float) -> None:
    # A synthetic spread-out history: a value many spreads beyond the max must flag.
    history = [center - spread, center, center + spread] * 3
    far = max(history) + 1000 * spread
    assert detect_anomaly(history, far, sensitivity=3.0).anomalous is True
