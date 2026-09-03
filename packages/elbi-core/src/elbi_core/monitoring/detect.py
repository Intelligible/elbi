"""Detect whether a monitored value is anomalous, given its recent history.

Two complementary tests, either of which can flag a value:

* **Learned baseline.** Over the recent history, estimate a center and a spread and flag
  a value that sits too many spreads away. The default (``"mad"``) uses the median and a
  scaled median absolute deviation, which is robust: a past spike does not inflate the
  band and mask the next one. ``"zscore"`` uses the mean and standard deviation, the
  familiar (less robust) alternative.
* **Static bounds.** A hard floor and/or ceiling the value must stay within, independent
  of history (a rate that must not exceed 1.0, a count that must stay positive).

The learned test needs ``min_history`` prior points; before that only the static
bounds apply, so a monitor still catches a hard-limit breach on day one. The function is
pure and dependency-free.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

#: Scale factor making the median absolute deviation a consistent estimator of the
#: standard deviation for normal data, so ``sensitivity`` reads in familiar sigma units.
_MAD_TO_SIGMA = 1.4826

#: Detection methods for the learned baseline.
METHODS = ("mad", "zscore")


@dataclass(frozen=True)
class AnomalyVerdict:
    """Whether a value is anomalous, with the band and reason behind the call.

    ``baseline`` is the estimated center (median or mean) and ``lower``/``upper`` the
    band the value was expected to fall in (from the learned test, static bounds, or
    both, whichever is tighter). ``score`` is the value's distance from the baseline in
    spread units (``None`` when no baseline could be learned). ``reason`` is a one-line
    explanation for an alert.
    """

    anomalous: bool
    value: float
    baseline: float | None
    lower: float | None
    upper: float | None
    score: float | None
    reason: str


def detect_anomaly(
    history: Sequence[float],
    value: float,
    *,
    method: str = "mad",
    sensitivity: float = 3.0,
    min_value: float | None = None,
    max_value: float | None = None,
    min_history: int = 5,
) -> AnomalyVerdict:
    """Return whether ``value`` is anomalous given ``history`` (oldest to newest).

    ``method`` selects the learned-baseline estimator (``"mad"`` or ``"zscore"``);
    ``sensitivity`` is how many spreads from the baseline are allowed before a flag
    (smaller is more sensitive). ``min_value``/``max_value`` are optional static bounds.
    The learned test is skipped until ``history`` has at least ``min_history`` points.

    Raises:
        ValueError: if ``method`` is unknown or ``sensitivity`` is not positive.
    """
    if method not in METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {', '.join(METHODS)}"
        )
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")

    # Static bounds bite first and unconditionally: a hard limit needs no history.
    if min_value is not None and value < min_value:
        return AnomalyVerdict(
            True,
            value,
            None,
            min_value,
            max_value,
            None,
            f"{value:g} is below the floor {min_value:g}",
        )
    if max_value is not None and value > max_value:
        return AnomalyVerdict(
            True,
            value,
            None,
            min_value,
            max_value,
            None,
            f"{value:g} is above the ceiling {max_value:g}",
        )

    points = [float(h) for h in history]
    if len(points) < min_history:
        return AnomalyVerdict(
            False,
            value,
            None,
            min_value,
            max_value,
            None,
            f"within static bounds; {len(points)} of {min_history} points needed "
            "to learn a baseline",
        )

    center, spread = _center_spread(points, method)
    # A flat history has zero spread; only an exact-match value is then "normal", and a
    # departure of any size is anomalous. Guard the division and say so plainly.
    if spread == 0:
        anomalous = value != center
        return AnomalyVerdict(
            anomalous,
            value,
            center,
            center,
            center,
            None,
            f"history is flat at {center:g}; value {value:g} "
            + ("departs from it" if anomalous else "matches it"),
        )

    lower = center - sensitivity * spread
    upper = center + sensitivity * spread
    score = (value - center) / spread
    anomalous = abs(score) > sensitivity
    if anomalous:
        direction = "above" if score > 0 else "below"
        reason = (
            f"{value:g} is {abs(score):.1f} std devs {direction} "
            f"the baseline {center:g}"
        )
    else:
        reason = f"{value:g} is within {sensitivity:g} std devs of baseline {center:g}"
    return AnomalyVerdict(anomalous, value, center, lower, upper, score, reason)


def _center_spread(points: list[float], method: str) -> tuple[float, float]:
    """The baseline center and one-unit spread for the chosen method."""
    if method == "zscore":
        mean = statistics.fmean(points)
        # Population stdev: the history is the whole record, not a sample of it.
        return mean, statistics.pstdev(points)
    median = statistics.median(points)
    mad = statistics.median([abs(p - median) for p in points])
    return median, mad * _MAD_TO_SIGMA
