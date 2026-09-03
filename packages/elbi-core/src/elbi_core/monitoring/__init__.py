"""Anomaly detection for monitored values, over a rolling history.

A monitor watches a number over time (a metric's value, a derivation's row count) and
asks, each time it takes a snapshot, whether the new value is anomalous given what the
value has been. The detection is a pure function of the history and the value, so it is
testable in isolation and identical wherever it runs; the app supplies the history,
persists snapshots, and raises alerts.

The default method is a robust learned baseline (median plus scaled median-absolute-
deviation), which the modern practice favors over a mean/standard-deviation band because
the median does not get dragged toward the very outliers the monitor is looking for.
"""

from .detect import AnomalyVerdict, detect_anomaly

__all__ = ["AnomalyVerdict", "detect_anomaly"]
