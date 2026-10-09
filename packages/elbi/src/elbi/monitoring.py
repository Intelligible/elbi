"""The monitoring service: watch a metric or derivation and alert on anomalies.

A monitor snapshots its target's value on each check, learns a baseline from the
recent snapshot history, and flags a value the anomaly detector finds out of
line (or outside a static bound). Consecutive anomalies fold into one **incident** so a
run of bad values raises one alert, not one per check; the incident closes when values
return to normal. A monitor only watches a certified metric or derivation, so an alert
includes the source's verification verdict and the number that moved is a verified one.

A source that cannot be read at all is an incident too, raised through the same alert
path: a monitor whose metric query or derivation stops evaluating would otherwise go
quiet, which reads exactly like a healthy value. Consecutive failures fold into one
incident, and the next readable value closes it, as a recovery or as a fresh anomaly.

A derivation monitor can also name a ``key`` column next to its measure. The reading
then carries the keys of the rows that breach (a positive measure value), every alert
lists them, and a breach that widens while its incident is open raises
``metric.breach_widened``. Without it, an incident opened by one failing row absorbs
every later one in silence: the sum moves from 1 to 2 and folds into the open incident,
so the second failure is never announced for as long as the first one stays broken.

The value source, the certified gate, and alert delivery are injected by the app; this
service owns the snapshot, detection, and incident bookkeeping.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from elbi_core import detect_anomaly

from .db import MetricMonitor, MetricSnapshot, MonitorIncident, Store
from .wire import Monitor as WireMonitor
from .wire import MonitorHistory as WireMonitorHistory
from .wire import MonitorIncident as WireMonitorIncident
from .wire import MonitorSnapshot as WireMonitorSnapshot

#: Alert payloads carry one of these events.
ANOMALY_DETECTED = "metric.anomaly_detected"
RECOVERED = "metric.recovered"
SOURCE_FAILED = "metric.source_failed"
BREACH_WIDENED = "metric.breach_widened"

#: The ``cause`` of an incident opened because the source could not be read. An anomaly
#: incident carries ``None``, which is also what rows predating the column hold.
_SOURCE_FAILED_CAUSE = "source_failed"

_KINDS = ("metric", "derivation")


class MonitorError(Exception):
    """A monitor operation failed for a reason worth showing the caller."""


class Reading:
    """A monitor value plus the keys of its breaching rows (``None`` when unkeyed)."""

    __slots__ = ("keys", "value")

    def __init__(self, value: float, keys: tuple[str, ...] | None = None) -> None:
        self.value = float(value)
        self.keys = keys


class MonitorService:
    """Create, run, and inspect anomaly monitors over metrics and derivations."""

    def __init__(
        self,
        store: Store,
        read_value: Callable[[MetricMonitor], float | Reading],
        source_certified: Callable[[str, str], bool],
        on_alert: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._store = store
        # read_value resolves a monitor's current scalar; source_certified gates the
        # target (and marks the snapshot's verdict); on_alert delivers a fired alert.
        self._read_value = read_value
        self._source_certified = source_certified
        self._on_alert = on_alert
        # The breaching keys each keyed monitor last read, to tell a widened breach from
        # one already reported. In memory: after a restart the first reading inside an
        # open incident only re-seeds it, so a restart never re-alerts on old keys.
        self._last_keys: dict[str, frozenset[str]] = {}

    def create(
        self,
        name: str,
        target_kind: str,
        target: str,
        config: dict[str, Any] | None = None,
        method: str = "mad",
        sensitivity: float = 3.0,
        min_value: float | None = None,
        max_value: float | None = None,
        window: int = 30,
        interval_hours: float = 1.0,
    ) -> WireMonitor:
        """Create a monitor over a certified metric or derivation.

        Raises:
            MonitorError: on an unknown kind or a target that is absent or uncertified.
        """
        if target_kind not in _KINDS:
            raise MonitorError(f"target_kind must be one of {', '.join(_KINDS)}")
        if not self._source_certified(target_kind, target):
            raise MonitorError(
                f"{target_kind} {target!r} is not certified; a monitor watches "
                "only verified numbers"
            )
        monitor = MetricMonitor(
            name=name.strip() or target,
            target_kind=target_kind,
            target=target,
            config_json=json.dumps(config or {}),
            method=method,
            sensitivity=sensitivity,
            min_value=min_value,
            max_value=max_value,
            window=window,
            interval_hours=interval_hours,
        )
        monitor_id = monitor.id
        self._store.save_metric_monitor(monitor)
        saved = self._store.get_metric_monitor(monitor_id)
        if saved is None:  # pragma: no cover - just written
            raise MonitorError("monitor could not be read back")
        return self._view(saved)

    def update(
        self,
        monitor_id: str,
        name: str | None = None,
        target_kind: str | None = None,
        target: str | None = None,
        config: dict[str, Any] | None = None,
        method: str | None = None,
        sensitivity: float | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        window: int | None = None,
        interval_hours: float | None = None,
    ) -> WireMonitor | None:
        """Change a monitor's settings in place; ``None`` if it does not exist.

        In place because a monitor's history is the monitor. Deleting one takes its
        snapshots and incidents with it, and those snapshots are the baseline anomaly
        detection compares against -- so re-creating a monitor to change a threshold
        silently resets the detector and discards every incident it raised.

        Retargeting is checked the way creating is: a monitor watches only certified
        numbers, and an edit must not be a way around that.

        Raises:
            MonitorError: on an unknown kind or a target that is absent or uncertified.
        """
        row = self._store.get_metric_monitor(monitor_id)
        if row is None:
            return None
        if target_kind is not None and target_kind not in _KINDS:
            raise MonitorError(f"target_kind must be one of {', '.join(_KINDS)}")
        kind = target_kind or row.target_kind
        watched = target if target is not None else row.target
        retargeted = target_kind is not None or target is not None
        if retargeted and not self._source_certified(kind, watched):
            raise MonitorError(
                f"{kind} {watched!r} is not certified; a monitor watches "
                "only verified numbers"
            )
        row.name = (name.strip() or watched) if name is not None else row.name
        row.target_kind = kind
        row.target = watched
        if config is not None:
            row.config_json = json.dumps(config)
        if method is not None:
            row.method = method
        if sensitivity is not None:
            row.sensitivity = sensitivity
        if window is not None:
            row.window = window
        if interval_hours is not None:
            row.interval_hours = interval_hours
        # A bound is cleared by naming it as null, so these are set unconditionally: a
        # threshold somebody removed from the file has to come off the monitor too.
        row.min_value = min_value
        row.max_value = max_value
        self._store.update_metric_monitor(row)
        saved = self._store.get_metric_monitor(monitor_id)
        return self._view(saved) if saved is not None else None

    def list_monitors(self) -> list[WireMonitor]:
        """The caller's monitors, each with its latest value and open-incident state."""
        return [self._view(m) for m in self._store.list_metric_monitors()]

    def get(self, monitor_id: str) -> WireMonitor | None:
        """A monitor's view, or ``None`` when it does not exist."""
        row = self._store.get_metric_monitor(monitor_id)
        return self._view(row) if row is not None else None

    def delete(self, monitor_id: str) -> bool:
        """Delete a monitor and its history; return whether it existed."""
        return self._store.delete_metric_monitor(monitor_id)

    def history(self, monitor_id: str) -> WireMonitorHistory:
        """A monitor's snapshot history (oldest to newest) and its incidents.

        Raises:
            MonitorError: if the monitor is unknown.
        """
        if self._store.get_metric_monitor(monitor_id) is None:
            raise MonitorError("monitor not found")
        snaps = list(reversed(self._store.list_metric_snapshots(monitor_id)))
        incidents = self._store.list_incidents(monitor_id)
        return WireMonitorHistory(
            snapshots=[_snapshot_view(s) for s in snaps],
            incidents=[_incident_view(i) for i in incidents],
        )

    def check(self, monitor_id: str) -> dict[str, Any]:
        """Snapshot a monitor's value now, detect an anomaly, and manage its incident.

        Raises:
            MonitorError: if the monitor is unknown or its value cannot be read.
        """
        monitor = self._store.get_metric_monitor(monitor_id)
        if monitor is None:
            raise MonitorError("monitor not found")
        return self.run(monitor)

    def run(self, monitor: MetricMonitor) -> dict[str, Any]:
        """Run one check of ``monitor`` (used by the scheduler and by ``check``).

        Raises:
            MonitorError: if the source could not be read, after alerting on it.
        """
        try:
            raw = self._read_value(monitor)
            if isinstance(raw, Reading):
                value, keys = raw.value, raw.keys
            else:
                value, keys = float(raw), None
        except Exception as exc:
            # Any failure to evaluate the source counts, not only the library's own
            # error types: a metric query raises its service's error, and an exception
            # escaping here would also stop the scheduler's pass over later monitors.
            error = MonitorError(f"could not read {monitor.target!r}: {exc}")
            failure = self._source_failed(monitor, str(error))
            if failure is not None and self._on_alert is not None:
                self._on_alert(failure)
            raise error from exc
        # History oldest-to-newest, excluding the value we are about to record.
        recent = self._store.list_metric_snapshots(monitor.id, monitor.window)
        history = [s.value for s in reversed(recent)]
        verdict = detect_anomaly(
            history,
            value,
            method=monitor.method,
            sensitivity=monitor.sensitivity,
            min_value=monitor.min_value,
            max_value=monitor.max_value,
        )
        source_verdict = (
            "sound"
            if self._source_certified(monitor.target_kind, monitor.target)
            else None
        )
        self._store.add_metric_snapshot(
            MetricSnapshot(
                monitor_id=monitor.id,
                value=value,
                anomalous=verdict.anomalous,
                baseline=verdict.baseline,
                lower=verdict.lower,
                upper=verdict.upper,
                score=verdict.score,
                reason=verdict.reason,
                source_verdict=source_verdict,
            )
        )
        alert = self._manage_incident(monitor, verdict, value, source_verdict)
        if keys is not None:
            alert = self._track_keys(
                monitor, verdict, value, source_verdict, keys, alert
            )
        self._store.touch_metric_monitor(monitor.id, _now())
        if alert is not None and self._on_alert is not None:
            self._on_alert(alert)
        return {
            "value": value,
            "anomalous": verdict.anomalous,
            "baseline": verdict.baseline,
            "lower": verdict.lower,
            "upper": verdict.upper,
            "score": verdict.score,
            "reason": verdict.reason,
            "alerted": alert is not None,
        }

    def _manage_incident(
        self,
        monitor: MetricMonitor,
        verdict: Any,
        value: float,
        source_verdict: str | None,
    ) -> dict[str, Any] | None:
        open_incident = self._store.open_incident(monitor.id)
        if open_incident is not None and open_incident.cause == _SOURCE_FAILED_CAUSE:
            # The source reads again, which ends the failure. A normal value is a
            # recovery; an anomalous one opens an incident of its own below.
            open_incident.closed_at = _now()
            self._store.save_incident(open_incident)
            if not verdict.anomalous:
                return self._alert(monitor, RECOVERED, verdict, value, source_verdict)
            open_incident = None
        if verdict.anomalous:
            if open_incident is None:
                self._store.save_incident(
                    MonitorIncident(
                        monitor_id=monitor.id,
                        peak_value=value,
                        peak_score=verdict.score,
                        reason=verdict.reason,
                    )
                )
                return self._alert(
                    monitor, ANOMALY_DETECTED, verdict, value, source_verdict
                )
            # Fold into the open incident: track the worst point, raise no new alert.
            open_incident.snapshots += 1
            if verdict.score is not None and (
                open_incident.peak_score is None
                or abs(verdict.score) > abs(open_incident.peak_score)
            ):
                open_incident.peak_value = value
                open_incident.peak_score = verdict.score
                open_incident.reason = verdict.reason
            self._store.save_incident(open_incident)
            return None
        if open_incident is not None:
            open_incident.closed_at = _now()
            open_incident.snapshots += 1
            self._store.save_incident(open_incident)
            return self._alert(monitor, RECOVERED, verdict, value, source_verdict)
        return None

    def _track_keys(
        self,
        monitor: MetricMonitor,
        verdict: Any,
        value: float,
        source_verdict: str | None,
        keys: tuple[str, ...],
        alert: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Attach the breaching keys to ``alert``, or raise one for a widened breach."""
        current = frozenset(keys)
        previous = self._last_keys.get(monitor.id)
        self._last_keys[monitor.id] = current
        if alert is not None:
            alert["breach_keys"] = sorted(current)
            return alert
        if not verdict.anomalous or previous is None:
            return None
        added = current - previous
        if not added:
            return None
        widened = self._alert(monitor, BREACH_WIDENED, verdict, value, source_verdict)
        widened["breach_keys"] = sorted(current)
        widened["new_keys"] = sorted(added)
        return widened

    def _source_failed(
        self, monitor: MetricMonitor, reason: str
    ) -> dict[str, Any] | None:
        """Record a failed read as an incident; the alert to raise, if it is new.

        A failure already open absorbs this one without a second alert. An open anomaly
        is closed first: its value can no longer be judged, so the failure is now the
        problem to report, and a later anomaly alerts afresh.
        """
        open_incident = self._store.open_incident(monitor.id)
        if open_incident is not None and open_incident.cause == _SOURCE_FAILED_CAUSE:
            open_incident.snapshots += 1
            open_incident.reason = reason
            self._store.save_incident(open_incident)
            return None
        if open_incident is not None:
            open_incident.closed_at = _now()
            self._store.save_incident(open_incident)
        self._store.save_incident(
            MonitorIncident(
                monitor_id=monitor.id, reason=reason, cause=_SOURCE_FAILED_CAUSE
            )
        )
        return {
            "event": SOURCE_FAILED,
            "monitor_id": monitor.id,
            "monitor": monitor.name,
            "target_kind": monitor.target_kind,
            "target": monitor.target,
            "reason": reason,
        }

    def _alert(
        self,
        monitor: MetricMonitor,
        event: str,
        verdict: Any,
        value: float,
        source_verdict: str | None,
    ) -> dict[str, Any]:
        return {
            "event": event,
            "monitor_id": monitor.id,
            "monitor": monitor.name,
            # Popped by the notification handler before the webhook sees the payload.
            "target_kind": monitor.target_kind,
            "target": monitor.target,
            "value": value,
            "baseline": verdict.baseline,
            "lower": verdict.lower,
            "upper": verdict.upper,
            "score": verdict.score,
            "reason": verdict.reason,
            "source_verdict": source_verdict,
        }

    def _view(self, monitor: MetricMonitor) -> WireMonitor:
        snaps = self._store.list_metric_snapshots(monitor.id, 1)
        latest = snaps[0] if snaps else None
        open_incident = self._store.open_incident(monitor.id)
        return WireMonitor(
            id=monitor.id,
            name=monitor.name,
            target_kind=monitor.target_kind,
            target=monitor.target,
            config=json.loads(monitor.config_json or "{}"),
            method=monitor.method,
            sensitivity=monitor.sensitivity,
            min_value=monitor.min_value,
            max_value=monitor.max_value,
            window=monitor.window,
            interval_hours=monitor.interval_hours,
            enabled=monitor.enabled,
            last_value=latest.value if latest is not None else None,
            last_checked_at=_iso(monitor.last_run_at),
            status="alerting" if open_incident is not None else "ok",
        )


def _snapshot_view(snapshot: MetricSnapshot) -> WireMonitorSnapshot:
    return WireMonitorSnapshot(
        at=_iso(snapshot.at),
        value=snapshot.value,
        anomalous=snapshot.anomalous,
        baseline=snapshot.baseline,
        lower=snapshot.lower,
        upper=snapshot.upper,
        score=snapshot.score,
        reason=snapshot.reason,
        source_verdict=snapshot.source_verdict,
    )


def _incident_view(incident: MonitorIncident) -> WireMonitorIncident:
    return WireMonitorIncident(
        id=incident.id,
        opened_at=_iso(incident.opened_at),
        closed_at=_iso(incident.closed_at),
        peak_value=incident.peak_value,
        peak_score=incident.peak_score,
        reason=incident.reason,
        snapshots=incident.snapshots,
        open=incident.closed_at is None,
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _now() -> datetime:
    return datetime.now(timezone.utc)
