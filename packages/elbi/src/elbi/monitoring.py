"""The monitoring service: watch a metric or derivation and alert on anomalies.

A monitor snapshots its target's value on each check, learns a baseline from the
recent snapshot history, and flags a value the anomaly detector finds out of
line (or outside a static bound). Consecutive anomalies fold into one **incident** so a
run of bad values raises one alert, not one per check; the incident closes when values
return to normal. Because a monitor only watches a certified metric or derivation, an
alert carries the source's oracle verdict -- the moved number was a verified one.

The value source, the certified gate, and alert delivery are injected by the app; this
service owns the snapshot, detection, and incident bookkeeping.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from elbi_core import detect_anomaly
from elbi_core.errors import ElbiError

from .db import MetricMonitor, MetricSnapshot, MonitorIncident, Store
from .wire import Monitor as WireMonitor
from .wire import MonitorHistory as WireMonitorHistory
from .wire import MonitorIncident as WireMonitorIncident
from .wire import MonitorSnapshot as WireMonitorSnapshot

#: Alert payloads carry one of these events.
ANOMALY_DETECTED = "metric.anomaly_detected"
RECOVERED = "metric.recovered"

_KINDS = ("metric", "derivation")


class MonitorError(Exception):
    """A monitor operation failed for a reason worth showing the caller."""


class MonitorService:
    """Create, run, and inspect anomaly monitors over metrics and derivations."""

    def __init__(
        self,
        store: Store,
        read_value: Callable[[MetricMonitor], float],
        source_certified: Callable[[str, str], bool],
        on_alert: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._store = store
        # read_value resolves a monitor's current scalar; source_certified gates the
        # target (and marks the snapshot's verdict); on_alert delivers a fired alert.
        self._read_value = read_value
        self._source_certified = source_certified
        self._on_alert = on_alert

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
        """Run one check of ``monitor`` (used by the scheduler and by ``check``)."""
        try:
            value = float(self._read_value(monitor))
        except (ElbiError, KeyError, TypeError, ValueError) as exc:
            raise MonitorError(f"could not read {monitor.target!r}: {exc}") from exc
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
