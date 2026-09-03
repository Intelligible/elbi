"""The monitoring capabilities the chat agent drives, as model-shaped text.

Wraps :class:`~elbi.monitoring.MonitorService` so the agent can set up a
monitor over a certified metric or derivation and report what is watched -- the same
anomaly monitoring a human configures in the UI. A monitor watches only a certified
target, so an alert always concerns a verified number.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .monitoring import MonitorError, MonitorService


class MonitorsAgent:
    """Bind a :class:`MonitorService` into the agent's ``mon_*`` capabilities."""

    def __init__(self, service: MonitorService) -> None:
        self._service = service

    def list_monitors(self) -> str:
        """List the monitors with their target, latest value, and alert status."""
        monitors = self._service.list_monitors()
        if not monitors:
            return "No monitors defined yet."
        lines = []
        for m in monitors:
            value = "?" if m.last_value is None else f"{m.last_value:g}"
            lines.append(
                f"- {m.name} watches {m.target_kind} {m.target!r}: "
                f"last {value}, status {m.status}"
            )
        return "Monitors:\n" + "\n".join(lines)

    def create(self, spec: Mapping[str, Any]) -> str:
        """Create a monitor from a spec object (target_kind, target, method, ...)."""
        try:
            view = self._service.create(
                name=str(spec.get("name") or ""),
                target_kind=str(spec.get("target_kind") or "metric"),
                target=str(spec.get("target") or ""),
                config=dict(spec.get("config") or {}),
                method=str(spec.get("method") or "mad"),
                sensitivity=float(spec.get("sensitivity", 3.0)),
                min_value=spec.get("min_value"),
                max_value=spec.get("max_value"),
                window=int(spec.get("window", 30)),
                interval_hours=float(spec.get("interval_hours", 1.0)),
            )
        except (MonitorError, ValueError) as exc:
            return f"Could not create the monitor: {exc}"
        return (
            f"Monitoring {view.target_kind} {view.target!r} as {view.name!r} "
            f"({view.method}, sensitivity {view.sensitivity:g})."
        )
