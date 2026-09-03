"""The metrics capabilities the chat agent drives, as model-shaped text.

Wraps :class:`~elbi.metrics.MetricService` so the agent can define a metric
over a certified derivation, list what exists, and query a metric by dimension and grain
-- the same semantic layer a human drives in the UI. Defining is governed the same way:
a simple metric's source must be certified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .metrics import MetricService, MetricServiceError

#: Cap a rendered metric query so a large result stays readable in context.
_MAX_ROWS = 50


class MetricsAgent:
    """Bind a :class:`MetricService` into the agent's ``metric_*`` capabilities."""

    def __init__(self, service: MetricService) -> None:
        self._service = service

    def list_metrics(self) -> str:
        """List the defined metrics with their type and source."""
        metrics = self._service.list_metrics()
        if not metrics:
            return "No metrics defined yet."
        lines = []
        for m in metrics:
            flag = "" if m.get("sourceCertified", True) else " (source uncertified)"
            detail = m.get("source") or f"{m.get('numerator')}/{m.get('denominator')}"
            lines.append(f"- {m['name']} [{m['type']}] over {detail}{flag}")
        return "Metrics:\n" + "\n".join(lines)

    def define(self, metric: Mapping[str, Any]) -> str:
        """Define (or replace) a metric from its spec manifest."""
        try:
            view = self._service.define(dict(metric))
        except MetricServiceError as exc:
            return f"Could not define the metric: {exc}"
        return (
            f"Defined metric {view['name']!r} ({view['type']}). "
            "Query it with query_metric."
        )

    def query(
        self, name: str, group_by: Sequence[str] = (), grain: str | None = None
    ) -> str:
        """Resolve a metric, grouped by dimensions and rolled up to a grain."""
        try:
            result = self._service.query(name, group_by=list(group_by), grain=grain)
        except MetricServiceError as exc:
            return str(exc)
        return _render_table(result["columns"], result["rows"])


def _render_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """A compact markdown table of a metric result, trimmed to the row cap."""
    if not rows:
        return "No rows."
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_cell(row.get(c)) for c in columns) + " |"
        for row in rows[:_MAX_ROWS]
    ]
    suffix = f"\n…({len(rows) - _MAX_ROWS} more rows)" if len(rows) > _MAX_ROWS else ""
    return "\n".join([header, divider, *body]) + suffix


def _cell(value: Any) -> str:
    return "" if value is None else str(value)
