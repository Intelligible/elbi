"""The dashboard capabilities the chat agent drives, as model-shaped text.

Wraps :class:`~elbi.dashboards.DashboardService` so the agent operates the same
dashboard surface a human does; see which certified derivations it can bind, list and
read dashboards, author or edit one's spec, and publish it. A dashboard is a declarative
artifact, so authoring is autonomous (the agent composes the spec), but publishing goes
through the same certification gate the UI enforces: a dashboard can only publish once
every derivation it binds is certified, so the gate, not the agent, is what lets a
number reach a viewer.

Everything the agent does shows up in the UI to inspect, edit, or take over.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .dashboards import DashboardError, DashboardService


class DashboardAgent:
    """Bind a :class:`DashboardService` into the agent's ``dash_*`` capabilities."""

    def __init__(self, service: DashboardService) -> None:
        self._service = service

    def list(self) -> str:
        """List the dashboards that exist."""
        rows = self._service.list_dashboards()
        if not rows:
            return "No dashboards yet. Create one with write_dashboard."
        lines = [
            f"- {r['title'] or r['name']} (id {r['id']}, "
            f"{r['status']}, v{r['version']})"
            for r in rows
        ]
        return "Dashboards:\n" + "\n".join(lines)

    def sources(self) -> str:
        """List the certified derivations a widget can bind, with their parameters."""
        catalog = self._service.catalog()
        if not catalog:
            return (
                "No certified derivations to bind yet. Author one with derive first; "
                "only certified derivations can back a dashboard widget."
            )
        lines = []
        for entry in catalog:
            params = ", ".join(entry.get("params", {}).keys()) or "no params"
            lines.append(f"- {entry['name']} ({params}): {entry['title']}")
        return "Certified derivations you can bind:\n" + "\n".join(lines)

    def read(self, dashboard_id: str) -> str:
        """Render a dashboard's variables, pages, widgets, and status."""
        try:
            view = self._service.get(dashboard_id)
        except DashboardError as exc:
            return str(exc)
        spec = view["spec"]
        parts = [
            f"# Dashboard: {view['title'] or view['name']} "
            f"(id {view['id']}, {view['status']}, v{view['version']})"
        ]
        variables = spec.get("variables", [])
        if variables:
            described = ", ".join(
                f"{v['name']} ({v.get('control', 'dropdown')})" for v in variables
            )
            parts.append("Variables: " + described)
        for page in spec.get("pages", []):
            parts.append(f"\n## page {page['name']}")
            for widget in page.get("widgets", []):
                bind = widget.get("bind", {})
                target = f" → {bind['derivation']}" if bind else ""
                pos = widget["gridPos"]
                parts.append(
                    f"- {widget['id']} ({widget['type']}){target} "
                    f"@ [{pos['x']},{pos['y']} {pos['w']}x{pos['h']}]"
                )
        return "\n".join(parts)

    def write(self, dashboard_id: str | None, spec: Mapping[str, Any]) -> str:
        """Create a dashboard (no id) or replace an existing one's spec.

        ``spec`` is the full Dashboard manifest. Validation errors are returned so the
        agent can fix and retry; a bound derivation need not be certified to save, only
        to publish.
        """
        manifest = dict(spec)
        try:
            if dashboard_id:
                result = self._service.save(dashboard_id, manifest)
            else:
                result = self._service.create(manifest)
        except DashboardError as exc:
            return f"Invalid dashboard (not saved): {exc}"
        return (
            f"Saved '{result['title']}' (id {result['id']}, v{result['version']}). "
            f"URL: /dashboards/{result['id']}. "
            "Publish it with publish_dashboard once its bindings are certified."
        )

    def publish(self, dashboard_id: str) -> str:
        """Publish a dashboard, gating on every bound derivation being certified."""
        try:
            result = self._service.publish(dashboard_id, require_certified=True)
        except DashboardError as exc:
            return str(exc)
        if result.ok:
            return (
                f"Published dashboard {dashboard_id}. Viewers now see it, and every "
                "tile is backed by a certified derivation."
            )
        return (
            "Not published: the certification gate refused it: "
            + result.detail
            + ". Author and certify the missing derivations (with derive), then retry."
        )
