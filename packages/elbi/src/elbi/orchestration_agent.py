"""The orchestration capabilities the chat agent drives, as model-shaped text.

Wraps :class:`~elbi.orchestration.OrchestrationService` so the agent can see which
assets are stale and materialize them in dependency order, the same operational control
a human has. Materializing is idempotent, so it is safe to call.
"""

from __future__ import annotations

from .orchestration import OrchestrationService

_STATUS_ICON = {"materialized": "✓", "stale": "△", "never": "·"}


class OrchestrationAgent:
    """Bind an :class:`OrchestrationService` into the agent's ``orch_*`` tools."""

    def __init__(self, service: OrchestrationService) -> None:
        self._service = service

    def status(self) -> str:
        """List each asset's freshness (materialized / stale / never-materialized)."""
        rows = self._service.status()
        if not rows:
            return "No assets to materialize."
        stale = [r["asset"] for r in rows if r["status"] in ("stale", "never")]
        lines = [
            f"{_STATUS_ICON.get(r['status'], '?')} {r['asset']} ({r['status']})"
            for r in rows
        ]
        summary = (
            f"{len(stale)} of {len(rows)} assets need materializing."
            if stale
            else "All assets are up to date."
        )
        return summary + "\n" + "\n".join(lines)

    def materialize(self, selection: str, assets: str) -> str:
        """Materialize assets in dependency order (selection 'all'/'stale' or names)."""
        names = [a.strip() for a in assets.split(",") if a.strip()] if assets else None
        result = self._service.materialize(
            selection=selection or "stale", assets=names, include_downstream=bool(names)
        )
        parts = []
        if result["materialized"]:
            parts.append(f"materialized {', '.join(result['materialized'])}")
        if result["skipped"]:
            parts.append(f"skipped {len(result['skipped'])} already-fresh")
        if result["failed"]:
            parts.append(f"FAILED {', '.join(result['failed'])}")
        if not parts:
            return "Nothing to do."
        return "Run complete: " + "; ".join(parts) + "."
