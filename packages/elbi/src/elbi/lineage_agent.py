"""The lineage/catalog capabilities the chat agent drives, as model-shaped text.

Wraps :class:`~elbi.lineage.LineageService` so the agent can answer the
questions a human uses the catalog for: find an artifact, trace where a number comes
from, and see what a change would break. Read-only.
"""

from __future__ import annotations

from .lineage import LineageError, LineageService

#: Cap a rendered catalog listing so a large project stays readable in context.
_MAX_ROWS = 40


class LineageAgent:
    """Bind a :class:`LineageService` into the agent's ``lin_*`` capabilities."""

    def __init__(self, service: LineageService) -> None:
        self._service = service

    def catalog(self, query: str) -> str:
        """Search the catalog of every artifact (datasets, derivations, models, …)."""
        records = self._service.catalog(query)
        if not records:
            return f"No artifacts match {query!r}." if query else "No artifacts yet."
        lines = []
        for record in records[:_MAX_ROWS]:
            verdict = f" [{record['verdict']}]" if record.get("verdict") else ""
            lines.append(f"- {record['type']}: {record['name']}{verdict}")
        suffix = (
            f"\n…({len(records) - _MAX_ROWS} more)" if len(records) > _MAX_ROWS else ""
        )
        header = f"Catalog matches for {query!r}:" if query else "Catalog:"
        return header + "\n" + "\n".join(lines) + suffix

    def lineage(self, node: str) -> str:
        """Trace a node's provenance (what feeds it) and dependents (what it feeds)."""
        # Three questions -- resolve, walk up, walk down -- off one graph.
        view = self._service.snapshot()
        node_id = view.resolve(node)
        if node_id is None:
            return f"No artifact {node!r} (or the name is ambiguous; use type:name)."
        try:
            up = view.provenance(node_id)
            down = view.impact(node_id)
        except LineageError as exc:
            return str(exc)
        return (
            f"# Lineage of {node_id}\n"
            f"Feeds from: {_render(up['feeds_from'])}\n"
            f"Feeds into: {_render(down['affected'])}"
        )

    def impact(self, node: str) -> str:
        """What changing or deleting a node would affect, by type."""
        view = self._service.snapshot()
        node_id = view.resolve(node)
        if node_id is None:
            return f"No artifact {node!r} (or the name is ambiguous; use type:name)."
        try:
            result = view.impact(node_id)
        except LineageError as exc:
            return str(exc)
        if result["count"] == 0:
            return f"Nothing depends on {node_id}; changing it is safe."
        return (
            f"Changing {node_id} affects {result['count']} artifact(s): "
            f"{_render(result['affected'])}"
        )


def _render(by_type: dict[str, list[str]]) -> str:
    if not by_type:
        return "(nothing)"
    return "; ".join(f"{t} {', '.join(names)}" for t, names in sorted(by_type.items()))
