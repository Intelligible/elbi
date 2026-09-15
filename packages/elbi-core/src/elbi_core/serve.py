"""Serve contracts: how a derivation's artifact is presented to an agent.

A :class:`Serve` value is declarative; it is part of the derivation manifest and
validates against the spec. Build one with the module-level helpers, which read
better at the call site than the initializer::

    serve.table(title="Churn risk", max_rows=50)
    serve.markdown(title="Pricing effects")
    serve.json(indent=2)
    serve.text()
    serve.components(title="Churn components")
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any, Literal

from .artifact import Artifact

ServeFormat = Literal["table", "markdown", "json", "text", "components"]


@dataclass(frozen=True)
class Serve:
    """The serve contract for a derivation.

    Prefer the :func:`table`, :func:`markdown`, :func:`json`, and :func:`text`
    builders over constructing this directly.
    """

    format: ServeFormat
    title: str | None = None
    columns: tuple[str, ...] | None = None
    max_rows: int = 100
    max_cells: int = 2000
    indent: int = 2

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to the manifest fragment for this format only.

        Only the fields meaningful for ``format`` are emitted, so the result
        satisfies the spec's per-format ``additionalProperties: false``.
        """
        manifest: dict[str, Any] = {"format": self.format}
        if self.title is not None:
            manifest["title"] = self.title
        if self.format == "table":
            if self.columns is not None:
                manifest["columns"] = list(self.columns)
            manifest["maxRows"] = self.max_rows
            manifest["maxCells"] = self.max_cells
        elif self.format == "json":
            manifest["indent"] = self.indent
        return manifest

    def render(self, artifact: Artifact) -> str:
        """Render the full served artifact to text (up to ``max_rows`` rows).

        The complete representation, as returned by a resource read. The inline
        tool text uses :meth:`preview` instead.
        """
        if self.format == "table":
            return self._render_table(artifact, limit=self.max_rows)
        if self.format == "json":
            body = _json.dumps(artifact.value, indent=self.indent, default=str)
            return self._with_title(body)
        if self.format == "components":
            return self._with_title(_render_components(artifact))
        # markdown and text both stringify the value as-is.
        return self._with_title(str(artifact.value))

    def preview(self, artifact: Artifact) -> str:
        """Render a context-sized preview for the inline tool response.

        Non-table formats preview as they render. A table is trimmed to fit the
        ``max_cells`` budget (rows x columns) and, when trimmed, notes the total
        row count and where the full result lives.
        """
        if self.format != "table":
            return self.render(artifact)
        rows = artifact.value if isinstance(artifact.value, list) else []
        columns = self._columns(rows)
        served = rows[: self.max_rows]
        budget_rows = (
            max(1, self.max_cells // len(columns)) if columns else self.max_rows
        )
        if len(served) <= budget_rows:
            return self.render(artifact)
        body = _markdown_table(columns, served[:budget_rows])
        body += (
            f"\n\n_Preview: first {budget_rows} of {len(rows)} rows. "
            "The full result is in this tool's structured content and its resource._"
        )
        return self._with_title(body)

    def structured(self, artifact: Artifact) -> dict[str, Any] | None:
        """Return the payload for an MCP ``structuredContent`` field, or None.

        For a table, the served rows (projected to ``columns``, capped at
        ``max_rows``) with their real types, plus the total ``row_count``. For
        ``components``, every component object in full (id/type/scope/statement/
        relations/structure/evidence/provenance) -- ``render``/``preview`` carry only
        the statements, so this is the one place the machine-checkable fields reach
        an agent. ``None`` for other formats.
        """
        if self.format == "components":
            items = artifact.value if isinstance(artifact.value, list) else []
            return {"components": items, "count": len(items)}
        if self.format != "table":
            return None
        rows = artifact.value if isinstance(artifact.value, list) else []
        columns = self._columns(rows)
        served = rows[: self.max_rows]
        projected = (
            [{column: row.get(column) for column in columns} for row in served]
            if self.columns
            else served
        )
        return {"rows": projected, "row_count": len(rows)}

    def _columns(self, rows: list[dict[str, Any]]) -> list[str]:
        return list(self.columns) if self.columns else _natural_columns(rows)

    def _render_table(self, artifact: Artifact, *, limit: int) -> str:
        rows = artifact.value if isinstance(artifact.value, list) else []
        columns = self._columns(rows)
        body = _markdown_table(columns, rows[:limit])
        if len(rows) > limit:
            body += f"\n\n_Showing {limit} of {len(rows)} rows._"
        return self._with_title(body)

    def _with_title(self, body: str) -> str:
        if self.title:
            return f"# {self.title}\n\n{body}"
        return body


def table(
    *,
    title: str | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    max_rows: int = 100,
    max_cells: int = 2000,
) -> Serve:
    """A table serve contract."""
    return Serve(
        format="table",
        title=title,
        columns=tuple(columns) if columns is not None else None,
        max_rows=max_rows,
        max_cells=max_cells,
    )


def markdown(*, title: str | None = None) -> Serve:
    """A Markdown serve contract."""
    return Serve(format="markdown", title=title)


def json(*, title: str | None = None, indent: int = 2) -> Serve:
    """A JSON serve contract."""
    return Serve(format="json", title=title, indent=indent)


def text(*, title: str | None = None) -> Serve:
    """A plain-text serve contract."""
    return Serve(format="text", title=title)


def components(*, title: str | None = None) -> Serve:
    """An ORC components serve contract."""
    return Serve(format="components", title=title)


def _render_components(artifact: Artifact) -> str:
    """Render components the way ORC's own spec composes them into a prompt.

    One bullet per statement, nothing else. The statement is the canonical
    representation; the rest of a component's fields reach the agent only through
    :meth:`Serve.structured`.
    """
    items = artifact.value if isinstance(artifact.value, list) else []
    if not items:
        return "_(no components)_"
    return "\n".join(f"- {item.get('statement', '')}" for item in items)


def _natural_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def _markdown_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not columns:
        return "_(no columns)_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = [_cell(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
