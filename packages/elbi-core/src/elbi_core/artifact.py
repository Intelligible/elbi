"""The output of a derivation.

An :class:`Artifact` wraps a computed value together with a hint about its shape
(``table``, ``markdown``, ``json``, ``text``, or ``opaque``). The serve contract
decides how a renderable value is finally presented to an agent; the artifact
only carries it.

An ``opaque`` artifact holds an arbitrary Python object (a trained model, a fitted
index) that a downstream derivation consumes but that is never rendered to an
agent. It is for internal derivations: a training derivation returns
``Artifact.opaque(model)`` and a prediction derivation reads it back with
``ctx.input("model").value``.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Literal

ArtifactKind = Literal["table", "markdown", "json", "text", "opaque"]

#: How many rows of a table an interactive display draws before it stops and says so.
DISPLAY_ROWS = 1000


@dataclass(frozen=True)
class Artifact:
    """A computed value plus a shape hint.

    Construct via the classmethods rather than the initializer; they make the
    intended shape explicit at the call site.
    """

    kind: ArtifactKind
    value: Any

    @classmethod
    def table(cls, rows: list[dict[str, Any]]) -> Artifact:
        """A tabular artifact from a list of row mappings."""
        return cls(kind="table", value=list(rows))

    @classmethod
    def markdown(cls, text: str) -> Artifact:
        """A Markdown-document artifact."""
        return cls(kind="markdown", value=str(text))

    @classmethod
    def json(cls, value: Any) -> Artifact:
        """A JSON-serializable artifact."""
        return cls(kind="json", value=value)

    @classmethod
    def text(cls, text: str) -> Artifact:
        """A plain-text artifact."""
        return cls(kind="text", value=str(text))

    def _repr_html_(self) -> str | None:
        """Draw a table as one, so a notebook echoing an artifact shows its rows.

        ``None`` for every other kind, which the display protocol reads as "no HTML
        rendering, use another": markdown has its own hook below, and text, JSON and
        opaque artifacts are better served by their repr than by invented markup.
        """
        if self.kind != "table":
            return None
        rows: list[dict[str, Any]] = self.value
        if not rows:
            return "<em>no rows</em>"
        columns = list(dict.fromkeys(key for row in rows for key in row))
        head = "".join(f"<th>{_cell(column)}</th>" for column in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{_cell(row.get(c))}</td>" for c in columns) + "</tr>"
            for row in rows[:DISPLAY_ROWS]
        )
        table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        if len(rows) <= DISPLAY_ROWS:
            return table
        note = f"showing {DISPLAY_ROWS:,} of {len(rows):,} rows"
        return f"{table}<em>{note}</em>"

    def _repr_markdown_(self) -> str | None:
        """The document itself for markdown; ``None`` for every other kind."""
        return self.value if self.kind == "markdown" else None

    @classmethod
    def opaque(cls, value: Any) -> Artifact:
        """An opaque object artifact (e.g. a trained model) for internal use.

        Carried and cached as-is and handed to downstream derivations. It has no
        serve rendering, so a derivation returning one should be internal (no
        serve contract).
        """
        return cls(kind="opaque", value=value)


def _cell(value: object) -> str:
    """One table cell's text, escaped, with ``None`` shown as an empty cell."""
    return escape("" if value is None else str(value))


def coerce_artifact(value: Any) -> Artifact:
    """Wrap a compute function's return value into an :class:`Artifact`.

    A function may return an :class:`Artifact` directly, or a raw value that is
    wrapped with a best-effort shape hint (a list of dicts becomes a table; a
    string becomes text; anything else becomes JSON).
    """
    if isinstance(value, Artifact):
        return value
    if isinstance(value, str):
        return Artifact.text(value)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return Artifact.table(value)
    return Artifact.json(value)
