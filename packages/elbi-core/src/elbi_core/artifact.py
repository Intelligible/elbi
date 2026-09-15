"""The output of a derivation.

An :class:`Artifact` wraps a computed value together with a hint about its shape
(``table``, ``markdown``, ``json``, ``text``, ``components``, or ``opaque``). The
serve contract decides how a renderable value is finally presented to an agent; the
artifact only carries it.

A ``components`` artifact holds a list of OpenReasoningComponents (ORC)-shaped
dicts: self-contained natural-language statements about the data, each optionally
carrying ``structure``/``evidence``/``relations``/``provenance``. Unlike ``table``,
the compute function authors these by hand today; nothing here generates them from
data statistics.

An ``opaque`` artifact holds an arbitrary Python object (a trained model, a fitted
index) that a downstream derivation consumes but that is never rendered to an
agent. It is for internal derivations: a training derivation returns
``Artifact.opaque(model)`` and a prediction derivation reads it back with
``ctx.input("model").value``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ArtifactKind = Literal["table", "markdown", "json", "text", "components", "opaque"]


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

    @classmethod
    def components(cls, items: list[dict[str, Any]]) -> Artifact:
        """A components artifact: a list of ORC-shaped natural-language facts."""
        return cls(kind="components", value=list(items))

    @classmethod
    def opaque(cls, value: Any) -> Artifact:
        """An opaque object artifact (e.g. a trained model) for internal use.

        Carried and cached as-is and handed to downstream derivations. It has no
        serve rendering, so a derivation returning one should be internal (no
        serve contract).
        """
        return cls(kind="opaque", value=value)


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
