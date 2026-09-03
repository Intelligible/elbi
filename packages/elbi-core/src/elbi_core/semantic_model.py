"""Governed semantic models bound as derivation inputs.

A :class:`SemanticModel` is a semantic model (metric definitions and the dimensions
they may be sliced by) that a derivation declares as an input, so the derivation's
gates run on top of definitions someone else governs. The type is deliberately
generic: the only place a wire format appears is :meth:`SemanticModel.from_osi`, so
another standard becomes another constructor rather than another input kind.

The document is held as canonical JSON text rather than a mapping. That keeps the
value genuinely immutable and hashable, which matters because the document's content
is what versions every derivation that reads it: a mutable mapping could be edited
in place after the derivation was defined, silently detaching results from the
definitions they came from.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .metrics.osi import from_osi
from .metrics.spec import MetricSet
from .versioning import canonical_json


@dataclass(frozen=True)
class SemanticModel:
    """A governed semantic model a derivation can consume.

    Args:
        name: The model name, matching ``^[a-z][a-z0-9_]*$``. Used as the input's
            ``ref`` in the manifest and as its lineage node.
        document_json: The source document as canonical JSON text.
        metrics: The definitions parsed from the document.
    """

    name: str
    document_json: str
    metrics: MetricSet

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SemanticModel name must be non-empty")

    @property
    def document(self) -> dict[str, Any]:
        """The source document, parsed fresh so a caller cannot mutate this model."""
        parsed: dict[str, Any] = json.loads(self.document_json)
        return parsed

    @classmethod
    def from_osi(
        cls, document: Mapping[str, Any], *, name: str | None = None
    ) -> SemanticModel:
        """Bind an Open Semantic Interchange document as a semantic model.

        The document is validated against the OSI schema and its metrics parsed, so
        an unusable model fails where the derivation is defined rather than when an
        agent calls it.

        Args:
            document: The OSI semantic-model document.
            name: Overrides the name taken from the document's first model. Pass one
                when the document's own name is not a valid input ``ref``.

        Raises:
            MetricError: if the document is not schema-conformant OSI, or carries a
                metric this adapter cannot map.
            ValueError: if no name is given and the document names no model.
        """
        metrics = from_osi(dict(document))
        resolved = name if name is not None else _first_model_name(document)
        return cls(
            name=resolved,
            document_json=canonical_json(document),
            metrics=metrics,
        )


def _first_model_name(document: Mapping[str, Any]) -> str:
    models = document.get("semantic_model") or []
    if not models:
        raise ValueError(
            "OSI document names no semantic model; pass an explicit name= instead"
        )
    return str(models[0]["name"])
