"""Gates a governed semantic model against the rows its metrics aggregate.

Shows a derivation consuming an Open Semantic Interchange document as an input: the
definitions arrive from someone else's governed model, and this derivation checks
they are actually computable over the data we hold. The model is loaded at import
time, so an unusable document fails when the project is discovered.
"""

from __future__ import annotations

import json
from pathlib import Path

from elbi_core import (
    Artifact,
    Context,
    Dataset,
    SemanticModel,
    derivation,
    serve,
)

_DOCUMENT = (
    Path(__file__).resolve().parent.parent / "semantics" / "sales_semantics.osi.json"
)
_MODEL = SemanticModel.from_osi(json.loads(_DOCUMENT.read_text(encoding="utf-8")))


@derivation(
    inputs={"model": _MODEL, "sales": Dataset("sales")},
    serve=serve.table(title="Metric gates", columns=["metric", "ok", "missing"]),
)
def metric_gates(ctx: Context) -> Artifact:
    """Report whether each governed metric is computable over the sales rows."""
    model: SemanticModel = ctx.input("model")
    rows = ctx.input("sales").rows
    available = set(rows[0]) if rows else set()

    report = []
    for metric in model.metrics.metrics:
        required = set(metric.dimensions)
        if metric.measure is not None and metric.measure.column is not None:
            required.add(metric.measure.column)
        missing = sorted(required - available)
        report.append(
            {
                "metric": metric.name,
                "ok": not missing,
                "missing": ", ".join(missing),
            }
        )
    return Artifact.table(report)
