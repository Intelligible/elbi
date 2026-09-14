"""Poke at the served model: what would the prediction be if a feature changed?

Reuses the same cached ``churn_model`` and the same ``_probability`` scorer
``predict_churn`` uses -- a what-if engine is not a second model, it is the
already-served one, re-scored against hypothetical records. Each scenario is a
partial record (only the features that change) merged onto a base record, so
an agent can ask "what if recency were half this" without knowing or
restating every other feature.
"""

from __future__ import annotations

from typing import Any

from elbi_core import Artifact, Context, derivation, param, serve

from .churn_model import churn_model
from .predict_churn import _probability


@derivation(
    inputs={"model": churn_model},
    params={
        "base": param.object(description="The baseline customer record"),
        "scenarios": param.array(
            items="object",
            description=(
                "Feature overrides to try; each is a partial record merged onto base"
            ),
        ),
    },
    serve=serve.table(title="Churn what-if"),
)
def predict_churn_whatif(ctx: Context) -> Artifact:
    """Score ``base`` plus each scenario's overrides under the same model."""
    model = ctx.input("model").value
    base: dict[str, Any] = ctx.param("base")
    base_probability = _probability(model, base)

    rows = []
    for index, overrides in enumerate(ctx.param("scenarios")):
        record = {**base, **overrides}
        probability = _probability(model, record)
        rows.append(
            {
                "scenario": index,
                "changed": ", ".join(f"{k}={v}" for k, v in overrides.items()),
                "churn_probability": probability,
                "delta": round(probability - base_probability, 4),
            }
        )
    return Artifact.table(rows)
