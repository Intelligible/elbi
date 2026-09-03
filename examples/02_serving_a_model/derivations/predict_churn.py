"""Serve churn predictions from the trained model over structured input.

``predict_churn`` takes one customer record (an ``object`` param); the model is an
input, so it is trained once and cached, not refit per call. ``predict_churn_batch``
scores a list of records (an ``array`` of objects) in one call.
"""

from __future__ import annotations

from math import exp
from typing import Any

from elbi_core import Artifact, Context, derivation, param, serve

from .churn_model import churn_model


def _probability(model: dict[str, Any], record: dict[str, Any]) -> float:
    """Logistic score of one record under the fitted weights."""
    z = 0.0
    for feature in model["features"]:
        stat = model["stats"][feature]
        standardized = (float(record[feature]) - stat["mean"]) / stat["std"]
        z += model["weight"][feature] * standardized
    return round(1.0 / (1.0 + exp(-z)), 4)


@derivation(
    inputs={"model": churn_model},
    params={
        "customer": param.object(
            description="One customer's recency, frequency, and monetary values"
        )
    },
    serve=serve.json(title="Churn prediction"),
)
def predict_churn(ctx: Context) -> Artifact:
    """Predict one customer's churn probability from a feature record."""
    model = ctx.input("model").value
    probability = _probability(model, ctx.param("customer"))
    return Artifact.json({"churn_probability": probability})


@derivation(
    inputs={"model": churn_model},
    params={
        "customers": param.array(
            items="object", description="A batch of customer feature records"
        )
    },
    serve=serve.table(title="Churn predictions"),
)
def predict_churn_batch(ctx: Context) -> Artifact:
    """Predict churn probability for each customer record in a batch."""
    model = ctx.input("model").value
    rows = []
    for index, record in enumerate(ctx.param("customers")):
        rows.append(
            {
                "id": record.get("id", index),
                "churn_probability": _probability(model, record),
            }
        )
    return Artifact.table(rows)
