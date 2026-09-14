"""Serve default-risk scores from the vetted model -- no parameters, one call.

An agent (or a human) gets this by calling ``run_default_risk``, not by writing
its own analysis over ``loans.csv``. The model behind it was reviewed once, by
a person, and every call re-scores current data through that same, unchanged
model -- see tests/test_example.py for what happens instead when a proposed
model tries to use a feature that leaks the outcome.
"""

from __future__ import annotations

from math import exp
from typing import Any

from elbi_core import Artifact, Context, Dataset, derivation, serve

from .default_model import default_model


def _probability(model: dict[str, Any], record: dict[str, Any]) -> float:
    """Logistic score of one record under the fitted weights."""
    z = 0.0
    for feature in model["features"]:
        stat = model["stats"][feature]
        standardized = (float(record[feature]) - stat["mean"]) / stat["std"]
        z += model["weight"][feature] * standardized
    return round(1.0 / (1.0 + exp(-z)), 4)


@derivation(
    inputs={"model": default_model, "loans": Dataset("loans")},
    serve=serve.table(title="Default risk", max_rows=200),
)
def default_risk(ctx: Context) -> Artifact:
    """Score every application in the loans dataset under the vetted model."""
    model = ctx.input("model").value
    rows = ctx.input("loans").rows
    scored = [
        {
            "application_id": row["application_id"],
            "default_probability": _probability(model, row),
        }
        for row in rows
    ]
    scored.sort(key=lambda r: r["default_probability"], reverse=True)
    return Artifact.table(scored)
