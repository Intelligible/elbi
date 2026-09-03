"""Train a churn model from the customers dataset, with no ML dependency.

The "model" is a pure-Python linear discriminant: for each feature, the
standardized separation between churned and active customers. It exists to show
the pattern rather than the algorithm, a training derivation that returns an
opaque model object a prediction derivation consumes, without pulling in a
compute library. Swap in scikit-learn, PyTorch, or anything else and the shape is
unchanged: fit, return ``Artifact.opaque(model)``.
"""

from __future__ import annotations

from statistics import mean, pstdev

from elbi_core import Artifact, Context, Dataset, derivation

FEATURES = ("recency", "frequency", "monetary")


@derivation(inputs={"customers": Dataset("customers")})  # internal: feeds predict
def churn_model(ctx: Context) -> Artifact:
    """Fit per-feature weights separating churned from active customers."""
    rows = ctx.input("customers").rows
    churned = [r for r in rows if r["churned"] == "1"]
    active = [r for r in rows if r["churned"] == "0"]

    stats: dict[str, dict[str, float]] = {}
    weight: dict[str, float] = {}
    for feature in FEATURES:
        values = [float(r[feature]) for r in rows]
        spread = pstdev(values) or 1.0
        separation = mean(float(r[feature]) for r in churned) - mean(
            float(r[feature]) for r in active
        )
        stats[feature] = {"mean": mean(values), "std": spread}
        weight[feature] = separation / spread

    return Artifact.opaque(
        {"features": list(FEATURES), "stats": stats, "weight": weight}
    )
