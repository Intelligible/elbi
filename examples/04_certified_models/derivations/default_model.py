"""Train a default-risk model from legitimate, available-at-application-time features.

No ML dependency: the same pure-Python standardized-weight discriminant as
examples/02_serving_a_model/derivations/churn_model.py. What matters here is
*which* features it is trained on: income, debt ratio, credit score, and loan
amount are all known before a loan is issued. ``days_past_due`` (in the same
CSV) is deliberately not among them -- it is only ever nonzero after a default
has already happened, so a model trained on it would be trained on the label.
tests/test_example.py proves the oracle catches that if an agent tries it.
"""

from __future__ import annotations

from statistics import mean, pstdev

from elbi_core import Artifact, Context, Dataset, derivation

FEATURES = ("income", "debt_ratio", "credit_score", "loan_amount")


@derivation(inputs={"loans": Dataset("loans")})  # internal: feeds default_risk
def default_model(ctx: Context) -> Artifact:
    """Fit per-feature weights separating defaulted from repaid loans."""
    rows = ctx.input("loans").rows
    defaulted = [r for r in rows if r["defaulted"] == "1"]
    repaid = [r for r in rows if r["defaulted"] == "0"]

    stats: dict[str, dict[str, float]] = {}
    weight: dict[str, float] = {}
    for feature in FEATURES:
        values = [float(r[feature]) for r in rows]
        spread = pstdev(values) or 1.0
        separation = mean(float(r[feature]) for r in defaulted) - mean(
            float(r[feature]) for r in repaid
        )
        stats[feature] = {"mean": mean(values), "std": spread}
        weight[feature] = separation / spread

    return Artifact.opaque(
        {"features": list(FEATURES), "stats": stats, "weight": weight}
    )
