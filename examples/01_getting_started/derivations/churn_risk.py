"""Per-customer churn-risk scores from the sales dataset."""

from __future__ import annotations

from elbi_core import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(title="Churn risk", columns=["customer_id", "risk"], max_rows=50),
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores derived from recent sales activity."""
    rows = ctx.input("sales").rows
    scored = [
        {
            "customer_id": row["customer_id"],
            "risk": round(1.0 / (1.0 + float(row["amount"])), 4),
        }
        for row in rows
    ]
    scored.sort(key=lambda r: r["risk"], reverse=True)
    return Artifact.table(scored)
