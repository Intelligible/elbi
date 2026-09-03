"""A Markdown summary built on top of churn_risk; shows composition."""

from __future__ import annotations

from elbi_core import Artifact, Context, derivation, serve

from .churn_risk import churn_risk


@derivation(
    inputs={"risk": churn_risk},
    serve=serve.markdown(title="Pricing effects"),
)
def pricing_effects(ctx: Context) -> Artifact:
    """Flag the highest-risk customers a price increase would most affect."""
    scored = ctx.input("risk").value
    at_risk = [row for row in scored if row["risk"] > 0.05]
    lines = [f"- **{row['customer_id']}**: risk {row['risk']}" for row in at_risk]
    body = "Customers most sensitive to a price increase:\n\n" + "\n".join(lines)
    return Artifact.markdown(body)
