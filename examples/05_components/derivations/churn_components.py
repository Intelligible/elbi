"""OpenReasoningComponents (ORC)-shaped facts about the customers dataset.

Unlike churn_risk in 01_getting_started (a table of scores), this derivation's
artifact is a list of self-contained natural-language statements -- grounded in
the same data, each carrying real evidence computed from it. `elbi serve`
exposes it as `run_churn_components`, whose structured content is the full
component objects (with provenance stamped by Elbi's own versioning), and as a
hit in `search_components`.

Everything here is pure standard library: no scipy/pandas, so this example
needs nothing beyond elbi itself, matching 01_getting_started.
"""

from __future__ import annotations

from statistics import mean, median, pstdev

from elbi_core import Artifact, Context, Dataset, derivation, serve

NAMESPACE = "elbi-examples"
DATASET = "customer_churn"


def _rate(rows: list[dict], predicate) -> float:
    matched = [r for r in rows if predicate(r)]
    return sum(1 for r in matched if r["churn"] == "true") / len(matched)


def _odds_ratio(rows: list[dict], predicate) -> tuple[float, int, int]:
    """Odds ratio of churn for rows matching `predicate` vs. rows that don't."""
    a = b = c = d = 0
    for row in rows:
        churned = row["churn"] == "true"
        if predicate(row):
            a += churned
            b += not churned
        else:
            c += churned
            d += not churned
    odds_ratio = (a / b) / (c / d) if b and c and d else float("nan")
    return odds_ratio, a + b, c + d


@derivation(
    inputs={"customers": Dataset("customers")},
    serve=serve.components(title="Churn components"),
)
def churn_components(ctx: Context) -> Artifact:
    """Grounded natural-language facts about customer churn."""
    rows = ctx.input("customers").rows
    n = len(rows)
    discounts = [float(r["discount"]) for r in rows]

    threshold = 0.20
    odds_ratio, n_high, n_low = _odds_ratio(
        rows, lambda r: float(r["discount"]) > threshold
    )
    churn_high = _rate(rows, lambda r: float(r["discount"]) > threshold)
    churn_low = _rate(rows, lambda r: float(r["discount"]) <= threshold)

    def _is_loyal_two_year(row: dict) -> bool:
        return int(row["tenure_months"]) > 36 and row["contract_type"] == "two_year"

    segment_rows = [r for r in rows if _is_loyal_two_year(r)]
    segment_churn = (
        sum(1 for r in segment_rows if r["churn"] == "true") / len(segment_rows)
        if segment_rows
        else float("nan")
    )
    overall_churn = sum(1 for r in rows if r["churn"] == "true") / n

    components = [
        {
            "id": f"{NAMESPACE}/customers_discount",
            "type": "column",
            "scope": {"dataset": DATASET},
            "statement": (
                "discount is a float column in the customers table representing "
                f"the discount rate applied to a customer's contract, ranging "
                f"from {min(discounts):.0%} to {max(discounts):.0%}."
            ),
            "structure": {
                "table": "customers",
                "column_name": "discount",
                "data_type": "float",
            },
        },
        {
            "id": f"{NAMESPACE}/discount_distribution",
            "type": "distribution",
            "scope": {"dataset": DATASET},
            "statement": (
                f"Customer discounts range from {min(discounts):.0%} to "
                f"{max(discounts):.0%}, with a median of {median(discounts):.0%}."
            ),
            "relations": [
                {
                    "type": "depends_on",
                    "target_id": f"{NAMESPACE}/customers_discount",
                }
            ],
            "structure": {
                "variable": "discount",
                "stats": {
                    "min": round(min(discounts), 3),
                    "max": round(max(discounts), 3),
                    "median": round(median(discounts), 3),
                    "mean": round(mean(discounts), 3),
                    "std": round(pstdev(discounts), 3),
                },
            },
            "evidence": {"sample_size": n},
        },
        {
            "id": f"{NAMESPACE}/churn_discount_threshold",
            "type": "threshold_rule",
            "scope": {"dataset": DATASET},
            "statement": (
                f"Customers receiving discounts above {threshold:.0%} churn "
                f"substantially more often ({churn_high:.0%} vs. {churn_low:.0%})."
            ),
            "relations": [
                {"type": "depends_on", "target_id": f"{NAMESPACE}/customers_discount"},
                {
                    "type": "depends_on",
                    "target_id": f"{NAMESPACE}/discount_distribution",
                },
            ],
            "structure": {
                "feature": "discount",
                "threshold": threshold,
                "direction": "above",
            },
            "evidence": {
                "odds_ratio": round(odds_ratio, 2),
                "sample_size": n_high + n_low,
                "churn_rate_above": round(churn_high, 3),
                "churn_rate_at_or_below": round(churn_low, 3),
            },
        },
        {
            "id": f"{NAMESPACE}/loyal_two_year_segment",
            "type": "segment",
            "scope": {"dataset": DATASET},
            "statement": (
                "A segment of long-tenure (>36 month) two-year-contract "
                f"customers (n={len(segment_rows)}) shows low churn "
                f"({segment_churn:.0%} vs. {overall_churn:.0%} overall)."
            ),
            "structure": {
                "conditions": [
                    {"feature": "tenure_months", "operator": ">", "value": 36},
                    {
                        "feature": "contract_type",
                        "operator": "=",
                        "value": "two_year",
                    },
                ]
            },
            "evidence": {
                "segment_size": len(segment_rows),
                "sample_size": n,
            },
        },
    ]
    return Artifact.components(components)
