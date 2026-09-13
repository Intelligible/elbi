"""The vetted model's finding, expressed as durable, searchable memory.

``default_risk`` (a table) answers "what is this application's risk" once,
per call. This derivation answers a different question: "what do we know
about default risk in general" -- and unlike a table an agent has to re-read
and re-interpret every time, each answer here is a natural-language statement
with its own evidence, servable as one more MCP tool (``run_default_risk_components``)
and, unlike a plain answer, discoverable later by *meaning* through
``search_components`` -- see docs/mcp.md and this session's `elbi_core.components`.

Elbi stamps ``provenance.derivation``/``provenance.derivation_version`` onto
each of these automatically at serve time (see
``elbi_cli.serving.Serving._present``), so "is this still true" reduces to
"is this derivation's version still current" -- the same staleness check
every other derivation already gets, with no separate machinery for
components.
"""

from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Any

from elbi_core import Artifact, Context, Dataset, derivation, serve

from .default_model import default_model

NAMESPACE = "elbi-examples"
DATASET = "loan_defaults"
DEBT_RATIO_THRESHOLD = 0.4


def _odds_ratio(rows: list[dict[str, Any]], predicate: Any) -> tuple[float, int, int]:
    """Odds ratio of default for rows matching ``predicate`` vs. rows that don't."""
    a = b = c = d = 0
    for row in rows:
        defaulted = row["defaulted"] == "1"
        if predicate(row):
            a += defaulted
            b += not defaulted
        else:
            c += defaulted
            d += not defaulted
    odds_ratio = (a / b) / (c / d) if b and c and d else float("nan")
    return odds_ratio, a + b, c + d


@derivation(
    inputs={"model": default_model, "loans": Dataset("loans")},
    serve=serve.components(title="Default risk components"),
)
def default_risk_components(ctx: Context) -> Artifact:
    """Grounded natural-language facts about default risk, from the vetted model."""
    model = ctx.input("model").value
    rows = ctx.input("loans").rows
    n = len(rows)
    debt_ratios = [float(r["debt_ratio"]) for r in rows]

    def above_threshold(row: dict[str, Any]) -> bool:
        return float(row["debt_ratio"]) > DEBT_RATIO_THRESHOLD

    odds_ratio, n_high, n_low = _odds_ratio(rows, above_threshold)
    rate_high = sum(
        1 for r in rows if above_threshold(r) and r["defaulted"] == "1"
    ) / max(1, sum(1 for r in rows if above_threshold(r)))
    rate_low = sum(
        1 for r in rows if not above_threshold(r) and r["defaulted"] == "1"
    ) / max(1, sum(1 for r in rows if not above_threshold(r)))

    components = [
        {
            "id": f"{NAMESPACE}/debt_ratio",
            "type": "column",
            "scope": {"dataset": DATASET},
            "statement": (
                "debt_ratio is a float column in the loans table representing "
                f"a loan applicant's debt-to-income ratio, ranging from "
                f"{min(debt_ratios):.0%} to {max(debt_ratios):.0%}."
            ),
            "structure": {
                "table": "loans",
                "column_name": "debt_ratio",
                "data_type": "float",
            },
        },
        {
            "id": f"{NAMESPACE}/debt_ratio_distribution",
            "type": "distribution",
            "scope": {"dataset": DATASET},
            "statement": (
                f"Applicant debt ratios range from {min(debt_ratios):.0%} to "
                f"{max(debt_ratios):.0%}, with a median of "
                f"{median(debt_ratios):.0%}."
            ),
            "relations": [
                {"type": "depends_on", "target_id": f"{NAMESPACE}/debt_ratio"}
            ],
            "structure": {
                "variable": "debt_ratio",
                "stats": {
                    "min": round(min(debt_ratios), 3),
                    "max": round(max(debt_ratios), 3),
                    "median": round(median(debt_ratios), 3),
                    "mean": round(mean(debt_ratios), 3),
                    "std": round(pstdev(debt_ratios), 3),
                },
            },
            "evidence": {"sample_size": n},
        },
        {
            "id": f"{NAMESPACE}/default_debt_ratio_threshold",
            "type": "threshold_rule",
            "scope": {"dataset": DATASET},
            "statement": (
                f"Applicants with a debt ratio above {DEBT_RATIO_THRESHOLD:.0%} "
                f"default substantially more often ({rate_high:.0%} vs. "
                f"{rate_low:.0%})."
            ),
            "relations": [
                {"type": "depends_on", "target_id": f"{NAMESPACE}/debt_ratio"},
                {
                    "type": "depends_on",
                    "target_id": f"{NAMESPACE}/debt_ratio_distribution",
                },
            ],
            "structure": {
                "feature": "debt_ratio",
                "threshold": DEBT_RATIO_THRESHOLD,
                "direction": "above",
            },
            "evidence": {
                "odds_ratio": round(odds_ratio, 2),
                "sample_size": n_high + n_low,
                "default_rate_above": round(rate_high, 3),
                "default_rate_at_or_below": round(rate_low, 3),
            },
            "provenance": {
                "source": "derived",
                "method": "certified_default_model",
                "model_features": model["features"],
            },
        },
    ]
    return Artifact.components(components)
