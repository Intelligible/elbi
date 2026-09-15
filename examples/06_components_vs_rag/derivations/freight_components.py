"""OpenReasoningComponents (ORC)-shaped facts about a freight accessorial rate schedule.

Unlike churn_components in 05_components (facts computed from raw rows), these are
human-authored: a rate schedule is asserted, not derived by statistics. Each row of
`fixtures/rate_schedule.csv` becomes one component, `provenance.source="human"`.

The rate schedule is append-only and versioned by row, not by file: an update (see
`../publish_update.py`) never edits a row in place. It marks the current row
superseded (`valid_until` gets a date) and appends a new one (`valid_until` empty).
Both rows come through here -- this derivation returns every row, superseded or not,
so the full history stays a queryable, citable audit trail. `../agent/demo_agent.py`
is what filters to "current" before answering; that split (a durable full history from
the derivation, a live view at the agent) is deliberate, and mirrors how any
versioned data source should be treated, not something special-cased for this demo.

Two rows depend on `diesel_price_index` (a stub -- see its own row): the fuel
surcharge tiers apply to a *band* of diesel prices, so their `depends_on` relation
names the real thing that would move them if it changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from elbi_core import Artifact, Context, Dataset, derivation, serve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ids import version_id as _version_id

DATASET = "freight_accessorials"


@derivation(
    inputs={"rate_schedule": Dataset("rate_schedule")},
    serve=serve.components(title="Freight accessorial components"),
)
def freight_components(ctx: Context) -> Artifact:
    """Every line item (and superseded prior version) of the accessorial tariff."""
    rows = ctx.input("rate_schedule").rows
    components = [_component(row) for row in rows]
    return Artifact.components(components)


def _component(row: dict[str, str]) -> dict:
    key = row["key"]
    version_id = _version_id(key, row["effective_date"])
    statement = f"{row['item']} costs {row['rate_display']}. {row['detail']}"

    relations = []
    for dep in row["depends_on"].split(";"):
        dep = dep.strip()
        if dep:
            relations.append(
                {"type": "depends_on", "target_id": _version_id(dep, "2024-01-01")}
            )
    supersedes = row["supersedes"].strip()
    if supersedes:
        relations.append({"type": "supersedes", "target_id": supersedes})

    return {
        "id": version_id,
        "type": row["type"],
        "scope": {"dataset": DATASET},
        "statement": statement,
        "relations": relations,
        "structure": {
            "key": key,
            "rate": float(row["rate"]),
            "unit": row["unit"],
            "effective_date": row["effective_date"],
            "valid_until": row["valid_until"] or None,
            "current": not row["valid_until"],
        },
        "evidence": {"citation": row["citation"]},
        "provenance": {"source": "human", "author": "rates-team@acme-freight.example"},
    }
