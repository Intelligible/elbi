"""Generate a synthetic customer-churn dataset for the components example.

Pure standard library, seeded for reproducibility, so the fixture can be
regenerated identically or regenerated larger (see examples/eval.py in the
openreasoningcomponents repo, which uses this to test whether the token gap
between raw data and components widens with dataset size).

Usage:
    python generate_customers.py -o customers.csv -n 500
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def generate(n: int, *, seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tenure = rng.randint(0, 72)
        contract = rng.choices(
            ["month_to_month", "one_year", "two_year"], weights=[3, 1, 1]
        )[0]
        discount = round(rng.uniform(0.0, 0.35), 3)
        monthly_charges = round(rng.uniform(20.0, 120.0), 2)

        # Churn probability: higher with a high discount (a real, if odd,
        # relationship -- steep discounts often mark a customer already
        # flagged as a retention risk), lower with tenure, higher for
        # month-to-month contracts. Deliberately not linear or independent, so
        # a components-format derivation has more than one true fact to state.
        risk = 0.10
        if discount > 0.20:
            risk += 0.25
        risk -= min(tenure, 48) / 48 * 0.15
        if contract == "month_to_month":
            risk += 0.10
        elif contract == "two_year":
            risk -= 0.05
        risk = min(max(risk, 0.01), 0.95)
        churn = rng.random() < risk

        rows.append(
            {
                "customer_id": f"c{i + 1}",
                "tenure_months": str(tenure),
                "contract_type": contract,
                "discount": f"{discount:.3f}",
                "monthly_charges": f"{monthly_charges:.2f}",
                "churn": "true" if churn else "false",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="customers.csv")
    parser.add_argument("-n", "--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = generate(args.rows, seed=args.seed)
    with Path(args.output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
