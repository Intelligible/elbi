"""Generate a synthetic loan-default dataset for the certified-models example.

Pure standard library, seeded for reproducibility, matching the style of the
other examples' fixture generators. ``days_past_due`` is
deliberately a *post-outcome* column: it is zero unless the loan already
defaulted, so a claim that includes it as a "feature" is really training on
the label. That is the leak the example's test proves the oracle catches.

Usage:
    python generate_loans.py -o loans.csv -n 300
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
        income = round(rng.uniform(30_000, 150_000), 2)
        debt_ratio = round(rng.uniform(0.0, 0.6), 3)
        credit_score = round(rng.uniform(550, 800))
        loan_amount = round(rng.uniform(2_000, 30_000), 2)

        # A real, if deliberately blunt, relationship: higher debt ratio and
        # loan-to-income both raise default risk; a higher credit score lowers
        # it. None of this is a "correct" underwriting model -- the point is
        # that it is a legitimate, available-at-application-time signal, not a
        # leak, and strong enough that a held-out fit clears chance (verified
        # directly against elbi_core.verification.verify_prediction, not just
        # asserted): see tests/test_example.py, which uses this same formula.
        loan_to_income = loan_amount / income
        risk = (
            -3.5
            + 10.0 * debt_ratio
            + 8.0 * loan_to_income
            - 0.02 * (credit_score - 550)
        )
        probability = 1.0 / (1.0 + pow(2.718281828, -risk))
        defaulted = rng.random() < probability

        # The leak: only ever nonzero *after* a default has already happened.
        days_past_due = rng.randint(30, 180) if defaulted else 0

        rows.append(
            {
                "application_id": f"a{i + 1}",
                "income": f"{income:.2f}",
                "debt_ratio": f"{debt_ratio:.3f}",
                "credit_score": str(int(credit_score)),
                "loan_amount": f"{loan_amount:.2f}",
                "days_past_due": str(days_past_due),
                "defaulted": "1" if defaulted else "0",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="loans.csv")
    parser.add_argument("-n", "--rows", type=int, default=300)
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
