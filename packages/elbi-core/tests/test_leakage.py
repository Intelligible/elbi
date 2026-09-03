"""Validation of the data-leakage screening gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_leakage

N = 400


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def test_leaky_feature_is_flagged_unsound() -> None:
    # `leak` is a near-perfect proxy for the binary target -> leakage
    rng = random.Random(0)
    rows = []
    for _ in range(N):
        y = rng.randint(0, 1)
        rows.append(
            {
                "x1": round(rng.gauss(0, 1), 3),
                "leak": y + round(rng.gauss(0, 0.01), 4),  # ~equals the label
                "y": y,
            }
        )
    report = verify_leakage(_rows(rows), "y")
    assert report.verdict == "unsound"
    assert "leak" in (report.pivotal or "")


def test_legitimate_features_are_sound() -> None:
    rng = random.Random(1)
    rows = _rows(
        {
            "x1": round(rng.gauss(0, 1), 3),
            "x2": round(rng.gauss(0, 1), 3),
            "y": rng.randint(0, 1),
        }
        for _ in range(N)
    )
    assert verify_leakage(rows, "y").verdict == "sound"


def test_continuous_target_leak_is_flagged() -> None:
    rng = random.Random(2)
    rows = _rows(
        {
            "x": round(v := rng.gauss(0, 1), 3),
            "leak": round(v + rng.gauss(0, 0.01), 4),
            "y": round(3 * v + rng.gauss(0, 0.5), 3),
        }
        for _ in range(N)
    )
    # leak correlates ~perfectly with y through the shared driver
    assert verify_leakage(rows, "y", features=["leak"]).verdict == "unsound"


def test_no_features_is_inconclusive() -> None:
    rng = random.Random(3)
    rows = _rows({"y": rng.randint(0, 1)} for _ in range(N))
    assert verify_leakage(rows, "y").verdict == "inconclusive"
