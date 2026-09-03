"""Validation of the fairness (subgroup disparity) gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_fairness

N = 2000


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def test_equal_treatment_is_sound() -> None:
    # both groups: same base rate, same (good) error rates
    rng = random.Random(0)
    rows = []
    for _ in range(N):
        g = rng.choice(["a", "b"])
        yt = rng.randint(0, 1)
        yp = yt if rng.random() < 0.85 else 1 - yt
        rows.append({"g": g, "yt": yt, "yp": yp})
    assert verify_fairness(_rows(rows), "g", "yt", "yp").verdict == "sound"


def test_disparate_error_rates_are_unsound() -> None:
    # group b's true positives are caught far less often (low TPR) -> disparity
    rng = random.Random(1)
    rows = []
    for _ in range(N):
        g = rng.choice(["a", "b"])
        yt = rng.randint(0, 1)
        acc = 0.9 if g == "a" else 0.55
        yp = yt if rng.random() < acc else 1 - yt
        rows.append({"g": g, "yt": yt, "yp": yp})
    report = verify_fairness(_rows(rows), "g", "yt", "yp")
    assert report.verdict == "unsound"


def test_too_few_groups_is_inconclusive() -> None:
    rng = random.Random(2)
    rows = _rows(
        {"g": "a", "yt": rng.randint(0, 1), "yp": rng.randint(0, 1)} for _ in range(N)
    )
    assert verify_fairness(rows, "g", "yt", "yp").verdict == "inconclusive"
