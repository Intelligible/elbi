"""Validation of the missing-data (MCAR) gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_missingness

N = 600


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def test_mcar_is_sound() -> None:
    # `y` is missing independently of everything observed
    rng = random.Random(0)
    rows = []
    for _ in range(N):
        x = rng.gauss(0, 1)
        y = "" if rng.random() < 0.3 else round(rng.gauss(0, 1), 4)
        rows.append({"x": round(x, 4), "y": y})
    assert verify_missingness(_rows(rows), "y").verdict == "sound"


def test_mar_is_unsound() -> None:
    # `y` is missing more often when `x` is high -> missingness depends on x
    rng = random.Random(1)
    rows = []
    for _ in range(N):
        x = rng.gauss(0, 1)
        prob = 0.6 if x > 0 else 0.05
        y = "" if rng.random() < prob else round(rng.gauss(0, 1), 4)
        rows.append({"x": round(x, 4), "y": y})
    report = verify_missingness(_rows(rows), "y")
    assert report.verdict == "unsound"
    assert "x" in (report.pivotal or "")


def test_too_few_missing_is_inconclusive() -> None:
    rng = random.Random(2)
    rows = _rows(
        {"x": round(rng.gauss(0, 1), 4), "y": round(rng.gauss(0, 1), 4)}
        for _ in range(N)
    )
    assert verify_missingness(rows, "y").verdict == "inconclusive"
