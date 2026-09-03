"""Validation of the normality goodness-of-fit gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_normality

N = 400


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def test_normal_column_is_sound() -> None:
    rng = random.Random(0)
    rows = _rows({"x": round(rng.gauss(0, 1), 4)} for _ in range(N))
    assert verify_normality(rows, "x").verdict == "sound"


def test_skewed_column_is_unsound() -> None:
    # an exponential column is strongly right-skewed -> not normal
    rng = random.Random(1)
    rows = _rows({"x": round(rng.expovariate(1.0), 4)} for _ in range(N))
    report = verify_normality(rows, "x")
    assert report.verdict == "unsound"


def test_too_few_is_inconclusive() -> None:
    rng = random.Random(2)
    rows = _rows({"x": round(rng.gauss(0, 1), 4)} for _ in range(10))
    assert verify_normality(rows, "x").verdict == "inconclusive"
