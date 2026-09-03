"""Validation of the instrumental-variables and regression-discontinuity gates."""

from __future__ import annotations

import random

from elbi_core.verification import verify_iv, verify_rdd

N = 500


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(round(v, 4)) for k, v in row.items()} for row in maker]


def test_strong_instrument_is_sound() -> None:
    rng = random.Random(0)
    rows = _rows(
        {
            "z": (z := rng.gauss(0, 1)),
            "d": 0.9 * z + rng.gauss(0, 0.4),
            "y": rng.gauss(0, 1),
        }
        for _ in range(N)
    )
    assert verify_iv(rows, "z", "d", "y").verdict == "sound"


def test_weak_instrument_is_unsound() -> None:
    # the instrument barely moves the treatment -> weak (first-stage F < 10)
    rng = random.Random(1)
    rows = _rows(
        {
            "z": (z := rng.gauss(0, 1)),
            "d": 0.04 * z + rng.gauss(0, 1),
            "y": rng.gauss(0, 1),
        }
        for _ in range(N)
    )
    report = verify_iv(rows, "z", "d", "y")
    assert report.verdict == "unsound"
    assert "weak instrument" in (report.pivotal or "")


def test_no_manipulation_is_sound() -> None:
    rng = random.Random(2)
    rows = _rows({"score": rng.gauss(0, 1)} for _ in range(N))
    assert verify_rdd(rows, "score", cutoff=0.0).verdict == "sound"


def test_manipulation_bunching_is_unsound() -> None:
    # units just below the cutoff push themselves just above it
    rng = random.Random(3)
    vals = []
    for _ in range(N):
        v = rng.gauss(0, 1)
        if -0.3 < v < 0.0 and rng.random() < 0.8:
            v = abs(v) * 0.1  # nudge to just above the cutoff
        vals.append({"score": v})
    report = verify_rdd(_rows(vals), "score", cutoff=0.0)
    assert report.verdict == "unsound"
