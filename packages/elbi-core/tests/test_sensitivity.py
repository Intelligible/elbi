"""Validation of the E-value sensitivity gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_sensitivity

N = 500


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(round(v, 4)) for k, v in row.items()} for row in maker]


def test_strong_effect_is_robust_to_confounding() -> None:
    rng = random.Random(0)
    rows = _rows(
        {"x": (x := rng.gauss(0, 1)), "y": 1.5 * x + rng.gauss(0, 1)} for _ in range(N)
    )
    report = verify_sensitivity(rows, "x", "y")
    assert report.verdict == "sound"
    assert report.effect > 1.25  # E-value


def test_weak_effect_is_fragile() -> None:
    # a barely-significant tiny effect: a weak confounder could explain it away
    rng = random.Random(1)
    rows = _rows(
        {"x": (x := rng.gauss(0, 1)), "y": 0.1 * x + rng.gauss(0, 1)} for _ in range(N)
    )
    assert verify_sensitivity(rows, "x", "y").verdict == "inconclusive"


def test_no_association_is_inconclusive() -> None:
    rng = random.Random(2)
    rows = _rows({"x": rng.gauss(0, 1), "y": rng.gauss(0, 1)} for _ in range(N))
    assert verify_sensitivity(rows, "x", "y").verdict == "inconclusive"


def test_e_value_grows_with_effect() -> None:
    rng = random.Random(3)
    weak = _rows(
        {"x": (x := rng.gauss(0, 1)), "y": 0.4 * x + rng.gauss(0, 1)} for _ in range(N)
    )
    strong = _rows(
        {"x": (x := rng.gauss(0, 1)), "y": 2.0 * x + rng.gauss(0, 1)} for _ in range(N)
    )
    assert (
        verify_sensitivity(strong, "x", "y").effect
        > verify_sensitivity(weak, "x", "y").effect
    )
