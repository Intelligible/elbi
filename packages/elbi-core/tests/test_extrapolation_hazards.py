"""Validation of the extrapolation and proportional-hazards gates."""

from __future__ import annotations

import random

from elbi_core.verification import (
    verify_extrapolation,
    verify_proportional_hazards,
)

N = 400


def _rows(maker) -> list[dict[str, str]]:
    return [
        {
            k: (str(round(v, 4)) if isinstance(v, int | float) else str(v))
            for k, v in row.items()
        }
        for row in maker
    ]


def test_in_support_query_is_sound() -> None:
    rng = random.Random(0)
    rows = _rows({"a": rng.gauss(0, 1), "b": rng.gauss(0, 1)} for _ in range(N))
    assert (
        verify_extrapolation(rows, ["a", "b"], {"a": 0.2, "b": -0.3}).verdict == "sound"
    )


def test_out_of_range_query_is_unsound() -> None:
    rng = random.Random(1)
    rows = _rows({"a": rng.gauss(0, 1), "b": rng.gauss(0, 1)} for _ in range(N))
    report = verify_extrapolation(rows, ["a", "b"], {"a": 12.0, "b": 0.0})
    assert report.verdict == "unsound"
    assert "a" in (report.pivotal or "")


def test_unobserved_combination_is_unsound() -> None:
    # a and b are tightly positively correlated; (a high, b low) is in-range per feature
    # but never observed together -> Mahalanobis flags it
    rng = random.Random(2)
    rows = _rows(
        {"a": (z := rng.gauss(0, 1)) + rng.gauss(0, 0.05), "b": z + rng.gauss(0, 0.05)}
        for _ in range(N)
    )
    report = verify_extrapolation(rows, ["a", "b"], {"a": 2.5, "b": -2.5})
    assert report.verdict == "unsound"


def test_proportional_hazards_sound() -> None:
    # group t has uniformly higher hazard -> curves separate but never cross
    rng = random.Random(3)
    rows = []
    for _ in range(N):
        g = rng.choice(["c", "t"])
        rate = 0.05 if g == "c" else 0.12
        t = rng.expovariate(rate)
        rows.append({"time": round(min(t, 60), 3), "event": 1 if t < 60 else 0, "g": g})
    assert (
        verify_proportional_hazards(_rows(rows), "time", "event", "g").verdict
        == "sound"
    )


def test_crossing_hazards_unsound() -> None:
    # control dies fast early then plateaus; treatment steady -> curves cross
    rng = random.Random(4)
    rows = []
    for _ in range(N):
        g = rng.choice(["c", "t"])
        if g == "c":
            t = (
                rng.expovariate(0.4)
                if rng.random() < 0.5
                else 50 + rng.expovariate(0.02)
            )
        else:
            t = rng.uniform(5, 40)
        rows.append({"time": round(min(t, 60), 3), "event": 1 if t < 60 else 0, "g": g})
    report = verify_proportional_hazards(_rows(rows), "time", "event", "g")
    assert report.verdict == "unsound"
