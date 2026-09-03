"""Validation of the count-overdispersion and propensity-overlap gates."""

from __future__ import annotations

import random

from elbi_core.verification import verify_counts, verify_overlap

N = 500


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def test_poisson_counts_are_sound() -> None:
    rng = random.Random(0)
    rows = _rows({"k": _poisson(rng, 3.0)} for _ in range(N))
    assert verify_counts(rows, "k").verdict == "sound"


def test_overdispersed_counts_are_unsound() -> None:
    # a negative-binomial-like mixture: Poisson mean itself varies -> overdispersed
    rng = random.Random(1)
    rows = _rows({"k": _poisson(rng, rng.expovariate(1 / 3.0))} for _ in range(N))
    report = verify_counts(rows, "k")
    assert report.verdict == "unsound"
    assert "overdispersed" in (report.pivotal or "")


def test_non_count_is_inconclusive() -> None:
    rng = random.Random(2)
    rows = _rows({"k": round(rng.gauss(0, 1), 3)} for _ in range(N))
    assert verify_counts(rows, "k").verdict == "inconclusive"


def test_good_overlap_is_sound() -> None:
    # treatment is essentially random in the covariate -> propensities overlap
    rng = random.Random(3)
    rows = _rows(
        {"x": round(rng.gauss(0, 1), 3), "t": rng.randint(0, 1)} for _ in range(N)
    )
    assert verify_overlap(rows, "t", ["x"]).verdict == "sound"


def test_poor_overlap_is_unsound() -> None:
    # treatment is almost perfectly separated by x -> propensities pile at 0 and 1
    rng = random.Random(4)
    rows = _rows(
        {"x": (x := round(rng.gauss(0, 1), 3)), "t": 1 if x > 0.3 else 0}
        for _ in range(N)
    )
    report = verify_overlap(rows, "t", ["x"])
    assert report.verdict == "unsound"


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm for a Poisson draw
    target = pow(2.718281828, -lam)
    k, prod = 0, 1.0
    while True:
        k += 1
        prod *= rng.random()
        if prod <= target:
            return k - 1
