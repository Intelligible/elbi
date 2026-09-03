"""Validation of the power-law goodness-of-fit gate (Clauset-Shalizi-Newman)."""

from __future__ import annotations

import random

from elbi_core.verification import verify_powerlaw

N = 600
_REPS = 150  # smaller bootstrap keeps the suite fast; verdicts are clear-cut


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(round(v, 4)) for k, v in row.items()} for row in maker]


def test_genuine_power_law_is_sound() -> None:
    # a Pareto draw: x = xmin * (1-u)^(-1/(alpha-1))
    rng = random.Random(0)
    rows = _rows({"v": 1.0 * (1 - rng.random()) ** (-1 / 1.5)} for _ in range(N))
    assert verify_powerlaw(rows, "v", reps=_REPS).verdict == "sound"


def test_lognormal_is_not_sound() -> None:
    # the core guarantee: heavy-tailed but lognormal data must NOT be certified a
    # power law (it cannot be distinguished from a lognormal)
    rng = random.Random(1)
    rows = _rows({"v": 2.718281828 ** rng.gauss(0, 1)} for _ in range(N))
    assert verify_powerlaw(rows, "v", reps=_REPS).verdict != "sound"


def test_exponential_is_not_sound() -> None:
    rng = random.Random(2)
    rows = _rows({"v": rng.expovariate(0.5)} for _ in range(N))
    assert verify_powerlaw(rows, "v", reps=_REPS).verdict != "sound"


def test_too_few_is_inconclusive() -> None:
    rng = random.Random(3)
    rows = _rows({"v": rng.expovariate(1.0)} for _ in range(20))
    assert verify_powerlaw(rows, "v", reps=_REPS).verdict == "inconclusive"


def test_reproducible() -> None:
    rng = random.Random(4)
    rows = _rows({"v": 1.0 * (1 - rng.random()) ** (-1 / 1.5)} for _ in range(N))
    a = verify_powerlaw(rows, "v", reps=_REPS)
    b = verify_powerlaw(rows, "v", reps=_REPS)
    assert a.verdict == b.verdict and a.effect == b.effect
