"""Validation of the time-series stationarity gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_stationarity

N = 300


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def test_white_noise_is_stationary() -> None:
    rng = random.Random(0)
    rows = _rows({"t": i, "v": round(rng.gauss(0, 1), 4)} for i in range(N))
    assert verify_stationarity(rows, "v", "t").verdict == "sound"


def test_mean_reverting_ar_is_stationary() -> None:
    rng = random.Random(1)
    out, x = [], 0.0
    for i in range(N):
        x = 0.3 * x + rng.gauss(0, 1)
        out.append({"t": i, "v": round(x, 4)})
    assert verify_stationarity(_rows(out), "v", "t").verdict == "sound"


def test_random_walk_is_unsound() -> None:
    rng = random.Random(2)
    out, x = [], 0.0
    for i in range(N):
        x += rng.gauss(0, 1)
        out.append({"t": i, "v": round(x, 4)})
    report = verify_stationarity(_rows(out), "v", "t")
    assert report.verdict == "unsound"
    assert "unit root" in (report.pivotal or "")


def test_deterministic_trend_is_unsound() -> None:
    rng = random.Random(3)
    rows = _rows({"t": i, "v": round(0.2 * i + rng.gauss(0, 1), 4)} for i in range(N))
    report = verify_stationarity(rows, "v", "t")
    assert report.verdict == "unsound"
    assert "trend" in (report.pivotal or "")


def test_too_short_is_inconclusive() -> None:
    rng = random.Random(4)
    rows = _rows({"t": i, "v": round(rng.gauss(0, 1), 4)} for i in range(20))
    assert verify_stationarity(rows, "v", "t").verdict == "inconclusive"
