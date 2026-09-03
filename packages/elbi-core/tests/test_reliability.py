"""Validation of the measurement-reliability (Cronbach's alpha) gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_reliability

N = 300


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(round(v, 4)) for k, v in row.items()} for row in maker]


def test_coherent_scale_is_sound() -> None:
    # five items all loading on one latent trait -> high internal consistency
    rng = random.Random(0)
    out = []
    for _ in range(N):
        trait = rng.gauss(0, 1)
        out.append({f"q{j}": trait + rng.gauss(0, 0.4) for j in range(5)})
    report = verify_reliability(_rows(out), [f"q{j}" for j in range(5)])
    assert report.verdict == "sound" and report.effect >= 0.7


def test_incoherent_items_are_unsound() -> None:
    # five independent noise columns -> alpha near 0
    rng = random.Random(1)
    rows = _rows({f"q{j}": rng.gauss(0, 1) for j in range(5)} for _ in range(N))
    report = verify_reliability(rows, [f"q{j}" for j in range(5)])
    assert report.verdict == "unsound"
    assert "alpha" in (report.pivotal or "")


def test_too_few_items_is_inconclusive() -> None:
    rng = random.Random(2)
    rows = _rows({"q0": rng.gauss(0, 1), "q1": rng.gauss(0, 1)} for _ in range(N))
    assert verify_reliability(rows, ["q0", "q1"]).verdict == "inconclusive"
