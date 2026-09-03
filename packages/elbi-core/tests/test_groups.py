"""Validation of the k-group comparison and equivalence (TOST) gates."""

from __future__ import annotations

import random

from elbi_core.verification import verify_equivalence, verify_groups

N = 600


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def test_real_k_group_difference_is_sound() -> None:
    rng = random.Random(0)
    rows = _rows(
        {
            "g": (g := rng.choice(["a", "b", "c"])),
            "v": {"a": 0.0, "b": 1.0, "c": 2.0}[g] + rng.gauss(0, 1),
        }
        for _ in range(N)
    )
    assert verify_groups(rows, "g", "v").verdict == "sound"


def test_no_k_group_difference_is_inconclusive() -> None:
    rng = random.Random(1)
    rows = _rows(
        {"g": rng.choice(["a", "b", "c"]), "v": rng.gauss(0, 1)} for _ in range(N)
    )
    assert verify_groups(rows, "g", "v").verdict == "inconclusive"


def test_outlier_driven_mean_shift_is_unsound() -> None:
    # group c is drawn from the same distribution but a few extreme outliers shift its
    # mean: the mean-based ANOVA fires, the rank-based Kruskal-Wallis does not
    rng = random.Random(2)
    rows = []
    for _ in range(N):
        g = rng.choice(["a", "b", "c"])
        v = rng.gauss(0, 1)
        rows.append({"g": g, "v": v})
    for r in rows:
        if r["g"] == "c" and rng.random() < 0.05:
            r["v"] = 40 + rng.gauss(0, 1)
    report = verify_groups(_rows(rows), "g", "v")
    assert report.verdict == "unsound"
    assert "Kruskal" in (report.pivotal or "")


def test_two_groups_defers_to_comparison() -> None:
    rng = random.Random(3)
    rows = _rows({"g": rng.choice(["a", "b"]), "v": rng.gauss(0, 1)} for _ in range(N))
    assert verify_groups(rows, "g", "v").verdict == "inconclusive"


def test_equivalent_groups_are_sound() -> None:
    rng = random.Random(4)
    rows = _rows(
        {"g": rng.choice(["a", "b"]), "v": rng.gauss(0, 1)} for _ in range(1200)
    )
    assert verify_equivalence(rows, "g", "v", sesoi=0.3).verdict == "sound"


def test_real_difference_is_unsound_for_equivalence() -> None:
    rng = random.Random(5)
    rows = _rows(
        {
            "g": (g := rng.choice(["a", "b"])),
            "v": (0.0 if g == "a" else 1.0) + rng.gauss(0, 1),
        }
        for _ in range(1200)
    )
    assert verify_equivalence(rows, "g", "v", sesoi=0.3).verdict == "unsound"


def test_underpowered_equivalence_is_inconclusive() -> None:
    # a tiny sample can neither establish equivalence nor a difference
    rng = random.Random(6)
    rows = _rows({"g": rng.choice(["a", "b"]), "v": rng.gauss(0, 1)} for _ in range(30))
    assert verify_equivalence(rows, "g", "v", sesoi=0.1).verdict == "inconclusive"


def test_equivalence_requires_a_sesoi() -> None:
    rng = random.Random(7)
    rows = _rows(
        {"g": rng.choice(["a", "b"]), "v": rng.gauss(0, 1)} for _ in range(400)
    )
    assert verify_equivalence(rows, "g", "v", sesoi=0).verdict == "inconclusive"
