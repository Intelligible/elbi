"""Metamorphic tests: verdicts must be invariant under changes that cannot matter.

These assert *relations*, not expected answers, so they need no oracle for the "right"
verdict and are not biased by whoever wrote them. A verdict may not depend on the order
of the rows, the names of the group levels, the units of a column (a positive affine
rescale), or the presence of an independent, irrelevant column. Any change to a verdict
under one of these is a bug in the gate, whatever the "correct" verdict is. This is the
check that caught a direction bug the hand-written fixtures could not; a relation holds
across the whole input space, not just the cases the author happened to imagine.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from elbi_core import verify_all

N_SEEDS = 8
Rows = list[dict[str, str]]


def _rows(dicts) -> Rows:
    return [{k: str(v) for k, v in d.items()} for d in dicts]


# --- metamorphic transformations (each preserves the true verdict) --------------


def _permute(rows: Rows, seed: int) -> Rows:
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def _relabel(col: str, mapping: dict[str, str]) -> Callable[[Rows, int], Rows]:
    return lambda rows, _seed: [{**r, col: mapping.get(r[col], r[col])} for r in rows]


def _affine(col: str, scale: float, shift: float) -> Callable[[Rows, int], Rows]:
    # a positive rescale of units: 2*x + 5. Significance and direction are unchanged.
    return lambda rows, _seed: [
        {**r, col: str(scale * float(r[col]) + shift)} for r in rows
    ]


def _add_noise_column(name: str) -> Callable[[Rows, int], Rows]:
    def apply(rows: Rows, seed: int) -> Rows:
        rng = random.Random(seed + 991)
        return [{**r, name: str(rng.gauss(0, 1))} for r in rows]

    return apply


def _assert_invariant(
    make: Callable[[int], Rows],
    claim: dict[str, object],
    transform: Callable[[Rows, int], Rows],
) -> None:
    for seed in range(N_SEEDS):
        rows = make(seed)
        before = verify_all(rows, **claim).verdict
        after = verify_all(transform(rows, seed), **claim).verdict
        assert before == after, f"seed {seed}: {before} -> {after} under transform"


# --- datasets spanning the gate families ---------------------------------------


def _effect(seed: int) -> Rows:
    rng = random.Random(seed)
    return _rows(
        {
            "x": (x := rng.gauss(0, 1)),
            "y": 0.8 * x + rng.gauss(0, 1),
            "z": rng.gauss(0, 1),
        }
        for _ in range(300)
    )


def _comparison(seed: int) -> Rows:
    rng = random.Random(seed)
    return _rows(
        {"g": ("a" if rng.random() < 0.5 else "b"), "v": rng.gauss(0, 1)}
        for _ in range(400)
    )


def _experiment(seed: int) -> Rows:
    rng = random.Random(seed)
    return _rows(
        {
            "variant": (v := "A" if rng.random() < 0.5 else "B"),
            "converted": int(rng.random() < (0.30 if v == "B" else 0.22)),
        }
        for _ in range(4000)
    )


def _screen(seed: int) -> Rows:
    rng = random.Random(seed)
    return _rows(
        {
            "t": (t := rng.randint(0, 1)),
            "m0": rng.gauss(0, 1) + 0.4 * t,
            "m1": rng.gauss(0, 1),
            "m2": rng.gauss(0, 1),
        }
        for _ in range(500)
    )


def _rtm(seed: int) -> Rows:
    rng = random.Random(seed)
    rows = []
    for _ in range(600):
        a = rng.gauss(50, 10)
        t1 = a + rng.gauss(0, 10)
        rows.append({"pre": t1, "post": a + rng.gauss(0, 10), "g": int(t1 < 40)})
    return _rows(rows)


# --- the relations --------------------------------------------------------------


def test_effect_is_invariant() -> None:
    claim = {"x": "x", "y": "y"}
    _assert_invariant(_effect, claim, _permute)
    _assert_invariant(_effect, claim, _affine("y", 2.0, 5.0))
    _assert_invariant(_effect, claim, _affine("x", 3.0, -1.0))
    _assert_invariant(_effect, claim, _add_noise_column("irrelevant"))


def test_comparison_is_invariant() -> None:
    claim = {"group": "g", "value": "v"}
    _assert_invariant(_comparison, claim, _permute)
    _assert_invariant(_comparison, claim, _relabel("g", {"a": "x", "b": "y"}))
    _assert_invariant(_comparison, claim, _affine("v", 10.0, 100.0))


def test_experiment_is_invariant() -> None:
    claim = {"variant": "variant", "metric": "converted"}
    _assert_invariant(_experiment, claim, _permute)
    _assert_invariant(
        _experiment, claim, _relabel("variant", {"A": "ctrl", "B": "test"})
    )
    _assert_invariant(_experiment, claim, _affine("converted", 4.0, 1.0))


def test_screen_is_invariant() -> None:
    claim = {"group": "t", "outcomes": ["m0", "m1", "m2"]}
    _assert_invariant(_screen, claim, _permute)
    _assert_invariant(_screen, claim, _relabel("t", {"0": "off", "1": "on"}))
    _assert_invariant(_screen, claim, _affine("m0", 2.0, 3.0))


def test_rtm_is_invariant() -> None:
    claim = {"before": "pre", "after": "post", "group": "g"}
    _assert_invariant(_rtm, claim, _permute)
    _assert_invariant(_rtm, claim, _relabel("g", {"0": "control", "1": "selected"}))
    _assert_invariant(_rtm, claim, _affine("post", 2.0, 10.0))
