"""Validation of the multiverse / specification-curve robustness engine.

Each test builds data with a known robustness and asserts the certificate: a genuine
effect is robust across the multiverse, each way an effect can be specification-
dependent is caught as fragile with the pivotal decision named, a bad control
(collider) is excluded rather than counted, the joint test is reproducible, and,
the headline, a conclusion that survives every perturbation *alone* but fails under a
*combination* is caught here where the single-axis effect gate certifies it sound.
"""

from __future__ import annotations

import random

from elbi_core.verification import verify_effect, verify_multiverse

N = 500
# a small resample count keeps the suite fast; verdicts are stable well below the
# default because the test cases are clearly robust or clearly fragile
_RESAMPLES = 120


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(round(v, 4)) for k, v in row.items()} for row in maker]


def _robust(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return _rows(
        {
            "x": (x := rng.gauss(0, 1)),
            "y": 0.8 * x + rng.gauss(0, 1),
            "z0": rng.gauss(0, 1),
        }
        for _ in range(N)
    )


def _confounded(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    out = []
    for _ in range(N):
        z = rng.gauss(0, 1)
        out.append(
            {"x": z + rng.gauss(0, 0.5), "y": 1.5 * z + rng.gauss(0, 0.5), "z": z}
        )
    return _rows(out)


def _collider(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    out = []
    for _ in range(N):
        x, y = rng.gauss(0, 1), rng.gauss(0, 1)
        out.append({"x": x, "y": y, "c": x + y + rng.gauss(0, 0.3)})
    return _rows(out)


def _noise(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return _rows(
        {"x": rng.gauss(0, 1), "y": rng.gauss(0, 1), "z0": rng.gauss(0, 1)}
        for _ in range(N)
    )


def _outlier_driven(seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    out = [
        {"x": rng.gauss(0, 1), "y": rng.gauss(0, 1), "z0": rng.gauss(0, 1)}
        for _ in range(N)
    ]
    for i in rng.sample(range(N), 8):
        out[i] = {
            "x": 8 + rng.gauss(0, 0.3),
            "y": 8 + rng.gauss(0, 0.3),
            "z0": rng.gauss(0, 1),
        }
    return _rows(out)


def _joint_fragile(seed: int) -> list[dict[str, str]]:
    # a spurious x-y link held up by an uncontrolled confounder w AND by outliers:
    # controlling w alone or dropping outliers alone each leaves it significant, but
    # both together erase it
    rng = random.Random(seed)
    out = []
    for _ in range(N):
        w = rng.gauss(0, 1)
        out.append({"x": w + rng.gauss(0, 0.5), "y": w + rng.gauss(0, 0.5), "w": w})
    for i in rng.sample(range(N), 18):
        out[i] = {
            "x": 6 + rng.gauss(0, 0.3),
            "y": 6 + rng.gauss(0, 0.3),
            "w": rng.gauss(0, 1),
        }
    return _rows(out)


def test_genuine_effect_is_robust() -> None:
    report = verify_multiverse(_robust(0), "x", "y", resamples=_RESAMPLES)
    assert report.verdict == "robust" and report.p_value < 0.05


def test_confounded_effect_is_fragile() -> None:
    report = verify_multiverse(_confounded(0), "x", "y", resamples=_RESAMPLES)
    assert report.verdict == "fragile"
    assert "z" in (report.pivotal or "")


def test_collider_is_excluded_not_counted() -> None:
    # controlling a collider would manufacture an association; the engine must not
    # vary it, so with x and y independent the certificate is inconclusive
    report = verify_multiverse(_collider(0), "x", "y", resamples=_RESAMPLES)
    assert report.verdict == "inconclusive"


def test_noise_is_inconclusive() -> None:
    report = verify_multiverse(_noise(0), "x", "y", resamples=_RESAMPLES)
    assert report.verdict == "inconclusive"


def test_outlier_driven_effect_is_not_robust() -> None:
    # the spurious link appears only in the raw, outliers-kept corner of the
    # multiverse, so it must never be certified robust (fragile or inconclusive)
    report = verify_multiverse(_outlier_driven(0), "x", "y", resamples=_RESAMPLES)
    assert report.verdict != "robust"


def test_joint_fragility_caught_where_single_gate_passes() -> None:
    # the headline: each perturbation alone leaves the effect, so verify_effect
    # certifies it sound, but the multiverse finds it fails under the combination
    rows = _joint_fragile(0)
    assert verify_effect(rows, "x", "y").verdict == "sound"
    assert verify_multiverse(rows, "x", "y", resamples=_RESAMPLES).verdict == "fragile"


def test_stable_across_seeds() -> None:
    assert all(
        verify_multiverse(_robust(s), "x", "y", resamples=_RESAMPLES).verdict
        == "robust"
        for s in range(4)
    )
    assert all(
        verify_multiverse(_confounded(s), "x", "y", resamples=_RESAMPLES).verdict
        == "fragile"
        for s in range(4)
    )


def test_reproducible() -> None:
    rows = _confounded(2)
    a = verify_multiverse(rows, "x", "y", resamples=_RESAMPLES)
    b = verify_multiverse(rows, "x", "y", resamples=_RESAMPLES)
    assert a.verdict == b.verdict and a.p_value == b.p_value and a.curve == b.curve
