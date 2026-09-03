"""Validation of the difference-in-differences pre-trends gate."""

from __future__ import annotations

import random

from elbi_core.verification import verify_did


def _rows(maker) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in row.items()} for row in maker]


def _panel(diverging: bool) -> list[dict[str, str]]:
    rng = random.Random(0 if not diverging else 1)
    out = []
    for unit in range(80):
        treated = unit % 2
        for t in range(10):
            slope = 0.5 + (0.3 * treated if diverging else 0.0)
            y = 2.0 * treated + slope * t + rng.gauss(0, 1)
            out.append({"g": "t" if treated else "c", "time": t, "y": round(y, 4)})
    return _rows(out)


def test_parallel_pretrends_is_sound() -> None:
    # both groups share the same pre-treatment slope -> parallel trends not contradicted
    assert (
        verify_did(_panel(diverging=False), "g", "time", "y", treatment_start=5).verdict
        == "sound"
    )


def test_diverging_pretrends_is_unsound() -> None:
    # the treated group was already trending up faster before treatment
    report = verify_did(_panel(diverging=True), "g", "time", "y", treatment_start=5)
    assert report.verdict == "unsound"
    assert "parallel trends" in (report.pivotal or "")


def test_too_few_pre_periods_is_inconclusive() -> None:
    rows = _panel(diverging=False)
    # treatment_start=1 leaves only one pre-period
    assert (
        verify_did(rows, "g", "time", "y", treatment_start=1).verdict == "inconclusive"
    )
