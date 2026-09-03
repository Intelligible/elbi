"""The regression net: every trap's gate must return its ground-truth verdict.

Parametrized over the loaded manifests, so each trap is its own case and a new trap
folder becomes a new case with no code change. This is what makes "reverting a gate
turns its trap red" a blocking check: a gate that stops catching a pitfall returns the
wrong verdict (or drops the pivotal reason), and the corresponding case fails.
"""

from __future__ import annotations

import pytest
from benchmark.harness import score
from benchmark.llm_baseline import load_baseline
from benchmark.loader import load_traps
from benchmark.report import PITFALL_ORDER
from benchmark.trap import Trap

_TRAPS = load_traps()
_BASELINE = load_baseline()


def test_catalog_is_non_empty() -> None:
    assert _TRAPS, "no trap manifests were loaded"


def test_every_pitfall_is_covered() -> None:
    covered = {t.pitfall for t in _TRAPS}
    assert covered == set(PITFALL_ORDER), (
        f"pitfalls not covered: {set(PITFALL_ORDER) - covered}"
    )


def test_inconclusive_is_represented() -> None:
    # The rubric's golden rule makes "inconclusive" a first-class correct verdict, so
    # the suite must prove the gate returns it, not only "unsound".
    assert any(t.expected_verdict == "inconclusive" for t in _TRAPS)


@pytest.mark.parametrize("trap", _TRAPS, ids=lambda t: t.id)
def test_gate_catches_the_trap(trap: Trap) -> None:
    result = score(trap, _BASELINE)
    assert result.verdict == trap.expected_verdict, (
        f"{trap.id} ({trap.fixture}): gate returned {result.verdict!r}, "
        f"expected {trap.expected_verdict!r}"
    )
    if trap.expected_pivotal is not None:
        assert trap.expected_pivotal in result.pivotal_text, (
            f"{trap.id} ({trap.fixture}): expected pivotal "
            f"{trap.expected_pivotal!r} not found in the report"
        )
    assert result.passed
