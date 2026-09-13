"""Runs the stale-skill example end to end (part of the repo CI suite).

The regression this guards: a skill file frozen at yesterday's close should
never agree with what ``risk_overlay`` computes once this morning's bar is in
the CSV. If a fixture edit ever makes them agree, this is the test that catches
it before the example does.
"""

from __future__ import annotations

from pathlib import Path

from elbi_core import Registry, Runner
from elbi_core.config import DataBindings
from elbi_core.discovery import discover

PROJECT = Path(__file__).resolve().parent.parent

# What skills/risk.md claims, frozen at yesterday's close.
SKILL_CUT_TTD = 0.028


def _runner() -> Runner:
    registry = Registry()
    discover(PROJECT / "derivations", registry=registry)
    bindings = DataBindings.load(PROJECT / "elbi.dev.yaml")
    return Runner(registry, bindings=bindings, base_dir=PROJECT)


def test_cov_matrix_is_opaque_and_internal() -> None:
    runner = _runner()
    artifact = runner.run("cov_matrix")
    assert artifact.kind == "opaque"
    assert "TTD" in artifact.value["symbols"]


def test_overlay_sees_this_mornings_bar() -> None:
    result = _runner().run("risk_overlay")
    assert result.value["as_of"] == "2026-08-07"  # the CSV's last row, not yesterday's


def test_overlay_cuts_far_more_than_the_stale_skill_claims() -> None:
    """The whole example in one assertion: today's cut dwarfs yesterday's."""
    live = _runner().run("risk_overlay").value
    live_cut = abs(live["cuts"]["TTD"])
    assert live_cut > 2 * SKILL_CUT_TTD
    assert set(live["cuts"]) == {"TTD"}  # the shock is isolated to one name
