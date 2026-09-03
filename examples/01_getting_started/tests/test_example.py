"""Runs the getting-started example end to end (part of the repo CI suite)."""

from __future__ import annotations

from pathlib import Path

from elbi_core import Registry, Runner
from elbi_core.config import DataBindings
from elbi_core.discovery import discover

PROJECT = Path(__file__).resolve().parent.parent


def _runner() -> Runner:
    registry = Registry()
    discover(PROJECT / "derivations", registry=registry)
    bindings = DataBindings.load(PROJECT / "elbi.dev.yaml")
    return Runner(registry, bindings=bindings, base_dir=PROJECT)


def test_churn_risk_runs() -> None:
    artifact = _runner().run("churn_risk")
    assert len(artifact.value) == 4
    assert {"customer_id", "risk"} <= artifact.value[0].keys()


def test_pricing_effects_composes_on_churn_risk() -> None:
    rendered = _runner().serve("pricing_effects")
    assert rendered.startswith("# Pricing effects")
    assert "risk" in rendered


def test_metric_gates_run_on_a_governed_semantic_model() -> None:
    report = _runner().run("metric_gates").value
    verdicts = {row["metric"]: row["ok"] for row in report}
    # Both governed metrics aggregate a column the sales rows actually carry.
    assert verdicts == {"total_amount": True, "avg_amount": True}
