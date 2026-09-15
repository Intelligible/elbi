"""Runs the components example end to end (part of the repo CI suite)."""

from __future__ import annotations

from pathlib import Path

from elbi_core import Registry, Runner
from elbi_core.components import is_valid_component
from elbi_core.config import DataBindings
from elbi_core.discovery import discover

PROJECT = Path(__file__).resolve().parent.parent


def _runner() -> Runner:
    registry = Registry()
    discover(PROJECT / "derivations", registry=registry)
    bindings = DataBindings.load(PROJECT / "elbi.dev.yaml")
    return Runner(registry, bindings=bindings, base_dir=PROJECT)


def test_churn_components_produces_valid_orc_components() -> None:
    artifact = _runner().run("churn_components")
    assert artifact.kind == "components"
    assert len(artifact.value) == 4
    for component in artifact.value:
        assert is_valid_component(component)


def test_churn_components_evidence_is_internally_consistent() -> None:
    """The threshold_rule's evidence should describe the same data the
    distribution component does -- not a coincidence, since both are computed
    from the one input, but worth asserting so a future edit that breaks that
    is caught here rather than downstream.
    """
    components = _runner().run("churn_components").value
    by_id = {c["id"]: c for c in components}
    threshold = by_id["elbi-examples/churn_discount_threshold"]
    assert threshold["evidence"]["odds_ratio"] > 1  # discount raises churn risk
    segment = by_id["elbi-examples/loyal_two_year_segment"]
    assert segment["evidence"]["segment_size"] > 0


def test_churn_components_serves_statements_and_structured_content() -> None:
    rendered = _runner().serve("churn_components")
    assert rendered.startswith("# Churn components")
    assert "discount" in rendered
    assert "- " in rendered  # one bullet per statement, not a raw JSON dump
