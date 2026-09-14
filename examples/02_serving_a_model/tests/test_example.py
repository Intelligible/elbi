"""Runs the model-serving example end to end (part of the repo CI suite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import DerivationError, Registry, Runner
from elbi_core.config import DataBindings
from elbi_core.discovery import discover

PROJECT = Path(__file__).resolve().parent.parent

CHURNY = {"recency": 110, "frequency": 1, "monetary": 30}
ACTIVE = {"recency": 6, "frequency": 18, "monetary": 480}


def _runner() -> Runner:
    registry = Registry()
    discover(PROJECT / "derivations", registry=registry)
    bindings = DataBindings.load(PROJECT / "elbi.dev.yaml")
    return Runner(registry, bindings=bindings, base_dir=PROJECT)


def test_model_is_opaque_and_internal() -> None:
    runner = _runner()
    artifact = runner.run("churn_model")
    assert artifact.kind == "opaque"  # an arbitrary object, not a rendered shape
    assert "weight" in artifact.value
    with pytest.raises(DerivationError, match="internal"):
        runner.serve("churn_model")  # the model itself is not served


def test_predict_churn_separates_active_from_churned() -> None:
    runner = _runner()
    churny = runner.run("predict_churn", {"customer": CHURNY})
    active = runner.run("predict_churn", {"customer": ACTIVE})
    assert churny.value["churn_probability"] > 0.5
    assert active.value["churn_probability"] < 0.5


def test_predict_batch_scores_each_record() -> None:
    runner = _runner()
    out = runner.run(
        "predict_churn_batch",
        {"customers": [{"id": "a", **CHURNY}, {"id": "b", **ACTIVE}]},
    )
    by_id = {row["id"]: row["churn_probability"] for row in out.value}
    assert by_id["a"] > 0.5 > by_id["b"]


def test_whatif_moves_prediction_in_the_expected_direction() -> None:
    """Poking at an active customer's recency should move the prediction the
    same direction increasing recency moves it for any customer: up.
    """
    runner = _runner()
    out = runner.run(
        "predict_churn_whatif",
        {
            "base": ACTIVE,
            "scenarios": [
                {"recency": CHURNY["recency"]},  # only recency changes
                {"recency": ACTIVE["recency"]},  # unchanged: delta should be ~0
            ],
        },
    )
    worse, unchanged = out.value
    assert worse["delta"] > 0  # a longer time since last purchase raises risk
    assert unchanged["delta"] == 0.0
