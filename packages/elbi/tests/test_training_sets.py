"""Feature-store training sets: the point-in-time join, persisted and trained on.

A training set is the materialized output of `get_historical_features`. These tests
prove the three things that make it useful: it persists (create/list/get/delete over the
API), it surfaces as a model training source, and a model actually trains on it: the
wiring that connects the feature store to the model registry, which did not exist
before.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.db import open_store
from elbi.features import FeatureStoreService
from elbi.ml import make_model_service
from elbi_core import Artifact, Context, Registry, Runner, derivation, serve
from elbi_core.registry import use_registry

pytestmark = pytest.mark.filterwarnings("ignore")

# One feature row per user, timestamped before any label event so the point-in-time
# join always has a value to attach. ``score`` separates the label cleanly, so a model
# trained on the join is learnable.
FEATURES = [
    {"user_id": f"u{i}", "event_timestamp": "2024-01-01", "score": i} for i in range(60)
]


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def score_src(ctx: Context) -> Artifact:
            return Artifact.table([dict(r) for r in FEATURES])

    return registry


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A store shared by a feature store and a model service, wired together."""
    tmp = tmp_path_factory.mktemp("ts")
    store = open_store(f"sqlite:{tmp / 'app.db'}")
    registry = _registry()
    fs = FeatureStoreService(
        store=store,
        make_runner=lambda: Runner(registry),
        is_certified=lambda name: any(
            d.name == name and d.is_certified for d in registry
        ),
    )
    models = make_model_service(
        store=store,
        cache_dir=tmp,
        load_datasets=lambda: {},
        list_training_sets=lambda: store.list_training_set_names(),
        load_training_set_rows=lambda name: store.training_set_rows(name),
    )
    assert models is not None  # the dev environment installs the ml extra
    return {"store": store, "fs": fs, "models": models}


def _build_training_set(fs: FeatureStoreService) -> dict[str, Any]:
    fs.define_entity(name="user", join_key="user_id")
    fs.define_feature_view(
        {
            "name": "scores",
            "entities": ["user"],
            "source": "score_src",
            "timestampField": "event_timestamp",
            "features": [{"name": "score"}],
        }
    )
    entity_df = [
        {
            "user_id": f"u{i}",
            "event_timestamp": "2024-06-01",
            "label": "1" if i >= 30 else "0",
        }
        for i in range(60)
    ]
    return fs.create_training_set(
        "churn_ts", ["scores:score"], entity_df, label="label"
    )


def _client(bundle: dict[str, Any]) -> TestClient:
    return TestClient(
        create_app(
            load_datasets=lambda: {},
            client=MagicMock(),
            store=bundle["store"],
            feature_store_service=bundle["fs"],
            model_service=bundle["models"],
        )
    )


def test_pit_join_persists_as_a_training_set(bundle: dict[str, Any]) -> None:
    summary = _build_training_set(bundle["fs"])
    # The join attached the feature onto each labelled entity row, leakage-free.
    assert summary["row_count"] == 60
    assert set(summary["columns"]) >= {"user_id", "label", "score"}
    with _client(bundle) as http:
        listed = http.get("/api/features/training-sets").json()
        assert "churn_ts" in [t["name"] for t in listed]
        got = http.get("/api/features/training-sets/churn_ts").json()
        assert got["rowCount"] == 60 and got["label"] == "label"
        assert len(got["sample"]) == 20


def test_training_set_is_offered_as_a_training_source(bundle: dict[str, Any]) -> None:
    _build_training_set(bundle["fs"])
    with _client(bundle) as http:
        sources = http.get("/api/registry/feature-sources").json()
        assert {"name": "churn_ts", "kind": "training_set"} in sources


def test_a_model_trains_on_the_training_set(bundle: dict[str, Any]) -> None:
    _build_training_set(bundle["fs"])
    report = bundle["models"].train_report(
        "churn_model",
        "churn_ts",
        "label",
        ("score",),
        "auto",
        5.0,
        None,
        source_kind="training_set",
    )
    # A dict, not a report object: the search runs in a child process and JSON is what
    # crosses back.
    assert report["version"] >= 1
    assert report["target"] == "label"
    assert "score" in report["features"]
    # The registered version records where it came from.
    versions = bundle["models"].registry().versions("churn_model")
    assert versions[0].tags["elbi.source_kind"] == "training_set"


def test_delete_training_set(bundle: dict[str, Any]) -> None:
    _build_training_set(bundle["fs"])
    with _client(bundle) as http:
        assert http.delete("/api/features/training-sets/churn_ts").status_code == 200
        assert "churn_ts" not in [
            t["name"] for t in http.get("/api/features/training-sets").json()
        ]


def test_unknown_training_set_is_a_404(bundle: dict[str, Any]) -> None:
    with _client(bundle) as http:
        assert http.get("/api/features/training-sets/ghost").status_code == 404
        assert http.delete("/api/features/training-sets/ghost").status_code == 404
