"""Feature-store monitoring over the HTTP surface: statistics snapshots and drift.

Drive the real stack (routes, service, DB, and the shared Evidently drift engine) the
way the model monitor is driven. The source derivation reads a mutable module list so a
test can shift the distribution between setting a baseline and checking for drift, which
is the only way to prove the check distinguishes a drifted view from a stable one.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.db import open_store
from elbi.features import FeatureStoreService
from elbi_core import Artifact, Context, Registry, Runner, derivation, serve
from elbi_core.registry import use_registry

# Importing Evidently pulls in transitive libraries that emit their own deprecation
# warnings; the drift engine itself is exercised here, not those. (Same stance as the
# model-drift tests.)
pytestmark = pytest.mark.filterwarnings("ignore")

DATASETS: dict[str, list[dict[str, Any]]] = {}

#: The source rows, mutated by :func:`_seed` between a baseline and a drift check.
_ROWS: list[dict[str, Any]] = []


def _seed(clicks: list[int]) -> None:
    """Replace the source with one row per click value, each a distinct entity."""
    _ROWS.clear()
    _ROWS.extend(
        {"user_id": f"u{i}", "event_timestamp": "2024-01-01", "clicks": value}
        for i, value in enumerate(clicks)
    )


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def user_stats_src(ctx: Context) -> Artifact:
            return Artifact.table([dict(r) for r in _ROWS])

    return registry


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()
    service = FeatureStoreService(
        store=store,
        make_runner=lambda: Runner(registry),
        is_certified=lambda name: any(
            d.name == name and d.is_certified for d in registry
        ),
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        feature_store_service=service,
    )
    with TestClient(app) as http:
        yield http


def _register(http: TestClient) -> None:
    assert (
        http.post(
            "/api/features/entities", json={"name": "user", "join_key": "user_id"}
        ).status_code
        == 200
    )
    response = http.post(
        "/api/features/views",
        json={
            "name": "user_stats",
            "entities": ["user"],
            "source": "user_stats_src",
            "timestampField": "event_timestamp",
            "features": [{"name": "clicks"}],
        },
    )
    assert response.status_code == 200, response.text


def test_statistics_snapshot_profiles_feature_columns(client: TestClient) -> None:
    _seed([1, 2, 3, 4, 5] * 8)  # 40 rows, five distinct click values
    _register(client)
    snapshot = client.post("/api/features/views/user_stats/statistics", json={}).json()
    assert snapshot["rowCount"] == 40
    # Only the feature column is profiled, not the join key or timestamp.
    assert [f["name"] for f in snapshot["features"]] == ["clicks"]
    clicks = snapshot["features"][0]
    assert clicks["completeness"] == 1.0
    assert clicks["distinct"] == 5

    history = client.get("/api/features/views/user_stats/statistics").json()
    assert len(history["snapshots"]) == 1
    assert history["snapshots"][0]["rowCount"] == 40


def test_drift_check_detects_a_shifted_distribution(client: TestClient) -> None:
    _seed(list(range(1, 41)))  # baseline: clicks 1..40
    _register(client)
    client.post(
        "/api/features/views/user_stats/statistics", json={"set_baseline": True}
    )
    _seed(list(range(1001, 1041)))  # current: clicks shifted far up
    result = client.post("/api/features/views/user_stats/drift-check", json={}).json()
    assert result["datasetDrift"] is True
    assert result["nCurrentRows"] == 40
    drifted = [c["column"] for c in result["columns"] if c["drifted"]]
    assert "clicks" in drifted

    history = client.get("/api/features/views/user_stats/drift").json()["checks"]
    assert len(history) == 1 and history[0]["datasetDrift"] is True


def test_drift_check_is_quiet_when_stable(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    client.post(
        "/api/features/views/user_stats/statistics", json={"set_baseline": True}
    )
    # The distribution has not moved since the baseline.
    result = client.post("/api/features/views/user_stats/drift-check", json={}).json()
    assert result["datasetDrift"] is False


def test_drift_check_needs_a_baseline(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    response = client.post("/api/features/views/user_stats/drift-check", json={})
    assert response.status_code == 409
    assert "baseline" in response.json()["detail"]


def test_scheduled_tick_checks_views_with_a_baseline(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    client.post(
        "/api/features/views/user_stats/statistics", json={"set_baseline": True}
    )
    _seed(list(range(1001, 1041)))  # drift the view since the baseline
    client.app.state.feature_drift_tick()  # type: ignore[attr-defined]
    history = client.get("/api/features/views/user_stats/drift").json()["checks"]
    assert len(history) == 1 and history[0]["datasetDrift"] is True


def test_scheduled_tick_skips_views_without_a_baseline(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    client.app.state.feature_drift_tick()  # type: ignore[attr-defined]
    assert client.get("/api/features/views/user_stats/drift").json()["checks"] == []


# -- acceptance: scheduled feature drift notifies the view's owner -------------


def test_scheduled_feature_drift_lands_in_the_inbox(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    client.post(
        "/api/features/views/user_stats/statistics", json={"set_baseline": True}
    )
    _seed(list(range(1001, 1041)))  # drift the view since the baseline
    client.app.state.feature_drift_tick()  # type: ignore[attr-defined]
    inbox = client.get("/api/notifications").json()
    assert inbox["unread"] == 1
    assert inbox["items"][0]["eventType"] == "feature.drift_detected"


def test_setting_a_new_baseline_retires_the_prior_one(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    client.post(
        "/api/features/views/user_stats/statistics", json={"set_baseline": True}
    )
    _seed(list(range(1001, 1041)))
    client.post(
        "/api/features/views/user_stats/statistics", json={"set_baseline": True}
    )
    # The baseline is now the shifted distribution, so an unchanged view is stable.
    result = client.post("/api/features/views/user_stats/drift-check", json={}).json()
    assert result["datasetDrift"] is False
