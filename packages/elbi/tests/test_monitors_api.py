"""Integration tests for the monitoring API: snapshot, detect, incident dedup, alert.

Drive the HTTP surface with a TestClient over a real store. The monitored value comes
from a controllable stub (so a test can feed a steady series then a spike), and alerts
are captured, so the whole loop is exercised: build a baseline, flag a spike once (not
per check), and recover.
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
from elbi.monitoring import MonitorService

# The current value the stub source returns; a test mutates it between checks.
NEXT: dict[str, float] = {"value": 100.0}
ALERTS: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    NEXT["value"] = 100.0
    ALERTS.clear()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    service = MonitorService(
        store=store,
        read_value=lambda monitor: NEXT["value"],
        source_certified=lambda kind, target: target != "uncertified",
        on_alert=ALERTS.append,
    )
    app = create_app(
        load_datasets=dict,
        client=MagicMock(),
        store=store,
        monitor_service=service,
    )
    with TestClient(app) as http:
        yield http


def _create(client: TestClient) -> str:
    response = client.post(
        "/api/monitors",
        json={
            "name": "revenue watch",
            "target_kind": "metric",
            "target": "revenue",
            "sensitivity": 3.0,
            "window": 30,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _check(client: TestClient, monitor_id: str, value: float) -> dict[str, Any]:
    NEXT["value"] = value
    response = client.post(f"/api/monitors/{monitor_id}/check")
    assert response.status_code == 200, response.text
    return response.json()


def test_create_and_list(client: TestClient) -> None:
    monitor_id = _create(client)
    monitors = client.get("/api/monitors").json()
    assert [m["id"] for m in monitors] == [monitor_id]
    assert monitors[0]["target"] == "revenue"
    assert monitors[0]["status"] == "ok"


def test_create_rejects_uncertified_target(client: TestClient) -> None:
    response = client.post(
        "/api/monitors",
        json={"target_kind": "metric", "target": "uncertified", "name": "x"},
    )
    assert response.status_code == 400
    assert "not certified" in response.json()["detail"]


def test_baseline_spike_dedup_and_recovery(client: TestClient) -> None:
    monitor_id = _create(client)
    # Build a steady baseline; no anomaly once enough history exists.
    for _ in range(8):
        result = _check(client, monitor_id, 100.0)
    assert result["anomalous"] is False

    # A spike flags and fires exactly one alert (a new incident opens).
    spike = _check(client, monitor_id, 1000.0)
    assert spike["anomalous"] is True
    assert spike["alerted"] is True
    assert len(ALERTS) == 1
    assert ALERTS[0]["event"] == "metric.anomaly_detected"
    assert ALERTS[0]["source_verdict"] == "sound"  # the moved number was verified

    # A second consecutive spike is still anomalous but folds into the open incident.
    again = _check(client, monitor_id, 1100.0)
    assert again["anomalous"] is True
    assert again["alerted"] is False
    assert len(ALERTS) == 1  # no duplicate alert

    # Returning to normal closes the incident and fires a recovery alert.
    recovered = _check(client, monitor_id, 100.0)
    assert recovered["anomalous"] is False
    assert len(ALERTS) == 2
    assert ALERTS[1]["event"] == "metric.recovered"


def test_updating_a_monitor_keeps_its_history(client: TestClient) -> None:
    """Changing a setting must not reset the detector.

    A monitor's snapshots *are* the baseline anomaly detection compares against, and
    its incidents are the record of what it caught. ``elbi sync`` used to push
    a changed monitor by deleting it and creating a replacement, and the delete cascades
    to both -- so editing a threshold in a file silently discarded the history and
    started the monitor over with nothing to compare to.
    """
    monitor_id = _create(client)
    for value in (100.0, 101.0, 99.0, 100.5):
        _check(client, monitor_id, value)
    before = client.get(f"/api/monitors/{monitor_id}").json()
    assert len(before["snapshots"]) == 4

    response = client.put(
        f"/api/monitors/{monitor_id}", json={"sensitivity": 2.0, "window": 14}
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["id"] == monitor_id, "the monitor was replaced, not updated"
    assert updated["sensitivity"] == 2.0
    assert updated["window"] == 14

    after = client.get(f"/api/monitors/{monitor_id}").json()
    assert len(after["snapshots"]) == 4, "the update discarded the baseline"


def test_updating_a_monitor_keeps_its_incidents(client: TestClient) -> None:
    monitor_id = _create(client)
    for value in (100.0, 100.0, 100.0, 100.0, 100.0):
        _check(client, monitor_id, value)
    _check(client, monitor_id, 900.0)  # a spike, which opens an incident
    before = client.get(f"/api/monitors/{monitor_id}").json()
    assert before["incidents"], "expected the spike to raise one"

    client.put(f"/api/monitors/{monitor_id}", json={"sensitivity": 4.0})

    after = client.get(f"/api/monitors/{monitor_id}").json()
    assert len(after["incidents"]) == len(before["incidents"])


def test_retargeting_a_monitor_still_needs_a_certified_target(
    client: TestClient,
) -> None:
    """An edit must not be a way around the gate that creating enforces."""
    monitor_id = _create(client)
    response = client.put(f"/api/monitors/{monitor_id}", json={"target": "uncertified"})
    assert response.status_code == 400
    assert "not certified" in response.json()["detail"]


def test_updating_a_missing_monitor_is_a_404(client: TestClient) -> None:
    assert (
        client.put("/api/monitors/nope", json={"sensitivity": 2.0}).status_code == 404
    )


def test_a_bound_can_be_cleared(client: TestClient) -> None:
    """A threshold removed from the file has to come off the monitor too."""
    monitor_id = _create(client)
    bounded = client.put(f"/api/monitors/{monitor_id}", json={"max_value": 500.0})
    assert bounded.json()["maxValue"] == 500.0
    cleared = client.put(f"/api/monitors/{monitor_id}", json={"sensitivity": 3.0})
    assert cleared.json()["maxValue"] is None


def test_static_bound_flags_before_any_history(client: TestClient) -> None:
    response = client.post(
        "/api/monitors",
        json={
            "target_kind": "metric",
            "target": "revenue",
            "name": "floor",
            "min_value": 0.0,
        },
    )
    monitor_id = response.json()["id"]
    result = _check(client, monitor_id, -5.0)  # below the floor, no history needed
    assert result["anomalous"] is True
    assert "below the floor" in result["reason"]


def test_history_and_incidents_endpoint(client: TestClient) -> None:
    monitor_id = _create(client)
    for value in [100.0, 101.0, 99.0, 100.0, 102.0, 98.0]:
        _check(client, monitor_id, value)
    _check(client, monitor_id, 5000.0)  # open an incident
    detail = client.get(f"/api/monitors/{monitor_id}").json()
    assert len(detail["snapshots"]) == 7
    assert len(detail["incidents"]) == 1
    assert detail["incidents"][0]["open"] is True


def test_delete_monitor(client: TestClient) -> None:
    monitor_id = _create(client)
    assert client.delete(f"/api/monitors/{monitor_id}").status_code == 200
    assert client.get(f"/api/monitors/{monitor_id}").status_code == 404


def test_scheduler_tick_runs_due_monitors(client: TestClient) -> None:
    # A freshly created monitor has never run, so the tick picks it up and snapshots it.
    monitor_id = _create(client)
    NEXT["value"] = 42.0
    client.app.state.monitor_tick()
    detail = client.get(f"/api/monitors/{monitor_id}").json()
    assert detail["snapshots"][-1]["value"] == 42.0
