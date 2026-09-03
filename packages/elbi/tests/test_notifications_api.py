"""Integration tests for the notification inbox: the acceptance criteria end to end.

Each test drives the real HTTP surface over a real store, so what reaches the inbox is
what the app actually wrote rather than what a stubbed service reported: a monitor's
alert, a run's failure, a finished training, and the unread count that tracks them.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from elbi import create_app, notifications
from elbi.db import Store, open_store
from elbi.monitoring import MonitorService

NEXT: dict[str, float] = {"value": 100.0}


@pytest.fixture(autouse=True)
def _reset() -> None:
    NEXT["value"] = 100.0


class _StubModels:
    """The minimal training surface: instant results, or a scripted failure.

    ``tracking_uri``/``registry``/``drift_result`` back only the scheduled
    drift-tick tests: a fresh scratch mlflow store means ``_last_drift_check_ms``
    finds no prior experiment and returns 0, so no real training is needed.
    """

    def __init__(self, fail: bool = False, tracking_dir: Path | None = None) -> None:
        self.fail = fail
        self._tracking_dir = tracking_dir or Path(
            tempfile.mkdtemp(prefix="stub-mlflow-")
        )

    def check_train(self, name: str, dataset: str, target: str, kind: str) -> None:
        return None

    def latest_version(self, name: str) -> int:
        return 0

    def tracking_uri(self) -> str:
        return f"sqlite:///{self._tracking_dir / 'mlflow.db'}"

    def registry(self) -> Any:
        return SimpleNamespace(models=lambda: [SimpleNamespace(name="churn")])

    def drift_result(self, name: str, current: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "dataset_drift": True,
            "version": 1,
            "share_drifted": 0.5,
            "n_drifted": 2,
        }

    def train_result(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("no signal in the data")
        return {
            "name": kwargs["name"],
            "version": 1,
            "task": "classification",
            "metrics": {"accuracy": 1.0},
            "oracle_verdict": "sound",
            "rendered": "trained",
        }


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'app.db'}")


@pytest.fixture(autouse=True)
def _no_mlflow_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stub model service has no MLflow store to mount a UI over."""
    monkeypatch.setattr("elbi.app.mount_mlflow_ui", lambda *a, **k: False)


def _inbox(http: TestClient) -> dict[str, Any]:
    response = http.get("/api/notifications")
    assert response.status_code == 200, response.text
    return response.json()


def _spike_monitor(http: TestClient) -> None:
    """Create a monitor, build a baseline, then feed it a spike."""
    monitor_id = http.post(
        "/api/monitors",
        json={"name": "p95 watch", "target_kind": "metric", "target": "p95"},
    ).json()["id"]
    for value in [100.0, 101.0, 99.0, 100.0, 102.0, 98.0, 100.0, 101.0]:
        NEXT["value"] = value
        http.post(f"/api/monitors/{monitor_id}/check")
    NEXT["value"] = 5000.0
    http.post(f"/api/monitors/{monitor_id}/check")


# -- acceptance: a finished training job notifies -------------------------------
def _await_jobs(http: TestClient) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        jobs = http.get("/api/jobs").json()
        if jobs and all(
            j["state"] in ("succeeded", "failed", "cancelled") for j in jobs
        ):
            return
        time.sleep(0.05)
    raise AssertionError("training job did not finish")


def _await_unread(http: TestClient, count: int) -> dict[str, Any]:
    """Poll the inbox until ``count`` unread rows appear (or fail after 5s).

    A job's terminal state lands in the store just before ``on_complete`` writes
    the notification, so a reader that races the worker briefly sees the old count.
    """
    deadline = time.time() + 5
    while time.time() < deadline:
        inbox = _inbox(http)
        if inbox["unread"] == count:
            return inbox
        time.sleep(0.05)
    raise AssertionError(f"inbox never reached {count} unread: {_inbox(http)}")


@pytest.fixture
def client(store: Store) -> Iterator[TestClient]:
    monitor_service = MonitorService(
        store=store,
        read_value=lambda monitor: NEXT["value"],
        source_certified=lambda kind, target: True,
        on_alert=notifications.monitor_alert_handler(store),
    )
    app = _app(store, monitor_service=monitor_service, model_service=_StubModels())
    with TestClient(app) as http:
        yield http


def _app(store: Store, **kwargs: Any) -> Any:
    return create_app(
        load_datasets=lambda: {"d": []},
        client=None,
        store=store,
        **kwargs,
    )


def test_a_finished_training_job_notifies(client: TestClient) -> None:
    started = client.post(
        "/api/registry/train",
        json={"name": "churn", "dataset": "d", "target": "y", "time_budget": 1},
    )
    assert started.status_code == 200, started.text
    _await_jobs(client)
    inbox = _inbox(client)
    assert inbox["unread"] == 1
    (item,) = inbox["items"]
    assert item["eventType"] == "model_version.created"
    assert item["targetId"] == "churn/v1"
    assert item["verdict"] == "sound"


def test_failed_training_notifies_without_a_webhook(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    webhook: list[str] = []
    monkeypatch.setattr(
        notifications,
        "fire_model_event",
        lambda store, action, data: webhook.append(action) or True,
    )
    app = _app(store, model_service=_StubModels(fail=True))
    with TestClient(app) as http:
        started = http.post(
            "/api/registry/train",
            json={"name": "churn", "dataset": "d", "target": "y", "time_budget": 1},
        )
        assert started.status_code == 200, started.text
        _await_jobs(http)
        inbox = _await_unread(http, 1)
        (item,) = inbox["items"]
        assert item["eventType"] == "training.failed"
        assert "no signal in the data" in item["body"]
    # training.failed is in-app only: the webhook stream never saw it.
    assert "training.failed" not in webhook


def test_cancelled_training_does_not_notify_success(store: Store) -> None:
    """Cancellation is cooperative: the work may run to completion anyway, after which
    the runner discards the result and stamps the job cancelled. A "registered"
    notification must not land for a job that was cancelled."""
    import threading

    gate = threading.Event()

    class _GatedModels(_StubModels):
        def train_result(self, **kwargs: Any) -> dict[str, Any]:
            gate.wait(timeout=10)
            return super().train_result(**kwargs)

    with TestClient(_app(store, model_service=_GatedModels())) as http:
        job_id = http.post(
            "/api/registry/train",
            json={"name": "churn", "dataset": "d", "target": "y", "time_budget": 1},
        ).json()["id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            if http.get(f"/api/jobs/{job_id}").json()["state"] == "running":
                break
            time.sleep(0.02)
        assert http.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        gate.set()
        _await_jobs(http)
        assert http.get(f"/api/jobs/{job_id}").json()["state"] == "cancelled"
        time.sleep(0.2)  # a (wrong) notification would land within this window
        assert _inbox(http) == {"items": [], "unread": 0}


def test_orphaned_training_notifies_on_restart(store: Store) -> None:
    """A restart strands a running training; the reconciler fails it with no user code
    running, and it must still be announced."""
    from elbi_core import Job

    store.job_store().create(
        Job(
            id="job_orphan1",
            key="k-orphan",
            label="train model churn",
            state="running",
            created_at=time.time(),
        )
    )
    # Building the app constructs the JobRunner, whose orphan reconciler fails the
    # stranded job and fires on_complete like any other terminal transition.
    with TestClient(_app(store, model_service=_StubModels())) as http:
        inbox = _await_unread(http, 1)
        (item,) = inbox["items"]
        assert item["eventType"] == "training.failed"
        assert "interrupted" in item["body"]


def test_the_inbox_reads_its_rows_and_its_count_together(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One read, so the badge and the list cannot contradict each other.

    They were fetched separately, and a notification committed between the two queries
    landed in one and not the other: a response carrying `unread: 1` with `items: []`.
    The list and the count now come from a single session, and this holds the endpoint
    to that -- splitting it again is what would reopen the window, and no amount of
    quiet-database assertion would notice.
    """
    reads: list[str] = []
    for name in (
        "notification_inbox",
        "list_notifications",
        "unread_notification_count",
    ):
        original = getattr(store, name)

        def spy(
            *args: Any, _name: str = name, _original: Any = original, **kwargs: Any
        ):
            reads.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, name, spy)

    notifications.dispatch_event(
        store,
        "run.failed",
        {"event": "run.failed", "run_id": "r1", "cause": "manual", "failed": ["a"]},
    )
    with TestClient(_app(store)) as http:
        inbox = _inbox(http)

    assert inbox["unread"] == len([i for i in inbox["items"] if i["readAt"] is None])
    assert reads == ["notification_inbox"], (
        f"the inbox took {reads} rather than a single combined read"
    )


def test_marking_a_missing_notification_read_is_a_404(
    store: Store, client: TestClient
) -> None:
    notifications.create_notifications(
        store, "run.failed", {"run_id": "r1", "cause": "manual"}
    )
    assert client.post("/api/notifications/nope/read").status_code == 404
    assert _inbox(client)["unread"] == 1


def test_endpoints_degrade_without_a_store() -> None:
    app = create_app(load_datasets=dict, client=None, store=None)
    with TestClient(app) as http:
        assert http.get("/api/notifications").json() == {"items": [], "unread": 0}
        assert http.post("/api/notifications/read-all").json()["count"] == 0
        assert http.post("/api/notifications/x/read").status_code == 404
        prefs = http.get("/api/notifications/preferences").json()
        assert {p["eventType"] for p in prefs["prefs"]} == set(
            notifications.EVENT_TYPES
        )
        assert http.put("/api/notifications/preferences", json={}).status_code == 404


# -- acceptance: read state drives the unread count -----------------------------
def test_read_and_read_all_update_the_unread_count(
    store: Store, client: TestClient
) -> None:
    for run in ("r1", "r2"):
        notifications.create_notifications(
            store, "run.failed", {"run_id": run, "cause": "manual"}
        )
    inbox = _inbox(client)
    assert inbox["unread"] == 2
    first_id = inbox["items"][0]["id"]
    assert client.post(f"/api/notifications/{first_id}/read").status_code == 200
    assert _inbox(client)["unread"] == 1
    done = client.post("/api/notifications/read-all")
    assert done.json() == {"ok": True, "count": 1}
    assert _inbox(client)["unread"] == 0
    # unread_only filters read rows out of the list.
    assert client.get("/api/notifications?unread_only=true").json()["items"] == []


def test_negative_limit_cannot_dump_the_table(store: Store, client: TestClient) -> None:
    for run in ("r1", "r2", "r3"):
        notifications.create_notifications(
            store, "run.failed", {"run_id": run, "cause": "manual"}
        )
    # A negative LIMIT is unbounded on SQLite and an error on Postgres; the
    # route must clamp it before it reaches SQL.
    response = client.get("/api/notifications?limit=-1")
    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) == 1
    assert len(client.get("/api/notifications?limit=2").json()["items"]) == 2


# -- acceptance: per-type preferences suppress creation --------------------------
def test_disabling_a_type_stops_its_notifications(
    store: Store, client: TestClient
) -> None:
    prefs = client.get("/api/notifications/preferences").json()
    assert prefs["emailAvailable"] in (True, False)
    by_type = {p["eventType"]: p for p in prefs["prefs"]}
    assert set(by_type) == set(notifications.EVENT_TYPES)
    assert by_type["metric.anomaly_detected"]["inApp"] is True

    updated = client.put(
        "/api/notifications/preferences",
        json={
            "prefs": [
                {
                    "event_type": "metric.anomaly_detected",
                    "in_app": False,
                    "email": False,
                }
            ]
        },
    )
    assert updated.status_code == 200, updated.text
    by_type = {p["eventType"]: p for p in updated.json()["prefs"]}
    assert by_type["metric.anomaly_detected"]["inApp"] is False

    _spike_monitor(client)
    assert _inbox(client) == {"items": [], "unread": 0}  # suppressed at creation


def test_unknown_preference_type_is_a_400(client: TestClient) -> None:
    response = client.put(
        "/api/notifications/preferences",
        json={"prefs": [{"event_type": "nope.event", "in_app": True}]},
    )
    assert response.status_code == 400
    assert "nope.event" in response.json()["detail"]


# -- acceptance: SMTP unconfigured degrades to in-app only -----------------------
def test_smtp_unconfigured_still_notifies_in_app(
    store: Store, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SMTP_HOST", raising=False)
    _spike_monitor(client)  # anomaly emails by default, but SMTP is absent
    inbox = _inbox(client)
    assert inbox["unread"] == 1
    prefs = client.get("/api/notifications/preferences").json()
    assert prefs["emailAvailable"] is False
