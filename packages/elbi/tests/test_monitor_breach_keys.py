"""A keyed monitor names its breaching rows and alerts when the breach widens.

The case this covers: a monitor sums a 0/1 breach column over one row per
workflow, with ``max_value: 0``. The first broken workflow opens an incident.
Before keys, a second workflow breaking later only moved the sum from 1 to 2,
which folded into the open incident, so nobody heard about it while the first
stayed broken.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from elbi import notifications
from elbi.db import Store, open_store
from elbi.monitoring import BREACH_WIDENED, MonitorService, Reading

READING: dict[str, Any] = {"value": Reading(0.0, ())}
ALERTS: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    READING["value"] = Reading(0.0, ())
    ALERTS.clear()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'app.db'}")


def _service(store: Store) -> MonitorService:
    return MonitorService(
        store=store,
        read_value=lambda monitor: READING["value"],
        source_certified=lambda kind, target: True,
        on_alert=ALERTS.append,
    )


def _monitor(service: MonitorService) -> str:
    return service.create(
        name="workflow errors",
        target_kind="derivation",
        target="workflow_health",
        config={"measure": {"column": "error_breach", "agg": "sum", "key": "id"}},
        max_value=0,
        window=2,
    ).id


def _read(service: MonitorService, monitor_id: str, *keys: str) -> dict[str, Any]:
    READING["value"] = Reading(float(len(keys)), tuple(sorted(keys)))
    return service.check(monitor_id)


def test_the_opening_alert_names_the_breaching_rows(store: Store) -> None:
    service = _service(store)
    monitor_id = _monitor(service)
    _read(service, monitor_id)
    _read(service, monitor_id, "compactor")
    assert [a["event"] for a in ALERTS] == ["metric.anomaly_detected"]
    assert ALERTS[0]["breach_keys"] == ["compactor"]


def test_a_second_breaking_row_alerts_while_the_first_is_still_open(
    store: Store,
) -> None:
    service = _service(store)
    monitor_id = _monitor(service)
    _read(service, monitor_id)
    _read(service, monitor_id, "compactor")
    _read(service, monitor_id, "compactor")  # same breach: folds, no alert
    _read(service, monitor_id, "compactor", "lead_capture")
    assert [a["event"] for a in ALERTS] == [
        "metric.anomaly_detected",
        BREACH_WIDENED,
    ]
    assert ALERTS[1]["new_keys"] == ["lead_capture"]
    assert ALERTS[1]["breach_keys"] == ["compactor", "lead_capture"]
    # Still one incident: widening reports, it does not reopen.
    incidents = service.history(monitor_id).incidents
    assert [i.closed_at is None for i in incidents] == [True]


def test_a_swap_at_the_same_count_still_alerts(store: Store) -> None:
    service = _service(store)
    monitor_id = _monitor(service)
    _read(service, monitor_id)
    _read(service, monitor_id, "compactor")
    _read(service, monitor_id, "lead_capture")  # sum unchanged at 1, row changed
    assert [a["event"] for a in ALERTS] == ["metric.anomaly_detected", BREACH_WIDENED]
    assert ALERTS[1]["new_keys"] == ["lead_capture"]


def test_a_shrinking_breach_is_quiet_and_recovery_lists_no_keys(store: Store) -> None:
    service = _service(store)
    monitor_id = _monitor(service)
    _read(service, monitor_id)
    _read(service, monitor_id, "compactor", "lead_capture")
    _read(service, monitor_id, "compactor")
    _read(service, monitor_id)
    assert [a["event"] for a in ALERTS] == [
        "metric.anomaly_detected",
        "metric.recovered",
    ]
    assert ALERTS[1]["breach_keys"] == []


def test_a_restart_inside_an_open_incident_does_not_realert(store: Store) -> None:
    first = _service(store)
    monitor_id = _monitor(first)
    _read(first, monitor_id)
    _read(first, monitor_id, "compactor")
    restarted = _service(store)  # fresh process: no remembered keys
    _read(restarted, monitor_id, "compactor")
    _read(restarted, monitor_id, "compactor", "lead_capture")
    assert [a["event"] for a in ALERTS] == [
        "metric.anomaly_detected",
        BREACH_WIDENED,
    ]


def test_an_unkeyed_monitor_behaves_as_before(store: Store) -> None:
    service = MonitorService(
        store=store,
        read_value=lambda monitor: float(READING["value"].value),
        source_certified=lambda kind, target: True,
        on_alert=ALERTS.append,
    )
    monitor_id = _monitor(service)
    for keys in ((), ("a",), ("a", "b")):
        READING["value"] = Reading(float(len(keys)), keys)
        service.check(monitor_id)
    assert [a["event"] for a in ALERTS] == ["metric.anomaly_detected"]
    assert "breach_keys" not in ALERTS[0]


def test_a_widened_breach_notifies_and_emails_by_default(store: Store) -> None:
    assert notifications.EVENT_TYPES[BREACH_WIDENED] is True
    notifications.create_notifications(
        store,
        BREACH_WIDENED,
        {
            "event": BREACH_WIDENED,
            "monitor_id": "m1",
            "monitor": "workflow errors",
            "new_keys": ["lead_capture"],
        },
    )
    [row] = store.list_notifications()
    assert row.title == "Monitor 'workflow errors' has new breaching rows"
    assert row.body == "Newly breaching: lead_capture"


def test_the_opening_notification_names_the_breaching_rows(store: Store) -> None:
    notifications.create_notifications(
        store,
        "metric.anomaly_detected",
        {
            "event": "metric.anomaly_detected",
            "monitor_id": "m1",
            "monitor": "workflow errors",
            "reason": "1 is above the ceiling 0",
            "breach_keys": ["compactor"],
        },
    )
    [row] = store.list_notifications()
    assert row.body == "1 is above the ceiling 0. Breaching: compactor"
