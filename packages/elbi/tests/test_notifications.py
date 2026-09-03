"""Unit tests for the notification dispatcher, store methods, and email guards.

The dispatcher's contract: notification rows are written before (and regardless
of) webhook delivery, the webhook payload never changes shape, a disabled
preference means the row is never created, and no notification failure can break
the operation that produced the event.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import select

from elbi import mailer, notifications
from elbi.db import Notification, Store, open_store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'notif.db'}")


@pytest.fixture
def webhook_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the webhook leg of dispatch, so tests see exactly what it gets."""
    calls: list[dict[str, Any]] = []

    def _capture(store: object, action: str, data: dict[str, Any]) -> bool:
        calls.append({"action": action, "data": data})
        return True

    monkeypatch.setattr(notifications, "fire_model_event", _capture)
    return calls


# -- dispatch ------------------------------------------------------------------
def test_dispatch_writes_row_and_still_fires_webhook(
    store: Store, webhook_calls: list[dict[str, Any]]
) -> None:
    notifications.dispatch_event(
        store,
        "run.failed",
        {"event": "run.failed", "run_id": "r1", "cause": "manual", "failed": ["a"]},
    )
    rows = store.list_notifications()
    assert len(rows) == 1
    assert rows[0].event_type == "run.failed"
    assert rows[0].target_type == "orchestration_run"
    assert rows[0].target_id == "r1"
    assert rows[0].read_at is None
    assert [c["action"] for c in webhook_calls] == ["run.failed"]


def test_dispatch_writes_rows_even_without_webhook(store: Store) -> None:
    # No webhook URL is configured on this store: fire_model_event returns False,
    # but the in-app row must exist anyway (the whole point of the dispatcher).
    delivered = notifications.dispatch_event(
        store,
        "metric.anomaly_detected",
        {"event": "metric.anomaly_detected", "monitor_id": "m1", "monitor": "p95"},
    )
    assert delivered is False
    assert len(store.list_notifications()) == 1


def test_disabled_in_app_pref_means_no_row(store: Store) -> None:
    store.set_notification_prefs({"metric.anomaly_detected": (False, False)})
    notifications.create_notifications(
        store,
        "metric.anomaly_detected",
        {"monitor_id": "m1", "monitor": "p95"},
    )
    assert store.list_notifications() == []
    # A different type still lands.
    notifications.create_notifications(
        store, "run.failed", {"run_id": "r1", "cause": "manual"}
    )
    assert len(store.list_notifications()) == 1


def test_unknown_event_type_writes_nothing(store: Store) -> None:
    notifications.create_notifications(store, "alias.updated", {})
    assert store.list_notifications() == []


def test_one_event_writes_one_row(store: Store) -> None:
    notifications.create_notifications(
        store, "run.failed", {"run_id": "r1", "cause": "manual"}
    )
    assert len(store.list_notifications()) == 1


def test_notification_failure_never_breaks_the_operation(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(rows: object) -> None:
        raise RuntimeError("db exploded")

    monkeypatch.setattr(store, "create_notifications", _boom)
    # Swallowed and logged, never raised.
    notifications.create_notifications(store, "run.failed", {"run_id": "r1"})


def test_oversized_payload_is_dropped_not_stored(store: Store) -> None:
    notifications.create_notifications(
        store,
        "run.failed",
        {"run_id": "r1", "cause": "manual", "failed": ["x" * 20000]},
    )
    row = store.list_notifications()[0]
    assert row.payload_json == '{"truncated": true}'


# -- read state ----------------------------------------------------------------
def test_read_state_and_unread_count(store: Store) -> None:
    for run in ("r1", "r2"):
        notifications.create_notifications(
            store, "run.failed", {"run_id": run, "cause": "manual"}
        )
    assert store.unread_notification_count() == 2
    first = store.list_notifications()[0]
    assert store.mark_notification_read(first.id) is True
    assert store.unread_notification_count() == 1
    assert [n.id for n in store.list_notifications(unread_only=True)] != [first.id]
    assert store.mark_all_notifications_read() == 1
    assert store.unread_notification_count() == 0


# -- retention -----------------------------------------------------------------
def test_maintenance_prunes_old_notifications(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from elbi.maintenance import run_once

    notifications.create_notifications(
        store, "run.failed", {"run_id": "r1", "cause": "manual"}
    )
    with store._write() as session:
        row = session.exec(select(Notification)).first()
        assert row is not None
        row.at = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=10)
        session.add(row)
        session.commit()
    monkeypatch.setenv("NOTIFICATION_RETENTION_DAYS", "0")  # disabled: keep all
    assert run_once(store)["notifications_pruned"] == 0
    monkeypatch.setenv("NOTIFICATION_RETENTION_DAYS", "1")
    assert run_once(store)["notifications_pruned"] == 1
    assert store.list_notifications() == []


# -- email ---------------------------------------------------------------------
def test_no_smtp_means_in_app_only_and_no_error(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SMTP_HOST", raising=False)
    # run.failed defaults to email on; without SMTP the guard skips silently.
    notifications.create_notifications(
        store, "run.failed", {"run_id": "r1", "cause": "manual"}
    )
    row = store.list_notifications()[0]
    assert row.emailed_at is None


def test_deliver_email_stamps_on_success(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    notifications.create_notifications(
        store, "run.failed", {"run_id": "r1", "cause": "manual"}
    )
    row = store.list_notifications()[0]
    monkeypatch.setattr(notifications.mailer, "send", lambda *a, **k: True)
    notifications._deliver_email(store, row.id, "alice@x.com", "s", "t", "h")
    assert store.list_notifications()[0].emailed_at is not None
    # A failed send leaves the stamp unset (no retry).
    other = Notification(event_type="run.failed")
    other_id = other.id  # read before the insert detaches the instance
    store.create_notifications([other])
    monkeypatch.setattr(notifications.mailer, "send", lambda *a, **k: False)
    notifications._deliver_email(store, other_id, "alice@x.com", "s", "t", "h")
    refetched = [n for n in store.list_notifications() if n.id == other_id]
    assert refetched[0].emailed_at is None


class _InlineThread:
    """A Thread stand-in that runs its target synchronously on start().

    Lets a test drive the email leg deterministically instead of joining a
    fire-and-forget daemon thread.
    """

    def __init__(self, *, target: Any, args: tuple[Any, ...], **_kwargs: Any) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


def test_configured_smtp_emails_through_the_real_path(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the email leg must survive the insert expiring the ORM rows.

    Goes through the public create path end to end (insert, guards, send, stamp)
    rather than calling the internals directly, which is how the detached-row bug
    slipped past the original suite.
    """
    monkeypatch.setenv("SMTP_HOST", "smtp.local")
    monkeypatch.setenv("SMTP_TO", "alice@x.com")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        notifications.mailer,
        "send",
        lambda to, subject, text, html=None: sent.append((to, subject)) or True,
    )
    monkeypatch.setattr(notifications.threading, "Thread", _InlineThread)
    notifications.create_notifications(
        store, "run.failed", {"run_id": "r1", "cause": "manual"}
    )
    assert [to for to, _subject in sent] == ["alice@x.com"]
    row = store.list_notifications()[0]
    assert row.emailed_at is not None


def test_notification_email_escapes_html() -> None:
    _subject, _text, html = mailer.notification_email(
        title="Monitor '<script>x()</script>' fired",
        body="<b>reason</b>",
        event_type="metric.anomaly_detected",
    )
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>reason</b>" not in html


def test_notification_email_subject_strips_newlines() -> None:
    subject, _text, _html = mailer.notification_email(
        title="line\r\nBcc: evil@x.com", body="", event_type="run.failed"
    )
    assert "\n" not in subject and "\r" not in subject
