"""In-app notifications, written from the platform's event vocabulary.

The webhook (``webhooks.fire_model_event``) delivers every event to one
configured URL and drops the event entirely when none is set, so nothing that
happens reaches the product itself. ``dispatch_event`` is the layer above that
channel: it writes durable notification rows (and emails the interrupt-worthy
types), then delegates to the webhook unchanged, so the external stream stays
byte-identical.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

from . import mailer
from .db import Notification, Store
from .webhooks import fire_model_event

logger = logging.getLogger("elbi")

TRAINING_FAILED = "training.failed"

# The notifying event types, mapped to their email default (on = worth
# interrupting someone over). In-app defaults on for all; stored preference
# rows override both. Also the vocabulary the preferences API validates.
EVENT_TYPES: dict[str, bool] = {
    "model_version.created": False,
    TRAINING_FAILED: True,
    "metric.anomaly_detected": True,
    "metric.recovered": False,
    "run.failed": True,
    "run.slow": False,
    "drift.detected": False,
    "feature.drift_detected": False,
    "feature.expectations_failed": True,
}

_PAYLOAD_CAP = 8192  # bytes of stored payload JSON; larger payloads are dropped
_ERROR_CAP = 500  # characters of error text carried into a notification body


def effective_prefs(store: Store) -> dict[str, tuple[bool, bool]]:
    """The full (in_app, email) matrix for one user: stored rows over defaults."""
    stored = store.get_notification_prefs()
    return {
        event_type: stored.get(event_type, (True, default_email))
        for event_type, default_email in EVENT_TYPES.items()
    }


def _render(action: str, data: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """A notification's (title, body, target_type, target_id) for an event.

    Defensive ``get``s throughout: a malformed payload must degrade to a vague
    notification, not an exception in the operation that produced the event.
    """
    if action == "metric.anomaly_detected":
        return (
            f"Monitor {data.get('monitor', '')!r} detected an anomaly",
            str(data.get("reason") or ""),
            "metric_monitor",
            str(data.get("monitor_id") or ""),
        )
    if action == "metric.recovered":
        return (
            f"Monitor {data.get('monitor', '')!r} recovered",
            str(data.get("reason") or ""),
            "metric_monitor",
            str(data.get("monitor_id") or ""),
        )
    if action == "run.failed":
        failed = ", ".join(str(a) for a in (data.get("failed") or []))
        return (
            f"Orchestration run failed ({data.get('cause', 'unknown')})",
            f"Failed assets: {failed}" if failed else "",
            "orchestration_run",
            str(data.get("run_id") or ""),
        )
    if action == "run.slow":
        try:
            duration = float(data.get("duration_ms") or 0)
            median = float(data.get("median_ms") or 0)
            body = f"Took {duration:.0f}ms against a median of {median:.0f}ms"
        except (TypeError, ValueError):
            body = ""
        return (
            f"Orchestration run slower than usual ({data.get('cause', 'unknown')})",
            body,
            "orchestration_run",
            str(data.get("run_id") or ""),
        )
    if action == "model_version.created":
        name = str(data.get("name") or "")
        version = data.get("version")
        return (
            f"Model {name!r} v{version} registered",
            f"Task: {data.get('task')}" if data.get("task") else "",
            "model_version",
            f"{name}/v{version}",
        )
    if action == TRAINING_FAILED:
        return (
            f"Training for model {data.get('name', '')!r} failed",
            str(data.get("error") or "")[:_ERROR_CAP],
            "model",
            str(data.get("name") or ""),
        )
    if action == "drift.detected":
        return (
            f"Drift detected on model {data.get('name', '')!r}",
            f"{data.get('n_drifted', 0)} drifted feature(s)",
            "model",
            str(data.get("name") or ""),
        )
    if action == "feature.drift_detected":
        return (
            f"Feature drift detected in {data.get('feature_view', '')!r}",
            f"{data.get('n_drifted', 0)} drifted feature(s)",
            "feature_view",
            str(data.get("feature_view") or ""),
        )
    if action == "feature.expectations_failed":
        return (
            f"Feature expectations failed in {data.get('feature_view', '')!r}",
            f"{data.get('n_violated', 0)} violated expectation(s)",
            "feature_view",
            str(data.get("feature_view") or ""),
        )
    return action, "", "", ""


def _deliver_email(
    store: Store, notification_id: str, address: str, subject: str, text: str, html: str
) -> None:
    """Send one notification email and stamp the row on success.

    ``mailer.send`` never raises; a failed send simply leaves ``emailed_at``
    unset, and nothing retries (the same doctrine as webhook delivery).
    """
    if mailer.send(address, subject, text, html):
        store.mark_notification_emailed(notification_id)


def _maybe_email(
    store: Store, notification_id: str, title: str, body: str, event_type: str
) -> None:
    """Email one notification if every guard passes, on a daemon thread.

    Takes scalars rather than the ORM row: by the time email runs the insert has
    committed and expired the instance, and reading a detached row raises. Guards
    run synchronously so the common cases (no SMTP, no owner, pref off upstream)
    spawn nothing; only an actual send leaves the calling thread.
    """
    if not mailer.is_configured():
        return
    address = mailer.recipient_address()
    if not address:
        return
    subject, text, html = mailer.notification_email(
        title=title, body=body, event_type=event_type
    )
    threading.Thread(
        target=_deliver_email,
        args=(store, notification_id, str(address), subject, text, html),
        name="notification-email",
        daemon=True,
    ).start()


def create_notifications(
    store: Store | None,
    action: str,
    data: Mapping[str, Any],
) -> None:
    """Write preference-filtered notification rows (and email where enabled).

    Swallows every exception: a notification must never break the training run,
    monitor tick, or orchestration finish it describes (the webhook's doctrine).
    A disabled in-app preference means the row is never created, which is why
    filtering happens here at write time rather than at read time.
    """
    if store is None or action not in EVENT_TYPES:
        return
    try:
        payload = json.dumps(dict(data), default=str)
        if len(payload) > _PAYLOAD_CAP:
            payload = json.dumps({"truncated": True})
        title, body, target_type, target_id = _render(action, data)
        verdict = data.get("source_verdict") or data.get("oracle_verdict")
        in_app, email = effective_prefs(store)[action]
        if not in_app:
            return
        row = Notification(
            event_type=action,
            title=title,
            body=body,
            target_type=target_type,
            target_id=target_id,
            verdict=verdict if verdict is None else str(verdict),
            payload_json=payload,
        )
        # Captured before the insert: committing expires the ORM instance, and
        # _maybe_email runs after the session closes.
        notification_id = row.id
        store.create_notifications([row])
        if email:
            _maybe_email(store, notification_id, title, body, action)
    except Exception:
        logger.exception("notification fan-out for %s failed", action)


def dispatch_event(
    store: Store | None,
    action: str,
    data: dict[str, Any],
) -> bool:
    """Notify the event's recipients in-app (and by email), then the webhook.

    Notifications are written first and unconditionally: the webhook's
    no-URL-configured early return lives inside ``fire_model_event``, and in-app
    delivery must not inherit "an operator configured a webhook" as a
    precondition. The payload reaches the webhook untouched.
    """
    create_notifications(store, action, data)
    return fire_model_event(store, action, data)


def monitor_alert_handler(store: Store) -> Callable[[dict[str, Any]], None]:
    """The production ``on_alert`` for ``MonitorService``: notify, webhook, audit."""

    def handle(payload: dict[str, Any]) -> None:
        dispatch_event(store, payload["event"], payload)
        store.record_audit(
            payload["event"],
            target_type="metric_monitor",
            target_id=str(payload.get("monitor_id", "")),
            verdict=payload.get("source_verdict"),
        )

    return handle


def orchestration_alert_handler(store: Store) -> Callable[[dict[str, Any]], None]:
    """The production ``on_notify`` for ``OrchestrationService``."""

    def handle(payload: dict[str, Any]) -> None:
        dispatch_event(store, payload["event"], payload)
        store.record_audit(
            payload["event"],
            target_type="orchestration_run",
            target_id=str(payload.get("run_id", "")),
        )

    return handle
