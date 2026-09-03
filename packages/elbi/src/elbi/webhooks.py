"""Model-registry webhooks, fired where registry writes actually happen.

MLflow 3.3 brought registry webhooks to open source, but they fire from the
tracking *server's* handler layer, and this app writes to the store through the
direct client, which never passes through those handlers. So the app fires its
own events at the operations themselves, with the same vocabulary (a model
version was created, an alias moved, something was deleted) and the same
integrity mechanism: an HMAC-SHA256 signature over the body, GitHub-style, in
``X-Elbi-Signature``, so a receiver can verify the sender holds the
shared secret.

Delivery is fire-and-forget on a daemon thread with a short timeout: a webhook
is a notification, and a slow or dead receiver must never block or fail the
registry operation it describes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.request
from typing import Any

from .db import Store

logger = logging.getLogger("elbi")

_URL_SETTING = "model_webhook_url"
_SECRET_SETTING = "model_webhook_secret"  # noqa: S105 - a setting key, not a secret

_TIMEOUT_SECONDS = 10


def webhook_config(store: Store | None) -> tuple[str, str]:
    """The configured webhook URL and secret ('' where unset)."""
    if store is None:
        return "", ""
    return (
        store.get_config(_URL_SETTING) or "",
        store.get_config(_SECRET_SETTING) or "",
    )


def fire_model_event(store: Store | None, action: str, data: dict[str, Any]) -> bool:
    """Send a registry event to the configured webhook; ``False`` when unset.

    ``action`` is the event name (``model_version.created``, ``alias.updated``,
    ``model_version.deleted``, ``registered_model.deleted``, ``drift.detected``,
    ``feature.drift_detected``, ``feature.expectations_failed``,
    ``metric.anomaly_detected``, ``metric.recovered``, ``run.failed``, ``run.slow``);
    ``data`` is the event's payload (model, feature-view, or monitor name, version,
    value, run id, and the like).
    """
    url, secret = webhook_config(store)
    if not url:
        return False
    body = json.dumps(
        {"action": action, "timestamp": int(time.time()), "data": data}
    ).encode()
    headers = {"Content-Type": "application/json"}
    if secret:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        headers["X-Elbi-Signature"] = f"sha256={digest}"

    def deliver() -> None:
        request = urllib.request.Request(  # noqa: S310 - operator-configured URL
            url, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=_TIMEOUT_SECONDS
            ):
                pass
        except Exception as exc:  # a notification must never break its operation
            logger.warning("model webhook delivery to %s failed: %s", url, exc)

    threading.Thread(target=deliver, name="model-webhook", daemon=True).start()
    return True
