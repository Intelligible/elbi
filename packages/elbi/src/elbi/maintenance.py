"""A small in-process scheduler for periodic housekeeping.

One background daemon thread runs a set of idempotent, cutoff-based cleanups on an
interval: pruning old serving traffic, notifications and (optionally) audit rows. No
external scheduler, task queue, or extra dependency: the minimal faithful version
of a maintenance cron for a single deployment. Every task is safe to re-run, so a missed
or doubled tick is harmless; horizontal scaling (multiple workers) is a platform concern
and would move this behind a shared lock.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence

from .db import Store
from .runtime_settings import value_of

logger = logging.getLogger(__name__)


# Each of these is an operational setting: the environment pins it, otherwise an admin's
# stored value applies, otherwise the built-in default. See runtime_settings for why
# that direction. The floors below are enforced here rather than there: a bad stored
# value should degrade to something sane, not stop housekeeping.
def _interval_seconds(store: Store | None = None) -> float:
    """The housekeeping cadence; at least a minute, default hourly."""
    try:
        return max(60.0, float(value_of("maintenance_interval_seconds", store)))
    except ValueError:
        return 3600.0


def _inference_retention_days(store: Store | None = None) -> int:
    """Days of serving-traffic history to keep (default 30; 0 keeps everything)."""
    try:
        return max(0, int(value_of("inference_retention_days", store)))
    except ValueError:
        return 30


def _audit_retention_days(store: Store | None = None) -> int:
    """Days of audit history to keep (0 = keep everything; retention disabled)."""
    try:
        return max(0, int(value_of("audit_retention_days", store)))
    except ValueError:
        return 0


def _notification_retention_days(store: Store | None = None) -> int:
    """Days of notifications to keep (default 90; 0 keeps everything).

    Unlike audit (a compliance trail, kept forever by default), notifications are
    transient operator signals: an unread "run failed" from months ago has no
    remaining value, so a bounded default keeps the inbox and the table small.
    """
    try:
        return max(0, int(value_of("notification_retention_days", store)))
    except ValueError:
        return 90


def _trash_retention_days(store: Store | None = None) -> int:
    """Days a trashed artifact stays restorable (default 30; 0 keeps trash forever)."""
    try:
        return max(0, int(value_of("trash_retention_days", store)))
    except ValueError:
        return 30


def run_once(store: Store) -> dict[str, int]:
    """Run one housekeeping pass and return per-task counts; never raises.

    Each task is isolated so one failure does not stop the others; all idempotent,
    so re-running is safe.
    """
    result: dict[str, int] = {}
    try:
        result["audit_pruned"] = store.prune_audit(_audit_retention_days(store))
    except Exception:
        logger.exception("maintenance: prune_audit failed")
    try:
        result["inference_pruned"] = store.prune_inference(
            _inference_retention_days(store)
        )
    except Exception:
        logger.exception("maintenance: prune_inference failed")
    try:
        result["notifications_pruned"] = store.prune_notifications(
            _notification_retention_days(store)
        )
    except Exception:
        logger.exception("maintenance: prune_notifications failed")
    try:
        result["trash_purged"] = store.purge_trash(_trash_retention_days(store))
    except Exception:
        logger.exception("maintenance: purge_trash failed")
    for name, task in store.housekeeping().items():
        try:
            result[name] = task()
        except Exception:
            logger.exception("maintenance: %s failed", name)
    return result


class Scheduler:
    """A daemon thread that runs :func:`run_once` on an interval until stopped.

    ``extra_tasks`` are app-provided periodic jobs (retraining policies, drift
    checks) run on the same tick, each isolated like the built-in tasks.
    """

    def __init__(
        self,
        store: Store,
        interval: float | None = None,
        extra_tasks: Sequence[Callable[[], None]] = (),
    ) -> None:
        self._store = store
        self._interval = interval if interval is not None else _interval_seconds(store)
        self._extra_tasks = tuple(extra_tasks)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background loop (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="maintenance", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to exit and join it briefly."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        # Run once at startup, then every interval; wait() returns at once once stopped
        # so shutdown is prompt.
        while not self._stop.is_set():
            counts = run_once(self._store)
            if any(counts.values()):
                logger.info("maintenance pass: %s", counts)
            for task in self._extra_tasks:
                try:
                    task()
                except Exception:
                    logger.exception("maintenance: extra task failed")
            self._stop.wait(self._interval)
