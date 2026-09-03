"""App-side wiring to export certified runs to a team's MLflow, when configured.

The export is opt-in: it runs only when a tracking URI is set, via the
``MLFLOW_TRACKING_URI`` environment variable or the ``mlflow_tracking_uri`` app setting
(the env var wins). It is fail-soft (see :mod:`elbi.tracking.mlflow`), so a
misconfigured or unreachable server never affects the certification that produced the
run; the run is already durably recorded in the app's own store regardless.
"""

from __future__ import annotations

import os

from elbi_core.tracking import CertifiedRun
from elbi_core.tracking.mlflow import emit_run

from .db import Store

_SETTING = "mlflow_tracking_uri"
_ENV = "MLFLOW_TRACKING_URI"


def tracking_uri(store: Store) -> str | None:
    """The configured MLflow tracking URI, env var over stored setting, or ``None``."""
    return os.environ.get(_ENV) or store.get_config(_SETTING)


def emit_configured_run(store: Store, run: CertifiedRun) -> bool:
    """Emit ``run`` to the configured MLflow server, or do nothing if none is set.

    Returns whether the run was sent. No configured URI is the common case (the run
    history and comparison work without MLflow), so it simply returns ``False``.
    """
    uri = tracking_uri(store)
    if not uri:
        return False
    return emit_run(run, tracking_uri=uri)
