"""Operational settings resolved from the environment or the store, source named.

Two kinds of configuration exist in this app and they deliberately resolve in opposite
directions. Getting them confused is the usual cause of "I changed it and nothing
happened", so the rule is written down here and the effective source is always reported.

**Operational settings**: retention windows, background-task cadences, the tracking
URI. One value each, and a platform team may need to pin them across a fleet. The
environment wins; the stored value is what an admin edits when it says nothing:

    env  >  stored setting  >  built-in default

**Product configuration**: LLM profiles, dashboards, monitors. Named, plural, and part
of what the app is for. The store wins and the environment is only a seed for a fresh
install, because a single ``LLM_MODEL`` cannot express a set of named profiles at all.
That resolution lives with the feature (see ``serve.resolve_client``), not here.

Every resolved value carries where it came from, so a UI can show a setting as managed
by the environment rather than offering a field whose edits would not take effect. Open
WebUI's tracker is full of exactly that confusion; naming the source avoids it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, fine for typing
    from .db import Store

Source = Literal["env", "setting", "default"]


@dataclass(frozen=True)
class Setting:
    """A resolved operational setting and where its value came from."""

    key: str
    env_var: str
    value: str
    source: Source
    #: False when the environment pins it, so a UI can render it read-only.
    editable: bool


@dataclass(frozen=True)
class SettingSpec:
    """Definition of one operational setting."""

    key: str
    env_var: str
    default: str
    description: str


#: Every operational setting, in the order a settings page should show them. Adding one
#: here is all that is needed for it to be readable, editable, and env-overridable.
SPECS: tuple[SettingSpec, ...] = (
    SettingSpec(
        key="audit_retention_days",
        env_var="AUDIT_RETENTION_DAYS",
        default="0",
        description="Days of audit history to keep. 0 keeps everything.",
    ),
    SettingSpec(
        key="inference_retention_days",
        env_var="INFERENCE_RETENTION_DAYS",
        default="30",
        description="Days of model-inference history to keep. 0 keeps everything.",
    ),
    SettingSpec(
        key="notification_retention_days",
        env_var="NOTIFICATION_RETENTION_DAYS",
        default="90",
        description="Days of in-app notifications to keep. 0 keeps everything.",
    ),
    SettingSpec(
        key="trash_retention_days",
        env_var="TRASH_RETENTION_DAYS",
        default="30",
        description=(
            "Days a trashed artifact stays restorable before it is permanently "
            "erased. 0 disables the sweep and keeps trash forever."
        ),
    ),
    SettingSpec(
        key="maintenance_interval_seconds",
        env_var="MAINTENANCE_INTERVAL_SECONDS",
        default="3600",
        description="How often housekeeping runs. Floored at 60 seconds.",
    ),
    SettingSpec(
        key="drift_check_interval_hours",
        env_var="DRIFT_CHECK_INTERVAL_HOURS",
        default="24",
        description="How often metrics are checked for drift. Floored at 1 hour.",
    ),
    SettingSpec(
        key="feature_drift_interval_hours",
        env_var="FEATURE_DRIFT_INTERVAL_HOURS",
        default="24",
        description="How often feature views are checked for drift. Floored at 1 hour.",
    ),
    SettingSpec(
        key="mlflow_tracking_uri",
        env_var="MLFLOW_TRACKING_URI",
        default="",
        description="Where model runs are tracked. Empty uses the bundled store.",
    ),
)

_BY_KEY = {spec.key: spec for spec in SPECS}


def resolve(spec: SettingSpec, store: Store | None) -> Setting:
    """Resolve one setting: environment first, then the store, then the default."""
    from_env = os.environ.get(spec.env_var)
    if from_env not in (None, ""):
        return Setting(spec.key, spec.env_var, str(from_env), "env", editable=False)

    stored = store.get_config(spec.key) if store is not None else None
    if stored not in (None, ""):
        return Setting(spec.key, spec.env_var, str(stored), "setting", editable=True)

    return Setting(spec.key, spec.env_var, spec.default, "default", editable=True)


def get(key: str, store: Store | None) -> Setting:
    """Resolve the setting named ``key``. Raises ``KeyError`` for an unknown key."""
    return resolve(_BY_KEY[key], store)


def all_settings(store: Store | None) -> list[Setting]:
    """Every operational setting, resolved."""
    return [resolve(spec, store) for spec in SPECS]


def value_of(key: str, store: Store | None) -> str:
    """Just the effective value, for call sites that do not care about the source."""
    return get(key, store).value


def describe(key: str) -> str:
    """The human description of ``key``, for a settings page."""
    return _BY_KEY[key].description


def is_known(key: str) -> bool:
    """Whether ``key`` names an operational setting this module manages."""
    return key in _BY_KEY
