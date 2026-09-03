"""Local lifecycle (certification) records.

A dependency-free sidecar at ``.elbi/lifecycle.yaml`` maps a derivation
name to its locally-recorded status. Discovered derivations keep their authored
status unless an entry here overrides it, so ``elbi certify`` can approve
an agent-proposed derivation without rewriting its source.

This is the *single-user* review record. Multi-party governance (who may certify,
signatures, audit trail) is the platform's job; this is its local analog, the same
open/closed seam as the cache store.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from elbi_core.errors import ConfigError

#: Filename of the lifecycle sidecar, under the project's ``.elbi/`` dir.
LIFECYCLE_FILENAME = "lifecycle.yaml"

_STATUSES = ("proposed", "certified")


class LifecycleStore:
    """Read and write locally-recorded derivation statuses."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def statuses(self) -> dict[str, str]:
        """Return the recorded ``name -> status`` map (empty if none)."""
        if not self._path.exists():
            return {}
        loaded = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{self._path}: top level must be a mapping")
        recorded: dict[str, str] = {}
        for name, status in loaded.items():
            if status not in _STATUSES:
                raise ConfigError(
                    f"{self._path}: status for {name!r} must be one of "
                    f"{list(_STATUSES)}"
                )
            recorded[str(name)] = status
        return recorded

    def set_status(self, name: str, status: str) -> None:
        """Record ``status`` for ``name``, creating the sidecar if needed."""
        if status not in _STATUSES:
            raise ConfigError(f"status must be one of {list(_STATUSES)}")
        recorded = self.statuses()
        recorded[name] = status
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            yaml.safe_dump(recorded, sort_keys=True), encoding="utf-8"
        )
