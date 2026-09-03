"""The run log: an append-only history of certified runs.

A run log records each distinct certified version of a derivation. Recording is
*version-collapsing*: appending a run whose ``(name, derivation_version)`` is already
present is a no-op, because an identical re-run is deterministic and produces the same
certified result. So the log holds one record per distinct certified state, and a
derivation's history is exactly its sequence of versions over time, newest first.

:class:`RunLog` is the seam; :class:`JsonlRunLog` is the local, single-user
implementation (one JSON object per line under the project directory). The app persists
the same :class:`~elbi.tracking.run.CertifiedRun` in its database instead, so
the record shape is shared across both surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from .run import CertifiedRun


@runtime_checkable
class RunLog(Protocol):
    """A durable, append-only store of certified runs."""

    def append(self, run: CertifiedRun) -> bool:
        """Record ``run``; return whether it was new (``False`` if it collapsed).

        A run whose ``(name, derivation_version)`` already exists is not recorded again,
        since a deterministic re-run yields the same certified result.
        """
        ...

    def runs(self, *, name: str | None = None) -> list[CertifiedRun]:
        """The recorded runs, newest first, optionally filtered to one derivation."""
        ...


class JsonlRunLog:
    """A :class:`RunLog` backed by a newline-delimited JSON file.

    Each :meth:`append` writes one JSON object; reads parse the whole file. This suits
    a local project's run history, tens to thousands of runs; the app keeps its own in a
    database.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, run: CertifiedRun) -> bool:
        """Append ``run`` unless its ``(name, version)`` is already logged."""
        if self._contains(run.name, run.derivation_version):
            return False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(run.to_dict(), separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return True

    def runs(self, *, name: str | None = None) -> list[CertifiedRun]:
        """Recorded runs, newest first, filtered to one derivation's name if given."""
        selected = [run for run in self._read() if name is None or run.name == name]
        selected.reverse()  # the file is oldest-first; present newest-first
        return selected

    def _read(self) -> list[CertifiedRun]:
        if not self._path.exists():
            return []
        out: list[CertifiedRun] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                out.append(CertifiedRun.from_dict(json.loads(stripped)))
        return out

    def _contains(self, name: str, version: str) -> bool:
        return any(
            run.name == name and run.derivation_version == version
            for run in self._read()
        )
