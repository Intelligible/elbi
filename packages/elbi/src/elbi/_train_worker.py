"""Run one AutoML search in a child process and report the result as JSON.

An AutoML search holds the frame, every candidate it fits, and the SHAP explainer, so in
the web process it can exceed the pod's memory limit and take the whole service with it
rather than just the run -- the reason queries run out of process too (see
``_query_worker``). The parent pipes in the rows it loads anyway, so the search's memory
belongs to a process whose death is a failed job.

Reads one JSON request on stdin and writes one JSON reply on stdout, so a crash is an
exit code the parent can name rather than a traceback on a dead pipe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _reply(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    """Train once from the request on stdin."""
    # Imported here so a missing ml extra is a reply the parent can surface.
    try:
        from elbi_core.ml import train_automl
    except Exception as exc:
        _reply({"error": f"the training worker could not start: {type(exc).__name__}"})
        return 0

    try:
        request = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        _reply({"error": "the training worker received a malformed request"})
        return 0

    rows = request.get("rows") or []
    kwargs = dict(request.get("kwargs") or {})
    artifact_root = kwargs.pop("artifact_root", "")
    try:
        report = train_automl(rows, artifact_root=Path(artifact_root), **kwargs)
    except Exception as exc:
        # Including ModelError, which the parent re-raises, so a bad spec still reads
        # as the caller's error rather than as a crashed worker.
        _reply({"error": f"{type(exc).__name__}: {exc}"})
        return 0

    _reply(
        {
            "report": {
                "name": report.name,
                "version": report.version,
                "task": report.task,
                "target": report.target,
                "features": list(report.features),
                "best_estimator": report.best_estimator,
                "metrics": report.metrics,
                "oracle_verdict": report.oracle_verdict,
                "champion": report.champion,
                "rendered": report.render(),
            }
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
