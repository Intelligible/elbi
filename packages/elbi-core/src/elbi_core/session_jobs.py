"""Run a long ``run_code`` snippet as a durable background job.

The interactive session is single-threaded and shared across a turn, so a long snippet
cannot run there without blocking the model's other calls. A background run_code job
therefore runs in its OWN isolated session (a fresh :class:`SessionManager` on the same
backend), sharing only the workspace directory: files the snippet writes land there, so
a later inline call can read them. Its in-memory namespace is isolated by design; the
filesystem is the bridge, matching the persistent-workspace model. Unlike a derivation
this is not content-addressed (exploratory code can have side effects and randomness),
so each submission is its own job. Shared by the app and the MCP server, so both
surfaces launch long exploration the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from .jobs import Job, JobRunner
from .session import SessionManager


def submit_code_job(
    job_runner: JobRunner,
    *,
    code: str,
    deps: Sequence[str],
    datasets: dict[str, list[dict[str, str]]],
    scratch_dir: Path | None = None,
    backend: str = "subprocess",
    egress: str | Sequence[str] = "full",
    image: str | None = None,
) -> Job:
    """Submit a run_code snippet to run in its own background session; return the job.

    The job spins up a throwaway session on the configured backend, runs the snippet,
    and returns its stdout/result/error. Files it writes to ``scratch_dir`` remain for
    the conversation's later inline calls.
    """

    def work(progress: Any, cancelled: Any) -> dict[str, Any]:
        progress("running")
        manager = SessionManager(
            datasets=datasets,
            workspace=scratch_dir,
            backend=backend,
            egress=egress,
            image=image,
        )
        try:
            result = manager.run_code(code, tuple(deps))
        finally:
            manager.close()
        return {
            "stdout": result.stdout,
            "result": result.result,
            "error": result.error,
        }

    # A fresh key per submission: exploratory code is not a pure function of its inputs,
    # so it is never deduped onto an earlier run the way a derivation is.
    return job_runner.submit(f"run_code:{uuid4().hex}", "run_code", work)
