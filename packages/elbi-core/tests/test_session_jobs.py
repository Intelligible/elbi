"""Running a run_code snippet as a background job: it executes in its own session on a
worker and its stdout/result come back through the job."""

from __future__ import annotations

from elbi_core import InMemoryJobStore, JobRunner, submit_code_job


def test_submit_code_job_runs_and_returns_stdout() -> None:
    runner = JobRunner(store=InMemoryJobStore(), max_workers=2)
    try:
        job = submit_code_job(
            runner,
            code="print('hi from job'); result = 6 * 7",
            deps=(),
            datasets={},
        )
        done = runner.wait(job.id, timeout=15)
        assert done is not None and done.state == "succeeded"
        assert "hi from job" in (done.result["stdout"] or "")
    finally:
        runner.close()


def test_code_jobs_are_not_deduped() -> None:
    # Exploratory code is impure, so two identical submissions are two distinct jobs.
    runner = JobRunner(store=InMemoryJobStore(), max_workers=2)
    try:
        a = submit_code_job(runner, code="result = 1", deps=(), datasets={})
        b = submit_code_job(runner, code="result = 1", deps=(), datasets={})
        assert a.id != b.id
    finally:
        runner.close()
