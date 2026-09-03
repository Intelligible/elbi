"""Running a derivation as a background job: it authors through the normal path, the
outcome is serialized as the job result, and an identical request dedupes by content."""

from __future__ import annotations

import threading

from elbi.derivation_jobs import (
    derivation_job_key,
    submit_derivation_job,
)
from elbi_agent import DeriveOutcome
from elbi_core import InMemoryJobStore, JobRunner


def _runner() -> JobRunner:
    return JobRunner(store=InMemoryJobStore(), max_workers=2)


def _submit(runner: JobRunner, derive, name: str = "m"):  # type: ignore[no-untyped-def]
    return submit_derivation_job(
        runner,
        derive,
        name=name,
        source="def m(ctx): return []",
        claim={"target": "price", "features": ["x"]},
        contract=None,
        fmt="table",
        assumptions=(),
        deps=["scikit-learn"],
        datasets=["houses"],
    )


def test_job_runs_the_derivation_and_serializes_the_outcome() -> None:
    def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
        return DeriveOutcome(
            certified=True, verdict="sound", rendered="ok", data_hash="h1"
        )

    runner = _runner()
    try:
        job = _submit(runner, derive)
        done = runner.wait(job.id, timeout=5)
        assert done is not None and done.state == "succeeded"
        assert done.result["certified"] is True
        assert done.result["verdict"] == "sound"
        assert done.result["data_hash"] == "h1"
    finally:
        runner.close()


def test_identical_request_dedupes_even_under_a_new_name() -> None:
    calls = 0
    lock = threading.Lock()

    def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
        nonlocal calls
        with lock:
            calls += 1
        return DeriveOutcome(certified=True, verdict="sound")

    runner = _runner()
    try:
        first = _submit(runner, derive, name="model_a")
        runner.wait(first.id, timeout=5)
        # Same source/claim/deps/datasets, different name: reuses the finished job.
        second = _submit(runner, derive, name="model_b")
        assert second.id == first.id
        assert calls == 1
    finally:
        runner.close()


def test_key_excludes_the_name() -> None:
    common = {"source": "s", "claim": None, "contract": None, "fmt": "table"}
    key_a = derivation_job_key("a", **common, deps=["numpy"], datasets=["d"])
    key_b = derivation_job_key("b", **common, deps=["numpy"], datasets=["d"])
    assert key_a == key_b
