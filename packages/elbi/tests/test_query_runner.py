"""Where warehouse SQL runs, and what happens when it goes wrong there.

DuckDB is an in-process engine, so a query executed in the web server competes with
request handling and an out-of-memory one ends the service rather than the query. These
tests are about the boundary that fixes it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from elbi.query_runner import (
    InProcessQueryRunner,
    QueryError,
    WorkerQueryRunner,
    build_runner,
)


def _execute(
    sql: str, *, max_rows: int
) -> tuple[list[str], list[dict[str, Any]], bool]:
    return (["n"], [{"n": 1}], False)


def test_the_default_runs_in_the_app_and_the_choice_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUERY_RUNNER", raising=False)
    # In-process is the laptop default: a worker pool on one machine buys isolation
    # nobody there needs.
    assert isinstance(build_runner(_execute), InProcessQueryRunner)

    monkeypatch.setenv("QUERY_RUNNER", "worker")
    runner = build_runner(_execute)
    try:
        assert isinstance(runner, WorkerQueryRunner)
    finally:
        runner.close()

    # A typo must not silently leave queries in the web server.
    monkeypatch.setenv("QUERY_RUNNER", "warehouse")
    with pytest.raises(QueryError, match="must be 'worker'"):
        build_runner(_execute)


def test_a_pool_needs_at_least_one_worker() -> None:
    with pytest.raises(ValueError, match="at least one worker"):
        WorkerQueryRunner(pool_size=0)


def test_a_worker_answers_a_real_query(tmp_path: Path) -> None:
    """End to end through a child process, against a real store.

    The worker rebuilds its own warehouse service from ``DB_URI``, so nothing crosses
    the boundary except the query and its result.
    """
    runner = WorkerQueryRunner(
        pool_size=1, env={"DB_URI": f"sqlite:{tmp_path / 'q.db'}"}, timeout=120
    )
    try:
        columns, rows, truncated = runner.run("select 41 + 1 as answer", max_rows=10)
        assert columns == ["answer"]
        assert rows == [{"answer": 42}]
        assert truncated is False

        # A second query reuses the same worker: the process is persistent, which is
        # what
        # makes this cheap enough to put in front of a dashboard.
        assert runner.run("select 1 as one", max_rows=10)[1] == [{"one": 1}]
    finally:
        runner.close()


def test_a_bad_query_is_the_users_problem_not_the_workers(tmp_path: Path) -> None:
    # A syntax error must come back as an error, and must not cost the worker that every
    # other request shares.
    runner = WorkerQueryRunner(
        pool_size=1, env={"DB_URI": f"sqlite:{tmp_path / 'q.db'}"}, timeout=120
    )
    try:
        with pytest.raises(QueryError):
            runner.run("select from where", max_rows=10)
        # Still alive.
        assert runner.run("select 7 as n", max_rows=10)[1] == [{"n": 7}]
    finally:
        runner.close()


def test_a_worker_that_dies_is_replaced_rather_than_mourned(tmp_path: Path) -> None:
    """A query that exhausts its memory ends its worker; the app survives.

    Simulated by killing the process between queries, which is what an OOM looks like
    from here: a pipe with no reader.
    """
    runner = WorkerQueryRunner(
        pool_size=1, env={"DB_URI": f"sqlite:{tmp_path / 'q.db'}"}, timeout=120
    )
    try:
        assert runner.run("select 1 as n", max_rows=10)[1] == [{"n": 1}]
        worker = runner._slots.get()
        assert worker._proc is not None
        worker._proc.kill()
        worker._proc.wait(timeout=10)
        runner._slots.put(worker)

        # The next query starts a fresh worker instead of failing.
        assert runner.run("select 2 as n", max_rows=10)[1] == [{"n": 2}]
    finally:
        runner.close()


def test_a_closed_runner_refuses_rather_than_starting_a_process(tmp_path: Path) -> None:
    runner = WorkerQueryRunner(
        pool_size=1, env={"DB_URI": f"sqlite:{tmp_path / 'q.db'}"}
    )
    runner.close()
    with pytest.raises(QueryError, match="closed"):
        runner.run("select 1", max_rows=1)


def test_the_pool_size_and_timeout_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUERY_RUNNER", "worker")
    monkeypatch.setenv("QUERY_POOL_SIZE", "4")
    runner = build_runner(_execute)
    try:
        assert runner._slots.qsize() == 4  # type: ignore[attr-defined]
    finally:
        runner.close()

    monkeypatch.setenv("QUERY_POOL_SIZE", "lots")
    with pytest.raises(QueryError, match="whole number"):
        build_runner(_execute)


def test_the_service_delegates_to_a_runner_and_reports_its_errors(
    tmp_path: Path,
) -> None:
    """The seam every SQL surface uses: Explore, metrics, dashboards, `sql()`."""
    from elbi.db import open_store
    from elbi.warehouse.service import WarehouseError, WarehouseService

    service = WarehouseService(open_store(f"sqlite:{tmp_path / 'w.db'}"))
    calls: list[str] = []

    class _Runner:
        def run(self, sql: str, max_rows: int) -> Any:
            calls.append(sql)
            if "boom" in sql:
                raise QueryError("Binder Error: no such column")
            return (["n"], [{"n": 5}], False)

        def close(self) -> None:
            pass

    service.set_runner(_Runner())
    assert service.query("select 5 as n")[1] == [{"n": 5}]
    assert calls == ["select 5 as n"]
    # A runner's error surfaces as the warehouse error the API already handles as a 400,
    # rather than as a new exception type every caller would have to learn.
    with pytest.raises(WarehouseError, match="Binder Error"):
        service.query("select boom")

    # Without a runner it executes here, which is the laptop path and what the worker
    # itself uses -- a worker delegating to another worker would recurse forever.
    service.set_runner(None)
    assert service.query("select 3 as n")[1] == [{"n": 3}]
    assert os.environ.get("QUERY_RUNNER") in (None, "", "inprocess", "worker")
