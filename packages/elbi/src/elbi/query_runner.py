"""Where warehouse SQL runs: in the app, or on compute of its own.

Every SQL surface, Explore, metrics, dashboards, and a notebook cell's ``sql()``,
funnels through :meth:`WarehouseService.query`. Until now that meant DuckDB executing
inside the web server, which is the one arrangement its own community warns about: an
in-process engine and the request handlers compete for the same pages, so a query that
runs out of memory takes the whole service down rather than failing.

That is also the failure the compute plan opens by complaining about for notebooks ("a
notebook can take down the app") and the reference products answer it by separating
query work outright. A Databricks SQL warehouse is a distinct compute resource, sized
and scaled on its own, so completely separate that it cannot run Python at all; Hex
pushes warehouse SQL down rather than through the kernel.

So there are two runners:

``InProcessQueryRunner``
    Executes in the app. The laptop and single-machine tier, and honest about what it
    is: no isolation, and a big enough query ends the process.

``WorkerQueryRunner``
    A small pool of persistent child processes that own DuckDB. A query's memory belongs
    to a worker, an out-of-memory query kills that worker rather than the app, and the
    pool replaces it. The pool is what keeps concurrency: serialising every dashboard
    behind one process would trade an availability problem for a throughput one.

DuckDB 1.5.2 added a native client-server protocol, which is the eventual shape for a
query service addressed over the network: a separately scaled Deployment rather than a
child process. This uses the line protocol already proven by the notebook kernel,
because the property that matters first is the isolation boundary, not where the
boundary lives.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from typing import Any, Protocol

logger = logging.getLogger("elbi")

#: How long one query may take before the worker running it is assumed lost. Generous:
#: a legitimate scan over a large table is the reason this exists.
DEFAULT_QUERY_TIMEOUT = 300.0

#: Workers held in the pool. Small on purpose (each is a whole DuckDB) and the point of
#: more than one is that a dashboard with several panels does not queue behind itself.
DEFAULT_POOL_SIZE = 2

QueryResult = tuple[list[str], list[dict[str, Any]], bool]


class QueryError(RuntimeError):
    """A query failed. Carries the engine's message, not a traceback."""


class QueryRunner(Protocol):
    """Runs read-only SQL over the warehouse and returns columns, rows, truncated."""

    def run(self, sql: str, max_rows: int) -> QueryResult:
        """Execute ``sql``, returning at most ``max_rows`` rows."""
        ...

    def close(self) -> None:
        """Release whatever the runner holds."""
        ...


class InProcessQueryRunner:
    """Executes in the app's own process. Single-machine, and unisolated by design."""

    def __init__(self, execute: Any) -> None:
        self._execute = execute

    def run(self, sql: str, max_rows: int) -> QueryResult:
        """Execute in this process; an OOM here ends the process."""
        result: QueryResult = self._execute(sql, max_rows=max_rows)
        return result

    def close(self) -> None:
        """Nothing to release."""


class WorkerQueryRunner:
    """A pool of child processes that own DuckDB, so the app never holds a scan."""

    def __init__(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        timeout: float = DEFAULT_QUERY_TIMEOUT,
        python_executable: str = sys.executable,
        env: dict[str, str] | None = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError("a query worker pool needs at least one worker")
        self._timeout = timeout
        self._python = python_executable
        self._env = env
        self._closed = False
        # A slot per worker rather than a worker per slot: a worker is started when a
        # query first needs one, so a deployment that never queries never pays for them.
        self._slots: queue.Queue[_Worker] = queue.Queue()
        for _ in range(pool_size):
            self._slots.put(_Worker())

    def run(self, sql: str, max_rows: int) -> QueryResult:
        """Execute ``sql`` on a pooled worker, waiting for a free one.

        The worker binds its views from its own store, so what a query can resolve is
        decided where the views are made rather than by a list travelling with it.
        """
        if self._closed:
            raise QueryError("the query runner is closed")
        worker = self._slots.get()
        try:
            reply = worker.ask(
                {"sql": sql, "max_rows": max_rows},
                timeout=self._timeout,
                start=self._start,
            )
        finally:
            self._slots.put(worker)
        error = reply.get("error")
        if error:
            raise QueryError(str(error))
        return (
            list(reply.get("columns") or []),
            list(reply.get("rows") or []),
            bool(reply.get("truncated")),
        )

    def _start(self) -> subprocess.Popen[str]:
        """Launch one worker process."""
        argv = [self._python, "-m", "elbi._query_worker"]
        return subprocess.Popen(  # noqa: S603
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, **(self._env or {})},
        )

    def close(self) -> None:
        """Shut every worker down."""
        self._closed = True
        while True:
            try:
                self._slots.get_nowait().close()
            except queue.Empty:
                return


class _Worker:
    """One child process, started on first use and replaced when it dies."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None

    def ask(
        self, request: dict[str, Any], timeout: float, start: Any
    ) -> dict[str, Any]:
        """Send one request and read its reply, restarting a dead worker first.

        A worker that died (the usual reason being a query that exhausted its memory) is
        replaced rather than mourned, and the caller is told which of the two happened:
        a query error is theirs to fix, a lost worker is not.
        """
        for attempt in (1, 2):
            if self._proc is None or self._proc.poll() is not None:
                self._replace(start)
            try:
                return self._exchange(request, timeout)
            except _WorkerLost:
                self._discard()
                if attempt == 2:
                    raise QueryError(
                        "the query worker stopped before answering, twice. A query "
                        "that exhausts its memory ends its worker; the app survives, "
                        "and this one needs a smaller result or an aggregate."
                    ) from None
        raise AssertionError("unreachable")  # pragma: no cover

    def _replace(self, start: Any) -> None:
        self._discard()
        self._proc = start()

    def _discard(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        proc.kill()
        # A killed process reaps immediately in practice; the timeout is here so a
        # wedged
        # one cannot hold the caller rather than because it is expected to fire.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)

    def _exchange(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        """One request, one reply, on this worker's pipes."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise _WorkerLost
        try:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise _WorkerLost from exc

        # Read on a thread so a worker that dies mid-query cannot wedge the caller: a
        # blocking readline on a pipe whose writer is gone returns, but one on a process
        # that is stuck does not.
        reply: list[dict[str, Any] | None] = []

        def read() -> None:
            line = proc.stdout.readline() if proc.stdout else ""
            try:
                reply.append(json.loads(line) if line.strip() else None)
            except ValueError:
                reply.append(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(timeout)
        if reader.is_alive():
            raise QueryError(
                f"the query took longer than {timeout:.0f}s and was abandoned. Its "
                "worker is replaced; the app was never blocked."
            )
        if not reply or reply[0] is None:
            raise _WorkerLost
        return reply[0]

    def close(self) -> None:
        """Stop this worker."""
        self._discard()


class _WorkerLost(Exception):
    """The worker exited without answering; the caller decides whether to retry."""


def build_runner(execute: Any, backend: str | None = None) -> QueryRunner:
    """The query runner for this deployment.

    ``QUERY_RUNNER`` chooses: ``worker`` for a pool of child processes, ``inprocess`` to
    execute in the app. The default is in-process, because that is what a laptop wants
    and a worker pool on a single machine buys isolation nobody there needs, but any
    deployment serving more than one person should set ``worker``, since the alternative
    is a dashboard query that can end the web server.
    """
    chosen = (backend or os.environ.get("QUERY_RUNNER", "") or "inprocess").strip()
    if chosen == "inprocess":
        return InProcessQueryRunner(execute)
    if chosen != "worker":
        raise QueryError(
            f"QUERY_RUNNER={chosen!r} must be 'worker' (a pool of query processes) or "
            "'inprocess' (execute in the app)"
        )
    size = _int_env("QUERY_POOL_SIZE", DEFAULT_POOL_SIZE)
    logger.info("warehouse queries run on a pool of %d worker process(es)", size)
    # No profile knob here on purpose. A worker's memory is bounded by the container the
    # app runs in, not by anything this code can set -- the same honest limitation the
    # subprocess sandbox has -- and a field that validated and then did nothing would be
    # worse than its absence.
    return WorkerQueryRunner(
        pool_size=size, timeout=_float_env("QUERY_TIMEOUT", DEFAULT_QUERY_TIMEOUT)
    )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise QueryError(f"{name} must be a whole number, not {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise QueryError(f"{name} must be a number, not {raw!r}") from exc
