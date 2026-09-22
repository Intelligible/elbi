"""The notebook kernel: a persistent interpreter one notebook's cells run against.

A kernel wraps :mod:`elbi.notebook._kernel_worker` in a child process whose namespace
survives across cells, so a dataframe loaded in one cell is there for the next: the
subprocess model :class:`elbi.ReplSession` uses for exploration, extended with the
notebook protocol: a cell run streams its outputs back as they happen, a SIGINT
interrupts a running cell without discarding state, and a restart swaps in a fresh
interpreter. The kernel's environment (its declared dependencies) is fixed at start,
like a Jupyter kernelspec; changing dependencies restarts it.

Two backends implement the same :class:`Kernel` protocol. :class:`SubprocessKernel` runs
the worker as a host child with a scrubbed environment and network denied: the local
default, a bounded sandbox but a shared kernel, not a security wall. The Docker backend
runs it inside an isolated container. Neither is the certified path: a notebook is
authoring, and a verified result still comes only from a derivation.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib.util
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from ..errors import DerivationError
from ..executor import _DEP_RE, _child_env


def _json_default(obj: object) -> Any:
    """Encode a value the bound datasets hold that JSON has no native form for.

    Typed warehouse columns arrive as dates, timestamps, and decimals (and occasionally
    NumPy scalars); each becomes its natural JSON form (an ISO string or a plain
    number) so a cell receives clean values rather than the kernel failing to start.
    """
    if isinstance(obj, (dt.date, dt.datetime, dt.time)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    item = getattr(obj, "item", None)  # a NumPy scalar exposes its Python value here
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def encode_datasets(data: dict[str, Any] | None) -> str:
    """JSON-encode the bound datasets seeded into a kernel, coercing typed values."""
    return json.dumps(data or {}, default=_json_default)


#: Grace period after an interrupt for the worker to emit its final messages before the
#: kernel force-kills a wedged child.
_INTERRUPT_GRACE = 5.0

#: How long a completion/inspection request waits for the worker's reply before it gives
#: up and returns an empty result.
_REQUEST_TIMEOUT = 5.0

#: While a cell is parked in ``input()`` the wall-clock budget is suspended (a person is
#: typing); the deadline is pushed out by this much each time input is requested.
_INPUT_GRACE = 3600.0

#: A callback invoked once per streamed output message (stream / display_data /
#: execute_result / error) during a cell run, so a caller can forward it live.
OutputSink = Callable[[dict[str, Any]], None]

#: A cell waiting on a server-side query is working, not wedged, so the wall-clock
#: budget is extended by this much while one is in flight. Generous, because the point
#: is that a scan happens where the data is and may legitimately take a while.
_QUERY_GRACE = 900.0

#: Bytes of output one cell may produce before the rest is dropped. A cell's outputs are
#: held in memory, persisted as JSON and shipped to a browser, so an unbounded ``print``
#: loop costs the host and the client, not the sandbox that wrote it. 10 MB matches the
#: limit Databricks publishes and defaults to for the same reason.  Enforced host-side
#: rather than in the worker: the worker is the thing running the user's code, so a
#: budget it keeps for itself is advisory. Truncating rather than failing the cell is
#: deliberate: the run's result is usually still what was wanted, and the wall-clock
#: timeout is what stops a genuine runaway.
MAX_OUTPUT_BYTES = 10 * 1024 * 1024

#: A callback invoked for each ipywidgets comm message the kernel emits, so the app's
#: comm bridge can relay it to the browser's widget manager.
CommListener = Callable[[dict[str, Any]], None]

#: Answers a cell's ``sql(...)`` or ``data['x']``. Called with ``(sql, table, limit)``,
#: exactly one of ``sql``/``table`` non-empty, and returns the reply the worker unpacks:
#: ``{"columns": [...], "rows": [...], "truncated": bool}`` or ``{"error": "..."}``.
#:
#: ``table`` is a name, never SQL, so the app resolves it against the tables it has
#: registered. A dataset name is user-controlled, and interpolating one into a query in
#: the kernel is the class of bug behind the Apache Polaris path-scoping CVEs.
QueryResolver = Callable[[str, str, int], dict[str, Any]]


class Kernel(Protocol):
    """A persistent interpreter for a notebook: run, interrupt, restart, close."""

    @property
    def alive(self) -> bool:
        """Whether the kernel process is still running."""
        ...

    #: How the last cell reached its data: ``"pushed_down"`` (the work ran in the
    #: warehouse and a bounded result came back), ``"materialised"`` (rows crossed
    #: into the kernel), or empty (the cell touched no data). An attribute rather than
    #: part of ``execute``'s return, because it is one optional fact and widening that
    #: signature would ripple through every backend for it.
    last_data_mode: str

    #: The queries the last cell pushed down: statement, duration, rows, whether the
    #: result was truncated. For a product whose scale story *is* pushdown, showing the
    #: SQL and what it cost is the difference between "this is fast" and a claim. The
    #: fields mirror what Databricks' query history records (statement text, duration,
    #: rows produced) minus the ones a single-node engine cannot honestly report.
    last_queries: list[dict[str, Any]]

    def execute(self, code: str, execution_count: int, on_output: OutputSink) -> str:
        """Run one cell, forwarding each output to ``on_output``; return final status.

        Status is ``"ok"``, ``"error"`` (the cell raised), ``"interrupted"`` (a SIGINT
        or
        timeout stopped it), or ``"dead"`` (the process was lost).
        """
        ...

    def interrupt(self) -> None:
        """Signal the running cell to stop, keeping the namespace intact."""
        ...

    def send_input(self, value: str) -> None:
        """Deliver a reply to a cell's pending ``input()``."""
        ...

    def set_query_resolver(self, resolver: QueryResolver | None) -> None:
        """Set the callable that answers a cell's ``sql(...)`` / ``data['x']``."""
        ...

    def complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        """Completions for ``code`` at ``cursor_pos`` (a ``complete_reply`` content)."""
        ...

    def inspect(
        self, code: str, cursor_pos: int, detail_level: int = 0
    ) -> dict[str, Any]:
        """Inspect the name at ``cursor_pos`` (Jupyter ``inspect_reply`` content)."""
        ...

    def variables(self) -> dict[str, Any]:
        """The user's data variables in the namespace (a ``variables_reply``)."""
        ...

    def set_comm_listener(self, listener: CommListener | None) -> None:
        """Route the worker's ipywidgets comm messages to ``listener``."""
        ...

    def send_comm(self, message: dict[str, Any]) -> None:
        """Forward a frontend comm message to the worker."""
        ...

    def close(self) -> None:
        """Shut the kernel down and release its process and temp files."""
        ...


def worker_path() -> str:
    """Filesystem path of the kernel worker, to run it by path in a uv environment."""
    spec = importlib.util.find_spec("elbi_core.notebook._kernel_worker")
    if spec is None or spec.origin is None:  # pragma: no cover - always importable
        raise DerivationError("notebook kernel worker module not found")
    return spec.origin


def _payload_size(message: Mapping[str, Any]) -> int:
    """Roughly how many bytes an output message costs to hold, store and ship.

    Sums the string payloads (a stream's ``text``, a display bundle's base64 image, a
    traceback's frames), which is where all of the size in an output message is. Walking
    the values is cheaper than re-serialising each message only to measure it, and the
    budget it feeds only needs to be right to within a message.
    """
    total = 0
    stack: list[Any] = [message]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            total += len(value)
        elif isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return total


def _truncation_notice(budget: int) -> dict[str, Any]:
    """The one message a cell gets when its output stops being forwarded."""
    size = (
        f"{budget / (1024 * 1024):.0f} MB" if budget >= 1024 * 1024 else f"{budget} B"
    )
    return {
        "type": "stream",
        "name": "stderr",
        "text": (
            f"\nOutput truncated at {size}. The cell is still running, and its result "
            "will still appear; the printing in between is discarded. Write large "
            "results to the warehouse, or print a summary.\n"
        ),
    }


class _StreamingKernel:
    """Shared protocol handling for a worker-backed kernel (host or container).

    A subclass launches the worker process (as ``self._proc``, an ``-i`` child speaking
    the notebook protocol on its stdin/stdout) and implements :attr:`alive`,
    :meth:`interrupt`, and :meth:`close`; this base owns the mechanics common to both
    backends: a reader thread draining stdout onto a queue, and a pump that streams a
    cell's outputs until ``done`` while enforcing the wall-clock timeout.
    """

    _proc: subprocess.Popen[str]
    _cell_timeout: float
    _alive: bool
    _max_output_bytes: int = MAX_OUTPUT_BYTES
    #: Set from each cell's ``done``; see :attr:`Kernel.last_data_mode`.
    last_data_mode: str = ""
    #: What the last cell pushed to the warehouse, one entry per query. See
    #: :attr:`Kernel.last_queries`.
    last_queries: list[dict[str, Any]]

    def _start_reader(self) -> None:
        """Start the background thread that feeds worker messages onto the queue.

        A cell can emit many messages at once, and a plain ``select`` on the pipe would
        race with the stream's own buffering (a batched write drains into the reader's
        buffer, leaving ``select`` reporting nothing while lines wait). A blocking
        readline on its own thread, feeding a queue the pump drains with a timeout,
        sidesteps that.
        """
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        # Serialize consumers of the message queue (a cell run and a completion/inspect
        # request never drain it at once) and writers to the worker's stdin.
        self._exec_lock = threading.Lock()
        self._stdin_lock = threading.Lock()
        # A sink for ipywidgets comm traffic (set by the app's comm bridge); comm
        # messages route here, not the run queue, so they flow with or without a run.
        self._comm_listener: CommListener | None = None
        # Answers a cell's `sql(...)`/`data['x']`. The kernel cannot query the warehouse
        # itself (scrubbed environment, blocked network), so the host runs it where the
        # engine and credentials already are. None when no warehouse is wired, which
        # answers with a clear error rather than hanging.
        self._query_resolver: QueryResolver | None = None
        self.last_queries = []
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _write(self, payload: dict[str, Any]) -> bool:
        """Write one JSON request line to the worker's stdin; return success.

        Encoded with ``_json_default`` for the same reason the seeded datasets are: a
        query reply carries whatever the warehouse column held, and a date, timestamp
        or decimal has no native JSON form. A ``TypeError`` here is a value the encoder
        does not cover, and it must not escape -- the worker is blocked reading this
        line, so a raised error would leave the cell waiting for a reply that never
        comes, until the cell deadline kills it with no indication of why.
        """
        stdin = self._proc.stdin
        if stdin is None:
            return False
        try:
            line = json.dumps(payload, default=_json_default) + "\n"
        except TypeError:
            return False
        with self._stdin_lock:
            try:
                stdin.write(line)
                stdin.flush()
            except (BrokenPipeError, ValueError):
                self._alive = False
                return False
        return True

    def set_query_resolver(self, resolver: QueryResolver | None) -> None:
        """Set the callable that answers a cell's warehouse queries."""
        self._query_resolver = resolver

    def _answer_query(self, request: dict[str, Any]) -> None:
        """Run a cell's query through the resolver and write the reply to the worker.

        Every failure path returns a reply rather than raising: a cell blocked in
        ``sql()`` reads until it gets one, so a resolver that throws must still produce
        an answer, or the cell hangs until the deadline kills it.
        """
        resolver = self._query_resolver
        sql = str(request.get("sql") or "")
        table = str(request.get("table") or "")
        limit = int(request.get("limit") or 0)
        started = time.monotonic()
        if resolver is None:
            content: dict[str, Any] = {
                "error": (
                    "No warehouse is available to this notebook, so sql() and "
                    "data[...] cannot run: this kernel has no query resolver."
                )
            }
        else:
            try:
                content = resolver(sql, table, limit)
            # Reported to the cell rather than raised: see the docstring above.
            except Exception as exc:
                content = {"error": f"{type(exc).__name__}: {exc}"}
        # Recorded whether it succeeded or not: a query that failed after twelve seconds
        # is exactly the one someone needs to see.
        self.last_queries.append(
            {
                "sql": sql or f"read table {table}",
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
                "rows": len(content.get("rows") or []),
                "truncated": bool(content.get("truncated")),
                "error": content.get("error"),
            }
        )
        if not self._write({"op": "query_reply", "content": content}):
            # The rows would not encode. Answer with the failure rather than nothing,
            # because the cell is blocked until some reply arrives.
            self._write(
                {
                    "op": "query_reply",
                    "content": {
                        "error": "the result holds a value that cannot be sent"
                    },
                }
            )

    def _death_reason(self) -> str:
        """Why the worker stopped, if the backend can find out. Empty when it cannot.

        A host subprocess leaves nothing to read after it exits; a pod has logs. The
        base returns nothing, so a backend opts in.
        """
        return ""

    def _read_loop(self) -> None:
        """Feed worker messages onto the queue; a ``None`` sentinel marks EOF.

        Comm messages (ipywidgets traffic) are routed to the comm listener rather than
        the run queue, so a widget interacts whether or not a cell is running.
        """
        stdout = self._proc.stdout
        if stdout is not None:
            try:
                for line in stdout:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        message = json.loads(stripped)
                    except ValueError:
                        continue
                    if message.get("type") in ("comm_open", "comm_msg", "comm_close"):
                        listener = self._comm_listener
                        if listener is not None:
                            listener(message)
                        continue
                    self._messages.put(message)
            except ValueError:
                # ``close()`` closes the pipe without joining this thread, so a read
                # in flight at shutdown raises here.
                if not stdout.closed:
                    raise
        self._messages.put(None)

    def set_comm_listener(self, listener: CommListener | None) -> None:
        """Route the worker's ipywidgets comm messages to ``listener`` (None: off)."""
        self._comm_listener = listener

    def send_comm(self, message: dict[str, Any]) -> None:
        """Forward a frontend comm message (``op``/``content``/``buffers``) in."""
        self._write(message)

    def execute(self, code: str, execution_count: int, on_output: OutputSink) -> str:
        """Run one cell, streaming outputs to ``on_output``; interrupt on timeout."""
        if not self.alive or self._proc.stdin is None:
            self._alive = False
            return "dead"
        with self._exec_lock:
            if not self._write({"code": code, "execution_count": execution_count}):
                return "dead"
            return self._pump(on_output)

    def send_input(self, value: str) -> None:
        """Deliver a reply to a cell's pending ``input()``, via the worker's stdin.

        Sent while a run holds the exec lock and is parked in ``input()``; it writes
        only to stdin, so it never contends with the pump draining the message queue.
        """
        self._write({"op": "input_reply", "value": value})

    def complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        """Completions for ``code`` at ``cursor_pos`` (a ``complete_reply`` content)."""
        empty = {"matches": [], "cursor_start": cursor_pos, "cursor_end": cursor_pos}
        return self._request(
            {"op": "complete", "code": code, "cursor_pos": cursor_pos},
            "complete_reply",
            empty,
        )

    def inspect(
        self, code: str, cursor_pos: int, detail_level: int = 0
    ) -> dict[str, Any]:
        """Inspect the name at ``cursor_pos`` (Jupyter ``inspect_reply`` content)."""
        return self._request(
            {
                "op": "inspect",
                "code": code,
                "cursor_pos": cursor_pos,
                "detail_level": detail_level,
            },
            "inspect_reply",
            {"found": False, "data": {}, "metadata": {}},
        )

    def variables(self) -> dict[str, Any]:
        """The user's data variables in the namespace (a ``variables_reply``)."""
        return self._request({"op": "variables"}, "variables_reply", {"variables": []})

    def _request(
        self, payload: dict[str, Any], reply_type: str, empty: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a request op to an idle worker and return its reply's content.

        Holds the exec lock so it cannot interleave with a cell run's message stream;
        returns ``empty`` if the kernel is dead or does not reply within the grace time.
        """
        if not self.alive:
            return empty
        with self._exec_lock:
            if not self._write(payload):
                return empty
            deadline = time.monotonic() + _REQUEST_TIMEOUT
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return empty
                try:
                    message = self._messages.get(timeout=remaining)
                except queue.Empty:
                    return empty
                if message is None:
                    self._alive = False
                    return empty
                if message.get("type") == reply_type:
                    return {k: v for k, v in message.items() if k != "type"}

    def _pump(self, on_output: OutputSink) -> str:
        """Drain streamed messages until ``done``, interrupting if a cell overruns.

        The budget is wall-clock: once ``cell_timeout`` elapses the cell is interrupted,
        then a grace period lets the worker emit its interrupted result and ``done``
        before a still-wedged child is force-killed. A cell that keeps producing output
        within the budget is not stuck, so the deadline bounds total time, not gaps.

        Output is bounded too, by :data:`MAX_OUTPUT_BYTES`. Past it the cell keeps
        running and the bulk of its output is dropped, having said once that it was: a
        cell that prints in a loop should not be able to exhaust the host it reports to.
        """
        deadline = time.monotonic() + self._cell_timeout
        interrupted = False
        spent = 0
        truncated = False
        self.last_queries = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not interrupted:
                    self.interrupt()
                    interrupted = True
                    deadline = time.monotonic() + _INTERRUPT_GRACE
                    continue
                self.close(kill=True)
                return "interrupted"
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                continue  # deadline reached; the top of the loop handles the interrupt
            if message is None:  # EOF sentinel: the worker exited mid-cell
                self._alive = False
                reason = self._death_reason()
                if reason:
                    # Delivered as cell output, because "dead" on its own sends someone
                    # to the logs of a pod that has already been deleted.
                    on_output(
                        {
                            "type": "error",
                            "ename": "KernelDied",
                            "evalue": "the kernel stopped before finishing this cell",
                            "traceback": reason.splitlines(),
                        }
                    )
                return "dead"
            if message.get("type") == "done":
                self.last_data_mode = str(message.get("data_mode", ""))
                if interrupted:
                    return "interrupted"
                return str(message.get("status", "ok"))
            if message.get("type") == "input_request" and not interrupted:
                # A cell parked in input() is not wedged; extend the budget so a person
                # has time to answer instead of being cut off by the wall-clock timeout.
                deadline = time.monotonic() + _INPUT_GRACE
            if message.get("type") == "query_request":
                # A cell waiting on a server-side scan is working, not wedged. Extend
                # the budget so a legitimately slow aggregate is not killed by the cell
                # deadline, then answer it. Not forwarded to on_output: this is protocol
                # traffic, not cell output.
                deadline = max(deadline, time.monotonic() + _QUERY_GRACE)
                self._answer_query(message)
                continue
            if truncated:
                # Past the budget, a cell's result and its traceback are still worth
                # more than the printing that crowded them out, so those two survive if
                # they fit in a second budget's worth. Everything bulky does not.
                kind = message.get("type")
                if kind in ("execute_result", "error") and (
                    _payload_size(message) <= self._max_output_bytes
                ):
                    on_output(message)
                continue
            spent += _payload_size(message)
            if spent > self._max_output_bytes:
                truncated = True
                on_output(_truncation_notice(self._max_output_bytes))
                continue
            on_output(message)

    @property
    def alive(self) -> bool:  # pragma: no cover - overridden by every backend
        """Whether the kernel process is still running."""
        raise NotImplementedError

    def interrupt(self) -> None:  # pragma: no cover - overridden by every backend
        """Signal the running cell to stop, keeping the namespace intact."""
        raise NotImplementedError

    def close(self, *, kill: bool = False) -> None:  # pragma: no cover - overridden
        """Shut the kernel down and release its resources."""
        raise NotImplementedError


class SubprocessKernel(_StreamingKernel):
    """A notebook kernel backed by a host subprocess (the local default).

    The child runs in its own process session (``start_new_session``) so a SIGINT can be
    delivered to it alone, interrupting a wedged cell, without touching the app. Its
    environment is scrubbed (host credentials withheld) and, unless ``allow_network``,
    outbound sockets are blocked. This is a bounded local sandbox, not isolation against
    hostile code, which is the Docker backend's job.
    """

    def __init__(
        self,
        *,
        deps: Sequence[str] = (),
        workspace: Path | None = None,
        data: dict[str, list[dict[str, Any]]] | None = None,
        dataset_names: Sequence[str] | None = None,
        cell_timeout: float = 120.0,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        python_executable: str = sys.executable,
        allow_network: bool = False,
        env_passthrough: tuple[str, ...] = (),
        env: Mapping[str, str] | None = None,
    ) -> None:
        if cell_timeout <= 0:
            raise ValueError("cell timeout must be positive")
        for dep in deps:
            if not _DEP_RE.match(dep):
                raise DerivationError(
                    f"refusing to provision invalid dependency {dep!r}"
                )
        self._cell_timeout = cell_timeout
        self._max_output_bytes = max_output_bytes
        self._alive = True
        self._io = tempfile.TemporaryDirectory(prefix="elbi-kernel-")
        # Two shapes, and only one is written. With `dataset_names` the kernel fetches
        # rows on demand and nothing is serialised up front; without it, every dataset
        # is written eagerly as before. See notebook/_data.py.
        data_path = Path(self._io.name) / "data.json"
        names_path: str | None = None
        if dataset_names is not None:
            names_path = str(Path(self._io.name) / "names.json")
            Path(names_path).write_text(
                json.dumps(list(dataset_names)), encoding="utf-8"
            )
            data_path.write_text("{}", encoding="utf-8")
        else:
            data_path.write_text(encode_datasets(data), encoding="utf-8")
        argv = self._argv(
            deps, str(data_path), names_path, allow_network, python_executable
        )
        child_env = _child_env(env_passthrough)
        # Force matplotlib's headless backend so a plotting cell captures a PNG rather
        # than trying to open a GUI window; harmless when matplotlib is absent.
        child_env["MPLBACKEND"] = "Agg"
        # Values, unlike `env_passthrough`, which admits a host variable by name.
        child_env.update(env or {})
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=str(workspace) if workspace is not None else self._io.name,
                env=child_env,
                text=True,
                start_new_session=os.name != "nt",
            )
        except FileNotFoundError as exc:
            self._alive = False
            raise DerivationError(f"could not start notebook kernel: {exc}") from exc
        self._start_reader()

    def _argv(
        self,
        deps: Sequence[str],
        data_path: str,
        names_path: str | None,
        allow_network: bool,
        python: str,
    ) -> list[str]:
        flags = ["-I", worker_path(), data_path]
        if names_path is not None:
            flags.append(names_path)
        if allow_network:
            flags.append("--allow-network")
        if not deps:
            return [python, *flags]
        with_args = [arg for dep in deps for arg in ("--with", dep)]
        return [
            "uv",
            "run",
            "--no-project",
            "--python",
            python,
            *with_args,
            "--",
            "python",
            *flags,
        ]

    @property
    def alive(self) -> bool:
        """Whether the kernel process is still running."""
        return self._alive and self._proc.poll() is None

    def interrupt(self) -> None:
        """Send SIGINT to the kernel's process group (KeyboardInterrupt in it)."""
        if not self.alive:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            if os.name == "nt":  # pragma: no cover - POSIX is the primary target
                self._proc.send_signal(signal.SIGINT)
            else:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)

    def close(self, *, kill: bool = False) -> None:
        """Shut the kernel down and release its process and temp files."""
        if self._alive and not kill and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired, ValueError):
                self._proc.kill()
        else:
            self._proc.kill()
        self._alive = False
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=5)
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
        self._io.cleanup()
