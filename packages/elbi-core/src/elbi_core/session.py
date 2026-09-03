"""A stateful exploration session: a persistent REPL the agent iterates against.

:class:`ReplSession` drives :mod:`elbi._repl_worker`, a long-lived child whose
namespace survives across calls, so a variable, import, or loaded dataset from one
``run_code`` is there for the next. This is the notebook-like loop the research on data
agents favors (stateful execution makes iteration cheaper and cuts undefined-name
errors), and it composes with the persistent workspace: the child's working directory is
that workspace, so both in-memory objects and files on disk carry forward.

It is exploration only. A verified answer never comes from a session; it comes from a
derivation, which runs one-shot and hermetic. The session's environment (its declared
dependencies) is fixed when it starts, like a kernel: to change dependencies, restart
the session, which is a clean reset of its state.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import select
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .errors import DerivationError
from .executor import _DEP_RE, CodeResult, _child_env


class Session(Protocol):
    """A stateful exploration session: run code and shell commands, then close.

    Implemented by :class:`ReplSession` (a host subprocess) and, in
    :mod:`elbi.container`, a Docker-backed session where ``run_code`` and
    ``run_bash`` share one isolated container. ``run_bash`` is only meaningful where the
    session is isolated; the host session declines it.
    """

    @property
    def alive(self) -> bool:
        """Whether the session can still run code (False after a timeout or exit)."""
        ...

    def run_code(self, code: str) -> CodeResult:
        """Run a Python snippet in the persistent namespace; return its output."""
        ...

    def run_bash(self, command: str) -> CodeResult:
        """Run a shell command in the session's environment; return its output."""
        ...

    def add_deps(self, deps: Sequence[str]) -> bool:
        """Install ``deps`` into the live session without losing state, if possible.

        Returns True when the running session now has them, so the manager keeps the
        namespace; False when it cannot (a fixed environment), so the manager restarts.
        """
        ...

    def close(self) -> None:
        """Shut the session down and release its resources."""
        ...


class _SessionTimeout(Exception):
    """A request outran the session timeout; the caller kills the session."""


class _SessionExited(Exception):
    """The worker exited without answering; the session is dead."""


def _exchange(
    proc: subprocess.Popen[str], request: dict[str, Any], timeout: float
) -> CodeResult:
    """Send one JSON request over the worker's pipes and read its JSON response.

    Raises :class:`_SessionTimeout` if no response arrives within ``timeout`` and
    :class:`_SessionExited` if the worker's pipes are gone. Shared by the host and
    container sessions, which differ only in how the worker process is launched.
    """
    if proc.stdin is None or proc.stdout is None:
        raise _SessionExited
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, ValueError) as exc:
        raise _SessionExited from exc
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise _SessionTimeout
    line = proc.stdout.readline()
    if not line:
        raise _SessionExited
    payload = json.loads(line)
    return CodeResult(
        stdout=payload.get("stdout", ""),
        result=payload.get("result"),
        error=payload.get("error"),
    )


def _worker_path() -> str:
    """Filesystem path of the REPL worker, to run it by path in a uv environment."""
    spec = importlib.util.find_spec("elbi_core._repl_worker")
    if spec is None or spec.origin is None:  # pragma: no cover - always importable
        raise DerivationError("repl worker module not found")
    return spec.origin


class ReplSession:
    """A persistent Python process whose namespace and workspace survive across calls.

    Start one per conversation, call :meth:`run` for each exploration snippet, and
    :meth:`close` when done. Requests and responses are newline-delimited JSON over the
    child's stdin/stdout; the child captures the user code's own stdout, so the protocol
    stream stays clean. A call that exceeds ``timeout`` kills the child and closes the
    session rather than blocking, since a persistent process cannot be interrupted.
    """

    def __init__(
        self,
        *,
        deps: Sequence[str] = (),
        workspace: Path | None = None,
        data: dict[str, list[dict[str, Any]]] | None = None,
        timeout: float = 300.0,
        python_executable: str = sys.executable,
        allow_network: bool = False,
        env_passthrough: tuple[str, ...] = (),
    ) -> None:
        if timeout <= 0:
            raise ValueError("session timeout must be positive")
        for dep in deps:
            if not _DEP_RE.match(dep):
                raise DerivationError(
                    f"refusing to provision invalid dependency {dep!r}"
                )
        self._timeout = timeout
        self._alive = True
        # The datasets seed the child's namespace as ``data``; a private temp file hands
        # them over at startup and lives for the session (the child reads it once).
        # default=str renders non-JSON scalars (dates, Decimals) as strings,
        # matching the string-valued cells the child already casts from.
        self._io = tempfile.TemporaryDirectory(prefix="elbi-session-")
        data_path = Path(self._io.name) / "data.json"
        data_path.write_text(json.dumps(data or {}, default=str), encoding="utf-8")
        argv = self._argv(deps, str(data_path), allow_network, python_executable)
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=str(workspace) if workspace is not None else self._io.name,
                env=_child_env(env_passthrough),
                text=True,
            )
        except FileNotFoundError as exc:
            self._alive = False
            raise DerivationError(
                f"could not start exploration session: {exc}"
            ) from exc

    def _argv(
        self, deps: Sequence[str], data_path: str, allow_network: bool, python: str
    ) -> list[str]:
        flags = ["-I", _worker_path(), data_path]
        if allow_network:
            flags.append("--allow-network")
        if not deps:
            return [python, *flags]
        # With deps, uv builds the environment; the interpreter after ``--`` must be the
        # bare ``python`` it puts on PATH (the provisioned env), not the host path, else
        # the ``--with`` packages are bypassed.
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
        """Whether the child still runs (False once a timeout or exit closed it)."""
        return self._alive

    def run_code(self, code: str) -> CodeResult:
        """Run ``code`` in the session and return its stdout, ``result``, and any error.

        A code error comes back as :attr:`CodeResult.error` (a traceback) and the
        session stays alive, so the agent iterates. A timeout or a dead child closes the
        session and is reported as an error rather than raised.
        """
        if not self._alive:
            return CodeResult(stdout="", result=None, error="the session is closed")
        try:
            return _exchange(self._proc, {"code": code}, self._timeout)
        except _SessionTimeout:
            self.close(kill=True)
            return CodeResult(
                stdout="",
                result=None,
                error=f"exploration exceeded the {self._timeout:g}s session timeout",
            )
        except _SessionExited:
            self._alive = False
            return CodeResult(stdout="", result=None, error="the session has exited")

    def run_bash(self, command: str) -> CodeResult:
        """Shell commands need the isolated docker backend; the host session declines.

        Running a shell on the host would execute against the user's own machine, which
        is the boundary the container backend exists to provide.
        """
        return CodeResult(
            stdout="",
            result=None,
            error="bash is only available with the docker sandbox backend",
        )

    def add_deps(self, deps: Sequence[str]) -> bool:
        """The host session's uv environment is fixed at start, so it adds no deps."""
        return False

    def close(self, *, kill: bool = False) -> None:
        """Shut the session down and release its process and temp files."""
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
        # Close the pipe file objects explicitly; left to garbage collection they raise
        # a ResourceWarning (which the test suite treats as an error).
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
        self._io.cleanup()


@dataclass
class SessionManager:
    """Lazily starts and reuses one exploration session, restarting it on new deps.

    Shared by the app runtime and the MCP server so both get the same stateful
    ``run_code`` plus ``run_bash``, with one place for backend selection and the
    deps-restart rule (a session's environment is fixed at creation, so a new dependency
    restarts it with the union, a clean reset). Exploration only; a derivation runs
    one-shot and hermetic through the executor, unaffected by any of this.
    """

    datasets: dict[str, list[dict[str, Any]]]
    workspace: Path | None = None
    backend: str = "subprocess"
    #: Outbound network policy for the docker backend: ``"full"``, ``"none"``, or a list
    #: of allowlisted hosts (ignored by the host-subprocess backend).
    egress: str | Sequence[str] = "full"
    #: Container image for the docker backend; ``None`` uses the slim default (a
    #: data-science image with the common stack baked in avoids per-session installs).
    image: str | None = None
    _session: Session | None = field(default=None, init=False, repr=False)
    _deps: tuple[str, ...] = field(default=(), init=False, repr=False)

    def run_code(self, code: str, deps: Sequence[str] = ()) -> CodeResult:
        """Run code in the session, (re)starting it when a call needs new deps."""
        return self._ensure(tuple(deps)).run_code(code)

    def run_bash(self, command: str) -> CodeResult:
        """Run a shell command in the session (declined unless the backend isolates)."""
        return self._ensure(()).run_bash(command)

    def _ensure(self, deps: tuple[str, ...]) -> Session:
        new = set(deps) - set(self._deps)
        # A prior call may have timed out or the worker may have exited, killing the
        # session. Drop the dead reference so a fresh one starts: its state is lost
        # (unavoidable once the worker is gone), but the next call runs instead of
        # failing forever with "the session is closed", so the conversation recovers.
        if self._session is not None and not self._session.alive:
            self._session = None
        if self._session is None:
            self._deps = tuple(sorted(set(self._deps) | new))
            self._session = self._make(self._deps)
        elif new:
            # Prefer adding the packages to the live session (docker installs them into
            # the running container), so the namespace survives; only restart if the
            # session's environment is fixed and cannot take them (the host backend).
            if self._session.add_deps(sorted(new)):
                self._deps = tuple(sorted(set(self._deps) | new))
            else:
                self._session.close()
                self._deps = tuple(sorted(set(self._deps) | new))
                self._session = self._make(self._deps)
        return self._session

    def _make(self, deps: tuple[str, ...]) -> Session:
        if self.backend == "docker":
            # Imported here, not at module load, because container imports this module.
            from .container import DockerReplSession

            return DockerReplSession(
                deps=deps,
                workspace=self.workspace,
                data=self.datasets,
                egress=self.egress,
                image=self.image,
            )
        return ReplSession(deps=deps, workspace=self.workspace, data=self.datasets)

    def close(self) -> None:
        """Tear down the session, if one was started."""
        if self._session is not None:
            self._session.close()
            self._session = None
