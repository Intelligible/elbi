"""Executors: where a derivation's compute runs.

Trusted human-authored derivations run in-process. Agent-authored derivations
carry untrusted generated source and run under an isolating executor instead.
:class:`Executor` is a pluggable protocol, so a platform can swap in a microVM or
hosted-sandbox backend with no derivation-code changes.

The subprocess sandbox is a hardened local default (scrubbed environment, an
ephemeral working directory, resource limits, denied network egress, and a
timeout). It is not a security boundary against hostile code; kernel-level
isolation belongs to a microVM executor reached through the same protocol.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .artifact import Artifact
from .context import Context
from .derivation import Derivation
from .errors import DerivationError

#: How a sandboxed derivation's compute function is found in its source: by a
#: function whose name matches the derivation name.
_CHILD_MODULE = "elbi_core._sandbox_child"

#: A declared dependency: a PEP 508 name (with optional extras and version), the
#: form `uv` accepts. Anything else is rejected before it reaches the resolver, so a
#: model-hallucinated or option-injecting string cannot become a uv argument; a name
#: that is well-formed but does not exist simply fails resolution (the cheap
#: existence check that defeats most slopsquatting).
_DEP_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?([<>=!~][^;\s]*)?$"
)

#: Environment variables the sandbox child is allowed to inherit. Everything else
#: (notably credentials in ``os.environ``) is withheld. These are what a normal
#: Python + numpy/pandas process needs to start and locate its config.
_SAFE_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
)


@dataclass(frozen=True)
class CodeResult:
    """The outcome of running exploratory code in the sandbox.

    ``error`` is a traceback string when the code raised (and ``None`` otherwise);
    a code error is captured here, not raised, so an agent can read it and iterate.
    ``result`` is the ``repr`` of a ``result`` variable the code may set.
    """

    stdout: str
    result: str | None = None
    error: str | None = None


@runtime_checkable
class Executor(Protocol):
    """A backend that runs a derivation's compute and returns its raw value."""

    def run(self, derivation: Derivation, context: Context) -> Any:
        """Compute ``derivation`` against ``context`` and return the value."""
        ...


class InProcessExecutor:
    """Run compute in the host process. The default; for trusted code only.

    An agent-authored derivation's ``compute`` refuses to run here (it raises),
    so untrusted generated source never executes in the host process by accident.
    """

    def run(self, derivation: Derivation, context: Context) -> Any:
        """Call the derivation's compute function directly."""
        return derivation.compute(context)


class SubprocessExecutor:
    """Run a derivation's source in a fresh, hardened Python subprocess.

    The child runs with a scrubbed environment (only :data:`_SAFE_ENV` plus
    ``env_passthrough`` are inherited), an ephemeral working directory, resource
    limits (CPU and file size always; address space when ``memory_limit`` is set),
    denied network egress unless ``allow_network`` is set, and a wall-clock
    ``timeout``. Inputs and params are pickled in; the result comes back as JSON.
    """

    #: Hard ceiling on bytes the child may write to disk (defends the filesystem).
    _FSIZE_LIMIT = 512 * 1024 * 1024

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        memory_limit: int | None = None,
        env_passthrough: tuple[str, ...] = (),
        allow_network: bool = False,
        python_executable: str = sys.executable,
    ) -> None:
        if timeout <= 0:
            raise ValueError("subprocess executor timeout must be positive")
        if memory_limit is not None and memory_limit <= 0:
            raise ValueError("subprocess executor memory_limit must be positive")
        self._timeout = timeout
        self._memory_limit = memory_limit
        self._env_passthrough = tuple(env_passthrough)
        self._allow_network = allow_network
        # The interpreter the child runs under. Defaults to this process's, so a
        # plain install just works; a deployment points it at a separate, curated
        # environment (the data-science stack, or a microVM's interpreter) to widen
        # what agent code may import and to harden isolation, without touching the
        # engine above this seam.
        self._python_executable = python_executable

    def run(self, derivation: Derivation, context: Context) -> Any:
        """Execute the derivation's source in a child process and return the value."""
        if derivation.source is None:
            raise DerivationError(
                f"derivation {derivation.name!r} has no source; the subprocess "
                "executor runs agent-authored derivations only"
            )
        job = {
            "source": derivation.source,
            "fn_name": derivation.name,
            "inputs": dict(context.inputs),
            "params": dict(context.params),
            "allow_network": self._allow_network,
        }
        # A derivation may declare third-party packages (numpy, scikit-learn); they are
        # provisioned into the sandbox the same way exploration deps are, and the child
        # runs as a module so elbi stays importable for its Context and inputs.
        result = self._invoke_child(
            job,
            f"derivation {derivation.name!r}",
            deps=derivation.deps,
            needs_package=True,
        )
        if not result.get("ok"):
            raise DerivationError(
                f"derivation {derivation.name!r} failed in sandbox: "
                f"{result.get('error')}"
            )
        return Artifact(kind=result["kind"], value=result["value"])

    def run_code(
        self,
        code: str,
        data: dict[str, list[dict[str, Any]]] | None = None,
        deps: Sequence[str] = (),
        *,
        workspace: Path | None = None,
    ) -> CodeResult:
        """Run exploratory ``code`` in the sandbox, with ``data`` rows in scope.

        ``data`` maps a name to a list of row dicts, exposed to the code as a
        ``data`` dict. ``deps`` names packages the code imports (e.g. ``interpret``
        for an EBM); they are provisioned on demand into a cached ephemeral
        environment with ``uv`` rather than bundled, so any analysis can pull the
        library it needs. This is the iterative explore/debug primitive that precedes
        authoring a derivation: it runs untrusted code in the same hardened sandbox
        as a derivation, but returns the code's stdout and ``result`` rather than a
        served artifact. A code error comes back as :attr:`CodeResult.error`, not an
        exception, so the caller can read it and refine.

        ``workspace`` is a persistent working directory that becomes the child's
        current directory, so a file one call writes (a fitted model, an intermediate
        table) is there for the next call to read. Only the filesystem persists: each
        call is still a fresh process, so in-memory variables do not carry over, and
        code that wants continuity must write to a file and read it back. This is
        exploration state only; a derivation runs with a private, empty directory (see
        :meth:`run`) so a certified result stays a pure function of its declared
        inputs. Left as ``None``, every call runs in its own throwaway directory.
        """
        job = {
            "mode": "exec",
            "code": code,
            "data": dict(data or {}),
            "allow_network": self._allow_network,
        }
        try:
            result = self._invoke_child(
                job, "exploration", deps=deps, workspace=workspace
            )
        except DerivationError as exc:
            # Provisioning/sandbox failures (e.g. an unresolvable dependency name)
            # come back as a readable error the agent can fix, not an exception.
            return CodeResult(stdout="", result=None, error=str(exc))
        return CodeResult(
            stdout=result.get("stdout", ""),
            result=result.get("result"),
            error=result.get("error"),
        )

    def _invoke_child(
        self,
        job: dict[str, Any],
        label: str,
        deps: Sequence[str] = (),
        *,
        workspace: Path | None = None,
        needs_package: bool = False,
    ) -> dict[str, Any]:
        """Run one job in a fresh hardened child and return its parsed JSON result.

        Raises :class:`DerivationError` on a sandbox-level failure (timeout or no
        result); the parsed dict is returned otherwise, for the caller to interpret.

        The job travels through a private, per-call temp directory (``job.pkl`` in,
        ``result.json`` out) that is always discarded. ``workspace``, when given, is
        the child's current directory instead, so files survive between calls; the
        transport files stay in the private directory and are addressed by absolute
        path, so a persistent workspace never mixes with them.
        """
        # validate deps before any filesystem work
        self._argv(deps, "", needs_package=needs_package)
        with tempfile.TemporaryDirectory(prefix="elbi-sandbox-") as tmp:
            io_dir = Path(tmp)
            # The pickle producer is trusted (our own process); the untrusted code
            # runs in the child and its output returns as JSON, never unpickled.
            (io_dir / "job.pkl").write_bytes(pickle.dumps(job))
            argv = self._argv(deps, str(io_dir), needs_package=needs_package)
            # The child reads job.pkl by the absolute path in argv, so its working
            # directory is free to be a persistent workspace when one is supplied.
            cwd = workspace if workspace is not None else io_dir
            try:
                # No shell and no agent input on the command line: the job travels in
                # job.pkl. With deps, uv resolves them into a cached env and runs the
                # child there; without deps, the child runs in this interpreter.
                completed = subprocess.run(  # noqa: S603
                    argv,
                    capture_output=True,
                    timeout=self._timeout,
                    check=False,
                    env=_child_env(self._env_passthrough),
                    cwd=str(cwd),
                    preexec_fn=self._resource_limiter() if not deps else None,
                )
            except subprocess.TimeoutExpired as exc:
                raise DerivationError(
                    f"{label} exceeded the {self._timeout:g}s sandbox timeout"
                ) from exc
            result_path = io_dir / "result.json"
            if not result_path.exists():
                stderr = completed.stderr.decode("utf-8", "replace").strip()
                raise DerivationError(
                    f"sandbox for {label} produced no result "
                    f"(exit {completed.returncode}): {stderr or '(no output)'}"
                )
            parsed: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
            return parsed

    def _argv(
        self, deps: Sequence[str], io_dir: str, *, needs_package: bool = False
    ) -> list[str]:
        """The child command: a plain interpreter, or uv provisioning ``deps`` first.

        Without ``deps`` the child runs under the configured interpreter, where
        elbi is already importable. With ``deps``, uv resolves them into a
        cached ephemeral environment. A derivation (``needs_package``) also needs
        elbi in that environment for its Context and pickled inputs, so it is
        installed editable and the child runs as a module; exploration code is
        stdlib-only, so it runs by file path with no such install.
        """
        if not deps:
            return [self._python_executable, "-I", "-m", _CHILD_MODULE, io_dir]
        for dep in deps:
            if not _DEP_RE.match(dep):
                raise DerivationError(
                    f"refusing to provision invalid dependency {dep!r}"
                )
        with_args = [arg for dep in deps for arg in ("--with", dep)]
        base = ["uv", "run", "--no-project", "--python", self._python_executable]
        if needs_package:
            return [
                *base,
                "--with-editable",
                _package_root(),
                *with_args,
                "--",
                "python",
                "-I",
                "-m",
                _CHILD_MODULE,
                io_dir,
            ]
        return [*base, *with_args, "--", "python", "-I", _child_path(), io_dir]

    def _resource_limiter(self) -> Callable[[], None] | None:
        """A ``preexec_fn`` that caps the child's resources, or None off POSIX."""
        if not hasattr(os, "fork"):  # pragma: no cover - POSIX-only resource limits
            return None
        cpu_seconds = max(1, int(self._timeout) + 1)
        fsize = self._FSIZE_LIMIT
        memory = self._memory_limit

        def apply() -> None:  # pragma: no cover - runs in the child after fork
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            if memory is not None:
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))

        return apply


class RoutingExecutor:
    """Route each derivation by isolation need.

    A derivation carrying generated ``source`` is untrusted and runs in the
    sandbox; trusted compute functions run in the host process.
    """

    def __init__(self, *, sandbox: Executor, trusted: Executor | None = None) -> None:
        self._sandbox = sandbox
        self._trusted = trusted if trusted is not None else InProcessExecutor()

    def run(self, derivation: Derivation, context: Context) -> Any:
        """Dispatch a source-carrying derivation to the sandbox, else in-process."""
        executor = self._sandbox if derivation.source is not None else self._trusted
        return executor.run(derivation, context)


def _child_env(passthrough: tuple[str, ...]) -> dict[str, str]:
    """Return the sandbox child's environment: allowlist plus passthrough only.

    Withholding the rest of ``os.environ`` keeps host credentials out of reach of
    untrusted code.
    """
    allowed = set(_SAFE_ENV) | set(passthrough)
    return {key: value for key, value in os.environ.items() if key in allowed}


def _child_path() -> str:
    """Filesystem path of the sandbox child, to run it by path in a uv env.

    Running by path means the ephemeral, dependency-only environment does not need
    ``elbi`` installed in it; the child is stdlib-only.
    """
    spec = importlib.util.find_spec(_CHILD_MODULE)
    if spec is None or spec.origin is None:  # pragma: no cover - always importable
        raise DerivationError("sandbox child module not found")
    return spec.origin


def _package_root() -> str:
    """The elbi project root (holding ``pyproject.toml``), to editable-install.

    A dependency-carrying derivation runs in a uv environment that must import
    elbi (for its Context and pickled inputs). Installing this source tree
    editable puts it on the child's path without needing a published release.
    """
    # _sandbox_child.py -> elbi/ -> src/ -> <project root>
    return str(Path(_child_path()).parents[2])


#: The default executor: in-process, for trusted code.
DEFAULT: Executor = InProcessExecutor()
