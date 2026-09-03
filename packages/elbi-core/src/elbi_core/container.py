"""A Docker-backed executor: run agent-authored derivations in a Linux container.

Same :class:`~elbi.executor.Executor` contract as
:class:`~elbi.executor.SubprocessExecutor`, but the derivation's untrusted source runs
inside an isolated Linux container instead of a host subprocess. That buys two things
the subprocess cannot: a namespace and cgroup boundary (generated code sees only the
container's filesystem, not the host's), run at least privilege (non-root as the host
uid, every capability dropped, no-new-privileges, and a read-only rootfs), so a
container escape does not land as host root; and a Linux userland where the manylinux
wheels for numpy, scikit-learn, and xgboost resolve without host system libraries (the
OpenMP/``libomp`` class of failure on macOS simply does not occur). This is a hardened
shared-kernel sandbox, not the kernel-level isolation a micro-VM backend would add
through the same :class:`~elbi.executor.Executor` seam.

A certified derivation stays hermetic. Dependencies are provisioned once, with network
access, into a cached volume keyed by the exact package set; the compute then runs with
networking disabled, mounting that volume read-only. So the output remains a pure
function of the source, its inputs, and its pinned dependencies, and the same result is
reproducible on re-run.

The :class:`~elbi.executor.Executor` seam is the extension point for remote
compute: a hosted micro-VM or third-party backend (AWS, Azure, GCP) is another ``run``
implementation, and nothing above this layer changes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifact import Artifact
from .context import Context
from .derivation import Derivation
from .errors import DerivationError
from .executor import _CHILD_MODULE, _DEP_RE, CodeResult, _child_path
from .session import _exchange, _SessionExited, _SessionTimeout

#: Packages ``import elbi`` needs to start inside the container. The numeric
#: verification gates import numpy/scipy lazily, so the core is this small; a
#: derivation's own libraries come from its declared ``deps``.
_CORE_DEPS = ("jsonschema", "pyyaml")

#: Default container image. A slim CPython; derivation dependencies install on top.
_DEFAULT_IMAGE = "python:3.12-slim"

#: A hostname allowed on an egress allowlist. Restricted so a name cannot smuggle extra
#: docker arguments; a bad name is rejected before any container is created.
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")

#: Label key stamped on every container and network an exploration session creates, with
#: the session id as its value. A crashed process cannot run :meth:`close`, so its
#: session container, egress proxy, and networks would otherwise leak; the label lets an
#: operator reap them (``docker ps -aq --filter label=elbi.session``).
_SESSION_LABEL = "elbi.session"


def _src_dir() -> str:
    """Host path of the ``elbi_core`` source root, mounted in the container."""
    # _sandbox_child.py -> elbi_core/ -> src/
    return str(Path(_child_path()).parents[1])


def _pkg_dir() -> str:
    """In-container path of the mounted package directory.

    Read off the child module's own location rather than spelled out, so a worker
    path cannot silently point at a package directory that no longer exists.
    """
    return f"/pkg/{Path(_child_path()).parent.name}"


def _hardening_args() -> list[str]:
    """Least-privilege docker flags for the hermetic derivation container.

    The derivation only runs ``python -m`` against read-only mounts, so it needs no
    Linux capabilities and no writable root filesystem. Dropping every capability,
    forbidding privilege escalation, a read-only rootfs with a small ``/tmp`` tmpfs for
    scratch, and running as the host uid (non-root, and matching the ``/io`` bind
    mount's owner so it stays writable) keep a container escape from landing as host
    root and bound the scratch a run can write. This is the real boundary (the
    subprocess backend is only a hardening), so it is posture-minimized, not run as
    container root.
    """
    args = [
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",  # noqa: S108 - a docker tmpfs mount, not a host temp path
    ]
    getuid, getgid = getattr(os, "getuid", None), getattr(os, "getgid", None)
    if getuid is not None and getgid is not None:  # POSIX host; matches the /io owner
        args += ["--user", f"{getuid()}:{getgid()}"]
    return args


class DockerExecutor:
    """Run a derivation's source inside a Linux container via the ``docker`` CLI.

    The container is ephemeral (``--rm``) and constrained: a memory cap, a CPU cap, a
    PID cap, no host environment, and no network during compute. Dependencies are
    provisioned into a cached named volume so repeated runs with the same package set
    do not re-download. The child protocol is identical to the subprocess sandbox (a
    pickled job in, JSON out), so the same ``elbi._sandbox_child`` is reused.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        timeout: float = 300.0,
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 256,
        allow_network: bool = False,
        docker_bin: str = "docker",
    ) -> None:
        if timeout <= 0:
            raise ValueError("docker executor timeout must be positive")
        self._image = image or _DEFAULT_IMAGE
        self._timeout = timeout
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._allow_network = allow_network
        self._docker = docker_bin

    def run(self, derivation: Derivation, context: Context) -> Any:
        """Execute the derivation's source in a container and return its value."""
        if derivation.source is None:
            raise DerivationError(
                f"derivation {derivation.name!r} has no source; the docker "
                "executor runs agent-authored derivations only"
            )
        deps = tuple(derivation.deps)
        for dep in deps:
            if not _DEP_RE.match(dep):
                raise DerivationError(
                    f"refusing to provision invalid dependency {dep!r}"
                )
        job = {
            "source": derivation.source,
            "fn_name": derivation.name,
            "inputs": dict(context.inputs),
            "params": dict(context.params),
            "allow_network": self._allow_network,
        }
        volume = ensure_deps_volume(self._image, deps, self._docker, self._timeout)
        label = f"derivation {derivation.name!r}"
        result = self._run_child(job, volume, label)
        if not result.get("ok"):
            raise DerivationError(
                f"derivation {derivation.name!r} failed in sandbox: "
                f"{result.get('error')}"
            )
        return Artifact(kind=result["kind"], value=result["value"])

    def _run_child(
        self, job: dict[str, Any], volume: str, label: str
    ) -> dict[str, Any]:
        """Run one job in a fresh, network-off container and return its JSON result."""
        name = f"elbi-run-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="elbi-docker-") as tmp:
            io_dir = Path(tmp)
            (io_dir / "job.pkl").write_bytes(pickle.dumps(job))
            argv = [
                self._docker,
                "run",
                "--rm",
                "--name",
                name,
                "--network",
                "bridge" if self._allow_network else "none",
                "--memory",
                self._memory,
                "--cpus",
                self._cpus,
                "--pids-limit",
                str(self._pids_limit),
                *_hardening_args(),
                "-v",
                f"{io_dir}:/io",
                "-v",
                f"{_src_dir()}:/pkg:ro",
                "-v",
                f"{volume}:/deps:ro",
                "-e",
                "PYTHONPATH=/pkg:/deps",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",  # rootfs is read-only; skip .pyc caching
                "-w",
                "/io",
                self._image,
                "python",
                "-m",
                _CHILD_MODULE,
                "/io",
            ]
            try:
                completed = subprocess.run(  # noqa: S603
                    argv,
                    capture_output=True,
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                # --rm does not fire on a killed run, so remove the container by name.
                subprocess.run(  # noqa: S603
                    [self._docker, "rm", "-f", name], capture_output=True, check=False
                )
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


def _docker_ok(docker_bin: str, args: list[str]) -> bool:
    """Whether a docker command exits cleanly (used to probe for the volume)."""
    try:
        completed = subprocess.run(  # noqa: S603
            [docker_bin, *args], capture_output=True, check=False
        )
    except FileNotFoundError as exc:
        raise DerivationError(
            f"docker executable {docker_bin!r} not found; install Docker or "
            "select the subprocess backend"
        ) from exc
    return completed.returncode == 0


def _docker_run_checked(
    docker_bin: str, args: list[str], what: str, timeout: float | None = None
) -> None:
    """Run a docker command, raising a readable error on failure."""
    try:
        completed = subprocess.run(  # noqa: S603
            [docker_bin, *args],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise DerivationError(
            f"docker executable {docker_bin!r} not found; install Docker or "
            "select the subprocess backend"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DerivationError(f"docker timed out trying to {what}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise DerivationError(f"docker failed to {what}: {stderr or '(no output)'}")


#: Written into a deps volume as the last provisioning step, so a volume counts as ready
#: only once its install has fully succeeded. `docker volume create` makes a volume
#: *exist* before `pip install` populates it, so existence alone does not mean ready: a
#: run (or a concurrent provisioner) that trusted existence would mount an empty or
#: half-installed `/deps`.
_READY_SENTINEL = "/deps/.elbi-ready"

#: Names of deps volumes this process has confirmed ready, so the hot path is a set
#: lookup rather than a docker call on every derivation run.
_ready_volumes: set[str] = set()

#: Serializes provisioning per volume name. The job runner is multi-threaded, and two
#: concurrent `pip install --target` into one volume interleave and corrupt it (pip
#: takes no lock on its target: pypa/pip#2746).
_provision_locks: dict[str, threading.Lock] = {}
_provision_guard = threading.Lock()


def _provision_lock(name: str) -> threading.Lock:
    """The lock guarding provisioning of the deps volume ``name``."""
    with _provision_guard:
        return _provision_locks.setdefault(name, threading.Lock())


def _volume_ready(image: str, name: str, docker_bin: str) -> bool:
    """Whether ``name`` exists and carries the completion sentinel (fully provisioned).

    Existence alone is not enough: `docker volume create` precedes the install.
    """
    if not _docker_ok(docker_bin, ["volume", "inspect", name]):
        return False
    return _docker_ok(
        docker_bin,
        ["run", "--rm", "-v", f"{name}:/deps", image, "test", "-f", _READY_SENTINEL],
    )


def ensure_deps_volume(
    image: str, deps: Sequence[str], docker_bin: str = "docker", timeout: float = 600.0
) -> str:
    """Provision (once) a named volume holding elbi's deps plus ``deps``.

    The volume name is content-addressed by the image and the exact package set, so the
    same set is provisioned only once and reused across derivation runs and sessions.
    Provisioning is the one step that gets network; callers that need a hermetic run
    mount the returned volume read-only with networking disabled.

    Provisioning is atomic and reproducible: it serializes per volume name so concurrent
    runs never share a half-installed volume, marks a volume ready only after its
    install fully succeeds, and discards a volume whose install failed or was
    interrupted rather than caching a partial one that a later run would mount.
    """
    packages = [*_CORE_DEPS, *deps]
    digest = hashlib.sha256("\n".join([image, *sorted(packages)]).encode()).hexdigest()[
        :16
    ]
    name = f"elbi-deps-{digest}"
    if name in _ready_volumes:
        return name
    with _provision_lock(name):
        if name in _ready_volumes:
            return name  # another thread provisioned it while we waited
        if _volume_ready(image, name, docker_bin):
            _ready_volumes.add(name)  # a prior process left a fully-provisioned volume
            return name
        # Discard any volume from a failed or interrupted earlier attempt: it may exist
        # (create precedes install) yet be empty or partial.
        _docker_ok(docker_bin, ["volume", "rm", "-f", name])
        _docker_run_checked(
            docker_bin, ["volume", "create", name], "create the deps volume"
        )
        try:
            _docker_run_checked(
                docker_bin,
                [
                    "run",
                    "--rm",
                    "-v",
                    f"{name}:/deps",
                    image,
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--target",
                    "/deps",
                    *packages,
                ],
                f"provision dependencies {packages}",
                timeout=max(timeout, 600.0),
            )
            # Mark ready only now, so a failed install never leaves a usable volume.
            _docker_run_checked(
                docker_bin,
                ["run", "--rm", "-v", f"{name}:/deps", image, "touch", _READY_SENTINEL],
                "mark the deps volume ready",
            )
        except DerivationError:
            # A failed or partial install must not be cached as valid: a later run would
            # mount an incomplete /deps and import the wrong thing (pypa/pip#10629).
            _docker_ok(docker_bin, ["volume", "rm", "-f", name])
            raise
        _ready_volumes.add(name)
    return name


def _wait_container_running(docker_bin: str, name: str) -> None:
    """Poll until container ``name`` is running, so an early call does not race."""
    inspect = [docker_bin, "inspect", "-f", "{{.State.Running}}", name]
    for _ in range(300):
        completed = subprocess.run(  # noqa: S603
            inspect, capture_output=True, text=True, check=False
        )
        if completed.stdout.strip() == "true":
            return
        time.sleep(0.1)


def _wait_container_listening(docker_bin: str, name: str, port: int) -> None:
    """Poll until ``name`` accepts a TCP connection on ``port`` (its proxy is up)."""
    probe = f"import socket; socket.create_connection(('127.0.0.1', {port}), 1).close()"
    for _ in range(300):
        completed = subprocess.run(  # noqa: S603
            [docker_bin, "exec", name, "python", "-c", probe],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return
        time.sleep(0.1)


def configure_egress(
    egress: str | Sequence[str],
    image: str,
    session_name: str,
    docker_bin: str = "docker",
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Build the container network flags for an egress policy, shared by both backends.

    Returns ``(net_args, env_args, containers, networks)``. ``"full"`` gives normal
    outbound access and ``"none"`` cuts it off (no extra resources). A list of hostnames
    is an allowlist enforced by a filtering proxy sidecar: the sandbox runs on an
    internal network with no route out, its ``HTTPS_PROXY`` points at the sidecar, and
    the sidecar (which alone reaches out) permits only those hosts. The returned
    network names are the caller's to remove on teardown.
    """
    if egress == "none":
        return ["--network", "none"], [], [], []
    if egress == "full":
        return ["--network", "bridge"], [], [], []
    hosts = [egress] if isinstance(egress, str) else list(egress)
    for host in hosts:
        if not _HOST_RE.match(host):
            raise DerivationError(f"invalid egress host {host!r}")
    ident = uuid.uuid4().hex[:12]
    internal = f"elbi-egnet-{ident}"
    outbound = f"elbi-egout-{ident}"
    proxy = f"elbi-egproxy-{ident}"
    label = ["--label", f"{_SESSION_LABEL}={session_name}"]
    networks: list[str] = []
    containers: list[str] = []
    _docker_run_checked(
        docker_bin,
        ["network", "create", "--internal", *label, internal],
        "create the egress network",
    )
    networks.append(internal)
    _docker_run_checked(
        docker_bin, ["network", "create", *label, outbound], "create the egress network"
    )
    networks.append(outbound)
    _docker_run_checked(
        docker_bin,
        [
            "run",
            "-d",
            "--name",
            proxy,
            *label,
            "--network",
            internal,
            "-v",
            f"{_src_dir()}:/pkg:ro",
            image,
            "python",
            f"{_pkg_dir()}/_egress_proxy.py",
            *hosts,
        ],
        "start the egress proxy",
    )
    containers.append(proxy)
    _docker_run_checked(
        docker_bin,
        ["network", "connect", outbound, proxy],
        "attach the egress proxy to the outbound network",
    )
    _wait_container_running(docker_bin, proxy)
    _wait_container_listening(docker_bin, proxy, 8888)
    # Only HTTPS_PROXY: the sidecar speaks CONNECT (HTTPS) only, and plain HTTP has no
    # route out from the internal network anyway, so this stays fail-closed.
    return (
        ["--network", internal],
        ["-e", f"HTTPS_PROXY=http://{proxy}:8888"],
        (containers),
        networks,
    )


class DockerReplSession:
    """A stateful exploration session whose namespace lives in a long-lived container.

    Both ``run_code`` (the persistent Python namespace) and ``run_bash`` (shell commands
    via ``docker exec``) run inside one container, so a package installed with ``bash``
    (``pip install``, ``apt-get``) is usable by the next ``run_code`` and files are
    shared. This is the isolated Linux box for exploration; a derivation still runs
    one-shot and hermetic through :class:`DockerExecutor`, so verification is intact.

    Network is enabled (exploration installs packages and fetches data), and the
    container is constrained by memory, CPU, and PID caps. It is not the certified path,
    so its state is never trusted; it is torn down on :meth:`close`.
    """

    def __init__(
        self,
        *,
        deps: Sequence[str] = (),
        workspace: Path | None = None,
        data: dict[str, list[dict[str, Any]]] | None = None,
        timeout: float = 300.0,
        image: str | None = None,
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 256,
        egress: str | Sequence[str] = "full",
        docker_bin: str = "docker",
    ) -> None:
        if timeout <= 0:
            raise ValueError("session timeout must be positive")
        image = image or _DEFAULT_IMAGE
        for dep in deps:
            if not _DEP_RE.match(dep):
                raise DerivationError(
                    f"refusing to provision invalid dependency {dep!r}"
                )
        self._timeout = timeout
        self._docker = docker_bin
        self._name = f"elbi-session-{uuid.uuid4().hex}"
        self._alive = True
        # Resources created for an allowlist egress policy, removed on close.
        self._egress_containers: list[str] = []
        self._egress_networks: list[str] = []
        net_args, env_args = self._setup_egress(egress, image)
        volume = ensure_deps_volume(image, deps, docker_bin, timeout)
        # A private host dir carries the datasets in and mounts at /io; the workspace
        # (if any) mounts at /work as the working dir so files persist across calls.
        self._io = tempfile.TemporaryDirectory(prefix="elbi-session-")
        (Path(self._io.name) / "data.json").write_text(
            json.dumps(data or {}), encoding="utf-8"
        )
        work_mount = str(workspace) if workspace is not None else self._io.name
        argv = [
            docker_bin,
            "run",
            "-i",
            "--name",
            self._name,
            "--label",
            f"{_SESSION_LABEL}={self._name}",
            *net_args,
            "--memory",
            memory,
            "--cpus",
            cpus,
            "--pids-limit",
            str(pids_limit),
            # The exploration box installs packages at runtime (pip, apt-get), so it
            # keeps a root, writable userland, not the derivation's read-only lockdown;
            # it is never the certified path. no-new-privileges is the one hardening
            # that does not break that (blocks setuid escalation, not package installs).
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{self._io.name}:/io",
            "-v",
            f"{_src_dir()}:/pkg:ro",
            "-v",
            f"{volume}:/deps:ro",
            "-v",
            f"{work_mount}:/work",
            "-e",
            "PYTHONPATH=/deps",
            *env_args,
            "-w",
            "/work",
            image,
            "python",
            f"{_pkg_dir()}/_repl_worker.py",
            "/io/data.json",
            "--allow-network",
        ]
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )
        except FileNotFoundError as exc:
            self._alive = False
            raise DerivationError(
                f"docker executable {docker_bin!r} not found; install Docker or "
                "select the subprocess backend"
            ) from exc
        # run_code waits on the pipe naturally, but run_bash (docker exec) needs the
        # container already up, so wait for it to be running before returning.
        self._wait_running(self._name)

    def _wait_running(self, name: str) -> None:
        """Poll until container ``name`` is running, so an early exec does not race."""
        inspect = [self._docker, "inspect", "-f", "{{.State.Running}}", name]
        for _ in range(300):
            completed = subprocess.run(  # noqa: S603
                inspect, capture_output=True, text=True, check=False
            )
            if completed.stdout.strip() == "true":
                return
            time.sleep(0.1)

    def _wait_listening(self, name: str, port: int) -> None:
        """Poll until ``name`` accepts a TCP connection on ``port`` (proxy is up)."""
        probe = (
            f"import socket; socket.create_connection(('127.0.0.1', {port}), 1).close()"
        )
        for _ in range(300):
            completed = subprocess.run(  # noqa: S603
                [self._docker, "exec", name, "python", "-c", probe],
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                return
            time.sleep(0.1)

    def _setup_egress(
        self, egress: str | Sequence[str], image: str
    ) -> tuple[list[str], list[str]]:
        """Configure the container's egress policy, tracking any resources for teardown.

        Delegates to the shared :func:`configure_egress`; the proxy container and nets
        an allowlist creates are recorded so :meth:`close` removes them.
        """
        net_args, env_args, containers, networks = configure_egress(
            egress, image, self._name, self._docker
        )
        self._egress_containers.extend(containers)
        self._egress_networks.extend(networks)
        return net_args, env_args

    @property
    def alive(self) -> bool:
        """Whether the container is still running (False after a timeout or exit)."""
        return self._alive

    def run_code(self, code: str) -> CodeResult:
        """Run ``code`` in the container's persistent namespace; return its output."""
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

    def add_deps(self, deps: Sequence[str]) -> bool:
        """Install ``deps`` into the running container, keeping the session's state.

        Instead of restarting the session (which would wipe the namespace) when a call
        needs a new package, pip-install it into the live container; the worker picks it
        up on the next ``import``, so a loaded dataframe or fitted model stays. The
        install is best-effort: if a package fails to build, the session is still kept
        (returns True), so the failure surfaces as an import error the model adapts to
        (pick another package) rather than silently losing all of its work to a restart.
        A dead session returns False so the manager recreates it.
        """
        if not self._alive:
            return False
        valid = [dep for dep in deps if _DEP_RE.match(dep)]
        if valid:
            subprocess.run(  # noqa: S603 - best-effort; a build failure is not fatal
                [
                    self._docker,
                    "exec",
                    self._name,
                    "pip",
                    "install",
                    "--no-cache-dir",
                    *valid,
                ],
                capture_output=True,
                timeout=self._timeout,
                check=False,
            )
        return True

    def run_bash(self, command: str) -> CodeResult:
        """Run a shell command in the live container via ``docker exec``.

        It shares the container's filesystem and installed packages with ``run_code``,
        so ``pip install`` or ``apt-get`` here is available to Python next call.
        """
        if not self._alive:
            return CodeResult(stdout="", result=None, error="the session is closed")
        try:
            completed = subprocess.run(  # noqa: S603
                [
                    self._docker,
                    "exec",
                    "-w",
                    "/work",
                    self._name,
                    "bash",
                    "-lc",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CodeResult(
                stdout="",
                result=None,
                error=f"the command exceeded the {self._timeout:g}s timeout",
            )
        stdout = (completed.stdout or "")[:8000]
        stderr = (completed.stderr or "").strip()
        error = (
            None
            if completed.returncode == 0
            else f"exit {completed.returncode}: {stderr}"
        )
        return CodeResult(stdout=stdout, result=None, error=error)

    def close(self, *, kill: bool = False) -> None:
        """Stop and remove the container and release its pipes and temp files."""
        if self._alive and not kill and self._proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError, ValueError):
                self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self._proc.stdin.flush()
        self._alive = False
        # Best-effort teardown of the container, the egress proxy, and its networks.
        removals = [["rm", "-f", self._name]]
        removals += [["rm", "-f", name] for name in self._egress_containers]
        removals += [["network", "rm", net] for net in self._egress_networks]
        for args in removals:
            subprocess.run(  # noqa: S603
                [self._docker, *args], capture_output=True, check=False
            )
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=5)
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
        self._io.cleanup()
