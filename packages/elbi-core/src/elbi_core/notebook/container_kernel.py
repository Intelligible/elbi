"""Docker-backed notebook kernel: an interpreter isolated from the host.

Same notebook protocol as :class:`~elbi.notebook.kernel.SubprocessKernel`, but the
worker runs inside a constrained container rather than a host process, the real
isolation boundary the subprocess backend cannot provide. It reuses the container
plumbing (the content-addressed dependency volume, the hardening flags, the source
mount) from :mod:`elbi.container`, so a notebook kernel and the certified-derivation
sandbox share one hardening story. Interrupting a cell sends SIGINT into the container;
restarting replaces it.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..container import (
    _DEFAULT_IMAGE,
    _SESSION_LABEL,
    _pkg_dir,
    _src_dir,
    configure_egress,
    ensure_deps_volume,
)
from ..errors import DerivationError
from ..executor import _DEP_RE
from .kernel import MAX_OUTPUT_BYTES, _StreamingKernel, encode_datasets


def _worker_in_container() -> str:
    """Path the kernel worker runs from inside the container.

    The package source is mounted read-only at ``/pkg``, so the worker runs by path
    without ``elbi_core`` installed.
    """
    return f"{_pkg_dir()}/notebook/_kernel_worker.py"


class DockerNotebookKernel(_StreamingKernel):
    """A notebook kernel whose interpreter runs in a hardened, disposable container.

    The container drops all capabilities it does not need, runs unprivileged, and mounts
    the dependency volume read-only; its network is the deployment's egress policy
    (``"full"`` or ``"none"``). It is not the certified path (a notebook is authoring),
    so, like the exploration container, it keeps a writable userland for interactive
    use.
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
        image: str | None = None,
        egress: str | Sequence[str] = "full",
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 256,
        credentials: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        docker_bin: str = "docker",
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
        self._docker = docker_bin
        self._name = f"elbi-nb-{uuid.uuid4().hex}"
        image = image or _DEFAULT_IMAGE
        # ``"full"``/``"none"``, or a host allowlist enforced by a proxy sidecar;
        # any resources it creates are torn down in ``close``.
        net_args, env_args, self._egress_containers, self._egress_networks = (
            configure_egress(egress, image, self._name, docker_bin)
        )
        volume = ensure_deps_volume(image, deps, docker_bin, cell_timeout + 480.0)
        self._io = tempfile.TemporaryDirectory(prefix="elbi-nb-")
        # Lazy mode writes only the dataset names; rows are fetched through the host
        # when a cell asks, so nothing large is serialised into the io mount.
        io_args = ["/io/data.json"]
        if dataset_names is not None:
            (Path(self._io.name) / "data.json").write_text("{}", encoding="utf-8")
            (Path(self._io.name) / "names.json").write_text(
                json.dumps(list(dataset_names)), encoding="utf-8"
            )
            io_args.append("/io/names.json")
        else:
            (Path(self._io.name) / "data.json").write_text(
                encode_datasets(data), encoding="utf-8"
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
            "--security-opt",
            "no-new-privileges",
            "-e",
            "PYTHONPATH=/deps",
            "-e",
            "MPLBACKEND=Agg",
            # Scoped to this session's prefixes and short-lived, so a cell reads the
            # warehouse directly instead of every byte crossing the app.
            *[
                arg
                for key, value in sorted({**(credentials or {}), **(env or {})}.items())
                for arg in ("-e", f"{key}={value}")
            ],
            *env_args,
            "-v",
            f"{self._io.name}:/io",
            "-v",
            f"{_src_dir()}:/pkg:ro",
            "-v",
            f"{volume}:/deps:ro",
            "-v",
            f"{work_mount}:/work",
            "-w",
            "/work",
            image,
            # The container's own network policy is authoritative, so let the worker's
            # in-process egress guard stand aside (``--allow-network``);
            # ``egress="none"`` still cuts the container off at the network layer.
            "python",
            _worker_in_container(),
            *io_args,
            "--allow-network",
        ]
        try:
            self._proc = subprocess.Popen(  # noqa: S603
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )
        except FileNotFoundError as exc:
            self._alive = False
            raise DerivationError(
                f"docker executable {docker_bin!r} not found; install Docker or select "
                "the subprocess backend"
            ) from exc
        self._start_reader()

    @property
    def alive(self) -> bool:
        """Whether the container is still running."""
        return self._alive and self._proc.poll() is None

    def interrupt(self) -> None:
        """Deliver SIGINT to the container's worker, raising KeyboardInterrupt in it."""
        if not self.alive:
            return
        subprocess.run(  # noqa: S603
            [self._docker, "kill", "--signal=INT", self._name],
            capture_output=True,
            check=False,
        )

    def close(self, *, kill: bool = False) -> None:
        """Shut the kernel down: ask the worker to stop, then remove the container."""
        if self._alive and not kill and self._proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError, ValueError):
                self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self._proc.stdin.flush()
        self._alive = False
        # Remove the kernel container and any egress proxy/nets an allowlist made.
        removals = [["rm", "-f", self._name]]
        removals += [["rm", "-f", name] for name in self._egress_containers]
        removals += [["network", "rm", name] for name in self._egress_networks]
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
