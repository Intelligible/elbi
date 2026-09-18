"""The notebook service: documents, live kernels, reactive execution, interchange.

This is the app-side surface over :mod:`elbi.notebook`. It persists notebooks and cells
through the :class:`~elbi.db.Store`, keeps one live kernel per open notebook (started
lazily, evicted when idle, restarted when its dependencies change), and turns a cell run
into a stream of output events the API relays to the browser over SSE.

Execution is reactive: running a cell re-runs exactly its transitive dependents, in
dependency order, computed by static analysis of the cells' code. Staleness (a cell
whose inputs changed since it last ran) is derived the same way, so the UI can flag what
is out of date without guessing. The kernel and its rich output are the SDK's; the
service adds the persistence, the dependency-aware run planning, and the ``.ipynb``
bridge.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from elbi_core.errors import DerivationError
from elbi_core.executor import _DEP_RE
from elbi_core.notebook import (
    MAX_OUTPUT_BYTES,
    Cell,
    CommListener,
    DependencyGraph,
    Kernel,
    SubprocessKernel,
    apply_overrides,
    resolve_lock,
)
from elbi_core.notebook import Notebook as NotebookDoc
from elbi_core.sandbox import (
    ComputeProfile,
    ComputeProfileError,
    ComputeProfiles,
    docker_resource_args,
)

from .db import NotebookCell, Store
from .duplicate import copy_label

if TYPE_CHECKING:
    from .authoring import DeriveFactory

#: What the app hands this service: the kernel-side resolver plus the caller whose
#: access the read is authorized against. The kernel's own protocol stays
#: ``(sql, table, limit)`` -- a kernel has no business knowing who it runs for.
QueryResolver = Callable[[str, str, int], dict[str, Any]]


logger = logging.getLogger("elbi")

#: The workspace setting holding the curated base environments (a JSON list of
#: ``{"name": ..., "deps": [...]}``), which a notebook can layer its own packages on.
BASE_ENV_SETTING = "notebook.base_environments"

#: A ``%pip install`` / ``!pip install`` / ``%uv pip install`` line, the notebook-scoped
#: install magic. The captured tail is the package specs (and any flags, which we drop).
_INSTALL_RE = re.compile(r"^\s*[%!]\s*(?:uv\s+)?pip\s+install\s+(?P<args>.+?)\s*$")


def _install_specs(source: str) -> list[str]:
    """The packages a cell installs, if every non-blank line is a pip-install magic.

    A cell used to add packages (``%pip install pandas``) is not Python; we treat it as
    an environment directive rather than code. A cell that mixes install lines with real
    code is left alone (its magic line will error in the kernel, prompting the user to
    separate the two: the same discipline Jupyter and Databricks expect).
    """
    specs: list[str] = []
    matched = False
    for line in source.splitlines():
        if not line.strip():
            continue
        match = _INSTALL_RE.match(line)
        if match is None:
            return []  # a real statement: a code cell, not an install cell
        matched = True
        for token in match.group("args").split():
            if not token.startswith("-") and _DEP_RE.match(token):
                specs.append(token)
    return specs if matched else []


def _dedupe(specs: Sequence[str]) -> list[str]:
    """Order-preserving de-duplication of dependency specs."""
    seen: set[str] = set()
    out: list[str] = []
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            out.append(spec)
    return out


Rows = list[dict[str, Any]]


def _hash(source: str) -> str:
    """A short content hash of a cell's source, for detecting edits since a run."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _derivation_name(source: str) -> str | None:
    """The name of a cell's derivation function: a top-level ``def <name>(ctx, ...)``.

    Prefers a function whose first parameter is ``ctx`` (the derivation convention); if
    none matches but a single top-level function is defined, that one is used.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in functions:
        args = node.args.args
        if args and args[0].arg == "ctx":
            return node.name
    return functions[0].name if len(functions) == 1 else None


def _to_output(message: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a kernel worker message to its nbformat output object.

    The worker tags messages with ``type``; nbformat names the same field
    ``output_type``.
    Everything else in the four shapes (stream/display_data/execute_result/error)
    already
    matches, so this is a rename, not a translation.
    """
    output = {key: value for key, value in message.items() if key != "type"}
    output["output_type"] = message["type"]
    return output


def _coalesce(outputs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-stream outputs into one, as Jupyter does on save."""
    merged: list[dict[str, Any]] = []
    for output in outputs:
        if (
            output.get("output_type") == "stream"
            and merged
            and merged[-1].get("output_type") == "stream"
            and merged[-1].get("name") == output.get("name")
        ):
            merged[-1] = {**merged[-1], "text": merged[-1]["text"] + output["text"]}
        else:
            merged.append(dict(output))
    return merged


@dataclass
class _Runtime:
    """A notebook's live kernel plus the bookkeeping reactive execution needs.

    ``ran`` records the source hash each cell held when it last ran successfully (an
    edit
    since then makes it stale), and ``run_seq`` the global counter value at that run (an
    upstream that ran later makes a downstream stale). ``lock`` serializes runs so a
    run-all and a manual run never drive the single kernel at once.
    """

    kernel: Kernel
    deps: tuple[str, ...]
    workspace: Path | None
    #: The profile this kernel started with, and the version that profile had at the
    #: time. A later edit changes the version but not this kernel, which is what
    #: ``runtime_state`` reports rather than pretending the change took effect.
    profile: ComputeProfile = field(
        default_factory=lambda: ComputeProfile(name="default")
    )
    profile_version: str = ""
    #: This notebook's effective idle timeout: its own request, clamped by the profile.
    idle_timeout: float = 1800.0
    #: ``interactive`` or ``batch``. A scheduled rerun and an abandoned session cost the
    #: same per second and mean entirely different things in a cost report.
    kind: str = "interactive"
    counter: int = 0
    ran: dict[str, str] = field(default_factory=dict)
    run_seq: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = 0.0
    started_at: float = 0.0
    #: Set while a cell is executing. The reaper never closes a busy kernel: reaping is
    #: for idleness, and killing a running cell would lose work rather than free waste.
    busy: bool = False


class _WarmPool:
    """Kernels started ahead of demand, so opening a notebook does not wait for a pod.

    A pooled kernel is interchangeable only with one that would have been started the
    same way, so the key is everything that shapes a kernel: the profile *as defined at
    that moment* (its version, so a resized profile does not hand out kernels built to
    the old size) and the exact dependency set. Nothing notebook-specific can be
    pre-warmed, which is why a per-notebook workspace disqualifies a session from using
    the pool.

    Off unless a profile asks for it: idle capacity is paid for whether or not a
    notebook opens, so the default is zero and the cost is documented rather than
    hidden.
    """

    def __init__(
        self, start: Callable[[ComputeProfile, tuple[str, ...]], Kernel]
    ) -> None:
        self._start = start
        self._ready: dict[tuple[str, str, tuple[str, ...]], list[Kernel]] = {}
        self._lock = threading.Lock()
        self._filling: set[tuple[str, str, tuple[str, ...]]] = set()

    @staticmethod
    def _key(
        profile: ComputeProfile, deps: tuple[str, ...]
    ) -> tuple[str, str, tuple[str, ...]]:
        return (profile.name, profile.version, deps)

    def take(self, profile: ComputeProfile, deps: tuple[str, ...]) -> Kernel | None:
        """A ready kernel of this shape, or None; refills the pool in the background."""
        if profile.warm_pool_size <= 0:
            return None
        key = self._key(profile, deps)
        kernel: Kernel | None = None
        with self._lock:
            while self._ready.get(key):
                candidate = self._ready[key].pop()
                # A pooled kernel can die while it waits (an evicted node, a cluster
                # restart). Handing out a dead one would look like a start failure.
                if candidate.alive:
                    kernel = candidate
                    break
        self._replenish(profile, deps)
        return kernel

    def _replenish(self, profile: ComputeProfile, deps: tuple[str, ...]) -> None:
        """Top the pool up to the profile's size, on a background thread."""
        key = self._key(profile, deps)
        with self._lock:
            if key in self._filling:
                return
            have = len([k for k in self._ready.get(key, []) if k.alive])
            want = profile.warm_pool_size - have
            if want <= 0:
                return
            self._filling.add(key)

        def fill() -> None:
            try:
                for _ in range(want):
                    try:
                        kernel = self._start(profile, deps)
                    except (DerivationError, ComputeProfileError) as exc:
                        # A pool that cannot be filled must not take the app with it,
                        # and must not retry forever: the next `take` retries.
                        logger.warning(
                            "could not pre-warm a %s kernel (%s); the pool stays short",
                            profile.name,
                            exc,
                        )
                        return
                    with self._lock:
                        self._ready.setdefault(key, []).append(kernel)
            finally:
                with self._lock:
                    self._filling.discard(key)

        threading.Thread(target=fill, daemon=True).start()

    def close(self) -> None:
        """Shut down every waiting kernel."""
        with self._lock:
            kernels = [k for group in self._ready.values() for k in group]
            self._ready.clear()
        for kernel in kernels:
            kernel.close()

    def depth(self) -> dict[str, int]:
        """How many kernels are waiting, per profile, for reporting."""
        with self._lock:
            counts: dict[str, int] = {}
            for (name, _, _), group in self._ready.items():
                counts[name] = counts.get(name, 0) + len([k for k in group if k.alive])
            return counts


#: Cells the editor does not draw: environment a derivation needs bound, not the
#: thing being edited. They still run.
SETUP_ROLE = "setup"


class NotebookService:
    """Owns notebook persistence and the pool of live kernels.

    One instance backs the whole app. Kernel configuration (backend, egress, image, cell
    timeout) mirrors the chat workspace's, so a notebook runs under the same sandbox the
    rest of the product does: a host subprocess locally, a container on the platform.
    """

    def __init__(
        self,
        store: Store,
        load_datasets: Callable[[], dict[str, Rows]],
        dataset_names: Callable[[], list[str]] | None = None,
        query_resolver: QueryResolver | None = None,
        backend: str = "subprocess",
        egress: str | Sequence[str] = "full",
        image: str | None = None,
        profiles: ComputeProfiles | None = None,
        runner_options: Callable[[], Mapping[str, Any]] | None = None,
        credentials: Callable[[], Mapping[str, str]] | None = None,
        kernel_env: Callable[[], Mapping[str, str]] | None = None,
        on_session_end: (
            Callable[[str, ComputeProfile, float, str], None] | None
        ) = None,
        scratch_root: Path | None = None,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        derive_factory: DeriveFactory | None = None,
    ) -> None:
        self._store = store
        self._load_datasets = load_datasets
        # With both of these a kernel starts empty and pulls what a cell asks for.
        # Without them (an embedding caller that has only `load_datasets`) the older
        # eager path still applies, so nothing that constructs this service breaks.
        self._dataset_names = dataset_names
        self._query_resolver = query_resolver
        self._backend = backend
        self._egress = egress
        self._image = image
        # Infrastructure defines the menu; a notebook picks from it. A deployment that
        # defines none gets a single profile at the built-in defaults, so the whole
        # mechanism is invisible until someone wants it.
        self._profiles = profiles or ComputeProfiles(
            profiles=(ComputeProfile(name="default", managed=False),), default="default"
        )
        # Mints the scoped, short-lived storage credentials a kernel starts with. None
        # when no broker is configured, in which case data reaches a cell through the
        # host's query resolver instead and the sandbox holds no credential at all.
        self._credentials = credentials
        # The deployment's own service addresses (the experiment-tracking URI), without
        # which a run logged from a cell goes to a directory that dies with the kernel.
        self._kernel_env = kernel_env
        # Called as a kernel ends, with what it ran on and for how long. Usage history
        # cannot be backfilled, so this is wired from the first session rather than when
        # somebody asks for a cost report.
        self._on_session_end = on_session_end
        # Cluster settings only the kubernetes runner understands (namespace, the deps
        # storage class, the session deadline). Passed through rather than enumerated,
        # so a runner can gain a knob without this service learning about it.
        self._runner_options = runner_options or (lambda: {})
        self._scratch_root = scratch_root
        self._max_output_bytes = max_output_bytes
        # The project's authoring factory (when the app was built with one), used to
        # promote a cell into a certified derivation via the same loop the chat uses.
        self._derive_factory = derive_factory
        self._live: dict[str, _Runtime] = {}
        self._live_lock = threading.Lock()
        # Pre-started kernels, for profiles that ask for them. A notebook with its own
        # scratch workspace cannot use one, since a pooled kernel is started before any
        # notebook is known.
        self._warm = _WarmPool(
            lambda profile, deps: self._make_kernel(deps, None, profile)
        )
        # Per-notebook ipywidgets comm subscribers (a browser tab's WebSocket sink). A
        # kernel forwards its comm traffic to a fan-out reaching every subscriber, so
        # widgets stay live across kernel restarts and multiple open tabs.
        self._comm_subscribers: dict[str, list[CommListener]] = {}

    # -- kernel lifecycle --------------------------------------------------------
    def _make_kernel(
        self, deps: tuple[str, ...], workspace: Path | None, profile: ComputeProfile
    ) -> Kernel:
        """Start a kernel on the backend, sized and bounded by ``profile``."""
        lazy = self._query_resolver is not None and self._dataset_names is not None
        # Eager only on the legacy path: reading every dataset here is what made kernel
        # start cost scale with bound data rather than with the notebook.
        datasets = None if lazy else self._load_datasets()
        names = list(self._dataset_names()) if lazy and self._dataset_names else None
        # A profile narrows the project's egress and never widens it, so a project that
        # denies the network cannot have a profile hand it back.
        profile = profile.narrowed_to(self._egress)
        credentials = self._credentials() if self._credentials else None
        common: dict[str, Any] = {
            "deps": deps,
            "workspace": workspace,
            "data": datasets,
            "dataset_names": names,
            "cell_timeout": profile.max_runtime,
            "max_output_bytes": self._max_output_bytes,
            "env": self._kernel_env() if self._kernel_env else None,
        }
        if self._backend == "kubernetes":
            from elbi_core.notebook.k8s_kernel import KubernetesNotebookKernel

            kernel: Kernel = KubernetesNotebookKernel(
                profile=profile,
                image=self._image,
                credentials=credentials,
                **self._runner_options(),
                **common,
            )
        elif self._backend == "docker":
            from elbi_core.notebook.container_kernel import DockerNotebookKernel

            kernel = DockerNotebookKernel(
                egress=profile.egress,
                image=profile.image or self._image,
                credentials=credentials,
                **docker_resource_args(profile),
                **common,
            )
        else:
            # The profile's egress decides here too, or a profile saying ``full`` is
            # quietly denied and what the kernel is meant to reach -- the deployment's
            # own tracking server -- fails as a connection error minutes later. Only
            # ``full`` opens it: a host allowlist needs the proxy the container backends
            # get, and approximating one as "anything" is a weaker control with the
            # same name.
            kernel = SubprocessKernel(allow_network=profile.egress == "full", **common)
        return kernel

    def _workspace(self, notebook_id: str) -> Path | None:
        """The per-notebook scratch dir, so files a cell writes persist across runs."""
        if self._scratch_root is None:
            return None
        path = self._scratch_root / "notebooks" / notebook_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def profiles(self) -> ComputeProfiles:
        """The compute menu this service offers, for the API to present and check."""
        return self._profiles

    def profile_for(self, notebook_id: str) -> ComputeProfile:
        """The profile a notebook runs under, falling back to the default.

        The profile belongs to the notebook rather than to whoever opens it: two people
        collaborating on one notebook otherwise get two different machines, and neither
        can reason about what the other saw.
        """
        return self._selected(notebook_id)[0]

    def _live_key(self, notebook_id: str) -> str:
        """Which entry in the live map this notebook's kernel is under.

        Its own, normally. A profile marked ``shared`` puts every notebook on it into
        one interpreter, so the key becomes the profile, and every lookup has to agree
        about that, or a restart would kill a kernel an interrupt could not find.
        """
        profile, _ = self._selected(notebook_id)
        return f"profile:{profile.name}" if profile.shared else notebook_id

    def _bind_query_resolver(self, kernel: Kernel, notebook_id: str) -> None:
        """Point a kernel's ``sql()`` at the warehouse, so a cell can query it.

        Without this, ``sql()`` in a cell has nothing to resolve against and the
        notebook is a longer way round to the data than the query API beside it.
        """
        if self._query_resolver is None:
            return
        resolver = self._query_resolver
        kernel.set_query_resolver(lambda sql, table, limit: resolver(sql, table, limit))

    def _idle_timeout(self, notebook_id: str, profile: ComputeProfile) -> float:
        """How long this notebook's kernel may sit idle.

        A notebook may ask for less than its profile allows and never more: an idle
        kernel costs whatever it is running on, so the profile sets the ceiling and the
        notebook's own setting only tightens it.
        """
        notebook = self._store.get_notebook(notebook_id)
        settings = json.loads(notebook.metadata_json or "{}") if notebook else {}
        raw = settings.get("idle_timeout")
        if not isinstance(raw, (int, float, str)):
            return profile.idle_timeout
        try:
            wanted = float(raw)
        except ValueError:
            # A notebook's metadata is user-editable, so a value that is not a number is
            # a typo rather than a request. Ignoring it beats a zero-second timeout that
            # reaps a kernel the moment it starts.
            return profile.idle_timeout
        return min(profile.idle_timeout, wanted) if wanted > 0 else profile.idle_timeout

    def _selected(self, notebook_id: str) -> tuple[ComputeProfile, str | None]:
        """The notebook's profile, and a note if it is not the one it asked for."""
        notebook = self._store.get_notebook(notebook_id)
        name = getattr(notebook, "compute_profile", None) if notebook else None
        try:
            return self._profiles.select(name), None
        except ComputeProfileError as exc:
            # The definition was removed or renamed underneath a notebook that named it.
            # Falling back keeps the notebook openable, but silently running something
            # other than what it asked for is how a resized limit goes unnoticed, so the
            # substitution is reported rather than absorbed.
            return self._profiles.select(None), str(exc)

    def _runtime(
        self,
        notebook_id: str,
        deps: Sequence[str],
        profile: ComputeProfile | None = None,
        kind: str = "interactive",
    ) -> _Runtime:
        """Return the notebook's live runtime, starting or restarting it as needed.

        A dependency change or a dead kernel means a fresh interpreter (its namespace is
        lost: unavoidable, and the kernel's declared environment is fixed at start). A
        profile change does *not* restart a live kernel: an admin who resizes a profile
        should not silently destroy the state of everyone currently working, so the new
        definition applies at the next start and the difference is reported until then.
        Idle kernels are swept first so an abandoned notebook does not hold a process.
        """
        self._evict_idle()
        want = tuple(deps)
        profile = profile or self.profile_for(notebook_id)
        # A shared profile keys its kernel by the profile rather than the notebook, so
        # every notebook on it lands in one interpreter. Opt-in for a reason: state one
        # notebook leaves behind is state the next one sees.
        key = f"profile:{profile.name}" if profile.shared else notebook_id
        with self._live_lock:
            runtime = self._live.get(key)
            if runtime is not None and (
                not runtime.kernel.alive or runtime.deps != want
            ):
                self._end_session(key, runtime)
                runtime = None
            if runtime is None:
                workspace = self._workspace(notebook_id)
                # Only a session that needs nothing notebook-specific can be served from
                # the pool, because a pooled kernel was started before this notebook was
                # known.
                kernel = None if workspace else self._warm.take(profile, want)
                if kernel is None:
                    kernel = self._make_kernel(want, workspace, profile)
                kernel.set_comm_listener(self._comm_fanout(notebook_id))
                # Bound here rather than at kernel creation: a pooled kernel starts
                # before any notebook is known, so it cannot carry the identity its
                # reads have to be authorized as.
                self._bind_query_resolver(kernel, notebook_id)
                now = time.monotonic()
                runtime = _Runtime(
                    kernel=kernel,
                    deps=want,
                    workspace=workspace,
                    profile=profile,
                    profile_version=profile.version,
                    idle_timeout=self._idle_timeout(notebook_id, profile),
                    kind=kind,
                    started_at=now,
                )
                self._live[key] = runtime
            runtime.last_used = time.monotonic()
            return runtime

    def _evict_idle(self) -> None:
        """Close kernels idle past their profile's timeout.

        Idle means no execution and nothing attached: a kernel running a cell is never
        reaped, however long it has been since the request that started it. Reaping is
        for reclaiming waste; killing a running cell destroys work instead.
        """
        now = time.monotonic()
        with self._live_lock:
            stale = [
                nb_id
                for nb_id, runtime in self._live.items()
                if not runtime.busy
                and not self._comm_subscribers.get(nb_id)
                and now - runtime.last_used > runtime.idle_timeout
            ]
            for nb_id in stale:
                self._end_session(nb_id, self._live.pop(nb_id))

    def runtime_state(self, notebook_id: str) -> dict[str, Any]:
        """What the notebook is running on, and whether that still matches the menu.

        ``drift`` is the Databricks compliance idea: after a profile definition changes,
        sessions already started keep the old one, and an admin who tightened a limit
        needs to be told that rather than assuming it took effect.
        """
        selected, unavailable = self._selected(notebook_id)
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        state: dict[str, Any] = {
            "profile": selected.to_dict(),
            "profile_version": selected.version,
            "unavailable": unavailable,
            "idle_timeout": self._idle_timeout(notebook_id, selected),
            "idle_timeout_cap": selected.idle_timeout,
            "status": (
                "idle" if runtime is None or not runtime.kernel.alive else "running"
            ),
            "drift": None,
        }
        if runtime is not None and runtime.kernel.alive:
            state["started_with"] = runtime.profile.to_dict()
            state["uptime_seconds"] = round(time.monotonic() - runtime.started_at, 1)
            if runtime.profile_version != selected.version:
                state["drift"] = (
                    f"running on {runtime.profile.name} as it was defined at version "
                    f"{runtime.profile_version}; the definition is now "
                    f"{selected.version}. Restart the kernel to pick it up."
                )
        return state

    def _end_session(self, notebook_id: str, runtime: _Runtime) -> None:
        """Close a kernel and attribute what it used.

        Attribution happens before the close, so a close that throws still leaves the
        session accounted for; the reverse would lose the record of the compute that was
        actually consumed.
        """
        if self._on_session_end is not None and runtime.started_at:
            elapsed = max(0.0, time.monotonic() - runtime.started_at)
            self._on_session_end(notebook_id, runtime.profile, elapsed, runtime.kind)
        runtime.kernel.close()

    def restart(self, notebook_id: str) -> None:
        """Kill a notebook's kernel; the next run starts a fresh one (state reset)."""
        with self._live_lock:
            runtime = self._live.pop(self._live_key(notebook_id), None)
        if runtime is not None:
            self._end_session(notebook_id, runtime)

    def interrupt(self, notebook_id: str) -> None:
        """Interrupt the notebook's currently running cell, keeping its namespace."""
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        if runtime is not None:
            runtime.kernel.interrupt()

    def kernel_status(self, notebook_id: str) -> str:
        """``running`` if a kernel is live for the notebook, else ``idle``."""
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        return "running" if runtime is not None and runtime.kernel.alive else "idle"

    def complete(self, notebook_id: str, code: str, cursor_pos: int) -> dict[str, Any]:
        """Completions from the notebook's live kernel (empty when none is running)."""
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        if runtime is None or not runtime.kernel.alive:
            return {"matches": [], "cursor_start": cursor_pos, "cursor_end": cursor_pos}
        return runtime.kernel.complete(code, cursor_pos)

    def inspect(
        self, notebook_id: str, code: str, cursor_pos: int, detail_level: int = 0
    ) -> dict[str, Any]:
        """Object help from the live kernel (not found when none is running)."""
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        if runtime is None or not runtime.kernel.alive:
            return {"found": False, "data": {}, "metadata": {}}
        return runtime.kernel.inspect(code, cursor_pos, detail_level)

    def variables(self, notebook_id: str) -> list[dict[str, str]]:
        """The user's data variables in the live kernel (empty when none is running)."""
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        if runtime is None or not runtime.kernel.alive:
            return []
        # variables() returns parsed introspection JSON (Any-valued); its "variables"
        # key holds the list of name/type/preview records this returns.
        variables = runtime.kernel.variables().get("variables", [])
        return cast("list[dict[str, str]]", variables)

    def send_input(self, notebook_id: str, value: str) -> bool:
        """Deliver a reply to a pending ``input()``; False if no kernel is live."""
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        if runtime is None or not runtime.kernel.alive:
            return False
        runtime.kernel.send_input(value)
        return True

    # -- ipywidgets comm ---------------------------------------------------------
    def _comm_fanout(self, notebook_id: str) -> CommListener:
        """A listener that relays a kernel's comm messages to every subscriber."""

        def fanout(message: dict[str, Any]) -> None:
            with self._live_lock:
                subscribers = list(self._comm_subscribers.get(notebook_id, ()))
            for send in subscribers:
                send(message)

        return fanout

    def comm_subscribe(
        self, notebook_id: str, send: CommListener
    ) -> Callable[[], None]:
        """Register a comm sink (a browser WebSocket); returns an unsubscribe fn."""
        with self._live_lock:
            self._comm_subscribers.setdefault(notebook_id, []).append(send)

        def unsubscribe() -> None:
            with self._live_lock:
                subscribers = self._comm_subscribers.get(notebook_id)
                if subscribers and send in subscribers:
                    subscribers.remove(send)

        return unsubscribe

    def comm_send(self, notebook_id: str, message: dict[str, Any]) -> None:
        """Forward a frontend comm message to the notebook's live kernel, if any."""
        with self._live_lock:
            runtime = self._live.get(self._live_key(notebook_id))
        if runtime is not None and runtime.kernel.alive:
            runtime.kernel.send_comm(message)

    def close(self) -> None:
        """Shut down every live kernel, and every kernel waiting in the pool."""
        with self._live_lock:
            for notebook_id, runtime in self._live.items():
                self._end_session(notebook_id, runtime)
            self._live.clear()
        self._warm.close()

    # -- environment -------------------------------------------------------------
    def base_environments(self) -> list[dict[str, Any]]:
        """The workspace's curated base environments, with their support windows.

        A base environment is an admin-defined package baseline a notebook layers its
        own dependencies on, so a common stack resolves and caches once instead of per
        notebook. Databricks serverless has the same idea under the same name.

        It is a consistency and startup-time mechanism, not a security boundary: nothing
        here verifies or allowlists what a base environment contains, and a notebook is
        free to add to it. What makes a resolved set trustworthy is the lock recorded
        against the notebook, not its membership of a base environment.

        ``end_of_support`` is the part a practitioner plans around. A lock says what you
        got; it does not say when it stops being maintained. Databricks pairs each
        serverless environment version with a dated end of support and, past it, stops
        offering that version for new work while existing workloads keep running, which
        is what ``expired`` reports here.
        """
        raw = self._store.get_config(BASE_ENV_SETTING)
        if not raw:
            return []
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            return []
        today = dt.date.today()
        environments = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            ends = str(entry.get("end_of_support") or "") or None
            expired = False
            if ends:
                try:
                    expired = dt.date.fromisoformat(ends) < today
                except ValueError:
                    # A malformed date is not an expiry: treating it as one would take
                    # an environment out of service over a typo in a settings field.
                    ends = None
            environments.append(
                {
                    "name": str(entry["name"]),
                    "deps": list(entry.get("deps", [])),
                    "version": str(entry.get("version") or ""),
                    "end_of_support": ends,
                    "expired": expired,
                }
            )
        return environments

    def selectable_environments(self) -> list[dict[str, Any]]:
        """Base environments a notebook may newly choose: everything still supported.

        An expired environment is neither deleted nor stopped: a notebook already on
        one keeps running, exactly as Databricks does at end of support. It simply stops
        being offered for new work, which is the only way an upgrade gets planned.
        """
        return [env for env in self.base_environments() if not env["expired"]]

    def _base_env_deps(self, name: str | None) -> list[str]:
        """The packages of the named base environment, or none when unset/unknown.

        Resolved from every environment rather than only the supported ones: a notebook
        already on an expired environment must keep running the packages it was pinned
        to, or end of support would silently become removal.
        """
        if not name:
            return []
        for env in self.base_environments():
            if env["name"] == name:
                return list(env["deps"])
        return []

    def _effective_deps(self, deps: Sequence[str], base_env: str | None) -> list[str]:
        """A notebook's full spec: its base environment plus its own packages."""
        return _dedupe([*self._base_env_deps(base_env), *deps])

    def _provision_deps(self, row: Any) -> list[str]:
        """Packages to start a kernel with: the pinned lock if present, else the spec.

        Provisioning from the lock is what makes a rerun reproducible, the same resolved
        versions every time, while an unlocked notebook still runs from its loose spec.
        """
        if row.lock_json:
            return list(json.loads(row.lock_json))
        metadata = json.loads(row.metadata_json or "{}")
        return self._effective_deps(
            json.loads(row.deps_json or "[]"), metadata.get("base_env")
        )

    def set_environment(
        self,
        notebook_id: str,
        deps: Sequence[str] | None = None,
        base_env: str | None = None,
        add_deps: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Change a notebook's environment, re-lock it, and restart its kernel.

        Setting the loose ``deps`` (or layering ``add_deps`` from a ``%pip`` cell) or
        the ``base_env`` re-resolves the effective spec to a pinned lock via uv and
        restarts the kernel so the new environment is provisioned from clean state: the
        front-load pattern that avoids the mid-session ``restartPython`` surprise. A
        failed resolution (a bad package) is reported and leaves the notebook on its
        prior lock.
        """
        row = self._store.get_notebook(notebook_id)
        if row is None:
            return {"ok": False, "error": "notebook not found"}
        metadata = json.loads(row.metadata_json or "{}")
        current = json.loads(row.deps_json or "[]")
        new_deps = list(deps) if deps is not None else current
        if add_deps:
            new_deps = _dedupe([*current, *add_deps])
        new_base = base_env if base_env is not None else metadata.get("base_env")
        # Moving *onto* an environment past its end of support is refused; staying on
        # one is not. That asymmetry is the mechanism: an expired environment keeps
        # working for what already uses it and stops taking on new work, which is what
        # makes an upgrade plannable rather than a surprise.
        if new_base and new_base != metadata.get("base_env"):
            expired = next(
                (
                    env
                    for env in self.base_environments()
                    if env["name"] == new_base and env["expired"]
                ),
                None,
            )
            if expired is not None:
                return {
                    "ok": False,
                    "error": (
                        f"base environment {new_base!r} reached end of support on "
                        f"{expired['end_of_support']} and cannot be selected for new "
                        "work. Notebooks already using it keep running."
                    ),
                }

        fields: dict[str, Any] = {}
        if new_deps != current:
            fields["deps_json"] = json.dumps(new_deps)
        if new_base != metadata.get("base_env"):
            metadata["base_env"] = new_base or None
            fields["metadata_json"] = json.dumps(metadata)
        if not fields and row.lock_json is not None:
            return {"ok": True, "deps": new_deps, "lock": json.loads(row.lock_json)}

        effective = self._effective_deps(new_deps, new_base)
        lock: list[str] | None = None
        error: str | None = None
        try:
            lock = resolve_lock(effective)
            fields["lock_json"] = json.dumps(lock)
        except DerivationError as exc:
            error = str(exc)  # keep the prior lock; the notebook stays runnable
            logger.warning("locking notebook %s failed: %s", notebook_id, error)
        if fields:
            self._store.update_notebook(notebook_id, **fields)
        self.restart(notebook_id)
        return {"ok": error is None, "deps": new_deps, "lock": lock, "error": error}

    def relock(self, notebook_id: str) -> dict[str, Any]:
        """Re-resolve the current spec to a fresh lock (pick up newer versions)."""
        row = self._store.get_notebook(notebook_id)
        if row is None:
            return {"ok": False, "error": "notebook not found"}
        metadata = json.loads(row.metadata_json or "{}")
        effective = self._effective_deps(
            json.loads(row.deps_json or "[]"), metadata.get("base_env")
        )
        try:
            lock = resolve_lock(effective)
        except DerivationError as exc:
            return {"ok": False, "error": str(exc)}
        self._store.update_notebook(notebook_id, lock_json=json.dumps(lock))
        self.restart(notebook_id)
        return {"ok": True, "lock": lock}

    # -- documents ---------------------------------------------------------------
    def _document(self, notebook_id: str) -> NotebookDoc:
        """Rebuild the nbformat document for a notebook from its stored cells."""
        cells = [
            Cell(
                id=row.id,
                cell_type=row.cell_type,  # type: ignore[arg-type]
                source=row.source,
                metadata=json.loads(row.metadata_json or "{}"),
                outputs=json.loads(row.outputs_json or "[]"),
                execution_count=row.execution_count,
            )
            for row in self._store.get_cells(notebook_id)
        ]
        return NotebookDoc(cells=cells)

    def graph_for(self, notebook_id: str) -> DependencyGraph:
        """The dataflow graph over the notebook's current code-cell sources."""
        return DependencyGraph(self._document(notebook_id).dependency_pairs())

    def _stale(
        self, runtime: _Runtime, doc: NotebookDoc, graph: DependencyGraph
    ) -> set[str]:
        """Run cells whose result is out of date (edited, or an input re-ran)."""
        source = {cell.id: cell.source for cell in doc.code_cells()}
        stale: set[str] = set()
        for cell_id in graph.topo_order():
            if cell_id not in runtime.run_seq:
                continue  # never run: pending, not stale
            if _hash(source.get(cell_id, "")) != runtime.ran.get(cell_id):
                stale.add(cell_id)
                continue
            for producer in graph.upstream.get(cell_id, ()):
                later = runtime.run_seq.get(producer, -1) > runtime.run_seq[cell_id]
                if producer in stale or later:
                    stale.add(cell_id)
                    break
        return stale

    def view(self, notebook_id: str) -> dict[str, Any] | None:
        """The notebook payload for the editor: metadata, cells, and the dep graph."""
        row = self._store.get_notebook(notebook_id)
        if row is None:
            return None
        graph = self.graph_for(notebook_id)
        cells = self._store.get_cells(notebook_id)
        metadata = json.loads(row.metadata_json or "{}")
        return {
            "id": row.id,
            "name": row.name,
            "copied_from": row.copied_from,
            "deps": json.loads(row.deps_json or "[]"),
            "metadata": metadata,
            "schedule": json.loads(row.schedule_json) if row.schedule_json else None,
            "kernel_status": self.kernel_status(notebook_id),
            "cells": [self._cell_view(cell) for cell in cells],
            "graph": self._graph_view(graph),
            "environment": {
                "base_env": metadata.get("base_env"),
                "base_environments": [e["name"] for e in self.base_environments()],
                "lock": json.loads(row.lock_json) if row.lock_json else None,
                "locked": row.lock_json is not None,
            },
        }

    def _cell_view(self, cell: NotebookCell) -> dict[str, Any]:
        """A cell as the editor consumes it (outputs and metadata parsed from JSON)."""
        return {
            "id": cell.id,
            "cell_type": cell.cell_type,
            "source": cell.source,
            "metadata": json.loads(cell.metadata_json or "{}"),
            "outputs": json.loads(cell.outputs_json or "[]"),
            "execution_count": cell.execution_count,
        }

    def _graph_view(self, graph: DependencyGraph) -> dict[str, Any]:
        """The dependency graph as JSON: per-cell defs/refs/edges, conflicts, cycles."""
        return {
            "cells": {
                cell_id: {
                    "defs": sorted(graph.deps[cell_id].defs),
                    "refs": sorted(graph.deps[cell_id].refs),
                    "upstream": sorted(graph.upstream.get(cell_id, ())),
                    "downstream": sorted(graph.downstream(cell_id)),
                    "syntax_error": graph.deps[cell_id].syntax_error,
                }
                for cell_id in graph.order
            },
            "conflicts": graph.conflicts,
            "cycle": sorted(graph.cycle_members()),
        }

    # -- execution ---------------------------------------------------------------
    def run_events(
        self, notebook_id: str, cell_ids: Sequence[str]
    ) -> Iterator[dict[str, Any]]:
        """Run cells (and, reactively, their dependents), yielding SSE-ready events.

        ``cell_ids`` are the roots the user asked to run; the reactive plan expands them
        to
        every transitive dependent, ordered so no cell runs before its inputs. Each cell
        emits a ``cell_start``, then its outputs as they stream, then a ``status``
        carrying
        the execution count and the recomputed stale set; a final ``done`` ends the
        stream.
        """
        row = self._store.get_notebook(notebook_id)
        if row is None:
            yield {"event": "error", "message": "notebook not found"}
            return
        metadata = json.loads(row.metadata_json or "{}")
        reactive = bool(metadata.get("reactive", True))
        doc = self._document(notebook_id)
        graph = self.graph_for(notebook_id)
        sources = {cell.id: cell.source for cell in doc.code_cells()}

        roots = [cid for cid in cell_ids if cid in sources]
        plan = graph.run_plan(roots) if reactive else list(roots)

        # A ``%pip install`` cell in the run is an environment change, not code: apply
        # the packages, re-lock, and restart before running, so cells execute against
        # the new environment (front-loaded, so any state reset happens before real
        # work).
        yield from self._apply_run_installs(notebook_id, plan, sources)
        row = self._store.get_notebook(notebook_id) or row
        runtime = self._runtime(notebook_id, self._provision_deps(row))

        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def work() -> None:
            try:
                with runtime.lock:
                    for cell_id in plan:
                        self._run_one(runtime, cell_id, sources[cell_id], events)
                    stale = self._stale(runtime, doc, graph)
                    events.put({"event": "stale", "cells": sorted(stale)})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield item
        yield {"event": "done"}

    def _apply_run_installs(
        self, notebook_id: str, plan: Sequence[str], sources: Mapping[str, str]
    ) -> Iterator[dict[str, Any]]:
        """Apply any ``%pip install`` cells in the plan to the environment first.

        Collects the packages every install cell in the run declares, adds the new ones
        to the notebook's spec, re-locks, and restarts: yielding
        ``installing``/``installed`` events so the client sees the environment change
        stream before its cells run.
        """
        wanted = _dedupe(
            [spec for cid in plan for spec in _install_specs(sources.get(cid, ""))]
        )
        if not wanted:
            return
        row = self._store.get_notebook(notebook_id)
        existing = set(json.loads(row.deps_json or "[]")) if row else set()
        fresh = [spec for spec in wanted if spec not in existing]
        if not fresh:
            return
        yield {"event": "installing", "packages": fresh}
        result = self.set_environment(notebook_id, add_deps=fresh)
        yield {
            "event": "installed",
            "packages": fresh,
            "ok": result["ok"],
            "error": result.get("error"),
        }

    def _run_one(
        self,
        runtime: _Runtime,
        cell_id: str,
        source: str,
        events: queue.Queue[dict[str, Any] | None],
    ) -> None:
        """Execute a cell, stream and persist its outputs, and update run state."""
        runtime.counter += 1
        count = runtime.counter
        runtime.last_used = time.monotonic()
        events.put({"event": "cell_start", "cell": cell_id, "execution_count": count})

        # An install cell is an environment directive, not code; the packages were
        # applied before the run. Show what it installed rather than running it.
        installs = _install_specs(source)
        if installs:
            note = "✓ installed into the notebook environment: " + ", ".join(installs)
            output = {"output_type": "stream", "name": "stdout", "text": note + "\n"}
            events.put({"event": "output", "cell": cell_id, "output": output})
            self._store.save_cell_result(cell_id, json.dumps([output]), count)
            events.put(
                {
                    "event": "status",
                    "cell": cell_id,
                    "status": "ok",
                    "execution_count": count,
                }
            )
            return

        collected: list[dict[str, Any]] = []

        def on_output(message: Mapping[str, Any]) -> None:
            if message.get("type") == "input_request":
                # A control message, not a cell output: the frontend prompts for a line
                # and replies via the input route, which sends it to the kernel's stdin.
                events.put(
                    {
                        "event": "input_request",
                        "cell": cell_id,
                        "prompt": message.get("prompt", ""),
                        "password": bool(message.get("password")),
                    }
                )
                return
            output = _to_output(message)
            collected.append(output)
            events.put({"event": "output", "cell": cell_id, "output": output})

        status = runtime.kernel.execute(source, count, on_output)
        outputs = _coalesce(collected)
        # Which side of the data boundary the cell landed on, recorded per run so the
        # editor can show it. Hex makes this a pill on the cell, and the reason to copy
        # that is the reason the split is explicit at all: a crossing nobody can see
        # only moves the memory failure later.
        self._store.save_cell_result(
            cell_id,
            json.dumps(outputs),
            count,
            data_mode=runtime.kernel.last_data_mode,
            queries=list(runtime.kernel.last_queries),
        )
        if status == "ok":
            runtime.ran[cell_id] = _hash(source)
            runtime.run_seq[cell_id] = count
        events.put(
            {
                "event": "status",
                "cell": cell_id,
                "status": status,
                "execution_count": count,
                "data_mode": runtime.kernel.last_data_mode,
                "queries": list(runtime.kernel.last_queries),
            }
        )

    def run_all_events(
        self, notebook_id: str, fresh: bool = False
    ) -> Iterator[dict[str, Any]]:
        """Run every code cell in dependency order; ``fresh`` restarts the kernel."""
        if fresh:
            self.restart(notebook_id)
        graph = self.graph_for(notebook_id)
        return self.run_events(notebook_id, graph.topo_order())

    def run_parameterized_events(
        self, notebook_id: str, overrides: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """Run the whole notebook top-to-bottom on a fresh kernel with parameters set.

        The overrides splice in as an injected-parameters cell right after the cell
        tagged ``parameters`` (papermill's model), then every code cell runs in order so
        the run is reproducible for the inputs. The injected cell executes for its
        effect on the namespace but is not persisted (it belongs to the run only).
        """
        row = self._store.get_notebook(notebook_id)
        if row is None:
            yield {"event": "error", "message": "notebook not found"}
            return
        self.restart(notebook_id)
        # Batch: nothing is watching, so this session both enforces the current profile
        # immediately and is attributed separately from someone's open notebook.
        runtime = self._runtime(notebook_id, self._provision_deps(row), kind="batch")
        effective = apply_overrides(self._document(notebook_id), overrides)
        code_cells = [(c.id, c.source) for c in effective.code_cells()]

        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def work() -> None:
            try:
                with runtime.lock:
                    for cell_id, source in code_cells:
                        self._run_one(runtime, cell_id, source, events)
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield item
        yield {"event": "done"}

    def run_scheduled(
        self, notebook_id: str, overrides: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run a notebook to completion for a scheduled tick; return a run summary.

        Drains a parameterized run synchronously (no client is watching), reporting how
        many cells ran and which failed, so the scheduler can log the outcome.
        """
        statuses: dict[str, str] = {}
        for event in self.run_parameterized_events(notebook_id, overrides):
            if event["event"] == "status":
                statuses[event["cell"]] = event["status"]
        failed = [cell for cell, status in statuses.items() if status != "ok"]
        return {"ran": len(statuses), "failed": failed, "ok": not failed}

    # -- bridges to derivations --------------------------------------------------
    def promote_cell(self, notebook_id: str, cell_id: str) -> dict[str, Any]:
        """Author a cell's derivation function as a governed, certified derivation.

        The cell must define a top-level ``def <name>(ctx): ...``: the derivation shape,
        reading inputs via ``ctx.input(...)``. Promoting runs the project's authoring
        loop (sandboxed run, oracle/golden verification, certification), the same path
        the chat and MCP ``propose_derivation`` use, so a notebook is a place to build a
        derivation, never an ungoverned way around one. Returns the outcome.
        """
        row = self._store.get_notebook(notebook_id)
        if row is None:
            return {"ok": False, "error": "notebook not found"}
        if self._derive_factory is None:
            return {"ok": False, "error": "derivation authoring is not configured"}
        cell = self._store.get_cell(cell_id)
        if cell is None or cell.notebook_id != notebook_id:
            return {"ok": False, "error": "cell not found"}
        name = _derivation_name(cell.source)
        if name is None:
            return {
                "ok": False,
                "error": "promote a cell that defines `def <name>(ctx): ...`: the "
                "derivation reads its inputs via ctx.input(...) and returns the result",
            }
        deps = json.loads(row.deps_json or "[]")
        derive = self._derive_factory(
            f"notebook:{notebook_id}", f"Promoted from notebook cell {cell_id}"
        )
        outcome = derive(name, cell.source, None, None, "table", (), deps)
        return {
            "ok": outcome.certified,
            "name": name,
            "certified": outcome.certified,
            "verdict": outcome.verdict,
            "rendered": outcome.rendered,
            "error": outcome.error,
            "detail": getattr(outcome, "detail", None),
        }

    def create(self, name: str) -> str:
        """Create an empty notebook and return its id."""
        return self._store.create_notebook(name)

    def summaries(self) -> list[dict[str, Any]]:
        """Notebook summaries (id, name, cell count) for listing."""
        return self._store.list_notebooks()

    def set_cells(self, notebook_id: str, cells: Sequence[Mapping[str, Any]]) -> None:
        """Replace a notebook's cells with ``{cell_type, source}`` objects, in order."""
        from elbi_core.notebook import new_id

        rows = []
        for position, cell in enumerate(cells):
            cell_type = str(cell.get("cell_type") or "code")
            if cell_type not in ("code", "markdown", "raw"):
                cell_type = "code"
            rows.append(
                NotebookCell(
                    id=new_id(),
                    notebook_id=notebook_id,
                    position=position,
                    cell_type=cell_type,
                    source=str(cell.get("source") or ""),
                )
            )
        self._store.replace_cells(notebook_id, rows)

    def set_schedule(
        self, notebook_id: str, schedule: Mapping[str, Any] | None
    ) -> None:
        """Set (or clear) a notebook's rerun schedule."""
        self._store.update_notebook(
            notebook_id, schedule_json=json.dumps(dict(schedule)) if schedule else None
        )

    def create_from_source(
        self, name: str, source: str, heading: str | None = None
    ) -> str:
        """Create a notebook seeded with a code cell (and an optional markdown heading).

        The scaffold for "open this in a notebook" flows (editing a derivation, or
        reproducing a model's training run), so the code lands ready to run and iterate.
        """
        from elbi_core.notebook import new_id

        notebook_id = self._store.create_notebook(name)
        seeded = self._store.get_cells(notebook_id)[0].id
        cells: list[NotebookCell] = []
        if heading:
            cells.append(
                NotebookCell(
                    id=seeded,
                    notebook_id=notebook_id,
                    position=0,
                    cell_type="markdown",
                    source=heading,
                )
            )
        cells.append(
            NotebookCell(
                id=new_id() if heading else seeded,
                notebook_id=notebook_id,
                position=len(cells),
                cell_type="code",
                source=source,
            )
        )
        self._store.replace_cells(notebook_id, cells)
        return notebook_id

    def create_from_sources(
        self,
        name: str,
        sources: Sequence[str],
        heading: str | None = None,
        setup: str | None = None,
    ) -> str:
        """Create a notebook seeded with several code cells, in the order given.

        The single-cell :meth:`create_from_source` cannot express "this needs that
        bound first", which a derivation reading an upstream derivation requires.

        ``setup`` is seeded as a hidden cell before them: it runs, binding what the
        visible cells reference, but the editor does not draw it. Imports and a data
        contract are environment, not the thing being edited.
        """
        from elbi_core.notebook import new_id

        notebook_id = self._store.create_notebook(name)
        seeded = self._store.get_cells(notebook_id)[0].id
        cells: list[NotebookCell] = []
        if heading:
            cells.append(
                NotebookCell(
                    id=seeded,
                    notebook_id=notebook_id,
                    position=0,
                    cell_type="markdown",
                    source=heading,
                )
            )
        if setup:
            cells.append(
                NotebookCell(
                    id=new_id() if cells else seeded,
                    notebook_id=notebook_id,
                    position=len(cells),
                    cell_type="code",
                    source=setup,
                    metadata_json=json.dumps({"elbi": {"role": SETUP_ROLE}}),
                )
            )
        for source in sources:
            cells.append(
                NotebookCell(
                    id=new_id() if cells else seeded,
                    notebook_id=notebook_id,
                    position=len(cells),
                    cell_type="code",
                    source=source,
                )
            )
        self._store.replace_cells(notebook_id, cells)
        return notebook_id

    def duplicate(self, notebook_id: str) -> str | None:
        """Copy a notebook into a new one; ``None`` if the original is not found.

        A duplicate is a new notebook, not a new version: it gets its own id and starts
        with a clean history.

        Cell outputs and the rerun schedule are left behind: a copy has not run, and
        inheriting the schedule would fire a cron nobody set. The environment lock *is*
        carried, so the copy resolves the same package versions rather than drifting on
        its first run. Cell ids are minted fresh: the store keys every cell globally, so
        reusing them would collide with the source.
        """
        from elbi_core.notebook import new_id

        row = self._store.get_notebook(notebook_id)
        if row is None:
            return None
        taken = {summary["name"] for summary in self._store.list_notebooks()}
        folder_id = row.folder_id
        new_notebook_id = self._store.create_notebook(
            copy_label(row.name, taken), folder_id=folder_id
        )
        self._store.replace_cells(
            new_notebook_id,
            [
                NotebookCell(
                    id=new_id(),
                    notebook_id=new_notebook_id,
                    position=position,
                    cell_type=cell.cell_type,
                    source=cell.source,
                    metadata_json=cell.metadata_json,
                )
                for position, cell in enumerate(self._store.get_cells(notebook_id))
            ],
        )
        self._store.update_notebook(
            new_notebook_id,
            deps_json=row.deps_json,
            metadata_json=row.metadata_json,
            lock_json=row.lock_json,
            compute_profile=row.compute_profile,
            copied_from=notebook_id,
        )
        return new_notebook_id

    def create_from_derivation(self, name: str) -> str | None:
        """Open a stored derivation's source in a new notebook, ready to run.

        Editing a certified derivation is a first-class task; this scaffolds a notebook
        with the derivation's code as a code cell (plus a heading), so a change can be
        explored here and re-promoted as a new version.

        A derivation's stored source is captured with ``inspect.getsource(fn)`` — the
        decorated function and nothing else, because imports live at module level.
        Pasted into a notebook alone it cannot run: the kernel binds ``data`` and
        ``sql`` and no more, so the first line dies on ``@derivation``. The scaffold
        seeds what the source needs before it: the SDK names it references, and any
        upstream derivation it reads, defined before the cell that reads it.

        ``None`` when there is no derivation of that name.
        """
        derivation = self._store.get_derivation(name)
        if derivation is None:
            return None
        heading = f"# Editing derivation `{name}`\n\n{derivation.question}".rstrip()
        from elbi_core._source import split_stored

        _, own = split_stored(derivation.source)
        setup, bodies = self._derivation_prelude(derivation.source)
        return self.create_from_sources(
            f"Editing {name}",
            [*bodies, own],
            heading=heading,
            setup=setup,
        )

    def _derivation_prelude(self, source: str) -> tuple[str, list[str]]:
        """What a derivation needs bound before it: its environment, then its upstreams.

        Each stored source is self-contained — it carries the imports, constants and
        helpers its module gave it — so seeding a derivation beside its upstream would
        repeat that context in every cell. It is split back out here and merged into one
        environment, leaving each visible cell as just a derivation.

        The closure matters, not just the one source: a derivation reading an upstream
        needs that upstream *and* whatever the upstream itself references.
        """
        from elbi_core._source import _free_names, split_stored

        preamble: list[str] = []
        bodies: list[str] = []
        seen: set[str] = set()

        def add(statements: list[str]) -> None:
            """Keep each distinct statement once, in the order first seen."""
            for statement in statements:
                if statement not in preamble:
                    preamble.append(statement)

        def walk(code: str) -> None:
            """Seed a source's upstreams depth-first, then take its own context."""
            statements, own = split_stored(code)
            names = _free_names(own) | _free_names("\n".join(statements))
            for ref in sorted(names):
                if ref in seen:
                    continue
                row = self._store.get_derivation(ref)
                if row is None:
                    continue
                # Marked before recursing, so a cycle terminates rather than seeding
                # each derivation forever.
                seen.add(ref)
                walk(row.source)
                inner_pre, inner_own = split_stored(row.source)
                add(inner_pre)
                bodies.append(inner_own)
            add(statements)

        walk(source)
        environment = " \n".join(preamble).replace(" \n", "\n") if preamble else ""
        return environment, bodies

    # -- interchange -------------------------------------------------------------
    def export_ipynb(
        self, notebook_id: str, include_outputs: bool = False
    ) -> dict[str, Any] | None:
        """The notebook as a nbformat 4.5 object (a writable ``.ipynb`` payload).

        The notebook's packages and rerun schedule ride in ``metadata.elbi`` so the
        ``.ipynb`` fully round-trips through ``elbi sync``: runtime state (last run /
        hash) is dropped so the repo form is stable.

        **Cell outputs are dropped unless asked for.** An output is a rendering of
        warehouse rows, so an export carrying them puts data wherever the file lands,
        which for ``elbi pull`` is somebody's laptop. Stripping costs nothing
        there: a notebook is executed on the deployment and never locally, so an output
        has no use on the far side.

        Off by default because that is the polarity every comparable tool chose.
        ``nbstripout`` exists to keep outputs out of version control, citing sensitive
        information among its reasons, and Databricks Git folders will not commit
        ``.ipynb`` output unless a workspace administrator enables it. The equivalent
        switch here is ``NOTEBOOK_EXPORT_OUTPUTS``, read by the route, so whether an
        output may leave stays a deployment's decision and not a caller's.
        """
        row = self._store.get_notebook(notebook_id)
        if row is None:
            return None
        doc = self._document(notebook_id)
        doc.metadata = json.loads(row.metadata_json or "{}")
        if not include_outputs:
            doc.clear_outputs()
        payload = doc.to_ipynb()
        config: dict[str, Any] = {}
        deps = json.loads(row.deps_json or "[]")
        if deps:
            config["deps"] = deps
        schedule = json.loads(row.schedule_json or "null")
        if isinstance(schedule, dict):
            config["schedule"] = {
                k: schedule[k]
                for k in (
                    "enabled",
                    "mode",
                    "cron",
                    "timezone",
                    "interval_hours",
                    "dataset",
                    "params",
                )
                if k in schedule
            }
        metadata = payload.setdefault("metadata", {})
        if config:
            metadata["elbi"] = config
        else:
            metadata.pop("elbi", None)
        return payload

    def import_ipynb(self, data: Mapping[str, Any], name: str) -> str:
        """Create a notebook from a parsed ``.ipynb`` object; return its id.

        For a notebook that already exists, use :meth:`replace_ipynb` instead: creating
        a replacement and deleting the old one changes the notebook's identity, and its
        folder is part of that identity.
        """
        notebook_id = self._store.create_notebook(name)
        self._apply_ipynb(notebook_id, data)
        return notebook_id

    def replace_ipynb(self, notebook_id: str, data: Mapping[str, Any]) -> bool:
        """Replace an existing notebook's contents in place; ``False`` if not found.

        In place is the whole point. The obvious alternative -- delete it and import the
        file as a new notebook -- produces the right cells and quietly destroys
        everything else the row carried:

        * **The folder.** A notebook reappearing at the root has silently left the
          folder it was organised into, and nothing reports the move.
        * **The id.** Anything pointing at the notebook -- a dashboard tile, a lineage
          edge, a saved link -- refers to a notebook that no longer exists.
        * **``created_at``**, so its history restarts.

        It is also atomic where delete-then-create is not: a failure part way through
        leaves the notebook as it was rather than deleted.
        """
        if self._store.get_notebook(notebook_id) is None:
            return False
        self._apply_ipynb(notebook_id, data)
        return True

    def _apply_ipynb(self, notebook_id: str, data: Mapping[str, Any]) -> None:
        """Write a parsed ``.ipynb`` over ``notebook_id``'s cells and configuration.

        Packages and a rerun schedule declared under ``metadata.elbi`` are
        applied, so ``elbi sync`` fully configures a notebook from its file.

        Cell ids are minted fresh rather than taken from the file. In nbformat an id is
        document-local, but the store keys every cell by it globally, so importing an
        exported notebook -- the same file pulled then pushed under a new name, or one
        copied between notebooks -- would collide with the cells it came from.
        """
        doc = NotebookDoc.from_ipynb(data)
        for cell in doc.cells:
            cell.id = uuid4().hex
        config = {}
        raw_meta = data.get("metadata")
        if isinstance(raw_meta, dict):
            candidate = raw_meta.get("elbi")
            if isinstance(candidate, dict):
                config = candidate
        doc.metadata.pop("elbi", None)  # app config lives in its own columns
        cells = [
            NotebookCell(
                id=cell.id,
                notebook_id=notebook_id,
                position=position,
                cell_type=cell.cell_type,
                source=cell.source,
                metadata_json=json.dumps(cell.metadata),
                outputs_json=json.dumps(cell.outputs),
                execution_count=cell.execution_count,
            )
            for position, cell in enumerate(doc.cells)
        ]
        self._store.replace_cells(notebook_id, cells)
        self._store.update_notebook(notebook_id, metadata_json=json.dumps(doc.metadata))
        deps = config.get("deps")
        if isinstance(deps, list):
            self._store.update_notebook(
                notebook_id, deps_json=json.dumps([str(d) for d in deps])
            )
        schedule = config.get("schedule")
        if isinstance(schedule, dict):
            self.set_schedule(notebook_id, schedule)
