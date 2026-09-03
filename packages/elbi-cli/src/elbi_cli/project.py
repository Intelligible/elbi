"""Loading a project into a registry + runner factory.

Shared by the ``dev`` and ``validate`` commands: locate ``elbi.yaml``,
discover derivations, and produce a factory that builds a
:class:`~elbi.Runner` bound to a persistent on-disk cache (under
``.elbi/cache``), so results survive across calls and restarts.
"""

from __future__ import annotations

import getpass
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from elbi_core import (
    Bm25Retriever,
    DockerExecutor,
    Executor,
    Registry,
    RoutingExecutor,
    Runner,
    SubprocessExecutor,
)
from elbi_core.cache import LocalCacheStore
from elbi_core.config import (
    DEV_BINDINGS_FILENAME,
    PROJECT_FILENAME,
    DataBindings,
    ProjectConfig,
)
from elbi_core.data import Table
from elbi_core.derivation import Status
from elbi_core.discovery import discover
from elbi_core.errors import ConfigError
from elbi_core.k8s_executor import KubernetesExecutor
from elbi_core.sandbox import ComputeProfile, docker_resource_args
from elbi_core.tracking import JsonlRunLog

from .authored import AUTHORED_DIRNAME, AuthoredStore
from .lifecycle import LIFECYCLE_FILENAME, LifecycleStore

#: Where a project's mutable state lives, relative to the project root by default:
#: the derivation cache, chat workspaces, the authored store, the run and audit logs.
CACHE_DIRNAME = ".elbi"

#: Overrides where that mutable state lives, so the project itself can be read-only.  A
#: deployment that delivers its project immutably needs this. The source may be a
#: directory baked into the container image, or a worktree a git-sync sidecar owns and
#: rewrites on every new commit: in both cases writing cache and logs into it is either
#: impossible or gets discarded. Pointing state at a volume separates what is delivered
#: from what accumulates, which is the same split Airflow draws between synced DAGs and
#: everything the scheduler writes.
STATE_DIR_ENV = "ELBI_STATE_DIR"

#: Where per-conversation scratch workspaces live within a project. Each holds the
#: files a chat's ``run_code`` calls write, so exploration state persists across calls.
WORKSPACES_DIRNAME = "workspaces"

#: Sandbox wall-clock budget for authoring a derivation. Generous because a derivation
#: may fit a model (cross-validated ensembles on tens of thousands of rows), which the
#: 30s library default cuts short; still bounded so a runaway cannot block forever.
SANDBOX_TIMEOUT_SECONDS = 300.0


def sandbox_executor(
    backend: str,
    image: str | None = None,
    profile: ComputeProfile | None = None,
    namespace: str | None = None,
) -> Executor:
    """Build the sandbox executor for the configured backend.

    ``"subprocess"`` is the dependency-free default (a hardened host subprocess);
    ``"docker"`` runs agent source in an isolated Linux container (``image`` selects it,
    defaulting to the slim base). ``profile`` sizes the container, so agent-written
    candidate code is bounded by the same menu a notebook picks from rather than by a
    number written into the library.

    ``"kubernetes"`` runs each candidate as a Job on the same runner notebook kernels
    use, which is where the plan says candidate code belongs. Anything else is refused
    rather than quietly treated as ``"subprocess"``: that fallback is the dangerous one,
    because a deployment which selected per-pod isolation would get agent-written code
    running as a child of the app process, weaker than it asked for and invisible unless
    somebody reads this function.
    """
    if backend == "docker":
        limits = docker_resource_args(profile) if profile else {}
        return DockerExecutor(
            timeout=SANDBOX_TIMEOUT_SECONDS,
            image=image,
            **{k: v for k, v in limits.items() if k != "gpus"},
        )
    if backend == "kubernetes":
        return KubernetesExecutor(
            timeout=SANDBOX_TIMEOUT_SECONDS,
            image=image,
            namespace=namespace,
        )
    if backend not in ("subprocess", ""):
        raise ConfigError(
            f"there is no {backend!r} executor for agent-written code; expected "
            "'kubernetes' (a Job per run), 'docker' (a container per run) or "
            "'subprocess' (a hardened child of the app)."
        )
    return SubprocessExecutor(timeout=SANDBOX_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class LoadedProject:
    """A discovered project ready to serve or validate."""

    config: ProjectConfig
    registry: Registry
    base_dir: Path
    derivations_dir: Path
    bindings_path: Path
    #: Where everything this project writes goes. ``base_dir / '.elbi'`` unless
    #: the deployment separates them; see :data:`STATE_DIR_ENV`.
    state_dir: Path
    cache_dir: Path
    workspaces_dir: Path
    discovered: tuple[str, ...]
    authored: tuple[str, ...]
    make_runner: Callable[..., Runner]
    load_dataset: Callable[[str], Table]
    lifecycle: LifecycleStore
    authored_store: AuthoredStore
    run_log: JsonlRunLog
    audit_log_path: Path
    certificate_key_dir: Path


def default_issuer_identity(project: LoadedProject) -> str:
    """The default certificate issuer identity: the OS user, or the project name.

    ``getpass.getuser()`` raises ``OSError`` when there is no matching login name in
    the environment and no passwd entry (e.g. some minimal containers or CI images
    running as an arbitrary UID); fall back to the project's own name rather than
    crashing the command outright.
    """
    try:
        return getpass.getuser()
    except OSError:
        return project.config.project


#: The certified-run history file, under the project cache: one JSON run per line.
RUN_LOG_FILENAME = "runs.jsonl"

#: The tamper-evident, hash-chained audit log, under the project cache.
AUDIT_LOG_FILENAME = "audit.jsonl"


def state_dir_for(base_dir: Path) -> Path:
    """Where this project's mutable state lives: ``ELBI_STATE_DIR``, else in it.

    Kept out of :func:`load_project` so a caller needing the path before loading (the
    entrypoint deciding what to create, a test asserting where a log went) resolves it
    the same way the loader will.
    """
    override = os.environ.get(STATE_DIR_ENV, "").strip()
    return Path(override) if override else base_dir / CACHE_DIRNAME


def load_project(base_dir: Path, *, state_dir: Path | None = None) -> LoadedProject:
    """Load and discover the project rooted at ``base_dir``.

    ``state_dir`` is where everything the project *writes* goes, defaulting to
    ``ELBI_STATE_DIR`` or ``<base_dir>/.elbi``. Pass it explicitly to
    serve a project directory that is read-only.

    Raises:
        ConfigError: if ``elbi.yaml`` is missing.
    """
    state = state_dir if state_dir is not None else state_dir_for(base_dir)
    project_path = base_dir / PROJECT_FILENAME
    if not project_path.exists():
        raise ConfigError(
            f"no {PROJECT_FILENAME} found in {base_dir}. "
            "Run `elbi init <name>` to scaffold a project."
        )

    config = ProjectConfig.load(project_path)
    bindings_path = base_dir / DEV_BINDINGS_FILENAME
    derivations_dir = base_dir / config.derivations_dir
    cache_dir = state / "cache"
    workspaces_dir = state / WORKSPACES_DIRNAME

    # Hybrid retrieval (BM25 + semantic) is the registry default; a project may opt
    # down to a purely lexical ranker. This governs `elbi mcp`, `serve`, and
    # the app, which all load through here.
    registry = Registry()
    if config.search == "lexical":
        registry.use_retriever(Bm25Retriever())
    discovered: tuple[str, ...] = ()
    if derivations_dir.exists():
        discovered = tuple(discover(derivations_dir, registry=registry))

    # Reload agent-authored derivations from the sidecar (never the source tree)
    # before applying lifecycle, so a human certification recorded for one still
    # takes effect on the reloaded value.
    authored_store = AuthoredStore(state / AUTHORED_DIRNAME)
    authored = authored_store.load_into(registry)

    lifecycle = LifecycleStore(state / LIFECYCLE_FILENAME)
    _apply_lifecycle(registry, lifecycle)

    run_log = JsonlRunLog(state / RUN_LOG_FILENAME)

    def make_runner(bindings: DataBindings | None = None) -> Runner:
        # ``bindings`` defaults to the project's dev bindings (file/SQL); a caller may
        # pass its own resolver instead: the app injects warehouse-backed bindings so
        # every derivation reads its dataset inputs from the warehouse.
        if bindings is None:
            bindings = DataBindings.load(bindings_path)
        # Route by provenance: human-authored derivations run in-process; agent-
        # authored ones are isolated in the configured sandbox (subprocess or docker).
        return Runner(
            registry,
            bindings=bindings,
            base_dir=base_dir,
            store=LocalCacheStore(cache_dir),
            executor=RoutingExecutor(
                sandbox=sandbox_executor(config.sandbox, config.sandbox_image)
            ),
        )

    def load_dataset(name: str) -> Table:
        return DataBindings.load(bindings_path).load_dataset(name, base_dir)

    return LoadedProject(
        config=config,
        registry=registry,
        base_dir=base_dir,
        derivations_dir=derivations_dir,
        state_dir=state,
        bindings_path=bindings_path,
        cache_dir=cache_dir,
        workspaces_dir=workspaces_dir,
        discovered=discovered,
        authored=authored,
        make_runner=make_runner,
        load_dataset=load_dataset,
        lifecycle=lifecycle,
        authored_store=authored_store,
        run_log=run_log,
        audit_log_path=state / AUDIT_LOG_FILENAME,
        certificate_key_dir=state,
    )


def _apply_lifecycle(registry: Registry, lifecycle: LifecycleStore) -> None:
    """Override each discovered derivation's status with its recorded one."""
    for name, status in lifecycle.statuses().items():
        if name in registry:
            current = registry.get(name)
            if current.status != status:
                # statuses() has already validated the value; map it to the typed
                # literal so the recorded status flows through with its real type.
                recorded: Status = "certified" if status == "certified" else "proposed"
                registry.replace(replace(current, status=recorded))


def reload_file_derivations(project: LoadedProject) -> dict[str, list[str]]:
    """Re-import ``derivations/*.py`` into the live registry, picking up edits.

    A running app calls this so a change to a derivation source file takes effect
    without a restart. File-authored (human) derivations are reconciled to match disk (a
    new file is registered, an edited one replaced, a deleted one removed), while
    agent-authored derivations (which live in the authored store, not the source tree)
    are left in place. Returns the reconciliation as added/reloaded/removed names.
    """
    fresh = Registry()
    if project.derivations_dir.exists():
        discover(project.derivations_dir, registry=fresh)
    _apply_lifecycle(fresh, project.lifecycle)
    on_disk = set(fresh.names())
    registry = project.registry

    removed = [
        name
        for name in registry.names()
        if not registry.get(name).is_agent_authored and name not in on_disk
    ]
    for name in removed:
        registry.remove(name)

    added, reloaded = [], []
    for name in sorted(on_disk):
        if name in registry:
            registry.replace(fresh.get(name))
            reloaded.append(name)
        else:
            registry.register(fresh.get(name))
            added.append(name)
    return {"added": added, "reloaded": reloaded, "removed": removed}
