"""Wire a project into the app and run it: ``elbi serve``.

Reuses the same project loading and MCP build as ``elbi mcp``, so the chat
app, the MCP endpoint, and the local data bindings are the one consistent surface.
The LLM client reads ``ANTHROPIC_API_KEY`` from the environment (bring-your-own-key).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
import textwrap
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from elbi_agent import LLMClient
from elbi_agent.anthropic_client import MODELS
from elbi_cli.guidance import compose_instructions
from elbi_cli.mcp_server import _unregister, build_server
from elbi_cli.project import load_project, sandbox_executor
from elbi_core import (
    Artifact,
    ChainedJsonlAuditSink,
    Context,
    RoutingExecutor,
    Runner,
    load_issuer,
)
from elbi_core import cache as cache_policies
from elbi_core import serve as serve_builders
from elbi_core.cache import LocalCacheStore, derivation_tag
from elbi_core.config import DataBindings, DatasetSpec, SourceSpec
from elbi_core.data import Table
from elbi_core.derivation import Derivation
from elbi_core.errors import ConfigError, DataBindingError, ElbiError, ModelError
from elbi_core.retrieval import OnnxEmbedder
from elbi_core.sandbox import ComputeProfile
from elbi_core.versioning import hash_file, hash_json

from . import datasources
from .app import LLMConfigError, Rows, create_app
from .authoring import make_derive_factory, make_draft_factory
from .compute import (
    credential_vendor,
    executor_backend,
    record_session,
    resolve_image,
    resolve_profiles,
    resolve_runner,
    runner_options,
    session_cost,
)
from .dashboards import DashboardService
from .db import Derivation as DerivationRow
from .db import PromotedQuery, open_store
from .env import env
from .explore import WAREHOUSE_SOURCE, ExploreService
from .extensions import service_class
from .features import FeatureStoreService
from .lineage import LineageService
from .metrics import MetricService
from .ml import kernel_tracking_env, make_model_service
from .monitoring import MonitorService
from .notebooks import NotebookService
from .notifications import monitor_alert_handler, orchestration_alert_handler
from .orchestration import OrchestrationService
from .query_runner import build_runner
from .search.build import Drain, SearchBuilder
from .search.document_map import EMBED_DIM, EMBED_MODEL
from .search.index import SearchIndex
from .search.invalidation import PendingWrites, install
from .search.retrievers import indexed_retriever
from .search.sources import Sources
from .titles import llm_title
from .warehouse import storage
from .warehouse.service import WarehouseError, WarehouseService, warehouse_available

logger = logging.getLogger("elbi")


def project_root(directory: str | Path) -> Path:
    """The project directory to serve, absolute but with a symlink left intact.

    Resolving looks harmless and breaks git delivery in two ways at once. git-sync hands
    the app a symlink and retargets it at a new worktree on every commit, so a resolved
    path pins the app to the worktree that happened to be current at startup (which
    git-sync later deletes) and the link it was supposed to watch is no longer anywhere
    in the loaded project. Reading through the link instead means every re-read gets the
    revision that is current now.
    """
    requested = Path(directory).expanduser()
    return requested.absolute() if requested.is_symlink() else requested.resolve()


#: How often to check whether the project's revision changed, in seconds. Roughly the
#: sidecar's own sync period: checking faster only finds the same revision again, and
#: the check is one ``readlink``. Zero switches it off, for a deployment that would
#: rather restart the pod to pick up a release.
_WATCH_INTERVAL_ENV = "PROJECT_WATCH_INTERVAL_SECONDS"
_DEFAULT_WATCH_INTERVAL = 30.0


def _watch_interval() -> float:
    """The revision-check cadence, from the environment."""
    raw = os.environ.get(_WATCH_INTERVAL_ENV, "").strip()
    if not raw:
        return _DEFAULT_WATCH_INTERVAL
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.0fs",
            _WATCH_INTERVAL_ENV,
            raw,
            _DEFAULT_WATCH_INTERVAL,
        )
        return _DEFAULT_WATCH_INTERVAL


def project_revision(path: Path) -> str | None:
    """What the project directory points at, or ``None`` for a plain directory.

    A deployment that delivers its project from git gives the app a symlink, and the
    sidecar retargets it at the new worktree on every commit. The link's target is
    therefore both the change signal and the name of the revision being served.
    Everything else (a project baked into an image, a directory on a laptop) is not a
    symlink and has no revision to watch, so watching costs one ``readlink`` and stops.
    """
    try:
        return str(path.readlink()) if path.is_symlink() else None
    except OSError:
        return None


#: Row ceiling when an external-source promoted derivation reads its source. A promoted
#: query should be bounded; this caps a runaway result rather than pull a whole table.
_EXTERNAL_PROMOTE_MAX_ROWS = 100_000
#: Cap on rows pulled when a synced warehouse table is read as a dataset (into a
#: notebook kernel, a model's training frame, etc.), so a big table won't exhaust RAM.
_WAREHOUSE_LOAD_MAX_ROWS = 1_000_000


def _aggregate_values(agg: str, values: list[float]) -> float:
    """Reduce a derivation column's values for a monitor (empty -> 0.0)."""
    if not values:
        return 0.0
    if agg == "mean":
        return sum(values) / len(values)
    if agg == "min":
        return min(values)
    if agg == "max":
        return max(values)
    if agg == "count":
        return float(len(values))
    return sum(values)


@dataclass(frozen=True)
class _RequestBindings(DataBindings):
    """Dataset bindings backed by request rows instead of files or SQL.

    Applying a feature derivation to rows that arrived over the wire means
    running it with its dataset inputs bound to those rows; everything else
    about execution (registry, sandbox routing, chaining) stays exactly the
    project's.
    """

    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def load_dataset(self, dataset: str, base_dir: Path) -> Table:
        if dataset not in self.tables:
            raise DataBindingError(
                f"dataset {dataset!r} is not bound in this scoring request"
            )
        return Table(rows=list(self.tables[dataset]))

    def version(self, dataset: str, base_dir: Path) -> tuple[str, Table | None]:
        table = self.load_dataset(dataset, base_dir)
        return hash_json(table.rows), table


@dataclass(frozen=True)
class _WarehouseBindings(DataBindings):
    """Dataset bindings that resolve every name to a warehouse Delta table.

    The runner uses these so a derivation's dataset inputs come from the warehouse.
    ``version`` returns the stored ingest fingerprint when present (a zero-read cache
    check that changes when a source is re-synced), otherwise a content hash of the
    rows, so the derivation cache invalidates exactly when the data changes.
    """

    warehouse: WarehouseService | None = None
    fingerprint: Callable[[str], str | None] | None = None
    max_rows: int = _WAREHOUSE_LOAD_MAX_ROWS

    def _rows(self, dataset: str) -> list[dict[str, Any]]:
        if self.warehouse is None:
            raise DataBindingError("the warehouse is not available")
        _cols, rows, _trunc = self.warehouse.read_any(dataset, max_rows=self.max_rows)
        return rows

    def load_dataset(self, dataset: str, base_dir: Path) -> Table:
        return Table(rows=self._rows(dataset))

    def version(self, dataset: str, base_dir: Path) -> tuple[str, Table | None]:
        if self.fingerprint is not None:
            stored = self.fingerprint(dataset)
            if stored is not None:
                return stored, None  # zero-read cache check
        rows = self._rows(dataset)
        return hash_json(rows), Table(rows=rows)


def build(
    directory: str | Path, *, model: str | None = None, with_mcp: bool = True
) -> FastAPI:
    """Construct the app for the project at ``directory`` (datasets, MCP, SPA)."""
    project = load_project(project_root(directory))

    # The native store: SQLite under the project's cache dir by default, or wherever
    # DB_URI points (a Postgres URI to deploy). Chat answers are authored here.
    store = open_store(env("DB_URI") or f"sqlite:{project.cache_dir / 'app.db'}")
    # The Delta Lake warehouse is the single source of data: every dataset the app
    # serves is a warehouse table (a project-declared source or a connector sync), so
    # the warehouse extra is required.
    if not warehouse_available():
        raise ConfigError(
            "serving a project requires the 'warehouse' extra (deltalake + duckdb): "
            "the warehouse is the only source of data. Install elbi[warehouse]."
        )
    # Warehouse SQL runs on compute of its own when configured: DuckDB is in-process, so
    # a query executed in the web server competes with request handling and an
    # out-of-memory one ends the service rather than the query.
    warehouse_service = service_class(WarehouseService)(store)
    warehouse_service.set_runner(build_runner(warehouse_service._query_here))

    def warehouse_tables() -> list[str]:
        """Every synced warehouse table name: the app's whole data namespace."""
        return [t["table"] for t in warehouse_service.tables()]

    def read_rows(name: str) -> Rows:
        """Read a warehouse table's rows by name: the one runtime read path.

        Bounded by ``_WAREHOUSE_LOAD_MAX_ROWS`` so a large table cannot exhaust memory.
        Raises :class:`WarehouseError` if the table does not exist.
        """
        _columns, rows, _truncated = warehouse_service.read_any(
            name, max_rows=_WAREHOUSE_LOAD_MAX_ROWS
        )
        return rows

    def dataset_fingerprint(name: str) -> str | None:
        """A content fingerprint of a table for staleness checks (or ``None`` if gone).

        Prefers the stored ingest fingerprint (cheap, no read) for a declared source;
        for a connector table it hashes the rows. Used by orchestration data-change
        sensors so "data changed" agrees with the derivation cache's ``version()``.
        """
        stored = store.get_config(f"warehouse.source.{name}.fingerprint")
        if stored is not None:
            return stored
        try:
            return hash_json(read_rows(name))
        except WarehouseError:
            return None

    def load_dataset(name: str) -> Table:
        """Load a dataset's rows from the warehouse."""
        return Table(rows=read_rows(name))

    def notebook_query(sql: str, table: str, limit: int) -> dict[str, Any]:
        """Answer a notebook cell's ``sql(...)`` or ``data['x']`` from the warehouse.

        Runs where the engine and the storage credentials already are, so the kernel
        (scrubbed environment, blocked network) needs neither. Only the bounded result
        crosses back.

        ``table`` is resolved by name against the synced tables, never interpolated into
        SQL: a dataset name is user-controlled, and building a query from one inside the
        kernel is the class of bug behind the Polaris path-scoping CVEs.
        """
        capped = max(1, min(limit, _WAREHOUSE_LOAD_MAX_ROWS))
        try:
            if table:
                columns, rows, truncated = warehouse_service.read_any(
                    table, max_rows=capped
                )
            else:
                columns, rows, truncated = warehouse_service.query(sql, max_rows=capped)
        except WarehouseError as exc:
            return {"error": str(exc)}
        return {"columns": columns, "rows": rows, "truncated": truncated}

    def load_datasets() -> dict[str, Rows]:
        """Every warehouse table as name→rows (the chat/notebooks/models namespace)."""
        loaded: dict[str, Rows] = {}
        for name in warehouse_tables():
            try:
                loaded[name] = read_rows(name)
            except WarehouseError:
                continue  # a table that fails to read is simply unavailable
        return loaded

    #: Every dataset the agent can discover and describe: the warehouse tables.
    def dataset_specs() -> tuple[DatasetSpec, ...]:
        return tuple(DatasetSpec(name=name) for name in warehouse_tables())

    def _source_fingerprint(spec: SourceSpec, config: dict[str, Any]) -> str:
        """A content fingerprint of a declared source, to skip an unchanged re-sync.

        A file source also hashes its bytes (cheap) so an edited fixture re-syncs;
        a remote connector fingerprints its config only, leaving ongoing refresh to the
        sync cadence rather than re-pulling the source on every boot.
        """
        payload: dict[str, Any] = {"type": spec.type, "config": config}
        path = config.get("path")
        if isinstance(path, str) and Path(path).exists():
            payload["bytes"] = hash_file(Path(path))
        return hash_json(payload)

    def _dev_binding_sources() -> tuple[SourceSpec, ...]:
        """Auto-derive a CSV source from each dataset's local dev.yaml file binding.

        So a project need not declare the same path twice (once for `elbi
        dev`, once for the warehouse) when both are meant to read the same file. An
        explicit `sources:` entry always wins (skipped here, not overridden). A
        SQL binding, an `env(VAR)` reference, or a path that isn't an existing
        .csv/.parquet file is left alone -- those need an explicit `sources:` entry,
        since there is no generic way to turn an arbitrary SQL connection into a
        warehouse connector automatically.
        """
        declared = {spec.name for spec in project.config.sources}
        bindings = DataBindings.load(project.bindings_path).bindings
        derived: list[SourceSpec] = []
        for name in project.config.dataset_names:
            if name in declared:
                continue
            raw = bindings.get(name)
            if not isinstance(raw, str):
                continue  # unbound, or a SQL mapping -- needs an explicit source
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (project.base_dir / candidate).resolve()
            if candidate.suffix not in (".csv", ".parquet") or not candidate.is_file():
                continue  # env(VAR) resolves to no such file, same as a bad path
            derived.append(SourceSpec(name=name, type="csv", config={"path": raw}))
        return tuple(derived)

    def _ensure_declared_sources() -> None:
        """Ensure each project-declared (or dev-binding-derived) source is synced.

        This is the only place a project's declaration turns into warehouse data.
        Idempotent: a source whose fingerprint is unchanged and whose table already
        exists is skipped; a changed declaration is rebuilt (drop, recreate, sync).
        Project sources land unprefixed so the table name equals the source name.
        """
        for spec in (*project.config.sources, *_dev_binding_sources()):
            config = dict(spec.config)
            path = config.get("path")
            if isinstance(path, str) and not Path(path).is_absolute():
                config["path"] = str((project.base_dir / path).resolve())
            fingerprint = _source_fingerprint(spec, config)
            key = f"warehouse.source.{spec.name}.fingerprint"
            existing = store.get_external_source_by_name(spec.name)
            table_ok = storage.table_location(spec.name) is not None
            if (
                existing is not None
                and store.get_config(key) == fingerprint
                and table_ok
            ):
                continue  # unchanged and already materialized
            if existing is not None:
                warehouse_service.delete_source(existing.id)
            source = warehouse_service.create_source(
                spec.type, spec.name, config, sync_frequency=spec.sync, prefix=""
            )
            warehouse_service.sync_source(source.id)
            store.set_config(key, fingerprint)

    _ensure_declared_sources()

    # Tables synced before there was anywhere to record their columns: an unchanged
    # declaration is skipped above, and a "manual" source may not sync again.
    _backfilled = warehouse_service.backfill_columns()
    if _backfilled:
        logger.info("recorded columns for %d existing table(s)", len(_backfilled))

    def _compute_source(derivation: Derivation) -> str:
        """The Python source of a repo derivation's compute, best-effort.

        Repo derivations carry their logic as a function (unlike agent-authored ones,
        whose source is stored text), so it is read back from the function object for
        display. Empty when the source is unavailable (a REPL- or C-defined compute).
        """
        try:
            return textwrap.dedent(inspect.getsource(derivation.compute))
        except (OSError, TypeError):
            return ""

    def _sync_repo_derivations() -> None:
        """Record repo-authored derivations in the store so the UI reads them there.

        Every UI surface is backed by the store, never the code tree: a derivation
        authored as a ``derivations/*.py`` file is mirrored into the derivations table
        (origin ``repo``) so it lists alongside chat-authored ones. The compute stays
        in the registry as the runtime that executes it; the store row is the catalog
        record the UI shows. Reconciled to the registry, so a deleted file drops.
        """
        # A human derivation is trusted (certified) but never oracle-run, so it carries
        # no soundness verdict: the same convention the catalog uses (verdict null for a
        # human derivation). ``origin`` marks it as repo-authored for the UI.
        rows = [
            DerivationRow(
                name=d.name,
                question=(d.description or "").strip(),
                # Mirror the compute source and serve contract so the detail view
                # shows them; the rendered output is filled in lazily on view, since
                # producing it means running the derivation against bound data.
                source=_compute_source(d),
                serve_json=(
                    json.dumps(d.serve.to_manifest()) if d.serve is not None else None
                ),
                verdict=None,
                origin="repo",
            )
            for d in project.registry
            if not d.is_agent_authored
        ]
        store.sync_repo_derivations(rows)

    _sync_repo_derivations()

    def _reload_project() -> dict[str, Any]:
        """Re-read the project from disk: derivation source files and source data.

        This is the server side of ``elbi sync``: after the CLI pushes the declarative
        artifacts over the API, it calls this so a change to a derivation ``.py`` or to
        a declared source's data takes effect without a restart. The live registry backs
        execution; the store row backs the UI, so both are refreshed.
        """
        from elbi_cli.project import reload_file_derivations

        derivations = reload_file_derivations(project)
        _sync_repo_derivations()
        _ensure_declared_sources()
        all_sources = (*project.config.sources, *_dev_binding_sources())
        return {
            "derivations": derivations,
            "sources": [spec.name for spec in all_sources],
        }

    served_revision = project_revision(project.base_dir)

    def _project_revision_tick() -> None:
        """Re-read the project when its revision changed underneath the running app.

        Without this, a sidecar that pulls a new commit leaves the files on disk ahead
        of what the app loaded; the deployment reports success and serves the previous
        derivations, which is the failure mode worth the most effort to avoid. Reloading
        on the revision rather than on a timer means an unchanged project costs one
        ``readlink``.
        """
        nonlocal served_revision
        current = project_revision(project.base_dir)
        if current is None or current == served_revision:
            return
        previous, served_revision = served_revision, current
        try:
            summary = _reload_project()
        except Exception:
            # Keep the old revision recorded: the app is still serving it, and reporting
            # the new one would make the audit trail claim a load that did not happen.
            served_revision = previous
            logger.exception(
                "the project revision changed to %s and it could not be loaded", current
            )
            return
        logger.info(
            "project revision %s -> %s (%d derivations)",
            previous,
            current,
            len(summary.get("derivations", {}).get("reloaded", []))
            + len(summary.get("derivations", {}).get("added", [])),
        )
        # The evidence a change-control auditor asks for: which revision the app is
        # serving, and when it started.
        store.record_audit(
            "project.revision_changed",
            target_type="project",
            target_id=str(current),
            data_hash=str(previous) if previous else None,
        )

    def _watch_project_revision(app: FastAPI) -> None:
        """Poll for a new revision, on a thread, for as long as the app is up.

        Its own thread and its own cadence rather than a housekeeping tick: housekeeping
        runs hourly, and an hour is not an acceptable delay between a merge landing and
        the app serving it. Started only when there is a link to watch, so a project
        baked into an image or sitting on a laptop pays nothing.
        """
        if project_revision(project.base_dir) is None:
            return
        interval = _watch_interval()
        if interval <= 0:
            return
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(interval):
                try:
                    _project_revision_tick()
                except Exception:
                    # A watcher that dies on one bad commit stops noticing every good
                    # one after it, silently. Log and keep polling.
                    logger.exception("the project revision check failed")

        # A daemon thread, so it never holds up shutdown. It exists only for a
        # git-delivered project, where the process lives as long as the pod does; the
        # stop event is on app.state so an embedder or a test can end it deliberately.
        threading.Thread(target=loop, name="project-revision", daemon=True).start()
        app.state.stop_project_revision_watch = stop.set

    # Derivations resolve dataset inputs from the warehouse through these bindings, so
    # metrics, the feature store, dashboards, and orchestration read the warehouse too.
    warehouse_bindings = _WarehouseBindings(
        warehouse=warehouse_service,
        fingerprint=lambda name: store.get_config(
            f"warehouse.source.{name}.fingerprint"
        ),
    )

    def make_runner(bindings: DataBindings | None = None) -> Runner:
        """A runner whose dataset inputs come from the warehouse."""
        return project.make_runner(bindings=bindings or warehouse_bindings)

    def render_derivation(name: str) -> str | None:
        """A repo derivation's rendered output for the detail view, best-effort.

        Run on view rather than at sync time, since producing it means executing the
        derivation against bound data; the runner caches, so a repeat view is cheap.
        Returns None when the derivation is internal (no serve contract) or its data
        is unavailable, so the detail view simply omits the output.

        Also None when a policy applies to the caller. The runner renders to text, so a
        column mask cannot be applied to the result after the fact, and serving it
        unnarrowed would disclose exactly what the policy withholds. The narrowed rows
        stay reachable through the served path.
        """
        # ponytail: withheld, not narrowed. Narrowing here needs the artifact rather
        # than the runner's rendering, which is what Serving already does; route this
        try:
            return make_runner().serve(name)
        except ElbiError:
            return None

    class _Operations:
        """Late-bound adapter exposing operate actions to the MCP server.

        Actions are materialize, run a workflow or notebook, and backfill. Bound once
        the services exist: the MCP tools are only invoked at runtime, so registering
        them up front is safe.
        """

        def __init__(self) -> None:
            self._orch: OrchestrationService | None = None
            self._nb: NotebookService | None = None

        def bind(
            self, orchestration: OrchestrationService, notebook: NotebookService
        ) -> None:
            self._orch, self._nb = orchestration, notebook

        @property
        def orchestration(self) -> OrchestrationService:
            if self._orch is None:
                raise RuntimeError("operations not bound to the orchestration service")
            return self._orch

        @property
        def notebook(self) -> NotebookService:
            if self._nb is None:
                raise RuntimeError("operations not bound to the notebook service")
            return self._nb

        def materialize(
            self, selection: str, assets: list[str] | None, include_downstream: bool
        ) -> dict[str, Any]:
            return self.orchestration.materialize(
                selection=selection,
                assets=assets,
                include_downstream=include_downstream,
            )

        def run_workflow(self, name: str) -> dict[str, Any]:
            match = next(
                (w for w in self.orchestration.workflows() if w["name"] == name), None
            )
            if match is None:
                raise KeyError(f"no workflow named {name!r}")
            return self.orchestration.run_workflow(match["id"])

        def run_notebook(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
            # Scoping the lookup is what scopes the run: `run_scheduled` takes an id it
            # trusts, and this is the only place that id is resolved from a name. RUN
            # rather than the listing default, so a notebook shared for reading is no
            # more runnable here than it is over `/api`.
            match = next(
                (n for n in self.notebook.summaries() if n["name"] == name),
                None,
            )
            if match is None:
                raise KeyError(f"no notebook named {name!r}")
            return self.notebook.run_scheduled(match["id"], params)

        def backfill(self, asset: str, param: str, values: list[str]) -> dict[str, Any]:
            return self.orchestration.backfill(asset=asset, param=param, values=values)

        def asset_status(self) -> list[dict[str, Any]]:
            return self.orchestration.status()

    operations = _Operations()
    # One signing identity for the deployment: the project certifies (the oracle runs
    # server-side and cannot be weakened by the agent), so the certifier is the system,
    # keyed by the project's signing key. Shared by the MCP server and the API export so
    # both surfaces sign under the same identity.
    issuer = load_issuer(project.certificate_key_dir, issuer=project.config.project)
    mcp_server = None

    def _stop_derivation_runtime(name: str) -> None:
        """Stop a derivation from running or serving: the registry and its MCP tool.

        Shared by trash and erase alike -- either way it should not keep going.
        """
        if name in project.registry:
            project.registry.remove(name)
        if mcp_server is not None:
            _unregister(mcp_server, name)

    def derivation_on_trash(name: str) -> bool:
        """Stamp a derivation trashed, then stop it running and hide its sidecar."""
        trashed = store.trash_derivation(name)
        if trashed:
            _stop_derivation_runtime(name)
            project.authored_store.trash(name)
        return trashed

    def derivation_on_restore(name: str) -> bool:
        """Clear a derivation's trash stamp and bring it back live.

        Only re-registers live when the store actually restored it, so restoring
        something that was never trashed leaves nothing running. A promoted
        query rebuilds through :func:`register_external_derivation`, the exact
        path startup already uses; an agent-authored one reloads from its
        sidecar. Execution, orchestration, lineage, catalog, and search all come
        back immediately either way; only its per-derivation MCP tool waits for
        the next server start (live registration needs the server's internal
        ``Serving`` plumbing, deliberately not reached from here).
        """
        if not store.restore_derivation(name):
            return False
        promoted = store.get_promoted_query(name)
        if promoted is not None:
            register_external_derivation(
                promoted.name, promoted.source_id, promoted.sql
            )
        else:
            project.authored_store.restore(name)
            project.authored_store.load_one(name, project.registry)
        return True

    def derivation_on_author(name: str, record: dict[str, Any]) -> bool:
        """Persist an MCP-authored derivation as a governed store row.

        The MCP layer has no database, so ``propose_derivation`` calls back here
        with the outcome's fields; this writes the same row shape the chat and
        explore-promote paths do (``authoring.py``), which is what makes an
        MCP-authored derivation visible in /api/derivations and holdable by
        trash. ``conversation_id`` is synthetic, the way explore's is.
        """
        attestation = record.get("attestation") or {}
        store.save_derivation(
            DerivationRow(
                name=name,
                conversation_id="mcp",
                question=record.get("question") or "Proposed via MCP",
                source=str(record.get("source") or ""),
                claim_json=(
                    json.dumps(record["claim"]) if record.get("claim") else None
                ),
                serve_json=json.dumps(
                    {
                        "format": record.get("format") or "table",
                        "deps": list(record.get("deps") or []),
                    }
                ),
                verdict=record.get("verdict"),
                rendered=str(record.get("rendered") or ""),
                attestation_json=json.dumps(attestation) if attestation else None,
                data_hash=(
                    attestation.get("data_hash")
                    or (attestation.get("contract") or {}).get("data_hash")
                ),
            )
        )
        return True

    def derivation_on_erase(name: str) -> bool:
        """Permanently erase a derivation: row, run history, sidecar, cache, promotion.

        Only touches the runtime side if the database erase actually happened.
        """
        if not store.erase_derivation(name):
            return False
        _stop_derivation_runtime(name)
        project.authored_store.remove(name)
        LocalCacheStore(project.cache_dir).invalidate_tag(derivation_tag(name))
        store.delete_promoted_query(name)
        return True

    if with_mcp:
        mcp_server = build_server(
            project.registry,
            make_runner,
            enable_propose=True,
            operations=operations,
            load_dataset=load_dataset,
            dataset_specs=dataset_specs(),
            # Warehouse-schema tools so an external coding agent learns the data.
            # Resolved per call, not bound here.
            warehouse_schema=lambda: warehouse_service.duckdb_catalog(),
            warehouse_sample=lambda name, limit: warehouse_service.read_any(
                name, max_rows=limit
            )[1],
            name=project.config.project,
            authored_store=project.authored_store,
            # The served app has a database, so an MCP-driven propose writes a
            # governed row and an MCP-driven delete trashes rather than erases
            # -- see build_server's own docstring on this.
            on_author=derivation_on_author,
            on_delete=derivation_on_trash,
            issuer=issuer,
            audit=ChainedJsonlAuditSink(project.audit_log_path),
            cache_dir=project.cache_dir,
            scratch_dir=project.workspaces_dir / "shared",
            backend=project.config.sandbox,
            egress=project.config.egress,
            image=resolve_image(project.config),
            instructions=compose_instructions(project.config.ai_context),
        )

    def list_derivations() -> list[str]:
        """Certified derivations, the trainable feature sources."""
        return [d.name for d in project.registry if d.is_certified]

    def is_certified(name: str) -> bool:
        """Whether a derivation exists in the project registry and is certified."""
        return any(d.name == name and d.is_certified for d in project.registry)

    def certified_catalog() -> list[dict[str, Any]]:
        """Certified derivations eligible to bind to a dashboard widget.

        Both the publish gate (which refuses uncertified bindings) and the editor's
        widget picker read this, so a dashboard is composed only over verified results.
        """
        catalog: list[dict[str, Any]] = []
        for derivation in project.registry:
            if not derivation.is_certified:
                continue
            manifest = derivation.to_manifest()
            catalog.append(
                {
                    "name": derivation.name,
                    "title": derivation.description or derivation.name,
                    "params": manifest.get("params", {}),
                    "served": derivation.is_served,
                }
            )
        return catalog

    def load_derivation_rows(name: str) -> list[dict[str, Any]]:
        """A certified derivation's output rows (cached, content-addressed)."""
        try:
            derivation = project.registry.get(name)
        except ElbiError as exc:
            raise ModelError(str(exc)) from exc
        if not derivation.is_certified:
            raise ModelError(f"derivation {name!r} is not certified")
        artifact = make_runner().run(name)
        value = artifact.value
        if not (isinstance(value, list) and value and isinstance(value[0], dict)):
            raise ModelError(
                f"derivation {name!r} does not produce tabular rows to train on"
            )
        return list(value)

    def apply_derivation(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run the feature derivation over request rows (the scoring transform).

        The derivation's dataset inputs are all bound to the incoming rows (a
        feature pipeline reads one logical dataset), and the run is fresh: request
        rows are never worth a cache entry.
        """
        try:
            derivation = project.registry.get(name)
        except ElbiError as exc:
            raise ModelError(str(exc)) from exc
        dataset_names = {
            dataset.name for dataset in derivation.dataset_inputs().values()
        }
        runner = Runner(
            project.registry,
            bindings=_RequestBindings(tables=dict.fromkeys(dataset_names, rows)),
            base_dir=project.base_dir,
            executor=RoutingExecutor(
                # Sized by the default profile, so agent-written code is bounded by the
                # same menu a notebook picks from rather than by a number written into
                # the library. The trusted half of the routing stays in-process: the
                # oracle must not run on compute a user can choose or starve.
                sandbox=sandbox_executor(
                    executor_backend(project.config),
                    resolve_image(project.config),
                    profile=compute_profiles.select(None),
                    namespace=runner_options().get("namespace"),
                )
            ),
        )
        try:
            artifact = runner.run(name, fresh=True)
        except ElbiError as exc:
            raise ModelError(f"feature derivation failed: {exc}") from exc
        value = artifact.value
        if not (isinstance(value, list) and all(isinstance(r, dict) for r in value)):
            raise ModelError(f"derivation {name!r} did not return rows")
        return list(value)

    # Resolve the LLM client from explicit config on demand, so a settings change (or a
    # per-turn model override) takes effect without a restart. Precedence: the chosen
    # profile, then the default profile, then the LLM_* environment / --model flag. An
    # unconfigured app raises LLMConfigError. Clients are cached by their
    # (model, api_key, base_url) signature.
    default_model = model or ""
    client_cache: dict[tuple[str, str, str, str], LLMClient] = {}

    def resolve_client(profile_name: str | None = None) -> LLMClient:
        # The model comes from the chosen profile, else the default profile, else the
        # LLM_MODEL env / --model flag. An unknown profile name resolves to no profile
        # and falls through the same way. With nothing configured, this raises.
        name = profile_name or store.get_config("llm.default_profile") or ""
        profile = store.get_profile(name) if name else None
        chosen = (
            (profile.model if profile else "")
            or os.environ.get("LLM_MODEL")
            or default_model
        )
        if not chosen:
            raise LLMConfigError(
                "No LLM model is configured. Add a model profile in Settings, set the "
                "LLM_MODEL environment variable, or pass --model: no default model is "
                "assumed."
            )
        api_key = (profile.api_key if profile else "") or env("LLM_API_KEY") or ""
        base_url = (
            (profile.base_url if profile else "")
            or os.environ.get("LLM_BASE_URL")
            or ""
        )
        effort = (
            (profile.reasoning_effort if profile else "")
            or os.environ.get("LLM_REASONING_EFFORT")
            or ""
        )
        signature = (chosen, api_key, base_url, effort)
        if signature not in client_cache:
            client_cache[signature] = _build_client(
                chosen, api_key or None, base_url or None, effort or None
            )
        return client_cache[signature]

    model_service = make_model_service(
        store=store,
        cache_dir=project.cache_dir,
        load_datasets=load_datasets,
        list_derivations=list_derivations,
        load_derivation_rows=load_derivation_rows,
        apply_derivation=apply_derivation,
        # Feature-store training sets (materialized point-in-time joins) become training
        # sources, read straight from the store where the feature store persisted them.
        list_training_sets=(
            (lambda: store.list_training_set_names()) if store is not None else None
        ),
        load_training_set_rows=(
            (lambda name: store.training_set_rows(name)) if store is not None else None
        ),
    )

    def explore_resolve_source(name: str) -> Any:
        """A dataset's queryable rows for the workbench, read from the warehouse."""
        return read_rows(name)

    def explore_read_schema(name: str) -> tuple[list[str], int]:
        """A dataset's columns and row count, read from the warehouse table."""
        return warehouse_service.table_shape(name)

    def register_external_derivation(name: str, source_id: str, sql: str) -> None:
        """Register a trusted in-process derivation reading a source or the warehouse.

        The ``source_id`` is a registered external data source or the synthetic
        :data:`WAREHOUSE_SOURCE`. The read reaches state the sandbox cannot, so it is
        human-origin, in-process, and cache-disabled; served as a table. Re-registering
        replaces any earlier derivation of the same name.
        """

        def compute(ctx: Context) -> Artifact:
            if source_id == WAREHOUSE_SOURCE:
                if warehouse_service is None:
                    raise ModelError("the data warehouse is not available")
                try:
                    _, rows, truncated = warehouse_service.query(
                        sql, max_rows=_EXTERNAL_PROMOTE_MAX_ROWS
                    )
                except WarehouseError as exc:
                    # Wrap the plain WarehouseError so promote_external's guard catches
                    # a bad query and leaves nothing registered (like explore.run_sql).
                    raise DataBindingError(str(exc)) from exc
                if truncated:
                    # Refuse rather than certify a silently-clipped partial answer.
                    raise DataBindingError(
                        f"query returns more than {_EXTERNAL_PROMOTE_MAX_ROWS:,} rows; "
                        "add a LIMIT or aggregate before promoting it"
                    )
                return Artifact.table(rows)
            source = store.get_data_source(source_id)
            if source is None:
                raise ModelError(f"data source {source_id!r} no longer exists")
            rows = datasources.load_rows(
                source, query=sql, max_rows=_EXTERNAL_PROMOTE_MAX_ROWS
            )
            return Artifact.table(rows)

        derivation = Derivation(
            name=name,
            compute=compute,
            serve=serve_builders.table(),
            origin="human",
            status="certified",
            cache=cache_policies.never(),
        )
        if name in project.registry:
            project.registry.remove(name)
        project.registry.register(derivation)

    def promote_external(name: str, sql: str, source_id: str) -> dict[str, Any]:
        """Promote an external-source or warehouse query to a trusted derivation.

        Registers it, runs it once to confirm the query executes, then persists it so it
        rebuilds on restart. A failing query leaves nothing registered.
        """
        if source_id == WAREHOUSE_SOURCE:
            if warehouse_service is None:
                return {"ok": False, "error": "the data warehouse is not available"}
            source_label = "the data warehouse"
        else:
            source = store.get_data_source(source_id)
            if source is None:
                return {"ok": False, "error": f"data source {source_id!r} not found"}
            source_label = f"data source {source.name!r}"
        register_external_derivation(name, source_id, sql)
        try:
            rendered = make_runner().serve(name)
        except ElbiError as exc:
            project.registry.remove(name)
            return {"ok": False, "error": f"query failed against the source: {exc}"}
        # Persist for restart (PromotedQuery rebuilds the registry entry) and for the
        # derivations UI (the store row lists it beside every other certified artifact).
        store.save_promoted_query(
            PromotedQuery(name=name, source_id=source_id, sql=sql)
        )
        store.save_derivation(
            DerivationRow(
                name=name,
                conversation_id="explore",
                question=f"Promoted from Explore ({source_label})",
                source=(
                    f"# Trusted in-process derivation reading {source_label}.\n\n"
                    f"{sql}\n"
                ),
                serve_json=json.dumps({"format": "table"}),
                rendered=rendered,
                origin="human",
            )
        )
        return {
            "ok": True,
            "name": name,
            "certified": True,
            "verdict": None,
            "rendered": rendered,
        }

    # Rebuild previously promoted external-source derivations into the registry, so a
    # certified artifact survives a restart like any other. A trashed one is skipped:
    # its row still exists (trash keeps rows, it only stamps them), but registering
    # it live again would undo the trash on every restart.
    _trashed_derivation_names = {
        item["id"] for item in store.list_trash() if item["type"] == "derivation"
    }
    for promoted in store.list_promoted_queries():
        if promoted.name in _trashed_derivation_names:
            continue
        try:
            register_external_derivation(
                promoted.name, promoted.source_id, promoted.sql
            )
        except ElbiError:
            logger.exception("could not restore promoted query %r", promoted.name)

    # Built once and shared: the dashboard service resolves metric tiles through it.
    metric_service = service_class(MetricService)(
        store=store, is_certified=is_certified, load_source=load_derivation_rows
    )

    def resolve_metric_tile(
        name: str,
        group_by: Sequence[str],
        grain: str | None,
        filters: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve a dashboard tile bound to a metric, to its rows."""
        result = metric_service.query(
            name, group_by=group_by, grain=grain, filters=[dict(f) for f in filters]
        )
        return list(result["rows"])

    def read_monitor_value(monitor: Any) -> float:
        """The scalar a monitor watches: a metric's total or a derivation stat."""
        config = json.loads(monitor.config_json or "{}")
        if monitor.target_kind == "metric":
            result = metric_service.query(
                monitor.target,
                group_by=config.get("group_by", []),
                grain=config.get("grain"),
                filters=config.get("filters", []),
            )
            rows = result["rows"]
            if not rows:
                raise ModelError(f"metric {monitor.target!r} returned no rows")
            return float(rows[0][monitor.target])
        rows = load_derivation_rows(monitor.target)
        measure = config.get("measure", "row_count")
        if measure == "row_count":
            return float(len(rows))
        column = measure.get("column") if isinstance(measure, dict) else None
        agg = measure.get("agg", "sum") if isinstance(measure, dict) else "sum"
        values = [
            float(r[column])
            for r in rows
            if column in r and isinstance(r[column], (int, float))
        ]
        return _aggregate_values(agg, values)

    def monitor_source_certified(kind: str, target: str) -> bool:
        """Whether a monitor's target is a certified metric or derivation."""
        if kind == "metric":
            return metric_service.get(target) is not None
        return is_certified(target)

    monitor_service = service_class(MonitorService)(
        store=store,
        read_value=read_monitor_value,
        source_certified=monitor_source_certified,
        # In-app notification, then webhook + audit.
        on_alert=monitor_alert_handler(store),
    )

    def model_edges() -> list[dict[str, Any]]:
        """Each registered model's oracle verdict and training source.

        ``source_kind`` and ``dataset`` name the training source (a derivation, a
        bound dataset, or a feature-store training set) so lineage links the model to
        the right upstream node; ``source`` is the legacy feature-derivation tag, kept
        for models trained directly on a derivation.
        """
        if model_service is None:
            return []
        try:
            registry = model_service.registry()
            edges: list[dict[str, Any]] = []
            for model in registry.models():
                versions = registry.versions(model.name)
                tags = versions[0].tags if versions else {}
                edges.append(
                    {
                        "name": model.name,
                        "verdict": tags.get("elbi.oracle_verdict"),
                        "source": tags.get("elbi.feature_derivation"),
                        "source_kind": tags.get("elbi.source_kind"),
                        "dataset": tags.get("elbi.dataset"),
                    }
                )
            return edges
        except Exception:
            return []

    # Resolve the default client eagerly when a model is configured, but let the server
    # boot with none set (so a profile can be added in the UI): the clear LLMConfigError
    # then surfaces when a request actually needs the LLM, not at startup.
    try:
        default_client: LLMClient | None = resolve_client()
    except LLMConfigError:
        default_client = None

    # The menu of sized shapes a notebook may run on. Infrastructure defines it; a
    # notebook picks from it. Resolved once at boot so a malformed profile fails there
    # rather than the first time somebody opens a notebook.
    compute_profiles = resolve_profiles(project.config)
    # Warned rather than refused: the directory may gain the group later.

    def record_compute(
        notebook_id: str, profile: ComputeProfile, seconds: float, kind: str
    ) -> None:
        """Attribute a finished kernel session to its notebook."""
        record_session(
            store,
            session_cost(
                profile,
                seconds=seconds,
                kind=kind,
                notebook_id=notebook_id,
                rates=compute_profiles.rates,
            ),
        )

    notebook_service = service_class(NotebookService)(
        store=store,
        load_datasets=load_datasets,
        # Datasets are named up front and fetched on demand, so starting a kernel costs
        # nothing and a table only crosses when a cell asks for it.
        dataset_names=warehouse_tables,
        query_resolver=notebook_query,
        profiles=compute_profiles,
        credentials=credential_vendor(),
        # Read per kernel start, so setting it needs no restart.
        kernel_env=kernel_tracking_env,
        on_session_end=record_compute,
        backend=resolve_runner(project.config),
        runner_options=runner_options,
        egress=project.config.egress,
        image=resolve_image(project.config),
        scratch_root=project.workspaces_dir,
        # The same authoring loop the chat uses, so a cell can be promoted to a
        # certified derivation.
        derive_factory=make_derive_factory(
            project, store, make_runner=make_runner, dataset_names=warehouse_tables
        ),
    )
    orchestration_service = OrchestrationService(
        store=store,
        registry_provider=lambda: project.registry,
        make_runner=make_runner,
        dataset_names=warehouse_tables,
        dataset_hash=dataset_fingerprint,
        on_notify=orchestration_alert_handler(store),
        compute_backend=project.config.sandbox,
    )
    # Platform-wide search: one DuckDB file beside the application database. The
    # registry is pointed at it here, so both consumers read one index.
    search_embedder = None if project.config.search == "lexical" else OnnxEmbedder()
    search_index = SearchIndex(
        project.cache_dir / "search.duckdb",
        # Declared, not measured: measuring loads the model, and the DDL needs the
        # width at open -- a weight download in every server's boot path.
        dim=EMBED_DIM,
        # A model change makes the stored vectors incomparable, so the file rebuilds.
        embedding_model="lexical" if search_embedder is None else EMBED_MODEL,
    )
    try:
        search_index.open()
    except Exception:
        # A second server on the same project, most often; also a read-only cache
        # volume. Search degrades rather than taking the deployment down.
        logger.warning("search index unavailable; ranking in memory", exc_info=True)
        search_builder = None
        search_drain = None
    else:
        search_builder = SearchBuilder(
            search_index, Sources(store=store, models=model_service), search_embedder
        )
        project.registry.use_retriever(indexed_retriever(search_index, search_embedder))
        # Every committed write reaches the index from here, through this queue.
        pending = PendingWrites()
        search_drain = Drain(
            search_builder,
            pending,
            on_stop=install(store._engine, pending),
        )
        search_drain.start()

    # The MCP operate tools drive these services directly (in-process), the thin-adapter
    # pattern: one core, exposed over both the API routes and the MCP.
    operations.bind(orchestration_service, notebook_service)

    app = create_app(
        load_datasets=load_datasets,
        client=default_client,
        mcp_server=mcp_server,
        static_dir=_spa_dir(),
        store=store,
        derive_factory=make_derive_factory(
            project, store, make_runner=make_runner, dataset_names=warehouse_tables
        ),
        # Titles run on the designated title profile (a cheap model) when one is set,
        # else the conversation's default profile. Resolved per call so a change applies
        # without a restart.
        generate_title=lambda question: llm_title(
            question, resolve_client(store.get_config("llm.title_profile") or None)
        ),
        # Derivation trash/restore/erase needs the live registry, the sidecar store,
        # and the MCP server all at once, which only this composition root holds.
        derivation_on_trash=derivation_on_trash,
        derivation_on_restore=derivation_on_restore,
        derivation_on_erase=derivation_on_erase,
        scratch_root=project.workspaces_dir,
        sandbox=project.config.sandbox,
        egress=project.config.egress,
        sandbox_image=resolve_image(project.config),
        make_client=resolve_client,
        models=[
            {"id": m.id, "name": m.name, "provider": m.provider, "default": m.default}
            for m in MODELS
        ],
        # Run periodic housekeeping (prune old traffic and audit rows) with the
        # server; off by default so tests and embedders don't spawn a background thread.
        enable_maintenance=True,
        # Model training/registry/serving, present when the ml extra is installed;
        # None degrades those surfaces to an install hint.
        model_service=model_service,
        # Notebooks: the authoring surface, running cells against a live kernel on the
        # same sandbox backend the chat exploration uses.
        notebook_service=notebook_service,
        # Dashboards: declarative presentation surfaces whose widgets bind to certified
        # derivations, resolved through the same project runner as everything else.
        dashboard_service=service_class(DashboardService)(
            store=store,
            make_runner=make_runner,
            certified_catalog=certified_catalog,
            # A tile may bind a semantic-layer metric; resolve and existence-check it
            # through the metric service so a metric renders like any other tile.
            resolve_metric=resolve_metric_tile,
            metric_exists=lambda name: metric_service.get(name) is not None,
        ),
        # Feature store: entities + feature views over certified derivations, with
        # point-in-time historical joins and a materialized online store.
        feature_store_service=service_class(FeatureStoreService)(
            store=store, make_runner=make_runner, is_certified=is_certified
        ),
        # Lineage + catalog: the cross-artifact dependency graph (datasets → derivations
        # → models / dashboards / feature views), impact analysis, and unified search.
        lineage_service=service_class(LineageService)(
            store=store,
            registry_provider=lambda: project.registry,
            dataset_names=warehouse_tables,
            models_provider=model_edges,
        ),
        # Exploration surface: ad-hoc SQL over bound datasets (DuckDB) or a registered
        # data source (read-only load_sql), schema browsing, and one-click profiling.
        # Ungoverned by design; promoting to a derivation is the opt-in bridge.
        explore_service=ExploreService(
            store=store,
            dataset_names=warehouse_tables,
            resolve_source=explore_resolve_source,
            read_schema=explore_read_schema,
            derive_factory=make_derive_factory(
                project, store, make_runner=make_runner, dataset_names=warehouse_tables
            ),
            draft_factory=make_draft_factory(
                project, make_runner=make_runner, dataset_names=warehouse_tables
            ),
            promote_external=promote_external,
            warehouse=warehouse_service,
        ),
        # Metrics: a semantic layer where each metric is an aggregation over a certified
        # derivation, sliced by dimensions and served the same everywhere; imports and
        # exports the OSI standard. A simple metric's source must be certified.
        metric_service=metric_service,
        # Monitoring: watch a metric or derivation over time, detect anomalies against a
        # learned baseline, and alert (webhook + audit) with the oracle verdict.
        monitor_service=monitor_service,
        # Data warehouse: sync external sources (SQL DBs, files, SaaS APIs) into a
        # portable Delta Lake lakehouse (local file:// or the user's own S3/GCS/Azure),
        # queryable by DuckDB. Present only with the 'warehouse' extra; otherwise the
        # API returns an install hint, like the ml surfaces.
        warehouse_service=warehouse_service,
        # Orchestration: materialize derivations (assets) in dependency order, track
        # staleness and runs, and fire cron schedules / data-change sensors.
        orchestration_service=orchestration_service,
        # Server side of `elbi sync`: re-read derivation source files and
        # declared source data from disk so those changes go live without a restart.
        reload_project=_reload_project,
        # Renders a repo derivation's output on demand for the detail view (repo rows
        # carry source but not a stored rendering).
        render_derivation=render_derivation,
        # Passed only when a policy is declared, so `policy_for is not None` inside the
        # app answers "is a policy configured" without a second flag.
        # Signs the exportable, tamper-evident certificate the API and UI serve, under
        # the same project identity as the MCP server above.
        certificate_issuer=issuer,
        # Nothing else calls a build, so passing this is what fills the index.
        search_builder=search_builder,
        search_drain=search_drain,
    )
    app.state.reload_project = _reload_project
    _watch_project_revision(app)
    return app


def _build_client(
    model: str,
    api_key: str | None,
    base_url: str | None,
    reasoning_effort: str | None = None,
) -> LLMClient:
    """A LiteLLM client for a model config, whatever the provider.

    Every model runs through LiteLLM, a ``provider/model`` string (``openai/gpt-5``,
    ``gemini/...``, ``ollama/...``) or a bare Claude id, which LiteLLM routes to
    Anthropic, so one path serves all providers and reports token usage and cost
    uniformly. A bare id is prefixed with ``anthropic/`` to name the provider.
    """
    from elbi_agent.litellm_client import LiteLLMClient

    litellm_model = model if "/" in model else f"anthropic/{model}"
    return LiteLLMClient(
        model=litellm_model,
        api_key=api_key,
        base_url=base_url or None,
        # A reasoning model's effort ("low"/"medium"/"high"). Resolved by the caller:
        # per-profile first, then LLM_REASONING_EFFORT as the seed for a fresh install.
        reasoning_effort=reasoning_effort or None,
    )


def _open_when_ready(url: str, timeout: float = 60.0) -> None:
    """Open ``url`` in the default browser once it actually responds.

    A fixed delay would race the warmup (model2vec load, MLflow table init can
    take 15-20s on first run): poll instead of guessing, so the browser never
    opens onto a still-refusing connection.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)  # noqa: S310 - localhost only
            break
        except Exception:
            time.sleep(0.3)
    else:
        return
    try:
        webbrowser.open(url)
    except Exception as exc:  # pragma: no cover - no display, no default browser, etc.
        # Not reaching a browser is not a startup failure: the server is already up
        # and the URL is on stdout. Recorded rather than swallowed so it is visible.
        logger.debug("could not open a browser at %s: %s", url, exc)


def serve(
    directory: str | Path = ".",
    host: str = "127.0.0.1",
    port: int = 7700,
    model: str | None = None,
    open_browser: bool = True,
) -> None:
    """Run the app with uvicorn, bound to localhost by default."""
    from .logging_config import configure_logging

    # Configure the root logger from LOG_LEVEL/LOG_FORMAT, then give uvicorn no config
    # of its own (log_config=None) so its logs use the same handler/format.
    configure_logging()
    url = f"http://{host}:{port}"
    logging.getLogger(__name__).info(
        "Starting the chat app at %s (first run can take ~15-20s while it warms up)...",
        url,
    )
    if open_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(build(directory, model=model), host=host, port=port, log_config=None)


def _spa_dir() -> Path | None:
    """The packaged SPA build: bundled into the wheel, or built here from source.

    A published wheel ships web/dist already built. A source checkout does not --
    it's git-ignored, normally produced by CI (or `just web-build`) before packaging
    -- so if the frontend's source tree is a sibling of this checkout with no build
    yet, or one older than its own sources, build it now instead of silently serving
    a chat app with no web UI.
    """
    dist_dir = Path(__file__).parent / "web" / "dist"
    web_src_dir = Path(__file__).parents[2] / "web"
    if web_src_dir.is_dir():
        _ensure_spa_built(web_src_dir, dist_dir)
    return dist_dir if dist_dir.is_dir() else None


def _spa_stale(web_src_dir: Path, dist_dir: Path) -> bool:
    """Whether the build predates its own sources (or doesn't exist at all)."""
    index_html = dist_dir / "index.html"
    if not index_html.is_file():
        return True
    built_at = index_html.stat().st_mtime
    return any(
        f.stat().st_mtime > built_at
        for f in (web_src_dir / "src").rglob("*")
        if f.is_file()
    )


def _ensure_spa_built(web_src_dir: Path, dist_dir: Path) -> None:
    """Run `npm run build` for a source checkout with no (or stale) SPA build."""
    if not _spa_stale(web_src_dir, dist_dir):
        return
    logger = logging.getLogger(__name__)
    npm = shutil.which("npm")
    if npm is None:
        logger.warning(
            "No frontend build found at %s and npm is not on PATH; the chat app "
            "will start with no web UI. Install Node.js and run `npm run build` in "
            "%s, then restart.",
            dist_dir,
            web_src_dir,
        )
        return
    try:
        if not (web_src_dir / "node_modules").is_dir():
            logger.info("Installing web UI dependencies (npm ci)...")
            # npm is resolved via shutil.which and the args are static, so no untrusted
            # input reaches the shell. Bounded so a hung npm cannot stall startup.
            subprocess.run(  # noqa: S603
                [npm, "ci"], cwd=web_src_dir, check=True, timeout=600
            )
        logger.info("Building web UI (npm run build)...")
        subprocess.run(  # noqa: S603
            [npm, "run", "build"], cwd=web_src_dir, check=True, timeout=600
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "Building the web UI failed (%s); the chat app will start with no web "
            "UI. Run `npm run build` in %s manually to see the full error.",
            exc,
            web_src_dir,
        )
