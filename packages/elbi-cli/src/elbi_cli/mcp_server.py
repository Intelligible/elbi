"""Build a local MCP server that exposes a project's derivations.

A served, certified derivation becomes a ``run_<name>`` tool, plus a
``elbi://derivation/<name>`` resource when it has no parameters. Internal
derivations (no serve contract) and proposed ones are not exposed.

``search_derivations`` is always registered, and ``propose_derivation`` (the
agent-authoring loop) when a sandbox runner is available. Serving goes through
:class:`~elbi_cli.serving.Serving` for stale-while-revalidate.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import importlib.util
import inspect
import logging
import os
import sys
import threading
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from mcp.server import MCPServer, NotificationOptions
from mcp.server.mcpserver import Context as McpContext
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    CallToolResult,
    ContentBlock,
    ResourceLink,
    TextContent,
    ToolAnnotations,
    ToolListChangedNotification,
)

from elbi_core import (
    AuditSink,
    CertificateIssuer,
    CertificationPolicy,
    DataContract,
    Dataset,
    Derivation,
    InMemoryJobStore,
    JobRunner,
    Param,
    Registry,
    Runner,
    Serve,
    SessionManager,
    analyze_structure,
    author,
    profile_columns,
    search_components,
    submit_code_job,
    suggest_contract,
    verify_all,
    verify_calibration,
    verify_certificate,
    verify_classification,
    verify_clusters,
    verify_comparison,
    verify_contract,
    verify_correlation,
    verify_counts,
    verify_did,
    verify_effect,
    verify_equivalence,
    verify_experiment,
    verify_extrapolation,
    verify_fairness,
    verify_forecast,
    verify_groups,
    verify_iv,
    verify_leakage,
    verify_logistic,
    verify_missingness,
    verify_multiverse,
    verify_normality,
    verify_overlap,
    verify_powerlaw,
    verify_prediction,
    verify_proportional_hazards,
    verify_proportions,
    verify_rdd,
    verify_regression,
    verify_reliability,
    verify_sensitivity,
    verify_stationarity,
    verify_survival,
    verify_trend,
)
from elbi_core import (
    serve as serve_builders,
)
from elbi_core.cache import LocalCacheStore, derivation_tag
from elbi_core.config import ColumnSpec, DatasetSpec
from elbi_core.data import Table
from elbi_core.errors import CertificateError, ElbiError
from elbi_core.tracking import CertifiedRun, RunLog, run_from_author
from elbi_core.tracking.mlflow import emit_run

from .authored import AuthoredStore
from .serving import ServeOutcome, Serving

logger = logging.getLogger(__name__)


def _emit_local_run(run: CertifiedRun) -> None:
    """Export a certified run to MLflow when ``MLFLOW_TRACKING_URI`` is set."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        emit_run(run, tracking_uri=uri)


#: Compute libraries an authoring agent commonly reaches for; reported by
#: ``sandbox_environment`` so it knows what it can import before writing code.
_COMMON_LIBS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "polars",
    "pyarrow",
    "matplotlib",
)

#: The `_meta` key a certified resource read carries its signed certificate under.
_CERTIFICATE_META_KEY = "intelligible.ai/certificate"

#: Ceiling on a caller-supplied ``train_model`` search budget, in seconds. The budget
#: arrives over the wire, and an unbounded one holds a worker thread for as long as it
#: asks for. The app's chat path clamps to the same figure before training inline.
_MAX_TRAIN_BUDGET = 240.0

#: How each serve format maps to a resource MIME type.
_MIME = {
    "table": "text/markdown",
    "markdown": "text/markdown",
    "json": "application/json",
    "text": "text/plain",
    "components": "application/json",
}


def resource_uri(name: str) -> str:
    """The MCP resource URI for a derivation."""
    return f"elbi://derivation/{name}"


def tool_name(name: str) -> str:
    """The MCP tool name for running a derivation."""
    return f"run_{name}"


# -- what each tool does to the world ---------------------------------------------
# The protocol's hints, and they are not decoration: a client must assume an unannotated
# tool is destructive, non-idempotent and open-world, so a gate that only computes a
# statistic is offered behind the same confirmation as a delete. Nothing here reaches
# beyond this deployment's own data, so `openWorldHint` is false throughout.

#: Computes or reads and changes nothing.
READS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
#: Writes, but only adds: a new derivation, a new model version, a materialized asset.
ADDS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
#: Writes, and calling it twice leaves the same state (an alias move, a cancellation).
SETTLES = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
#: Removes or overwrites something a person may still want.
REMOVES = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
#: Runs caller-supplied code, so its effects are whatever that code did. Sandboxed, but
#: the hint describes what the tool may do rather than what it is confined to.
EXECUTES = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)

#: Every statically-registered tool. Coverage is asserted, so adding a tool without
#: deciding what it does is an error here rather than a pessimistic default in a client.
_ANNOTATIONS: dict[str, ToolAnnotations] = {
    # discovery and description
    "describe_warehouse": READS,
    "table_schema": READS,
    "sample_rows": READS,
    "describe_dataset": READS,
    "profile_dataset": READS,
    "structure_map": READS,
    "search_derivations": READS,
    "search_components": READS,
    "asset_status": READS,
    "job_status": READS,
    "sandbox_environment": READS,
    "list_models": READS,
    "suggest_contract": READS,
    # the oracle: every gate computes a verdict and writes nothing
    **{
        f"verify_{gate}": READS
        for gate in (
            "analysis",
            "comparison",
            "groups",
            "equivalence",
            "leakage",
            "fairness",
            "sensitivity",
            "reliability",
            "normality",
            "missingness",
            "iv",
            "rdd",
            "did",
            "counts",
            "overlap",
            "powerlaw",
            "correlation",
            "trend",
            "regression",
            "prediction",
            "experiment",
            "proportions",
            "forecast",
            "calibration",
            "logistic",
            "survival",
            "clusters",
            "robustness",
            "contract",
            "classification",
            "extrapolation",
            "proportional_hazards",
            "stationarity",
        )
    },
    "verify": READS,
    # authoring and runs
    "propose_derivation": ADDS,
    "train_model": ADDS,
    "materialize": ADDS,
    "backfill": ADDS,
    "run_workflow": ADDS,
    # a notebook cell is caller-supplied code
    "run_notebook": EXECUTES,
    "run_code": EXECUTES,
    "bash": EXECUTES,
    # which version serves, and stopping work already started
    "promote_model": SETTLES,
    "cancel_job": SETTLES,
    # scoring records an inference, so it is not read-only
    "predict": ADDS,
    "delete_derivation": REMOVES,
}


def tool_effects() -> dict[str, ToolAnnotations]:
    """Each statically-registered tool's declared effect on the world.

    Exposed so a client can act on what a tool does without re-deriving it: the table
    here is the same one every tool is annotated from, and a tool added without an entry
    fails the build rather than defaulting.

    A per-derivation ``run_<name>`` tool is deliberately absent. It only reads, so an
    unlisted tool of that shape can be treated as read-only, and anything else unlisted
    as unknown.
    """
    return dict(_ANNOTATIONS)


def _annotate(server: MCPServer) -> None:
    """Attach each registered tool's hints, and refuse an undeclared one."""
    tools = server._tool_manager._tools
    for name, tool in tools.items():
        if tool.annotations is not None:
            continue  # a derivation's own tool, annotated where it is registered
        annotations = _ANNOTATIONS.get(name)
        if annotations is None:
            raise RuntimeError(
                f"MCP tool {name!r} declares no behaviour hints. Add it to "
                "_ANNOTATIONS: an unannotated tool is presented to every client as "
                "destructive and non-idempotent."
            )
        tool.annotations = annotations


def build_server(
    registry: Registry,
    make_runner: Callable[[], Runner],
    *,
    enable_propose: bool = False,
    certification: CertificationPolicy | None = None,
    load_dataset: Callable[[str], Table] | None = None,
    dataset_specs: tuple[DatasetSpec, ...] = (),
    warehouse_schema: Callable[[], dict[str, Any]] | None = None,
    warehouse_sample: Callable[[str, int], list[dict[str, Any]]] | None = None,
    name: str = "elbi",
    audit: AuditSink | None = None,
    authored_store: AuthoredStore | None = None,
    issuer: CertificateIssuer | None = None,
    run_log: RunLog | None = None,
    cache_dir: Path | None = None,
    scratch_dir: Path | None = None,
    backend: str = "subprocess",
    egress: str | Sequence[str] = "full",
    image: str | None = None,
    instructions: str | None = None,
    operations: Operations | None = None,
    on_author: Callable[[str, dict[str, Any]], object] | None = None,
    on_delete: Callable[[str], object] | None = None,
    embedder: Any | None = None,
) -> MCPServer:
    """Construct an :class:`MCPServer` exposing certified, served derivations.

    Transport settings (host, port, mount path) belong to whoever serves the result:
    :meth:`MCPServer.run` for a standalone process,
    :meth:`MCPServer.streamable_http_app`
    for one mounted inside another ASGI app.

    Proposed (uncertified) derivations are never exposed. ``search_derivations`` is
    always registered; ``propose_derivation`` (the agent-authoring loop) is
    registered when ``enable_propose`` is set. ``make_runner`` must route agent-
    authored derivations to a sandbox (see :class:`~elbi.RoutingExecutor`).

    ``certification`` is the policy that decides whether a verified proposal is
    served immediately; it defaults to auto-certify-on-verify
    (:data:`~elbi.authoring.DEFAULT_CERTIFICATION`). Pass
    :class:`~elbi.ManualCertification` to require an explicit
    ``elbi certify`` instead.

    ``audit`` is the observability seam: the default records nothing, while a
    deployment passes a sink that writes an append-only trail of every invocation.

    ``scratch_dir`` is a persistent working directory for the exploration session (see
    :func:`_register_run_code`), so files survive across calls; ``None`` keeps each call
    ephemeral. ``backend`` selects where that session runs: ``"subprocess"`` (default)
    or ``"docker"`` (an isolated container where ``run_code`` and ``bash`` share it).

    ``embedder`` is reused for ``search_components``
    (see :func:`_register_component_search`), the same object already loaded for
    derivation search where one is available, so it is loaded once rather than
    twice. ``None`` (the default, and the only option for ``elbi-cli``, which does
    not depend on an embedding model) falls back to lexical search alone for
    components.
    """
    server = MCPServer(name, instructions=instructions)
    # Agents add and remove derivation tools at runtime, so the tool list is
    # dynamic: advertise it (MCP `tools.listChanged`) and emit the notification on
    # change, so a connected client can discover a just-authored `run_<name>`.
    _advertise_tool_changes(server)
    serving = Serving(registry, make_runner, audit=audit)
    for derivation in registry:
        if derivation.is_served and derivation.is_certified:
            # Restore a stored verification attestation, and its signed certificate,
            # so a derivation certified through the effect gate still serves both
            # after a restart. The certificate is persisted at certification time and
            # re-served verbatim; a derivation certified before this feature has an
            # attestation but no certificate and simply serves without one.
            record = (
                authored_store.read(derivation.name)
                if authored_store is not None
                else None
            )
            attestation = record.get("attestation") if record else None
            certificate = record.get("certificate") if record else None
            if certificate is not None:
                try:
                    verify_certificate(
                        certificate,
                        public_key=issuer.public_key if issuer is not None else None,
                    )
                except CertificateError as exc:
                    logger.warning(
                        "dropping stored certificate for %r: %s",
                        derivation.name,
                        exc,
                    )
                    certificate = None
            _register(server, derivation, serving, attestation, certificate)
    _register_search(server, registry)
    _register_component_search(server, registry, serving, embedder)
    if operations is not None:
        _register_operate(server, operations)
    if warehouse_schema is not None:
        _register_warehouse(server, warehouse_schema, warehouse_sample)
    if load_dataset is not None:
        _register_describe(server, load_dataset, dataset_specs)
        _register_structure(server, load_dataset, dataset_specs)
        _register_verify(server, load_dataset)
        _register_contract(server, load_dataset)
    if enable_propose:
        _register_environment(server)
        _register_run_code(
            server, load_dataset, dataset_specs, scratch_dir, backend, egress, image
        )
        _register_delete(server, registry, authored_store, cache_dir, on_delete)
        _register_ml(server, load_dataset, cache_dir)
        _register_propose(
            server,
            registry,
            make_runner,
            serving,
            certification,
            authored_store,
            issuer,
            run_log,
            load_dataset,
            on_author,
        )
    # Last, so it sees every tool this build registered.
    _annotate(server)
    return server


def _advertise_tool_changes(server: MCPServer) -> None:
    """Make a handshake-era client see `tools.listChanged`.

    At 2026-07-28 the flag is derived from serving `subscriptions/listen`, which
    ``MCPServer`` always does, so that era needs nothing. The handshake era still builds
    its initialize response from notification options with every flag off, and the
    session manager calls the builder with no arguments, so the capability is never
    advertised otherwise. Wrap the builder to turn the tools flag on while leaving any
    explicit caller options untouched.
    """
    low = server._lowlevel_server
    build = low.create_initialization_options

    def with_tools_changed(
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
        extensions: dict[str, dict[str, Any]] | None = None,
    ) -> Any:
        return build(
            notification_options=notification_options
            or NotificationOptions(tools_changed=True),
            experimental_capabilities=experimental_capabilities,
            extensions=extensions,
        )

    low.create_initialization_options = with_tools_changed  # type: ignore[method-assign]


async def _notify_tools_changed(ctx: McpContext[Any, Any]) -> None:
    """Emit `tools/list_changed` so a connected client refreshes its tool list.

    Both eras are served, and they deliver differently: 2026-07-28 publishes to
    `subscriptions/listen` subscribers, while a handshake-era client is told on this
    call's own stream, which needs the originating request id because a per-request
    connection has no standalone channel. Best-effort on each: a transport with no live
    channel is skipped, and a failed notification never breaks the originating call.
    """
    with contextlib.suppress(Exception):
        await ctx.notify_tools_changed()
    with contextlib.suppress(Exception):
        await ctx.session.send_notification(
            ToolListChangedNotification(), related_request_id=ctx.request_id
        )


def _register(
    server: MCPServer,
    derivation: Derivation,
    serving: Serving,
    attestation: dict[str, Any] | None = None,
    certificate: dict[str, Any] | None = None,
) -> None:
    serve_contract = derivation.serve
    if serve_contract is None:  # pragma: no cover - filtered by caller
        return
    name = derivation.name
    description = _describe(derivation)

    if derivation.params:
        server.tool(name=tool_name(name), description=description, annotations=READS)(
            _build_param_tool(
                name, derivation.params, serving, attestation, certificate
            )
        )
        return

    mime = _MIME.get(serve_contract.format, "text/plain")

    async def read() -> str:
        # A resource read returns the full rendering, not the trimmed preview.
        return (await serving.serve(name)).text

    read.__name__ = name
    read.__doc__ = description
    # The signed certificate rides in the resource's `_meta` so every read (and the
    # resource listing) carries the proof.
    meta = {_CERTIFICATE_META_KEY: certificate} if certificate is not None else None
    server.resource(
        resource_uri(name),
        name=name,
        description=description,
        mime_type=mime,
        meta=meta,
    )(read)

    async def run() -> CallToolResult:
        return _tool_result(await serving.serve(name), name, attestation, certificate)

    run.__name__ = tool_name(name)
    run.__doc__ = f"Run the {name} derivation and return its served output."
    server.tool(name=tool_name(name), description=run.__doc__, annotations=READS)(run)


def _build_param_tool(
    name: str,
    params: dict[str, Param] | Any,
    serving: Serving,
    attestation: dict[str, Any] | None = None,
    certificate: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Build an async tool whose signature mirrors the declared parameters."""

    async def run(**kwargs: Any) -> CallToolResult:
        return _tool_result(
            await serving.serve(name, kwargs), name, attestation, certificate
        )

    signature_params: list[inspect.Parameter] = []
    for pname, spec in params.items():
        base = spec.annotation()
        if spec.required:
            annotation: Any = base
            default: Any = inspect.Parameter.empty
        else:
            annotation = base | None
            default = spec.default
        signature_params.append(
            inspect.Parameter(
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
                default=default,
            )
        )

    run.__name__ = tool_name(name)
    # MCPServer reads __signature__ to build the input schema; set it through an
    # Any-typed view since functions don't declare that attribute statically.
    configurable: Any = run
    configurable.__signature__ = inspect.Signature(signature_params)
    run.__annotations__ = {p.name: p.annotation for p in signature_params} | {
        "return": CallToolResult
    }
    return run


def _tool_result(
    outcome: ServeOutcome,
    name: str,
    attestation: dict[str, Any] | None = None,
    certificate: dict[str, Any] | None = None,
) -> CallToolResult:
    """Shape a serve outcome into an MCP tool result.

    The text content is the preview. A table also carries the full rows in
    ``structuredContent`` and a ``resource_link`` to its resource; other formats
    return text only. A verified effect derivation adds its verification record
    under ``structuredContent.verification`` and its signed certificate under
    ``structuredContent.certificate`` so any client can trust and export the result.

    """
    content: list[ContentBlock] = [
        TextContent(type="text", text=outcome.preview or outcome.text)
    ]
    if outcome.structured is not None:
        content.append(
            ResourceLink(
                type="resource_link",
                uri=resource_uri(name),
                name=name,
                description=f"The full {name} result.",
                mime_type="application/json",
            )
        )
    extras: dict[str, Any] = {}
    if attestation is not None:
        extras["verification"] = attestation
    if certificate is not None:
        extras["certificate"] = certificate
    structured = (
        {**(outcome.structured or {}), **extras}
        if (outcome.structured or extras)
        else None
    )
    return CallToolResult(content=content, structured_content=structured)


def _describe(derivation: Derivation) -> str:
    description = derivation.description or f"The {derivation.name} derivation."
    if not derivation.params:
        return description
    lines = [description, "", "Parameters:"]
    for pname, spec in derivation.params.items():
        kind: str = spec.type
        if spec.type == "array" and spec.items:
            kind = f"array of {spec.items}"
        suffix = "" if spec.required else " (optional)"
        detail = f": {spec.description}" if spec.description else ""
        lines.append(f"- {pname} ({kind}){suffix}{detail}")
    return "\n".join(lines)


def _register_warehouse(
    server: MCPServer,
    warehouse_schema: Callable[[], dict[str, Any]],
    warehouse_sample: Callable[[str, int], list[dict[str, Any]]] | None,
) -> None:
    """Register warehouse-schema tools so an agent can learn the data before building.

    ``describe_warehouse`` lists every table with its columns and row counts;
    ``table_schema`` returns one table's columns and types; ``sample_rows`` returns a
    few example rows. This is the context a coding agent needs to author a derivation,
    metric, or dashboard against the warehouse without guessing column names or types.
    """
    import json

    def _tables() -> list[dict[str, Any]]:
        return list(warehouse_schema().get("tables", []))

    async def describe_warehouse() -> str:
        """List every warehouse table with its columns, types, and row count."""
        tables = _tables()
        if not tables:
            return "The warehouse has no tables yet. Add a source and sync it first."
        lines = [f"{len(tables)} warehouse table(s):", ""]
        for table in tables:
            cols = ", ".join(
                f"{c['name']}:{c.get('type', '?')}" for c in table.get("columns", [])
            )
            rows = table.get("rows")
            count = f" ({rows} rows)" if rows is not None else ""
            lines.append(f"- {table['name']}{count}: {cols}")
        return "\n".join(lines)

    async def table_schema(name: str) -> str:
        """The columns and types of one warehouse table, as JSON."""
        for table in _tables():
            if table["name"] == name:
                return json.dumps(
                    {
                        "table": name,
                        "rows": table.get("rows"),
                        "columns": table.get("columns", []),
                    },
                    indent=2,
                )
        available = ", ".join(t["name"] for t in _tables()) or "(none)"
        raise MCPError(
            INVALID_PARAMS,
            f"No warehouse table named {name!r}. Available: {available}.",
        )

    async def sample_rows(name: str, limit: int = 10) -> str:
        """A few example rows from a warehouse table, as JSON (default 10)."""
        if warehouse_sample is None:
            return "Row sampling is not available on this server."
        known = {t["name"] for t in _tables()}
        if name not in known:
            available = ", ".join(sorted(known)) or "(none)"
            raise MCPError(
                INVALID_PARAMS,
                f"No warehouse table named {name!r}. Available: {available}.",
            )
        rows = await asyncio.to_thread(warehouse_sample, name, max(1, min(limit, 100)))
        return json.dumps(rows, indent=2, default=str)

    server.tool(name="describe_warehouse", description=describe_warehouse.__doc__)(
        describe_warehouse
    )
    server.tool(name="table_schema", description=table_schema.__doc__)(table_schema)
    # Advertised only when there is a sampler to call: a listed tool that can only
    # refuse reads to an agent as a broken tool, and it will retry.
    if warehouse_sample is not None:
        server.tool(name="sample_rows", description=sample_rows.__doc__)(sample_rows)


class Operations(Protocol):
    """The side-effecting operations an agent may trigger over MCP.

    A thin seam over the app's orchestration and notebook services: each method runs
    to completion and returns a plain result dict (no app types leak into the MCP
    layer), so ``build_server`` can wrap them as tools without depending on the app.
    """

    def materialize(
        self, *, selection: str, assets: list[str] | None, include_downstream: bool
    ) -> dict[str, Any]:
        """Materialize a selection of assets; return the run summary."""
        ...

    def run_workflow(self, name: str) -> dict[str, Any]:
        """Run a named workflow; return the run summary with per-step outcomes."""
        ...

    def run_notebook(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run a named notebook to completion; return a cell-run summary."""
        ...

    def backfill(self, *, asset: str, param: str, values: list[str]) -> dict[str, Any]:
        """Backfill a parameterized asset across partition values; return the run."""
        ...

    def asset_status(self) -> list[dict[str, Any]]:
        """Every asset's freshness and verdict."""
        ...


def _register_operate(server: MCPServer, operations: Operations) -> None:
    """Register the operate tools: materialize, run a workflow/notebook, backfill.

    These are state-changing (the client is expected to confirm before calling) and each
    runs synchronously and reports the outcome, the way an agent wants to observe a run
    rather than poll for it.
    """

    def _run_line(result: dict[str, Any]) -> str:
        ok = "ok" if result.get("ok") else "FAILED"
        return f"run {result.get('run_id')}: {ok}"

    async def materialize(
        selection: str = "stale",
        assets: list[str] | None = None,
        include_downstream: bool = False,
    ) -> str:
        """Materialize assets, running to completion.

        ``selection`` is 'stale' or 'all', or pass an explicit ``assets`` list (with
        ``include_downstream`` to add dependents). Reports materialized/skipped/failed.
        """
        result = await asyncio.to_thread(
            operations.materialize,
            selection=selection,
            assets=assets,
            include_downstream=include_downstream,
        )
        return (
            f"{_run_line(result)}\n"
            f"  materialized: {result.get('materialized', [])}\n"
            f"  skipped: {result.get('skipped', [])}\n"
            f"  failed: {result.get('failed', [])}"
        )

    async def run_workflow(name: str) -> str:
        """Run a workflow by name to completion.

        Reports each step's outcome (succeeded / failed / excluded by its run-if rule).
        """
        try:
            result = await asyncio.to_thread(operations.run_workflow, name)
        except (KeyError, ValueError) as exc:
            return f"Error: {exc}"
        steps = ", ".join(f"{s['id']}={s['state']}" for s in result.get("steps", []))
        return f"{_run_line(result)}\n  steps: {steps}"

    async def run_notebook(name: str, params: dict[str, Any] | None = None) -> str:
        """Run a notebook by name to completion.

        ``params`` overrides are spliced in as an injected cell. Reports how many cells
        ran and which failed.
        """
        try:
            result = await asyncio.to_thread(
                operations.run_notebook, name, params or {}
            )
        except (KeyError, ValueError) as exc:
            return f"Error: {exc}"
        failed = result.get("failed", [])
        status = "ok" if result.get("ok") else f"FAILED: {failed}"
        return f"notebook {name!r}: ran {result.get('ran', 0)} cell(s), {status}"

    async def backfill(asset: str, param: str, values: list[str]) -> str:
        """Backfill a parameterized asset once per partition value.

        Each value runs the derivation with ``{param: value}``. Reports failed
        partitions.
        """
        result = await asyncio.to_thread(
            operations.backfill, asset=asset, param=param, values=values
        )
        return (
            f"{_run_line(result)}\n"
            f"  partitions: {result.get('partitions')}, failed: {result.get('failed')}"
        )

    async def asset_status() -> str:
        """List every asset with its freshness and oracle verdict (read-only)."""
        rows = await asyncio.to_thread(operations.asset_status)
        if not rows:
            return "No assets."
        lines = [
            f"- {r['asset']}: {r['status']}"
            + (f" [{r['verdict']}]" if r.get("verdict") else "")
            for r in rows
        ]
        return "\n".join(lines)

    server.tool(name="asset_status", description=asset_status.__doc__)(asset_status)
    server.tool(name="materialize", description=materialize.__doc__)(materialize)
    server.tool(name="run_workflow", description=run_workflow.__doc__)(run_workflow)
    server.tool(name="run_notebook", description=run_notebook.__doc__)(run_notebook)
    server.tool(name="backfill", description=backfill.__doc__)(backfill)


def _register_search(server: MCPServer, registry: Registry) -> None:
    """Register the retrieval entry point over the certified, served derivations."""

    async def search_derivations(query: str, limit: int = 5) -> str:
        # Rank only served, certified derivations so a proposed one can never crowd
        # a real match out of the top `limit`.
        eligible = [d for d in registry if d.is_served and d.is_certified]
        matches = registry.retriever.search(query, eligible, limit=limit)
        if not matches:
            return f"No derivations match {query!r}."
        lines = [f"Found {len(matches)} derivation(s) for {query!r}:"]
        for derivation in matches:
            summary = derivation.description or "(no description)"
            lines.append(
                f"- {derivation.name}: {summary} (run: {tool_name(derivation.name)})"
            )
        return "\n".join(lines)

    search_derivations.__doc__ = (
        "Search the available derivations by keyword and return the best matches "
        "with the tool name to run each. Use this to find a derivation instead of "
        "scanning every tool."
    )
    server.tool(name="search_derivations", description=search_derivations.__doc__)(
        search_derivations
    )


def _register_component_search(
    server: MCPServer,
    registry: Registry,
    serving: Serving,
    embedder: Any | None,
) -> None:
    """Register ``search_components``.

    Retrieval over every served, certified ``components``-format derivation's
    current facts. A thin, rebuild-every-call index (see
    :func:`elbi_core.components.search_components`'s docstring for why): each call
    re-serves every eligible derivation through the same :class:`Serving` cache
    ``run_<name>`` uses, so it costs nothing beyond what is already cached, then
    ranks the combined corpus. There is no persistence across restarts and no
    incremental re-indexing; :mod:`elbi.search`'s chunked, persisted index (already
    used for a derivation's own fields) is the right home once this needs to run at
    real scale.
    """

    async def search_components_tool(query: str, limit: int = 5) -> CallToolResult:
        # Recomputed per call, like search_derivations: a derivation can be authored
        # or trashed between calls.
        candidates = [
            derivation
            for derivation in registry
            if derivation.is_served
            and derivation.is_certified
            and derivation.serve is not None
            and derivation.serve.format == "components"
        ]
        corpus: list[dict[str, Any]] = []
        for derivation in candidates:
            try:
                outcome = await serving.serve(derivation.name)
            except Exception:
                # One derivation failing to serve (e.g. a required param with no
                # default, a transient error) must not sink the whole search.
                logger.warning(
                    "search_components: could not serve %r",
                    derivation.name,
                    exc_info=True,
                )
                continue
            if outcome.structured is not None:
                corpus.extend(outcome.structured.get("components", []))

        matches = search_components(query, corpus, limit=limit, embedder=embedder)
        if not matches:
            text = f"No components match {query!r}."
        else:
            lines = [f"Found {len(matches)} component(s) for {query!r}:"]
            lines.extend(f"- {match.get('statement', '')}" for match in matches)
            text = "\n".join(lines)
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content={"components": matches, "count": len(matches)},
        )

    search_components_tool.__doc__ = (
        "Search grounded, natural-language facts (OpenReasoningComponents) drawn "
        "from every served components-format derivation. Returns matched "
        "statements as text, and the full component objects -- with evidence, "
        "relations and provenance -- as structured content."
    )
    server.tool(name="search_components", description=search_components_tool.__doc__)(
        search_components_tool
    )


def _register_structure(
    server: MCPServer,
    load_dataset: Callable[[str], Table],
    dataset_specs: tuple[DatasetSpec, ...],
) -> None:
    """Register structure discovery: the dataset's relational skeleton.

    Deterministic, computed from values: which columns are identifiers, which are
    derived/collinear, which determine others, and which share information (the
    candidates for a confounder). An agent reads this to ground a hypothesis before
    analyzing, rather than guessing how columns relate.
    """
    listed = ", ".join(s.name for s in dataset_specs) or "(none declared)"

    async def structure_map(name: str) -> str:
        try:
            table = load_dataset(name)
        except ElbiError as exc:
            raise MCPError(
                INVALID_PARAMS,
                f"Could not load dataset {name!r}: {exc}\nAvailable: {listed}.",
            ) from exc
        return analyze_structure(table.rows, table.columns).render()

    structure_map.__doc__ = (
        "Discover a dataset's relational structure before analyzing it: identifier "
        "columns, derived/collinear columns, functional dependencies, and which "
        "columns share information (a likely confounder when one drives another). "
        "Computed deterministically from the data. Use it to ground an analysis: to "
        "pick controls for a 'what affects X' question, avoid collinear regressors, "
        "and skip identifiers. Pair with describe_dataset (meaning) and run_code "
        "(verify)."
    )
    server.tool(name="structure_map", description=structure_map.__doc__)(structure_map)


def _load_or_refuse(load_dataset: Callable[[str], Table], dataset: str) -> Table:
    """Load a dataset, or refuse the call.

    A dataset the caller named but the project does not have is a bad argument, not an
    answer, so it leaves as an MCP error rather than as text a model has to read to
    discover the call failed.
    """
    try:
        return load_dataset(dataset)
    except ElbiError as exc:
        raise MCPError(
            INVALID_PARAMS, f"Could not load dataset {dataset!r}: {exc}"
        ) from exc


def _register_verify(server: MCPServer, load_dataset: Callable[[str], Table]) -> None:
    """Register the deterministic verification oracle.

    The soundness gate for a claimed effect: it re-estimates the effect under
    confounding, collider-control, fragility, outlier, and negative-control
    perturbations and reports whether it survives. It is the data-decidable half of
    the oracle; causal direction is surfaced as a caveat for the agent to resolve
    with world knowledge rather than assumed.
    """

    async def verify_analysis(
        dataset: str,
        x: str,
        y: str,
        controls: list[str] | None = None,
        negative_controls: list[str] | None = None,
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_effect,
            table.rows,
            x,
            y,
            controls=controls or [],
            negative_controls=negative_controls or [],
        )
        return report.render()

    verify_analysis.__doc__ = (
        "Verify that a claimed effect of `x` on `y` in a dataset is sound, before "
        "you report it. Tries to break the claim the way a careful reviewer would: "
        "an observed confounder that overturns it, a collider among `controls` that "
        "created it, fragility under resampling, reliance on outliers, and (given "
        "`negative_controls`, variables x cannot cause) latent confounding. Returns "
        "sound / unsound / inconclusive with the pivotal issue. It does NOT decide "
        "causal direction (x→y vs y→x); confirm that with domain knowledge."
    )
    server.tool(name="verify_analysis", description=verify_analysis.__doc__)(
        verify_analysis
    )

    _load = partial(_load_or_refuse, load_dataset)

    async def verify_comparison_tool(
        dataset: str, group: str, value: str, subgroup: str | None = None
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_comparison, table.rows, group, value, subgroup=subgroup
        )
        return f"Difference in '{value}' across '{group}':\n\n{report.render()}"

    verify_comparison_tool.__doc__ = (
        "Verify a difference in `value` between the two groups of `group` before you "
        "report it: significance, effect size, reliance on outliers, and, given a "
        "`subgroup` column, whether the difference reverses within subgroups "
        "(Simpson's paradox). Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_comparison", description=verify_comparison_tool.__doc__)(
        verify_comparison_tool
    )

    async def verify_groups_tool(dataset: str, group: str, value: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_groups, table.rows, group, value)
        head = f"Difference in '{value}' across the groups of '{group}':"
        return f"{head}\n\n{report.render()}"

    verify_groups_tool.__doc__ = (
        "Verify a difference in `value` across three or more groups of `group` "
        "(Welch's ANOVA, with a rank-based Kruskal-Wallis robustness check and Dunn "
        "post-hoc to name the differing pairs). Use this instead of verify_comparison "
        "when the group has more than two levels. Returns sound / unsound / "
        "inconclusive."
    )
    server.tool(name="verify_groups", description=verify_groups_tool.__doc__)(
        verify_groups_tool
    )

    async def verify_equivalence_tool(
        dataset: str, group: str, value: str, sesoi: float
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_equivalence, table.rows, group, value, sesoi=sesoi
        )
        head = f"Equivalence of '{value}' between the groups of '{group}':"
        return f"{head}\n\n{report.render()}"

    verify_equivalence_tool.__doc__ = (
        "Verify that two groups are *equivalent* on `value` within ±`sesoi` (two "
        "one-sided tests). A non-significant difference is NOT equivalence; use this "
        "to soundly claim 'no meaningful difference'. `sesoi` is the smallest effect "
        "that would matter, in the units of `value`; it is a domain judgement the "
        "data cannot supply. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_equivalence", description=verify_equivalence_tool.__doc__)(
        verify_equivalence_tool
    )

    async def verify_leakage_tool(
        dataset: str, target: str, features: list[str] | None = None
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_leakage, table.rows, target, features=features
        )
        return f"Leakage screen for predicting '{target}':\n\n{report.render()}"

    verify_leakage_tool.__doc__ = (
        "Screen the features for data leakage before trusting a model that predicts "
        "`target`: flag any single feature that predicts the target near-perfectly (a "
        "likely proxy or consequence of the label, unavailable at prediction time). "
        "This is the 'too good to be true' check; confirm a flagged feature's "
        "provenance. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_leakage", description=verify_leakage_tool.__doc__)(
        verify_leakage_tool
    )

    async def verify_fairness_tool(
        dataset: str, group: str, y_true: str, y_pred: str
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_fairness, table.rows, group, y_true, y_pred
        )
        return f"Fairness across '{group}':\n\n{report.render()}"

    verify_fairness_tool.__doc__ = (
        "Verify a binary classifier does not disparately harm a subgroup: compare the "
        "selection rate, true/false-positive rates, and predictive value across the "
        "levels of `group`, flagging gaps beyond 0.1 or a selection ratio under the "
        "four-fifths rule. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_fairness", description=verify_fairness_tool.__doc__)(
        verify_fairness_tool
    )

    async def verify_sensitivity_tool(dataset: str, x: str, y: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_sensitivity, table.rows, x, y)
        return f"Confounding sensitivity of '{x}' on '{y}':\n\n{report.render()}"

    verify_sensitivity_tool.__doc__ = (
        "Quantify how robust the effect of `x` on `y` is to UNMEASURED confounding: "
        "the thing the data cannot test. Returns the E-value: the minimum risk-ratio "
        "association a hidden confounder would need with both `x` and `y` to explain "
        "the effect away. A large E-value means only an implausibly strong confounder "
        "could overturn it. Returns sound (robust) / inconclusive (fragile)."
    )
    server.tool(name="verify_sensitivity", description=verify_sensitivity_tool.__doc__)(
        verify_sensitivity_tool
    )

    async def verify_reliability_tool(dataset: str, items: list[str]) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_reliability, table.rows, items)
        return f"Internal consistency of {items}:\n\n{report.render()}"

    verify_reliability_tool.__doc__ = (
        "Verify that several columns (`items`) form an internally consistent scale "
        "measuring one construct, via Cronbach's alpha, before building a score or "
        "correlation on them. An unreliable scale (alpha < 0.7) attenuates every "
        "downstream result. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_reliability", description=verify_reliability_tool.__doc__)(
        verify_reliability_tool
    )

    async def verify_stationarity_tool(
        dataset: str, value: str, time: str | None = None
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_stationarity, table.rows, value, time)
        return f"Stationarity of the '{value}' series:\n\n{report.render()}"

    verify_stationarity_tool.__doc__ = (
        "Check a time series for a unit root before correlating or regressing it on "
        "another series: regressing two non-stationary series produces large, "
        "significant, entirely spurious relationships (Granger & Newbold). Returns "
        "sound (stationary, safe) / unsound (random walk or trending: difference or "
        "detrend first) / inconclusive."
    )
    server.tool(
        name="verify_stationarity", description=verify_stationarity_tool.__doc__
    )(verify_stationarity_tool)

    async def verify_normality_tool(dataset: str, column: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_normality, table.rows, column)
        return f"Normality of '{column}':\n\n{report.render()}"

    verify_normality_tool.__doc__ = (
        "Check whether `column` can be treated as normally distributed (D'Agostino-"
        "Pearson K² and Anderson-Darling) before using a parametric test or control "
        "limits on it. Returns sound (normal) / unsound (skewed or heavy-tailed: "
        "transform or use a rank method) / inconclusive."
    )
    server.tool(name="verify_normality", description=verify_normality_tool.__doc__)(
        verify_normality_tool
    )

    async def verify_missingness_tool(dataset: str, column: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_missingness, table.rows, column)
        return f"Missingness mechanism for '{column}':\n\n{report.render()}"

    verify_missingness_tool.__doc__ = (
        "Check whether `column` is missing completely at random before dropping or "
        "mean-imputing those rows, which only avoids bias under MCAR. Tests whether "
        "any observed column predicts the missingness. Returns sound (MCAR not "
        "contradicted) / unsound (missingness depends on the data: use multiple "
        "imputation) / inconclusive."
    )
    server.tool(name="verify_missingness", description=verify_missingness_tool.__doc__)(
        verify_missingness_tool
    )

    async def verify_iv_tool(
        dataset: str, instrument: str, treatment: str, outcome: str
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_iv, table.rows, instrument, treatment, outcome
        )
        head = f"Instrument strength of '{instrument}' for '{treatment}':"
        return f"{head}\n\n{report.render()}"

    verify_iv_tool.__doc__ = (
        "Check whether an instrumental variable is strong enough to trust before an IV "
        "/ 2SLS estimate: reports the first-stage F (weak below 10, biasing the "
        "estimate) and reminds you the exclusion restriction is untestable. Returns "
        "sound / unsound / inconclusive."
    )
    server.tool(name="verify_iv", description=verify_iv_tool.__doc__)(verify_iv_tool)

    async def verify_rdd_tool(dataset: str, running: str, cutoff: float) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_rdd, table.rows, running, cutoff)
        return (
            f"RDD manipulation check on '{running}' at {cutoff}:\n\n{report.render()}"
        )

    verify_rdd_tool.__doc__ = (
        "Check a regression-discontinuity design for manipulation of the `running` "
        "variable at the `cutoff`: bunching just past the threshold means units gamed "
        "their score and the design is invalid (a binomial density test). Returns "
        "sound / unsound / inconclusive."
    )
    server.tool(name="verify_rdd", description=verify_rdd_tool.__doc__)(verify_rdd_tool)

    async def verify_extrapolation_tool(
        dataset: str, features: list[str], query: dict[str, float]
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_extrapolation, table.rows, features, query
        )
        return f"Extrapolation check for {query}:\n\n{report.render()}"

    verify_extrapolation_tool.__doc__ = (
        "Check whether a prediction `query` (a {feature: value} point) lies within the "
        "training support of `features` before trusting the model there: flags a value "
        "outside a feature's observed range, or an unobserved combination "
        "(Mahalanobis). Returns sound (in support) / unsound (extrapolating) / "
        "inconclusive."
    )
    server.tool(
        name="verify_extrapolation", description=verify_extrapolation_tool.__doc__
    )(verify_extrapolation_tool)

    async def verify_proportional_hazards_tool(
        dataset: str, time: str, event: str, group: str
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_proportional_hazards, table.rows, time, event, group
        )
        return f"Proportional-hazards check across '{group}':\n\n{report.render()}"

    verify_proportional_hazards_tool.__doc__ = (
        "Check whether a Cox hazard ratio is meaningful between two groups: if the "
        "Kaplan-Meier survival curves cross, the hazards are not proportional and a "
        "single hazard ratio misleads (report time-specific survival instead). Returns "
        "sound / unsound / inconclusive."
    )
    server.tool(
        name="verify_proportional_hazards",
        description=verify_proportional_hazards_tool.__doc__,
    )(verify_proportional_hazards_tool)

    async def verify_did_tool(
        dataset: str, group: str, time: str, value: str, treatment_start: float
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_did, table.rows, group, time, value, treatment_start
        )
        return f"DiD pre-trends check across '{group}':\n\n{report.render()}"

    verify_did_tool.__doc__ = (
        "Check the parallel-trends assumption of a difference-in-differences design by "
        "testing for a differential pre-treatment trend between the groups (before "
        "`treatment_start`). A significant pre-trend means the groups were already "
        "diverging, biasing the DiD estimate. A pass means parallel trends are not "
        "contradicted, not proven. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_did", description=verify_did_tool.__doc__)(verify_did_tool)

    async def verify_counts_tool(
        dataset: str, count: str, predictors: list[str] | None = None
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_counts, table.rows, count, predictors=predictors
        )
        return f"Count-model check for '{count}':\n\n{report.render()}"

    verify_counts_tool.__doc__ = (
        "Check whether a Poisson model fits a `count` outcome or whether it is "
        "overdispersed (variance > mean), which makes Poisson standard errors too "
        "small; use negative-binomial / quasi-Poisson instead. Returns sound / "
        "unsound / inconclusive."
    )
    server.tool(name="verify_counts", description=verify_counts_tool.__doc__)(
        verify_counts_tool
    )

    async def verify_overlap_tool(
        dataset: str, treatment: str, covariates: list[str]
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_overlap, table.rows, treatment, covariates
        )
        return f"Overlap / positivity for '{treatment}':\n\n{report.render()}"

    verify_overlap_tool.__doc__ = (
        "Check whether a treated and control group overlap enough to be compared "
        "(positivity) before propensity matching or weighting: estimates the "
        "propensity and flags poor overlap (units piled at the propensity extremes or "
        "a low overlap coefficient), where an estimate extrapolates rather than "
        "compares. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_overlap", description=verify_overlap_tool.__doc__)(
        verify_overlap_tool
    )

    async def verify_powerlaw_tool(dataset: str, column: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_powerlaw, table.rows, column)
        return f"Power-law fit for '{column}':\n\n{report.render()}"

    verify_powerlaw_tool.__doc__ = (
        "Check whether a 'power-law' / 'scale-free' claim about `column` is actually "
        "supported (Clauset-Shalizi-Newman): fits the tail by MLE, tests goodness-of-"
        "fit by bootstrap, and compares against a lognormal and exponential. Asserts a "
        "power law only when it beats the alternatives; the common honest result is "
        "'cannot be distinguished from a lognormal'. Returns sound / unsound / other."
    )
    server.tool(name="verify_powerlaw", description=verify_powerlaw_tool.__doc__)(
        verify_powerlaw_tool
    )

    async def verify_correlation_tool(dataset: str, x: str, y: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_correlation, table.rows, x, y)
        return f"Association between '{x}' and '{y}':\n\n{report.render()}"

    verify_correlation_tool.__doc__ = (
        "Verify a linear association between `x` and `y` before you report it: "
        "significance, reliance on outliers, and whether an observed covariate "
        "explains it away. Always caveats that correlation is not causation."
    )
    server.tool(name="verify_correlation", description=verify_correlation_tool.__doc__)(
        verify_correlation_tool
    )

    async def verify_trend_tool(dataset: str, time: str, value: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_trend, table.rows, time, value)
        return f"Trend in '{value}' over '{time}':\n\n{report.render()}"

    verify_trend_tool.__doc__ = (
        "Verify a trend in `value` over `time` before you report it: significance "
        "and robustness to autocorrelation, the first/last points, and outliers. "
        "Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_trend", description=verify_trend_tool.__doc__)(
        verify_trend_tool
    )

    async def verify_regression_tool(
        dataset: str, y: str, x: str, controls: list[str] | None = None
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_regression, table.rows, y, x, controls=controls or []
        )
        return f"Coefficient of '{x}' on '{y}':\n\n{report.render()}"

    verify_regression_tool.__doc__ = (
        "Verify an OLS regression coefficient (the slope of `x` on `y`, holding "
        "`controls`) before you report it: multicollinearity (VIF), heteroskedastic "
        "errors re-tested with robust standard errors, influential points, and "
        "functional-form misspecification. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_regression", description=verify_regression_tool.__doc__)(
        verify_regression_tool
    )

    async def verify_prediction_tool(
        dataset: str,
        features: list[str],
        target: str,
        time_order: str | None = None,
        group: str | None = None,
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_prediction,
            table.rows,
            features,
            target,
            time_order=time_order,
            group=group,
        )
        return f"Predicting '{target}':\n\n{report.render()}"

    verify_prediction_tool.__doc__ = (
        "Verify a predictive-performance claim before you report it. Re-evaluates the "
        "model leakage-free: it screens for a single feature that predicts the target "
        "almost perfectly (a leaked proxy), rows shared across the split, and, given "
        "`time_order` or `group`, temporal or entity leakage, then reports the "
        "held-out skill it measures itself. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_prediction", description=verify_prediction_tool.__doc__)(
        verify_prediction_tool
    )

    async def verify_experiment_tool(dataset: str, variant: str, metric: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_experiment, table.rows, variant, metric)
        return f"A/B test on '{metric}':\n\n{report.render()}"

    verify_experiment_tool.__doc__ = (
        "Verify an A/B test result before trusting it: first a sample-ratio-mismatch "
        "check (a broken split voids the whole result), then the difference in the "
        "metric between variants with its effect size. Returns sound / unsound / "
        "inconclusive."
    )
    server.tool(name="verify_experiment", description=verify_experiment_tool.__doc__)(
        verify_experiment_tool
    )

    async def verify_proportions_tool(dataset: str, group: str, outcome: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_proportions, table.rows, group, outcome)
        return f"Association between '{group}' and '{outcome}':\n\n{report.render()}"

    verify_proportions_tool.__doc__ = (
        "Verify a categorical association (contingency table) before reporting it. "
        "Uses an exact test instead of chi-square on sparse tables, catching a "
        "chi-square that is significant only because its approximation is invalid."
    )
    server.tool(name="verify_proportions", description=verify_proportions_tool.__doc__)(
        verify_proportions_tool
    )

    async def verify_forecast_tool(
        dataset: str, time: str, actual: str, forecast: str, seasonal_period: int = 1
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_forecast,
            table.rows,
            time,
            actual,
            forecast,
            seasonal_period=seasonal_period,
        )
        return f"Forecast of '{actual}':\n\n{report.render()}"

    verify_forecast_tool.__doc__ = (
        "Verify a forecast before reporting it useful: the mean absolute scaled error "
        "(MASE) against the (seasonal-)naive baseline. A MASE >= 1 means the forecast "
        "is no better than copying the last season. Returns sound / inconclusive."
    )
    server.tool(name="verify_forecast", description=verify_forecast_tool.__doc__)(
        verify_forecast_tool
    )

    async def verify_classification_tool(
        dataset: str,
        y_true: str,
        y_pred: str | None = None,
        y_score: str | None = None,
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_classification, table.rows, y_true, y_pred=y_pred, y_score=y_score
        )
        return f"Classifier on '{y_true}':\n\n{report.render()}"

    verify_classification_tool.__doc__ = (
        "Verify that a classifier has real skill, not the accuracy paradox. Balanced "
        "accuracy decides, because the majority-class rule scores 0.5 on it at any "
        "prevalence; raw accuracy and the majority baseline are reported alongside "
        "but do not gate. Given `y_score`, ranking must also beat prevalence (PR-AUC) "
        "and chance (ROC). Returns sound / inconclusive, naming which fell short."
    )
    server.tool(
        name="verify_classification", description=verify_classification_tool.__doc__
    )(verify_classification_tool)

    async def verify_calibration_tool(
        dataset: str, probability: str, outcome: str
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_calibration, table.rows, probability, outcome
        )
        return f"Calibration of '{probability}':\n\n{report.render()}"

    verify_calibration_tool.__doc__ = (
        "Verify that predicted probabilities can be trusted as risks, not just ranks: "
        "the expected calibration error. A model can rank well (high AUC) yet be badly "
        "miscalibrated. Returns sound / unsound."
    )
    server.tool(name="verify_calibration", description=verify_calibration_tool.__doc__)(
        verify_calibration_tool
    )

    async def verify_logistic_tool(
        dataset: str, x: str, y: str, controls: list[str] | None = None
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_logistic, table.rows, x, y, controls=controls or []
        )
        return f"Logistic effect of '{x}' on '{y}':\n\n{report.render()}"

    verify_logistic_tool.__doc__ = (
        "Verify a logistic-regression odds ratio of `x` on a binary `y`. Catches "
        "separation: when a predictor splits the outcome and the maximum-likelihood "
        "odds ratio is infinite/not identifiable. Returns sound / unsound / "
        "inconclusive."
    )
    server.tool(name="verify_logistic", description=verify_logistic_tool.__doc__)(
        verify_logistic_tool
    )

    async def verify_survival_tool(
        dataset: str, time: str, event: str, group: str
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(
            verify_survival, table.rows, time, event, group
        )
        return f"Survival difference across '{group}':\n\n{report.render()}"

    verify_survival_tool.__doc__ = (
        "Verify a survival difference between two groups with the log-rank test, which "
        "honours censoring, and flag when a naive mean-time comparison would mislead. "
        "Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_survival", description=verify_survival_tool.__doc__)(
        verify_survival_tool
    )

    async def verify_clusters_tool(
        dataset: str, features: list[str], k: int = 3
    ) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_clusters, table.rows, features, k=k)
        return f"Cluster structure in {features}:\n\n{report.render()}"

    verify_clusters_tool.__doc__ = (
        "Verify that a claimed cluster structure is real, not imposed: cluster "
        "tendency (Hopkins) and the silhouette of a k-means partition. k-means returns "
        "k clusters even on a single blob. Returns sound / unsound / inconclusive."
    )
    server.tool(name="verify_clusters", description=verify_clusters_tool.__doc__)(
        verify_clusters_tool
    )

    async def verify_robustness_tool(dataset: str, x: str, y: str) -> str:
        table = _load(dataset)
        report = await asyncio.to_thread(verify_multiverse, table.rows, x, y)
        return f"Robustness of the effect of '{x}' on '{y}':\n\n{report.render()}"

    verify_robustness_tool.__doc__ = (
        "Certify how robustly `x` affects `y` across the whole defensible analytical "
        "multiverse (every confounder-adjustment set, keep/drop outliers, raw/rank "
        "fit), with the joint resampling-under-the-null test of specification-curve "
        "analysis. Catches a conclusion that holds under the one analysis you ran but "
        "fails under a defensible alternative or combination. Returns robust / fragile "
        "/ inconclusive with the pivotal decision and breaking specification."
    )
    server.tool(name="verify_robustness", description=verify_robustness_tool.__doc__)(
        verify_robustness_tool
    )

    async def verify_tool(
        dataset: str,
        x: str | None = None,
        y: str | None = None,
        controls: list[str] | None = None,
        group: str | None = None,
        value: str | None = None,
        sesoi: float | None = None,
        variant: str | None = None,
        metric: str | None = None,
        outcome: str | None = None,
        time: str | None = None,
        actual: str | None = None,
        forecast: str | None = None,
        seasonal_period: int | None = None,
        y_true: str | None = None,
        y_pred: str | None = None,
        y_score: str | None = None,
        probability: str | None = None,
        event: str | None = None,
        features: list[str] | None = None,
        target: str | None = None,
        column: str | None = None,
        instrument: str | None = None,
        treatment: str | None = None,
        running: str | None = None,
        cutoff: float | None = None,
        treatment_start: float | None = None,
        count: str | None = None,
        predictors: list[str] | None = None,
        covariates: list[str] | None = None,
    ) -> str:
        table = _load(dataset)
        claim = {
            k: v
            for k, v in {
                "x": x,
                "y": y,
                "controls": controls,
                "group": group,
                "value": value,
                "sesoi": sesoi,
                "variant": variant,
                "metric": metric,
                "outcome": outcome,
                "time": time,
                "actual": actual,
                "forecast": forecast,
                "seasonal_period": seasonal_period,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_score": y_score,
                "probability": probability,
                "event": event,
                "features": features,
                "target": target,
                "column": column,
                "instrument": instrument,
                "treatment": treatment,
                "running": running,
                "cutoff": cutoff,
                "treatment_start": treatment_start,
                "count": count,
                "predictors": predictors,
                "covariates": covariates,
            }.items()
            if v is not None
        }
        report = await asyncio.to_thread(verify_all, table.rows, **claim)
        return report.render()

    verify_tool.__doc__ = (
        "Verify a claim by running every applicable check at once. Assign the columns "
        "of your claim to roles (x/y for an effect, group/value for a comparison, "
        "variant/metric for an A/B test, y_true/y_pred for a classifier, time/event/"
        "group for survival, features for clusters, …); the engine selects the checks "
        "whose preconditions the data satisfies, runs them all, and returns one "
        "verdict (sound / unsound / invalid / inconclusive): a validity failure "
        "(sample-ratio mismatch) voids the result. Prefer this over picking a single "
        "verify_* tool when you are not sure which check applies."
    )
    server.tool(name="verify", description=verify_tool.__doc__)(verify_tool)


def _format_result(result: Any) -> str:
    """Render a session result (stdout/result/error) the way the agent reads it."""
    parts: list[str] = []
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.result is not None:
        parts.append(f"result: {result.result}")
    if result.error:
        parts.append(f"error:\n{result.error}")
    return "\n\n".join(parts) or (
        "(ran with no output; print() to inspect, or set a `result` variable)"
    )


def _register_run_code(
    server: MCPServer,
    load_dataset: Callable[[str], Table] | None,
    dataset_specs: tuple[DatasetSpec, ...] = (),
    scratch_dir: Path | None = None,
    backend: str = "subprocess",
    egress: str | Sequence[str] = "full",
    image: str | None = None,
) -> None:
    """Register the stateful explore/debug loop: ``run_code`` plus ``bash``.

    This precedes authoring a derivation: the agent inspects the data, prototypes the
    computation, checks the number, then crystallizes it with ``propose_derivation``.
    The two tools share one persistent session (variables, imports, and files carry
    across calls, and, on the docker backend, ``bash`` and ``run_code`` share one
    container), the same experience the app gives a chat. The session is scoped to this
    server (the MCP surface has no conversation to key on), a local-dev convenience; a
    derivation still runs one-shot and hermetic, so a certified result is unaffected.
    """
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    holder: dict[str, SessionManager] = {}
    datasets_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def _datasets() -> dict[str, list[dict[str, Any]]]:
        if "d" not in datasets_cache:
            datasets: dict[str, list[dict[str, Any]]] = {}
            for spec in dataset_specs:
                if load_dataset is None:
                    break
                with contextlib.suppress(ElbiError):
                    datasets[spec.name] = list(load_dataset(spec.name).rows)
            datasets_cache["d"] = datasets
        return datasets_cache["d"]

    # The session is one process with one working directory, so two calls cannot share
    # it. On the event loop that was enforced by accident; off it, the lock is what
    # keeps a second call from interleaving with the first.
    session_lock = threading.Lock()

    def _locked(call: Callable[[], Any]) -> Any:
        with session_lock:
            return call()

    def _manager() -> SessionManager:
        if "m" not in holder:
            manager = SessionManager(
                datasets=_datasets(),
                workspace=scratch_dir,
                backend=backend,
                egress=egress,
                image=image,
            )
            holder["m"] = manager
            atexit.register(manager.close)
        return holder["m"]

    # Background exploration jobs (long computations): kept in-memory for this server's
    # lifetime, observed and cancelled with the job tools below.
    jobs = JobRunner(store=InMemoryJobStore())
    atexit.register(jobs.close)

    async def run_code(
        code: str, deps: list[str] | None = None, background: bool = False
    ) -> str:
        if background:
            job = submit_code_job(
                jobs,
                code=code,
                deps=deps or [],
                datasets=_datasets(),
                scratch_dir=scratch_dir,
                backend=backend,
                egress=egress,
                image=image,
            )
            return (
                f"Launched background run_code job {job.id}. It runs in its own "
                "isolated session, so have it WRITE results to a workspace file a "
                "later inline call can read. Poll job_status; cancel_job stops it."
            )
        result = await asyncio.to_thread(
            _locked, lambda: _manager().run_code(code, deps or [])
        )
        return _format_result(result)

    run_code.__doc__ = (
        "Run Python in a persistent sandbox session to explore data and prototype a "
        "computation before authoring a derivation. The declared datasets are there "
        "as `data['<name>']` (a list of row dicts; values are strings, convert as "
        "needed). `deps` lists third-party packages the code imports: named packages "
        "are installed on demand (e.g. deps=['xgboost']); only the standard library is "
        "available without deps. run_code and bash share a live session: variables, "
        "imports, and files persist from one call to the next, so build up state "
        "incrementally. print() to inspect; set `result` to return a value; a code "
        "error comes back as a traceback so you can fix it and re-run. For a long "
        "computation, set background=true: it runs as a job in its own isolated "
        "session (write results to a workspace file); poll job_status and cancel with "
        "cancel_job. Iterate here, then capture it with propose_derivation."
    )
    server.tool(name="run_code", description=run_code.__doc__)(run_code)

    async def job_status(job_id: str) -> str:
        job = jobs.get(job_id)
        if job is None:
            return f"error:\nno job {job_id}"
        lines = [f"job {job.id} [{job.state}]"]
        if job.progress:
            lines.append(f"progress: {job.progress}")
        if job.result is not None:
            lines.append(f"result: {job.result}")
        if job.error:
            lines.append(f"error: {job.error}")
        return "\n".join(lines)

    job_status.__doc__ = (
        "Report a background job's state (queued/running/succeeded/failed/cancelled), "
        "its latest progress, and, once finished, its result or error. Poll this after "
        "launching work with run_code(background=true)."
    )
    server.tool(name="job_status", description=job_status.__doc__)(job_status)

    async def cancel_job(job_id: str) -> str:
        job = jobs.cancel(job_id)
        if job is None:
            return f"error:\nno job {job_id}"
        return f"job {job.id} [{job.state}]"

    cancel_job.__doc__ = (
        "Request cancellation of a background job by id; returns its resulting state. "
        "A queued job is cancelled at once, a running one stops at its next checkpoint."
    )
    server.tool(name="cancel_job", description=cancel_job.__doc__)(cancel_job)

    async def bash(command: str) -> str:
        result = await asyncio.to_thread(_locked, lambda: _manager().run_bash(command))
        return _format_result(result)

    bash.__doc__ = (
        "Run a shell command in the same sandbox session as run_code, sharing its "
        "filesystem and installed packages: install a system lib (apt-get), inspect "
        "files, or run a CLI tool, and a package you install here is available to the "
        "next run_code. Requires the docker sandbox backend (an isolated container); "
        "with the default subprocess backend it is declined, since a host shell would "
        "run on the user's machine. Exploration only; the answer comes from a "
        "derivation."
    )
    server.tool(name="bash", description=bash.__doc__)(bash)


def _register_contract(server: MCPServer, load_dataset: Callable[[str], Table]) -> None:
    """Register the data-cleaning tools: profile, suggest a contract, and check one.

    These are the read-only half of verifiable cleaning: an agent profiles a dataset,
    lets the profiler propose a contract (thresholds are confidence bounds, not sample
    rates), edits it, and checks it. To *certify* a cleaning step, the agent authors a
    derivation whose output must satisfy the contract via ``propose_derivation``.
    """
    import json

    _load = partial(_load_or_refuse, load_dataset)

    async def profile_dataset(dataset: str) -> str:
        table = _load(dataset)
        profiles = [p.to_dict() for p in profile_columns(table.rows)]
        return json.dumps({"dataset": dataset, "columns": profiles}, indent=2)

    profile_dataset.__doc__ = (
        "Profile a dataset for data cleaning: per column, its completeness, distinct "
        "count, inferred type, numeric range, and most common values. Read this before "
        "proposing a data contract. It reports the data's shape; it asserts nothing."
    )
    server.tool(name="profile_dataset", description=profile_dataset.__doc__)(
        profile_dataset
    )

    async def suggest_contract_tool(dataset: str) -> str:
        table = _load(dataset)
        contract = await asyncio.to_thread(suggest_contract, table.rows)
        return json.dumps(contract.to_manifest(), indent=2)

    suggest_contract_tool.__doc__ = (
        "Propose a data contract for a dataset by profiling it: per-field types and "
        "constraints (required, unique, ranges, enums), a primary key, and per-field "
        "tolerances set to a confidence lower bound so they do not overfit the sample. "
        "The result is a starting point to review and tighten, not a verdict; check it "
        "with verify_contract, and attach it to a cleaning derivation via "
        "propose_derivation to certify it."
    )
    server.tool(name="suggest_contract", description=suggest_contract_tool.__doc__)(
        suggest_contract_tool
    )

    async def verify_contract_tool(
        dataset: str,
        contract: dict[str, Any],
        references: dict[str, str] | None = None,
    ) -> str:
        table = _load(dataset)
        try:
            parsed = DataContract.from_manifest(contract)
        except ElbiError as exc:
            return f"Invalid contract: {exc}"
        resolved: dict[str, list[dict[str, Any]]] = {}
        for resource, ref_dataset in (references or {}).items():
            resolved[resource] = _load(ref_dataset).rows
        report = await asyncio.to_thread(
            verify_contract, table.rows, parsed, references=resolved
        )
        return report.render()

    verify_contract_tool.__doc__ = (
        "Check a dataset against a data contract (see suggest_contract for the shape) "
        "and return a three-valued verdict with the offending rows: sound (the data "
        "meets the bar), unsound (it does not, with located violations), or "
        "inconclusive (a clause could not be evaluated). `references` maps a foreign "
        "key's referenced resource name to the dataset that holds it, for referential-"
        "integrity checks. This checks without certifying; to certify a cleaning step, "
        "attach the contract to a derivation via propose_derivation."
    )
    server.tool(name="verify_contract", description=verify_contract_tool.__doc__)(
        verify_contract_tool
    )


def _register_propose(
    server: MCPServer,
    registry: Registry,
    make_runner: Callable[[], Runner],
    serving: Serving,
    certification: CertificationPolicy | None,
    authored_store: AuthoredStore | None,
    issuer: CertificateIssuer | None,
    run_log: RunLog | None,
    load_dataset: Callable[[str], Table] | None,
    on_author: Callable[[str, dict[str, Any]], object] | None = None,
) -> None:
    """Register the agent-authoring loop.

    Propose source, verify under isolation, and, per the certification policy, serve
    it immediately or hold it for review. A derivation that declares a ``claim`` (the
    conclusion it asserts) is gated on the verification oracle too: an unsound
    conclusion is held, not served, and the verdict is returned for the agent to fix
    and propose again. The verifier runs server-side, so the agent cannot weaken it.
    """

    async def propose_derivation(
        name: str,
        source: str,
        format: str = "table",
        title: str | None = None,
        inputs: list[str] | None = None,
        claim: dict[str, Any] | None = None,
        contract: dict[str, Any] | None = None,
        deps: list[str] | None = None,
        *,
        ctx: McpContext[Any, Any],
    ) -> str:
        try:
            serve_contract = _serve_for(format, title)
            dataset_inputs = {key: Dataset(key) for key in inputs or []}
            data_contract = DataContract.from_manifest(contract) if contract else None
            outcome = await asyncio.to_thread(
                lambda: author(
                    name,
                    source,
                    runner=make_runner(),
                    serve=serve_contract,
                    inputs=dataset_inputs,
                    claim=claim,
                    contract=data_contract,
                    deps=deps,
                    policy=certification,
                    registry=registry,
                )
            )
        except ElbiError as exc:
            return f"Proposal for {name!r} rejected: {exc}"

        result = outcome.result
        if result.error:
            return (
                f"Proposed {name!r} but it failed to run in the sandbox:\n"
                f"{result.error}\n\nFix the source and propose again."
            )
        # A declared `claim` whose conclusion the oracle did not find sound: hold it
        # (never served) and return the oracle's reason so the agent can fix it.
        if result.oracle_verdict is not None and result.oracle_verdict != "sound":
            if authored_store is not None:
                authored_store.save(outcome.derivation)
            return (
                f"Proposed {name!r} but its declared conclusion did not pass "
                f"verification (oracle: {result.oracle_verdict}), so it is held (not "
                f"served). Fix the analysis and propose again.\n\n"
                f"{result.oracle_detail or ''}"
            )
        # A declared `contract` the output did not satisfy: hold it and return the
        # failing clause so the agent can fix the cleaning and propose again.
        if result.contract_verdict is not None and result.contract_verdict != "sound":
            if authored_store is not None:
                authored_store.save(outcome.derivation)
            return (
                f"Proposed {name!r} but its output did not satisfy its data contract "
                f"(contract: {result.contract_verdict}), so it is held (not served). "
                f"Fix the cleaning and propose again.\n\n"
                f"{result.contract_detail or ''}"
            )
        if not result.ok:
            return (
                f"Proposed {name!r} but it did not pass its golden cases; it is held. "
                "Fix the source and propose again."
            )
        # The verification attestation (present when a claim was declared and verified
        # sound) rides with the served result and is persisted so it survives a
        # restart; the derivation is still reconstructed as agent-origin and runs only
        # under the sandbox.
        attestation = result.oracle_attestation
        if result.contract_attestation is not None:
            attestation = {
                **(attestation or {}),
                "contract": result.contract_attestation,
            }
        # Build the certified run once: it seeds both the version history and, signed,
        # the exportable certificate that rides with every serve of this derivation.
        # The certificate is persisted so a restart re-serves the identical bytes.
        run = run_from_author(outcome)
        certificate = (
            issuer.issue(run) if issuer is not None and run is not None else None
        )
        if authored_store is not None:
            authored_store.save(outcome.derivation, attestation, certificate)
        # Record the certified run for the version history and comparison (a no-op on a
        # deterministic re-run: the run collapses to its existing version), and export
        # it to MLflow if MLFLOW_TRACKING_URI is set.
        if run_log is not None and run is not None and run_log.append(run):
            _emit_local_run(run)
        preview = result.rendered or "(produced no served output)"
        if outcome.certified:
            # The served app persists a governed row for a certified proposal
            # (the same moment the chat path does), which is what makes it
            # visible in /api/derivations and holdable by trash. Held
            # proposals stay sidecar-only, exactly like the chat path.
            if on_author is not None:
                on_author(
                    name,
                    {
                        "source": source,
                        "question": (
                            f"Proposed via MCP: {title}"
                            if title
                            else "Proposed via MCP"
                        ),
                        "format": format,
                        "deps": list(deps or []),
                        "claim": claim,
                        "verdict": result.oracle_verdict or result.contract_verdict,
                        "rendered": result.rendered or "",
                        "attestation": attestation,
                    },
                )
            # The propose tool always supplies a serve contract, so a certified
            # proposal is served; register its tool on the live server and tell
            # connected clients the tool list changed.
            _register(server, outcome.derivation, serving, attestation, certificate)
            await _notify_tools_changed(ctx)
            verified = (
                f"\n\nVerified sound: {result.oracle_detail}"
                if result.oracle_verdict == "sound"
                else ""
            )
            return (
                f"Authored and certified {name!r} (origin=agent), and ran it. The "
                f"result is below: report it directly as the answer. You do NOT "
                f"need to call `{tool_name(name)}` again unless you want to re-run "
                f"it with different parameters.\n\n"
                f"Result:\n{preview}{verified}"
            )
        return (
            f"Proposed {name!r} (origin=agent, status=proposed). It ran cleanly in "
            f"the sandbox.\n\nPreview:\n{preview}\n\nIt is held for review and will "
            f"not be served until certified: `elbi certify {name}`."
        )

    propose_derivation.__doc__ = (
        "Author a new derivation from generated Python source. The source must "
        "define a function named the same as `name`, taking a Context and "
        "returning a value. It runs once under sandbox isolation; if it verifies "
        "cleanly it is certified and served immediately (unless this server is "
        "configured to hold proposals for human review). `inputs` lists dataset "
        "names the function reads via ctx.input(name). `deps` names third-party "
        "packages the source imports (e.g. ['numpy','scikit-learn']); the derivation "
        "sandbox is stdlib-only, so any non-stdlib import must be declared here or it "
        "fails. First call describe_dataset "
        "for the schema and sandbox_environment for importable libraries; do not "
        "author throwaway probes to discover them, and delete any you no longer need "
        "with delete_derivation. When the derivation draws a CONCLUSION (an effect, "
        "an A/B result, a model evaluation, a trend...), output the analytic rows it "
        "concerns and declare `claim` as the verification column roles (e.g. "
        '{"x": "dose", "y": "response"} or {"variant": "arm", "metric": '
        '"converted"}). The derivation is then gated on the verification oracle and '
        "is NOT served unless the conclusion is sound; if it comes back unsound, fix "
        "the analysis per the verdict and propose again. When the derivation CLEANS "
        "data (dedupe, fix types, enforce a schema), declare a `contract` (see "
        "suggest_contract / verify_contract) the output must satisfy; it is gated on "
        "the contract and NOT served unless the output is valid, so a certified clean "
        "table carries a verified quality bar."
    )
    server.tool(name="propose_derivation", description=propose_derivation.__doc__)(
        propose_derivation
    )


def _register_describe(
    server: MCPServer,
    load_dataset: Callable[[str], Table],
    dataset_specs: tuple[DatasetSpec, ...],
) -> None:
    """Register dataset introspection, so an agent reads the schema, not probes it."""
    specs = {spec.name: spec for spec in dataset_specs}
    listed = ", ".join(specs) or "(none declared)"

    async def describe_dataset(name: str = "") -> str:
        if not name:
            return (
                f"Available datasets: {listed}. "
                "Call describe_dataset(name=...) for a dataset's schema."
            )
        try:
            table = load_dataset(name)
        except ElbiError as exc:
            raise MCPError(
                INVALID_PARAMS,
                f"Could not load dataset {name!r}: {exc}\n"
                f"Available datasets: {listed}.",
            ) from exc
        return _render_dataset(name, table, specs.get(name))

    describe_dataset.__doc__ = (
        "Describe a bound dataset before writing a derivation over it: its row "
        "count, columns, an example row, each column's inferred type, and any "
        "author-declared meaning (description, unit, role, synonyms). Call with no "
        "name to list the available datasets. Row values are strings; cast them in "
        "your derivation (int(...), float(...))."
    )
    server.tool(name="describe_dataset", description=describe_dataset.__doc__)(
        describe_dataset
    )


def _register_environment(server: MCPServer) -> None:
    """Register sandbox-capability introspection (Python version + importable libs)."""

    async def sandbox_environment() -> str:
        return _render_environment()

    sandbox_environment.__doc__ = (
        "Report the sandbox a proposed derivation runs in: the Python version and "
        "which common compute libraries (numpy, pandas, ...) are importable. Check "
        "this before writing a derivation so you don't import something unavailable."
    )
    server.tool(name="sandbox_environment", description=sandbox_environment.__doc__)(
        sandbox_environment
    )


def _register_ml(
    server: MCPServer,
    load_dataset: Callable[[str], Table] | None,
    cache_dir: Path | None,
) -> None:
    """Register the model lifecycle: AutoML training, the registry, and scoring.

    Registered only when datasets are bound and the optional ml stack (the ``ml``
    extra) is importable, so a server without it simply does not offer the tools
    and the advertised capability list stays honest. The tracking/registry store is
    ``MLFLOW_TRACKING_URI`` when set (a team's MLflow server), else a SQLite store
    under the project's cache directory.
    """
    if load_dataset is None:
        return
    try:
        from elbi_core.ml import require_ml

        require_ml()
    except ElbiError:
        return
    from elbi_core.ml import ModelRegistry, train_automl

    dataset_loader = load_dataset

    def _uri() -> str:
        env = os.environ.get("MLFLOW_TRACKING_URI")
        if env:
            return env
        root = cache_dir if cache_dir is not None else Path(".elbi")
        root.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{root / 'mlflow.db'}"

    def _registry() -> ModelRegistry:
        return ModelRegistry(_uri())

    def _rows(dataset: str) -> list[dict[str, Any]]:
        """The dataset's rows, refusing a name the project does not have."""
        return list(_load_or_refuse(dataset_loader, dataset).rows)

    async def train_model(
        name: str,
        dataset: str,
        target: str,
        features: list[str] | None = None,
        task: str = "auto",
        time_budget: float = 60.0,
        metric: str | None = None,
        ensemble: bool = False,
        time_col: str | None = None,
        horizon: int | None = None,
        engine: str = "flaml",
    ) -> str:
        if time_budget <= 0:
            # FLAML reads a negative budget as "no search budget", which is an
            # unbounded run rather than a small one, so the ceiling below only
            # bounds anything once the floor is known.
            raise MCPError(
                INVALID_PARAMS,
                f"time_budget must be a positive number of seconds, not {time_budget}.",
            )
        rows = await asyncio.to_thread(_rows, dataset)
        try:
            report = await asyncio.to_thread(
                train_automl,
                rows,
                name=name,
                target=target,
                features=features or (),
                task=task,
                time_budget=min(time_budget, _MAX_TRAIN_BUDGET),
                metric=metric,
                tracking_uri=_uri(),
                artifact_root=(
                    cache_dir / "mlartifacts" if cache_dir is not None else None
                ),
                ensemble=ensemble,
                dataset=dataset,
                time_col=time_col,
                horizon=horizon,
                engine=engine,
            )
        except ElbiError as exc:
            return f"error: {exc}"
        return report.render()

    train_model.__doc__ = (
        "Train a predictive model with AutoML (FLAML) and register it in the MLflow "
        "model registry. `target` is the column to predict; `features` defaults to "
        "every other column; `task` is classification/regression/ts_forecast/auto "
        "(ts_forecast needs time_col and horizon); `time_budget` bounds the search "
        "in seconds and must be positive, capped at 240; `ensemble` stacks the "
        "searched learners. `dataset` may also "
        "name a certified derivation (engineered features). Metrics reported are "
        "held-out. The first sound version becomes the champion alias; later "
        "versions need promote_model."
    )
    server.tool(name="train_model", description=train_model.__doc__)(train_model)

    async def list_models() -> str:
        models = _registry().models()
        if not models:
            return "No registered models yet. train_model creates one."
        lines = ["name | latest | champion"]
        for m in models:
            champion = f"v{m.champion_version}" if m.champion_version else "(none)"
            lines.append(f"{m.name} | v{m.latest_version} | {champion}")
        return "\n".join(lines)

    list_models.__doc__ = (
        "List the registered models with their latest version and where the "
        "champion alias points."
    )
    server.tool(name="list_models", description=list_models.__doc__)(list_models)

    async def promote_model(name: str, version: int, alias: str = "champion") -> str:
        try:
            _registry().promote(name, version, alias)
        except ElbiError as exc:
            return f"error: {exc}"
        return f"{name} v{version} is now @{alias}."

    promote_model.__doc__ = (
        "Point a registered model's alias (default: champion) at a version. The "
        "champion is what a bare model name serves."
    )
    server.tool(name="promote_model", description=promote_model.__doc__)(promote_model)

    async def predict(
        model: str, rows: list[dict[str, Any]], version: str | None = None
    ) -> str:
        from elbi_core.ml import parse_invocations

        registry = _registry()
        try:
            resolved = registry.resolve(model, version)
            loaded = await asyncio.to_thread(registry.load, model, str(resolved))
            frame, _ = parse_invocations({"dataframe_records": rows})
            predictions = await asyncio.to_thread(loaded.predict, frame)
        except ElbiError as exc:
            return f"error: {exc}"
        values = (
            predictions.tolist()
            if hasattr(predictions, "tolist")
            else list(predictions)
        )
        return f"{model} v{resolved} predictions: {values}"

    predict.__doc__ = (
        "Score feature records with a registered model. `rows` are records shaped "
        "like the training columns (target omitted); `version` is a version number "
        "or alias, defaulting to the champion (or newest)."
    )
    server.tool(name="predict", description=predict.__doc__)(predict)


def _register_delete(
    server: MCPServer,
    registry: Registry,
    authored_store: AuthoredStore | None,
    cache_dir: Path | None,
    on_delete: Callable[[str], object] | None = None,
) -> None:
    """Register deletion of agent-authored derivations, for cleaning up probes.

    ``on_delete`` is the served app's hook (see ``serve.py``) to stamp the
    database row trashed. When it is set and reports the stamp took, this
    trashes rather than erases: the sidecar moves aside instead of being
    removed, and the cache is left alone so a restore stays cheap. When the
    stamp did not take (no governed row: a held proposal, a legacy probe) or
    the hook is ``None`` -- the standalone ``elbi mcp`` process, which
    has no database to stamp -- deletion stays exactly as permanent as it
    always was, and says so.
    """

    async def delete_derivation(name: str, *, ctx: McpContext[Any, Any]) -> str:
        try:
            derivation = registry.get(name)
        except ElbiError as exc:
            return f"Cannot delete {name!r}: {exc}"
        if not derivation.is_agent_authored:
            return (
                f"Refusing to delete {name!r}: it is human-authored. Only "
                "agent-authored derivations can be deleted here."
            )
        # The hook runs before any runtime teardown, and only a truthy return
        # takes the trash path: a False means there was no governed row to
        # stamp (a held proposal, a legacy probe), and claiming "moved to
        # trash" for one of those promises a restore nothing can perform.
        if on_delete is not None and on_delete(name):
            # The hook may have stopped it already (the served app's does, since
            # it holds the registry and the sidecar), so every step here is
            # guarded or idempotent rather than assumed to be first.
            if name in registry:
                registry.remove(name)
            _unregister(server, name)
            await _notify_tools_changed(ctx)
            if authored_store is not None:
                authored_store.trash(name)
            return (
                f"Moved {name!r} to trash; it is no longer registered or served. "
                "Restore it to bring it back."
            )
        if name in registry:
            registry.remove(name)
        _unregister(server, name)
        await _notify_tools_changed(ctx)
        if authored_store is not None:
            authored_store.remove(name)  # also drop the persisted record
        if cache_dir is not None:
            # Purge its cached results so they are not left orphaned on disk.
            LocalCacheStore(cache_dir).invalidate_tag(derivation_tag(name))
        return f"Deleted {name!r}; it is no longer registered or served."

    delete_derivation.__doc__ = (
        "Delete an agent-authored derivation you no longer need (e.g. a throwaway "
        "probe), removing it from the registry and unserving its tool. "
        "Human-authored derivations are protected and cannot be deleted here."
    )
    server.tool(name="delete_derivation", description=delete_derivation.__doc__)(
        delete_derivation
    )


def _unregister(server: MCPServer, name: str) -> None:
    """Remove a derivation's tool (and resource, if any) from the live server."""
    if tool_name(name) in server._tool_manager._tools:
        server.remove_tool(tool_name(name))
    server._resource_manager._resources.pop(resource_uri(name), None)


def _render_dataset(name: str, table: Table, spec: DatasetSpec | None) -> str:
    rows = table.rows
    columns = table.columns
    sample = rows[0] if rows else {}
    types = _infer_types(rows, columns)
    column_specs = {c.name: c for c in spec.columns} if spec else {}
    lines = [f"# Dataset: {name}", ""]
    if spec and spec.description:
        lines += [spec.description, ""]
    lines += [
        f"{len(rows):,} rows, {len(columns)} columns. Values are strings; cast as "
        "needed (int(...), float(...)).",
        "",
        "| column | example | inferred | meaning |",
        "| --- | --- | --- | --- |",
    ]
    for column in columns:
        example = str(sample.get(column, "")).replace("|", "\\|")
        meaning = _column_meaning(column_specs.get(column)).replace("|", "\\|")
        lines.append(f"| {column} | {example} | {types[column]} | {meaning} |")
    return "\n".join(lines)


def _column_meaning(spec: ColumnSpec | None) -> str:
    """The author-declared meaning for a column, as a compact cell. Empty if none."""
    if spec is None:
        return ""
    parts: list[str] = []
    if spec.description:
        parts.append(spec.description)
    tags = [t for t in (spec.role, spec.unit) if t]
    if tags:
        parts.append(f"({', '.join(tags)})")
    if spec.synonyms:
        parts.append(f"aka {', '.join(spec.synonyms)}")
    if spec.sample_values:
        parts.append(f"e.g. {', '.join(spec.sample_values)}")
    return " ".join(parts)


def _infer_types(
    rows: list[dict[str, Any]], columns: list[str], limit: int = 200
) -> dict[str, str]:
    """Best-effort per-column type from the first ``limit`` rows of string values."""
    types: dict[str, str] = {}
    for column in columns:
        seen = False
        all_int = all_float = True
        for row in rows[:limit]:
            value = row.get(column, "")
            if value in ("", None):
                continue
            seen = True
            try:
                int(value)
            except (ValueError, TypeError):
                all_int = False
            try:
                float(value)
            except (ValueError, TypeError):
                all_float = False
        if not seen:
            types[column] = "empty"
        elif all_int:
            types[column] = "integer"
        elif all_float:
            types[column] = "number"
        else:
            types[column] = "string"
    return types


def _render_environment() -> str:
    version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    available = [m for m in _COMMON_LIBS if importlib.util.find_spec(m) is not None]
    libs = ", ".join(available) if available else "(none of the common ones)"
    return (
        f"Sandbox: a fresh, isolated Python {version} subprocess with a wall-clock "
        "timeout.\nThe Python standard library is always available.\n"
        f"Importable compute libraries here: {libs}.\n"
        "To use a library that isn't listed, add it to the project's environment."
    )


def _serve_for(format: str, title: str | None) -> Serve:
    builders = {
        "table": serve_builders.table,
        "markdown": serve_builders.markdown,
        "json": serve_builders.json,
        "text": serve_builders.text,
    }
    if format not in builders:
        raise ElbiError(
            f"unknown serve format {format!r}; choose one of {sorted(builders)}"
        )
    return builders[format](title=title)
