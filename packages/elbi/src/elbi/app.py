"""The FastAPI app: a verified-chat endpoint over the runtime, with the MCP mounted.

:func:`create_app` builds an ASGI app that (1) streams a run of the verifying runtime
over Server-Sent Events at ``POST /api/chat`` (the browser sees each tool the model
calls and a final result it could not fake) (2) optionally mounts the project's MCP
server at ``/mcp`` (sharing its lifespan, which the streamable-HTTP transport requires),
and (3) serves the built single-page app when present. The frontend is a decoupled SPA
that calls these endpoints, so the same app runs as one local process or as a separately
deployed API behind a CDN-hosted SPA.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_args
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from mcp.server.transport_security import TransportSecuritySettings
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from sse_starlette import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from elbi_agent import (
    AnswerResult,
    DeriveFn,
    LLMClient,
    Transcript,
    Workspace,
    answer,
    condense_turns,
    stream,
)
from elbi_core import (
    CertificateIssuer,
    JobRunner,
    SubprocessExecutor,
    new_job_id,
    submit_code_job,
    verify_certificate,
)
from elbi_core.errors import CertificateError, ElbiError, ModelError
from elbi_core.executor import _DEP_RE
from elbi_core.sandbox import ComputeProfileError
from elbi_core.tracking import CertifiedRun, with_changes
from elbi_core.versioning import hash_json

from . import crypto, datasources, extensions, mailer, runtime_settings
from .authoring import DeriveFactory
from .casing import CamelCaseResponses, SnakeCaseRequests, snakeify
from .certificate_pdf import render_certificate_pdf
from .compute import over_budget, spend_limit
from .dashboards import DashboardError, DashboardService
from .db import Conversation, DataSource, Derivation, LlmProfile, Message, Secret, Store
from .derivation_jobs import derivation_job_key, submit_derivation_job
from .explore import ExploreService
from .features import (
    FeatureStoreError,
    FeatureStoreService,
)
from .lineage import LineageError, LineageService
from .metrics import MetricService, MetricServiceError
from .ml import INLINE_TRAIN_BUDGET, ModelService, mount_mlflow_ui
from .monitoring import MonitorError, MonitorService
from .nl2sql import draft_query, generate_sql
from .notebooks import BASE_ENV_SETTING, NotebookService
from .notifications import (
    EVENT_TYPES,
    TRAINING_FAILED,
    create_notifications,
    dispatch_event,
    effective_prefs,
)
from .orchestration import OrchestrationService, cron_due
from .search.build import Drain, SearchBuilder, log_counts
from .search.index import DEFAULT_CANDIDATES, MAX_SEARCH_CANDIDATES, SearchIndex
from .search.snippet import snippet
from .search.sql import filter_sql
from .telemetry import setup_tracing
from .warehouse.service import WarehouseError, WarehouseService
from .webhooks import fire_model_event, webhook_config
from .wire import AuditEvent as WireAuditEvent
from .wire import Budget as WireBudget
from .wire import ComputeProfiles as WireComputeProfiles
from .wire import ComputeSession as WireComputeSession
from .wire import ComputeUsage as WireComputeUsage
from .wire import Created as WireCreated
from .wire import Dataset as WireDataset
from .wire import DataSource as WireDataSource
from .wire import DerivationDetail as WireDerivationDetail
from .wire import DerivationSummary as WireDerivationSummary
from .wire import Install as WireInstall
from .wire import Job as WireJob
from .wire import LineageGraph as WireLineageGraph
from .wire import MlflowSettings as WireMlflowSettings
from .wire import Monitor as WireMonitor
from .wire import MonitorHistory as WireMonitorHistory
from .wire import NotificationItem as WireNotificationItem
from .wire import NotificationList as WireNotificationList
from .wire import NotificationPref as WireNotificationPref
from .wire import NotificationPreferences as WireNotificationPreferences
from .wire import Ok as WireOk
from .wire import Version as WireVersion
from .wire import WebhookSettings as WireWebhookSettings

logger = logging.getLogger(__name__)


class LLMConfigError(ElbiError):
    """Raised when an LLM call is attempted with no model explicitly configured.

    A model must be named explicitly (by a profile in Settings, the ``LLM_MODEL``
    environment variable, or the ``--model`` flag), so which model runs is always a
    visible choice. Defined here (not in ``serve``, which imports it from here) so
    the chat route below can catch it specifically -- ``serve`` builds on ``app``,
    so the dependency has to run this direction.
    """


def _register_build_info() -> None:
    """Register the ``app_build_info`` gauge (value 1) once, idempotently.

    Guarded on the collector name so building the app more than once (in tests, or a
    reloader) does not raise a duplicate-timeseries error.
    """
    from prometheus_client import REGISTRY

    if "app_build_info" in getattr(REGISTRY, "_names_to_collectors", {}):
        return
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    # The distribution is "elbi"; "elbi-app" is only the console script. Looking up the
    # script name always raised, so this gauge reported "unknown" in every deployment:
    # useless for telling which build is running.
    try:
        app_version = _pkg_version("elbi")
    except PackageNotFoundError:  # pragma: no cover - only in a non-installed checkout
        app_version = "unknown"
    Gauge("app_build_info", "Application build metadata", ["version"]).labels(
        version=app_version
    ).set(1)


class _SPAStaticFiles(StaticFiles):
    """Serve built files, falling back to index.html so client-side routes resolve.

    The SPA owns paths like ``/derivations`` that have no file on disk; a hard refresh
    there must still load the app rather than 404, so a missing path returns index.html
    and the client router takes over.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            response = await super().get_response("index.html", scope)
        # index.html references content-hashed asset filenames, so it must revalidate on
        # every load: otherwise a browser serves a stale index that points at an old
        # bundle and never picks up a new build. Hashed assets (with a file extension)
        # keep their default long-lived cache.
        last_segment = path.rsplit("/", 1)[-1]
        is_index = path in ("", ".", "index.html") or "." not in last_segment
        if is_index:
            response.headers["Cache-Control"] = "no-cache"
        return response


Rows = list[dict[str, str]]

#: Wall-clock budget for a chat's run_code calls. Generous because the model prototypes
#: model fitting here (the 30s library default cut cross-validated training short),
#: matching the derivation authoring budget; still bounded so a runaway cannot hang.
_RUN_CODE_TIMEOUT_SECONDS = 300.0

#: How many recent conversation messages to replay into the model's context (a sliding
#: window; older turns are represented by the certified-derivation memory instead).
_HISTORY_MESSAGES = 12
#: How many of a conversation's certified derivations to list as durable memory.
_MEMORY_LIMIT = 12
#: A stateful session is kept alive between a conversation's turns, then evicted once
#: idle this long, and the fleet is capped so long-lived containers cannot accumulate.
_SESSION_IDLE_SECONDS = 1800.0
_MAX_LIVE_SESSIONS = 32

#: LLM profile names: 1-64 chars, starting alphanumeric, then alphanumerics, spaces,
#: or ./-/_ (they are display labels, e.g. "Sonnet 5"; the value is trimmed first).
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
#: The most profiles a user can register.
_MAX_PROFILES = 10


def _reasoning_efforts() -> frozenset[str]:
    """The reasoning budgets a model may be given, taken from LiteLLM not listed here.

    Hardcoding this set was the original mistake. It read `{"low", "medium", "high"}`,
    which left out `none` -- and OpenAI's newer reasoning models apply an effort of
    their own and then refuse function tools alongside it, so a model whose only
    workable value was `none` could not be configured at all and failed on every
    question. Listing values by hand also missed `minimal` and `xhigh`, and would have
    missed the next one.

    LiteLLM is what has to honour the value, so LiteLLM decides what is sayable. The
    fallback is only for an install without the extra, and is deliberately the widest
    set rather than the narrowest: refusing a value the provider accepts is the failure
    this replaces.
    """
    try:
        from litellm.types.llms.openai import REASONING_EFFORT

        return frozenset(get_args(REASONING_EFFORT))
    except Exception:  # pragma: no cover - only without the litellm extra
        return frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


def _turn_failure_message(exc: Exception) -> str:
    """What to tell the user when a turn failed, and whether trying again is advice.

    A provider that rejects the request with a 4xx will reject it identically every
    time, so "please try again" is not merely unhelpful -- it sends someone round a loop
    that cannot terminate. It also hides the one sentence that says what to change: a
    model refusing function tools alongside its own reasoning effort took a log dive to
    find, and the provider had said so plainly.

    Anything else is treated as transient, the safer assumption for a timeout, a
    disconnect or a tool that fell over.
    """
    status = getattr(exc, "status_code", None)
    detail = str(exc).strip()
    if isinstance(status, int) and 400 <= status < 500 and detail:
        return (
            f"The model provider rejected this request: {detail} This is a "
            "configuration problem rather than a transient one, so it will keep "
            "happening until the model or its settings change."
        )
    return (
        "This analysis could not be completed due to a server error. Please try again."
    )


#: Providers whose name is not its own best label.
_PROVIDER_LABELS: dict[str, str] = {
    "azure_ai": "Azure AI",
    "azure": "Azure OpenAI",
    "bedrock_mantle": "Amazon Bedrock",
    "bedrock": "Amazon Bedrock",
    "github_copilot": "GitHub Copilot",
    "hosted_vllm": "vLLM",
    "lm_studio": "LM Studio",
    "nvidia_nim": "NVIDIA NIM",
    "oci": "Oracle Cloud",
    "sap": "SAP",
    "ollama_chat": "Ollama",
    "openai": "OpenAI",
    "sagemaker_chat": "Amazon SageMaker",
    "sagemaker_nova": "Amazon SageMaker",
    "sagemaker": "Amazon SageMaker",
    "vercel_ai_gateway": "Vercel AI Gateway",
    "vertex_ai_beta": "Google Vertex AI",
    "vertex_ai": "Google Vertex AI",
    "watsonx_text": "IBM watsonx",
    "watsonx": "IBM watsonx",
    "xai": "xAI",
}


def _provider_label(provider: str) -> str:
    """A provider's display name, title-cased from its key when none is given."""
    if not provider:
        return ""
    return _PROVIDER_LABELS.get(
        provider, provider.replace("_", " ").title().replace(" Ai", " AI")
    )


def _model_provider(model: str) -> str:
    """Which provider a model string routes to, resolved rather than parsed.

    Splitting on "/" in the browser gets this wrong in both directions: a bare
    `gpt-5.6-terra` has no prefix and is still OpenAI, while `bedrock/us.anthropic.…`
    has one that is not the model's family. LiteLLM already resolves this for routing,
    so the same answer is reused instead of guessed, and the UI is told rather than
    asked to infer.
    """
    if not model.strip():
        return ""
    try:
        import litellm

        return str(litellm.get_llm_provider(model)[1] or "")
    except Exception:
        # Unknown or no litellm: fall back to the prefix, which is right often enough
        # to pick an icon and never load-bearing.
        return model.split("/")[0].lower() if "/" in model else ""


def _model_reasoning_efforts(model: str) -> list[str]:
    """The efforts this model accepts, for a picker that offers only what will work.

    Read from LiteLLM's model registry, which carries a flag per effort. Presentation
    only: the registry reports gpt-5.6-terra as supporting reasoning *and* function
    calling, which is true separately and false together, so what a model will actually
    accept is settled by the provider at call time, not here. Being wrong in this
    function costs a dropdown entry; being wrong about validity costs a broken chat.
    """
    if not model.strip():
        return []
    try:
        import litellm

        if not litellm.supports_reasoning(model):
            return []
        info = litellm.get_model_info(model)
    except Exception:
        return []
    # A flag that is explicitly False is a refusal; None means unspecified, and the
    # registry only carries flags for the values that vary, so absence means "usual".
    # Ordered by how much thinking they ask for, not alphabetically, because this list
    # populates a control someone reads left to right.
    ladder = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    sayable = _reasoning_efforts()
    return [
        effort
        for effort in ladder
        if effort in sayable
        and info.get(f"supports_{effort}_reasoning_effort") is not False
    ]


def _profile_view(profile: LlmProfile) -> dict[str, Any]:
    """The wire form of a profile: its config without the secret (masked to a bool)."""
    return {
        "name": profile.name,
        "model": profile.model,
        "base_url": profile.base_url,
        "reasoning_effort": profile.reasoning_effort,
        "api_key_set": bool(profile.api_key),
        # Resolved server-side: the browser cannot do either of these without shipping
        # LiteLLM's model registry to it.
        "provider": _model_provider(profile.model),
        "provider_label": _provider_label(_model_provider(profile.model)),
        "reasoning_efforts": _model_reasoning_efforts(profile.model),
    }


def _history_from(messages: Sequence[Any]) -> list[tuple[str, str]]:
    """The prior turns as (role, text), oldest first, for replay into context.

    Only the user's questions and the assistant's answers are carried; the tool traces
    are dropped (their durable outcome rides in the derivation memory instead), which is
    the tool-output clearing that keeps a long conversation from bloating the context.
    The caller windows this to the recent turns and condenses the rest.
    """
    turns: list[tuple[str, str]] = []
    for message in messages:
        text = (message.content or "").strip()
        if not text:
            continue
        role = "assistant" if message.role == "assistant" else "user"
        turns.append((role, text))
    return turns


#: Cap stored request/prediction JSON per inference event; the table is a
#: monitoring window, not a byte-perfect archive of every payload.
_INFERENCE_JSON_CAP = 20_000


def _request_rows(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The feature records inside an invocations payload, best-effort.

    Normalizes the framings that carry named records; positional tensor payloads
    come back empty (there is nothing named to monitor drift on).
    """
    data = body.get("dataframe_records") or body.get("instances") or body.get("inputs")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [r for r in data if isinstance(r, dict)]
    split = body.get("dataframe_split")
    if isinstance(split, dict):
        columns = split.get("columns")
        rows = split.get("data")
        if isinstance(columns, list) and isinstance(rows, list):
            return [
                dict(zip(columns, row, strict=False))
                for row in rows
                if isinstance(row, list)
            ]
    return []


def _train_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """A training spec mapping as ``ModelService.train_result`` keyword args."""
    return {
        "name": str(spec["name"]),
        "dataset": str(spec["dataset"]),
        "target": str(spec["target"]),
        "features": list(spec.get("features") or []),
        "task": str(spec.get("task") or "auto"),
        "time_budget": float(spec.get("time_budget") or 60.0),
        "metric": spec.get("metric"),
        "ensemble": bool(spec.get("ensemble", False)),
        "time_col": spec.get("time_col"),
        "horizon": int(spec["horizon"]) if spec.get("horizon") is not None else None,
        "groups": spec.get("groups"),
        "source_kind": str(spec.get("source_kind") or "dataset"),
        "engine": str(spec.get("engine") or "flaml"),
    }


def _source_of(body: Mapping[str, Any]) -> tuple[str, str]:
    """The training/scoring source in a request: (kind, name).

    ``derivation`` names a certified feature derivation; ``training_set`` a
    materialized point-in-time join; ``dataset`` a bound dataset. Exactly one may be
    given.
    """
    named = [
        (kind, str(body.get(kind) or "").strip())
        for kind in ("derivation", "training_set", "dataset")
    ]
    given = [(kind, name) for kind, name in named if name]
    if len(given) > 1:
        raise HTTPException(
            status_code=400,
            detail="give exactly one of dataset, derivation, or training_set",
        )
    return given[0] if given else ("dataset", "")


def _inline_job(label: str, result: dict[str, Any]) -> WireJob:
    """A finished job for work that ran in the request rather than on a runner."""
    now = time.time()
    return WireJob(
        id=new_job_id(),
        label=label,
        state="succeeded",
        progress="",
        result=result,
        error=None,
        created_at=now,
        started_at=now,
        finished_at=now,
    )


def _job_payload(job: Any) -> WireJob:
    """Serialize a background job for the API (state, progress, and its result)."""
    return WireJob(
        id=job.id,
        label=job.label,
        state=job.state,
        progress=job.progress,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _memory_from(derivations: Sequence[Any]) -> str:
    """A compact note of the conversation's certified derivations, as durable memory."""
    lines: list[str] = []
    for d in list(derivations)[-_MEMORY_LIMIT:]:
        claim = json.loads(d.claim_json) if getattr(d, "claim_json", None) else {}
        roles = ", ".join(f"{k}={v}" for k, v in claim.items()) if claim else ""
        detail = f": {roles}" if roles else ""
        lines.append(f"- {d.name} [{d.verdict}]{detail}")
    return "\n".join(lines)


#: Bounds on a workbench query's row cap; a client value is clamped into this range.
_MIN_QUERY_ROWS = 1
_MAX_QUERY_ROWS = 10_000
_DEFAULT_QUERY_ROWS = 1000


def _clamp_rows(value: Any) -> int:
    """Clamp a client-supplied row cap into range, defaulting when absent or bad."""
    try:
        rows = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_QUERY_ROWS
    return max(_MIN_QUERY_ROWS, min(rows, _MAX_QUERY_ROWS))


def create_app(
    *,
    load_datasets: Callable[[], dict[str, Rows]],
    client: LLMClient | None = None,
    mcp_server: Any | None = None,
    static_dir: Path | None = None,
    cors_origins: Sequence[str] = (),
    store: Store | None = None,
    derive_factory: DeriveFactory | None = None,
    generate_title: Callable[[str], str | None] | None = None,
    make_client: Callable[[str | None], LLMClient] | None = None,
    models: Sequence[dict[str, Any]] = (),
    scratch_root: Path | None = None,
    sandbox: str = "subprocess",
    egress: str | Sequence[str] = "full",
    sandbox_image: str | None = None,
    enable_maintenance: bool = False,
    model_service: ModelService | None = None,
    notebook_service: NotebookService | None = None,
    dashboard_service: DashboardService | None = None,
    feature_store_service: FeatureStoreService | None = None,
    lineage_service: LineageService | None = None,
    orchestration_service: OrchestrationService | None = None,
    explore_service: ExploreService | None = None,
    metric_service: MetricService | None = None,
    monitor_service: MonitorService | None = None,
    warehouse_service: WarehouseService | None = None,
    reload_project: Callable[[], dict[str, Any]] | None = None,
    render_derivation: Callable[[str], str | None] | None = None,
    withhold_rendering: Callable[[str], bool] | None = None,
    certificate_issuer: CertificateIssuer | None = None,
    search_builder: SearchBuilder | None = None,
    search_drain: Drain | None = None,
    derivation_on_trash: Callable[[str], bool] | None = None,
    derivation_on_restore: Callable[[str], bool] | None = None,
    derivation_on_erase: Callable[[str], bool] | None = None,
) -> FastAPI:
    """Build the app over a dataset loader and an LLM client.

    ``mcp_server`` is an optional :class:`mcp.server.MCPServer` (its
    streamable-HTTP app is mounted at ``/mcp``). ``static_dir`` serves a built SPA;
    ``cors_origins`` enables a decoupled SPA hosted elsewhere. ``store`` persists
    conversations, messages, and authored derivations; ``derive_factory`` gives each
    run the capability to author a derivation as its answer. ``models`` is the picker's
    catalogue and ``make_client`` builds a client for a chosen model id, so a request
    can select one (falling back to ``client``). All come from the project in
    :func:`elbi.serve.build`.

    ``scratch_root`` roots the per-conversation ``run_code`` workspaces: each
    conversation gets ``scratch_root/<conversation_id>``, so a chat's exploration
    files persist across its calls but never leak into another chat. It needs a
    ``store`` (a conversation to key on); ``None`` keeps every call ephemeral.

    ``model_service`` (the ml extra) powers the chat's model-lifecycle tools, the
    registry API under ``/api/registry``, and MLflow-protocol scoring under
    ``/api/serving``; ``None`` degrades those surfaces to an install hint.

    Trust model: the app is local-first and single-user. Its endpoints are
    unauthenticated and everything it holds -- conversations, jobs, derivations -- is
    the one person's, so it assumes a trusted user on a trusted machine. Reaching it
    from anywhere else means putting something that authenticates in front of it. The
    oracle's verdict stays unforgeable either way, so this is a data-isolation
    boundary, not a soundness one.
    """
    mcp_app = None
    if mcp_server is not None:
        mcp_app = mcp_server.streamable_http_app(
            # Mounted at /mcp, so the sub-app answers at its own root.
            streamable_http_path="/",
            # The mounted sub-app does not own the socket, so it does not decide which
            # Host values are legitimate. MCP auto-enables DNS-rebinding protection with
            # a localhost-only allowlist whenever no settings are passed, which answers
            # 421 to every deployed hostname.
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            ),
            # Deployments scale this app horizontally with no sticky routing, so a
            # handshake-era client must not depend on which replica answered its
            # initialize: a session id lives in one process, and the next POST landing
            # elsewhere is answered "session not found". A born-ready transport per
            # request costs the standalone notification stream, which is why a tool-list
            # change is sent on the originating request instead.
            # `test_app.py` asserts no session header is issued.
            stateless_http=True,
        )

    def _default_client() -> LLMClient:
        """The default LLM client: the injected one, else built from configuration.

        Raises when no model is configured (an :class:`ElbiError` subclass), so
        a surface that needs the LLM fails with a clear, actionable message instead of
        silently defaulting to a model or key. Used where no per-request profile applies
        (background jobs, NL→SQL); the chat resolves per conversation via
        ``make_client``.
        """
        if client is not None:
            return client
        if make_client is not None:
            return make_client(None)
        raise ElbiError(
            "No LLM model is configured. Add a model profile in Settings or set "
            "LLM_MODEL."
        )

    # Live exploration workspaces, keyed by conversation id, kept between a
    # conversation's turns so its session (namespace, fitted models, files) survives.
    # Assumes turns of one conversation are sequential, as a chat UI sends them.
    live_workspaces: dict[str, Workspace] = {}
    workspace_seen: dict[str, float] = {}

    # Durable background jobs (long training runs): submitted from a conversation, they
    # run detached, persist to the store, and are observed via the /api/jobs endpoints.
    # A finished derivation job re-pings the agent through _on_job_complete (below), so
    # the promised follow-up actually happens instead of the job dropping into oblivion.
    # Keyed by the job's content address (not its id) and holding every conversation
    # that asked for it: a job shared by two conversations (content-addressed dedupe)
    # posts a follow-up to each, and the mapping is registered before the work can
    # finish so a fast job never completes before its conversation is known.
    job_followups: dict[str, set[str]] = {}
    job_followups_lock = threading.Lock()

    def _on_job_complete(job: Any) -> None:
        """Re-enter every conversation waiting on a certified background derivation.

        The completion is a channel-style event: the certified result (already checked
        by the oracle) is injected as context into an off-request turn running the same
        agent loop, seeded with each conversation's history and the machine-checked
        attestation, so the model narrates the finding and the answer is posted back
        verified. Code jobs (no ``certified`` key) and uncertified results are left to
        be read from /api/jobs; only a sound derivation earns a follow-up answer.
        """
        with job_followups_lock:
            conversations = job_followups.pop(job.key, set())
        # Notified here, not in the work closure, so restart-orphaned jobs (where
        # no user code runs) are covered too. In-app only: no webhook analogue.
        if job.state == "failed" and job.label.startswith("train model "):
            create_notifications(
                store,
                TRAINING_FAILED,
                {
                    "name": job.label.removeprefix("train model "),
                    "error": job.error or "",
                },
            )
        result = job.result if isinstance(job.result, dict) else {}
        if (
            not conversations
            or store is None
            or job.state != "succeeded"
            or not result.get("certified")
        ):
            return
        checks = tuple(tuple(check) for check in result.get("checks", []))
        verdict = result.get("verdict")
        attestation = AnswerResult(
            verified=bool(verdict),
            verdict=str(verdict or "unverified"),
            narrative="",
            checks=checks,
            data_hash=result.get("data_hash"),
            spec={"derivation": job.label},
        )
        prompt = (
            f"Your background training job for derivation '{job.label}' has finished "
            f"and CERTIFIED (oracle verdict: {attestation.verdict}). Its verified "
            f"output:\n{result.get('rendered', '')}\n\nInterpret this certified result "
            "and call `answer` now with the finding for the user, in plain language. "
            "The analysis is already certified, so do not run tools or derive again."
        )
        for conversation_id in conversations:
            try:
                final = answer(
                    prompt,
                    Workspace(datasets=load_datasets()),
                    _default_client(),
                    history=_history_from(store.messages(conversation_id)),
                    memory=_memory_from(store.derivations_for(conversation_id)),
                    verified=attestation,
                )
            except Exception:
                # The result is already certified; a narration failure (an LLM or API
                # error) must not silently swallow the follow-up the user was promised.
                # Log it, then post the verified result verbatim so the answer arrives.
                logger.exception(
                    "background follow-up narration failed for job %s (%s)",
                    job.id,
                    job.label,
                )
                final = replace(
                    attestation,
                    narrative=(
                        f"The certified result for '{job.label}' is ready "
                        f"(verdict: {attestation.verdict}).\n\n"
                        f"{result.get('rendered', '')}"
                    ),
                )
            try:
                store.add_message(
                    conversation_id,
                    "assistant",
                    content=final.narrative,
                    result=_result_payload(final),
                )
            except Exception:
                logger.exception(
                    "failed to persist background follow-up for job %s (%s)",
                    job.id,
                    job.label,
                )

    job_runner = (
        JobRunner(store=store.job_store(), on_complete=_on_job_complete)
        if store is not None
        else None
    )

    def _gc_workspaces() -> None:
        """Evict and close workspaces that have gone idle or overflow the cap."""
        now = time.monotonic()
        drop = {
            c
            for c, seen in workspace_seen.items()
            if now - seen > _SESSION_IDLE_SECONDS
        }
        keep = [(s, c) for c, s in workspace_seen.items() if c not in drop]
        if len(keep) > _MAX_LIVE_SESSIONS:
            keep.sort()  # oldest-seen first
            drop |= {c for _, c in keep[: len(keep) - _MAX_LIVE_SESSIONS]}
        for c in drop:
            workspace_seen.pop(c, None)
            ws = live_workspaces.pop(c, None)
            if ws is not None:
                with suppress(Exception):
                    ws.close()

    def _after_training(result: dict[str, Any], *, notify: bool = True) -> None:
        """Record and announce a registered version (audit, notification, webhook).

        ``notify=False`` keeps the audit row and the webhook (the version really
        was registered) but suppresses the in-app notification: a cancelled job
        whose work ran to completion anyway must not tell its submitter that it
        succeeded.
        """
        target = f"{result['name']}/v{result['version']}"
        if store is not None:
            store.record_audit(
                "train_model",
                target_type="model_version",
                target_id=target,
            )
        created = {
            "name": result["name"],
            "version": result["version"],
            "task": result["task"],
            "metrics": result["metrics"],
            "oracle_verdict": result["oracle_verdict"],
        }
        if notify:
            dispatch_event(store, "model_version.created", created)
        else:
            fire_model_event(store, "model_version.created", created)
        if result.get("champion"):
            fire_model_event(
                store,
                "alias.updated",
                {
                    "name": result["name"],
                    "alias": "champion",
                    "version": result["version"],
                },
            )

    def _submit_training(spec: Mapping[str, Any]) -> Any:
        """Launch a training run on the job runner (shared by the UI and chat).

        Identical requests dedupe onto the live job (a double-click trains once),
        but the key carries the model's current latest version, so training the
        same spec again after it finishes produces the next version rather than
        replaying the old result.
        """
        service = model_service
        if service is None or job_runner is None:  # bound only when both exist
            raise RuntimeError("training jobs need the model service and a store")
        name = str(spec["name"])
        dataset = str(spec["dataset"])
        budget = float(spec.get("time_budget") or 60.0)

        def work(
            progress: Callable[[str], None], cancelled: Callable[[], bool]
        ) -> dict[str, Any]:
            progress(f"AutoML search over {dataset!r} ({budget:.0f}s budget)")
            result = service.train_result(**_train_kwargs(spec))
            progress(f"registered {result['name']} v{result['version']}")
            # The runner discards a cancelled job's result; announce no success.
            _after_training(result, notify=not cancelled())
            return result

        key = hash_json(
            _train_kwargs(spec) | {"onto_version": service.latest_version(name)}
        )
        # The submitter rides on the job row, so a restart can still attribute it.
        return job_runner.submit(key, f"train model {name}", work)

    def _chat_train(spec: Mapping[str, Any]) -> str:
        """The chat's TrainFn: inline for short budgets, a background job beyond.

        An inline tool call blocks the model's turn, so a budget past the inline
        allowance goes to the job runner (the same path the UI's train button
        uses) and the model reports the job instead of stalling the conversation.
        """
        service = model_service
        if service is None:  # bound onto workspaces only when the service exists
            return "error:\nmodel training is unavailable"
        name = str(spec["name"])
        time_budget = float(spec.get("time_budget") or 60.0)
        if job_runner is not None and time_budget > INLINE_TRAIN_BUDGET:
            try:
                service.check_train(
                    name,
                    str(spec["dataset"]),
                    str(spec["target"]),
                    str(spec.get("source_kind") or "dataset"),
                )
            except ModelError as exc:
                return f"error:\n{exc}"
            job = _submit_training(spec)
            return (
                f"Training {name!r} launched as background job {job.id} with a "
                f"{time_budget:.0f}s budget (long budgets run server-side rather "
                "than blocking this conversation). The new version appears in the "
                "registry when it finishes; check list_models later or watch the "
                "jobs bar. Tell the user it is training, then continue."
            )
        try:
            result = service.train_result(
                **_train_kwargs(
                    dict(spec) | {"time_budget": min(time_budget, INLINE_TRAIN_BUDGET)}
                )
            )
        except ModelError as exc:
            return f"error:\n{exc}"
        _after_training(result)
        rendered: str = result["rendered"]
        return rendered + (
            f"\n\nServing: POST /api/serving/{result['name']}/invocations with an "
            'MLflow scoring payload (e.g. {"dataframe_records": [...]}).'
        )

    def _retrain_tick() -> None:
        """Run due retraining policies (called by the maintenance scheduler).

        ``on_data_change`` policies key on the dataset's content hash, the same
        identity the platform uses everywhere; ``interval`` policies key on the
        clock. Each due policy submits the standard training job (dedupe makes a
        doubled tick harmless), then records what it saw, so a policy fires once
        per change or period rather than every tick.
        """
        if store is None or model_service is None or job_runner is None:
            return
        for policy in store.list_retrain_policies():
            if not policy.enabled:
                continue
            try:
                if policy.mode == "interval":
                    due = policy.last_run_at is None or (
                        datetime.now(timezone.utc)
                        - policy.last_run_at.replace(tzinfo=timezone.utc)
                    ) >= timedelta(hours=policy.interval_hours)
                    if not due:
                        continue
                    data_hash = None
                else:
                    try:
                        rows = model_service.load_source(
                            policy.source_kind, policy.dataset
                        )
                    except ModelError:
                        continue
                    data_hash = hash_json(rows)
                    if data_hash == policy.last_data_hash:
                        continue
                _submit_training(
                    {
                        "name": policy.model,
                        "dataset": policy.dataset,
                        "source_kind": policy.source_kind,
                        "engine": policy.engine,
                        "target": policy.target,
                        "features": list(json.loads(policy.features_json or "[]")),
                        "task": policy.task,
                        "time_budget": policy.time_budget,
                        "metric": policy.metric,
                        "ensemble": policy.ensemble,
                        "time_col": policy.time_col,
                        "horizon": policy.horizon,
                        "groups": policy.groups,
                    }
                )
                store.mark_retrain(policy.model, data_hash=data_hash)
                logger.info("retraining %s (%s policy)", policy.model, policy.mode)
            except Exception:
                logger.exception("retrain policy for %s failed", policy.model)

    def _drift_tick() -> None:
        """Run scheduled drift checks (called by the maintenance scheduler).

        Every registered model with enough stored traffic gets checked at most
        once per interval; a dataset-drift outcome fires the webhook, so the
        loop is Lakehouse-Monitoring-shaped: traffic in, drift verdicts and
        alerts out, no per-model setup required.
        """
        if store is None or model_service is None:
            return
        from elbi_core.ml import MIN_DRIFT_ROWS

        try:
            interval_hours = max(
                1.0,
                float(runtime_settings.value_of("drift_check_interval_hours", store)),
            )
        except ValueError:
            interval_hours = 24.0
        for model in model_service.registry().models():
            try:
                last = _last_drift_check_ms(model.name)
                age_hours = (time.time() * 1000 - last) / 3_600_000 if last else None
                if age_hours is not None and age_hours < interval_hours:
                    continue
                current = store.inference_inputs(model.name)
                if len(current) < MIN_DRIFT_ROWS:
                    continue
                result = model_service.drift_result(model.name, current)
                if result["dataset_drift"]:
                    dispatch_event(
                        store,
                        "drift.detected",
                        {
                            "name": model.name,
                            "version": result["version"],
                            "share_drifted": result["share_drifted"],
                            "n_drifted": result["n_drifted"],
                        },
                    )
                logger.info(
                    "drift check for %s: share=%.2f drift=%s",
                    model.name,
                    result["share_drifted"],
                    result["dataset_drift"],
                )
            except Exception:
                logger.exception("drift check for %s failed", model.name)

    def _feature_drift_tick() -> None:
        """Run scheduled feature-view drift checks (maintenance scheduler).

        Every feature view with a baseline is checked at most once per interval; a
        dataset-drift outcome fires the webhook. A view without a baseline is skipped,
        so monitoring is opt-in per view: the same traffic-in, alerts-out shape as the
        model drift loop.
        """
        if store is None or feature_store_service is None:
            return
        try:
            interval_hours = max(
                1.0,
                float(runtime_settings.value_of("feature_drift_interval_hours", store)),
            )
        except ValueError:
            interval_hours = 24.0
        now = datetime.now(timezone.utc)
        for view in feature_store_service.catalog():
            name = view["name"]
            try:
                if store.get_feature_baseline(name) is None:
                    continue
                recent = store.list_feature_drift(name, limit=1)
                if recent:
                    last_at = recent[0].at
                    if last_at.tzinfo is None:
                        last_at = last_at.replace(tzinfo=timezone.utc)
                    if (now - last_at).total_seconds() / 3600 < interval_hours:
                        continue
                result = feature_store_service.drift(name)
                if result["dataset_drift"]:
                    dispatch_event(
                        store,
                        "feature.drift_detected",
                        {
                            "feature_view": name,
                            "share_drifted": result["share_drifted"],
                            "n_drifted": result["n_drifted"],
                        },
                    )
                logger.info(
                    "feature drift check for %s: share=%.2f drift=%s",
                    name,
                    result["share_drifted"],
                    result["dataset_drift"],
                )
            except FeatureStoreError:
                continue  # e.g. too few rows this cycle: retry next tick
            except Exception:
                logger.exception("feature drift check for %s failed", name)

    def _notebook_tick() -> None:
        """Run notebooks whose schedule is due (called by the maintenance scheduler).

        Three modes, the shape a scheduler is expected to offer: ``cron`` for a
        wall-clock time in a named zone ("the 1st at 06:00"), ``interval`` for a plain
        period, and ``on_data_change`` keyed on a dataset's content hash, the same
        identity the retrain policies use. A due notebook runs to completion with its
        stored parameters on a fresh kernel, then records when it ran (and the hash
        seen), firing once per period or change.

        An interval cannot express a calendar time: 720 hours drifts off the 1st by the
        length of each run, so a monthly job wanders.
        """
        if store is None or notebook_service is None:
            return
        now = datetime.now(timezone.utc)
        for row in store.notebooks_with_schedule():
            try:
                schedule = json.loads(row.schedule_json or "{}")
                if not schedule.get("enabled"):
                    continue
                params = schedule.get("params") or {}
                data_hash: str | None = None
                mode = schedule.get("mode")
                last = schedule.get("last_run_at")
                if mode == "on_data_change":
                    rows = load_datasets().get(schedule.get("dataset") or "")
                    if rows is None:
                        continue
                    data_hash = hash_json(rows)
                    if data_hash == schedule.get("last_data_hash"):
                        continue
                elif mode == "cron":
                    if not cron_due(
                        str(schedule.get("cron") or ""),
                        datetime.fromisoformat(last) if last else None,
                        now,
                        str(schedule.get("timezone") or ""),
                    ):
                        continue
                else:
                    hours = float(schedule.get("interval_hours") or 24)
                    interval = timedelta(hours=hours)
                    if last and now - datetime.fromisoformat(last) < interval:
                        continue
                result = notebook_service.run_scheduled(row.id, params)
                schedule["last_run_at"] = now.isoformat()
                if data_hash is not None:
                    schedule["last_data_hash"] = data_hash
                store.update_notebook(row.id, schedule_json=json.dumps(schedule))
                store.record_audit(
                    "notebook.run", target_type="notebook", target_id=row.id
                )
                logger.info("ran scheduled notebook %s: %s", row.id, result)
            except Exception:
                logger.exception("scheduled notebook %s failed", row.id)

    def _deliver_dashboard(
        channel: str, recipients: list[str], subject: str, body: str
    ) -> None:
        """Deliver a rendered snapshot over a subscription's channel (email/webhook)."""
        if channel == "email":
            from .mailer import send as send_email

            for to in recipients:
                send_email(str(to), subject, body)
            return
        if channel == "webhook":
            import http.client
            from urllib.parse import urlparse

            payload = json.dumps({"subject": subject, "body": body}).encode("utf-8")
            for url in recipients:
                parsed = urlparse(str(url))
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    continue
                connection: http.client.HTTPConnection
                if parsed.scheme == "https":
                    connection = http.client.HTTPSConnection(
                        parsed.hostname, parsed.port, timeout=10
                    )
                else:
                    connection = http.client.HTTPConnection(
                        parsed.hostname, parsed.port, timeout=10
                    )
                path = parsed.path or "/"
                if parsed.query:
                    path = f"{path}?{parsed.query}"
                try:
                    connection.request(
                        "POST",
                        path,
                        body=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    connection.getresponse().read()
                except Exception:
                    logger.exception("dashboard webhook to %s failed", url)
                finally:
                    connection.close()

    def _dashboard_delivery_tick() -> None:
        """Deliver due dashboard subscriptions (called by the maintenance scheduler).

        A subscription fires at most once a day: it delivers a snapshot of its page,
        resolved from the published dashboard under the saved filter state, then records
        when it ran so a missed or doubled tick never double-sends within the window.
        """
        if store is None or dashboard_service is None:
            return
        now = datetime.now(timezone.utc)
        for subscription in store.active_dashboard_subscriptions():
            try:
                last = subscription.last_run_at
                if last is not None:
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    if now - last < timedelta(hours=24):
                        continue
                subject, body = dashboard_service.render_delivery(subscription)
                recipients = json.loads(subscription.recipients_json or "[]")
                _deliver_dashboard(subscription.channel, recipients, subject, body)
                store.touch_dashboard_subscription(subscription.id)
                store.record_audit(
                    "dashboard.deliver",
                    target_type="dashboard",
                    target_id=subscription.dashboard_id,
                )
                logger.info("delivered dashboard subscription %s", subscription.id)
            except Exception:
                logger.exception("dashboard delivery %s failed", subscription.id)

    def _orchestration_tick() -> None:
        """Fire due materialization schedules (called by the maintenance scheduler)."""
        if orchestration_service is None:
            return
        try:
            fired = orchestration_service.run_due_schedules()
            if fired:
                logger.info("materialized due schedules: %s", fired)
        except Exception:
            logger.exception("orchestration tick failed")

    def _monitor_tick() -> None:
        """Run each monitor whose interval has elapsed (called by the scheduler).

        Each monitor is isolated: a target that fails to resolve logs and does not stop
        the others. Anomalies raise alerts through the monitor's own incident dedup.
        """
        if monitor_service is None or store is None:
            return
        for monitor in store.due_metric_monitors(datetime.now(timezone.utc)):
            try:
                monitor_service.run(monitor)
            except MonitorError:
                logger.exception("monitor %s failed", monitor.id)

    def _derivation_trash_tick() -> None:
        """Erase derivations whose trash retention window has elapsed.

        Not part of ``Store.purge_trash``: erasing a derivation needs the
        runtime hooks (the live registry, the sidecar), which the store cannot
        reach on its own. Uses the same ``derivation_on_erase`` hook the API's
        immediate-erase action calls, so the sweep and the button are provably
        the same code path.
        """
        if store is None or derivation_on_erase is None:
            return
        from .maintenance import _trash_retention_days

        days = _trash_retention_days(store)
        for name in store.expired_trashed_derivations(days):
            derivation_on_erase(name)

    def _warehouse_sync_tick() -> None:
        """Sync warehouse sources whose cadence has elapsed (maintenance scheduler)."""
        if warehouse_service is None:
            return
        synced = warehouse_service.sync_due(datetime.now(timezone.utc))
        if synced:
            logger.info("synced due warehouse sources: %s", synced)

    def _search_tick() -> None:
        """Bring the search index up to date (maintenance scheduler).

        The only thing that calls a build. The last attempt shipped an indexer wired to
        every consumer with nothing on this side of the seam, so the index stayed empty
        and search returned nothing -- through a full test suite, because every test
        wrote its own documents first.

        Idempotent by digest: on the housekeeping cadence it costs a scan of what exists
        and re-embeds only what changed.
        """
        if search_builder is None:
            return
        try:
            report = search_builder.build()
        except Exception:
            logger.exception("search index build failed")
            return
        if report.indexed or report.removed:
            log_counts(report)

    def _last_drift_check_ms(name: str) -> int:
        """Epoch ms of the model's newest drift-check run, or 0."""
        import mlflow

        from elbi_core.ml.training import scoped_tracking

        service = model_service
        if service is None:
            return 0
        with scoped_tracking(service.tracking_uri()):
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is None:
                return 0
            runs = mlflow.search_runs(
                [experiment.experiment_id],
                filter_string="tags.`elbi.drift_check` = 'true'",
                max_results=1,
                output_format="list",
            )
        return int(runs[0].info.start_time) if runs else 0

    def _bind_model_tools(ws: Workspace) -> Workspace:
        """Give a workspace the model-lifecycle capabilities, when the extra is in."""
        if model_service is not None:
            ws.train = lambda spec: _chat_train(spec)
            ws.list_models = model_service.render_models
            ws.predict_model = model_service.render_predict
            ws.promote_model = _chat_promote
            ws.batch_score = model_service.render_batch_score
        _bind_notebook_tools(ws)
        _bind_dashboard_tools(ws)
        _bind_feature_tools(ws)
        _bind_lineage_tools(ws)
        _bind_metric_tools(ws)
        _bind_monitor_tools(ws)
        _bind_orchestration_tools(ws)
        return ws

    def _bind_notebook_tools(ws: Workspace) -> None:
        """Give a workspace the notebook-operating capabilities, when a store exists.

        Lets the agent build, run, inspect, promote from, and schedule notebooks: the
        same surface a human drives in the UI, and everything it does is visible and
        editable there.
        """
        if notebook_service is None:
            return
        from .notebook_agent import NotebookAgent

        agent = NotebookAgent(notebook_service)
        ws.nb_write = agent.write
        ws.nb_run = agent.run
        ws.nb_read = agent.read
        ws.nb_list = agent.list
        ws.nb_promote_cell = agent.promote_cell
        ws.nb_schedule = agent.schedule

    def _bind_dashboard_tools(ws: Workspace) -> None:
        """Give a workspace the dashboard-operating capabilities, when one exists.

        Lets the agent list, read, author, and publish dashboards: the same surface a
        human drives in the UI. Everything it does is visible and editable there, and
        publishing stays governed by the certified-binding gate.
        """
        if dashboard_service is None:
            return
        from .dashboard_agent import DashboardAgent

        agent = DashboardAgent(dashboard_service)
        ws.dash_list = agent.list
        ws.dash_sources = agent.sources
        ws.dash_read = agent.read
        ws.dash_write = agent.write
        ws.dash_publish = agent.publish

    def _bind_feature_tools(ws: Workspace) -> None:
        """Give a workspace the feature-store capabilities, when one exists.

        Lets the agent discover, define, materialize, and retrieve features: the same
        surface a human drives. Serving stays gated on a certified source derivation.
        """
        if feature_store_service is None:
            return
        from .feature_store_agent import FeatureStoreAgent

        agent = FeatureStoreAgent(feature_store_service)
        ws.fs_list = agent.list
        ws.fs_define = agent.define
        ws.fs_materialize = agent.materialize
        ws.fs_online = agent.get_online
        ws.fs_historical = agent.get_historical
        ws.fs_statistics = agent.statistics
        ws.fs_drift = agent.drift
        ws.fs_expectations = agent.check_expectations
        ws.fs_training_set = agent.create_training_set

    def _bind_lineage_tools(ws: Workspace) -> None:
        """Give a workspace the lineage/catalog capabilities, when a project exists.

        Lets the agent search the catalog and trace provenance/impact across artifacts:
        the same discovery a human does. Read-only.
        """
        if lineage_service is None:
            return
        from .lineage_agent import LineageAgent

        agent = LineageAgent(lineage_service)
        ws.lin_catalog = agent.catalog
        ws.lin_lineage = agent.lineage
        ws.lin_impact = agent.impact

    def _bind_metric_tools(ws: Workspace) -> None:
        """Give a workspace the semantic-layer metric capabilities, given a project.

        Lets the agent define, list, and query metrics over certified derivations: the
        same semantic layer a human drives. Defining a simple metric is gated on its
        source being certified.
        """
        if metric_service is None:
            return
        from .metrics_agent import MetricsAgent

        agent = MetricsAgent(metric_service)
        ws.metric_list = agent.list_metrics
        ws.metric_define = agent.define
        ws.metric_query = agent.query

    def _bind_monitor_tools(ws: Workspace) -> None:
        """Give a workspace the anomaly-monitor capabilities, given a project."""
        if monitor_service is None:
            return
        from .monitoring_agent import MonitorsAgent

        agent = MonitorsAgent(monitor_service)
        ws.monitor_list = agent.list_monitors
        ws.monitor_create = agent.create

    def _bind_orchestration_tools(ws: Workspace) -> None:
        """Give a workspace the orchestration capabilities, when a project exists.

        Lets the agent see stale assets and materialize them in dependency order: the
        same operational control a human has. Read + idempotent materialization.
        """
        if orchestration_service is None:
            return
        from .orchestration_agent import OrchestrationAgent

        agent = OrchestrationAgent(orchestration_service)
        ws.orch_status = agent.status
        ws.orch_materialize = agent.materialize

    def _chat_promote(name: str, version: int, alias: str) -> str:
        """The chat's promote: the registry move plus the audit/webhook trail."""
        service = model_service
        if service is None:
            return "error:\nmodel promotion is unavailable"
        outcome = service.render_promote(name, version, alias)
        if not outcome.startswith("error"):
            if store is not None:
                store.record_audit(
                    "promote_model",
                    target_type="model_version",
                    target_id=f"{name}/v{version}",
                )
            fire_model_event(
                store,
                "alias.updated",
                {"name": name, "alias": alias, "version": version},
            )
        return outcome

    def _acquire_workspace(
        conversation_id: str,
        derive: DeriveFn | None,
        scratch_dir: Path | None,
    ) -> Workspace:
        """Reuse this conversation's live workspace, or start one; refresh derive."""
        _gc_workspaces()
        ws = live_workspaces.get(conversation_id)
        if ws is None:
            ws = _bind_model_tools(
                Workspace(
                    datasets=load_datasets(),
                    derive=derive,
                    scratch_dir=scratch_dir,
                    stateful=True,
                    backend=sandbox,
                    egress=egress,
                    image=sandbox_image,
                    executor=SubprocessExecutor(timeout=_RUN_CODE_TIMEOUT_SECONDS),
                ),
            )
            live_workspaces[conversation_id] = ws
        else:
            ws.derive = (
                derive  # bind this turn's question for authored-derivation metadata
            )
        workspace_seen[conversation_id] = time.monotonic()
        return ws

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # The mounted MCP app needs its session manager started; nested lifespans
        # are not run automatically, so its context is entered here explicitly. On
        # shutdown, close every live session so no container/process is left running.
        scheduler = None
        if enable_maintenance and store is not None:
            from .maintenance import Scheduler

            scheduler = Scheduler(
                store,
                extra_tasks=(
                    _retrain_tick,
                    _drift_tick,
                    _feature_drift_tick,
                    _notebook_tick,
                    _dashboard_delivery_tick,
                    _orchestration_tick,
                    _monitor_tick,
                    _derivation_trash_tick,
                    _warehouse_sync_tick,
                    _search_tick,
                ),
            )
            scheduler.start()
        try:
            if mcp_app is not None:
                async with mcp_app.router.lifespan_context(mcp_app):
                    yield
            else:
                yield
        finally:
            if scheduler is not None:
                scheduler.stop()
            for ws in list(live_workspaces.values()):
                with suppress(Exception):
                    ws.close()
            live_workspaces.clear()
            workspace_seen.clear()
            if notebook_service is not None:
                with suppress(Exception):
                    notebook_service.close()
            if job_runner is not None:
                with suppress(Exception):
                    job_runner.close()
            if search_drain is not None:
                with suppress(Exception):
                    search_drain.stop()
            if search_builder is not None:
                with suppress(Exception):
                    search_builder.close()

    app = FastAPI(title="elbi", lifespan=lifespan)
    # Exposed for embedders and tests: run the retraining-policy check
    # synchronously (the maintenance scheduler calls the same function).
    app.state.retrain_tick = _retrain_tick
    # Exposed for tests: deliver due dashboard subscriptions synchronously.
    app.state.dashboard_delivery_tick = _dashboard_delivery_tick
    # Exposed for tests: fire due materialization schedules synchronously.
    app.state.orchestration_tick = _orchestration_tick
    # Exposed for tests: run due feature-view drift checks synchronously.
    app.state.feature_drift_tick = _feature_drift_tick
    # Exposed for tests: run due model drift checks synchronously.
    app.state.drift_tick = _drift_tick
    # Exposed for tests: run due metric/derivation monitors synchronously.
    app.state.monitor_tick = _monitor_tick
    app.state.withhold_rendering = withhold_rendering
    # Exposed for tests: sync due warehouse sources synchronously.
    app.state.warehouse_sync_tick = _warehouse_sync_tick
    # Exposed for tests: build the search index synchronously.
    app.state.search_tick = _search_tick
    # The store on app.state so a request-scoped dependency can reach it.
    app.state.store = store

    async def _conversation(
        conversation_id: str, request: Request
    ) -> Conversation | None:
        """The conversation of this id, or ``None`` when there is none."""
        return store.get_conversation(conversation_id) if store else None

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Standardize the JSON API on camelCase field names: responses are camelCased and
    # request bodies snake-cased at the boundary, so handlers keep snake_case internally
    # while the wire matches the TypeScript client and the camelCase spec manifests.
    # OSI/notebook documents and tabular row data are exempt (see casing.py).
    app.add_middleware(CamelCaseResponses)
    app.add_middleware(SnakeCaseRequests)

    @app.get("/health")
    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        """Liveness: the process is up (cheap; ``/health`` aliases it)."""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready(response: Response) -> dict[str, str]:
        """Readiness: the store is reachable (503 pulls the replica from rotation)."""
        if store is not None and not store.ping():
            response.status_code = 503
            return {"status": "unavailable"}
        return {"status": "ok"}

    @app.get("/api/version")
    async def app_version() -> WireVersion:
        """What is running here, and the command that upgrades it.

        Reports only what is already on this machine. It does not ask PyPI whether
        anything newer exists, and neither does anything else the server does: a check
        the operator did not request is exactly the outbound connection ``docs/privacy``
        promises this app never makes. ``elbi update`` is where that question is asked,
        by a person, on purpose.

        The upgrade command is included because the interface has no other way to know
        it -- a browser cannot see whether the process behind it came from a container,
        a uv tool install or a virtualenv, and the answer differs for each.
        """
        from elbi_cli.update import detect_install, distribution

        from . import __version__

        package = distribution()
        install = detect_install(package)
        return WireVersion(
            version=__version__,
            package=package,
            install=WireInstall(
                kind=install.kind, command=install.command, note=install.note
            ),
        )

    @app.get("/api/datasets")
    async def datasets() -> list[WireDataset]:
        """Every queryable dataset and its shape.

        Every dataset is a warehouse table (a project-declared source or a connector
        sync) (the warehouse is the only source of data), so they share one origin.
        """
        return [
            WireDataset(
                name=name,
                rows=len(rows),
                columns=len(rows[0]) if rows else 0,
                origin="warehouse",
            )
            for name, rows in load_datasets().items()
        ]

    @app.get("/api/models")
    async def list_models() -> list[dict[str, Any]]:
        """The models the picker offers (empty when the server fixes one model)."""
        return list(models)

    # -- explore (SQL workbench + profiling) -------------------------------------
    def _explore() -> ExploreService:
        """The exploration service, or a 503 when the store it needs is absent."""
        if explore_service is None:
            raise HTTPException(status_code=503, detail="exploration requires a store")
        return explore_service

    @app.get("/api/explore/sources")
    async def explore_sources() -> list[dict[str, Any]]:
        """The query targets: bound datasets plus registered external data sources."""
        return _explore().sources()

    @app.get("/api/explore/catalog")
    async def explore_catalog(
        request: Request, source_id: str | None = None
    ) -> dict[str, Any]:
        """The tables and columns for a source (bound datasets when unset)."""
        service = _explore()
        try:
            return await run_in_threadpool(service.catalog, source_id)
        except ElbiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/explore/query")
    async def explore_query(request: Request) -> dict[str, Any]:
        """Run read-only SQL and return columns, bounded rows, and a truncation flag."""
        service = _explore()
        body = await request.json()
        sql = str(body.get("sql") or "").strip()
        if not sql:
            raise HTTPException(status_code=400, detail="a query is required")
        max_rows = _clamp_rows(body.get("max_rows"))
        try:
            return await run_in_threadpool(
                lambda: service.run_sql(
                    sql,
                    source_id=body.get("source_id"),
                    max_rows=max_rows,
                )
            )
        except ElbiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/explore/profile")
    async def explore_profile(request: Request) -> dict[str, Any]:
        """Profile a bound dataset (in full) or a query result (sampled)."""
        service = _explore()
        body = await request.json()
        dataset = body.get("dataset")
        sql = str(body.get("sql") or "").strip() or None
        if dataset is None and sql is None:
            raise HTTPException(
                status_code=400, detail="a dataset or query is required"
            )
        try:
            return await run_in_threadpool(
                lambda: service.profile(
                    dataset=dataset,
                    sql=sql,
                    source_id=body.get("source_id"),
                )
            )
        except ElbiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/explore/nl2sql")
    async def explore_nl2sql(request: Request) -> dict[str, str]:
        """Draft a SQL query from a natural-language question, grounded in the schema.

        A convenience, not a governed step: the draft is a starting point to edit and
        run, and only a promoted query is ever certified.
        """
        service = _explore()
        body = await request.json()
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="a prompt is required")
        source_id = body.get("source_id")
        # The chosen source's catalog, so generated SQL names tables that exist rather
        # than ones the model invented.
        try:
            catalog = await run_in_threadpool(service.catalog, source_id)
        except ElbiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        source = store.get_data_source(source_id) if store and source_id else None
        dialect = source.kind if source is not None else "DuckDB"
        try:
            sql = await run_in_threadpool(
                generate_sql,
                prompt,
                catalog["tables"],
                _default_client(),
                dialect=dialect,
            )
        except ElbiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"sql": sql}

    @app.get("/api/explore/queries")
    async def list_saved_queries(request: Request) -> list[dict[str, Any]]:
        """Saved queries, most-recently-updated first."""
        return _explore().list_queries()

    @app.post("/api/explore/queries")
    async def save_saved_query(request: Request) -> dict[str, Any]:
        """Create or update a saved query (an ungoverned exploration artifact)."""
        service = _explore()
        body = await request.json()
        sql = str(body.get("sql") or "").strip()
        if not sql:
            raise HTTPException(status_code=400, detail="a query is required")
        return service.save_query(
            name=str(body.get("name") or ""),
            sql=sql,
            source_id=body.get("source_id"),
            query_id=body.get("id"),
        )

    @app.get("/api/explore/queries/{query_id}")
    async def get_saved_query(request: Request, query_id: str) -> dict[str, Any]:
        """A saved query by id."""
        view = _explore().get_query(query_id)
        if view is None:
            raise HTTPException(status_code=404, detail="query not found")
        return view

    @app.post("/api/explore/queries/{query_id}/duplicate")
    async def duplicate_saved_query(request: Request, query_id: str) -> dict[str, Any]:
        """Copy a saved query into a new one owned by the caller."""
        service = _explore()
        view = service.duplicate_query(query_id)
        if view is None:
            raise HTTPException(status_code=404, detail="query not found")
        _require_store().record_audit(
            "saved_query.duplicate",
            target_type="saved_query",
            target_id=query_id,
            verdict=str(view["id"]),
        )
        return view

    @app.delete("/api/explore/queries/{query_id}")
    async def delete_saved_query(
        request: Request, query_id: str, permanent: bool = False
    ) -> WireOk:
        """Move a saved query to trash, or erase it immediately with permanent=true."""
        service = _explore()
        deleted = service.delete_query(query_id, permanent=permanent)
        if not deleted:
            raise HTTPException(status_code=404, detail="query not found")
        return WireOk()

    @app.post("/api/explore/promote")
    async def promote_query(request: Request) -> dict[str, Any]:
        """Author a bound-dataset query as a certified derivation (opt-in, human).

        This is the one path from exploration into the governed world; it runs the same
        authoring loop the chat and notebooks use. It does not gate exploration itself.
        """
        service = _explore()
        body = await request.json()
        name = str(body.get("name") or "").strip()
        sql = str(body.get("sql") or "").strip()
        if not name or not sql:
            raise HTTPException(status_code=400, detail="a name and query are required")
        return await run_in_threadpool(
            lambda: service.promote(name, sql, source_id=body.get("source_id"))
        )

    @app.post("/api/explore/draft")
    async def explore_draft(request: Request) -> dict[str, Any]:
        """Draft a derivation from NL or SQL and return the verdict, certifying nothing.

        The first half of the one-click loop: the ask becomes a proposed derivation,
        verified under the sandbox, its verdict returned for review. Certifying is the
        follow-up ``/api/explore/certify`` call; loop latency is logged on both calls,
        correlated by ``flow_id``.
        """
        service = _explore()
        body = await request.json()
        name = str(body.get("name") or "").strip()
        sql = str(body.get("sql") or "").strip()
        prompt = str(body.get("prompt") or "").strip()
        source_id = body.get("source_id")
        flow_id = str(body.get("flow_id") or "").strip() or None
        if not name:
            raise HTTPException(status_code=400, detail="a derivation name is required")
        if not sql and not prompt:
            raise HTTPException(
                status_code=400, detail="a prompt or a query is required"
            )

        claim = body.get("claim") if isinstance(body.get("claim"), dict) else None
        started = time.monotonic()
        nl2sql_ms: float | None = None
        # A typed question wins over leftover editor SQL; one completion drafts the
        # SQL and extracts the claim so the oracle can rule on the draft.
        if prompt:
            try:
                catalog = await run_in_threadpool(service.catalog, source_id)
            except ElbiError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            source = store.get_data_source(source_id) if store and source_id else None
            dialect = source.kind if source is not None else "DuckDB"
            try:
                sql, claim = await run_in_threadpool(
                    draft_query,
                    prompt,
                    catalog["tables"],
                    _default_client(),
                    dialect=dialect,
                )
            except ElbiError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            nl2sql_ms = (time.monotonic() - started) * 1000
            if not sql:
                raise HTTPException(
                    status_code=400, detail="could not draft SQL for the question"
                )

        verify_started = time.monotonic()
        result = await run_in_threadpool(
            lambda: service.draft(name, sql, source_id=source_id, claim=claim)
        )
        now = time.monotonic()
        result["sql"] = sql
        result["claim"] = claim
        result["flow_id"] = flow_id
        logger.info(
            "explore.draft",
            extra={
                "flow_id": flow_id,
                "derivation": name,
                "nl2sql_ms": nl2sql_ms,
                "verify_ms": (now - verify_started) * 1000,
                "draft_ms": (now - started) * 1000,
                "verdict": result.get("verdict"),
                "claimed": claim is not None,
                "certifiable": result.get("ok"),
            },
        )
        return result

    @app.post("/api/explore/certify")
    async def explore_certify(request: Request) -> dict[str, Any]:
        """Certify a reviewed draft with one click -- the human-in-the-loop half.

        Runs the governed promote path: a cache-backed re-verify, then certify and
        persist only when verification passes, so a rejected draft can never serve.
        Latency is logged, correlated with the draft call by ``flow_id``.
        """
        service = _explore()
        body = await request.json()
        name = str(body.get("name") or "").strip()
        sql = str(body.get("sql") or "").strip()
        flow_id = str(body.get("flow_id") or "").strip() or None
        claim = body.get("claim") if isinstance(body.get("claim"), dict) else None
        if not name or not sql:
            raise HTTPException(status_code=400, detail="a name and query are required")
        started = time.monotonic()
        result = await run_in_threadpool(
            lambda: service.promote(
                name, sql, source_id=body.get("source_id"), claim=claim
            )
        )
        result["flow_id"] = flow_id
        logger.info(
            "explore.certify",
            extra={
                "flow_id": flow_id,
                "derivation": name,
                "certify_ms": (time.monotonic() - started) * 1000,
                "certified": result.get("certified"),
                "verdict": result.get("verdict"),
            },
        )
        return result

    # -- data warehouse (sync external sources into a Delta Lake lakehouse) -------
    def _warehouse() -> WarehouseService:
        """The warehouse service, or a 503 hint when the ``warehouse`` extra is off."""
        if warehouse_service is None:
            raise HTTPException(
                status_code=503,
                detail="the data warehouse requires the 'warehouse' extra "
                "(pip install 'elbi-app[warehouse]')",
            )
        return warehouse_service

    @app.get("/api/warehouse/catalog")
    async def warehouse_catalog() -> dict[str, Any]:
        """The connection wizard's catalog: connector form schemas + categories."""
        return _warehouse().catalog()

    @app.post("/api/warehouse/interest")
    async def warehouse_interest(request: Request) -> WireOk:
        """Record a notify-me for a coming-soon connector."""
        service = _warehouse()
        body = await request.json()
        source_type = str(body.get("source_type") or "").strip()
        if not source_type:
            raise HTTPException(status_code=400, detail="a source_type is required")
        service.register_interest(source_type)
        return WireOk()

    @app.get("/api/warehouse/sources")
    async def warehouse_sources() -> list[dict[str, Any]]:
        """Every configured source with its table counts and last-sync state."""
        return _warehouse().list_sources_view()

    @app.post("/api/warehouse/uploads")
    async def warehouse_upload_file(file: UploadFile = File(...)) -> dict[str, str]:
        """Store an uploaded CSV/Parquet file and return its warehouse path.

        The returned ``path`` is what the CSV/Parquet source's ``path`` field takes, so
        a user can upload a local file instead of typing a path or object-store URL.
        """
        from .warehouse.uploads import save_upload

        # UploadFile.file is a blocking spooled stream; copy it to disk in the
        # threadpool so a large upload neither blocks the event loop nor buffers whole.
        try:
            path = await run_in_threadpool(
                save_upload, file.filename or "upload", file.file
            )
        except WarehouseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()
        return {"path": path, "filename": file.filename or ""}

    @app.post("/api/warehouse/sources")
    async def warehouse_create_source(request: Request) -> dict[str, Any]:
        """Validate a connection, discover its tables, and persist the source.

        Creating a connection is the most privileged act in the product: it mints a new
        path to data that the rest of the permission model does not mediate. Gated like
        the equivalent elsewhere -- Unity Catalog reserves CREATE STORAGE CREDENTIAL and
        CREATE EXTERNAL LOCATION to metastore admins for the same reason.
        """
        service = _warehouse()
        body = await request.json()
        source_type = str(body.get("source_type") or "").strip()
        name = str(body.get("name") or "").strip()
        config = body.get("config") or {}
        if not source_type or not name:
            raise HTTPException(
                status_code=400, detail="a source_type and name are required"
            )
        frequency = str(body.get("sync_frequency") or "day")
        description = str(body.get("description") or "")
        # An omitted/blank prefix means "default to the source type" (None), not the
        # unprefixed sentinel ("") reserved for project-declared sources.
        prefix = str(body["prefix"]).strip() if body.get("prefix") else None
        try:
            source = await run_in_threadpool(
                lambda: service.create_source(
                    source_type,
                    name,
                    config,
                    sync_frequency=frequency,
                    description=description,
                    prefix=prefix,
                )
            )
        except WarehouseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return service.source_detail_view(source.id)

    @app.patch("/api/warehouse/sources/{source_id}")
    async def warehouse_update_source(
        source_id: str, request: Request
    ) -> dict[str, Any]:
        """Change a source's auto-sync cadence (e.g. daily, 6-hourly, manual)."""
        service = _warehouse()
        body = await request.json()
        frequency = body.get("sync_frequency")
        if frequency is None:
            raise HTTPException(status_code=400, detail="sync_frequency is required")
        try:
            return service.set_sync_frequency(source_id, str(frequency))
        except WarehouseError as exc:
            status = 404 if "not found" in str(exc).lower() else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.get("/api/warehouse/sources/{source_id}")
    async def warehouse_source_detail(source_id: str) -> dict[str, Any]:
        """A source with its schemas (the source detail page)."""
        try:
            return _warehouse().source_detail_view(source_id)
        except WarehouseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/warehouse/sources/{source_id}")
    async def warehouse_delete_source(source_id: str) -> WireOk:
        """Delete a source, its schemas, and the warehouse tables it produced."""
        service = _warehouse()
        if not await run_in_threadpool(service.delete_source, source_id):
            raise HTTPException(status_code=404, detail="source not found")
        return WireOk()

    @app.post("/api/warehouse/sources/{source_id}/test")
    async def warehouse_test_source(source_id: str) -> dict[str, Any]:
        """Re-run a source's connection test and report what it found.

        Runs off the main loop: a connection test fetches, and a source whose paginator
        loops would otherwise block every other request until its deadline expires.
        """
        service = _warehouse()
        try:
            ok, errors = await run_in_threadpool(service.test_source, source_id)
        except WarehouseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": ok, "errors": errors}

    @app.post("/api/warehouse/sources/{source_id}/sync")
    async def warehouse_sync_source(source_id: str) -> dict[str, Any]:
        """Sync every enabled table of a source into the Delta Lake warehouse."""
        service = _warehouse()
        try:
            outcomes = await run_in_threadpool(service.sync_source, source_id)
        except WarehouseError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "outcomes": [
                {"table": o.table, "rows": o.rows, "ok": o.ok, "error": o.error}
                for o in outcomes
            ],
            "source": service.source_detail_view(source_id),
        }

    @app.patch("/api/warehouse/schemas/{schema_id}")
    async def warehouse_update_schema(
        schema_id: str, request: Request
    ) -> dict[str, Any]:
        """Toggle a table on/off or change how it syncs (full-refresh/incremental)."""
        service = _warehouse()
        body = await request.json()
        try:
            return service.update_schema(
                schema_id,
                should_sync=body.get("should_sync"),
                sync_type=body.get("sync_type"),
                incremental_field=body.get("incremental_field"),
            )
        except WarehouseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/warehouse/tables")
    async def warehouse_tables(request: Request) -> list[dict[str, Any]]:
        """Synced warehouse tables the caller may at least discover.

        Tables they cannot read are listed too, carrying the level they hold: knowing
        the data exists is how someone knows to ask for it.
        """
        return await run_in_threadpool(lambda: _warehouse().tables())

    # -- metrics (semantic layer) ------------------------------------------------
    def _metrics() -> MetricService:
        """The metrics service, or a 503 when the store it needs is absent."""
        if metric_service is None:
            raise HTTPException(status_code=503, detail="metrics require a store")
        return metric_service

    @app.get("/api/metrics")
    async def list_metrics_defs(request: Request) -> list[dict[str, Any]]:
        """The caller's metrics, each with its source's certification state."""
        return _metrics().list_metrics()

    @app.post("/api/metrics")
    async def define_metric(request: Request) -> dict[str, Any]:
        """Define (or replace) a metric from its spec manifest."""
        service = _metrics()
        body = await request.json()
        try:
            return service.define(body)
        except MetricServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/metrics/osi")
    async def export_metrics_osi(request: Request) -> dict[str, Any]:
        """Export the caller's metrics as an OSI semantic-model document."""
        try:
            return _metrics().export_osi()
        except MetricServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/metrics/osi")
    async def import_metrics_osi(request: Request) -> dict[str, Any]:
        """Import metrics from an OSI semantic-model document."""
        service = _metrics()
        body = await request.json()
        document = body.get("document")
        if not isinstance(document, dict):
            raise HTTPException(status_code=400, detail="an OSI 'document' is required")
        # A record from /api/exports/... is the other product: a definition plus the
        # evidence behind it, never an interchange document. Name that here, or the
        # caller gets a jsonschema complaint about a `version` key they never omitted.
        if str(document.get("schema", "")).startswith("elbi.export/"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "that is an exported record, not an OSI document. Export OSI "
                    "(GET /api/metrics/osi) produces a file this route accepts."
                ),
            )
        try:
            imported = service.import_osi(document, source_for=body.get("source_for"))
        except MetricServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"imported": imported}

    @app.get("/api/metrics/overview")
    async def metrics_overview(request: Request) -> list[dict[str, Any]]:
        """Each metric's current value and small trend, for the catalog sparklines."""
        service = _metrics()
        return await run_in_threadpool(lambda: service.overview())

    @app.post("/api/metrics/ask")
    async def ask_metric(request: Request) -> dict[str, Any]:
        """Answer a natural-language question by resolving it to a metric query.

        The LLM never writes SQL: it picks a metric and its dimensions/grain/filters
        from the catalog, the choice is validated against that metric, and the
        deterministic resolver runs it, so a question either resolves to a verified
        number or is declined, never a plausible-but-wrong figure.
        """
        service = _metrics()
        body = await request.json()
        question = str(body.get("question") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="a 'question' is required")
        views = service.list_metrics()
        if not views:
            raise HTTPException(status_code=400, detail="no metrics are defined yet")
        try:
            plan = await run_in_threadpool(
                _resolve_metric_question, question, views, _default_client()
            )
            result = await run_in_threadpool(
                lambda: service.query(
                    plan["metric_name"],
                    group_by=plan["group_by"],
                    grain=plan["grain"],
                    filters=plan["filters"],
                )
            )
        except (MetricServiceError, MetricQuestionError, ElbiError) as exc:
            # ElbiError covers "no model configured"; the others are validation or
            # resolution failures: all are the caller's to act on, never a 500.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **result,
            "resolved_query": plan,
            "explanation": plan.get("explanation"),
        }

    @app.get("/api/metrics/{name}")
    async def get_metric_def(request: Request, name: str) -> dict[str, Any]:
        """A metric's definition and its source certification state."""
        view = _metrics().get(name)
        if view is None:
            raise HTTPException(status_code=404, detail="metric not found")
        return view

    @app.post("/api/metrics/{name}/duplicate")
    async def duplicate_metric(request: Request, name: str) -> dict[str, Any]:
        """Copy a metric definition into a new one owned by the caller.

        400 rather than 404 when the copy cannot be defined, most often because the
        source derivation is no longer certified.
        """
        service = _metrics()
        try:
            view = await run_in_threadpool(service.duplicate, name)
        except MetricServiceError as exc:
            status = 404 if "not found" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        _require_store().record_audit(
            "metric.duplicate",
            target_type="metric",
            target_id=name,
            verdict=str(view["name"]),
        )
        return view

    @app.delete("/api/metrics/{name}")
    async def delete_metric_def(
        request: Request, name: str, permanent: bool = False
    ) -> WireOk:
        """Move a metric to trash, or erase it immediately with ``permanent=true``."""
        deleted = _metrics().delete(name, permanent=permanent)
        if not deleted:
            raise HTTPException(status_code=404, detail="metric not found")
        return WireOk()

    @app.post("/api/metrics/{name}/compile")
    async def compile_metric_def(request: Request, name: str) -> dict[str, str]:
        """The SQL a metric query compiles to (the generated-SQL trust affordance)."""
        service = _metrics()
        body = snakeify(await request.json())  # verbatim route; snake the app body here
        try:
            sql = await run_in_threadpool(
                lambda: service.compile_sql(
                    name,
                    group_by=[str(g) for g in body.get("group_by", [])],
                    grain=body.get("grain"),
                    filters=list(body.get("filters", [])),
                )
            )
        except MetricServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"sql": sql}

    @app.get("/api/metrics/{name}/history")
    async def metric_history(request: Request, name: str) -> list[dict[str, Any]]:
        """A metric's definition change log, newest first."""
        return _metrics().history(name)

    @app.post("/api/metrics/{name}/query")
    async def query_metric_def(request: Request, name: str) -> dict[str, Any]:
        """Resolve a metric: group by dimensions, roll up to a grain, filter."""
        service = _metrics()
        # /api/metrics is a verbatim route (its define body is a camelCase manifest), so
        # this app-shaped query body is snake-cased here instead of by the middleware.
        body = snakeify(await request.json())
        try:
            return await run_in_threadpool(
                lambda: service.query(
                    name,
                    group_by=[str(g) for g in body.get("group_by", [])],
                    grain=body.get("grain"),
                    filters=list(body.get("filters", [])),
                )
            )
        except MetricServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- monitors (anomaly detection & alerting) ---------------------------------
    def _monitors() -> MonitorService:
        """The monitoring service, or a 503 when the store it needs is absent."""
        if monitor_service is None:
            raise HTTPException(status_code=503, detail="monitors require a store")
        return monitor_service

    @app.get("/api/monitors")
    async def list_monitors(request: Request) -> list[WireMonitor]:
        """Every monitor, each with its latest value and alert status."""
        return _monitors().list_monitors()

    @app.post("/api/monitors")
    async def create_monitor(request: Request) -> WireMonitor:
        """Create a monitor over a certified metric or derivation."""
        service = _monitors()
        body = await request.json()
        try:
            return service.create(
                name=str(body.get("name") or ""),
                target_kind=str(body.get("target_kind") or "metric"),
                target=str(body.get("target") or ""),
                config=body.get("config"),
                method=str(body.get("method") or "mad"),
                sensitivity=float(body.get("sensitivity", 3.0)),
                min_value=body.get("min_value"),
                max_value=body.get("max_value"),
                window=int(body.get("window", 30)),
                interval_hours=float(body.get("interval_hours", 1.0)),
            )
        except (MonitorError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/monitors/{monitor_id}")
    async def update_monitor(monitor_id: str, request: Request) -> WireMonitor:
        """Change a monitor's settings, keeping its snapshots and its incidents.

        What ``elbi sync`` uses for a monitor that already exists. Re-creating
        one would take its history with it, and those snapshots are the baseline the
        detector compares against, so changing a threshold would quietly reset it.
        """
        service = _monitors()
        body = await request.json()
        numeric = {
            "sensitivity": float,
            "window": int,
            "interval_hours": float,
        }
        fields: dict[str, Any] = {
            key: cast(body[key]) for key, cast in numeric.items() if key in body
        }
        for key in ("name", "target_kind", "target", "method"):
            if key in body:
                fields[key] = str(body[key] or "")
        if "config" in body:
            fields["config"] = body["config"] or {}
        try:
            view = service.update(
                monitor_id,
                min_value=body.get("min_value"),
                max_value=body.get("max_value"),
                **fields,
            )
        except (MonitorError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if view is None:
            raise HTTPException(status_code=404, detail="no such monitor")
        return view

    @app.get("/api/monitors/{monitor_id}")
    async def get_monitor(request: Request, monitor_id: str) -> WireMonitorHistory:
        """A monitor's history (snapshots) and incidents."""
        service = _monitors()
        if service.get(monitor_id) is None:
            raise HTTPException(status_code=404, detail="monitor not found")
        try:
            return service.history(monitor_id)
        except MonitorError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/monitors/{monitor_id}")
    async def delete_monitor(request: Request, monitor_id: str) -> WireOk:
        """Delete a monitor and its history."""
        if not _monitors().delete(monitor_id):
            raise HTTPException(status_code=404, detail="monitor not found")
        return WireOk()

    @app.post("/api/monitors/{monitor_id}/check")
    async def check_monitor(request: Request, monitor_id: str) -> dict[str, Any]:
        """Run a monitor now: snapshot, detect an anomaly, manage its incident."""
        service = _monitors()
        try:
            return await run_in_threadpool(lambda: service.check(monitor_id))
        except MonitorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- notebooks ---------------------------------------------------------------
    def _notebooks() -> NotebookService:
        """The notebook service, or a 503 when the store it needs is absent."""
        if notebook_service is None:
            raise HTTPException(status_code=503, detail="notebooks require a store")
        return notebook_service

    def _require_store() -> Store:
        """The store, or a 503 when it is absent (routes that write always need one)."""
        if store is None:
            raise HTTPException(status_code=503, detail="this action requires a store")
        return store

    @app.get("/api/notebooks")
    async def list_notebooks(request: Request) -> list[dict[str, Any]]:
        """The caller's notebooks, most-recently-updated first."""
        _notebooks()
        return store.list_notebooks() if store else []

    @app.post("/api/notebooks")
    async def create_notebook(request: Request) -> dict[str, Any]:
        """Create a notebook (one empty cell) in a folder and return its id."""
        _notebooks()
        body = await request.json()
        name = str(body.get("name") or "Untitled notebook").strip()
        folder_id = str(body["folder_id"]) if body.get("folder_id") else None
        _require_folder(folder_id)
        return {"id": _require_store().create_notebook(name, folder_id=folder_id)}

    def _require_folder(folder_id: str | None) -> None:
        """404 unless ``folder_id`` names a live folder (``None`` is the root)."""
        if folder_id and store is not None and store.get_folder(folder_id) is None:
            raise HTTPException(status_code=404, detail="folder not found")

    @app.get("/api/notebooks/folders")
    async def list_notebook_folders(request: Request) -> list[dict[str, Any]]:
        """The caller's folders as a flat list; the client rebuilds the tree."""
        _notebooks()
        return store.list_folders() if store else []

    @app.post("/api/notebooks/folders")
    async def create_notebook_folder(request: Request) -> dict[str, Any]:
        """Create a folder under ``parent_id`` (omit or null for the root)."""
        _notebooks()
        body = await request.json()
        name = str(body.get("name") or "New folder").strip() or "New folder"
        parent_id = str(body["parent_id"]) if body.get("parent_id") else None
        _require_folder(parent_id)
        folder_id = _require_store().create_folder(name, parent_id)
        return {"id": folder_id}

    @app.put("/api/notebooks/folders/{folder_id}")
    async def update_notebook_folder(request: Request, folder_id: str) -> WireOk:
        """Rename a folder and/or reparent it (a cycle-forming move is a 409)."""
        _notebooks()
        db = _require_store()
        _require_folder(folder_id)
        body = await request.json()
        if "name" in body:
            db.rename_folder(folder_id, str(body["name"]).strip())
        if "parent_id" in body:
            parent_id = str(body["parent_id"]) if body["parent_id"] else None
            _require_folder(parent_id)
            if not db.move_folder(folder_id, parent_id):
                raise HTTPException(
                    status_code=409,
                    detail="cannot move a folder into itself or a descendant",
                )
        return WireOk()

    @app.delete("/api/notebooks/folders/{folder_id}")
    async def delete_notebook_folder(
        request: Request,
        folder_id: str,
        recursive: bool = False,
        permanent: bool = False,
    ) -> WireOk:
        """Move a folder to trash, or erase it immediately with ``permanent=true``.

        ``recursive=true`` also takes its subtree with it; a non-empty folder
        trashed without it is a 409, so the caller can confirm before removing
        its contents. ``permanent`` ignores that check entirely: an explicit
        permanent-delete action already is the confirmation.
        """
        _notebooks()
        db = _require_store()
        _require_folder(folder_id)
        if permanent:
            db.erase_folder(folder_id)
            return WireOk()
        if not db.trash_folder(folder_id, recursive):
            raise HTTPException(status_code=409, detail="folder is not empty")
        return WireOk()

    @app.post("/api/notebooks/{notebook_id}/move")
    async def move_notebook(request: Request, notebook_id: str) -> WireOk:
        """Move a notebook into a folder (null ``folder_id`` == root)."""
        _notebooks()
        body = await request.json()
        folder_id = str(body["folder_id"]) if body.get("folder_id") else None
        _require_folder(folder_id)
        _require_store().move_notebook(notebook_id, folder_id)
        return WireOk()

    @app.get("/api/notebooks/{notebook_id}")
    async def get_notebook(request: Request, notebook_id: str) -> dict[str, Any]:
        """The full notebook: metadata, cells with outputs, and the dependency graph."""
        service = _notebooks()
        view = service.view(notebook_id)
        if view is None:
            raise HTTPException(status_code=404, detail="notebook not found")
        return view

    @app.put("/api/notebooks/{notebook_id}")
    async def update_notebook(request: Request, notebook_id: str) -> dict[str, Any]:
        """Update a notebook's name, environment, or reactive/schedule settings.

        A change to ``deps`` or ``base_env`` goes through the environment path, re-lock
        and restart, and returns the lock result; other fields are a plain update.
        """
        service = _notebooks()
        body = await request.json()
        fields: dict[str, Any] = {}
        if "name" in body:
            fields["name"] = str(body["name"]).strip() or "Untitled notebook"
        if "metadata" in body:
            fields["metadata_json"] = json.dumps(dict(body["metadata"]))
        if "schedule" in body:
            fields["schedule_json"] = (
                json.dumps(dict(body["schedule"])) if body["schedule"] else None
            )
        if "compute_profile" in body:
            requested = body["compute_profile"]
            # A separate right from editing the notebook. Editing changes what the
            # analysis says; this changes what it costs, and Databricks splits the two
            # the same way ("can restart" versus "can manage").
            # Checked here rather than at kernel start, so a name that does not exist
            # or that this caller may not use is refused while they are still looking at
            # the picker. Selecting a profile is an edit to the notebook, so anyone who
            # can edit it can change what it runs on.
            try:
                service.profiles.select(requested)
            except ComputeProfileError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            fields["compute_profile"] = requested
            # Audited, because anyone who can edit a notebook can change what it runs
            # on, and "who moved this onto the GPU tier" is a question with a cost
            # attached. The usage record says what ran; this says who decided it would.
            _require_store().record_audit(
                "notebook.compute_profile",
                target_type="notebook",
                target_id=notebook_id,
                verdict=str(requested) if requested else "default",
            )
        if fields:
            _require_store().update_notebook(notebook_id, **fields)
        if "deps" in body or "base_env" in body:
            return await run_in_threadpool(
                service.set_environment,
                notebook_id,
                deps=list(body["deps"]) if "deps" in body else None,
                base_env=body.get("base_env") if "base_env" in body else None,
            )
        return {"ok": True}

    @app.post("/api/notebooks/{notebook_id}/relock")
    async def relock_notebook(request: Request, notebook_id: str) -> dict[str, Any]:
        """Re-resolve the notebook's environment to a fresh pinned lock."""
        service = _notebooks()
        return await run_in_threadpool(service.relock, notebook_id)

    @app.get("/api/compute/notebooks/{notebook_id}")
    async def notebook_compute(request: Request, notebook_id: str) -> dict[str, Any]:
        """What the notebook runs on, and whether its kernel still matches that.

        Under ``/api/compute`` rather than ``/api/notebooks`` because that family passes
        its JSON through verbatim for nbformat's sake, and a compute payload has nothing
        to do with nbformat; leaving it there would make it the one snake_case response
        in the compute surface.
        """
        service = _notebooks()
        return service.runtime_state(notebook_id)

    @app.get("/api/compute/profiles")
    async def list_compute_profiles(request: Request) -> WireComputeProfiles:
        """The compute menu, as this caller may use it.

        Infrastructure defines this and the app does not edit it, so there is no PUT
        here: a field the UI offered but could not change would be worse than its
        absence. Profiles the caller may not select are omitted rather than shown
        disabled, since a restricted profile they cannot use is not information they
        can act on.
        """
        profiles = _notebooks().profiles
        rates = profiles.rates
        return WireComputeProfiles(
            default=profiles.default,
            max_cost_per_hour=profiles.max_cost_per_hour,
            profiles=[profile.presented(rates) for profile in profiles.profiles],
        )

    @app.get("/api/compute/usage")
    async def compute_usage(request: Request, days: int = 30) -> WireComputeUsage:
        """Compute attributed to (user, notebook, profile, duration) over a window.

        Reported, not enforced. Databricks is candid that its own spend limits for
        compute are notification-only and that they do "not proactively terminate
        resources to maintain the limit"; terminating someone's session to save the
        overage destroys work to recover cents.
        """
        store = _require_store()
        since = datetime.now(timezone.utc) - timedelta(days=max(1, days))
        rows = store.compute_usage(since=since)
        spent = store.compute_spend(since=since)
        limit = spend_limit()
        return WireComputeUsage(
            since=since.isoformat(),
            spend=round(spent, 4),
            # Split out because these mean different things: batch is work someone
            # scheduled, interactive is work someone is doing, and only the second is
            # worth chasing when it is idle.
            by_kind={
                kind: round(total, 4)
                for kind, total in store.compute_spend_by_kind(since=since).items()
            },
            limit=limit,
            alert=over_budget(spent, limit),
            sessions=[
                WireComputeSession(
                    ended_at=row.ended_at.isoformat(),
                    kind=row.kind,
                    notebook_id=row.notebook_id,
                    profile=row.profile,
                    profile_version=row.profile_version,
                    seconds=row.seconds,
                    cost=row.cost,
                )
                for row in rows
            ],
        )

    @app.get("/api/settings/notebook-environments")
    async def get_notebook_environments(all: bool = False) -> list[dict[str, Any]]:
        """The curated base environments notebooks build on.

        Only the supported ones by default, since that is what a notebook may newly
        choose; ``all=true`` includes those past end of support, which an admin needs to
        see in order to manage them.
        """
        service = _notebooks()
        return service.base_environments() if all else service.selectable_environments()

    @app.put("/api/settings/notebook-environments")
    async def set_notebook_environments(request: Request) -> WireOk:
        """Define the curated base environments (admin only); validates each package."""
        _notebooks()
        body = await request.json()
        environments = []
        for entry in body.get("environments", []):
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            deps = [str(d).strip() for d in entry.get("deps", []) if str(d).strip()]
            for dep in deps:
                if not _DEP_RE.match(dep):
                    raise HTTPException(
                        status_code=400, detail=f"invalid dependency {dep!r}"
                    )
            ends = str(entry.get("end_of_support") or "").strip()
            if ends:
                try:
                    date.fromisoformat(ends)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"end_of_support {ends!r} must be a date (YYYY-MM-DD); an "
                            "unparseable one would look like no support window at all"
                        ),
                    ) from exc
            environments.append(
                {
                    "name": name,
                    "deps": deps,
                    "version": str(entry.get("version") or "").strip(),
                    "end_of_support": ends or None,
                }
            )
        _require_store().set_config(BASE_ENV_SETTING, json.dumps(environments))
        return WireOk()

    @app.delete("/api/notebooks/{notebook_id}")
    async def delete_notebook(
        request: Request, notebook_id: str, permanent: bool = False
    ) -> WireOk:
        """Move a notebook to trash, or erase it immediately with ``permanent=true``.

        Either way the live kernel is stopped first: a trashed notebook should not
        keep running, and an erased one certainly should not outlive its row.
        """
        service = _notebooks()
        # _own_notebook rather than _manage_notebook: same MANAGE check, but it
        # answers 403 when the caller can see the notebook and merely may not delete
        # it, where _manage_notebook says 404 either way.
        service.restart(notebook_id)
        db = _require_store()
        if permanent:
            db.erase_notebook(notebook_id)
        else:
            db.trash_notebook(notebook_id)
        return WireOk()

    @app.post("/api/notebooks/{notebook_id}/duplicate")
    async def duplicate_notebook(request: Request, notebook_id: str) -> WireCreated:
        """Copy a notebook into a new one owned by the caller."""
        service = _notebooks()
        new_id = await run_in_threadpool(service.duplicate, notebook_id)
        if new_id is None:
            raise HTTPException(status_code=404, detail="notebook not found")
        _require_store().record_audit(
            "notebook.duplicate",
            target_type="notebook",
            target_id=notebook_id,
            verdict=new_id,
        )
        return WireCreated(id=new_id)

    @app.post("/api/notebooks/{notebook_id}/cells")
    async def add_cell(request: Request, notebook_id: str) -> dict[str, Any]:
        """Add a cell (code or markdown), optionally after a given cell."""
        _notebooks()
        body = await request.json()
        cell = _require_store().add_cell(
            notebook_id,
            cell_type=str(body.get("cell_type") or "code"),
            after=str(body.get("after")) if body.get("after") else None,
            source=str(body.get("source") or ""),
        )
        return {"id": cell.id}

    @app.put("/api/notebooks/{notebook_id}/cells/{cell_id}")
    async def update_cell(request: Request, notebook_id: str, cell_id: str) -> WireOk:
        """Update a cell's source, type, or metadata (e.g. tag it ``parameters``)."""
        _notebooks()
        body = await request.json()
        fields: dict[str, Any] = {}
        if "source" in body:
            fields["source"] = str(body["source"])
        if "cell_type" in body:
            fields["cell_type"] = str(body["cell_type"])
        if "metadata" in body:
            fields["metadata_json"] = json.dumps(dict(body["metadata"]))
        if fields:
            _require_store().update_cell(cell_id, **fields)
        return WireOk()

    @app.delete("/api/notebooks/{notebook_id}/cells/{cell_id}")
    async def delete_cell(request: Request, notebook_id: str, cell_id: str) -> WireOk:
        """Delete a cell from the notebook."""
        _notebooks()
        _require_store().delete_cell(cell_id)
        return WireOk()

    @app.post("/api/notebooks/{notebook_id}/cells/reorder")
    async def reorder_cells(request: Request, notebook_id: str) -> WireOk:
        """Set the notebook's cell order to the given list of cell ids."""
        _notebooks()
        body = await request.json()
        order = [str(c) for c in body.get("order", [])]
        _require_store().reorder_cells(notebook_id, order)
        return WireOk()

    @app.post("/api/notebooks/{notebook_id}/run")
    async def run_notebook(request: Request, notebook_id: str) -> EventSourceResponse:
        """Run cells and their dependents, streaming outputs to the client over SSE.

        The body names the roots to run (``cells``) or asks to run everything
        (``run_all``, optionally ``fresh`` to restart the kernel first). Execution is
        reactive: each root's transitive dependents run in dependency order.
        """
        service = _notebooks()
        body = await request.json()
        if isinstance(body.get("params"), dict) and body["params"]:
            events = service.run_parameterized_events(notebook_id, body["params"])
        elif body.get("run_all"):
            events = service.run_all_events(notebook_id, fresh=bool(body.get("fresh")))
        else:
            cells = [str(c) for c in body.get("cells", [])]
            events = service.run_events(notebook_id, cells)

        async def stream() -> AsyncIterator[dict[str, str]]:
            async for item in iterate_in_threadpool(events):
                yield {"data": json.dumps(item)}
            yield {"data": "[DONE]"}

        return EventSourceResponse(stream())

    @app.post("/api/notebooks/{notebook_id}/interrupt")
    async def interrupt_notebook(request: Request, notebook_id: str) -> WireOk:
        """Interrupt the notebook's currently running cell (namespace preserved)."""
        service = _notebooks()
        service.interrupt(notebook_id)
        return WireOk()

    @app.post("/api/notebooks/{notebook_id}/restart")
    async def restart_notebook(request: Request, notebook_id: str) -> WireOk:
        """Restart the kernel: a fresh interpreter with an empty namespace."""
        service = _notebooks()
        service.restart(notebook_id)
        return WireOk()

    @app.post("/api/notebooks/{notebook_id}/complete")
    async def complete_notebook(request: Request, notebook_id: str) -> dict[str, Any]:
        """Tab-completions for a cell's code at a cursor (from the live kernel)."""
        service = _notebooks()
        body = await request.json()
        return await run_in_threadpool(
            service.complete,
            notebook_id,
            str(body.get("code", "")),
            int(body.get("cursor_pos", 0)),
        )

    @app.get("/api/notebooks/{notebook_id}/variables")
    async def notebook_variables(
        request: Request, notebook_id: str
    ) -> list[dict[str, str]]:
        """The user's data variables in the live kernel (name, type, summary)."""
        service = _notebooks()
        return await run_in_threadpool(service.variables, notebook_id)

    @app.post("/api/notebooks/{notebook_id}/inspect")
    async def inspect_notebook(request: Request, notebook_id: str) -> dict[str, Any]:
        """Object help (signature + docstring) for the name under the cursor."""
        service = _notebooks()
        body = await request.json()
        return await run_in_threadpool(
            service.inspect,
            notebook_id,
            str(body.get("code", "")),
            int(body.get("cursor_pos", 0)),
            int(body.get("detail_level", 0)),
        )

    @app.post("/api/notebooks/{notebook_id}/input")
    async def input_notebook(request: Request, notebook_id: str) -> dict[str, bool]:
        """Deliver a reply to a cell's pending ``input()`` prompt."""
        service = _notebooks()
        body = await request.json()
        delivered = service.send_input(notebook_id, str(body.get("value", "")))
        return {"ok": delivered}

    @app.websocket("/api/notebooks/{notebook_id}/comm")
    async def notebook_comm(websocket: WebSocket, notebook_id: str) -> None:
        """Bidirectional ipywidgets comm channel between the browser and the kernel.

        The kernel's comm messages (widget state) are pushed to the socket; messages
        from the browser's widget manager (a slider drag) are forwarded to the kernel.
        The kernel emits comm traffic from its reader thread, so those messages cross
        onto the event loop through a thread-safe queue before being sent.
        """
        if notebook_service is None or store is None:
            await websocket.close(code=1011)
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def on_comm(message: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(outbound.put_nowait, message)

        unsubscribe = notebook_service.comm_subscribe(notebook_id, on_comm)

        async def pump() -> None:
            while True:
                await websocket.send_json(await outbound.get())

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                notebook_service.comm_send(notebook_id, await websocket.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            pump_task.cancel()
            unsubscribe()

    @app.get("/api/notebooks/{notebook_id}/export")
    async def export_notebook(
        request: Request, notebook_id: str, outputs: bool = False
    ) -> Response:
        """Download the notebook as a nbformat 4.5 ``.ipynb`` file.

        Outputs are stripped unless asked for *and* permitted. A cell's output is a
        rendering of warehouse rows, and this is the route ``elbi pull`` reads,
        so the default keeps data on the deployment rather than on a laptop.

        ``NOTEBOOK_EXPORT_OUTPUTS`` is what permits it, and it is deliberately a
        deployment setting rather than something the caller decides: whether data may
        leave is a question about the company, not about the person asking. Databricks
        draws the line in the same place, refusing to commit ``.ipynb`` output from a
        Git folder until a workspace administrator turns it on.
        """
        service = _notebooks()
        allowed = os.environ.get("NOTEBOOK_EXPORT_OUTPUTS", "").strip() == "1"
        payload = service.export_ipynb(notebook_id, include_outputs=outputs and allowed)
        if payload is None:
            raise HTTPException(status_code=404, detail="notebook not found")
        return Response(
            content=json.dumps(payload, indent=1),
            media_type="application/x-ipynb+json",
            headers={
                "Content-Disposition": f'attachment; filename="{notebook_id}.ipynb"'
            },
        )

    @app.post("/api/notebooks/{notebook_id}/cells/{cell_id}/promote")
    async def promote_cell(
        request: Request, notebook_id: str, cell_id: str
    ) -> dict[str, Any]:
        """Author a cell's ``def <name>(ctx)`` as a certified, governed derivation."""
        service = _notebooks()
        return await run_in_threadpool(service.promote_cell, notebook_id, cell_id)

    @app.post("/api/notebooks/from-derivation/{name}")
    async def notebook_from_derivation(request: Request, name: str) -> dict[str, Any]:
        """Open a stored derivation's source in a new notebook for iteration."""
        service = _notebooks()
        notebook_id = service.create_from_derivation(name)
        if notebook_id is None:
            raise HTTPException(status_code=404, detail="derivation not found")
        return {"id": notebook_id}

    @app.post("/api/notebooks/from-model/{name}")
    async def notebook_from_model(request: Request, name: str) -> dict[str, Any]:
        """Open a model version's training script in a notebook to re-run or adapt."""
        service = _notebooks()
        if model_service is None:
            raise HTTPException(status_code=503, detail="models require the ml extra")
        version = request.query_params.get("version")
        try:
            script = await run_in_threadpool(
                model_service.training_script, name, version
            )
        except ModelError as exc:
            # The same 404 this route already gives a known model with no script; an
            # unknown name is no more of a server fault than an unscripted one.
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if script is None:
            raise HTTPException(
                status_code=404, detail="that model version has no training script"
            )
        heading = (
            f"# Training `{name}`\n\nThe glass-box script that reproduces this model's "
            "winning run. Edit and run it to retrain."
        )
        notebook_id = await run_in_threadpool(
            lambda: service.create_from_source(
                f"Training {name}", script, heading=heading
            )
        )
        return {"id": notebook_id}

    @app.post("/api/notebooks/import")
    async def import_notebook(request: Request) -> dict[str, Any]:
        """Create a notebook from an uploaded ``.ipynb`` document."""
        service = _notebooks()
        body = await request.json()
        ipynb = body.get("ipynb")
        if not isinstance(ipynb, dict):
            raise HTTPException(status_code=400, detail="missing ipynb document")
        name = str(body.get("name") or "Imported notebook").strip()
        notebook_id = service.import_ipynb(ipynb, name=name)
        return {"id": notebook_id}

    @app.put("/api/notebooks/{notebook_id}/ipynb")
    async def replace_notebook_ipynb(
        notebook_id: str, request: Request
    ) -> dict[str, Any]:
        """Replace a notebook's contents from an ``.ipynb``, keeping its identity.

        What ``elbi sync`` uses for a notebook that already exists. Deleting it and
        importing the file as a new one produces the right cells but a new id, which
        every link, schedule and dashboard tile pointing at the old one then misses.
        """
        service = _notebooks()
        body = await request.json()
        ipynb = body.get("ipynb")
        if not isinstance(ipynb, dict):
            raise HTTPException(status_code=400, detail="missing ipynb document")
        if not service.replace_ipynb(notebook_id, ipynb):
            raise HTTPException(status_code=404, detail="notebook not found")
        name = body.get("name")
        if isinstance(name, str) and name.strip():
            # A rename travels with the contents, since a file's name is the notebook's
            # name in the repo: without this, renaming a file would push its cells and
            # leave the old name behind.
            store_or_404 = _require_store()
            store_or_404.update_notebook(notebook_id, name=name.strip())
        return {"id": notebook_id}

    # -- dashboards --------------------------------------------------------------
    def _dashboards() -> DashboardService:
        """The dashboard service, or a 503 when the project it needs is absent."""
        if dashboard_service is None:
            raise HTTPException(status_code=503, detail="dashboards require a project")
        return dashboard_service

    @app.get("/api/dashboards")
    async def list_dashboards(request: Request) -> list[dict[str, Any]]:
        """The caller's dashboards, most-recently-updated first."""
        service = _dashboards()
        return service.list_dashboards()

    @app.get("/api/dashboards/catalog")
    async def dashboard_catalog() -> list[dict[str, Any]]:
        """The certified derivations available to bind to a widget."""
        return _dashboards().catalog()

    @app.get("/api/dashboards/schema")
    async def dashboard_schema() -> dict[str, Any]:
        """The DashboardSpec JSON Schema, so an editor can check a spec as it is typed.

        The same document the server validates against, rather than a copy of its rules
        kept in the client: a rule that drifts would report an error the save accepts,
        or accept one it refuses.
        """
        from elbi_core.dashboard.spec import load_dashboard_schema

        return load_dashboard_schema()

    @app.post("/api/dashboards")
    async def create_dashboard(request: Request) -> dict[str, Any]:
        """Validate a dashboard manifest and store it as a new draft."""
        service = _dashboards()
        body = await request.json()
        try:
            return service.create(dict(body))
        except DashboardError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dashboards/{dashboard_id}")
    async def get_dashboard(request: Request, dashboard_id: str) -> dict[str, Any]:
        """A dashboard's full state: draft spec, published spec, status, version."""
        service = _dashboards()
        try:
            return service.get(dashboard_id)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/dashboards/{dashboard_id}")
    async def save_dashboard(request: Request, dashboard_id: str) -> dict[str, Any]:
        """Validate and replace a dashboard's draft spec, appending a version."""
        service = _dashboards()
        body = await request.json()
        spec = body.get("spec", body)
        try:
            return service.save(dashboard_id, dict(spec))
        except DashboardError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dashboards/{dashboard_id}/duplicate")
    async def duplicate_dashboard(
        request: Request, dashboard_id: str
    ) -> dict[str, Any]:
        """Copy a dashboard into a new unpublished draft owned by the caller."""
        service = _dashboards()
        try:
            view = service.duplicate(dashboard_id)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        _require_store().record_audit(
            "dashboard.duplicate",
            target_type="dashboard",
            target_id=dashboard_id,
            verdict=str(view["id"]),
        )
        return view

    @app.delete("/api/dashboards/{dashboard_id}")
    async def delete_dashboard(
        request: Request, dashboard_id: str, permanent: bool = False
    ) -> WireOk:
        """Move a dashboard to trash, or erase it immediately with permanent=true."""
        service = _dashboards()
        try:
            service.delete(dashboard_id, permanent=permanent)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return WireOk()

    @app.post("/api/dashboards/{dashboard_id}/publish")
    async def publish_dashboard(request: Request, dashboard_id: str) -> dict[str, Any]:
        """Publish a dashboard (human path), refusing only if a binding is missing."""
        service = _dashboards()
        try:
            result = service.publish(dashboard_id)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not result.ok:
            raise HTTPException(status_code=409, detail=result.detail)
        return {"ok": True}

    @app.get("/api/dashboards/{dashboard_id}/versions")
    async def dashboard_versions(
        request: Request, dashboard_id: str
    ) -> list[dict[str, Any]]:
        """The dashboard's saved revisions, newest first."""
        service = _dashboards()
        try:
            return service.versions(dashboard_id)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dashboards/{dashboard_id}/pages/{page}/data")
    async def resolve_dashboard_page(
        request: Request, dashboard_id: str, page: str
    ) -> dict[str, Any]:
        """Resolve a page's widgets against the viewer's variable selections.

        Runs each data-bound widget's derivation (through the project runner, so
        cached results are reused) and returns the rows plus any per-widget error.
        ``published`` reads the frozen published spec; otherwise the working draft.
        """
        service = _dashboards()
        body = await request.json()
        variables = dict(body.get("variables", {}))
        published = bool(body.get("published", False))
        try:
            widgets = await run_in_threadpool(
                service.resolve,
                dashboard_id,
                page,
                variables,
                published=published,
            )
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"widgets": widgets}

    @app.get("/api/dashboards/{dashboard_id}/variables/{variable}/options")
    async def dashboard_variable_options(
        request: Request, dashboard_id: str, variable: str
    ) -> dict[str, Any]:
        """The selectable options for a filter control (static or derivation-backed)."""
        service = _dashboards()
        try:
            options = await run_in_threadpool(service.options, dashboard_id, variable)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"options": options}

    @app.get("/api/dashboards/{dashboard_id}/subscriptions")
    async def list_dashboard_subscriptions(
        request: Request, dashboard_id: str
    ) -> list[dict[str, Any]]:
        """A dashboard's scheduled snapshot deliveries."""
        service = _dashboards()
        try:
            return service.subscriptions(dashboard_id)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/dashboards/{dashboard_id}/subscriptions")
    async def create_dashboard_subscription(
        request: Request, dashboard_id: str
    ) -> dict[str, Any]:
        """Register a scheduled snapshot delivery for a dashboard."""
        service = _dashboards()
        body = await request.json()
        try:
            subscription_id = service.subscribe(
                dashboard_id,
                page=str(body.get("page", "")),
                cron=str(body.get("cron", "")),
                recipients=list(body.get("recipients", [])),
                variable_state=dict(body.get("variables", {})),
                fmt=str(body.get("fmt", "png")),
                channel=str(body.get("channel", "email")),
            )
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"id": subscription_id}

    @app.delete("/api/dashboards/{dashboard_id}/subscriptions/{subscription_id}")
    async def delete_dashboard_subscription(
        request: Request, dashboard_id: str, subscription_id: str
    ) -> WireOk:
        """Remove a scheduled delivery."""
        service = _dashboards()
        try:
            service.unsubscribe(dashboard_id, subscription_id)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return WireOk()

    # -- feature store -----------------------------------------------------------
    def _features() -> FeatureStoreService:
        """The feature-store service, or a 503 when the project it needs is absent."""
        if feature_store_service is None:
            raise HTTPException(status_code=503, detail="feature store needs a project")
        return feature_store_service

    @app.get("/api/features/entities")
    async def list_feature_entities(request: Request) -> list[dict[str, Any]]:
        """The registered entities (join keys)."""
        service = _features()
        return service.list_entities()

    @app.post("/api/features/entities")
    async def define_feature_entity(request: Request) -> WireOk:
        """Register (or replace) an entity."""
        service = _features()
        body = await request.json()
        service.define_entity(
            name=str(_required(body, "name")),
            join_key=str(_required(body, "join_key")),
            value_type=str(body.get("value_type", "string")),
            description=body.get("description"),
        )
        return WireOk()

    @app.get("/api/features/views")
    async def list_feature_views(request: Request) -> list[dict[str, Any]]:
        """The registered feature views, for discovery."""
        service = _features()
        return service.catalog()

    @app.get("/api/features/views/{name}")
    async def get_feature_view(request: Request, name: str) -> dict[str, Any]:
        """A single feature view's registry entry."""
        service = _features()
        try:
            return service.get_feature_view(name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/features/views/{name}/detail")
    async def feature_view_detail(request: Request, name: str) -> dict[str, Any]:
        """A feature view's registry entry, freshness, and monitoring history."""
        service = _features()
        try:
            return service.view_detail(name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/features/views")
    async def define_feature_view(request: Request) -> dict[str, Any]:
        """Validate a feature-view manifest against the registry and persist it."""
        service = _features()
        body = await request.json()
        try:
            return service.define_feature_view(dict(body))
        except FeatureStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/features/views/{name}")
    async def delete_feature_view(
        request: Request, name: str, permanent: bool = False
    ) -> WireOk:
        """Move a feature view to trash, or erase it immediately with permanent=true."""
        service = _features()
        try:
            service.delete_feature_view(name, permanent=permanent)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return WireOk()

    @app.post("/api/features/materialize")
    async def materialize_features(request: Request) -> dict[str, Any]:
        """Refresh the online store with the latest value per entity for each view."""
        service = _features()
        body = await request.json()
        views = body.get("feature_views")
        try:
            written = await run_in_threadpool(
                service.materialize,
                feature_views=list(views) if views is not None else None,
            )
        except FeatureStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"written": written}

    @app.post("/api/features/online")
    async def online_features(request: Request) -> dict[str, Any]:
        """Read the latest materialized features for a batch of entity rows."""
        service = _features()
        body = await request.json()
        rows = service.get_online(
            list(body.get("entity_rows", [])),
            list(body.get("features", [])),
        )
        return {"rows": rows}

    @app.post("/api/features/historical")
    async def historical_features(request: Request) -> dict[str, Any]:
        """Point-in-time join of features onto an entity dataframe."""
        service = _features()
        body = await request.json()
        try:
            rows = await run_in_threadpool(
                service.get_historical,
                list(body.get("entity_df", [])),
                list(body.get("features", [])),
            )
        except FeatureStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"rows": rows}

    @app.get("/api/features/views/{name}/statistics")
    async def feature_statistics(request: Request, name: str) -> dict[str, Any]:
        """A feature view's profile snapshots, newest first."""
        service = _features()
        try:
            return {"snapshots": service.statistics(name)}
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/features/views/{name}/statistics")
    async def snapshot_feature_statistics(
        request: Request, name: str
    ) -> dict[str, Any]:
        """Profile a view's feature columns now, optionally as the drift baseline."""
        service = _features()
        body = await request.json()
        try:
            return await run_in_threadpool(
                service.snapshot_statistics,
                name,
                set_baseline=bool(body.get("set_baseline", False)),
            )
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/features/views/{name}/drift")
    async def feature_drift_history(request: Request, name: str) -> dict[str, Any]:
        """A feature view's recorded drift checks, newest first."""
        service = _features()
        try:
            return {"checks": service.drift_history(name)}
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/features/views/{name}/drift-check")
    async def run_feature_drift_check(request: Request, name: str) -> dict[str, Any]:
        """Compare a view's current values against its baseline; record the result."""
        service = _features()
        try:
            result = await run_in_threadpool(service.drift, name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if result.get("dataset_drift"):
            dispatch_event(
                store,
                "feature.drift_detected",
                {
                    "feature_view": name,
                    "share_drifted": result.get("share_drifted"),
                    "n_drifted": result.get("n_drifted"),
                },
            )
        return result

    @app.get("/api/features/views/{name}/contract")
    async def get_feature_contract(request: Request, name: str) -> dict[str, Any]:
        """A feature view's attached data contract, or ``{"contract": null}``."""
        service = _features()
        try:
            contract = service.get_contract(name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"contract": contract}

    @app.put("/api/features/views/{name}/contract")
    async def set_feature_contract(request: Request, name: str) -> dict[str, Any]:
        """Attach (or clear, with an empty manifest) a view's data contract."""
        service = _features()
        body = await request.json()
        try:
            stored = service.set_contract(name, body.get("contract"))
        except FeatureStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"contract": stored}

    @app.post("/api/features/views/{name}/contract/suggest")
    async def suggest_feature_contract(request: Request, name: str) -> dict[str, Any]:
        """Profile the view and propose a data contract to start from (not attached)."""
        service = _features()
        try:
            proposed = await run_in_threadpool(service.suggest_contract, name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"contract": proposed}

    @app.get("/api/features/views/{name}/expectations")
    async def feature_expectations_history(
        request: Request, name: str
    ) -> dict[str, Any]:
        """A feature view's recorded contract checks, newest first."""
        service = _features()
        try:
            return {"checks": service.expectations_history(name)}
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/features/views/{name}/expectations-check")
    async def run_feature_expectations(request: Request, name: str) -> dict[str, Any]:
        """Check a view's values against its data contract; record the result."""
        service = _features()
        try:
            result = await run_in_threadpool(service.verify_expectations, name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if result.get("verdict") == "unsound":
            dispatch_event(
                store,
                "feature.expectations_failed",
                {"feature_view": name, "n_violated": result.get("n_violated")},
            )
        return result

    @app.get("/api/features/training-sets")
    async def list_training_sets(request: Request) -> list[dict[str, Any]]:
        """The materialized training sets, newest first."""
        service = _features()
        return service.list_training_sets()

    @app.post("/api/features/training-sets")
    async def create_training_set(request: Request) -> dict[str, Any]:
        """Materialize a point-in-time join into a named, reusable training set."""
        service = _features()
        body = await request.json()
        try:
            return await run_in_threadpool(
                service.create_training_set,
                str(_required(body, "name")),
                list(body.get("features", [])),
                list(body.get("entity_df", [])),
                label=body.get("label"),
            )
        except FeatureStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/features/training-sets/{name}")
    async def get_training_set(request: Request, name: str) -> dict[str, Any]:
        """A training set's metadata and a small row sample."""
        service = _features()
        try:
            return service.get_training_set(name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/features/training-sets/{name}")
    async def delete_training_set(request: Request, name: str) -> WireOk:
        """Remove a materialized training set."""
        service = _features()
        try:
            service.delete_training_set(name)
        except FeatureStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return WireOk()

    # -- lineage & catalog -------------------------------------------------------
    def _lineage() -> LineageService:
        """The lineage service, or a 503 when the project it needs is absent."""
        if lineage_service is None:
            raise HTTPException(status_code=503, detail="lineage needs a project")
        return lineage_service

    @app.get("/api/catalog")
    async def catalog(request: Request, q: str = "") -> list[dict[str, Any]]:
        """Every artifact as a searchable catalog record."""
        service = _lineage()
        return await run_in_threadpool(service.catalog, q)

    @app.get("/api/lineage/graph")
    async def lineage_graph(request: Request) -> WireLineageGraph:
        """The whole cross-artifact lineage graph."""
        service = _lineage()
        return WireLineageGraph.model_validate(await run_in_threadpool(service.graph))

    @app.get("/api/lineage/subgraph")
    async def lineage_subgraph(request: Request, node: str) -> dict[str, Any]:
        """A node's neighborhood: its ancestors and descendants."""
        service = _lineage()
        return await run_in_threadpool(service.subgraph, node)

    @app.get("/api/lineage/impact")
    async def lineage_impact(request: Request, node: str) -> dict[str, Any]:
        """What a change to a node would affect, grouped by artifact type."""
        service = _lineage()
        try:
            return await run_in_threadpool(service.impact, node)
        except LineageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/lineage/provenance")
    async def lineage_provenance(request: Request, node: str) -> dict[str, Any]:
        """What feeds a node, grouped by artifact type (its upstream lineage).

        The mirror of ``/impact``, and what the share dialog asks before handing an
        artifact out: sharing a dashboard shares the numbers it renders, so the
        upstream it reads is worth seeing at the moment of the decision.
        """
        service = _lineage()
        try:
            return await run_in_threadpool(service.provenance, node)
        except LineageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # -- orchestration -----------------------------------------------------------
    def _orchestration() -> OrchestrationService:
        """The orchestration service, or a 503 when the project it needs is absent."""
        if orchestration_service is None:
            raise HTTPException(status_code=503, detail="orchestration needs a project")
        return orchestration_service

    def _start_run(
        service: OrchestrationService,
        *,
        cause: str,
        label: str,
        run: Callable[[str], Any],
        parent_run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        """Open a run and drive ``run(run_id)`` on the job runner, returning at once.

        The run row and its per-asset steps are written to the store as the loop
        progresses, so the client polls the run-detail endpoint for live progress. With
        no job runner (e.g. a test), it runs inline and returns the finished run.
        """
        run_id = service.open_run(
            cause=cause,
            parent_run_id=parent_run_id,
            workflow_id=workflow_id,
        )

        def work(_progress: Callable[[str], None], _cancel: Callable[[], bool]) -> None:
            # Cancellation flows through the service's own flag (the /cancel route),
            # which materialize checks per asset; the job's args are unused here.
            try:
                run(run_id)
            except Exception:  # never leave the run stuck 'running'
                service.finish_failed(run_id)
                raise

        if job_runner is not None:
            job_runner.submit(f"{cause}:{run_id}", label, work)
            return {"run_id": run_id, "status": "running"}
        work(lambda _: None, lambda: False)  # no job runner: run inline
        return {"run_id": run_id, "status": "done"}

    def _start_materialization(
        service: OrchestrationService,
        *,
        cause: str,
        parent_run_id: str | None = None,
        **materialize_args: Any,
    ) -> dict[str, Any]:
        """Open and drive a materialization run on the job runner (see _start_run)."""
        return _start_run(
            service,
            cause=cause,
            label="materialize assets",
            parent_run_id=parent_run_id,
            run=lambda run_id: service.materialize(run_id=run_id, **materialize_args),
        )

    @app.get("/api/orchestration/status")
    async def orchestration_status(request: Request) -> list[dict[str, Any]]:
        """Every asset's freshness (materialized / stale / never) and verdict."""
        service = _orchestration()
        return await run_in_threadpool(service.status)

    @app.get("/api/orchestration/graph")
    async def orchestration_graph(request: Request) -> dict[str, Any]:
        """The asset dependency DAG: nodes (freshness + verdict) and edges."""
        service = _orchestration()
        return await run_in_threadpool(service.graph)

    @app.get("/api/orchestration/history")
    async def orchestration_history(request: Request) -> dict[str, Any]:
        """The run-history matrix: assets and recent runs with per-asset cells."""
        service = _orchestration()
        return await run_in_threadpool(service.history)

    @app.get("/api/orchestration/settings")
    async def get_orchestration_settings() -> dict[str, Any]:
        """Orchestration policy: per-asset retry count and the compute backend."""
        service = _orchestration()
        return {"max_retries": service.retry_policy(), "compute": service.compute()}

    @app.put("/api/orchestration/settings")
    async def put_orchestration_settings(request: Request) -> dict[str, int]:
        """Update the per-asset retry count."""
        service = _orchestration()
        body = await request.json()
        service.set_retry_policy(int(_required(body, "max_retries")))
        return {"max_retries": service.retry_policy()}

    @app.get("/api/orchestration/checks")
    async def list_asset_checks(request: Request) -> list[dict[str, Any]]:
        """Data-quality checks, optionally filtered by ``?asset=``."""
        service = _orchestration()
        asset = request.query_params.get("asset")
        return await run_in_threadpool(service.checks, asset)

    @app.post("/api/orchestration/checks")
    async def upsert_asset_check(request: Request) -> WireCreated:
        """Create or update a data-quality check on an asset (a boolean SQL expr)."""
        service = _orchestration()
        body = await request.json()
        check_id = service.upsert_check(
            check_id=body.get("id"),
            asset=str(_required(body, "asset")),
            name=str(_required(body, "name")),
            expr=str(_required(body, "expr")),
            severity=str(body.get("severity", "warn")),
            enabled=bool(body.get("enabled", True)),
        )
        return WireCreated(id=check_id)

    @app.delete("/api/orchestration/checks/{check_id}")
    async def delete_asset_check(request: Request, check_id: str) -> WireOk:
        """Remove a data-quality check."""
        if not _orchestration().delete_check(check_id):
            raise HTTPException(status_code=404, detail="check not found")
        return WireOk()

    @app.post("/api/orchestration/materialize")
    async def orchestration_materialize(request: Request) -> dict[str, Any]:
        """Materialize a selection of assets (async); returns the run to poll."""
        service = _orchestration()
        body = await request.json()
        assets = body.get("assets")
        return await run_in_threadpool(
            lambda: _start_materialization(
                service,
                cause="manual",
                selection=str(body.get("selection", "stale")),
                assets=list(assets) if assets else None,
                include_downstream=bool(body.get("include_downstream", False)),
            )
        )

    @app.post("/api/orchestration/backfill")
    async def orchestration_backfill(request: Request) -> dict[str, Any]:
        """Backfill a parameterized asset across a set of partition values (async)."""
        service = _orchestration()
        body = await request.json()
        asset = str(_required(body, "asset"))
        param = str(_required(body, "param"))
        values = list(_required(body, "values"))
        return await run_in_threadpool(
            lambda: _start_run(
                service,
                cause="backfill",
                label=f"backfill {asset}",
                run=lambda run_id: service.backfill(
                    run_id=run_id,
                    asset=asset,
                    param=param,
                    values=values,
                ),
            )
        )

    @app.get("/api/orchestration/runs")
    async def orchestration_runs(request: Request) -> list[dict[str, Any]]:
        """Run history, newest first."""
        service = _orchestration()
        return service.runs()

    @app.get("/api/orchestration/runs/{run_id}")
    async def orchestration_run(request: Request, run_id: str) -> dict[str, Any]:
        """A run with its per-asset steps (poll this for live progress)."""
        service = _orchestration()
        result = service.run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="run not found")
        return result

    @app.post("/api/orchestration/runs/{run_id}/retry")
    async def orchestration_retry(request: Request, run_id: str) -> dict[str, Any]:
        """Retry a run from failure: re-materialize its failed assets + downstream."""
        service = _orchestration()
        failed = await run_in_threadpool(service.failed_assets, run_id)
        if not failed:
            raise HTTPException(
                status_code=400, detail="run has no failed assets to retry"
            )
        return await run_in_threadpool(
            lambda: _start_materialization(
                service,
                cause="retry",
                parent_run_id=run_id,
                assets=failed,
                include_downstream=True,
            )
        )

    @app.post("/api/orchestration/runs/{run_id}/cancel")
    async def orchestration_cancel(request: Request, run_id: str) -> WireOk:
        """Ask a running materialization to stop at the next asset boundary."""
        service = _orchestration()
        if service.run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        service.request_cancel(run_id)
        return WireOk()

    @app.get("/api/orchestration/workflows")
    async def list_workflows(request: Request) -> list[dict[str, Any]]:
        """Workflow definitions (a DAG of run-if-gated materialization steps)."""
        service = _orchestration()
        return service.workflows()

    @app.post("/api/orchestration/workflows")
    async def upsert_workflow(request: Request) -> WireCreated:
        """Create or replace a workflow."""
        service = _orchestration()
        body = await request.json()
        workflow_id = service.upsert_workflow(
            workflow_id=body.get("id"),
            name=str(_required(body, "name")),
            steps=list(body.get("steps", [])),
        )
        return WireCreated(id=workflow_id)

    @app.delete("/api/orchestration/workflows/{workflow_id}")
    async def delete_workflow(request: Request, workflow_id: str) -> WireOk:
        """Remove a workflow."""
        if not _orchestration().delete_workflow(workflow_id):
            raise HTTPException(status_code=404, detail="workflow not found")
        return WireOk()

    @app.post("/api/orchestration/workflows/{workflow_id}/run")
    async def run_workflow(request: Request, workflow_id: str) -> dict[str, Any]:
        """Run a workflow (async); returns the run to poll."""
        service = _orchestration()
        if service.get_workflow(workflow_id) is None:
            raise HTTPException(status_code=404, detail="workflow not found")
        return await run_in_threadpool(
            lambda: _start_run(
                service,
                cause="workflow",
                label=f"workflow {workflow_id}",
                workflow_id=workflow_id,
                run=lambda run_id: service.run_workflow(workflow_id, run_id=run_id),
            )
        )

    @app.get("/api/orchestration/schedules")
    async def list_schedules(request: Request) -> list[dict[str, Any]]:
        """The materialization schedules."""
        service = _orchestration()
        return service.schedules()

    @app.post("/api/orchestration/schedules")
    async def create_schedule(request: Request) -> WireCreated:
        """Create or replace a materialization schedule (cron or data-change sensor)."""
        service = _orchestration()
        body = await request.json()
        schedule_id = service.upsert_schedule(
            name=str(_required(body, "name")),
            selection=str(body.get("selection", "stale")),
            mode=str(body.get("mode", "cron")),
            cron=str(body.get("cron", "")),
            dataset=body.get("dataset"),
        )
        return WireCreated(id=schedule_id)

    @app.delete("/api/orchestration/schedules/{schedule_id}")
    async def delete_schedule(request: Request, schedule_id: str) -> WireOk:
        """Remove a schedule."""
        if not _orchestration().delete_schedule(schedule_id):
            raise HTTPException(status_code=404, detail="schedule not found")
        return WireOk()

    @app.get("/api/settings/llm/profiles")
    async def list_llm_profiles() -> dict[str, Any]:
        """The saved LLM profiles (keys masked) and which one is the default.

        Each profile is a named model config; a conversation runs on one of them. The
        api_key value is never returned, only whether one is set, the way a settings
        surface masks a secret. ``configured`` is a live check of the same
        precedence chat runs on (chosen profile, else default profile, else
        LLM_MODEL/--model) -- true whenever a message would actually go through,
        so the UI can point an unconfigured install at Settings before the first
        failed send, not just after it.
        """
        try:
            _default_client()
            configured = True
        except ElbiError:
            configured = False
        if store is None:
            return {"profiles": [], "default": "", "configured": configured}
        return {
            "profiles": [_profile_view(p) for p in store.list_profiles()],
            "default": store.get_config("llm.default_profile") or "",
            # Profile used for the cheap title call ("" = use the default profile).
            "title_profile": store.get_config("llm.title_profile") or "",
            "configured": configured,
        }

    @app.put("/api/settings/llm/profiles/{name}")
    async def save_llm_profile(name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Create or update a named LLM profile (a provider/model, key, base URL).

        Omitting ``api_key`` keeps the stored one (so editing other fields does not
        require re-entering the secret); an explicit empty string clears it. The first
        profile saved becomes the default. Names are 1-64 chars starting alphanumeric.
        """
        if store is None:
            raise HTTPException(status_code=404, detail="settings are not enabled")
        name = name.strip()
        if not _PROFILE_NAME_RE.match(name):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid profile name: use 1-64 characters (letters, digits, "
                    "spaces, and . _ -), starting with a letter or digit."
                ),
            )
        existing = store.get_profile(name)
        if existing is None and len(store.list_profiles()) >= _MAX_PROFILES:
            raise HTTPException(
                status_code=400, detail=f"at most {_MAX_PROFILES} profiles"
            )
        api_key = (
            str(body["api_key"])
            if "api_key" in body
            else (existing.api_key if existing else "")
        )
        model = str(body.get("model", existing.model if existing else "")).strip()
        base_url = str(
            body.get("base_url", existing.base_url if existing else "")
        ).strip()
        effort = str(
            body.get("reasoning_effort", existing.reasoning_effort if existing else "")
        ).strip()
        if effort and effort not in _reasoning_efforts():
            raise HTTPException(
                status_code=400,
                detail=(
                    "reasoning_effort must be one of "
                    f"{', '.join(sorted(_reasoning_efforts()))}, or empty to "
                    "leave it to the model's default."
                ),
            )
        key = api_key.strip()
        store.save_profile(
            LlmProfile(
                name=name,
                model=model,
                api_key=key,
                base_url=base_url,
                reasoning_effort=effort,
            )
        )
        if not store.get_config("llm.default_profile"):
            store.set_config("llm.default_profile", name)
        # The same view the listing returns, so the two endpoints cannot describe a
        # profile differently -- that divergence is how the resolved provider came to be
        # present on one and missing from the other. Built from a second, unpersisted
        # instance rather than the one just saved: that row is detached after the commit
        # and reading an attribute off it raises.
        return _profile_view(
            LlmProfile(
                name=name,
                model=model,
                api_key=key,
                base_url=base_url,
                reasoning_effort=effort,
            )
        )

    @app.delete("/api/settings/llm/profiles/{name}")
    async def delete_llm_profile(name: str) -> WireOk:
        """Delete a profile; if it was the default, promote another (or clear it)."""
        if store is None:
            raise HTTPException(status_code=404, detail="settings are not enabled")
        if not store.delete_profile(name):
            raise HTTPException(status_code=404, detail=f"no profile {name!r}")
        if store.get_config("llm.default_profile") == name:
            remaining = store.list_profiles()
            store.set_config(
                "llm.default_profile", remaining[0].name if remaining else ""
            )
        return WireOk()

    @app.put("/api/settings/llm/default")
    async def set_default_profile(body: dict[str, Any]) -> dict[str, str]:
        """Set the default profile (used by conversations that do not choose one)."""
        if store is None:
            raise HTTPException(status_code=404, detail="settings are not enabled")
        name = str(body.get("name") or "").strip()
        if name and store.get_profile(name) is None:
            raise HTTPException(status_code=404, detail=f"no profile {name!r}")
        store.set_config("llm.default_profile", name)
        return {"default": name}

    @app.put("/api/settings/llm/title-profile")
    async def set_title_profile(body: dict[str, Any]) -> dict[str, str]:
        """Set the profile titles run on (a cheap model); "" uses the default."""
        if store is None:
            raise HTTPException(status_code=404, detail="settings are not enabled")
        name = str(body.get("name") or "").strip()
        if name and store.get_profile(name) is None:
            raise HTTPException(status_code=404, detail=f"no profile {name!r}")
        store.set_config("llm.title_profile", name)
        return {"title_profile": name}

    @app.get("/api/audit")
    async def get_audit(request: Request, limit: int = 100) -> list[WireAuditEvent]:
        """The audit trail: what ran, against what, and whether it was sound."""
        if store is None:
            return []
        return [
            WireAuditEvent(
                id=e.id,
                at=_iso_utc(e.at),
                action=e.action,
                target_type=e.target_type,
                target_id=e.target_id,
                verdict=e.verdict,
                data_hash=e.data_hash,
            )
            for e in store.list_audit(min(limit, 500))
        ]

    # -- trash --------------------------------------------------------------------
    @app.get("/api/trash")
    async def list_trash(request: Request) -> list[dict[str, Any]]:
        """Trashed artifacts the caller may restore: own, or everyone's for an admin."""
        db = _require_store()
        return db.list_trash()

    @app.post("/api/trash/{kind}/{item_id}/restore")
    async def restore_trash_item(request: Request, kind: str, item_id: str) -> WireOk:
        """Restore a trashed artifact by kind and id."""
        db = _require_store()
        if kind == "derivation":
            # Not in Store's generic dispatch: restoring one needs the served
            # app's runtime hooks (live registry, sidecar), which db.py cannot
            # reach on its own.
            if derivation_on_restore is None:
                raise HTTPException(status_code=404, detail="not found in trash")
            return WireOk()
        try:
            restored = db.restore_trashed(kind, item_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not restored:
            raise HTTPException(status_code=404, detail="not found in trash")
        return WireOk()

    @app.delete("/api/trash/{kind}/{item_id}")
    async def erase_trash_item(request: Request, kind: str, item_id: str) -> WireOk:
        """Permanently erase a trashed artifact by kind and id."""
        db = _require_store()
        if kind == "derivation":
            if derivation_on_erase is None:
                raise HTTPException(status_code=404, detail="not found in trash")
            return WireOk()
        try:
            erased = db.erase_trashed(kind, item_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not erased:
            raise HTTPException(status_code=404, detail="not found in trash")
        return WireOk()

    # -- notifications -----------------------------------------------------------
    @app.get("/api/notifications")
    async def get_notifications(
        request: Request, limit: int = 50, unread_only: bool = False
    ) -> WireNotificationList:
        """Notifications, newest first, with the unread count.

        The count rides on the list so the inbox indicator and the page share one poll,
        and both come from one read so they cannot disagree.
        """
        if store is None:
            return WireNotificationList(items=[], unread=0)
        rows, unread = store.notification_inbox(
            # Clamp both ways: a negative LIMIT is unbounded on SQLite.
            limit=max(1, min(limit, 500)),
            unread_only=unread_only,
        )
        items = [
            WireNotificationItem(
                id=n.id,
                at=_iso_utc(n.at),
                event_type=n.event_type,
                title=n.title,
                body=n.body,
                target_type=n.target_type,
                target_id=n.target_id,
                verdict=n.verdict,
                read_at=_iso_utc(n.read_at) if n.read_at else None,
            )
            for n in rows
        ]
        return WireNotificationList(items=items, unread=unread)

    @app.post("/api/notifications/read-all")
    async def read_all_notifications(request: Request) -> dict[str, Any]:
        """Mark every unread notification of the caller read."""
        if store is None:
            return {"ok": True, "count": 0}
        count = store.mark_all_notifications_read()
        return {"ok": True, "count": count}

    @app.post("/api/notifications/{notification_id}/read")
    async def read_notification(
        request: Request, notification_id: str
    ) -> dict[str, Any]:
        """Mark one notification read; 404 for missing and not-yours alike."""
        if store is None or not store.mark_notification_read(notification_id):
            raise HTTPException(status_code=404, detail="no such notification")
        return {"ok": True}

    @app.get("/api/notifications/preferences")
    async def get_notification_preferences(
        request: Request,
    ) -> WireNotificationPreferences:
        """The caller's full per-event-type matrix (stored rows over defaults).

        The complete matrix is returned (not just deviations) so the client never
        hardcodes defaults; ``email_available`` tells it whether the email column
        is actionable at all on this deployment.
        """
        if store is None:
            prefs = {t: (True, e) for t, e in EVENT_TYPES.items()}
        else:
            prefs = effective_prefs(store)
        return WireNotificationPreferences(
            email_available=mailer.is_configured(),
            prefs=[
                WireNotificationPref(event_type=t, in_app=in_app, email=email)
                for t, (in_app, email) in prefs.items()
            ],
        )

    @app.put("/api/notifications/preferences")
    async def set_notification_preferences(
        body: dict[str, Any], request: Request
    ) -> WireNotificationPreferences:
        """Store the per-event-type switches; unknown types are a 400."""
        if store is None:
            raise HTTPException(
                status_code=404, detail="notification preferences need a store"
            )
        updates: dict[str, tuple[bool, bool]] = {}
        for item in body.get("prefs") or []:
            if not isinstance(item, dict):
                continue
            event_type = str(item.get("event_type") or "")
            if event_type not in EVENT_TYPES:
                raise HTTPException(
                    status_code=400, detail=f"unknown event type {event_type!r}"
                )
            updates[event_type] = (
                bool(item.get("in_app", True)),
                bool(item.get("email", False)),
            )
        if updates:
            store.set_notification_prefs(updates)
        prefs = effective_prefs(store)
        return WireNotificationPreferences(
            email_available=mailer.is_configured(),
            prefs=[
                WireNotificationPref(event_type=t, in_app=in_app, email=email)
                for t, (in_app, email) in prefs.items()
            ],
        )

    # -- API keys (hashed) ------------------------------------------------------

    # -- secrets vault ----------------------------------------------------------
    @app.get("/api/secrets")
    async def list_secrets(request: Request) -> list[dict[str, Any]]:
        """The caller's secrets, masked (name + description, never the value)."""
        if store is None:
            return []
        return [
            {"name": s.name, "description": s.description} for s in store.list_secrets()
        ]

    @app.post("/api/secrets")
    async def save_secret(body: dict[str, Any], request: Request) -> dict[str, str]:
        """Store a secret, encrypted at rest (needs ``APP_SECRET_KEY``)."""
        if store is None:
            raise HTTPException(status_code=404, detail="secrets are not enabled")
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not crypto.is_configured():
            raise HTTPException(
                status_code=400, detail=f"{crypto.SECRET_ENV} must be set"
            )
        store.save_secret(
            Secret(
                name=name,
                value=crypto.encrypt(str(body.get("value") or "")),
                description=str(body.get("description") or ""),
            )
        )
        return {"name": name}

    @app.delete("/api/secrets/{name}")
    async def delete_secret(name: str, request: Request) -> WireOk:
        if store is None or not store.delete_secret(name):
            raise HTTPException(status_code=404, detail="no such secret")
        return WireOk()

    @app.get("/api/settings/budget")
    async def get_budget(request: Request) -> WireBudget:
        """The spend cap (USD), its window, and the spend so far."""
        empty = WireBudget(max_budget=0.0, window="30d", spend=0.0)
        if store is None:
            return empty
        budget = store.get_budget()
        if budget is None:
            return empty
        return WireBudget(
            max_budget=budget.max_budget,
            window=budget.window,
            spend=budget.spend,
        )

    @app.put("/api/settings/budget")
    async def put_budget(body: dict[str, Any]) -> dict[str, Any]:
        """Set the spend cap (USD) and window (``30d``), resetting the window."""
        if store is None:
            raise HTTPException(status_code=404, detail="budget is not enabled")
        max_budget = float(body.get("max_budget") or 0.0)
        window = str(body.get("window") or "30d").strip() or "30d"
        store.set_budget(max_budget, window)
        return {"max_budget": max_budget, "window": window, "spend": 0.0}

    @app.get("/api/settings/runtime")
    async def get_runtime_settings() -> dict[str, Any]:
        """Operational settings, their effective values, and where each comes from.

        ``source`` is ``env``, ``setting``, or ``default``; ``editable`` is false when
        the environment pins the value, so a UI can show it as managed rather than
        offering a field whose edits would not take effect.
        """
        return {
            "settings": [
                {
                    "key": s.key,
                    "env_var": s.env_var,
                    "value": s.value,
                    "source": s.source,
                    "editable": s.editable,
                    "description": runtime_settings.describe(s.key),
                }
                for s in runtime_settings.all_settings(store)
            ]
        }

    @app.put("/api/settings/runtime/{key}")
    async def put_runtime_setting(key: str, body: dict[str, Any]) -> dict[str, Any]:
        """Store a value for one operational setting.

        Accepted even when the environment currently pins the setting (the stored value
        is kept and applies if the variable is later removed), but the response still
        reports the resolved source, so a caller can see it is not yet in force.
        """
        if store is None:
            raise HTTPException(status_code=404, detail="settings are not enabled")
        if not runtime_settings.is_known(key):
            raise HTTPException(status_code=404, detail=f"unknown setting {key!r}")
        store.set_config(key, str(body.get("value") or "").strip())
        resolved = runtime_settings.get(key, store)
        return {
            "key": resolved.key,
            "value": resolved.value,
            "source": resolved.source,
            "editable": resolved.editable,
        }

    @app.get("/api/settings/mlflow")
    async def get_mlflow() -> WireMlflowSettings:
        """The MLflow tracking URI certified runs export to, and where it is set.

        The ``MLFLOW_TRACKING_URI`` environment variable overrides the stored setting;
        ``source`` reports which is in effect so an operator can see where exports go.
        Resolved through the shared operational-settings path, so it behaves like every
        other setting on ``/api/settings/runtime``.
        """
        resolved = runtime_settings.get("mlflow_tracking_uri", store)
        return WireMlflowSettings(
            tracking_uri=resolved.value,
            # "default" is reported as "none" here: this setting's default is the empty
            # string, and the original contract for this endpoint said "none" for unset.
            source="none" if resolved.source == "default" else resolved.source,
            editable=resolved.editable,
        )

    @app.put("/api/settings/mlflow")
    async def put_mlflow(body: dict[str, Any]) -> dict[str, Any]:
        """Set or clear the MLflow tracking URI certified runs export to.

        An empty value disables the export; certified runs are still recorded in the
        app's own history and comparison either way. The environment variable, when set,
        still overrides this at export time.
        """
        if store is None:
            raise HTTPException(status_code=404, detail="settings are not enabled")
        uri = str(body.get("tracking_uri") or "").strip()
        store.set_config("mlflow_tracking_uri", uri)
        return {"tracking_uri": uri}

    # -- models (registry + MLflow-protocol scoring) ------------------------------
    def _model_service() -> ModelService:
        """The model service, or the 503 that names the extra to install."""
        if model_service is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "model registry/serving needs the ml extra: "
                    "pip install 'elbi-app[ml]'"
                ),
            )
        return model_service

    @app.get("/api/registry/models")
    async def list_registered_models() -> list[dict[str, Any]]:
        """Every registered model: latest version, champion, and tags."""
        return _model_service().models_view()

    @app.get("/api/registry/feature-sources")
    async def feature_sources() -> list[dict[str, str]]:
        """Everything a model can train on: bound datasets and certified derivations."""
        return _model_service().feature_sources()

    @app.get("/api/datasets/{name}/columns")
    async def dataset_columns(name: str) -> list[dict[str, Any]]:
        """A dataset's columns with a numeric hint, for the train-model form."""
        try:
            return _model_service().columns_view(name)
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/registry/train")
    async def train_registered_model(body: dict[str, Any], request: Request) -> WireJob:
        """Launch an AutoML training run from the UI, as a background job.

        The request is validated up front (a bad name or dataset is a 400, not a
        failed job); the search itself runs on the job runner so the UI can watch
        it in the jobs bar and refetch the registry when it lands. Identical
        requests dedupe onto the live job (a double-click trains once), but the
        key carries the model's current latest version, so training the same spec
        again after it finishes produces the next version rather than replaying
        the old result.
        """
        service = _model_service()
        name = str(body.get("name") or "").strip()
        source_kind, dataset = _source_of(body)
        target = str(body.get("target") or "").strip()
        if not name or not dataset or not target:
            raise HTTPException(
                status_code=400,
                detail="name, a data source (dataset or derivation), and target "
                "are required",
            )
        raw_features = body.get("features")
        features = (
            [str(f).strip() for f in raw_features if str(f).strip()]
            if isinstance(raw_features, list)
            else []
        )
        task = str(body.get("task") or "auto")
        try:
            time_budget = float(body.get("time_budget") or 60.0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="time_budget must be a number of seconds"
            ) from exc
        metric = str(body["metric"]).strip() if body.get("metric") else None
        horizon = None
        if body.get("horizon") is not None:
            try:
                horizon = int(body["horizon"])
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="horizon must be an integer"
                ) from exc
        from elbi_core.ml import ENGINES

        engine = str(body.get("engine") or "flaml")
        if engine not in ENGINES:
            raise HTTPException(
                status_code=400, detail=f"engine must be one of {', '.join(ENGINES)}"
            )
        spec = {
            "name": name,
            "dataset": dataset,
            "source_kind": source_kind,
            "engine": engine,
            "target": target,
            "features": features,
            "task": task,
            "time_budget": time_budget,
            "metric": metric,
            "ensemble": bool(body.get("ensemble", False)),
            "time_col": str(body["time_col"]).strip() if body.get("time_col") else None,
            "horizon": horizon,
            # The entity a row belongs to. Repeated observations of one entity on both
            # sides of the holdout score the model on what it trained on.
            "groups": str(body["groups"]).strip() if body.get("groups") else None,
        }
        try:
            service.check_train(name, dataset, target, source_kind)
        except ModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if job_runner is None:
            # No store means no durable jobs; train inline and answer in the same
            # shape a finished job would have, so the caller handles one contract.
            try:
                result = await run_in_threadpool(
                    lambda: service.train_result(**_train_kwargs(spec))
                )
            except ModelError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            _after_training(result)
            return _inline_job(f"train model {name}", result)
        return _job_payload(_submit_training(spec))

    @app.post("/api/registry/models/{name}/batch-score")
    async def batch_score_model(name: str, body: dict[str, Any]) -> WireJob:
        """Score a whole dataset with a registered model, as a background job.

        Same job contract as training: a job payload to poll (or, with no store,
        the finished-job shape inline). The full predictions land as a
        ``predictions.csv`` artifact on a scoring run in the model's experiment;
        the job result carries the summary and the run id.
        """
        service = _model_service()
        source_kind, dataset = _source_of(body)
        if not dataset:
            raise HTTPException(
                status_code=400, detail="a dataset or derivation is required"
            )
        ref = str(body["version"]).strip() if body.get("version") else None
        limit = None
        if body.get("limit") is not None:
            try:
                limit = int(body["limit"])
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="limit must be an integer"
                ) from exc
        try:
            service.registry().resolve(name, ref)
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if source_kind == "dataset" and dataset not in load_datasets():
            raise HTTPException(status_code=400, detail=f"no dataset named {dataset!r}")

        def work(
            progress: Callable[[str], None], cancelled: Callable[[], bool]
        ) -> dict[str, Any]:
            progress(f"scoring {dataset!r} with {name}")
            return service.batch_score_result(name, dataset, ref, limit, source_kind)

        if job_runner is None:
            try:
                result = await run_in_threadpool(
                    service.batch_score_result, name, dataset, ref, limit, source_kind
                )
            except ModelError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return _inline_job(f"batch score {name}", result)
        key = hash_json(
            {
                "batch": name,
                "dataset": dataset,
                "kind": source_kind,
                "ref": ref,
                "limit": limit,
            }
        )
        return _job_payload(job_runner.submit(key, f"batch score {name}", work))

    @app.post("/api/serving/{name}/raw-invocations")
    async def invoke_model_raw(name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Score RAW rows through the model's recorded feature derivation.

        The skew-proof path for models trained on a certified feature pipeline:
        the same code that engineered the training features engineers these rows
        before scoring, so callers send source-shaped records instead of
        reproducing the feature logic. Logged to the inference table in the
        engineered space, so drift comparisons stay apples-to-apples.
        """
        service = _model_service()
        started = time.monotonic()
        try:
            result = await run_in_threadpool(service.invoke_raw, name, body)
        except ModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if store is not None:
            try:
                store.record_inference(
                    name,
                    0,
                    n_rows=int(result.get("n_scored_rows") or 0),
                    latency_ms=(time.monotonic() - started) * 1000,
                    request_json=json.dumps(result.get("engineered") or [])[
                        :_INFERENCE_JSON_CAP
                    ],
                    predictions_json=json.dumps(result.get("predictions"))[
                        :_INFERENCE_JSON_CAP
                    ],
                )
            except Exception:
                logger.exception("failed to record raw inference event")
        return result

    @app.get("/api/registry/models/{name}/retrain")
    async def get_retrain_policy(name: str) -> dict[str, Any]:
        """The model's continuous-training policy (enabled or not)."""
        _model_service()
        policy = store.get_retrain_policy(name) if store is not None else None
        if policy is None:
            return {"configured": False}
        return {
            "configured": True,
            "dataset": policy.dataset,
            "source_kind": policy.source_kind,
            "engine": policy.engine,
            "target": policy.target,
            "features": json.loads(policy.features_json or "[]"),
            "task": policy.task,
            "time_budget": policy.time_budget,
            "metric": policy.metric,
            "ensemble": policy.ensemble,
            "mode": policy.mode,
            "interval_hours": policy.interval_hours,
            "time_col": policy.time_col,
            "horizon": policy.horizon,
            "groups": policy.groups,
            "enabled": policy.enabled,
            "last_run_at": policy.last_run_at.isoformat()
            if policy.last_run_at
            else None,
        }

    @app.get("/api/registry/retrain-policies")
    async def list_retrain_policies() -> list[dict[str, Any]]:
        """Every model's continuous-training policy, for config-as-code pull."""
        if store is None:
            return []
        return [
            {
                "model": p.model,
                "dataset": p.dataset,
                "source_kind": p.source_kind,
                "target": p.target,
                "features": json.loads(p.features_json or "[]"),
                "task": p.task,
                "engine": p.engine,
                "time_budget": p.time_budget,
                "metric": p.metric,
                "ensemble": p.ensemble,
                "mode": p.mode,
                "interval_hours": p.interval_hours,
                "time_col": p.time_col,
                "horizon": p.horizon,
                "groups": p.groups,
                "enabled": p.enabled,
            }
            for p in store.list_retrain_policies()
        ]

    @app.put("/api/registry/models/{name}/retrain")
    async def put_retrain_policy(
        name: str, body: dict[str, Any], request: Request
    ) -> dict[str, Any]:
        """Set the model's continuous-training policy.

        ``mode`` is ``on_data_change`` (retrain when the dataset's content hash
        moves; the current hash is baselined now so enabling never retrains
        immediately) or ``interval`` (every ``interval_hours``). The rest is the
        training spec the retrain submits verbatim.
        """
        from .db import RetrainPolicy

        service = _model_service()
        if store is None:
            raise HTTPException(status_code=404, detail="policies need a store")
        source_kind, dataset = _source_of(body)
        target = str(body.get("target") or "").strip()
        mode = str(body.get("mode") or "on_data_change")
        if mode not in ("on_data_change", "interval"):
            raise HTTPException(
                status_code=400, detail="mode is on_data_change or interval"
            )
        try:
            service.check_train(name, dataset, target, source_kind)
        except ModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raw_features = body.get("features")
        features = (
            [str(f).strip() for f in raw_features if str(f).strip()]
            if isinstance(raw_features, list)
            else []
        )
        try:
            time_budget = float(body.get("time_budget") or 300.0)
            interval_hours = float(body.get("interval_hours") or 24.0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        baseline = None
        if mode == "on_data_change":
            try:
                baseline = hash_json(service.load_source(source_kind, dataset))
            except ModelError:
                baseline = None
        store.set_retrain_policy(
            RetrainPolicy(
                model=name,
                dataset=dataset,
                source_kind=source_kind,
                target=target,
                features_json=json.dumps(features),
                task=str(body.get("task") or "auto"),
                time_budget=time_budget,
                metric=str(body["metric"]).strip() if body.get("metric") else None,
                ensemble=bool(body.get("ensemble", False)),
                mode=mode,
                interval_hours=interval_hours,
                enabled=bool(body.get("enabled", True)),
                last_data_hash=baseline,
                engine=str(body.get("engine") or "flaml"),
                time_col=str(body["time_col"]).strip()
                if body.get("time_col")
                else None,
                horizon=int(body["horizon"])
                if body.get("horizon") is not None
                else None,
                groups=str(body["groups"]).strip() if body.get("groups") else None,
            )
        )
        store.record_audit("set_retrain_policy", target_type="model", target_id=name)
        return await get_retrain_policy(name)

    @app.delete("/api/registry/models/{name}/retrain")
    async def delete_retrain_policy(name: str, request: Request) -> dict[str, Any]:
        """Remove the model's continuous-training policy."""
        _model_service()
        if store is None or not store.delete_retrain_policy(name):
            raise HTTPException(status_code=404, detail="no policy for this model")
        store.record_audit("delete_retrain_policy", target_type="model", target_id=name)
        return {"deleted": name}

    @app.get("/api/registry/models/{name}/inference")
    async def model_inference_log(
        name: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """The model's recent serving traffic (newest first), from the store."""
        _model_service()
        if store is None:
            return []
        return [
            {
                "id": e.id,
                "at": e.at.isoformat(),
                "version": e.version,
                "n_rows": e.n_rows,
                "latency_ms": round(e.latency_ms, 2),
                "status": e.status,
                "error": e.error,
                "predictions": json.loads(e.predictions_json)
                if e.predictions_json
                else None,
            }
            for e in store.list_inference(name, limit=min(limit, 500), offset=offset)
        ]

    @app.post("/api/registry/models/{name}/drift-check")
    async def run_drift_check(name: str, request: Request) -> dict[str, Any]:
        """Compare recent serving traffic against the training reference now.

        Uses the inference table's stored inputs; a dataset-drift outcome also
        fires the ``drift.detected`` webhook. Answers 400 when there is not yet
        enough traffic to test.
        """
        service = _model_service()
        if store is None:
            raise HTTPException(status_code=404, detail="drift needs a store")
        current = store.inference_inputs(name)
        try:
            result = await run_in_threadpool(service.drift_result, name, current)
        except ModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if result["dataset_drift"]:
            # Unlike the scheduled tick, this on-demand check has a requester.
            dispatch_event(
                store,
                "drift.detected",
                {
                    "name": name,
                    "version": result["version"],
                    "share_drifted": result["share_drifted"],
                    "n_drifted": result["n_drifted"],
                },
            )
        return result

    @app.get("/api/registry/models/{name}/drift")
    async def drift_history(name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Recent drift checks for the model, newest first."""
        from elbi_core.ml.training import scoped_tracking

        service = _model_service()
        import mlflow

        with scoped_tracking(service.tracking_uri()):
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is None:
                raise HTTPException(status_code=404, detail="no such model")
            runs = mlflow.search_runs(
                [experiment.experiment_id],
                filter_string="tags.`elbi.drift_check` = 'true'",
                max_results=min(limit, 100),
                output_format="list",
            )
        return [
            {
                "run_id": run.info.run_id,
                "at": run.info.start_time,
                "version": run.data.tags.get("elbi.model_version"),
                "n_drifted": run.data.metrics.get("n_drifted"),
                "share_drifted": run.data.metrics.get("share_drifted"),
                "dataset_drift": bool(run.data.metrics.get("dataset_drift")),
                "n_current_rows": run.data.metrics.get("n_current_rows"),
            }
            for run in runs
        ]

    @app.get("/api/registry/models/{name}/versions/{version}/schema")
    async def registered_model_schema(name: str, version: str) -> dict[str, Any]:
        """A version's serving contract: signature and the logged input example."""
        try:
            return await run_in_threadpool(
                _model_service().registry().schema, name, version
            )
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/registry/models/{name}")
    async def delete_registered_model(name: str, request: Request) -> dict[str, Any]:
        """Delete a registered model and every version of it."""
        try:
            _model_service().registry().delete(name)
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if store is not None:
            store.record_audit("delete_model", target_type="model", target_id=name)
        fire_model_event(store, "registered_model.deleted", {"name": name})
        return {"deleted": name}

    @app.delete("/api/registry/models/{name}/versions/{version}")
    async def delete_registered_model_version(
        name: str, version: int, request: Request
    ) -> dict[str, Any]:
        """Delete one version of a registered model."""
        try:
            _model_service().registry().delete_version(name, version)
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if store is not None:
            store.record_audit(
                "delete_model_version",
                target_type="model_version",
                target_id=f"{name}/v{version}",
            )
        fire_model_event(
            store, "model_version.deleted", {"name": name, "version": version}
        )
        return {"deleted": name, "version": version}

    @app.get("/api/registry/models/{name}")
    async def get_registered_model(name: str) -> dict[str, Any]:
        """One model's versions, each with its run's metrics and params."""
        try:
            return _model_service().model_view(name)
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/registry/models/{name}/promote")
    async def promote_registered_model(
        name: str, body: dict[str, Any], request: Request
    ) -> dict[str, Any]:
        """Point an alias (default ``champion``) at a version of ``name``."""
        service = _model_service()
        try:
            version = int(body.get("version", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="version must be an integer"
            ) from exc
        alias = str(body.get("alias") or "champion").strip() or "champion"
        try:
            service.registry().promote(name, version, alias)
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if store is not None:
            store.record_audit(
                "promote_model",
                target_type="model_version",
                target_id=f"{name}/v{version}",
            )
        fire_model_event(
            store, "alias.updated", {"name": name, "alias": alias, "version": version}
        )
        return {"name": name, "version": version, "alias": alias}

    @app.post("/api/serving/{name}/invocations")
    async def invoke_model(
        name: str, body: dict[str, Any], version: str | None = None
    ) -> dict[str, Any]:
        """Score with a registered model, speaking MLflow's ``/invocations`` protocol.

        The body is one of ``dataframe_split``/``dataframe_records``/``instances``/
        ``inputs`` (plus optional ``params``) and the response is ``{"predictions"}``,
        so any MLflow-scoring client can call a registered model here. ``version``
        selects a version or alias; the default serves the champion (or newest).
        """
        service = _model_service()
        try:
            resolved = service.registry().resolve(name, version)
        except ModelError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        def _log(
            payload: dict[str, Any] | None, status: str, error: str | None, ms: float
        ) -> None:
            # The inference table: capped request/response JSON per event, so
            # production traffic is inspectable and drift-monitorable later.
            if store is None:
                return
            request_rows = _request_rows(body)
            try:
                store.record_inference(
                    name,
                    resolved,
                    n_rows=len(request_rows),
                    latency_ms=ms,
                    status=status,
                    error=error,
                    request_json=json.dumps(request_rows)[:_INFERENCE_JSON_CAP],
                    predictions_json=(
                        json.dumps(payload.get("predictions"))[:_INFERENCE_JSON_CAP]
                        if payload is not None
                        else None
                    ),
                )
            except Exception:
                logger.exception("failed to record inference event")

        started = time.monotonic()
        try:
            result = await run_in_threadpool(service.invoke, name, body, version)
        except ModelError as exc:
            _log(None, "error", str(exc), (time.monotonic() - started) * 1000)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _log(result, "ok", None, (time.monotonic() - started) * 1000)
        return result

    @app.get("/api/settings/webhooks")
    async def get_webhooks() -> WireWebhookSettings:
        """The model-registry webhook target (secret masked to a boolean)."""
        url, secret = webhook_config(store)
        return WireWebhookSettings(url=url, secret_set=bool(secret))

    @app.put("/api/settings/webhooks")
    async def put_webhooks(body: dict[str, Any]) -> dict[str, Any]:
        """Set or clear the registry webhook: events on version/alias/deletion.

        Deliveries are HMAC-signed (X-Elbi-Signature) when a secret is
        set; an empty URL disables the webhook.
        """
        if store is None:
            raise HTTPException(status_code=404, detail="settings are not enabled")
        url = str(body.get("url") or "").strip()
        store.set_config("model_webhook_url", url)
        if "secret" in body:
            store.set_config("model_webhook_secret", str(body.get("secret") or ""))
        return {"url": url, "secret_set": bool(webhook_config(store)[1])}

    # -- data sources -----------------------------------------------------------
    def _source_from_body(
        body: dict[str, Any], existing: DataSource | None
    ) -> DataSource:
        """Build a DataSource from a request body, encrypting the secret.

        An absent secret (or the mask) keeps the stored one; a stored secret requires
        ``APP_SECRET_KEY`` so it is never persisted in the clear.
        """
        secret = existing.secret if existing else None
        raw = body.get("secret")
        if isinstance(raw, str) and raw and raw != datasources.PASSWORD_MASK:
            if not crypto.is_configured():
                raise HTTPException(
                    status_code=400,
                    detail=f"{crypto.SECRET_ENV} must be set to store a secret",
                )
            secret = crypto.encrypt(raw)
        elif raw == "":
            secret = None
        source = DataSource(
            name=str(body.get("name", existing.name if existing else "")).strip(),
            kind=str(body.get("kind", existing.kind if existing else "")).strip(),
            host=body.get("host", existing.host if existing else None),
            port=body.get("port", existing.port if existing else None),
            database=body.get("database", existing.database if existing else None),
            username=body.get("username", existing.username if existing else None),
            secret=secret,
            secret_env=body.get(
                "secret_env", existing.secret_env if existing else None
            ),
            extra=body.get("extra", existing.extra if existing else None),
        )
        if existing is not None:  # preserve identity + creation time on update
            source.id = existing.id
            source.created_at = existing.created_at
        return source

    @app.get("/api/data-sources")
    async def list_data_sources() -> list[WireDataSource]:
        """Every registered data source (secrets masked)."""
        if store is None:
            return []
        return [datasources.view(s) for s in store.list_data_sources()]

    @app.post("/api/data-sources")
    async def create_data_source(
        body: dict[str, Any], request: Request
    ) -> WireDataSource:
        """Register a data source; the secret is encrypted at rest before storing."""
        if store is None:
            raise HTTPException(status_code=404, detail="data sources are not enabled")
        name = str(body.get("name") or "").strip()
        if not name or not str(body.get("kind") or "").strip():
            raise HTTPException(status_code=400, detail="name and kind are required")
        if store.get_data_source_by_name(name) is not None:
            raise HTTPException(status_code=400, detail=f"{name!r} already exists")
        source = _source_from_body(body, None)
        # View the transient object before the commit expires its attributes.
        wire = datasources.view(source)
        store.save_data_source(source)
        store.record_audit(
            "data_source.create",
            target_type="data_source",
            target_id=wire.id,
        )
        return wire

    @app.get("/api/data-sources/{source_id}")
    async def get_data_source(source_id: str) -> WireDataSource:
        source = store.get_data_source(source_id) if store else None
        if source is None:
            raise HTTPException(status_code=404, detail="no such data source")
        return datasources.view(source)

    @app.put("/api/data-sources/{source_id}")
    async def update_data_source(
        source_id: str, body: dict[str, Any]
    ) -> WireDataSource:
        existing = store.get_data_source(source_id) if store else None
        if store is None or existing is None:
            raise HTTPException(status_code=404, detail="no such data source")
        source = _source_from_body(body, existing)
        wire = datasources.view(source)  # before the commit expires it
        store.save_data_source(source)
        return wire

    @app.delete("/api/data-sources/{source_id}")
    async def delete_data_source(source_id: str) -> WireOk:
        if store is None or not store.delete_data_source(source_id):
            raise HTTPException(status_code=404, detail="no such data source")
        return WireOk()

    @app.post("/api/data-sources/test")
    async def test_data_source(body: dict[str, Any]) -> dict[str, Any]:
        """Test an unsaved connection payload before saving it.

        Gated to data-source managers: it opens a connection to an arbitrary host, so an
        unprivileged caller must not probe internal services through it. Private-network
        hosts are intentionally allowed (on-prem data lives there), so the permission
        gate, not IP filtering, is the control.
        """
        return datasources.test_connection(_source_from_body(body, None))

    @app.post("/api/data-sources/{source_id}/test")
    async def test_saved_data_source(source_id: str) -> dict[str, Any]:
        source = store.get_data_source(source_id) if store else None
        if source is None:
            raise HTTPException(status_code=404, detail="no such data source")
        return datasources.test_connection(source)

    @app.post("/api/data-sources/{source_id}/preview")
    async def preview_data_source(
        source_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Sample rows from a source (``{query}`` or ``{table}``) via the runtime."""
        source = store.get_data_source(source_id) if store else None
        if source is None:
            raise HTTPException(status_code=404, detail="no such data source")
        query = str(body.get("query") or "").strip() or None
        table = str(body.get("table") or "").strip() or None
        limit = min(int(body.get("max_rows") or 100), 1000)
        try:
            rows = datasources.load_rows(
                source, query=query, table=table, max_rows=limit
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"rows": rows, "count": len(rows)}

    @app.post("/api/chat")
    async def chat(request: Request) -> EventSourceResponse:
        # The request is an AI SDK `useChat` payload: a list of UI messages (plus an
        # optional model id). The question is the last user message's text.
        body = await request.json()
        # Governance gate: refuse the turn before it spends anything.
        if store is not None and store.is_over_budget():
            raise HTTPException(status_code=402, detail="budget exceeded")
        question = _question_from(body)
        context = str(body.get("context") or "")
        # Any images attached to this turn, as data URLs; bounded so a request cannot
        # ship an unbounded payload. The model reads them alongside the question.
        images = [u for u in (body.get("images") or []) if isinstance(u, str)][:4]
        # Edit-and-resend: id of a past user message being edited. Its turn and every
        # later one are dropped before this question runs, so the chat re-runs here.
        edit_message_id = str(body.get("editMessageId") or "").strip() or None
        # The LLM profile the composer's switcher selected for this turn, if any.
        requested_profile = str(body.get("profile") or "").strip() or None

        # Continue the conversation the client names (its stable id), or start one.
        # Continuing lets the run see prior turns and the durable results already
        # established, so a follow-up like "now tune it" resolves instead of restarting.
        client_conversation_id = str(body.get("conversationId") or "").strip() or None
        conversation_id: str | None = None
        derive: DeriveFn | None = None
        history: list[tuple[str, str]] = []
        memory = ""
        summary = ""
        all_turns: list[tuple[str, str]] = []
        existing_conv = None
        # The profile to run on: the switcher's pick, else the conversation's own, else
        # the default. make_client resolves it to a client (loading its model + key).
        profile = requested_profile
        if store is not None and question:
            existing = (
                store.get_conversation(client_conversation_id)
                if client_conversation_id is not None
                else None
            )
            if existing is not None:
                existing_conv = existing
                conversation_id = existing.id
                # An edit drops the edited turn and everything after it, so the resend
                # replaces that point in the conversation instead of appending.
                if edit_message_id is not None:
                    store.delete_messages_from(conversation_id, edit_message_id)
                # Load prior turns and durable memory BEFORE recording this question.
                all_turns = _history_from(store.messages(conversation_id))
                memory = _memory_from(store.derivations_for(conversation_id))
                is_new = False
                profile = requested_profile or existing.profile
                if requested_profile and requested_profile != existing.profile:
                    store.set_conversation_profile(conversation_id, requested_profile)
            else:
                # Profile precedence for a new chat: the switcher's pick, then the
                # configured default.
                profile = requested_profile or store.get_config("llm.default_profile")
                conversation_id = store.create_conversation(
                    _title(question),
                    conversation_id=client_conversation_id,
                    profile=profile,
                )
                is_new = True
            store.add_message(conversation_id, "user", content=question)
            if derive_factory is not None:
                derive = derive_factory(conversation_id, question)
            # A new conversation starts with the plain truncated question as its title;
            # generate a better one from the LLM in the background (off the request
            # path) and replace it when ready. Any failure leaves the plain title in
            # place, so this can never block a turn or break a conversation.
            if is_new and generate_title is not None:
                _generate_title_async(store, conversation_id, question, generate_title)
        # Resolve the client from the chosen profile (loading its model + key), so a
        # switch takes effect this turn. Tests inject a fixed client and no factory.
        # With no model configured, answer with a clear 400 rather than a default.
        # LLMConfigError gets a `code` the frontend keys off to point at Settings,
        # rather than showing this specific, actionable message as a generic failure.
        try:
            run_client = (
                make_client(profile) if make_client is not None else _default_client()
            )
        except LLMConfigError as exc:
            raise HTTPException(
                status_code=400,
                detail={"message": str(exc), "code": "llm_not_configured"},
            ) from exc
        except ElbiError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Replay only the recent window of turns; condense everything before it into a
        # short cached summary so a long conversation keeps its earlier goals without
        # replaying every message. The summary is recomputed only when the number of
        # older turns changes, so a normal follow-up costs no extra model call.
        history = all_turns[-_HISTORY_MESSAGES:]
        if (
            store is not None
            and conversation_id is not None
            and existing_conv is not None
            and len(all_turns) > _HISTORY_MESSAGES
        ):
            older = all_turns[:-_HISTORY_MESSAGES]
            if existing_conv.summary and existing_conv.summary_turns == len(older):
                summary = existing_conv.summary
            else:
                summary = condense_turns(run_client, older)
                store.set_summary(conversation_id, summary, len(older))

        scratch_dir: Path | None = None
        if scratch_root is not None and conversation_id is not None:
            scratch_dir = scratch_root / conversation_id
            scratch_dir.mkdir(parents=True, exist_ok=True)

        # With a conversation to key on, the workspace (and its live session: namespace,
        # fitted models, files) is kept across turns, so iterative work is cheap. With
        # no store, it is a throwaway per-request session, closed when the run ends.
        persistent = conversation_id is not None
        if conversation_id is not None:
            workspace = _acquire_workspace(conversation_id, derive, scratch_dir)
        else:
            workspace = _bind_model_tools(
                Workspace(
                    datasets=load_datasets(),
                    derive=derive,
                    scratch_dir=scratch_dir,
                    stateful=True,
                    backend=sandbox,
                    egress=egress,
                    image=sandbox_image,
                    executor=SubprocessExecutor(timeout=_RUN_CODE_TIMEOUT_SECONDS),
                ),
            )

        # Let this turn launch background jobs. `submit` runs a long derivation (bound
        # to this turn's derive, so the job authors and persists against this chat);
        # `submit_code` runs a long run_code snippet in its own session sharing the
        # workspace files. Both are refreshed each turn, like `derive` itself.
        if job_runner is not None:
            ws_datasets = dict(workspace.datasets)
            ws_scratch = workspace.scratch_dir
            workspace.submit_code = lambda code, deps: (
                submit_code_job(
                    job_runner,
                    code=code,
                    deps=deps,
                    datasets=ws_datasets,
                    scratch_dir=ws_scratch,
                    backend=sandbox,
                    egress=egress,
                    image=sandbox_image,
                ).id
            )
            if derive is not None and conversation_id is not None:
                turn_derive = derive
                turn_conversation = conversation_id
                dataset_names = list(workspace.datasets.keys())

                def _submit(
                    name: str,
                    source: str,
                    claim: Any,
                    contract: Any,
                    fmt: str,
                    assumptions: Any,
                    deps: Any,
                ) -> str:
                    # Register this conversation for the follow-up BEFORE submitting,
                    # keyed by the same content address the job uses, so a fast job
                    # cannot finish (fire on_complete) before it is recorded as waiting.
                    key = derivation_job_key(
                        name, source, claim, contract, fmt, deps, dataset_names
                    )
                    with job_followups_lock:
                        job_followups.setdefault(key, set()).add(turn_conversation)
                    job = submit_derivation_job(
                        job_runner,
                        turn_derive,
                        name=name,
                        source=source,
                        claim=claim,
                        contract=contract,
                        fmt=fmt,
                        assumptions=assumptions,
                        deps=deps,
                        datasets=dataset_names,
                    )
                    # Deduped onto a finished job: on_complete will not fire again, so
                    # deliver this conversation's follow-up from the cached result.
                    if job.done:
                        _on_job_complete(job)
                    return job.id

                workspace.submit = _submit

        # The turn's trace and outcome, shared with publisher() so the assistant message
        # is persisted exactly once in its finally: whether the run produced a clean
        # answer, hit a server error, or was stopped/disconnected mid-stream. Without
        # this a failed or stopped turn leaves the user's question saved with no reply.
        trace: list[dict[str, Any]] = []
        final: AnswerResult | None = None
        error: AnswerResult | None = None

        async def publisher() -> AsyncIterator[dict[str, str]]:
            # Emit the AI SDK UI Message Stream: a start, the agent's steps as custom
            # data parts, the answer as a text part, the verification payload as a
            # custom data part, then finish. `useChat` reassembles these into one
            # message with typed parts the UI renders. A per-request (non-persistent)
            # session is closed in the finally; a persistent one is kept for the
            # conversation's next turn and reaped by the idle sweep instead.
            try:
                async for chunk in _run():
                    yield chunk
            finally:
                # Persist the assistant turn here, not inside _run, so it runs even when
                # the client disconnects mid-stream and the generator is closed: a clean
                # answer, an error, or a stopped record, so the turn is never orphaned.
                if question and store is not None and conversation_id is not None:
                    outcome = final or error or _stopped()
                    store.add_message(
                        conversation_id,
                        "assistant",
                        content=outcome.narrative,
                        tools=trace,
                        result=_result_payload(outcome),
                    )
                    # Add this turn's tokens and cost to the conversation totals, and
                    # its cost to the governing budget window(s) (user and global).
                    store.add_usage(
                        conversation_id,
                        outcome.usage.prompt_tokens,
                        outcome.usage.completion_tokens,
                        outcome.usage.cost,
                    )
                    store.add_spend(outcome.usage.cost)
                    # Attributed audit trail: who asked, and whether it was sound.
                    store.record_audit(
                        "answer",
                        target_type="conversation",
                        target_id=conversation_id,
                        verdict=outcome.verdict,
                        data_hash=outcome.data_hash,
                    )
                if not persistent:
                    workspace.close()

        async def _run() -> AsyncIterator[dict[str, str]]:
            yield _data({"type": "start", "messageId": uuid4().hex})
            yield _data({"type": "start-step"})
            nonlocal final, error
            if question:
                # The runtime is synchronous (the LLM client blocks), so step it in a
                # worker thread and forward each event; stop early if the client leaves.
                # The reasoning, tool calls (with input), and tool outputs stream in
                # order so the UI can show the model's work as it happens.
                events = stream(
                    question,
                    workspace,
                    run_client,
                    context=context,
                    history=history,
                    memory=memory,
                    summary=summary,
                    images=images,
                )
                reasoning_buf = ""
                try:
                    async for event in iterate_in_threadpool(events):
                        if await request.is_disconnected():
                            break
                        item: dict[str, Any]
                        if event.kind == "reasoning":
                            item = {"kind": "reasoning", "text": event.text}
                            trace.append(item)
                            yield _data({"type": "data-reasoning", "data": item})
                        elif event.kind == "reasoning_delta":
                            # A streaming client emits the model's prose token by token;
                            # forward each slice live and accumulate it so the persisted
                            # trace still carries the whole reasoning for this step.
                            reasoning_buf += event.text or ""
                            yield _data(
                                {
                                    "type": "data-reasoning-delta",
                                    "data": {"text": event.text},
                                }
                            )
                        elif event.kind == "tool":
                            reasoning_buf = _flush_reasoning(trace, reasoning_buf)
                            item = {
                                "kind": "tool",
                                "name": event.tool,
                                "arguments": event.arguments,
                            }
                            trace.append(item)
                            yield _data({"type": "data-tool", "data": item})
                        elif event.kind == "tool_result":
                            item = {"kind": "tool_output", "output": event.output}
                            trace.append(item)
                            yield _data({"type": "data-tool-output", "data": item})
                        elif event.result is not None:
                            reasoning_buf = _flush_reasoning(trace, reasoning_buf)
                            final = event.result
                except Exception as exc:
                    # The LLM call or a tool failed mid-turn. Do not abort the stream
                    # truncated (no result, no finish): log it and finish with a visible
                    # error so the client shows a message instead of hanging silently.
                    logger.exception(
                        "chat turn failed for conversation %s", conversation_id
                    )
                    final = None
                    error = _empty(_turn_failure_message(exc))
            result = error or final or _empty("ask a question")

            text_id = uuid4().hex
            yield _data({"type": "text-start", "id": text_id})
            if result.narrative:
                yield _data(
                    {"type": "text-delta", "id": text_id, "delta": result.narrative}
                )
            yield _data({"type": "text-end", "id": text_id})
            yield _data({"type": "data-result", "data": _result_payload(result)})
            viz = _viz_for(result)
            if viz is not None:
                yield _data({"type": "data-viz", "data": viz})
            yield _data({"type": "finish-step"})
            yield _data({"type": "finish"})
            yield {"data": "[DONE]"}

        return EventSourceResponse(
            publisher(), headers={"x-vercel-ai-ui-message-stream": "v1"}
        )

    @app.get("/api/conversations")
    async def conversations(
        request: Request,
        limit: int = 20,
        page_id: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        """A page of past conversations, most-recently-active first.

        Cursor-paginated ``{items, next_page_id}`` so the history list scales: pass the
        returned ``next_page_id`` back as ``page_id`` for the following page (``None``
        when the list is exhausted). ``q`` filters to conversations whose title contains
        it (case-insensitive), so it stays searchable. An authenticated caller sees only
        their own conversations; single-user mode returns all. Empty with no store.
        """
        if store is None:
            return {"items": [], "next_page_id": None}
        rows, next_page_id = store.search_conversations(
            limit=limit, page_id=page_id, query=q
        )
        return {
            "items": [
                {
                    "id": c.id,
                    "title": c.title,
                    "created_at": _iso_utc(c.created_at),
                    "updated_at": _iso_utc(c.updated_at),
                }
                for c in rows
            ],
            "next_page_id": next_page_id,
        }

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str, request: Request) -> WireOk:
        """Delete a conversation and its messages.

        Returns 404 when there is no such conversation. On success its live workspace
        (the sandbox session, the namespace and the files) is torn down, so a deleted
        conversation leaves nothing running.
        """
        if store is None:
            raise HTTPException(status_code=404, detail="conversations are not enabled")
        if await _conversation(conversation_id, request) is None:
            raise HTTPException(
                status_code=404, detail=f"no conversation {conversation_id!r}"
            )
        store.delete_conversation(conversation_id)
        workspace_seen.pop(conversation_id, None)
        ws = live_workspaces.pop(conversation_id, None)
        if ws is not None:
            with suppress(Exception):
                ws.close()
        return WireOk()

    @app.patch("/api/conversations/{conversation_id}")
    async def rename_conversation(
        conversation_id: str, body: dict[str, Any], request: Request
    ) -> dict[str, str]:
        """Rename a conversation. 404 if it is missing, 400 if the title is empty."""
        if store is None:
            raise HTTPException(status_code=404, detail="conversations are not enabled")
        title = str(body.get("title", "")).strip()
        if not title:
            raise HTTPException(status_code=400, detail="title must not be empty")
        title = title[:120]
        if await _conversation(conversation_id, request) is None:
            raise HTTPException(
                status_code=404, detail=f"no conversation {conversation_id!r}"
            )
        store.rename_conversation(conversation_id, title)
        return {"id": conversation_id, "title": title}

    @app.get("/api/conversations/{conversation_id}/messages")
    async def conversation_messages(
        conversation_id: str, request: Request
    ) -> list[dict[str, Any]]:
        """The turns of the caller's conversation, with their tools and results."""
        if store is None or await _conversation(conversation_id, request) is None:
            return []
        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "tools": json.loads(m.tools_json),
                "result": json.loads(m.result_json) if m.result_json else None,
                "feedback": m.feedback,
            }
            for m in store.messages(conversation_id)
        ]

    @app.put("/api/messages/{message_id}/feedback")
    async def set_message_feedback(
        message_id: str, body: dict[str, Any], request: Request
    ) -> dict[str, str | None]:
        """Rate an assistant turn."""
        if store is None:
            raise HTTPException(status_code=404, detail="messages are not enabled")
        feedback = body.get("feedback")
        if feedback not in ("up", "down", None):
            raise HTTPException(status_code=400, detail="feedback must be up/down/null")
        message = store.get_message(message_id)
        if (
            message is None
            or await _conversation(message.conversation_id, request) is None
        ):
            raise HTTPException(status_code=404, detail=f"no message {message_id!r}")
        store.set_feedback(message_id, feedback)
        return {"feedback": feedback}

    @app.get("/api/conversations/{conversation_id}/export")
    async def export_conversation(conversation_id: str, request: Request) -> Response:
        """Download the caller's conversation as a self-contained JSON document."""
        conversation = await _conversation(conversation_id, request)
        if store is None or conversation is None:
            raise HTTPException(status_code=404, detail="no such conversation")
        document = _conversation_document(conversation, store.messages(conversation_id))
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", conversation.title).strip("-") or "chat"
        return Response(
            content=json.dumps(document, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{stem}.json"'},
        )

    @app.get("/api/conversations/{conversation_id}/usage")
    async def conversation_usage(
        conversation_id: str, request: Request
    ) -> dict[str, Any]:
        """The caller's conversation's accumulated token counts and cost."""
        conversation = await _conversation(conversation_id, request)
        if conversation is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        return {
            "prompt_tokens": conversation.prompt_tokens or 0,
            "completion_tokens": conversation.completion_tokens or 0,
            "cost": conversation.cost or 0.0,
        }

    def _search_offset(page_id: str | None) -> int:
        """The cursor as an offset, matching ``search_conversations``.

        A malformed cursor is page one rather than an error: the client's alternative is
        a 400 for a value it did not compose itself.
        """
        try:
            return max(0, int(page_id)) if page_id is not None else 0
        except ValueError:
            return 0

    def _search_filters(types: str, verdict: str) -> dict[str, list[str]]:
        """The caller's facet choices, as the columns the ranking statement knows.

        Unknown columns cannot arrive here (the names are fixed) and unknown *values*
        match nothing, so neither needs rejecting.
        """
        chosen = {
            "entity_type": [t.strip() for t in types.split(",") if t.strip()],
            "verdict": [v.strip() for v in verdict.split(",") if v.strip()],
        }
        return {column: values for column, values in chosen.items() if values}

    def _facet_counts(
        index: SearchIndex,
        column: str,
        query: str,
        filters: dict[str, list[str]],
        vector: Sequence[float] | None = None,
    ) -> list[tuple[Any, int]]:
        """One facet's buckets, narrowed by the query and every filter but its own."""
        where_sql, where_params = filter_sql(filters or None, skip=column)
        return index.facet_counts(
            column,
            query=query,
            vector=vector,
            where_sql=where_sql,
            where_params=where_params,
        )

    def _search_item(
        row: dict[str, Any], query: str, may_read_body: bool
    ) -> dict[str, Any]:
        """One result: what it is, why it matched, and where to go.

        The snippet is drawn from the body only when the caller may read it. For a
        browse-only document it comes from the fields that document could be *matched*
        on, so the snippet never says more than the ranking already did.
        """
        readable = f"{row['title']} {row['body']}" if may_read_body else row["title"]
        return {
            "type": row["entity_type"],
            "id": row["entity_id"],
            "name": row["title"],
            "snippet": snippet(f"{readable} {row['columns']}".strip(), query),
            "verdict": row["verdict"],
            "status": row["status"],
            "created_at": _iso_utc(row["created_at"]) if row["created_at"] else None,
            # Empty for an entity the SPA has no page for; a result that is not a link
            # beats one that 404s.
            "route": row["route"] or None,
        }

    def _search() -> SearchBuilder:
        """The builder, or a 503 when its index could not be opened.

        The builder rather than the index alone: the reader needs its embedder to ask
        the dense half anything, and it is the one object that holds both.

        ``serve`` treats an unopenable index as a degradation rather than a failure, so
        the deployment is up and this one surface is not.
        """
        if search_builder is None:
            raise HTTPException(status_code=503, detail="search is not available")
        return search_builder

    @app.get("/api/search")
    async def search(
        request: Request,
        q: str = "",
        types: str = "",
        verdict: str = "",
        limit: int = 20,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """Hybrid-ranked results across every indexed artifact, scoped to the caller.

        Cursor-paginated ``{items, next_page_id}`` like ``/api/conversations``: pass the
        returned ``next_page_id`` back as ``page_id``. ``types`` and ``verdict`` are
        comma-separated and narrow *inside* the ranking statement, so a rare type still
        fills a page instead of being crowded out before the filter runs.

        Authorization is a predicate in that same statement rather than a pass over its
        output, which is what keeps a scoped caller's page full and their facet counts
        honest.
        """
        builder = _search()
        index = builder.index
        if not q.strip():
            return {"items": [], "next_page_id": None, "facets": {}}
        filters = _search_filters(types, verdict)
        where_sql, where_params = filter_sql(filters or None)
        limit = max(1, min(limit, 100))
        offset = _search_offset(page_id)

        # ``candidates`` moves with the offset: it caps the ranking statement's own
        # LIMIT, so a fixed one strands every page past the first window.
        wanted = offset + limit + 1
        # Both halves or neither: without a vector ``search`` silently ranks lexically,
        # so every stored embedding and the HNSW index over them go unread and a query
        # that means the right thing without sharing a word finds nothing.
        vector = await run_in_threadpool(builder.embed_query, q)

        # Chunks collapse to one result per entity *after* fusion, so a window of
        # documents is not a window of results: enough chunks of one entity fill it and
        # the page comes back holding a single item with no cursor, while entities that
        # matched sit below the cut. Widen until the page is full or the rankings are
        # spent, rather than assuming a fixed multiple is enough for every corpus.
        candidates = max(DEFAULT_CANDIDATES, wanted)
        while True:
            doc_ids = await run_in_threadpool(
                index.search,
                q,
                vector=vector,
                limit=wanted,
                candidates=candidates,
                where_sql=where_sql,
                where_params=where_params,
            )
            if len(doc_ids) >= wanted or candidates >= MAX_SEARCH_CANDIDATES:
                break
            widened = min(candidates * 4, MAX_SEARCH_CANDIDATES)
            if widened == candidates:
                break
            candidates = widened
        page = doc_ids[offset : offset + limit]
        rows = await run_in_threadpool(index.hydrate, page)
        return {
            "items": [_search_item(row, q, True) for row in rows],
            "next_page_id": (
                str(offset + limit) if len(doc_ids) > offset + limit else None
            ),
            "facets": {
                column: dict(
                    await run_in_threadpool(
                        _facet_counts, index, column, q, filters, vector
                    )
                )
                for column in ("entity_type", "verdict")
            },
        }

    @app.get("/api/jobs")
    async def jobs() -> list[WireJob]:
        """Background jobs (long training runs), newest first."""
        if job_runner is None:
            return []
        return [_job_payload(j) for j in job_runner.store.list()]

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> WireJob:
        """One background job: poll it for progress and its certified result."""
        if job_runner is None:
            raise HTTPException(
                status_code=404, detail="background jobs are not enabled"
            )
        job = job_runner.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return _job_payload(job)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> WireJob:
        """Request cancellation of a background job; returns its resulting state."""
        if job_runner is None:
            raise HTTPException(
                status_code=404, detail="background jobs are not enabled"
            )
        job = job_runner.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return _job_payload(job)

    @app.get("/api/derivations")
    async def derivations(request: Request) -> list[WireDerivationSummary]:
        """The authored derivations: durable cached answers, newest first."""
        if store is None:
            return []
        return [_derivation_summary(d) for d in store.list_derivations()]

    def _withholds(name: str) -> bool:
        """Whether a stored rendering must not be served for this derivation.

        Read off ``app.state`` per request rather than closed over, because whatever
        answers this may be installed after the app is built -- an extension is handed
        the app, not the arguments it was made from. The constructor argument seeds it,
        so a caller passing one directly is unaffected.
        """
        answer = getattr(app.state, "withhold_rendering", None)
        return answer is not None and bool(answer(name))

    @app.get("/api/derivations/{name}")
    async def derivation(name: str, request: Request) -> WireDerivationDetail:
        """One authored derivation in full: source, claim, output, attestation."""
        row = store.get_derivation(name) if store else None
        if row is None:
            raise HTTPException(status_code=404, detail=f"no derivation named {name!r}")
        detail = _derivation_detail(row)
        # A stored rendering is rows already computed, so it cannot be narrowed after
        # the fact. It is dropped for a caller an extension says to withhold it from,
        # and the on-demand path below withholds too. The rest of the detail serves:
        # what is withheld is rows, not the derivation's existence.
        if _withholds(name):
            detail.rendered = None
        # A repo derivation stores no rendering; produce its output on demand.
        if not detail.rendered and render_derivation is not None:
            rendered = await run_in_threadpool(render_derivation, name)
            if rendered:
                detail.rendered = rendered
        return detail

    @app.get("/api/derivations/{name}/history")
    async def derivation_history(name: str, request: Request) -> list[dict[str, Any]]:
        """A derivation's result history: its certified versions, newest first.

        Each version carries the oracle's certified estimate and its verdict, plus
        ``changed``: the input dimensions (data, code, controls, params, claim) that
        moved it from the next-older version, so a reader sees not just that the number
        changed but why. An identical re-derive collapses to the existing version rather
        than adding a row. Scoped to what the caller reaches, matching its detail view.
        """
        if store is None:
            raise HTTPException(status_code=404, detail=f"no derivation named {name!r}")
        row = store.get_derivation(name)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no derivation named {name!r}")
        history = with_changes(store.runs_for_derivation(name))
        return [
            {**_run_dict(run), "changed": list(changed)} for run, changed in history
        ]

    @app.delete("/api/derivations/{name}")
    async def delete_derivation_route(
        request: Request, name: str, permanent: bool = False
    ) -> WireOk:
        """Move a derivation to trash, or erase it immediately with ``permanent=true``.

        Refused with 409 for a repo-origin derivation: those are source files
        under version control, wiped and rebuilt on every start, which would
        silently undo a trash stamp the next time the project reloads.
        """
        db = _require_store()
        origin = db.derivation_origin(name)
        if origin is None:
            raise HTTPException(status_code=404, detail=f"no derivation named {name!r}")
        if origin == "repo":
            raise HTTPException(
                status_code=409,
                detail="repo-origin derivations are managed in the source tree",
            )
        hook = derivation_on_erase if permanent else derivation_on_trash
        if hook is None:
            raise HTTPException(
                status_code=503, detail="derivation trash is not enabled"
            )
        if not hook(name):
            raise HTTPException(status_code=404, detail=f"no derivation named {name!r}")
        return WireOk()

    if certificate_issuer is not None:

        def _build_certificate_document(name: str) -> dict[str, Any]:
            """The synchronous half: store reads, the verification gate, and signing.

            Deterministic: it is issued from the derivation's stored run and timestamp,
            so repeated exports (and a pull/sync round-trip) produce identical bytes.
            """
            row = store.get_derivation(name) if store else None
            if store is None or row is None:
                raise HTTPException(
                    status_code=404, detail=f"no derivation named {name!r}"
                )
            attestation = (
                json.loads(row.attestation_json) if row.attestation_json else {}
            )
            if not (row.verdict or attestation.get("verdict")):
                raise HTTPException(
                    status_code=404,
                    detail=f"derivation {name!r} has no verified claim to certify",
                )
            run = _certificate_run(
                row, store.latest_run_for_derivation(name), attestation
            )
            skipped = tuple((str(s), "") for s in attestation.get("skipped", []))
            return certificate_issuer.issue(run, skipped=skipped)

        async def _certificate_document(name: str, request: Request) -> dict[str, Any]:
            """Build the signed certificate for a caller-visible certified derivation.

            The DB reads and the Ed25519 sign are synchronous CPU/IO work, offloaded
            to a thread so they do not block the event loop.
            """
            return await run_in_threadpool(_build_certificate_document, name)

        # These live under /api/certificates, not /api/derivations, so the camelCase
        # boundary leaves them byte-for-byte (see casing._STANDARD_ROUTES): the
        # signature is over the exact snake_case bytes and any key rewrite would break
        # it.
        @app.get("/api/certificates/{name}")
        async def derivation_certificate(name: str, request: Request) -> Response:
            """The derivation's signed, tamper-evident certificate, as JSON."""
            document = await _certificate_document(name, request)
            return Response(
                content=json.dumps(document, indent=2, sort_keys=True),
                media_type="application/json",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{_safe_filename_part(name)}'
                        '-certificate.json"'
                    )
                },
            )

        @app.get("/api/certificates/{name}/pdf")
        async def derivation_certificate_pdf(name: str, request: Request) -> Response:
            """The derivation's certificate rendered as a one-page PDF download."""
            document = await _certificate_document(name, request)
            pdf = await run_in_threadpool(render_certificate_pdf, document)
            return Response(
                content=pdf,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{_safe_filename_part(name)}'
                        '-certificate.pdf"'
                    )
                },
            )

        @app.post("/api/certificates/{name}")
        async def verify_derivation_certificate(
            name: str, request: Request
        ) -> dict[str, Any]:
            """Verify a submitted certificate against this derivation (the sync gate).

            400 if the signature does not verify (tampered); 409 if it verifies but its
            payload is not this derivation's current certificate (stale); else ``ok``.
            No state is mutated: the certificate is derived from the stored run.
            """
            try:
                body = await request.json()
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail="certificate must be valid JSON"
                ) from exc
            try:
                verify_certificate(body, public_key=certificate_issuer.public_key)
            except CertificateError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            current = await _certificate_document(name, request)
            if body.get("certificate") != current.get("certificate"):
                raise HTTPException(
                    status_code=409,
                    detail=f"certificate does not match the current {name!r}",
                )
            return {"ok": True, "verified": True}

    # -- data-portability exports (IP-31) -----------------------------------------
    #
    # A derivation's proof (verdict, attestation, result history), a dashboard's
    # definition plus current values, and a metric's definition plus history -- as
    # downloadable documents. Every route composes the same views the app already
    # builds for its own detail routes; nothing here re-derives what a derivation,
    # dashboard, or metric "is".
    #
    # Routed under /api/exports/ rather than e.g. /api/derivations/{name}/export:
    # the casing middleware's exemption list matches whole route *prefixes*
    # (casing._STANDARD_ROUTES), so a leaf route cannot opt out on its own, and an
    # export embedding a certificate's signed bytes must never be camelCased --
    # that would silently invalidate the signature.

    def _export_certificate(row: Derivation) -> dict[str, Any] | None:
        """The derivation's certificate for export, or ``None`` -- never a 404.

        ``/api/certificates/{name}`` 404s when nothing is certified yet, which is
        right for "give me the certificate" and wrong for "give me everything you
        have": an export describes an uncertified derivation completely, just
        without one. Reimplements the certify step rather than calling the
        conditionally-defined ``_build_certificate_document`` above, so it degrades
        instead of raising.
        """
        if certificate_issuer is None or store is None:
            return None  # certificates need the 'certificate' extra
        attestation = json.loads(row.attestation_json) if row.attestation_json else {}
        if not (row.verdict or attestation.get("verdict")):
            return None  # nothing verified yet to certify
        run = _certificate_run(
            row, store.latest_run_for_derivation(row.name), attestation
        )
        skipped = tuple((str(s), "") for s in attestation.get("skipped", []))
        return certificate_issuer.issue(run, skipped=skipped)

    async def _derivation_export_document(row: Derivation) -> dict[str, Any]:
        """One derivation's auditable record.

        Source, claim, verdict, attestation, result history, and certificate -- the
        data-portability answer for it. Every field is read from what was recorded;
        nothing here executes the derivation. The detail route renders a repo derivation
        on view (``render_derivation`` runs it against bound data), and inheriting that
        would make exporting a workspace run it -- and for an export whose job is to
        hand over evidence, an output computed at export time is not evidence anyway. A
        chat-authored derivation's stored ``rendered`` still travels.
        """
        assert store is not None  # noqa: S101 - only called once a row is in hand
        detail = _derivation_detail(row)
        if _withholds(row.name):
            detail.rendered = None  # matches the detail route
        history = with_changes(store.runs_for_derivation(row.name))
        return {
            # Dumped by field name, so the document stays snake_case.
            **detail.model_dump(),
            "history": [
                {**_run_dict(run), "changed": list(changed)} for run, changed in history
            ],
            "certificate": _export_certificate(row),
            # After the spread, not before: _derivation_summary carries a `created_at`
            # of its own (the derivation's), and a key later in the literal wins. Ahead
            # of it, the envelope's timestamp was silently overwritten on every export.
            "schema": "elbi.export/v1",
            "kind": "derivation",
            "created_at": _iso_utc(datetime.now(timezone.utc)),
        }

    @app.get("/api/exports/derivations/{name}")
    async def export_derivation(name: str, request: Request) -> Response:
        """A derivation's complete auditable record, for data portability."""
        row = store.get_derivation(name) if store else None
        if row is None:
            raise HTTPException(status_code=404, detail=f"no derivation named {name!r}")
        document = await _derivation_export_document(row)
        return _download_json(document, f"{name}-record")

    def _dashboard_record_document(dashboard_id: str) -> dict[str, Any]:
        """One dashboard's definition and saved versions -- nothing resolved.

        What a dashboard *records*: its spec, and every version of that spec. Both are
        plain reads; resolving a page instead would run its widgets.

        Raises ``DashboardError`` for an unknown or unreachable id, exactly as
        ``DashboardService.get`` does -- the route below turns that into a 404.
        """
        service = _dashboards()
        return {
            **service.get(dashboard_id),
            "versions": service.versions(dashboard_id),
            "schema": "elbi.export/v1",
            "kind": "dashboard",
            "created_at": _iso_utc(datetime.now(timezone.utc)),
        }

    async def _dashboard_export_document(dashboard_id: str) -> dict[str, Any]:
        """The record above, plus every page's current resolved values.

        The values are the expensive half and the reason this one is async: a dashboard
        has no equivalent of a derivation's run history, so resolving its widgets is the
        only evidence it can produce of what it actually showed. That is worth the cost
        for someone exporting *a dashboard*, and not for someone exporting *a person*.
        """
        document = _dashboard_record_document(dashboard_id)
        published = document["status"] == "published"
        spec_source = (
            document["published_spec"] if published else document["spec"]
        ) or document["spec"]
        values: dict[str, list[dict[str, Any]]] = {}
        for page in spec_source.get("pages", []):
            page_name = str(page["name"])
            values[page_name] = await run_in_threadpool(
                _dashboards().resolve,
                dashboard_id,
                page_name,
                {},
                published=published,
            )
        return {**document, "values": values}

    @app.get("/api/exports/dashboards/{dashboard_id}")
    async def export_dashboard(request: Request, dashboard_id: str) -> Response:
        """A dashboard's definition, saved versions, and current values."""
        try:
            document = await _dashboard_export_document(dashboard_id)
        except DashboardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # The URL carries an opaque id; the document carries the name a person knows it
        # by, and that is what the saved file should be called.
        stem = str(document.get("name") or dashboard_id)
        return _download_json(document, f"{stem}-record")

    def _metric_export_document(name: str) -> dict[str, Any] | None:
        """One metric's manifest plus its full definition history, or ``None``."""
        service = _metrics()
        view = service.get(name)
        if view is None:
            return None
        return {
            "schema": "elbi.export/v1",
            "kind": "metric",
            "created_at": _iso_utc(datetime.now(timezone.utc)),
            **view,
            "versions": service.history(name),
        }

    @app.get("/api/exports/metrics/{name}")
    async def export_metric(request: Request, name: str) -> Response:
        """A metric's definition plus its version history."""
        document = _metric_export_document(name)
        if document is None:
            raise HTTPException(status_code=404, detail="metric not found")
        return _download_json(document, f"{name}-record")

    if reload_project is not None:

        @app.post("/api/project/reload")
        async def reload_project_route() -> dict[str, Any]:
            """Re-read derivation source files and source data from disk (no restart).

            The server side of ``elbi sync`` for the parts that are code and
            data rather than declarative config: a change to a ``derivations/*.py`` or
            to a declared source's data takes effect in the running app.
            """
            return reload_project()

    # Observability: expose Prometheus metrics at /internal/metrics and, when
    # configured, OTLP tracing. The scrape path is deliberately NOT /metrics: that URL
    # is the Metrics tab's SPA route, and a scrape endpoint there would shadow the app
    # on a hard refresh. Added before the SPA catch-all mount either way.
    _register_build_info()
    Instrumentator().instrument(app).expose(
        app, endpoint="/internal/metrics", include_in_schema=False
    )
    setup_tracing(app)

    # Installed before the routes below are mounted, so an extension can register its
    # own paths, and before the single-page app's catch-all, which would shadow them.
    added = extensions.install_all(
        extensions.ExtensionContext(
            app=app,
            store=store,
            warehouse=warehouse_service,
            notebooks=notebook_service,
            metrics=metric_service,
            monitors=monitor_service,
        )
    )
    if added:
        logger.info("extensions installed: %s", ", ".join(added))

    if mcp_app is not None:
        # A client configured with the advertised URL posts to `/mcp` exactly, and the
        # mount below serves only paths *under* it. Starlette would normally redirect
        # the bare path itself, but the single-page app's catch-all is mounted at the
        # root and matches first -- and it serves files, so it answers a POST with 405.
        # 307 rather than 301/302 because only 307 obliges the client to repeat the
        # POST with its body intact.
        @app.api_route(
            "/mcp", methods=["GET", "POST", "DELETE"], include_in_schema=False
        )
        async def _mcp_root(request: Request) -> Response:
            query = request.url.query
            return RedirectResponse(
                f"/mcp/?{query}" if query else "/mcp/", status_code=307
            )

        app.mount("/mcp", mcp_app)
    if model_service is not None:
        # The real MLflow UI (experiments, run charts, registry, comparisons) on
        # the same store the platform trains into; the Models page links into it.
        mount_mlflow_ui(app, model_service)
    if static_dir is not None and static_dir.is_dir():
        # Mounted last so it does not shadow the API routes; index.html fallback so
        # client-side routes (e.g. /derivations) resolve on a direct load.
        app.mount("/", _SPAStaticFiles(directory=static_dir, html=True), name="spa")

    return app


def _required(body: Any, name: str) -> Any:
    """A required field from a JSON body, or a 400 naming what is missing.

    Indexing the body directly turns an absent field into a ``KeyError`` and a 500,
    which tells the caller only that something broke. The status is 400 rather than
    422 because 422 is what FastAPI raises from its own schema validation; an error
    raised here is this app's, and keeping the two apart lets a client tell them apart.
    """
    try:
        return body[name]
    except (KeyError, TypeError):
        raise HTTPException(status_code=400, detail=f"a {name!r} is required") from None


def _iso_utc(dt: datetime) -> str:
    """ISO 8601 with an explicit UTC offset.

    Timestamps are written UTC-aware, but SQLite returns them naive on read, so a bare
    ``isoformat()`` omits the zone and a browser parses the value as *local* time:
    shifting every conversation into the future, which the sidebar's relative clock
    clamps to "just now". Stamp a naive value as UTC so the client reads the instant it
    was actually recorded.
    """
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _conversation_document(
    conversation: Conversation, messages: list[Message]
) -> dict[str, Any]:
    """One conversation as a self-contained JSON document.

    Defined here so a conversation's exported shape lives in exactly one place.
    """
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": _iso_utc(conversation.created_at),
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tools": json.loads(m.tools_json),
                "result": json.loads(m.result_json) if m.result_json else None,
                "created_at": _iso_utc(m.created_at),
            }
            for m in messages
        ],
    }


def _data(chunk: dict[str, Any]) -> dict[str, str]:
    """Render one AI SDK UI Message Stream chunk as an SSE ``data:`` frame."""
    return {"data": json.dumps(chunk)}


def _question_from(body: dict[str, Any]) -> str:
    """The question: the last user message's text in an AI SDK `useChat` payload.

    Falls back to a top-level ``question`` field so the endpoint is still callable
    directly (tests, scripts) without constructing a message list.
    """
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not (isinstance(message, dict) and message.get("role") == "user"):
                continue
            parts = message.get("parts")
            if isinstance(parts, list):
                text = " ".join(
                    str(p.get("text", ""))
                    for p in parts
                    if isinstance(p, dict) and p.get("type") == "text"
                )
                return text.strip()
    return str(body.get("question", "")).strip()


def _flush_reasoning(trace: list[dict[str, Any]], buffer: str) -> str:
    """Fold accumulated streamed reasoning into the trace as one item; return empty."""
    if buffer.strip():
        trace.append({"kind": "reasoning", "text": buffer})
    return ""


def _result_payload(result: AnswerResult | None) -> dict[str, Any]:
    """The wire form of a final result, including the verification material."""
    if result is None:  # pragma: no cover - a result event always carries a result
        return _result_payload(_empty("no result"))
    return {
        "verified": result.verified,
        "verdict": result.verdict,
        "narrative": result.narrative,
        "checks": [
            {"name": name, "verdict": verdict, "detail": detail}
            for name, verdict, detail in result.checks
        ],
        "assumptions": list(result.assumptions),
        "spec": result.spec,
        "data_hash": result.data_hash,
        "steps": result.steps,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "cost": result.usage.cost,
        },
    }


def _empty(reason: str) -> AnswerResult:
    return AnswerResult(verified=False, verdict="unverified", narrative=reason, steps=0)


def _stopped() -> AnswerResult:
    """The record for a turn the user stopped or that lost its connection mid-stream.

    Persisting it means the conversation shows the turn ended without an answer, instead
    of orphaning the user's question with no reply.
    """
    return AnswerResult(
        verified=False,
        verdict="stopped",
        narrative="This analysis was stopped before it finished.",
        steps=0,
    )


def _generate_title_async(
    store: Store,
    conversation_id: str,
    question: str,
    generate_title: Callable[[str], str | None],
) -> None:
    """Generate a conversation's title in a daemon thread, off the request path.

    Best-effort: a failure (or a ``None`` result) is logged and the plain truncated-
    question title is kept, so titling never blocks a turn or breaks a conversation.
    """

    def work() -> None:
        try:
            title = generate_title(question)
        except Exception:
            logger.exception("title generation failed for %s", conversation_id)
            return
        if title:
            with suppress(Exception):
                store.rename_conversation(conversation_id, title)

    threading.Thread(target=work, daemon=True).start()


#: Fallback default when the model attaches no chart of its own: a Vega-Lite spec built
#: from the claim's declared roles (a map for coordinates, a scatter for an x-y effect).
_VIZ_FALLBACKS: tuple[tuple[dict[str, tuple[str, ...]], str], ...] = (
    (
        {"latitude": ("latitude", "lat"), "longitude": ("longitude", "long", "lon")},
        "geo",
    ),
    ({"x": ("x",), "y": ("y",)}, "point"),
)


def _viz_for(result: AnswerResult) -> dict[str, Any] | None:
    """The chart to render for a verified result, as {spec, rows}.

    The model's own chart wins: a full Vega-Lite spec it declared on ``answer`` and the
    runtime already validated (it references only certified columns). Absent one, fall
    back to a minimal spec built from the claim's roles. Either way we inject the
    certified rows as the data downstream, so the chart is always a view of exactly what
    the oracle checked: any Vega-Lite visualization, no fixed menu of chart types.
    """
    if not result.verified or not result.rows:
        return None
    rows = [dict(r) for r in result.rows]
    if isinstance(result.viz, dict) and result.viz:  # the model's validated Vega-Lite
        return {"spec": result.viz, "rows": rows}
    columns = set(result.rows[0])
    claim = (result.spec or {}).get("claim") or {}

    def declared(candidates: tuple[str, ...]) -> str | None:
        for role in candidates:
            col = claim.get(role)
            if isinstance(col, str) and col in columns:
                return col
        return None

    for channels, kind in _VIZ_FALLBACKS:
        resolved = {ch: declared(roles) for ch, roles in channels.items()}
        if all(resolved.values()):
            encoding = {ch: {"field": col} for ch, col in resolved.items()}
            if kind == "geo" and (value := declared(("value", "target", "metric"))):
                encoding["color"] = {"field": value, "type": "quantitative"}
            spec: dict[str, Any] = {"mark": "point", "encoding": encoding}
            return {"spec": spec, "rows": rows}
    return None


def _title(question: str) -> str:
    """A short conversation title from the opening question."""
    trimmed = question.strip().splitlines()[0] if question.strip() else "analysis"
    return trimmed[:60] + ("…" if len(trimmed) > 60 else "")


def _summary_fields(d: Derivation) -> dict[str, Any]:
    """The fields the list and detail forms share."""
    return {
        "name": d.name,
        "question": d.question,
        "verdict": d.verdict,
        "data_hash": d.data_hash,
        "created_at": _iso_utc(d.created_at),
        "origin": d.origin,
    }


def _derivation_summary(d: Derivation) -> WireDerivationSummary:
    """The list form of a derivation, whether repo- or chat-authored."""
    return WireDerivationSummary(**_summary_fields(d))


class MetricQuestionError(Exception):
    """A natural-language metric question could not be resolved to a valid query."""


_ASK_SYSTEM = (
    "You translate a question into a query over a fixed catalog of governed metrics. "
    "You never write SQL. Choose exactly ONE metric and only dimensions and filter "
    "columns the catalog lists for that metric. Reply with ONLY a JSON object:\n"
    '{"metric_name": string|null, "group_by": [string], '
    '"grain": "hour"|"day"|"week"|"month"|"quarter"|"year"|null, '
    '"filters": [{"column": string, "op": "eq"|"ne"|"lt"|"le"|"gt"|"ge"|"in", '
    '"value": any}], "explanation": string}\n'
    'If the question cannot be answered from the catalog, reply {"metric_name": null}.'
)


def _metric_dimensions(view: dict[str, Any]) -> list[str]:
    """A metric view's groupable columns: its dimensions plus any time dimension."""
    dims = [str(d) for d in (view.get("dimensions") or [])]
    time_dimension = view.get("timeDimension")
    if isinstance(time_dimension, dict) and time_dimension.get("column"):
        dims.append(str(time_dimension["column"]))
    return dims


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text``, tolerating Markdown code fences."""
    cleaned = (
        text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    )
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise MetricQuestionError("the model did not return a JSON query")
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise MetricQuestionError(f"the model returned invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise MetricQuestionError("the model did not return a JSON object")
    return obj


def _resolve_metric_question(
    question: str, views: list[dict[str, Any]], client: LLMClient
) -> dict[str, Any]:
    """Resolve an NL question to a validated metric query (text-to-semantic-query).

    Grounds the model in the catalog, then validates its choice against the chosen
    metric's declared dimensions, grain, and filter operators, so an invalid field is
    rejected (and the caller declines) rather than compiled into a wrong-but-plausible
    answer. The model picks a metric; the deterministic resolver runs it.
    """
    from elbi_core.metrics import FILTER_OPS, GRAINS

    catalog = {
        v["name"]: {
            "description": v.get("description") or v.get("label") or "",
            "dimensions": _metric_dimensions(v),
        }
        for v in views
    }
    transcript = Transcript(system=_ASK_SYSTEM)
    transcript.add_user_text(
        f"Catalog (metric -> description, dimensions):\n{json.dumps(catalog, indent=2)}"
        f"\n\nQuestion: {question}"
    )
    try:
        step = client.step(transcript, ())
    except Exception as exc:  # any provider/transport error, surfaced to the caller
        raise MetricQuestionError(f"the model could not be reached: {exc}") from exc

    plan = _parse_json_object(step.text or "")
    name = plan.get("metric_name")
    if not name or name not in catalog:
        raise MetricQuestionError(
            "that question does not map to a defined metric: try naming one directly."
        )
    allowed = set(catalog[name]["dimensions"])
    group_by = [str(d) for d in (plan.get("group_by") or [])]
    unknown = [d for d in group_by if d not in allowed]
    if unknown:
        raise MetricQuestionError(
            f"metric {name!r} has no dimension(s) {unknown}; "
            f"available: {sorted(allowed) or '(none)'}"
        )
    grain = plan.get("grain")
    if grain is not None and grain not in GRAINS:
        raise MetricQuestionError(f"unknown grain {grain!r}")
    filters: list[dict[str, Any]] = []
    for clause in plan.get("filters") or []:
        column = str(clause.get("column", ""))
        op = str(clause.get("op", "eq"))
        if column not in allowed:
            raise MetricQuestionError(f"cannot filter {name!r} on {column!r}")
        if op not in FILTER_OPS:
            raise MetricQuestionError(f"unknown filter operator {op!r}")
        filters.append({"column": column, "op": op, "value": clause.get("value")})
    return {
        "metric_name": name,
        "group_by": group_by,
        "grain": grain,
        "filters": filters,
        "explanation": str(plan.get("explanation") or ""),
    }


def _derivation_detail(d: Derivation) -> WireDerivationDetail:
    """The full form of an authored derivation, for inspection or reuse."""
    return WireDerivationDetail(
        **_summary_fields(d),
        conversation_id=d.conversation_id,
        source=d.source,
        claim=json.loads(d.claim_json) if d.claim_json else None,
        serve=json.loads(d.serve_json) if d.serve_json else None,
        narrative=d.narrative,
        rendered=d.rendered,
        attestation=(
            json.loads(d.attestation_json) or None if d.attestation_json else None
        ),
        assumptions=json.loads(d.assumptions_json) if d.assumptions_json else [],
    )


def _run_dict(run: CertifiedRun) -> dict[str, Any]:
    """The API form of a certified version: its record plus display-ready fields."""
    return {
        **run.to_dict(),
        "short_version": run.short_version,
        "metrics": run.metrics(),
    }


def _safe_filename_part(name: str) -> str:
    """``name`` with anything but word characters, dots, and hyphens replaced.

    A derivation name has no charset constraint at creation time and is embedded
    directly in a Content-Disposition header; a quote or control character would
    otherwise let it break out of the quoted filename or inject header content.
    """
    return re.sub(r"[^\w.-]", "_", name)


def _download_json(document: dict[str, Any], stem: str) -> Response:
    """One export document as a downloadable file: indented, sorted, and named.

    Compact JSON is right for an API a client parses. These routes serve attachments a
    person opens, so they match what the CLI archive's ``_write_record`` writes instead:
    the same record is the same bytes whichever way it was fetched.
    """
    return Response(
        # The trailing newline is what makes "same bytes either way" exact: the CLI
        # archive's _write_record ends every record file with one.
        content=json.dumps(document, indent=2, sort_keys=True) + "\n",
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{_safe_filename_part(stem)}.json"'
            )
        },
    )


def _certificate_run(
    row: Derivation, latest: CertifiedRun | None, attestation: dict[str, Any]
) -> CertifiedRun:
    """The certified run to certify: the newest recorded run, or one from the row.

    A chat-authored derivation has a recorded run carrying the content-addressed
    version identity; a repo derivation may have only the stored attestation, so fall
    back to a run built from the row with empty version components.
    """
    if latest is not None:
        return latest
    return CertifiedRun(
        name=row.name,
        derivation_version="",
        verdict=row.verdict or str(attestation.get("verdict") or "unverified"),
        created_at=_iso_utc(row.created_at),
        data_hash=row.data_hash,
        estimate=attestation.get("estimate"),
        estimate_label=attestation.get("estimate_label"),
        adjusted_for=tuple(attestation.get("adjusted_for") or ()),
        claim=json.loads(row.claim_json) if row.claim_json else {},
        checks=tuple(
            (c["name"], c["verdict"], c["detail"])
            for c in attestation.get("checks", [])
        ),
        question=row.question,
    )
