"""Native persistence: conversations, messages, authored derivations, and config.

One set of SQLModel models runs on either backend through SQLAlchemy, chosen by a
``DB_URI`` in nao's ``dbConfig`` style: SQLite by default (``sqlite:./elbi.db``:
embedded, zero-config, local-first) or Postgres when deployed (``postgres://…``, which
needs the ``postgres`` extra). Answers are authored derivations, so they live here as
durable, cached records rather than as ephemeral chat turns; this store is the app's own
surface for them, not an MCP registration.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import (
    Connection,
    Engine,
    MetaData,
    Table,
    UniqueConstraint,
    event,
    func,
    insert,
    inspect,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, col, create_engine, select

from elbi_core import Job, JobState
from elbi_core.tracking import CertifiedRun

from .resources import ResourceType

logger = logging.getLogger("elbi")

DEFAULT_DB_URI = "sqlite:./elbi.db"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex


def _as_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime (SQLite drops tzinfo), for safe comparison."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _trash_entry(kind: str, item_id: str, name: str, row: Any) -> dict[str, Any]:
    """One trash listing row, the same shape regardless of the artifact kind.

    ``row`` only needs ``deleted_at``, which every trashable model has, so this works
    across the seven unrelated table classes without them sharing a base class.
    """
    return {
        "type": kind,
        "id": item_id,
        "name": name,
        "deleted_at": _as_utc(row.deleted_at).isoformat(),
    }


def _workflow_view(row: Workflow) -> dict[str, Any]:
    """Serialize a workflow definition for the API."""
    return {
        "id": row.id,
        "name": row.name,
        "steps": json.loads(row.steps_json or "[]"),
    }


def _asset_check_view(row: AssetCheck) -> dict[str, Any]:
    """Serialize a data-quality check for the API."""
    return {
        "id": row.id,
        "asset": row.asset,
        "name": row.name,
        "expr": row.expr,
        "severity": row.severity,
        "enabled": row.enabled,
    }


#: Budget window units in seconds for the ``<n>h``/``<n>d``/``<n>mo`` grammar.
_WINDOW_UNITS = {"h": 3600, "d": 86400, "mo": 30 * 86400}


def _window_seconds(window: str) -> float:
    """Parse a window (``"24h"``, ``"30d"``, ``"1mo"``) to seconds; default 30d."""
    import re

    match = re.fullmatch(r"\s*(\d+)\s*(h|d|mo)\s*", window or "")
    if match is None:
        return 30 * 86400.0
    return int(match.group(1)) * _WINDOW_UNITS[match.group(2)]


def _window_elapsed(budget: Budget) -> bool:
    """Whether a budget's rolling window has elapsed (so its spend resets)."""
    start = budget.window_start
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return (_now() - start).total_seconds() >= _window_seconds(budget.window)


def sqlalchemy_url(db_uri: str) -> str:
    """Translate a nao-style ``DB_URI`` into a SQLAlchemy URL.

    ``sqlite:./x.db`` becomes ``sqlite:///./x.db``; ``postgres://…`` becomes
    ``postgresql+psycopg://…`` (the psycopg v3 driver); a bare path is read as a
    SQLite file so an unprefixed value still works.
    """
    if db_uri.startswith(("postgres://", "postgresql://")):
        _, rest = db_uri.split("://", 1)
        return f"postgresql+psycopg://{rest}"
    if db_uri.startswith("sqlite:"):
        return "sqlite:///" + db_uri[len("sqlite:") :]
    return "sqlite:///" + db_uri


class Conversation(SQLModel, table=True):
    """One chat session; its questions and answers hang off it as messages."""

    id: str = Field(default_factory=_new_id, primary_key=True)
    title: str
    created_at: datetime = Field(default_factory=_now)
    # Bumped on each new message, so the history can order by recent activity (a chat
    # you just replied to rises to the top) rather than by creation time.
    updated_at: datetime = Field(default_factory=_now)
    # The user who owns this conversation, or None in single-user/unauthenticated mode.
    # Stamped from the request identity; queries scope by it once auth is enabled.

    # The LLM profile this conversation runs on (a named model config), or None to use
    # the default profile. Lets one chat use, say, OpenAI while another uses Claude.
    profile: str | None = Field(default=None)
    # Token counts and cost accumulated over the conversation's turns, so the UI can
    # show what a chat spent. Nullable for rows created before the columns existed.
    prompt_tokens: int | None = Field(default=0)
    completion_tokens: int | None = Field(default=0)
    cost: float | None = Field(default=0.0)
    # A cached summary of the conversation's older turns, and how many turns it covers,
    # so a long chat replays a short summary plus the recent window rather than every
    # message. Recomputed only when the number of older turns changes.
    summary: str | None = Field(default=None)
    summary_turns: int | None = Field(default=0)


class Message(SQLModel, table=True):
    """A single turn: the user's question or the assistant's verified answer.

    ``tools_json`` and ``result_json`` carry the agent's tool trace and the verified
    result payload as JSON so a conversation reloads exactly as it streamed.
    """

    id: str = Field(default_factory=_new_id, primary_key=True)
    conversation_id: str = Field(index=True, foreign_key="conversation.id")
    role: str
    content: str = ""
    tools_json: str = "[]"
    result_json: str | None = None
    # A thumbs rating on an assistant turn: "up", "down", or None (unrated / cleared).
    feedback: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)


class Derivation(SQLModel, table=True):
    """A chat-authored derivation: the durable, certified answer to a question.

    The compute ``source``, the ``claim`` it was gated on, the ``rendered`` output,
    and the ``attestation`` are all kept, so the answer is reproducible and auditable
    rather than a one-off chat turn.
    """

    name: str = Field(primary_key=True)
    conversation_id: str | None = Field(default=None, index=True)

    question: str = ""
    source: str = ""
    claim_json: str | None = None
    serve_json: str | None = None
    verdict: str | None = None
    narrative: str = ""
    rendered: str = ""
    attestation_json: str | None = None
    #: Declared, sourced premises the conclusion is conditioned on (JSON list of
    #: strings): the audited assumptions from the user's context, if any.
    assumptions_json: str | None = None
    data_hash: str | None = None
    origin: str = "agent"
    created_at: datetime = Field(default_factory=_now)
    #: Trash stamp, see :class:`Notebook` and :class:`MetricRow` (same name-PK
    #: situation). Only ever set for ``origin != "repo"``: a repo-origin
    #: row is a source file under version control, re-synced from disk on every start
    #: (:func:`Store.sync_repo_derivations`), which would silently clear this stamp, so
    #: trashing a repo derivation is refused rather than accepted and then undone.
    deleted_at: datetime | None = Field(default=None)


class DerivationRun(SQLModel, table=True):
    """One certified version of a derivation: an entry in its result history.

    Append-only history keyed on the content-addressed ``derivation_version``: an
    identical re-derive collapses to the existing row, while a change to the data, the
    code, or the controls appends a new one, so a derivation's rows are its certified
    versions over time. Every estimate here is the oracle's verified number, not a
    self-reported one, which is what makes the history trustworthy. The scalar fields
    are columns (filtered and sorted on); the structured attestation and version
    components ride as JSON, so a diff can attribute a move to code, data, or controls.
    """

    __tablename__ = "derivation_run"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = Field(index=True)
    derivation_version: str = Field(index=True)
    verdict: str
    estimate: float | None = None
    estimate_label: str | None = None
    data_hash: str | None = None
    adjusted_for_json: str | None = None
    claim_json: str | None = None
    params_json: str | None = None
    deps_json: str | None = None
    checks_json: str | None = None
    code_version: str | None = None
    input_versions_json: str | None = None
    question: str = ""
    conversation_id: str | None = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=_now, index=True)


def _run_to_row(run: CertifiedRun) -> DerivationRun:
    """Map a :class:`CertifiedRun` to its persisted row (tuples/dicts become JSON)."""
    return DerivationRun(
        name=run.name,
        derivation_version=run.derivation_version,
        verdict=run.verdict,
        estimate=run.estimate,
        estimate_label=run.estimate_label,
        data_hash=run.data_hash,
        adjusted_for_json=json.dumps(list(run.adjusted_for)),
        claim_json=json.dumps(run.claim),
        params_json=json.dumps(run.params),
        deps_json=json.dumps(list(run.deps)),
        checks_json=json.dumps([list(c) for c in run.checks]),
        code_version=run.code_version,
        input_versions_json=json.dumps(run.input_versions),
        question=run.question,
        conversation_id=run.conversation_id,
        created_at=_as_utc(datetime.fromisoformat(run.created_at)),
    )


def _row_to_run(row: DerivationRun) -> CertifiedRun:
    """Rebuild a :class:`CertifiedRun` from its persisted row (JSON becomes tuples)."""
    return CertifiedRun(
        name=row.name,
        derivation_version=row.derivation_version,
        verdict=row.verdict,
        created_at=_as_utc(row.created_at).isoformat(),
        data_hash=row.data_hash,
        estimate=row.estimate,
        estimate_label=row.estimate_label,
        adjusted_for=tuple(json.loads(row.adjusted_for_json or "[]")),
        claim=json.loads(row.claim_json or "{}"),
        params=json.loads(row.params_json or "{}"),
        deps=tuple(json.loads(row.deps_json or "[]")),
        checks=tuple(tuple(c) for c in json.loads(row.checks_json or "[]")),
        code_version=row.code_version,
        input_versions=json.loads(row.input_versions_json or "{}"),
        question=row.question,
        conversation_id=row.conversation_id,
    )


class Notebook(SQLModel, table=True):
    """A notebook: the human authoring surface whose cells run against a live kernel.

    The document (its cells) hangs off this row as :class:`NotebookCell` records,
    ordered
    by ``position``. ``deps_json`` is the kernel's fixed dependency set (PEP 508
    strings),
    ``metadata_json`` carries nbformat metadata plus app settings (whether reactive
    execution is on), and ``schedule_json`` holds the parameterized-rerun policy,
    mirroring
    a model's retraining policy in shape.
    """

    __tablename__ = "notebook"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = "Untitled notebook"

    # The containing folder (``None`` == the tree's root). An adjacency-list edge, the
    # same shape as :class:`NotebookFolder.parent_id`, so the tree is one join deep.
    folder_id: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    deps_json: str = "[]"
    metadata_json: str = "{}"
    schedule_json: str | None = None
    # The resolved, pinned environment (a uv-compiled lockfile as a JSON list of
    # ``name==version`` specs) for the notebook's effective dependencies. Provisioned
    # from in preference to the loose ``deps_json`` so a rerun gets the same versions.
    lock_json: str | None = None
    # Which compute profile this notebook runs on. A property of the notebook rather
    # than of whoever opens it, so two collaborators do not get two different machines.
    # None means the deployment's default.
    compute_profile: str | None = None
    #: When this was moved to trash, or None while it is live. A timestamp rather
    #: than a flag because the retention sweep needs to know how long it has been
    #: there, and the trash listing shows it. Every read path filters on this being
    #: NULL, so a trashed row is invisible to lists, detail reads, lineage, and the
    #: search index.
    deleted_at: datetime | None = Field(default=None)
    #: Who moved it to trash, for the trash listing. Not a foreign key, the same choice
    #:.
    # The notebook this was duplicated from. A record of where the content came from,
    # not a dependency: the source can be deleted, so never dereference it blind.
    copied_from: str | None = None


class ComputeUsage(SQLModel, table=True):
    """One finished compute session, attributed when it ends.

    Written from day one rather than when someone asks for a cost report, because usage
    history cannot be backfilled: an attribution missing from a past session stays
    missing. ``profile_version`` records the definition the session actually ran under,
    so a later edit to the profile does not silently rewrite what was charged.
    """

    __tablename__ = "compute_usage"

    id: str = Field(default_factory=_new_id, primary_key=True)
    ended_at: datetime = Field(default_factory=_now, index=True)
    # ``interactive`` or ``batch``. Databricks bills these to different SKUs and does
    # "not recommend" running production jobs on all-purpose compute; without the
    # distinction a scheduled 3am rerun and an abandoned session look identical in a
    # cost report, and only one of them is worth acting on.
    kind: str = Field(default="interactive", index=True)
    notebook_id: str = Field(index=True)
    profile: str = Field(index=True)
    profile_version: str = ""
    seconds: float = 0.0
    cost: float = 0.0


class NotebookCell(SQLModel, table=True):
    """One cell of a notebook, in nbformat shape, ordered within its notebook.

    ``outputs_json`` is the cell's last rendered outputs (the four nbformat output
    shapes)
    so a reopened notebook shows its results without re-running; ``execution_count`` is
    the
    kernel counter at its last run (``None`` when never run or cleared).
    """

    __tablename__ = "notebook_cell"

    id: str = Field(primary_key=True)
    notebook_id: str = Field(index=True, foreign_key="notebook.id")
    position: int = 0
    cell_type: str = "code"
    source: str = ""
    metadata_json: str = "{}"
    outputs_json: str = "[]"
    execution_count: int | None = None


class NotebookFolder(SQLModel, table=True):
    """A folder in the notebook workspace tree: the unit of organization.

    An adjacency-list node. ``parent_id`` points at the containing folder, or is
    ``None`` for a folder that sits at the tree's root; notebooks hang off a folder
    through :attr:`Notebook.folder_id`. Kept deliberately flat (one parent edge, no
    materialized path) so a move is a single field write and the tree is rebuilt from
    the folder list on the client.
    """

    __tablename__ = "notebook_folder"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = "New folder"
    parent_id: str | None = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    #: Trash stamp, see :class:`Notebook`. Trashing a folder stamps its whole subtree
    #: with one shared timestamp, which is how restore knows what it took down.
    deleted_at: datetime | None = Field(default=None)


class Dashboard(SQLModel, table=True):
    """A dashboard: a declarative presentation surface over certified derivations.

    The spec (:class:`~elbi.dashboard.DashboardSpec`) is held as a manifest in
    ``spec_json``, the working draft, and, once published, a frozen snapshot in
    ``published_spec_json`` (what viewers see, invisible to edits until re-published).
    Every save appends a :class:`DashboardVersion`, so a dashboard has the same
    diffable, roll-back-able history as code.
    """

    __tablename__ = "dashboard"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = Field(index=True)
    title: str = ""
    status: str = "draft"
    spec_json: str = "{}"
    published_spec_json: str | None = None
    version: int = 1
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    #: Trash stamp, see :class:`Notebook`. Versions and subscriptions are left alone
    #: while trashed, so a restore brings the whole history back with it.
    deleted_at: datetime | None = Field(default=None)
    #: The dashboard this was duplicated from, see :class:`Notebook.copied_from`.
    copied_from: str | None = None


class DashboardVersion(SQLModel, table=True):
    """One saved revision of a dashboard's spec: its dashboards-as-code history.

    Append-only. A save labels the row ``'saved'``; a publish labels it
    ``'published'``. The snapshot is the full manifest, so any prior revision can be
    diffed or restored.
    """

    __tablename__ = "dashboard_version"

    id: str = Field(default_factory=_new_id, primary_key=True)
    dashboard_id: str = Field(index=True, foreign_key="dashboard.id")
    version: int
    spec_json: str
    label: str = "saved"
    author_id: str | None = None
    created_at: datetime = Field(default_factory=_now, index=True)


class DashboardSubscription(SQLModel, table=True):
    """A scheduled delivery of a dashboard snapshot to a set of recipients.

    The scheduler renders the named ``page`` under the saved ``variable_state`` on the
    ``cron`` cadence and delivers it as ``fmt`` (png/pdf/csv) over ``channel``
    (email/webhook). Shaped to mirror a model's retrain policy and a notebook's
    schedule, so one scheduler drives them all.
    """

    __tablename__ = "dashboard_subscription"

    id: str = Field(default_factory=_new_id, primary_key=True)
    dashboard_id: str = Field(index=True, foreign_key="dashboard.id")

    cron: str = ""
    page: str = ""
    recipients_json: str = "[]"
    variable_state_json: str = "{}"
    fmt: str = "png"
    channel: str = "email"
    active: bool = True
    last_run_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)


class FeatureEntity(SQLModel, table=True):
    """A registered entity: a named join key features are keyed by."""

    __tablename__ = "feature_entity"

    name: str = Field(primary_key=True)

    join_key: str = ""
    value_type: str = "string"
    description: str | None = None
    created_at: datetime = Field(default_factory=_now)


class FeatureViewRow(SQLModel, table=True):
    """A registered feature view: a group of features from a certified derivation."""

    __tablename__ = "feature_view"

    name: str = Field(primary_key=True)

    entities_json: str = "[]"
    source: str = ""
    timestamp_field: str | None = None
    ttl_seconds: int | None = None
    features_json: str = "[]"
    description: str | None = None
    #: An optional data contract: the quality bar its feature values must meet.
    contract_json: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    #: Trash stamp, see :class:`Notebook` and :class:`MetricRow` (same name-PK
    #: situation).
    deleted_at: datetime | None = Field(default=None)


class OnlineFeatureRow(SQLModel, table=True):
    """The latest materialized feature values for one entity key of one view."""

    __tablename__ = "online_feature"

    id: str = Field(default_factory=_new_id, primary_key=True)
    feature_view: str = Field(index=True)
    entity_key: str = Field(index=True)
    values_json: str = "{}"
    updated_at: datetime = Field(default_factory=_now)


class FeatureStatisticsRow(SQLModel, table=True):
    """A profile snapshot of a feature view's feature columns at a point in time.

    Every snapshot carries the per-feature profiles and row count; the one flagged
    ``is_baseline`` also retains a capped row sample, the reference a later
    drift check compares against.
    """

    __tablename__ = "feature_statistics"

    id: str = Field(default_factory=_new_id, primary_key=True)
    feature_view: str = Field(index=True)

    at: datetime = Field(default_factory=_now, index=True)
    row_count: int = 0
    profiles_json: str = "[]"
    is_baseline: bool = Field(default=False, index=True)
    sample_json: str = ""


class FeatureDriftRow(SQLModel, table=True):
    """One drift check of a feature view's current values against its baseline."""

    __tablename__ = "feature_drift"

    id: str = Field(default_factory=_new_id, primary_key=True)
    feature_view: str = Field(index=True)

    at: datetime = Field(default_factory=_now, index=True)
    n_columns: int = 0
    n_drifted: int = 0
    share_drifted: float = 0.0
    dataset_drift: bool = False
    n_current_rows: int = 0
    report_json: str = "{}"


class FeatureExpectationRow(SQLModel, table=True):
    """One check of a feature view's values against its attached data contract."""

    __tablename__ = "feature_expectation"

    id: str = Field(default_factory=_new_id, primary_key=True)
    feature_view: str = Field(index=True)

    at: datetime = Field(default_factory=_now, index=True)
    verdict: str = ""
    n_clauses: int = 0
    n_violated: int = 0
    row_count: int = 0
    report_json: str = "{}"


class FeatureTrainingSet(SQLModel, table=True):
    """A materialized, named training set: the frozen output of a point-in-time join.

    Persisting the join output makes it a reproducible training source (like a bound
    dataset), rather than a query that must be re-run identically to reproduce a model.
    """

    __tablename__ = "feature_training_set"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = Field(index=True)

    created_at: datetime = Field(default_factory=_now)
    features_json: str = "[]"
    label: str | None = None
    row_count: int = 0
    rows_json: str = "[]"


class OrchestrationRun(SQLModel, table=True):
    """One materialization run over a set of assets, with its trigger and outcome."""

    __tablename__ = "orchestration_run"

    id: str = Field(default_factory=_new_id, primary_key=True)

    #: What triggered the run: 'manual', 'schedule', 'sensor', or 'retry'.
    cause: str = "manual"
    #: 'running', 'succeeded' (all ok), 'failed' (an asset failed), or 'cancelled'.
    status: str = "running"
    #: The run this one retries from (a retry re-materializes its failed assets +
    #: downstream), or None for an original run: the retry-from-failure audit trail.
    parent_run_id: str | None = Field(default=None, index=True)
    #: The workflow this run executed (cause 'workflow'), or None for a plain run.
    workflow_id: str | None = Field(default=None, index=True)
    #: JSON list of per-step outcomes for a workflow run (id/state), else empty.
    workflow_steps_json: str = "[]"
    started_at: datetime = Field(default_factory=_now, index=True)
    finished_at: datetime | None = None


class AssetRun(SQLModel, table=True):
    """The materialization of one asset within a run."""

    __tablename__ = "asset_run"

    id: str = Field(default_factory=_new_id, primary_key=True)
    run_id: str = Field(index=True, foreign_key="orchestration_run.id")
    asset: str = Field(index=True)
    state: str = "succeeded"
    data_version: str | None = None
    verdict: str | None = None
    error: str | None = None
    attempts: int = 0
    duration_ms: int = 0
    #: The step's captured stdout/stderr, for the run-log view.
    logs: str = ""
    #: JSON list of data-quality check outcomes for this step (name/severity/failed/…).
    checks_json: str = "[]"
    created_at: datetime = Field(default_factory=_now, index=True)


class AssetCheck(SQLModel, table=True):
    """A data-quality check on a derivation's output, evaluated at materialization.

    ``expr`` is a boolean SQL expression over the asset's output columns (evaluated with
    DuckDB, like a Delta Live Tables expectation); ``severity`` is 'warn' (record the
    failure) or 'error' (fail the asset's run step). Keyed to one asset by name.
    """

    __tablename__ = "asset_check"

    id: str = Field(default_factory=_new_id, primary_key=True)

    asset: str = Field(index=True)
    name: str
    expr: str
    #: 'warn' (record failing rows) or 'error' (fail the run step, like expect_or_fail).
    severity: str = "warn"
    enabled: bool = True
    created_at: datetime = Field(default_factory=_now)


class Workflow(SQLModel, table=True):
    """A procedural workflow: a DAG of steps, each a materialization gated by a rule.

    ``steps_json`` is a list of ``{id, selection, assets, includeDownstream, dependsOn,
    runIf}``. Steps run in dependency order; a step whose ``runIf`` trigger rule isn't
    met by its upstreams is *excluded* (skipped), the Databricks/Airflow control-flow
    model: letting a workflow branch on success/failure rather than only fan out assets.
    """

    __tablename__ = "workflow"

    id: str = Field(default_factory=_new_id, primary_key=True)

    name: str
    steps_json: str = "[]"
    created_at: datetime = Field(default_factory=_now)


class MaterializationSchedule(SQLModel, table=True):
    """A schedule that materializes a selection of assets on a cadence or a sensor.

    ``mode`` is 'cron' (fire when the cron expression is due) or 'on_data_change' (fire
    when the ``dataset``'s content hash moves). ``selection`` is 'all', 'stale', or a
    comma-separated list of asset names.
    """

    __tablename__ = "materialization_schedule"

    id: str = Field(default_factory=_new_id, primary_key=True)

    name: str = ""
    selection: str = "stale"
    mode: str = "cron"
    cron: str = ""
    dataset: str | None = None
    enabled: bool = True
    last_run_at: datetime | None = None
    last_data_hash: str | None = None
    created_at: datetime = Field(default_factory=_now)


class Setting(SQLModel, table=True):
    """App/project config as key/value, so settings persist alongside the data."""

    key: str = Field(primary_key=True)
    value: str


class LlmProfile(SQLModel, table=True):
    """A named LLM configuration the user can switch between: model plus credentials.

    Registering several (an OpenAI setup, a Claude setup, a cheap model for titles) lets
    a conversation pick which one it runs on. The ``api_key`` is stored but never
    returned by the API, only whether one is set, the way a settings surface masks a
    secret.
    """

    name: str = Field(primary_key=True)
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    #: Reasoning budget for a thinking model ("low"/"medium"/"high"). Per-profile
    #: because it belongs to the model chosen, not the deployment: a profile on a
    #: reasoning model and one on a cheap model for titles want different answers.
    reasoning_effort: str = ""
    created_at: datetime = Field(default_factory=_now)


class DataSource(SQLModel, table=True):
    """A registered database a derivation can read from, beyond the app's own store.

    The readable DSN parts (host/port/db/user) stay cleartext so connections can be
    listed and filtered, while the password lives in ``secret`` encrypted at rest
    (:mod:`elbi.crypto`). ``secret_env`` is the dbt/nao alternative: the name of an
    environment variable the operator supplies the password through, so nothing
    sensitive is stored. Resolving a source yields a connection URL that flows into the
    runtime's existing ``load_sql``.
    """

    __tablename__ = "data_source"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = Field(index=True, unique=True)  # referenced by dataset bindings
    kind: str  # postgres | mysql | sqlite | mssql
    host: str | None = Field(default=None)
    port: int | None = Field(default=None)
    database: str | None = Field(default=None)
    username: str | None = Field(default=None)
    #: The password, encrypted at rest. None when the secret comes from ``secret_env``.
    secret: str | None = Field(default=None)
    #: Name of an env var holding the password instead of storing it (dbt/nao style).
    secret_env: str | None = Field(default=None)
    #: Non-secret extra config as JSON (sslmode, connect options).
    extra: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


#: Auto-sync cadences a warehouse source can be set to. ``"manual"`` is absent here; it
#: means never auto-sync.
SYNC_INTERVALS: dict[str, timedelta] = {
    "30min": timedelta(minutes=30),
    "1hour": timedelta(hours=1),
    "6hour": timedelta(hours=6),
    "12hour": timedelta(hours=12),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
}


class ExternalDataSource(SQLModel, table=True):
    """A configured data-warehouse connector instance (a Postgres/CSV connection).

    ``config_encrypted`` holds the user's connection values (matching the connector's
    ``SourceConfig`` fields) as an encrypted JSON blob, since it may carry credentials.
    Each source owns one or more :class:`ExternalDataSchema` rows, the tables/endpoints
    selected to sync into the lakehouse.
    """

    __tablename__ = "external_data_source"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = Field(index=True, unique=True)
    source_type: str  # registry key: "sql" | "csv" | "stripe" | ...
    config_encrypted: str
    # Optional human note and a table-name prefix (defaults to the source type).
    # Both are asked once for the whole source rather than per table.
    description: str = ""
    prefix: str = ""
    # How often the maintenance scheduler auto-syncs this source; "manual" = never
    # (on-demand only).
    sync_frequency: str = "day"
    status: str = "idle"  # idle | syncing | error
    last_error: str | None = Field(default=None)
    last_synced_at: datetime | None = Field(default=None)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ExternalDataSchema(SQLModel, table=True):
    """One table/endpoint of an external source, synced into a single warehouse table.

    ``should_sync`` toggles a table on, and
    ``sync_type`` is ``full_refresh`` or ``incremental`` (with ``incremental_field`` the
    cursor column and ``cursor`` its last high-water mark).
    """

    __tablename__ = "external_data_schema"

    id: str = Field(default_factory=_new_id, primary_key=True)
    source_id: str = Field(index=True)
    name: str  # source-side table/endpoint name
    table: str  # warehouse Delta table name it lands in
    should_sync: bool = True
    sync_type: str = "full_refresh"  # full_refresh | incremental
    incremental_field: str | None = Field(default=None)  # chosen cursor column
    # Columns the connector offered as cursor candidates (JSON list), so the UI can let
    # the user switch to incremental without a live reconnect to the source.
    incremental_fields: str = Field(default="[]")
    cursor: str | None = Field(default=None)  # last high-water mark, stringified
    row_count: int | None = Field(default=None)
    status: str = "pending"  # pending | syncing | synced | error
    last_error: str | None = Field(default=None)
    last_synced_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)


class WarehouseColumn(SQLModel, table=True):
    """One column of one synced warehouse table, captured when the table is written.

    Column metadata otherwise lives only in the Delta log, so answering "which datasets
    have a ``customer_id`` column" means opening every table -- fine per page load, not
    per keystroke.

    ``source_id`` is denormalised so a read that needs to know which connection a
    table came from does not join per row to find out.

    One row per column per table, enforced here rather than trusted. Replacement runs
    in a transaction, but that lock is one process's, and every replica backfills at
    boot -- so two can each find no rows and each insert the whole set. The database is
    the only place that sees both.
    """

    __tablename__ = "warehouse_column"
    __table_args__ = (UniqueConstraint("table", "name", name="uq_warehouse_column"),)

    id: str = Field(default_factory=_new_id, primary_key=True)
    table: str = Field(index=True)  # warehouse Delta table the column belongs to
    source_id: str = Field(index=True)  # the connection it was synced from
    name: str = Field(index=True)  # what a column search matches on
    data_type: str = ""  # the Delta type, rendered
    nullable: bool = True
    ordinal: int = 0  # position in the table's schema
    #: The column's comment, read from its Delta field metadata. None for every
    #: connector shipped today, none of which describes its columns.
    description: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)


class SavedQuery(SQLModel, table=True):
    """A saved SQL query from the exploration workbench.

    ``source_id`` names the data source the query runs against, or is ``None`` for a
    query over the project's bound datasets. A saved query is a human exploration
    artifact, never a certified derivation and never gated: promoting one to a
    derivation is a separate, opt-in step.
    """

    __tablename__ = "saved_query"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str
    sql: str
    source_id: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    #: Trash stamp, see :class:`Notebook`.
    deleted_at: datetime | None = Field(default=None)
    #: The saved query this was duplicated from, see :class:`Notebook.copied_from`.
    copied_from: str | None = None


class PromotedQuery(SQLModel, table=True):
    """A query promoted to a certified derivation that reads an external data source.

    A bound-dataset promotion becomes an ordinary agent-authored derivation. An
    external-source promotion cannot run in the sandbox (which has no database access),
    so it is a trusted, human-origin derivation that reads the source in-process through
    ``load_sql``. This record lets that derivation be rebuilt into the registry on
    restart, so it survives like any other artifact.
    """

    __tablename__ = "promoted_query"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = Field(index=True, unique=True)  # a registry-unique derivation name
    source_id: str
    sql: str

    created_at: datetime = Field(default_factory=_now)


class MetricRow(SQLModel, table=True):
    """A semantic-layer metric definition, stored as its spec manifest.

    A metric names an aggregation over a certified derivation (its ``source``) plus the
    dimensions it may be sliced by. The full definition lives in ``manifest_json`` (a
    ``metric`` object from the Metric Spec); ``source`` is kept apart for the certified
    gate and for lineage. The set of all rows forms the project's metric set.
    """

    __tablename__ = "metric"

    name: str = Field(primary_key=True)
    manifest_json: str
    source: str | None = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    #: Trash stamp, see :class:`Notebook`. ``name`` is this row's primary key, so a
    #: trashed row blocks a fresh create with the same name until it is restored or
    #: erased -- ``upsert_metric`` erases a trashed row first rather than
    #: resurrecting it.
    deleted_at: datetime | None = Field(default=None)
    #: The metric this was duplicated from, see :class:`Notebook.copied_from`. Holds a
    #: *name* rather than an id, because a metric has no id: its name is the key.
    copied_from: str | None = None


class MetricVersion(SQLModel, table=True):
    """One version of a metric's definition: its change log.

    Append-only: a new row lands each time the definition's content hash changes (a
    no-op re-save collapses), so the table is the metric's history. Following the
    git-commit model (who/when/snapshot) plus the oracle: ``author`` records whether a
    human or the agent made the change, ``verdict`` the source's certification at that
    version, and ``manifest_json`` the full snapshot (so a rollback or "what did it look
    like then" needs no diff replay). Definitions edited through the UI or the agent,
    not git, are why this is a first-class table rather than commit history.
    """

    __tablename__ = "metric_version"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str = Field(index=True)

    version: int
    manifest_json: str
    definition_hash: str = Field(index=True)
    author: str = "human"  # "human" | "agent"
    verdict: str | None = None
    change_summary: str | None = None
    created_at: datetime = Field(default_factory=_now, index=True)


class MetricMonitor(SQLModel, table=True):
    """A monitor watching a metric or derivation's value over time for anomalies.

    ``target_kind``/``target`` name what is watched; ``config_json`` carries how to read
    its scalar value (a metric's group/grain/filters, or a derivation's measure). The
    detector settings (``method``, ``sensitivity``, ``min_value``/``max_value``,
    ``window``) drive the anomaly test over the snapshot history. ``interval_hours`` and
    ``last_run_at`` pace the scheduled check.
    """

    __tablename__ = "metric_monitor"

    id: str = Field(default_factory=_new_id, primary_key=True)
    name: str
    target_kind: str  # "metric" | "derivation"
    target: str
    config_json: str = "{}"
    method: str = "mad"
    sensitivity: float = 3.0
    min_value: float | None = Field(default=None)
    max_value: float | None = Field(default=None)
    window: int = 30
    interval_hours: float = 1.0
    enabled: bool = True

    last_run_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)


class MetricSnapshot(SQLModel, table=True):
    """One observed value of a monitored target, with its anomaly verdict."""

    __tablename__ = "metric_snapshot"

    id: str = Field(default_factory=_new_id, primary_key=True)
    monitor_id: str = Field(index=True)

    at: datetime = Field(default_factory=_now, index=True)
    value: float
    anomalous: bool = False
    baseline: float | None = Field(default=None)
    lower: float | None = Field(default=None)
    upper: float | None = Field(default=None)
    score: float | None = Field(default=None)
    reason: str = ""
    # The source's oracle verdict at snapshot time: a monitored number rides on a
    # certified metric/derivation, so an alert can say the moved number was verified.
    source_verdict: str | None = Field(default=None)


class MonitorIncident(SQLModel, table=True):
    """An open (or resolved) run of anomalies for a monitor, so one alert fires per run.

    Consecutive anomalous snapshots fold into one incident rather than one alert each;
    ``closed_at`` is set when values return to normal.
    """

    __tablename__ = "monitor_incident"

    id: str = Field(default_factory=_new_id, primary_key=True)
    monitor_id: str = Field(index=True)

    opened_at: datetime = Field(default_factory=_now, index=True)
    closed_at: datetime | None = Field(default=None)
    peak_value: float = 0.0
    peak_score: float | None = Field(default=None)
    reason: str = ""
    snapshots: int = 1


class AuditEvent(SQLModel, table=True):
    """An attributed record of a consequential action, for the compliance trail.

    Who did what, against which object, and, for a certified answer, the verdict and
    data hash, so the trail answers "who ran what, and was it sound".
    """

    id: str = Field(default_factory=_new_id, primary_key=True)
    at: datetime = Field(default_factory=_now, index=True)
    action: str  # e.g. "answer", "data_source.create", "budget.set"
    target_type: str = ""
    target_id: str = ""
    verdict: str | None = None
    data_hash: str | None = None


class Notification(SQLModel, table=True):
    """A durable record of a platform event, with read state.

    The webhook is fire-and-forget to one instance-wide URL, so an event has no way to
    be heard about except by looking. A notification row is the addressed counterpart:
    what happened, and whether it has been seen (``read_at``). ``emailed_at`` records
    the single email attempt so delivery is never repeated.
    """

    # The two queries every open browser polls: the list and the unread count.
    id: str = Field(default_factory=_new_id, primary_key=True)

    at: datetime = Field(default_factory=_now, index=True)
    event_type: str = Field(index=True)  # e.g. "run.failed", "metric.anomaly_detected"
    title: str = ""
    body: str = ""
    target_type: str = ""  # "metric_monitor" | "orchestration_run" | "model_version"
    target_id: str = ""
    verdict: str | None = None  # the source's oracle verdict, where the event has one
    payload_json: str = "{}"  # full event payload, for a detail view
    read_at: datetime | None = Field(default=None, index=True)
    emailed_at: datetime | None = None


class NotificationPreference(SQLModel, table=True):
    """Per-event-type notification switches.

    An absent row means the defaults (in-app on, email per the event type's
    default in ``notifications.EVENT_TYPES``). A stored row holds whatever was
    last explicitly saved, which may equal the defaults: the Settings form
    submits the full matrix, so saving pins every type at the values it showed.
    """

    __tablename__ = "notification_preference"

    event_type: str = Field(primary_key=True)
    in_app: bool = True
    email: bool = False
    updated_at: datetime = Field(default_factory=_now)


class InferenceEvent(SQLModel, table=True):
    """One serving request against a registered model (the inference table).

    The request payload and predictions are stored (capped) so production traffic
    is inspectable and monitorable after the fact: what was sent, what came back,
    how fast, and whether it failed. This is the raw material drift monitoring
    reads; without it, "how is the model doing in production" is unanswerable.
    """

    id: str = Field(default_factory=_new_id, primary_key=True)
    at: datetime = Field(default_factory=_now, index=True)
    model: str = Field(index=True)
    version: int = 0
    n_rows: int = 0
    latency_ms: float = 0.0
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    request_json: str | None = None
    predictions_json: str | None = None


class RetrainPolicy(SQLModel, table=True):
    """Continuous training for one model: when to retrain, and on what.

    Two modes, matching how retraining is actually operated: ``on_data_change``
    retrains when the bound dataset's content hash moves (the platform already
    knows every dataset's content identity), and ``interval`` retrains on a
    fixed cadence. The policy stores the full training spec so the retrain is
    the same job a person would have submitted, not a diverging variant.
    """

    model: str = Field(primary_key=True)
    dataset: str
    source_kind: str = "dataset"  # or "derivation" (a certified feature pipeline)
    target: str
    features_json: str = "[]"
    task: str = "auto"
    engine: str = "flaml"
    time_budget: float = 300.0
    metric: str | None = None
    ensemble: bool = False
    mode: str = "on_data_change"  # or "interval"
    interval_hours: float = 24.0
    time_col: str | None = None
    #: Column naming a row's entity, kept whole across the holdout and out of the
    #: features. Same meaning as AutoGluon's ``groups``.
    groups: str | None = None
    horizon: int | None = None
    enabled: bool = True
    last_data_hash: str | None = None
    last_run_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)


#: Primary key of the one :class:`Budget` row.
BUDGET_ID = "instance"


class Budget(SQLModel, table=True):
    """The instance's spend cap over a rolling window (in-process governance).

    A ``max_budget`` over a rolling window with accumulated ``spend``, enforced
    in-process against our own usage store rather than a separate gateway service. One
    row: ``id`` is fixed so a second cap cannot be written and then disagree with the
    first about what applies.
    """

    id: str = Field(default=BUDGET_ID, primary_key=True)
    max_budget: float = 0.0  # USD; 0 means no cap
    window: str = "30d"  # duration: <n>h | <n>d | <n>mo
    window_start: datetime = Field(default_factory=_now)
    spend: float = 0.0


class Secret(SQLModel, table=True):
    """A named secret (a data-source password, an API token), encrypted at rest.

    One row per secret, with ``value`` encrypted via :mod:`elbi.crypto`.
    """

    id: str = Field(default_factory=_new_id, primary_key=True)

    name: str = Field(index=True)
    value: str = ""  # encrypted at rest
    description: str = ""
    created_at: datetime = Field(default_factory=_now)


class JobRow(SQLModel, table=True):
    """A durable background job, so a long training survives a restart.

    Mirrors :class:`elbi.Job`; ``key`` is the content address the job was
    submitted under (so an identical submission dedupes) and ``result_json`` holds the
    serialized result once the job succeeds. Persisting these is what lets the app come
    back up mid-train and reconcile rather than losing the work.
    """

    id: str = Field(primary_key=True)
    key: str = Field(index=True)
    label: str = ""
    state: str = "queued"
    progress: str = ""
    result_json: str | None = None
    error: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None


class Store:
    """Thin session-per-operation wrapper over the models above."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # Held by every write (here and in the job store) so a background-job write and
        # a request write never hit the single-writer database at once. Reads skip it.
        self._lock = threading.Lock()
        # Dispose the pool when this store becomes unreachable, for the many callers
        # that open a store and simply drop it. Without this the pooled connections are
        # finalized individually, and CPython 3.13 raises "unclosed database" from
        # sqlite3.Connection.__del__ at an arbitrary later moment.
        self._finalizer = weakref.finalize(self, engine.dispose)

    def close(self) -> None:
        """Dispose the connection pool, closing every pooled connection."""
        self._finalizer()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self._engine) as session:
            yield session

    def ping(self) -> bool:
        """Whether the database is reachable (a cheap ``SELECT 1``), for readiness."""
        try:
            with self._session() as session:
                session.connection().execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @contextmanager
    def _write(self) -> Iterator[Session]:
        """A session that holds the shared lock, so writes serialize in-process.

        The lock is shared with the job store, so a background-job write and a request
        write never hit the database at once (SQLite allows one writer). Reads use
        :meth:`_session` without the lock and, under WAL, run alongside a writer.
        """
        with self._lock, Session(self._engine) as session:
            yield session

    # -- the seam an extension queries through -----------------------------------
    #
    # An extension that adds its own surfaces needs the same database, and opening a
    # second engine against it would mean a second connection pool and, on SQLite, a
    # second writer. These hand out this store's own sessions so an extension shares
    # the pool and the write lock rather than competing with them.

    @contextmanager
    def reading(self) -> Iterator[Session]:
        """A read session on this store's engine, for an extension's own queries."""
        with self._session() as session:
            yield session

    @contextmanager
    def writing(self) -> Iterator[Session]:
        """A write session holding this store's lock, so writes stay serialized."""
        with self._write() as session:
            yield session

    # -- conversations & messages ------------------------------------------------
    def create_conversation(
        self, title: str, conversation_id: str | None = None, profile: str | None = None
    ) -> str:
        """Start a conversation and return its id.

        ``conversation_id`` lets the client supply the id (so its stable per-chat id is
        the key the server continues across turns); omitted, one is generated.
        ``profile`` records which LLM profile the conversation runs on. It is
        the authenticated user (``None`` in single-user mode).
        """
        row = (
            Conversation(title=title, profile=profile)
            if conversation_id is None
            else Conversation(id=conversation_id, title=title, profile=profile)
        )
        with self._write() as session:
            session.add(row)
            session.commit()
            return row.id

    def set_conversation_profile(
        self, conversation_id: str, profile: str | None
    ) -> None:
        """Record the LLM profile a conversation runs on (its per-chat model choice)."""
        with self._write() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.profile = profile
                session.add(conversation)
                session.commit()

    def set_summary(self, conversation_id: str, summary: str, turns: int) -> None:
        """Cache a conversation's summary and how many older turns it covers."""
        with self._write() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.summary = summary
                conversation.summary_turns = turns
                session.add(conversation)
                session.commit()

    def add_usage(
        self, conversation_id: str, prompt: int, completion: int, cost: float
    ) -> None:
        """Add a turn's token counts and cost to a conversation's running totals."""
        with self._write() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.prompt_tokens = (conversation.prompt_tokens or 0) + prompt
                conversation.completion_tokens = (
                    conversation.completion_tokens or 0
                ) + completion
                conversation.cost = (conversation.cost or 0.0) + cost
                session.add(conversation)
                session.commit()

    # -- audit -------------------------------------------------------------------
    def record_audit(
        self,
        action: str,
        *,
        target_type: str = "",
        target_id: str = "",
        verdict: str | None = None,
        data_hash: str | None = None,
    ) -> None:
        """Append an audit event (best-effort; never blocks the action it records)."""
        with self._write() as session:
            session.add(
                AuditEvent(
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    verdict=verdict,
                    data_hash=data_hash,
                )
            )
            session.commit()

    def list_audit(self, limit: int = 100) -> list[AuditEvent]:
        """Recent audit events, newest first, scoped to one actor."""
        with self._session() as session:
            stmt = (
                select(AuditEvent)
                .order_by(AuditEvent.at.desc())  # type: ignore[attr-defined]
                .limit(limit)
            )
            return list(session.exec(stmt))

    # -- notifications -------------------------------------------------------------

    def create_notifications(self, rows: Sequence[Notification]) -> None:
        """Insert notification rows in one write (a multi-recipient fan-out)."""
        if not rows:
            return
        with self._write() as session:
            for row in rows:
                session.add(row)
            session.commit()

    def list_notifications(
        self, limit: int = 50, unread_only: bool = False
    ) -> list[Notification]:
        """Notifications, newest first."""
        with self._session() as session:
            stmt = select(Notification)
            if unread_only:
                stmt = stmt.where(col(Notification.read_at).is_(None))
            stmt = stmt.order_by(col(Notification.at).desc()).limit(limit)
            return list(session.exec(stmt))

    def unread_notification_count(self) -> int:
        """How many of this user's notifications are unread."""
        with self._session() as session:
            count = session.exec(
                select(func.count(col(Notification.id))).where(
                    col(Notification.read_at).is_(None)
                )
            ).one()
            return int(count or 0)

    def notification_inbox(
        self, limit: int = 50, unread_only: bool = False
    ) -> tuple[list[Notification], int]:
        """One user's notifications and their unread count, which always agree.

        The count is derived from the rows rather than queried separately. Two queries
        cannot be relied on to see one snapshot: SQLAlchemy's pysqlite driver stays in
        autocommit for select-only work, so each runs in its own transaction and a row
        committed between them lands in one and not the other -- a response carrying
        `unread: 1` with `items: []`, which is what a submitter saw when a training job
        finished mid-poll. Sharing one session was the first attempt, and not enough.

        One extra row is fetched to tell a full page from a truncated one. A count is
        issued only when the page is truncated, since only then can the rows not answer
        it -- and a caller reading page one of many has already accepted that the two
        describe different amounts of data.
        """
        with self._session() as session:
            stmt = select(Notification)
            if unread_only:
                stmt = stmt.where(col(Notification.read_at).is_(None))
            rows = list(
                session.exec(
                    stmt.order_by(col(Notification.at).desc()).limit(limit + 1)
                )
            )
            truncated = len(rows) > limit
            rows = rows[:limit]
            if not truncated:
                return rows, sum(1 for n in rows if n.read_at is None)
            unread = session.exec(
                select(func.count(col(Notification.id))).where(
                    col(Notification.read_at).is_(None)
                )
            ).one()
            return rows, int(unread or 0)

    def mark_notification_read(self, notification_id: str) -> bool:
        """Set ``read_at`` if the row is this user's; ``False`` when it is not.

        Missing and not-yours are indistinguishable, so the API can 404 both.
        """
        with self._write() as session:
            row = session.get(Notification, notification_id)
            if row is None:
                return False
            if row.read_at is None:
                row.read_at = _now()
                session.add(row)
                session.commit()
            return True

    def mark_all_notifications_read(self) -> int:
        """Mark every unread notification of this user read; return the count."""
        with self._write() as session:
            rows = session.exec(
                select(Notification).where(col(Notification.read_at).is_(None))
            ).all()
            now = _now()
            for row in rows:
                row.read_at = now
                session.add(row)
            if rows:
                session.commit()
            return len(rows)

    def mark_notification_emailed(self, notification_id: str) -> None:
        """Stamp the (single) email attempt for this notification as delivered."""
        with self._write() as session:
            row = session.get(Notification, notification_id)
            if row is None:
                return
            row.emailed_at = _now()
            session.add(row)
            session.commit()

    def get_notification_prefs(self) -> dict[str, tuple[bool, bool]]:
        """Stored ``(in_app, email)`` switches by event type (absent = defaults)."""
        with self._session() as session:
            rows = session.exec(select(NotificationPreference)).all()
            return {row.event_type: (row.in_app, row.email) for row in rows}

    def set_notification_prefs(self, prefs: Mapping[str, tuple[bool, bool]]) -> None:
        """Upsert the per-event-type ``(in_app, email)`` switches."""
        with self._write() as session:
            for event_type, (in_app, email) in prefs.items():
                row = session.get(NotificationPreference, event_type)
                if row is None:
                    row = NotificationPreference(
                        event_type=event_type, in_app=in_app, email=email
                    )
                else:
                    row.in_app = in_app
                    row.email = email
                    row.updated_at = _now()
                session.add(row)
            session.commit()

    # -- model serving -----------------------------------------------------------

    def record_inference(
        self,
        model: str,
        version: int,
        *,
        n_rows: int,
        latency_ms: float,
        status: str = "ok",
        error: str | None = None,
        request_json: str | None = None,
        predictions_json: str | None = None,
    ) -> None:
        """Append a serving event (best-effort; never blocks the request path)."""
        with self._write() as session:
            session.add(
                InferenceEvent(
                    model=model,
                    version=version,
                    n_rows=n_rows,
                    latency_ms=latency_ms,
                    status=status,
                    error=error,
                    request_json=request_json,
                    predictions_json=predictions_json,
                )
            )
            session.commit()

    def list_inference(
        self, model: str, limit: int = 100, offset: int = 0
    ) -> list[InferenceEvent]:
        """A model's recent serving events, newest first."""
        with self._session() as session:
            stmt = (
                select(InferenceEvent)
                .where(InferenceEvent.model == model)
                .order_by(InferenceEvent.at.desc())  # type: ignore[attr-defined]
                .offset(offset)
                .limit(limit)
            )
            return list(session.exec(stmt).all())

    def inference_inputs(
        self, model: str, limit_events: int = 1000
    ) -> list[dict[str, Any]]:
        """The feature records recently sent to ``model``, for drift analysis.

        Flattens the stored request payloads back into rows (each request may
        carry several); events without a stored payload are skipped.
        """
        rows: list[dict[str, Any]] = []
        for record in self.list_inference(model, limit=limit_events):
            if not record.request_json:
                continue
            try:
                parsed = json.loads(record.request_json)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                rows.extend(r for r in parsed if isinstance(r, dict))
        return rows

    def set_retrain_policy(self, policy: RetrainPolicy) -> None:
        """Create or replace a model's retraining policy."""
        with self._write() as session:
            existing = session.get(RetrainPolicy, policy.model)
            if existing is not None:
                session.delete(existing)
                session.commit()
            session.add(policy)
            session.commit()

    def get_retrain_policy(self, model: str) -> RetrainPolicy | None:
        """The model's retraining policy, or ``None``."""
        with self._session() as session:
            return session.get(RetrainPolicy, model)

    def list_retrain_policies(self) -> list[RetrainPolicy]:
        """Every retraining policy."""
        with self._session() as session:
            return list(session.exec(select(RetrainPolicy)).all())

    def delete_retrain_policy(self, model: str) -> bool:
        """Remove a model's retraining policy; whether one existed."""
        with self._write() as session:
            existing = session.get(RetrainPolicy, model)
            if existing is None:
                return False
            session.delete(existing)
            session.commit()
            return True

    def mark_retrain(self, model: str, *, data_hash: str | None = None) -> None:
        """Record that the policy ran now (and against which data identity)."""
        with self._write() as session:
            existing = session.get(RetrainPolicy, model)
            if existing is None:
                return
            existing.last_run_at = _now()
            if data_hash is not None:
                existing.last_data_hash = data_hash
            session.add(existing)
            session.commit()

    def prune_inference(self, older_than_days: int) -> int:
        """Delete serving events older than N days; return the count (0 = keep)."""
        if older_than_days <= 0:
            return 0
        cutoff = _now() - timedelta(days=older_than_days)
        with self._write() as session:
            rows = session.exec(
                select(InferenceEvent).where(InferenceEvent.at < cutoff)
            ).all()
            for row in rows:
                session.delete(row)
            if rows:
                session.commit()
            return len(rows)

    def prune_audit(self, older_than_days: int) -> int:
        """Delete audit rows older than N days; return the count (0 = keep all)."""
        if older_than_days <= 0:
            return 0
        cutoff = _now() - timedelta(days=older_than_days)
        with self._write() as session:
            rows = session.exec(select(AuditEvent).where(AuditEvent.at < cutoff)).all()
            for row in rows:
                session.delete(row)
            if rows:
                session.commit()
            return len(rows)

    def prune_notifications(self, older_than_days: int) -> int:
        """Delete notifications older than N days; return the count (0 = keep all)."""
        if older_than_days <= 0:
            return 0
        cutoff = _now() - timedelta(days=older_than_days)
        with self._write() as session:
            rows = session.exec(
                select(Notification).where(Notification.at < cutoff)
            ).all()
            for row in rows:
                session.delete(row)
            if rows:
                session.commit()
            return len(rows)

    # -- secrets vault -----------------------------------------------------------
    def save_secret(self, secret: Secret) -> None:
        """Insert or replace a secret by name; ``value`` is already encrypted."""
        with self._write() as session:
            existing = session.exec(
                select(Secret).where(Secret.name == secret.name)
            ).first()
            if existing is not None:
                session.delete(existing)
                session.commit()
            session.add(secret)
            session.commit()

    def get_secret(self, name: str) -> Secret | None:
        """The secret of this name, or ``None``."""
        with self._session() as session:
            return session.exec(select(Secret).where(Secret.name == name)).first()

    def list_secrets(self) -> list[Secret]:
        """Every secret (values still encrypted), newest first."""
        with self._session() as session:
            stmt = (
                select(Secret).order_by(Secret.created_at.desc())  # type: ignore[attr-defined]
            )
            return list(session.exec(stmt))

    def delete_secret(self, name: str) -> bool:
        """Delete a secret; return whether it existed."""
        with self._write() as session:
            secret = session.exec(select(Secret).where(Secret.name == name)).first()
            if secret is None:
                return False
            session.delete(secret)
            session.commit()
            return True

    # -- budgets -----------------------------------------------------------------
    def set_budget(self, max_budget: float, window: str) -> Budget:
        """Set the spend cap and window, resetting the window to now."""
        with self._write() as session:
            budget = session.get(Budget, BUDGET_ID) or Budget()
            budget.max_budget = max_budget
            budget.window = window
            budget.window_start = _now()
            budget.spend = 0.0
            session.add(budget)
            session.commit()
            session.refresh(budget)
            return budget

    def housekeeping(self) -> dict[str, Callable[[], int]]:
        """Cutoff-based cleanups this store contributes beyond the built-in ones.

        Keyed by the name the result is reported under. Each returns how many rows it
        removed and is safe to re-run, the same contract the built-in tasks hold.
        """
        return {}

    def get_budget(self) -> Budget | None:
        """The spend cap, or ``None`` when none is set."""
        with self._session() as session:
            return session.get(Budget, BUDGET_ID)

    def is_over_budget(self) -> bool:
        """Whether the window's spend has reached the cap."""
        budget = self.get_budget()
        if budget is None or budget.max_budget <= 0:
            return False
        if _window_elapsed(budget):  # a lapsed window resets to zero on the next spend
            return False
        return budget.spend >= budget.max_budget

    def add_spend(self, cost: float) -> None:
        """Add a turn's cost to the budget.

        The window is reset lazily when it has elapsed, before the cost is applied, so a
        rolling window needs no background scheduler.
        """
        if cost <= 0:
            return
        with self._write() as session:
            budget = session.get(Budget, BUDGET_ID)
            if budget is None:
                return
            if _window_elapsed(budget):
                budget.window_start = _now()
                budget.spend = 0.0
            budget.spend += cost
            session.add(budget)
            session.commit()

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """The conversation of this id, or ``None`` if it does not exist yet."""
        with self._session() as session:
            return session.get(Conversation, conversation_id)

    def search_conversations(
        self, limit: int = 20, page_id: str | None = None, query: str | None = None
    ) -> tuple[list[Conversation], str | None]:
        """A page of conversations, most-recently-active first, with a cursor.

        Cursor pagination for the conversation list: the ``page_id`` cursor is an
        offset, one extra row is fetched to learn whether more remain, and the returned
        next cursor is the offset of the following page (``None`` when the list is
        exhausted). Ordered by ``updated_at`` so a chat just added to rises to the top
        of the history. A ``query`` filters to conversations whose title contains it

        user when set (an authenticated caller sees only their own chats);
        ``None`` returns all (single-user mode).
        """
        limit = max(1, min(limit, 100))
        try:
            offset = max(0, int(page_id)) if page_id is not None else 0
        except ValueError:
            offset = 0
        with self._session() as session:
            stmt = select(Conversation)
            if query and query.strip():
                stmt = stmt.where(
                    Conversation.title.ilike(f"%{query.strip()}%")  # type: ignore[attr-defined]
                )
            stmt = (
                stmt.order_by(Conversation.updated_at.desc())  # type: ignore[attr-defined]
                .offset(offset)
                .limit(limit + 1)  # one extra row tells us whether a next page exists
            )
            rows = list(session.exec(stmt))
        has_more = len(rows) > limit
        next_page_id = str(offset + limit) if has_more else None
        return rows[:limit], next_page_id

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """Set a conversation's title; return whether it existed.

        The title is otherwise derived from the opening question and never changes, so
        this is how a chat gets a name that reflects what it became.
        """
        with self._write() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return False
            conversation.title = title
            session.add(conversation)
            session.commit()
            return True

    def add_message(
        self,
        conversation_id: str,
        role: str,
        *,
        content: str = "",
        tools: Sequence[Mapping[str, Any]] = (),
        result: Mapping[str, Any] | None = None,
    ) -> str:
        """Append a turn to a conversation and return the message id."""
        row = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tools_json=json.dumps(list(tools)),
            result_json=None if result is None else json.dumps(result),
        )
        with self._write() as session:
            session.add(row)
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.updated_at = _now()  # lift it in the recent-activity order
                session.add(conversation)
            session.commit()
            return row.id

    def delete_messages_from(self, conversation_id: str, message_id: str) -> int:
        """Delete a message and every later one in the conversation; return the count.

        Backs edit-and-resend: editing a past question drops it and the turns after it,
        so the conversation re-runs from that point instead of branching. Nothing is
        deleted if the message is unknown or belongs to another conversation.
        """
        with self._write() as session:
            anchor = session.get(Message, message_id)
            if anchor is None or anchor.conversation_id != conversation_id:
                return 0
            stmt = select(Message).where(
                Message.conversation_id == conversation_id,
                Message.created_at >= anchor.created_at,
            )
            rows = [
                m for m in session.exec(stmt) if m.created_at > anchor.created_at
            ] + [anchor]
            for row in rows:
                session.delete(row)
            session.commit()
            return len(rows)

    def get_message(self, message_id: str) -> Message | None:
        """A single message by id (to resolve its conversation for access checks)."""
        with self._session() as session:
            return session.get(Message, message_id)

    def set_feedback(self, message_id: str, feedback: str | None) -> bool:
        """Rate an assistant turn ("up"/"down"/None); return whether it existed."""
        with self._write() as session:
            message = session.get(Message, message_id)
            if message is None:
                return False
            message.feedback = feedback
            session.add(message)
            session.commit()
            return True

    def messages(self, conversation_id: str) -> list[Message]:
        """The turns of one conversation, in the order they were sent."""
        with self._session() as session:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)  # type: ignore[arg-type]
            )
            return list(session.exec(stmt))

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages; return whether it existed.

        The conversation and its turns are removed. Authored derivations are durable,
        certified artifacts shown on their own page and reusable across conversations,
        so they are left intact (not cascaded); only their back-reference to this
        conversation goes stale.
        """
        with self._write() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return False
            messages = session.exec(
                select(Message).where(Message.conversation_id == conversation_id)
            )
            for message in messages:
                session.delete(message)
            # Flush the messages before marking the conversation, so the child rows are
            # gone from the database by the time the parent row is deleted.
            #
            # Marking both and committing once is not equivalent. The unit of work
            # orders deletes from mapper relationships, and there is none between these
            # two -- only a foreign key on the table, which informs schema sorting and
            # not flush order. So the order is arbitrary, and parent-first makes
            # Postgres reject the whole transaction. SQLite does not enforce foreign
            # keys by default, which is why every local run and the whole test suite
            # passed while a real deployment returned 500.
            session.flush()
            session.delete(conversation)
            session.commit()
            return True

    # -- derivations -------------------------------------------------------------
    def save_derivation(self, derivation: Derivation) -> None:
        """Insert the derivation, or replace an earlier one of the same name.

        A trashed row of the same name is erased first, its run history purged and an
        audit event recorded, rather than silently resurrected and overwritten -- the
        same guard :meth:`upsert_metric` uses for its own name-keyed table.
        """
        erased_trashed = False
        with self._write() as session:
            # Captured now, while the row is still live: commit() below expires
            # its attributes, and record_audit runs after the session closes.
            name = derivation.name
            existing = session.get(Derivation, derivation.name)
            if existing is not None:
                if existing.deleted_at is not None:
                    erased_trashed = True
                    for run in session.exec(
                        select(DerivationRun).where(
                            DerivationRun.name == derivation.name
                        )
                    ):
                        session.delete(run)
                session.delete(existing)
                session.commit()
            session.add(derivation)
            session.commit()
        if erased_trashed:
            self.record_audit(
                "derivation.erase",
                target_type=ResourceType.DERIVATION.value,
                target_id=name,
            )

    def delete_derivation(self, name: str) -> None:
        """Permanently remove a derivation and its certified run history.

        There was previously no delete for derivations on ``Store`` at all: the
        MCP tool removed the live registration, the sidecar, and the cache, but
        left this row and every :class:`DerivationRun` behind forever. This is
        the piece that actually erases them.
        """
        with self._write() as session:
            for run in session.exec(
                select(DerivationRun).where(DerivationRun.name == name)
            ):
                session.delete(run)
            row = session.get(Derivation, name)
            if row is not None:
                session.delete(row)
            session.commit()

    def sync_repo_derivations(self, rows: list[Derivation]) -> None:
        """Reconcile the store's repo-authored derivations to match ``rows``.

        Repo derivations mirror the project's ``derivations/*.py`` files so the UI reads
        them from the store like any other. They are project-global (no owner), so this
        replaces every ``origin='repo'`` row with ``rows``, preserving an existing row's
        ``created_at`` and dropping any whose source file is gone. Chat-authored rows
        (a different origin) are untouched.
        """
        wanted = {row.name: row for row in rows}
        with self._write() as session:
            for prev in session.exec(
                select(Derivation).where(Derivation.origin == "repo")
            ).all():
                replacement = wanted.get(prev.name)
                if replacement is not None:
                    replacement.created_at = prev.created_at
                session.delete(prev)
            session.commit()
            for row in wanted.values():
                session.add(row)
            session.commit()

    def get_derivation(self, name: str) -> Derivation | None:
        """The derivation of this name, or ``None`` when it is absent or trashed."""
        with self._session() as session:
            row = session.get(Derivation, name)
            if row is None or row.deleted_at is not None:
                return None
            return row

    def list_derivations(self) -> list[Derivation]:
        """Authored derivations, newest first."""
        with self._session() as session:
            stmt = select(Derivation).where(col(Derivation.deleted_at).is_(None))
            stmt = stmt.order_by(col(Derivation.created_at).desc())
            return list(session.exec(stmt))

    def derivations_for(self, conversation_id: str) -> list[Derivation]:
        """The derivations authored in one conversation, oldest first (its memory)."""
        with self._session() as session:
            stmt = (
                select(Derivation)
                .where(Derivation.conversation_id == conversation_id)
                .order_by(Derivation.created_at)  # type: ignore[arg-type]
            )
            return list(session.exec(stmt))

    # -- certified runs (derivation result history) ------------------------------
    def append_run(self, run: CertifiedRun) -> bool:
        """Record a certified run; return whether it was new (``False`` if collapsed).

        Version-collapsing: a run whose ``(name, derivation_version)`` is already stored
        is not recorded again, because a deterministic re-derive yields the same
        certified result. Inherits its conversation's owner when unset, so the run is
        attributed like the derivation it belongs to.
        """
        with self._write() as session:
            existing = session.exec(
                select(DerivationRun).where(
                    DerivationRun.name == run.name,
                    DerivationRun.derivation_version == run.derivation_version,
                )
            ).first()
            if existing is not None:
                return False
            row = _run_to_row(run)
            session.add(row)
            session.commit()
            return True

    def runs_for_derivation(self, name: str) -> list[CertifiedRun]:
        """A derivation's certified versions, newest first (its result history)."""
        with self._session() as session:
            stmt = (
                select(DerivationRun)
                .where(DerivationRun.name == name)
                .order_by(DerivationRun.created_at.desc())  # type: ignore[attr-defined]
            )
            return [_row_to_run(row) for row in session.exec(stmt)]

    def latest_run_for_derivation(self, name: str) -> CertifiedRun | None:
        """A derivation's newest certified run, or ``None`` if it has none.

        Callers that only need the newest run (a certificate export, not the full
        history) should use this over :meth:`runs_for_derivation`: the database does
        the ``LIMIT 1`` instead of the caller fetching and deserializing every run.
        """
        with self._session() as session:
            stmt = (
                select(DerivationRun)
                .where(DerivationRun.name == name)
                .order_by(col(DerivationRun.created_at).desc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            return _row_to_run(row) if row is not None else None

    # -- orchestration -----------------------------------------------------------
    def create_orchestration_run(
        self,
        *,
        cause: str = "manual",
        parent_run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> str:
        """Open a materialization run in the 'running' state and return its id."""
        row = OrchestrationRun(
            cause=cause, parent_run_id=parent_run_id, workflow_id=workflow_id
        )
        with self._write() as session:
            session.add(row)
            session.commit()
            return row.id

    def set_workflow_steps(self, run_id: str, steps: list[dict[str, Any]]) -> None:
        """Record a workflow run's per-step outcomes (id/state/run_if)."""
        with self._write() as session:
            row = session.get(OrchestrationRun, run_id)
            if row is not None:
                row.workflow_steps_json = json.dumps(steps)
                session.add(row)
                session.commit()

    # -- workflows ---------------------------------------------------------------
    def list_workflows(self) -> list[dict[str, Any]]:
        """All workflow definitions."""
        with self._session() as session:
            stmt = select(Workflow)
            stmt = stmt.order_by(col(Workflow.created_at))
            return [_workflow_view(row) for row in session.exec(stmt)]

    def get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        """One workflow definition, or None."""
        with self._session() as session:
            row = session.get(Workflow, workflow_id)
            return _workflow_view(row) if row is not None else None

    def upsert_workflow(
        self, *, workflow_id: str | None = None, name: str, steps: list[dict[str, Any]]
    ) -> str:
        """Create or replace a workflow definition; returns its id."""
        with self._write() as session:
            row = session.get(Workflow, workflow_id) if workflow_id else None
            if row is None:
                row = Workflow(name=name)
            row.name = name
            row.steps_json = json.dumps(steps)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def delete_workflow(self, workflow_id: str) -> bool:
        """Remove a workflow definition; return whether it was there to remove."""
        with self._write() as session:
            row = session.get(Workflow, workflow_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def record_asset_run(
        self,
        run_id: str,
        *,
        asset: str,
        state: str,
        data_version: str | None,
        verdict: str | None,
        error: str | None,
        attempts: int,
        duration_ms: int,
        logs: str = "",
        checks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append one asset's materialization result to a run."""
        with self._write() as session:
            session.add(
                AssetRun(
                    run_id=run_id,
                    asset=asset,
                    state=state,
                    data_version=data_version,
                    verdict=verdict,
                    error=error,
                    attempts=attempts,
                    duration_ms=duration_ms,
                    logs=logs,
                    checks_json=json.dumps(checks or []),
                )
            )
            session.commit()

    def finish_orchestration_run(self, run_id: str, status: str) -> None:
        """Mark a run terminal ('succeeded' or 'failed')."""
        with self._write() as session:
            row = session.get(OrchestrationRun, run_id)
            if row is not None:
                row.status = status
                row.finished_at = _now()
                session.add(row)
                session.commit()

    def list_orchestration_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Run summaries with per-state asset counts, newest first."""
        with self._session() as session:
            stmt = select(OrchestrationRun)
            stmt = stmt.order_by(col(OrchestrationRun.started_at).desc()).limit(limit)
            runs = list(session.exec(stmt))
            summaries: list[dict[str, Any]] = []
            for run in runs:
                steps = session.exec(
                    select(AssetRun).where(AssetRun.run_id == run.id)
                ).all()
                counts: dict[str, int] = {}
                for step in steps:
                    counts[step.state] = counts.get(step.state, 0) + 1
                summaries.append(
                    {
                        "id": run.id,
                        "cause": run.cause,
                        "status": run.status,
                        "parent_run_id": run.parent_run_id,
                        "workflow_id": run.workflow_id,
                        "started_at": _as_utc(run.started_at).isoformat(),
                        "finished_at": _as_utc(run.finished_at).isoformat()
                        if run.finished_at
                        else None,
                        "counts": counts,
                    }
                )
            return summaries

    def get_orchestration_run(self, run_id: str) -> dict[str, Any] | None:
        """A run with its per-asset steps, or ``None``."""
        with self._session() as session:
            run = session.get(OrchestrationRun, run_id)
            if run is None:
                return None
            steps = session.exec(
                select(AssetRun)
                .where(AssetRun.run_id == run_id)
                .order_by(col(AssetRun.created_at))
            )
            return {
                "id": run.id,
                "cause": run.cause,
                "status": run.status,
                "parent_run_id": run.parent_run_id,
                "workflow_id": run.workflow_id,
                "workflow_steps": json.loads(run.workflow_steps_json or "[]"),
                "started_at": _as_utc(run.started_at).isoformat(),
                "finished_at": _as_utc(run.finished_at).isoformat()
                if run.finished_at
                else None,
                "steps": [
                    {
                        "asset": s.asset,
                        "state": s.state,
                        "data_version": s.data_version,
                        "verdict": s.verdict,
                        "error": s.error,
                        "attempts": s.attempts,
                        "duration_ms": s.duration_ms,
                        "logs": s.logs,
                        "checks": json.loads(s.checks_json or "[]"),
                    }
                    for s in steps
                ],
            }

    def last_materialized_version(self, asset: str) -> str | None:
        """The content version of the asset's most recent successful materialization."""
        with self._session() as session:
            stmt = (
                select(AssetRun)
                .where(AssetRun.asset == asset)
                .where(AssetRun.state == "succeeded")
                .order_by(col(AssetRun.created_at).desc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            return row.data_version if row is not None else None

    def latest_materializations(self) -> dict[str, dict[str, Any]]:
        """Each asset's most recent successful materialization: when and how long.

        Powers the 'last materialized N ago' freshness hint in the UI.
        """
        with self._session() as session:
            rows = session.exec(
                select(AssetRun)
                .where(AssetRun.state == "succeeded")
                .order_by(col(AssetRun.created_at).desc())
            )
            latest: dict[str, dict[str, Any]] = {}
            for row in rows:  # newest first, so the first row seen per asset wins
                latest.setdefault(
                    row.asset,
                    {
                        "at": _as_utc(row.created_at).isoformat(),
                        "duration_ms": row.duration_ms,
                    },
                )
            return latest

    def asset_states_for_runs(
        self, run_ids: Sequence[str]
    ) -> dict[str, dict[str, str]]:
        """For each run id, its per-asset step state: the run-history matrix cells."""
        if not run_ids:
            return {}
        with self._session() as session:
            rows = session.exec(
                select(AssetRun).where(col(AssetRun.run_id).in_(list(run_ids)))
            )
            cells: dict[str, dict[str, str]] = {rid: {} for rid in run_ids}
            for row in rows:
                cells[row.run_id][row.asset] = row.state
            return cells

    # -- data-quality checks -----------------------------------------------------
    def list_asset_checks(
        self, asset: str | None = None, *, enabled_only: bool = False
    ) -> list[dict[str, Any]]:
        """Data-quality checks, optionally for one asset and/or only enabled ones."""
        with self._session() as session:
            stmt = select(AssetCheck)
            if asset is not None:
                stmt = stmt.where(AssetCheck.asset == asset)
            if enabled_only:
                stmt = stmt.where(col(AssetCheck.enabled).is_(True))
            stmt = stmt.order_by(col(AssetCheck.asset), col(AssetCheck.created_at))
            return [_asset_check_view(row) for row in session.exec(stmt)]

    def upsert_asset_check(
        self,
        *,
        check_id: str | None = None,
        asset: str,
        name: str,
        expr: str,
        severity: str = "warn",
        enabled: bool = True,
    ) -> str:
        """Create or update a data-quality check; returns its id."""
        with self._write() as session:
            row = session.get(AssetCheck, check_id) if check_id else None
            if row is None:
                row = AssetCheck(asset=asset, name=name, expr=expr)
            row.asset = asset
            row.name = name
            row.expr = expr
            row.severity = severity
            row.enabled = enabled
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def delete_asset_check(self, check_id: str) -> bool:
        """Remove a data-quality check; return whether it was there to remove."""
        with self._write() as session:
            row = session.get(AssetCheck, check_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def upsert_materialization_schedule(
        self,
        *,
        name: str,
        selection: str,
        mode: str,
        cron: str = "",
        dataset: str | None = None,
    ) -> str:
        """Create or replace a materialization schedule (by name) and return its id."""
        with self._write() as session:
            existing = session.exec(
                select(MaterializationSchedule).where(
                    MaterializationSchedule.name == name
                )
            ).first()
            row = existing or MaterializationSchedule(name=name)
            row.selection = selection
            row.mode = mode
            row.cron = cron
            row.dataset = dataset
            session.add(row)
            session.commit()
            return row.id

    def list_materialization_schedules(self) -> list[MaterializationSchedule]:
        """The materialization schedules."""
        with self._session() as session:
            stmt = select(MaterializationSchedule)
            return list(session.exec(stmt))

    def active_materialization_schedules(self) -> list[MaterializationSchedule]:
        """Every enabled schedule, for the scheduler to evaluate."""
        with self._session() as session:
            stmt = select(MaterializationSchedule).where(
                col(MaterializationSchedule.enabled).is_(True)
            )
            return list(session.exec(stmt))

    def delete_materialization_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule; return whether it was there to remove."""
        with self._write() as session:
            row = session.get(MaterializationSchedule, schedule_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def touch_materialization_schedule(
        self, schedule_id: str, *, last_data_hash: str | None = None
    ) -> None:
        """Record that a schedule just fired (and, for a sensor, the hash it saw)."""
        with self._write() as session:
            row = session.get(MaterializationSchedule, schedule_id)
            if row is not None:
                row.last_run_at = _now()
                if last_data_hash is not None:
                    row.last_data_hash = last_data_hash
                session.add(row)
                session.commit()

    # -- feature store -----------------------------------------------------------
    def upsert_feature_entity(
        self,
        *,
        name: str,
        join_key: str,
        value_type: str = "string",
        description: str | None = None,
    ) -> None:
        """Create or replace a registered entity."""
        with self._write() as session:
            row = session.get(FeatureEntity, name) or FeatureEntity(name=name)
            row.join_key = join_key
            row.value_type = value_type
            row.description = description
            session.add(row)
            session.commit()

    def list_feature_entities(self) -> list[FeatureEntity]:
        """The registered entities."""
        with self._session() as session:
            stmt = select(FeatureEntity)
            return list(session.exec(stmt))

    def delete_feature_entity(self, name: str) -> None:
        """Remove a registered entity."""
        with self._write() as session:
            row = session.get(FeatureEntity, name)
            if row is not None:
                session.delete(row)
                session.commit()

    def upsert_feature_view(
        self,
        *,
        name: str,
        entities: Sequence[str],
        source: str,
        timestamp_field: str | None,
        ttl_seconds: int | None,
        features: Sequence[dict[str, Any]],
        description: str | None = None,
    ) -> bool:
        """Create or replace a registered feature view; whether it was written.

        A view name is the primary key, so writing one replaces any view already under
        that name.

        A trashed row of the same name is erased first (its child tables purged, an
        audit event recorded) rather than silently resurrected and overwritten in place.
        """
        erased_trashed = False
        with self._write() as session:
            existing = session.get(FeatureViewRow, name)
            if existing is not None and existing.deleted_at is not None:
                erased_trashed = True
                for online in session.exec(
                    select(OnlineFeatureRow).where(
                        OnlineFeatureRow.feature_view == name
                    )
                ):
                    session.delete(online)
                for stat in session.exec(
                    select(FeatureStatisticsRow).where(
                        FeatureStatisticsRow.feature_view == name
                    )
                ):
                    session.delete(stat)
                for drift in session.exec(
                    select(FeatureDriftRow).where(FeatureDriftRow.feature_view == name)
                ):
                    session.delete(drift)
                for check in session.exec(
                    select(FeatureExpectationRow).where(
                        FeatureExpectationRow.feature_view == name
                    )
                ):
                    session.delete(check)
                session.delete(existing)
                session.commit()
            row = session.get(FeatureViewRow, name) or FeatureViewRow(name=name)
            row.entities_json = json.dumps(list(entities))
            row.source = source
            row.timestamp_field = timestamp_field
            row.ttl_seconds = ttl_seconds
            row.features_json = json.dumps(list(features))
            row.description = description
            row.updated_at = _now()
            session.add(row)
            session.commit()
        if erased_trashed:
            self.record_audit(
                "feature_view.erase",
                target_type=ResourceType.FEATURE_VIEW.value,
                target_id=name,
            )
        return True

    def list_feature_views(self) -> list[FeatureViewRow]:
        """The registered feature views the caller may list."""
        with self._session() as session:
            stmt = select(FeatureViewRow).where(
                col(FeatureViewRow.deleted_at).is_(None)
            )
            return list(session.exec(stmt))

    def get_feature_view(self, name: str) -> FeatureViewRow | None:
        """A feature view row, or ``None`` when it is absent or trashed."""
        with self._session() as session:
            row = session.get(FeatureViewRow, name)
            if row is None or row.deleted_at is not None:
                return None
            return row

    def delete_feature_view(self, name: str) -> bool:
        """Remove a feature view, its online values, and its monitoring history.

        Returns whether it existed. Deleting a view discards the materialized values
        and history beneath it, which editing does not.
        """
        if self.get_feature_view(name) is None:
            return False
        with self._write() as session:
            for online in session.exec(
                select(OnlineFeatureRow).where(OnlineFeatureRow.feature_view == name)
            ):
                session.delete(online)
            for stat in session.exec(
                select(FeatureStatisticsRow).where(
                    FeatureStatisticsRow.feature_view == name
                )
            ):
                session.delete(stat)
            for drift in session.exec(
                select(FeatureDriftRow).where(FeatureDriftRow.feature_view == name)
            ):
                session.delete(drift)
            for check in session.exec(
                select(FeatureExpectationRow).where(
                    FeatureExpectationRow.feature_view == name
                )
            ):
                session.delete(check)
            view = session.get(FeatureViewRow, name)
            if view is not None:
                session.delete(view)
            session.commit()
        return True

    def online_upsert(
        self, feature_view: str, records: Sequence[tuple[str, dict[str, Any]]]
    ) -> None:
        """Replace the stored feature values for each entity key of a view."""
        with self._write() as session:
            for entity_key, values in records:
                existing = session.exec(
                    select(OnlineFeatureRow)
                    .where(OnlineFeatureRow.feature_view == feature_view)
                    .where(OnlineFeatureRow.entity_key == entity_key)
                ).first()
                row = existing or OnlineFeatureRow(
                    feature_view=feature_view, entity_key=entity_key
                )
                row.values_json = json.dumps(values)
                row.updated_at = _now()
                session.add(row)
            session.commit()

    def online_lookup(
        self, feature_view: str, entity_keys: Sequence[str]
    ) -> list[dict[str, Any] | None]:
        """The stored feature values for each entity key, or ``None`` when absent."""
        with self._session() as session:
            stmt = select(OnlineFeatureRow).where(
                OnlineFeatureRow.feature_view == feature_view
            )
            stored = {
                row.entity_key: json.loads(row.values_json)
                for row in session.exec(stmt)
            }
            return [stored.get(key) for key in entity_keys]

    def online_stats(self, feature_view: str) -> tuple[int, datetime | None]:
        """The materialized key count and last-materialized time for one view.

        Both derive from the online rows already written per key (their ``updated_at``
        is stamped on each upsert), so freshness needs no separate bookkeeping.
        """
        with self._session() as session:
            count, last = session.exec(
                select(
                    func.count(col(OnlineFeatureRow.id)),
                    func.max(OnlineFeatureRow.updated_at),
                ).where(OnlineFeatureRow.feature_view == feature_view)
            ).one()
            return int(count or 0), last

    def online_stats_all(self) -> dict[str, tuple[int, datetime | None]]:
        """Per-view key count and last-materialized time, in one grouped query."""
        with self._session() as session:
            rows = session.exec(
                select(
                    OnlineFeatureRow.feature_view,
                    func.count(col(OnlineFeatureRow.id)),
                    func.max(OnlineFeatureRow.updated_at),
                ).group_by(OnlineFeatureRow.feature_view)
            )
            return {view: (int(count or 0), last) for view, count, last in rows}

    def add_feature_statistics(
        self,
        feature_view: str,
        *,
        row_count: int,
        profiles: Sequence[dict[str, Any]],
        is_baseline: bool = False,
        sample: Sequence[dict[str, Any]] | None = None,
    ) -> str:
        """Record a profile snapshot; when a baseline, retire the view's prior one."""
        with self._write() as session:
            if is_baseline:
                for prior in session.exec(
                    select(FeatureStatisticsRow)
                    .where(FeatureStatisticsRow.feature_view == feature_view)
                    .where(col(FeatureStatisticsRow.is_baseline).is_(True))
                ):
                    prior.is_baseline = False
                    prior.sample_json = ""
                    session.add(prior)
            row = FeatureStatisticsRow(
                feature_view=feature_view,
                row_count=row_count,
                profiles_json=json.dumps(list(profiles), default=str),
                is_baseline=is_baseline,
                sample_json=json.dumps(list(sample), default=str) if sample else "",
            )
            session.add(row)
            session.commit()
            return row.id

    def list_feature_statistics(
        self, feature_view: str, limit: int = 50
    ) -> list[FeatureStatisticsRow]:
        """A feature view's profile snapshots, newest first."""
        with self._session() as session:
            stmt = select(FeatureStatisticsRow).where(
                FeatureStatisticsRow.feature_view == feature_view
            )
            stmt = stmt.order_by(col(FeatureStatisticsRow.at).desc()).limit(limit)
            return list(session.exec(stmt))

    def get_feature_baseline(self, feature_view: str) -> FeatureStatisticsRow | None:
        """The view's baseline snapshot (with its reference sample), or ``None``."""
        with self._session() as session:
            stmt = (
                select(FeatureStatisticsRow)
                .where(FeatureStatisticsRow.feature_view == feature_view)
                .where(col(FeatureStatisticsRow.is_baseline).is_(True))
            )
            return session.exec(stmt).first()

    def add_feature_drift(
        self,
        feature_view: str,
        *,
        n_columns: int,
        n_drifted: int,
        share_drifted: float,
        dataset_drift: bool,
        n_current_rows: int,
        report: dict[str, Any],
    ) -> str:
        """Record the outcome of a feature-drift check."""
        with self._write() as session:
            row = FeatureDriftRow(
                feature_view=feature_view,
                n_columns=n_columns,
                n_drifted=n_drifted,
                share_drifted=share_drifted,
                dataset_drift=dataset_drift,
                n_current_rows=n_current_rows,
                report_json=json.dumps(report, default=str),
            )
            session.add(row)
            session.commit()
            return row.id

    def list_feature_drift(
        self, feature_view: str, limit: int = 50
    ) -> list[FeatureDriftRow]:
        """A feature view's drift checks, newest first."""
        with self._session() as session:
            stmt = select(FeatureDriftRow).where(
                FeatureDriftRow.feature_view == feature_view
            )
            stmt = stmt.order_by(col(FeatureDriftRow.at).desc()).limit(limit)
            return list(session.exec(stmt))

    def set_feature_view_contract(self, name: str, contract_json: str | None) -> bool:
        """Attach (or, with ``None``, clear) a view's contract; False if unknown."""
        with self._write() as session:
            row = session.get(FeatureViewRow, name)
            if row is None:
                return False
            row.contract_json = contract_json
            row.updated_at = _now()
            session.add(row)
            session.commit()
            return True

    def add_feature_expectation(
        self,
        feature_view: str,
        *,
        verdict: str,
        n_clauses: int,
        n_violated: int,
        row_count: int,
        report: dict[str, Any],
    ) -> str:
        """Record the outcome of a data-contract check on a feature view."""
        with self._write() as session:
            row = FeatureExpectationRow(
                feature_view=feature_view,
                verdict=verdict,
                n_clauses=n_clauses,
                n_violated=n_violated,
                row_count=row_count,
                report_json=json.dumps(report, default=str),
            )
            session.add(row)
            session.commit()
            return row.id

    def list_feature_expectations(
        self, feature_view: str, limit: int = 50
    ) -> list[FeatureExpectationRow]:
        """A feature view's contract checks, newest first."""
        with self._session() as session:
            stmt = select(FeatureExpectationRow).where(
                FeatureExpectationRow.feature_view == feature_view
            )
            stmt = stmt.order_by(col(FeatureExpectationRow.at).desc()).limit(limit)
            return list(session.exec(stmt))

    def upsert_training_set(
        self,
        *,
        name: str,
        features: Sequence[str],
        label: str | None,
        rows: Sequence[dict[str, Any]],
    ) -> None:
        """Create or replace a named, materialized training set."""
        with self._write() as session:
            existing = session.exec(
                select(FeatureTrainingSet).where(FeatureTrainingSet.name == name)
            ).first()
            row = existing or FeatureTrainingSet(name=name)
            row.features_json = json.dumps(list(features))
            row.label = label
            row.row_count = len(rows)
            row.rows_json = json.dumps(list(rows), default=str)
            row.created_at = _now()
            session.add(row)
            session.commit()

    def list_training_sets(self) -> list[FeatureTrainingSet]:
        """The materialized training sets, newest first."""
        with self._session() as session:
            stmt = select(FeatureTrainingSet)
            stmt = stmt.order_by(col(FeatureTrainingSet.created_at).desc())
            return list(session.exec(stmt))

    def get_training_set(self, name: str) -> FeatureTrainingSet | None:
        """A training set by name."""
        with self._session() as session:
            stmt = select(FeatureTrainingSet).where(FeatureTrainingSet.name == name)
            return session.exec(
                stmt.order_by(col(FeatureTrainingSet.created_at).desc())
            ).first()

    def list_training_set_names(self) -> list[str]:
        """Every training set's name (for the model-training source list)."""
        with self._session() as session:
            stmt = select(FeatureTrainingSet).order_by(
                col(FeatureTrainingSet.created_at).desc()
            )
            return [row.name for row in session.exec(stmt)]

    def training_set_rows(self, name: str) -> list[dict[str, Any]]:
        """The materialized rows of a training set (for training); empty if absent."""
        row = self.get_training_set(name)
        return json.loads(row.rows_json) if row is not None else []

    def delete_training_set(self, name: str) -> None:
        """Remove a training set."""
        with self._write() as session:
            stmt = select(FeatureTrainingSet).where(FeatureTrainingSet.name == name)
            for row in session.exec(stmt):
                session.delete(row)
            session.commit()

    # -- dashboards --------------------------------------------------------------
    def create_dashboard(
        self, *, name: str, title: str, spec_json: str, copied_from: str | None = None
    ) -> str:
        """Create a draft dashboard, record its first version, and return its id.

        ``copied_from`` names the dashboard this one was duplicated from, so a copy is
        recorded in the same write that creates it rather than in a follow-up update.
        """
        row = Dashboard(
            name=name, title=title, spec_json=spec_json, copied_from=copied_from
        )
        with self._write() as session:
            session.add(row)
            session.add(
                DashboardVersion(
                    dashboard_id=row.id, version=1, spec_json=spec_json, label="saved"
                )
            )
            session.commit()
            return row.id

    def list_dashboards(self) -> list[dict[str, Any]]:
        """Dashboard summaries (id, name, title, status), newest activity first."""
        with self._session() as session:
            stmt = select(Dashboard)
            stmt = stmt.where(col(Dashboard.deleted_at).is_(None))
            stmt = stmt.order_by(col(Dashboard.updated_at).desc())
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "title": row.title,
                    "status": row.status,
                    "version": row.version,
                    "updated_at": _as_utc(row.updated_at).isoformat(),
                    "copied_from": row.copied_from,
                }
                for row in session.exec(stmt)
            ]

    def get_dashboard(self, dashboard_id: str) -> Dashboard | None:
        """A dashboard row the caller reaches at ``minimum``, else ``None``."""
        with self._session() as session:
            row = session.get(Dashboard, dashboard_id)
            if row is None or row.deleted_at is not None:
                return None
            return row

    def save_dashboard(
        self, dashboard_id: str, spec_json: str, *, title: str | None = None
    ) -> Dashboard | None:
        """Replace the draft spec, bump the version, and append it to the history."""
        with self._write() as session:
            row = session.get(Dashboard, dashboard_id)
            if row is None:
                return None
            row.spec_json = spec_json
            if title is not None:
                row.title = title
            row.version += 1
            row.updated_at = _now()
            session.add(row)
            session.add(
                DashboardVersion(
                    dashboard_id=row.id,
                    version=row.version,
                    spec_json=spec_json,
                    label="saved",
                )
            )
            session.commit()
            session.refresh(row)
            return row

    def publish_dashboard(self, dashboard_id: str) -> Dashboard | None:
        """Snapshot the current draft as the published spec viewers see."""
        with self._write() as session:
            row = session.get(Dashboard, dashboard_id)
            if row is None:
                return None
            row.published_spec_json = row.spec_json
            row.status = "published"
            row.updated_at = _now()
            session.add(row)
            session.add(
                DashboardVersion(
                    dashboard_id=row.id,
                    version=row.version,
                    spec_json=row.spec_json,
                    label="published",
                )
            )
            session.commit()
            session.refresh(row)
            return row

    def delete_dashboard(self, dashboard_id: str) -> None:
        """Delete a dashboard and its versions and subscriptions."""
        with self._write() as session:
            versions = session.exec(
                select(DashboardVersion).where(
                    DashboardVersion.dashboard_id == dashboard_id
                )
            )
            subscriptions = session.exec(
                select(DashboardSubscription).where(
                    DashboardSubscription.dashboard_id == dashboard_id
                )
            )
            for child in [*versions, *subscriptions]:
                session.delete(child)
            dashboard = session.get(Dashboard, dashboard_id)
            if dashboard is not None:
                session.delete(dashboard)
            session.commit()

    def dashboard_versions(self, dashboard_id: str) -> list[dict[str, Any]]:
        """A dashboard's saved revisions, newest first."""
        with self._session() as session:
            stmt = (
                select(DashboardVersion)
                .where(DashboardVersion.dashboard_id == dashboard_id)
                .order_by(col(DashboardVersion.created_at).desc())
            )
            return [
                {
                    "id": row.id,
                    "version": row.version,
                    "label": row.label,
                    "created_at": _as_utc(row.created_at).isoformat(),
                }
                for row in session.exec(stmt)
            ]

    def create_dashboard_subscription(
        self,
        *,
        dashboard_id: str,
        page: str,
        cron: str,
        recipients: Sequence[str],
        variable_state: dict[str, Any],
        fmt: str,
        channel: str,
    ) -> str:
        """Register a scheduled snapshot delivery and return its id."""
        row = DashboardSubscription(
            dashboard_id=dashboard_id,
            page=page,
            cron=cron,
            recipients_json=json.dumps(list(recipients)),
            variable_state_json=json.dumps(variable_state),
            fmt=fmt,
            channel=channel,
        )
        with self._write() as session:
            session.add(row)
            session.commit()
            return row.id

    def list_dashboard_subscriptions(
        self, dashboard_id: str
    ) -> list[DashboardSubscription]:
        """A dashboard's subscriptions."""
        with self._session() as session:
            stmt = select(DashboardSubscription).where(
                DashboardSubscription.dashboard_id == dashboard_id
            )
            return list(session.exec(stmt))

    def active_dashboard_subscriptions(self) -> list[DashboardSubscription]:
        """Every active subscription across dashboards, for the scheduler to fire."""
        with self._session() as session:
            live_dashboards = select(Dashboard.id).where(
                col(Dashboard.deleted_at).is_(None)
            )
            stmt = select(DashboardSubscription).where(
                col(DashboardSubscription.active).is_(True),
                col(DashboardSubscription.dashboard_id).in_(live_dashboards),
            )
            return list(session.exec(stmt))

    def delete_dashboard_subscription(self, subscription_id: str) -> None:
        """Remove a subscription."""
        with self._write() as session:
            row = session.get(DashboardSubscription, subscription_id)
            if row is not None:
                session.delete(row)
                session.commit()

    def touch_dashboard_subscription(self, subscription_id: str) -> None:
        """Record that a subscription just delivered, for its next-due calculation."""
        with self._write() as session:
            row = session.get(DashboardSubscription, subscription_id)
            if row is not None:
                row.last_run_at = _now()
                session.add(row)
                session.commit()

    def create_notebook(
        self, name: str, notebook_id: str | None = None, folder_id: str | None = None
    ) -> str:
        """Create a notebook with one empty code cell and return its id."""
        row = (
            Notebook(name=name, folder_id=folder_id)
            if notebook_id is None
            else Notebook(id=notebook_id, name=name, folder_id=folder_id)
        )
        with self._write() as session:
            session.add(row)
            session.add(NotebookCell(id=_new_id(), notebook_id=row.id, position=0))
            session.commit()
            return row.id

    def list_notebooks(self) -> list[dict[str, Any]]:
        """Notebook summaries (id, name, timestamps, cell count), newest first."""
        with self._session() as session:
            stmt = select(Notebook)
            stmt = stmt.where(col(Notebook.deleted_at).is_(None))
            stmt = stmt.order_by(Notebook.updated_at.desc())  # type: ignore[attr-defined]
            summaries: list[dict[str, Any]] = []
            for row in session.exec(stmt):
                count = len(
                    session.exec(
                        select(NotebookCell.id).where(
                            NotebookCell.notebook_id == row.id
                        )
                    ).all()
                )
                summaries.append(
                    {
                        "id": row.id,
                        "name": row.name,
                        "folder_id": row.folder_id,
                        "copied_from": row.copied_from,
                        "created_at": _as_utc(row.created_at).isoformat(),
                        "updated_at": _as_utc(row.updated_at).isoformat(),
                        "cell_count": count,
                    }
                )
            return summaries

    def get_notebook(self, notebook_id: str) -> Notebook | None:
        """A notebook row, or ``None`` when it does not exist or is trashed."""
        with self._session() as session:
            row = session.get(Notebook, notebook_id)
            if row is None or row.deleted_at is not None:
                return None
            return row

    # -- compute usage ----------------------------------------------------------
    def record_compute_usage(
        self,
        *,
        notebook_id: str,
        profile: str,
        seconds: float,
        cost: float,
        kind: str = "interactive",
        profile_version: str = "",
    ) -> None:
        """Append one finished session's attribution."""
        with self._write() as session:
            session.add(
                ComputeUsage(
                    notebook_id=notebook_id,
                    profile=profile,
                    profile_version=profile_version,
                    seconds=seconds,
                    cost=cost,
                    kind=kind,
                )
            )
            session.commit()

    def compute_usage(
        self, *, since: datetime | None = None, limit: int = 500
    ) -> list[ComputeUsage]:
        """Recent compute sessions, newest first."""
        with self._session() as session:
            stmt = select(ComputeUsage)
            if since is not None:
                stmt = stmt.where(ComputeUsage.ended_at >= since)
            stmt = stmt.order_by(col(ComputeUsage.ended_at).desc()).limit(limit)
            return list(session.exec(stmt))

    def compute_spend(self, *, since: datetime | None = None) -> float:
        """Total compute cost since ``since`` (all time when None)."""
        with self._session() as session:
            stmt = select(func.coalesce(func.sum(ComputeUsage.cost), 0.0))
            if since is not None:
                stmt = stmt.where(ComputeUsage.ended_at >= since)
            return float(session.exec(stmt).one())

    def compute_spend_by_kind(
        self, *, since: datetime | None = None
    ) -> dict[str, float]:
        """Cost split by interactive versus batch."""
        with self._session() as session:
            stmt = select(
                ComputeUsage.kind, func.coalesce(func.sum(ComputeUsage.cost), 0.0)
            )
            if since is not None:
                stmt = stmt.where(ComputeUsage.ended_at >= since)
            stmt = stmt.group_by(col(ComputeUsage.kind))
            return {
                kind or "interactive": float(total)
                for kind, total in session.exec(stmt)
            }

    def get_cells(self, notebook_id: str) -> list[NotebookCell]:
        """A notebook's cells in document order (by ``position``)."""
        with self._session() as session:
            stmt = (
                select(NotebookCell)
                .where(NotebookCell.notebook_id == notebook_id)
                .order_by(NotebookCell.position)  # type: ignore[arg-type]
            )
            return list(session.exec(stmt))

    def get_cell(self, cell_id: str) -> NotebookCell | None:
        """A single cell by id."""
        with self._session() as session:
            return session.get(NotebookCell, cell_id)

    def update_notebook(self, notebook_id: str, **fields: Any) -> None:
        """Update a notebook's scalar fields (name, deps, metadata, schedule)."""
        with self._write() as session:
            row = session.get(Notebook, notebook_id)
            if row is None:
                return
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = _now()
            session.add(row)
            session.commit()

    def add_cell(
        self,
        notebook_id: str,
        *,
        cell_type: str = "code",
        after: str | None = None,
        cell_id: str | None = None,
        source: str = "",
        metadata_json: str = "{}",
    ) -> NotebookCell:
        """Insert a new cell, after ``after`` (or at the end), renumbering positions."""
        with self._write() as session:
            cells = list(
                session.exec(
                    select(NotebookCell)
                    .where(NotebookCell.notebook_id == notebook_id)
                    .order_by(NotebookCell.position)  # type: ignore[arg-type]
                )
            )
            index = len(cells)
            if after is not None:
                for position, cell in enumerate(cells):
                    if cell.id == after:
                        index = position + 1
                        break
            new_cell = NotebookCell(
                id=cell_id or _new_id(),
                notebook_id=notebook_id,
                cell_type=cell_type,
                source=source,
                metadata_json=metadata_json,
            )
            cells.insert(index, new_cell)
            for position, cell in enumerate(cells):
                cell.position = position
                session.add(cell)
            self._touch(session, notebook_id)
            session.commit()
            session.refresh(new_cell)
            return new_cell

    def update_cell(self, cell_id: str, **fields: Any) -> None:
        """Update a cell's editable fields (source, cell_type, metadata_json)."""
        with self._write() as session:
            cell = session.get(NotebookCell, cell_id)
            if cell is None:
                return
            for key, value in fields.items():
                setattr(cell, key, value)
            session.add(cell)
            self._touch(session, cell.notebook_id)
            session.commit()

    def save_cell_result(
        self,
        cell_id: str,
        outputs_json: str,
        execution_count: int | None,
        data_mode: str | None = None,
        queries: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist a cell's rendered outputs and the counter from its last run.

        ``data_mode`` records how the cell reached its data and ``queries`` what it
        pushed to the warehouse. Both go into the cell's nbformat metadata under our own
        key, so they survive an export and cannot collide with a key Jupyter defines.
        """
        with self._write() as session:
            cell = session.get(NotebookCell, cell_id)
            if cell is None:
                return
            cell.outputs_json = outputs_json
            cell.execution_count = execution_count
            if data_mode is not None:
                metadata = json.loads(cell.metadata_json or "{}")
                ours = dict(metadata.get("elbi") or {})
                if data_mode:
                    ours["data_mode"] = data_mode
                else:
                    ours.pop("data_mode", None)
                if queries:
                    ours["queries"] = queries
                else:
                    ours.pop("queries", None)
                metadata["elbi"] = ours
                cell.metadata_json = json.dumps(metadata)
            session.add(cell)
            session.commit()

    def delete_cell(self, cell_id: str) -> None:
        """Delete a cell and renumber the notebook's remaining positions."""
        with self._write() as session:
            cell = session.get(NotebookCell, cell_id)
            if cell is None:
                return
            notebook_id = cell.notebook_id
            session.delete(cell)
            session.flush()
            remaining = list(
                session.exec(
                    select(NotebookCell)
                    .where(NotebookCell.notebook_id == notebook_id)
                    .order_by(NotebookCell.position)  # type: ignore[arg-type]
                )
            )
            for position, other in enumerate(remaining):
                other.position = position
                session.add(other)
            self._touch(session, notebook_id)
            session.commit()

    def reorder_cells(self, notebook_id: str, ordered_ids: Sequence[str]) -> None:
        """Order cells by ``ordered_ids``; unlisted ids keep their relative order."""
        with self._write() as session:
            cells = {
                cell.id: cell
                for cell in session.exec(
                    select(NotebookCell).where(NotebookCell.notebook_id == notebook_id)
                )
            }
            position = 0
            for cell_id in ordered_ids:
                cell = cells.pop(cell_id, None)
                if cell is not None:
                    cell.position = position
                    session.add(cell)
                    position += 1
            for cell in cells.values():  # any unlisted cells trail, order preserved
                cell.position = position
                session.add(cell)
                position += 1
            self._touch(session, notebook_id)
            session.commit()

    def replace_cells(self, notebook_id: str, cells: Sequence[NotebookCell]) -> None:
        """Replace a notebook's entire cell list (used by ``.ipynb`` import)."""
        with self._write() as session:
            for existing in session.exec(
                select(NotebookCell).where(NotebookCell.notebook_id == notebook_id)
            ):
                session.delete(existing)
            session.flush()
            for position, cell in enumerate(cells):
                cell.notebook_id = notebook_id
                cell.position = position
                session.add(cell)
            self._touch(session, notebook_id)
            session.commit()

    def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook and all its cells."""
        with self._write() as session:
            for cell in session.exec(
                select(NotebookCell).where(NotebookCell.notebook_id == notebook_id)
            ):
                session.delete(cell)
            row = session.get(Notebook, notebook_id)
            if row is not None:
                session.delete(row)
            session.commit()

    def notebooks_with_schedule(self) -> list[Notebook]:
        """Every notebook with a run schedule set (for the maintenance scheduler)."""
        with self._session() as session:
            stmt = select(Notebook).where(
                col(Notebook.schedule_json).is_not(None),
                col(Notebook.deleted_at).is_(None),
            )
            return list(session.exec(stmt))

    # -- notebook folders --------------------------------------------------------
    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        """Create a folder under ``parent_id`` (``None`` == root) and return its id."""
        row = NotebookFolder(name=name, parent_id=parent_id)
        with self._write() as session:
            session.add(row)
            session.commit()
            return row.id

    def list_folders(self) -> list[dict[str, Any]]:
        """The caller's folders as a flat list; the client rebuilds the tree."""
        with self._session() as session:
            stmt = select(NotebookFolder).where(
                col(NotebookFolder.deleted_at).is_(None)
            )
            stmt = stmt.order_by(col(NotebookFolder.name))
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "parent_id": row.parent_id,
                    "created_at": _as_utc(row.created_at).isoformat(),
                    "updated_at": _as_utc(row.updated_at).isoformat(),
                }
                for row in session.exec(stmt)
            ]

    def get_folder(self, folder_id: str) -> NotebookFolder | None:
        """A folder row, or ``None``."""
        with self._session() as session:
            row = session.get(NotebookFolder, folder_id)
            if row is None or row.deleted_at is not None:
                return None
            return row

    def rename_folder(self, folder_id: str, name: str) -> None:
        """Rename a folder in place."""
        with self._write() as session:
            row = session.get(NotebookFolder, folder_id)
            if row is None:
                return
            row.name = name
            row.updated_at = _now()
            session.add(row)
            session.commit()

    def move_folder(self, folder_id: str, new_parent_id: str | None) -> bool:
        """Reparent a folder, rejecting a move into itself or one of its descendants.

        Returns ``False`` (leaving the folder in place) when the move would form a
        cycle. Walking up from ``new_parent_id`` and reaching ``folder_id`` means the
        proposed parent is a descendant of the folder, which a tree cannot allow.
        """
        with self._write() as session:
            row = session.get(NotebookFolder, folder_id)
            if row is None:
                return False
            cursor = new_parent_id
            seen: set[str] = set()
            while cursor is not None and cursor not in seen:
                if cursor == folder_id:
                    return False
                seen.add(cursor)
                parent = session.get(NotebookFolder, cursor)
                cursor = parent.parent_id if parent is not None else None
            row.parent_id = new_parent_id
            row.updated_at = _now()
            session.add(row)
            session.commit()
            return True

    def delete_folder(self, folder_id: str, recursive: bool = False) -> bool:
        """Delete a folder; with ``recursive`` also delete its whole subtree.

        Returns ``False`` without deleting when the folder still holds child folders or
        notebooks and ``recursive`` is off, so the caller can surface a conflict.
        A missing folder is a no-op and returns ``True``.
        """
        with self._write() as session:
            row = session.get(NotebookFolder, folder_id)
            if row is None:
                return True
            # The subtree rooted here, gathered breadth-first over parent edges.
            subtree = [folder_id]
            frontier = [folder_id]
            while frontier:
                children = session.exec(
                    select(NotebookFolder.id).where(
                        col(NotebookFolder.parent_id).in_(frontier)
                    )
                ).all()
                subtree.extend(children)
                frontier = list(children)
            notebooks = session.exec(
                select(Notebook.id).where(col(Notebook.folder_id).in_(subtree))
            ).all()
            if not recursive and (len(subtree) > 1 or notebooks):
                return False
            for cell in session.exec(
                select(NotebookCell).where(col(NotebookCell.notebook_id).in_(notebooks))
            ):
                session.delete(cell)
            for nb in session.exec(
                select(Notebook).where(col(Notebook.folder_id).in_(subtree))
            ):
                session.delete(nb)
            for fid in subtree:
                folder = session.get(NotebookFolder, fid)
                if folder is not None:
                    session.delete(folder)
            session.commit()
            return True

    def move_notebook(self, notebook_id: str, folder_id: str | None) -> None:
        """Move a notebook into ``folder_id`` (``None`` == root)."""
        with self._write() as session:
            row = session.get(Notebook, notebook_id)
            if row is None:
                return
            row.folder_id = folder_id
            row.updated_at = _now()
            session.add(row)
            session.commit()

    def _touch(self, session: Session, notebook_id: str) -> None:
        """Bump a notebook's ``updated_at`` within an open write session."""
        row = session.get(Notebook, notebook_id)
        if row is not None:
            row.updated_at = _now()
            session.add(row)

    # -- config ------------------------------------------------------------------
    def get_config(self, key: str) -> str | None:
        """The stored value for ``key``, or ``None``."""
        with self._session() as session:
            row = session.get(Setting, key)
            return None if row is None else row.value

    def set_config(self, key: str, value: str) -> None:
        """Store ``value`` under ``key``, replacing any earlier value."""
        with self._write() as session:
            row = session.get(Setting, key)
            if row is None:
                session.add(Setting(key=key, value=value))
            else:
                row.value = value
                session.add(row)
            session.commit()

    # -- LLM profiles ------------------------------------------------------------
    def save_profile(self, profile: LlmProfile) -> None:
        """Insert or replace an LLM profile by name."""
        with self._write() as session:
            existing = session.get(LlmProfile, profile.name)
            if existing is not None:
                session.delete(existing)
                session.commit()
            session.add(profile)
            session.commit()

    def get_profile(self, name: str) -> LlmProfile | None:
        """The profile of this name, or ``None``."""
        with self._session() as session:
            return session.get(LlmProfile, name)

    def list_profiles(self) -> list[LlmProfile]:
        """Every LLM profile, oldest first."""
        with self._session() as session:
            stmt = select(LlmProfile).order_by(LlmProfile.created_at)  # type: ignore[arg-type]
            return list(session.exec(stmt))

    def delete_profile(self, name: str) -> bool:
        """Delete a profile; return whether it existed."""
        with self._write() as session:
            profile = session.get(LlmProfile, name)
            if profile is None:
                return False
            session.delete(profile)
            session.commit()
            return True

    # -- data sources ------------------------------------------------------------
    def save_data_source(self, source: DataSource) -> None:
        """Insert or replace a data source by id."""
        with self._write() as session:
            existing = session.get(DataSource, source.id)
            if existing is not None:
                session.delete(existing)
                session.commit()
            source.updated_at = _now()
            session.add(source)
            session.commit()

    def get_data_source(self, source_id: str) -> DataSource | None:
        """The data source of this id, or ``None``."""
        with self._session() as session:
            return session.get(DataSource, source_id)

    def get_data_source_by_name(self, name: str) -> DataSource | None:
        """The data source registered under ``name``, or ``None`` (used by bindings)."""
        with self._session() as session:
            stmt = select(DataSource).where(DataSource.name == name)
            return session.exec(stmt).first()

    def list_data_sources(self) -> list[DataSource]:
        """Every registered data source, newest first."""
        with self._session() as session:
            stmt = select(DataSource).order_by(DataSource.created_at.desc())  # type: ignore[attr-defined]
            return list(session.exec(stmt))

    def delete_data_source(self, source_id: str) -> bool:
        """Delete a data source; return whether it existed."""
        with self._write() as session:
            source = session.get(DataSource, source_id)
            if source is None:
                return False
            session.delete(source)
            session.commit()
            return True

    # --- Warehouse: external sources and their per-table schemas ---------------

    def save_external_source(self, source: ExternalDataSource) -> None:
        """Insert or replace a warehouse source by id."""
        with self._write() as session:
            existing = session.get(ExternalDataSource, source.id)
            if existing is not None:
                session.delete(existing)
                session.commit()
            source.updated_at = _now()
            session.add(source)
            session.commit()

    def get_external_source(self, source_id: str) -> ExternalDataSource | None:
        """The warehouse source of this id, or ``None``."""
        with self._session() as session:
            return session.get(ExternalDataSource, source_id)

    def get_external_source_by_name(self, name: str) -> ExternalDataSource | None:
        """The warehouse source registered under ``name``, or ``None``."""
        with self._session() as session:
            stmt = select(ExternalDataSource).where(ExternalDataSource.name == name)
            return session.exec(stmt).first()

    def list_external_sources(self) -> list[ExternalDataSource]:
        """Every warehouse source, newest first."""
        with self._session() as session:
            stmt = select(ExternalDataSource).order_by(
                ExternalDataSource.created_at.desc()  # type: ignore[attr-defined]
            )
            return list(session.exec(stmt))

    def delete_external_source(self, source_id: str) -> bool:
        """Delete a warehouse source, its schemas and their columns; did it exist?

        Nothing else collects the columns and the search index reads them directly, so
        one left behind is a table a search goes on offering after it is gone.
        """
        with self._write() as session:
            source = session.get(ExternalDataSource, source_id)
            if source is None:
                return False
            schemas = session.exec(
                select(ExternalDataSchema).where(
                    ExternalDataSchema.source_id == source_id
                )
            )
            for schema in schemas:
                session.delete(schema)
            columns = session.exec(
                select(WarehouseColumn).where(WarehouseColumn.source_id == source_id)
            )
            for column in columns:
                session.delete(column)
            session.delete(source)
            session.commit()
            return True

    def save_external_schema(self, schema: ExternalDataSchema) -> None:
        """Insert or replace a warehouse schema by id."""
        with self._write() as session:
            existing = session.get(ExternalDataSchema, schema.id)
            if existing is not None:
                session.delete(existing)
                session.commit()
            session.add(schema)
            session.commit()

    def get_external_schema(self, schema_id: str) -> ExternalDataSchema | None:
        """The warehouse schema of this id, or ``None``."""
        with self._session() as session:
            return session.get(ExternalDataSchema, schema_id)

    def list_external_schemas(self, source_id: str) -> list[ExternalDataSchema]:
        """Every schema belonging to a warehouse source."""
        with self._session() as session:
            stmt = select(ExternalDataSchema).where(
                ExternalDataSchema.source_id == source_id
            )
            return list(session.exec(stmt))

    def list_all_external_schemas(self) -> list[ExternalDataSchema]:
        """Every warehouse schema across all sources (for the tables listing)."""
        with self._session() as session:
            return list(session.exec(select(ExternalDataSchema)))

    def get_external_schema_by_table(self, table: str) -> ExternalDataSchema | None:
        """The schema that produced a warehouse table, by its qualified table name.

        A query names a table, not a schema id, so finding what produced one starts
        from the name.
        """
        with self._session() as session:
            return session.exec(
                select(ExternalDataSchema).where(ExternalDataSchema.table == table)
            ).first()

    def due_external_sources(self, now: datetime) -> list[ExternalDataSource]:
        """Sources whose auto-sync cadence has elapsed since their last attempt.

        A source on ``sync_frequency == "manual"`` is never due. A never-synced source
        is due immediately for its first sync, unless its last attempt errored: a failed
        source waits a full interval (gated on ``updated_at``, the last-attempt time),
        so a broken connection is retried on cadence rather than every tick.
        """
        due: list[ExternalDataSource] = []
        with self._session() as session:
            for row in session.exec(select(ExternalDataSource)):
                interval = SYNC_INTERVALS.get(row.sync_frequency)
                if interval is None or row.status == "syncing":
                    continue
                if row.status == "error":
                    elapsed = (now - _as_utc(row.updated_at)).total_seconds()
                elif row.last_synced_at is None:
                    due.append(row)
                    continue
                else:
                    elapsed = (now - _as_utc(row.last_synced_at)).total_seconds()
                if elapsed >= interval.total_seconds():
                    due.append(row)
        return due

    def update_external_source(self, source_id: str, **fields: Any) -> None:
        """Update a warehouse source's scalar fields in place."""
        with self._write() as session:
            row = session.get(ExternalDataSource, source_id)
            if row is None:
                return
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = _now()
            session.add(row)
            session.commit()

    def update_external_schema(self, schema_id: str, **fields: Any) -> None:
        """Update a warehouse schema's scalar fields in place."""
        with self._write() as session:
            row = session.get(ExternalDataSchema, schema_id)
            if row is None:
                return
            for key, value in fields.items():
                setattr(row, key, value)
            session.add(row)
            session.commit()

    def replace_warehouse_columns(
        self, table: str, source_id: str, columns: Sequence[WarehouseColumn]
    ) -> bool:
        """Make ``columns`` the table's whole column set, replacing what was there.

        One transaction, so a reader never catches the table mid-swap. Replacing
        wholesale is what reflects added *and* removed columns without a duplicate:
        upserting what is present can never drop what is not.

        Returns whether this writer's set is the one that landed. ``_write`` serializes
        writers in *this* process, and every replica backfills at boot, so two can each
        read no rows and each insert. The unique constraint on the table catches that;
        losing means somebody else wrote the same table's columns concurrently, which is
        a race to report rather than a failure to raise through a caller's sync.
        """
        try:
            with self._write() as session:
                existing = session.exec(
                    select(WarehouseColumn).where(WarehouseColumn.table == table)
                )
                for row in existing:
                    session.delete(row)
                session.flush()
                for column in columns:
                    column.table = table
                    column.source_id = source_id
                    session.add(column)
                session.commit()
        except IntegrityError:
            logger.warning(
                "columns for %s were written concurrently; keeping the other writer's",
                table,
            )
            return False
        return True

    def list_warehouse_columns(
        self, tables: Sequence[str] | None = None
    ) -> list[WarehouseColumn]:
        """Persisted columns, for ``tables`` or for every table, in schema order."""
        with self._session() as session:
            stmt = select(WarehouseColumn)
            if tables is not None:
                if not tables:
                    return []
                stmt = stmt.where(col(WarehouseColumn.table).in_(list(tables)))
            stmt = stmt.order_by(
                col(WarehouseColumn.table), col(WarehouseColumn.ordinal)
            )
            return list(session.exec(stmt))

    def save_saved_query(self, query: SavedQuery) -> bool:
        """Insert or replace a saved query by id; whether it was written.

        A replacement keeps the original ``created_at``, so re-saving does not move a
        query to the top of a list ordered by when it was first written.
        """
        query_id = query.id
        with self._session() as session:
            existing = session.get(SavedQuery, query_id)
            prior = None if existing is None else existing.created_at
        if prior is not None:
            query.created_at = prior
        with self._write() as session:
            stale = session.get(SavedQuery, query_id)
            if stale is not None:
                session.delete(stale)
                session.commit()
            query.updated_at = _now()
            session.add(query)
            session.commit()
        return True

    def list_saved_queries(self) -> list[SavedQuery]:
        """Saved queries the caller may list, most-recently-updated first."""
        with self._session() as session:
            stmt = select(SavedQuery).where(col(SavedQuery.deleted_at).is_(None))
            stmt = stmt.order_by(col(SavedQuery.updated_at).desc())
            return list(session.exec(stmt))

    def get_saved_query(self, query_id: str) -> SavedQuery | None:
        """A saved query, or ``None`` when it is absent or trashed."""
        with self._session() as session:
            row = session.get(SavedQuery, query_id)
            if row is None or row.deleted_at is not None:
                return None
            return row

    def delete_saved_query(self, query_id: str) -> bool:
        """Delete a saved query; return whether it existed."""
        if self.get_saved_query(query_id) is None:
            return False
        with self._write() as session:
            row = session.get(SavedQuery, query_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def save_promoted_query(self, query: PromotedQuery) -> None:
        """Insert or replace a promoted external-source query, keyed by its name."""
        with self._write() as session:
            existing = session.exec(
                select(PromotedQuery).where(PromotedQuery.name == query.name)
            ).first()
            if existing is not None:
                session.delete(existing)
                session.commit()
            session.add(query)
            session.commit()

    def list_promoted_queries(self) -> list[PromotedQuery]:
        """Every promoted external-source query, for rebuilding them on startup."""
        with self._session() as session:
            return list(session.exec(select(PromotedQuery)))

    def get_promoted_query(self, name: str) -> PromotedQuery | None:
        """The promoted query of this name, or ``None``.

        Lets a derivation restore tell "this was a promoted external-source
        query" apart from "this was agent-authored" -- the two need different
        code to bring their live registration back.
        """
        with self._session() as session:
            return session.exec(
                select(PromotedQuery).where(PromotedQuery.name == name)
            ).first()

    def delete_promoted_query(self, name: str) -> bool:
        """Delete a promoted external-source query by name; return if it existed."""
        with self._write() as session:
            row = session.exec(
                select(PromotedQuery).where(PromotedQuery.name == name)
            ).first()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def upsert_metric(
        self, *, name: str, manifest_json: str, source: str | None
    ) -> bool:
        """Insert or replace a metric definition by name.

        A metric name is the primary key, so writing one replaces any definition
        already under that name.

        A trashed row of the same name is erased first rather than silently resurrected
        and overwritten: creating over it is exactly what "restore succeeds, or the old
        one is gone for good" means for a name-keyed artifact.
        """
        with self._session() as session:
            existing = session.get(MetricRow, name)
            prior = None if existing is None else existing.created_at
        created = prior if prior is not None else _now()
        erased_trashed = False
        with self._write() as session:
            stale = session.get(MetricRow, name)
            if stale is not None:
                if stale.deleted_at is not None:
                    erased_trashed = True
                    for version in session.exec(
                        select(MetricVersion).where(MetricVersion.name == name)
                    ):
                        session.delete(version)
                session.delete(stale)
                session.commit()
            session.add(
                MetricRow(
                    name=name,
                    manifest_json=manifest_json,
                    source=source,
                    created_at=created,
                )
            )
            session.commit()
        if erased_trashed:
            self.record_audit(
                "metric.erase", target_type=ResourceType.METRIC.value, target_id=name
            )
        return True

    def append_metric_version(
        self,
        *,
        name: str,
        manifest_json: str,
        definition_hash: str,
        author: str = "human",
        verdict: str | None = None,
        change_summary: str | None = None,
    ) -> bool:
        """Append a definition version, unless the content hash is unchanged.

        Returns whether a version was written: a re-save with an identical definition
        collapses onto the latest version (no history spam), mirroring dbt's
        ``state:modified`` content check.
        """
        with self._write() as session:
            prior = session.exec(
                select(MetricVersion).order_by(col(MetricVersion.version).desc())
            ).first()
            if prior is not None and prior.definition_hash == definition_hash:
                return False
            session.add(
                MetricVersion(
                    name=name,
                    version=(prior.version + 1) if prior is not None else 1,
                    manifest_json=manifest_json,
                    definition_hash=definition_hash,
                    author=author,
                    verdict=verdict,
                    change_summary=change_summary,
                )
            )
            session.commit()
            return True

    def list_metric_versions(self, name: str) -> list[MetricVersion]:
        """A metric's definition history, newest version first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(MetricVersion)
                    .where(MetricVersion.name == name)
                    .order_by(col(MetricVersion.version).desc())
                ).all()
            )

    def list_metrics(self) -> list[MetricRow]:
        """Metric definitions the caller may list, newest first."""
        with self._session() as session:
            stmt = select(MetricRow).where(col(MetricRow.deleted_at).is_(None))
            stmt = stmt.order_by(col(MetricRow.created_at).desc())
            return list(session.exec(stmt))

    def get_metric(self, name: str) -> MetricRow | None:
        """A metric definition, or ``None`` when it is absent or trashed."""
        with self._session() as session:
            row = session.get(MetricRow, name)
            if row is None or row.deleted_at is not None:
                return None
            return row

    def set_metric_copied_from(self, name: str, source: str) -> None:
        """Record that metric ``name`` was duplicated from metric ``source``."""
        with self._write() as session:
            row = session.get(MetricRow, name)
            if row is None:
                return
            row.copied_from = source
            session.add(row)
            session.commit()

    def delete_metric(self, name: str) -> bool:
        """Delete a metric definition and its version history.

        Returns whether the caller reached it at ``manage``.
        """
        if self.get_metric(name) is None:
            return False
        with self._write() as session:
            row = session.get(MetricRow, name)
            if row is None:
                return False
            for version in session.exec(
                select(MetricVersion).where(MetricVersion.name == name)
            ):
                session.delete(version)
            session.delete(row)
            session.commit()
            return True

    # -- metric monitors ---------------------------------------------------------
    def save_metric_monitor(self, monitor: MetricMonitor) -> None:
        """Insert or replace a monitor by id."""
        with self._write() as session:
            existing = session.get(MetricMonitor, monitor.id)
            if existing is not None:
                session.delete(existing)
                session.commit()
            session.add(monitor)
            session.commit()

    def update_metric_monitor(self, monitor: MetricMonitor) -> None:
        """Write a monitor's fields, leaving its snapshots and incidents alone.

        Distinct from :meth:`save_metric_monitor`, which deletes the row before writing
        it. That is fine for a fresh id and wrong for an existing monitor: the delete
        cascades to the history, and the snapshots are the baseline anomaly detection
        compares against.
        """
        with self._write() as session:
            session.merge(monitor)
            session.commit()

    def list_metric_monitors(self) -> list[MetricMonitor]:
        """The caller's monitors, newest first."""
        with self._session() as session:
            stmt = select(MetricMonitor)
            stmt = stmt.order_by(col(MetricMonitor.created_at).desc())
            return list(session.exec(stmt))

    def get_metric_monitor(self, monitor_id: str) -> MetricMonitor | None:
        """A monitor by id, or ``None``."""
        with self._session() as session:
            row = session.get(MetricMonitor, monitor_id)
            if row is None:
                return None
            return row

    def delete_metric_monitor(self, monitor_id: str) -> bool:
        """Delete a monitor and its snapshots/incidents; return whether it existed."""
        with self._write() as session:
            row = session.get(MetricMonitor, monitor_id)
            if row is None:
                return False
            for snap in session.exec(
                select(MetricSnapshot).where(MetricSnapshot.monitor_id == monitor_id)
            ):
                session.delete(snap)
            for incident in session.exec(
                select(MonitorIncident).where(MonitorIncident.monitor_id == monitor_id)
            ):
                session.delete(incident)
            session.delete(row)
            session.commit()
            return True

    def due_metric_monitors(self, now: datetime) -> list[MetricMonitor]:
        """Enabled monitors whose interval has elapsed since their last run."""
        due: list[MetricMonitor] = []
        with self._session() as session:
            for row in session.exec(select(MetricMonitor).where(MetricMonitor.enabled)):
                if row.last_run_at is None:
                    due.append(row)
                    continue
                elapsed = (now - _as_utc(row.last_run_at)).total_seconds()
                if elapsed >= row.interval_hours * 3600:
                    due.append(row)
        return due

    def touch_metric_monitor(self, monitor_id: str, when: datetime) -> None:
        """Stamp a monitor's last-run time."""
        with self._write() as session:
            row = session.get(MetricMonitor, monitor_id)
            if row is not None:
                row.last_run_at = when
                session.add(row)
                session.commit()

    def add_metric_snapshot(self, snapshot: MetricSnapshot) -> None:
        """Append a value observation for a monitor."""
        with self._write() as session:
            session.add(snapshot)
            session.commit()

    def list_metric_snapshots(
        self, monitor_id: str, limit: int = 200
    ) -> list[MetricSnapshot]:
        """A monitor's snapshots, newest first (capped)."""
        with self._session() as session:
            stmt = (
                select(MetricSnapshot)
                .where(MetricSnapshot.monitor_id == monitor_id)
                .order_by(col(MetricSnapshot.at).desc())
                .limit(limit)
            )
            return list(session.exec(stmt))

    def open_incident(self, monitor_id: str) -> MonitorIncident | None:
        """The monitor's currently-open incident, if any."""
        with self._session() as session:
            stmt = select(MonitorIncident).where(
                MonitorIncident.monitor_id == monitor_id,
                col(MonitorIncident.closed_at).is_(None),
            )
            return session.exec(stmt).first()

    def save_incident(self, incident: MonitorIncident) -> None:
        """Insert or update an incident by id.

        ``merge`` reconciles a possibly-detached instance (the open incident is read in
        one session, mutated, and saved in another), inserting when new and updating
        otherwise, without the stale-identity trap of delete-then-add.
        """
        with self._write() as session:
            session.merge(incident)
            session.commit()

    def list_incidents(
        self, monitor_id: str, limit: int = 100
    ) -> list[MonitorIncident]:
        """A monitor's incidents, newest first (capped)."""
        with self._session() as session:
            stmt = (
                select(MonitorIncident)
                .where(MonitorIncident.monitor_id == monitor_id)
                .order_by(col(MonitorIncident.opened_at).desc())
                .limit(limit)
            )
            return list(session.exec(stmt))

    def job_store(self) -> DbJobStore:
        """A durable :class:`~elbi.JobStore` over this store's database."""
        return DbJobStore(self._engine, self._lock, owner=self)

    # -- trash and erasure --------------------------------------------------------
    #
    # Soft delete for the artifacts a person authors by hand: notebooks, folders,
    # dashboards, saved queries, metrics, and feature views. A trash stamp
    # (``deleted_at``) hides a row from every read path above without
    # touching its content or history, so restore is just clearing the stamp.
    # Erasure -- immediate, unconditional removal -- calls the very same
    # ``delete_*`` cascades the retention sweep calls at the end of the trash
    # window: an accidental delete that ages out and a "delete this now, for
    # real" request are the same hard delete, only triggered on a different
    # schedule.
    #
    # Derivations get their own trash/restore/erase in the served app
    # (serve.py), not here: their live registration is an in-memory registry no
    # database row can see, so trashing one has to reach outside the store.

    def trash_notebook(self, notebook_id: str) -> None:
        """Move a notebook to trash.

        Same authorization contract :meth:`delete_notebook` has always had: the
        caller is expected to have already checked access via
        :meth:`get_notebook` (now at ``Level.MANAGE``) before calling this.
        """
        trashed = False
        with self._write() as session:
            row = session.get(Notebook, notebook_id)
            if row is not None and row.deleted_at is None:
                trashed = True
                row.deleted_at = _now()
                session.add(row)
                session.commit()
        if trashed:
            self.record_audit(
                "notebook.trash",
                target_type=ResourceType.NOTEBOOK.value,
                target_id=notebook_id,
            )

    def restore_notebook(self, notebook_id: str) -> bool:
        """Restore a trashed notebook. ``False`` if it is not there to restore."""
        with self._write() as session:
            row = session.get(Notebook, notebook_id)
            if row is None or row.deleted_at is None:
                return False
            row.deleted_at = None
            if row.folder_id is not None:
                folder = session.get(NotebookFolder, row.folder_id)
                if folder is None or folder.deleted_at is not None:
                    row.folder_id = None  # its container is gone or still trashed
            session.add(row)
            session.commit()
        self.record_audit(
            "notebook.restore",
            target_type=ResourceType.NOTEBOOK.value,
            target_id=notebook_id,
        )
        return True

    def erase_notebook(self, notebook_id: str) -> bool:
        """Permanently erase a notebook, live or trashed: the row and its cells.

        Works on a live row too, which is what makes ``permanent=true`` a real
        bypass of trash rather than a second step after it.
        """
        with self._session() as session:
            row = session.get(Notebook, notebook_id)
            if row is None:
                return False
        self.delete_notebook(notebook_id)
        self.record_audit(
            "notebook.erase",
            target_type=ResourceType.NOTEBOOK.value,
            target_id=notebook_id,
        )
        return True

    def _folder_subtree(
        self, session: Session, folder_id: str
    ) -> tuple[list[str], list[str]]:
        """The subtree rooted at ``folder_id`` (folder ids, notebook ids within it)."""
        subtree = [folder_id]
        frontier = [folder_id]
        while frontier:
            children = session.exec(
                select(NotebookFolder.id).where(
                    col(NotebookFolder.parent_id).in_(frontier)
                )
            ).all()
            subtree.extend(children)
            frontier = list(children)
        notebook_ids = list(
            session.exec(
                select(Notebook.id).where(col(Notebook.folder_id).in_(subtree))
            )
        )
        return subtree, notebook_ids

    def trash_folder(self, folder_id: str, recursive: bool = False) -> bool:
        """Move a folder to trash; with ``recursive`` also trash its whole subtree.

        Mirrors :meth:`delete_folder`'s conflict contract: ``False`` without
        trashing anything when the folder still holds live children and
        ``recursive`` is off. A missing folder is a no-op returning ``True``.
        Every row in the subtree gets the *same* trash timestamp, captured once,
        which is what lets :meth:`restore_folder` tell "went down with this
        folder" apart from "was already trashed on its own before this."
        """
        result = True
        with self._write() as session:
            row = session.get(NotebookFolder, folder_id)
            if row is None:
                return True
            if row.deleted_at is not None:
                return True  # already trashed; nothing new happened to audit
            subtree, all_notebook_ids = self._folder_subtree(session, folder_id)
            live_notebooks = [
                nb
                for nb in session.exec(
                    select(Notebook).where(col(Notebook.id).in_(all_notebook_ids))
                )
                if nb.deleted_at is None
            ]
            if not recursive and (len(subtree) > 1 or live_notebooks):
                result = False
            else:
                stamp = _now()
                for nb in live_notebooks:
                    nb.deleted_at = stamp
                    session.add(nb)
                for fid in subtree:
                    folder = session.get(NotebookFolder, fid)
                    if folder is not None and folder.deleted_at is None:
                        folder.deleted_at = stamp
                        session.add(folder)
                session.commit()
        if result:
            self.record_audit(
                "folder.trash",
                target_type=ResourceType.FOLDER.value,
                target_id=folder_id,
            )
        return result

    def restore_folder(self, folder_id: str) -> bool:
        """Restore a trashed folder and the subtree trashed along with it.

        Only descendants whose ``deleted_at`` exactly matches this folder's own
        stamp come back -- a notebook trashed on its own before the folder was
        trashed keeps its separate trash entry.
        """
        with self._write() as session:
            row = session.get(NotebookFolder, folder_id)
            if row is None or row.deleted_at is None:
                return False
            stamp = row.deleted_at
            if row.parent_id is not None:
                parent = session.get(NotebookFolder, row.parent_id)
                if parent is None or parent.deleted_at is not None:
                    row.parent_id = None
            row.deleted_at = None
            session.add(row)
            subtree, notebook_ids = self._folder_subtree(session, folder_id)
            for fid in subtree:
                if fid == folder_id:
                    continue
                folder = session.get(NotebookFolder, fid)
                if folder is not None and folder.deleted_at == stamp:
                    folder.deleted_at = None
                    session.add(folder)
            for nb in session.exec(
                select(Notebook).where(col(Notebook.id).in_(notebook_ids))
            ):
                if nb.deleted_at == stamp:
                    nb.deleted_at = None
                    session.add(nb)
            session.commit()
        self.record_audit(
            "folder.restore", target_type=ResourceType.FOLDER.value, target_id=folder_id
        )
        return True

    def erase_folder(self, folder_id: str) -> bool:
        """Permanently erase a folder and its subtree, live or trashed."""
        with self._session() as session:
            row = session.get(NotebookFolder, folder_id)
            if row is None:
                return False
        self.delete_folder(folder_id, recursive=True)
        self.record_audit(
            "folder.erase", target_type=ResourceType.FOLDER.value, target_id=folder_id
        )
        return True

    def trash_dashboard(self, dashboard_id: str) -> None:
        """Move a dashboard to trash.

        Versions and subscriptions are left alone, so a restore brings the
        whole revision history and delivery schedule back with it. Same
        authorization contract :meth:`delete_dashboard` has always had.
        """
        trashed = False
        with self._write() as session:
            row = session.get(Dashboard, dashboard_id)
            if row is not None and row.deleted_at is None:
                trashed = True
                row.deleted_at = _now()
                session.add(row)
                session.commit()
        if trashed:
            self.record_audit(
                "dashboard.trash",
                target_type=ResourceType.DASHBOARD.value,
                target_id=dashboard_id,
            )

    def restore_dashboard(self, dashboard_id: str) -> bool:
        """Restore a trashed dashboard. ``False`` if it is not there to restore."""
        with self._write() as session:
            row = session.get(Dashboard, dashboard_id)
            if row is None or row.deleted_at is None:
                return False
            row.deleted_at = None
            session.add(row)
            session.commit()
        self.record_audit(
            "dashboard.restore",
            target_type=ResourceType.DASHBOARD.value,
            target_id=dashboard_id,
        )
        return True

    def erase_dashboard(self, dashboard_id: str) -> bool:
        """Permanently erase a dashboard: its row, versions, and subscriptions."""
        with self._session() as session:
            row = session.get(Dashboard, dashboard_id)
            if row is None:
                return False
        self.delete_dashboard(dashboard_id)
        self.record_audit(
            "dashboard.erase",
            target_type=ResourceType.DASHBOARD.value,
            target_id=dashboard_id,
        )
        return True

    def trash_saved_query(self, query_id: str) -> bool:
        """Move a saved query to trash; return whether it was there to move."""
        trashed = False
        with self._write() as session:
            row = session.get(SavedQuery, query_id)
            if row is None:
                return False
            if row.deleted_at is None:
                trashed = True
                row.deleted_at = _now()
                session.add(row)
                session.commit()
        if trashed:
            self.record_audit(
                "saved_query.trash",
                target_type=ResourceType.SAVED_QUERY.value,
                target_id=query_id,
            )
        return True

    def restore_saved_query(self, query_id: str) -> bool:
        """Restore a trashed saved query. ``False`` if it is not there to restore."""
        with self._write() as session:
            row = session.get(SavedQuery, query_id)
            if row is None or row.deleted_at is None:
                return False
            row.deleted_at = None
            session.add(row)
            session.commit()
        self.record_audit(
            "saved_query.restore",
            target_type=ResourceType.SAVED_QUERY.value,
            target_id=query_id,
        )
        return True

    def erase_saved_query(self, query_id: str) -> bool:
        """Permanently erase a saved query, live or trashed."""
        with self._session() as session:
            row = session.get(SavedQuery, query_id)
            if row is None:
                return False
        self.delete_saved_query(query_id)
        self.record_audit(
            "saved_query.erase",
            target_type=ResourceType.SAVED_QUERY.value,
            target_id=query_id,
        )
        return True

    def trash_metric(self, name: str) -> bool:
        """Move a metric to trash; return whether it was there to move."""
        trashed = False
        with self._write() as session:
            row = session.get(MetricRow, name)
            if row is None:
                return False
            if row.deleted_at is None:
                trashed = True
                row.deleted_at = _now()
                session.add(row)
                session.commit()
        if trashed:
            self.record_audit(
                "metric.trash", target_type=ResourceType.METRIC.value, target_id=name
            )
        return True

    def restore_metric(self, name: str) -> bool:
        """Restore a trashed metric. ``False`` if it is not there to restore."""
        with self._write() as session:
            row = session.get(MetricRow, name)
            if row is None or row.deleted_at is None:
                return False
            row.deleted_at = None
            session.add(row)
            session.commit()
        self.record_audit(
            "metric.restore", target_type=ResourceType.METRIC.value, target_id=name
        )
        return True

    def erase_metric(self, name: str) -> bool:
        """Permanently erase a metric and its version history, live or trashed."""
        with self._session() as session:
            row = session.get(MetricRow, name)
            if row is None:
                return False
        self.delete_metric(name)
        self.record_audit(
            "metric.erase", target_type=ResourceType.METRIC.value, target_id=name
        )
        return True

    def trash_feature_view(self, name: str) -> bool:
        """Move a feature view to trash; return whether it was there to move."""
        trashed = False
        with self._write() as session:
            row = session.get(FeatureViewRow, name)
            if row is None:
                return False
            if row.deleted_at is None:
                trashed = True
                row.deleted_at = _now()
                session.add(row)
                session.commit()
        if trashed:
            self.record_audit(
                "feature_view.trash",
                target_type=ResourceType.FEATURE_VIEW.value,
                target_id=name,
            )
        return True

    def restore_feature_view(self, name: str) -> bool:
        """Restore a trashed feature view. ``False`` if it is not there to restore."""
        with self._write() as session:
            row = session.get(FeatureViewRow, name)
            if row is None or row.deleted_at is None:
                return False
            row.deleted_at = None
            session.add(row)
            session.commit()
        self.record_audit(
            "feature_view.restore",
            target_type=ResourceType.FEATURE_VIEW.value,
            target_id=name,
        )
        return True

    def erase_feature_view(self, name: str) -> bool:
        """Permanently erase a feature view and its child tables, live or trashed."""
        with self._session() as session:
            row = session.get(FeatureViewRow, name)
            if row is None:
                return False
        self.delete_feature_view(name)
        self.record_audit(
            "feature_view.erase",
            target_type=ResourceType.FEATURE_VIEW.value,
            target_id=name,
        )
        return True

    def derivation_origin(self, name: str) -> str | None:
        """A derivation's ``origin`` regardless of trash state, or ``None`` if missing.

        Deliberately bypasses the live-only filter :meth:`get_derivation` applies:
        the trash/restore/erase route needs to see a trashed *or* repo-origin row
        to tell a genuinely missing name (404) apart from one that exists but is
        refused (409, for a repo-origin row) before calling any of the methods
        below.
        """
        with self._session() as session:
            row = session.get(Derivation, name)
            return None if row is None else row.origin

    def trash_derivation(self, name: str) -> bool:
        """Move an agent-authored or promoted derivation's row to trash.

        Only the database half. Removing it from the live registry and moving
        its sidecar aside is the served app's job (serve.py), which is the only
        place holding both; this file has no reach into either. A repo-origin
        row is refused here as well as at the route: :meth:`sync_repo_derivations`
        wipes and rebuilds those rows on every start, which would silently clear
        the stamp, and the MCP delete path never runs the route's pre-check.
        """
        trashed = False
        with self._write() as session:
            row = session.get(Derivation, name)
            if row is not None and row.deleted_at is None and row.origin != "repo":
                trashed = True
                row.deleted_at = _now()
                session.add(row)
                session.commit()
        if trashed:
            self.record_audit(
                "derivation.trash",
                target_type=ResourceType.DERIVATION.value,
                target_id=name,
            )
        return trashed

    def restore_derivation(self, name: str) -> bool:
        """Restore a trashed derivation's row. ``False`` if not there to restore.

        Only the database half; re-registering it live is the served app's job,
        run after this returns ``True``.
        """
        with self._write() as session:
            row = session.get(Derivation, name)
            if row is None or row.deleted_at is None:
                return False
            row.deleted_at = None
            session.add(row)
            session.commit()
        self.record_audit(
            "derivation.restore",
            target_type=ResourceType.DERIVATION.value,
            target_id=name,
        )
        return True

    def erase_derivation(self, name: str) -> bool:
        """Permanently erase a derivation's row and run history, live or trashed.

        Only the database half; invalidating its cache, removing its sidecar,
        and unregistering it live are the served app's job, run after this
        returns ``True``.
        """
        with self._session() as session:
            row = session.get(Derivation, name)
            if row is None:
                return False
        self.delete_derivation(name)
        self.record_audit(
            "derivation.erase",
            target_type=ResourceType.DERIVATION.value,
            target_id=name,
        )
        return True

    def list_trash(self) -> list[dict[str, Any]]:
        """Trashed artifacts, newest first: kind, id, name, and when.

        A folder or notebook trashed as part of a folder's subtree is left off this
        list on its own -- only the subtree's root appears, found by comparing its
        exact trash timestamp against its container's.
        """
        with self._session() as session:
            items: list[dict[str, Any]] = []

            folders = list(
                session.exec(
                    select(NotebookFolder).where(
                        col(NotebookFolder.deleted_at).is_not(None)
                    )
                )
            )
            folder_by_id = {f.id: f for f in folders}
            for folder_row in folders:
                parent = (
                    folder_by_id.get(folder_row.parent_id)
                    if folder_row.parent_id
                    else None
                )
                if parent is not None and parent.deleted_at == folder_row.deleted_at:
                    continue  # a subtree member; its root is listed instead
                items.append(
                    _trash_entry("folder", folder_row.id, folder_row.name, folder_row)
                )

            for nb_row in session.exec(
                select(Notebook).where(col(Notebook.deleted_at).is_not(None))
            ):
                container = (
                    folder_by_id.get(nb_row.folder_id) if nb_row.folder_id else None
                )
                if container is not None and container.deleted_at == nb_row.deleted_at:
                    continue  # trashed with its folder, already represented above
                items.append(_trash_entry("notebook", nb_row.id, nb_row.name, nb_row))

            dashboard_stmt = select(Dashboard).where(
                col(Dashboard.deleted_at).is_not(None)
            )
            for dash_row in session.exec(dashboard_stmt):
                items.append(
                    _trash_entry("dashboard", dash_row.id, dash_row.name, dash_row)
                )

            query_stmt = select(SavedQuery).where(
                col(SavedQuery.deleted_at).is_not(None)
            )
            for query_row in session.exec(query_stmt):
                items.append(
                    _trash_entry("saved_query", query_row.id, query_row.name, query_row)
                )

            metric_stmt = select(MetricRow).where(
                col(MetricRow.deleted_at).is_not(None)
            )
            for metric_row in session.exec(metric_stmt):
                items.append(
                    _trash_entry("metric", metric_row.name, metric_row.name, metric_row)
                )

            fv_stmt = select(FeatureViewRow).where(
                col(FeatureViewRow.deleted_at).is_not(None)
            )
            for fv_row in session.exec(fv_stmt):
                items.append(
                    _trash_entry("feature_view", fv_row.name, fv_row.name, fv_row)
                )

            deriv_stmt = select(Derivation).where(
                col(Derivation.deleted_at).is_not(None)
            )
            for deriv_row in session.exec(deriv_stmt):
                items.append(
                    _trash_entry(
                        "derivation", deriv_row.name, deriv_row.name, deriv_row
                    )
                )

            items.sort(key=lambda item: item["deleted_at"], reverse=True)
            return items

    #: Kind name to (restore, erase) methods, for the generic ``/api/trash``
    #: routes. Excludes ``derivation``, whose restore/erase also has to touch the
    #: live registry and is handled in the served app instead.
    def _trash_dispatch(
        self, kind: str
    ) -> tuple[Callable[..., bool], Callable[..., bool]]:
        table: dict[str, tuple[Callable[..., bool], Callable[..., bool]]] = {
            "notebook": (self.restore_notebook, self.erase_notebook),
            "folder": (self.restore_folder, self.erase_folder),
            "dashboard": (self.restore_dashboard, self.erase_dashboard),
            "saved_query": (self.restore_saved_query, self.erase_saved_query),
            "metric": (self.restore_metric, self.erase_metric),
            "feature_view": (self.restore_feature_view, self.erase_feature_view),
        }
        if kind not in table:
            raise ValueError(f"unknown trash kind: {kind}")
        return table[kind]

    def restore_trashed(self, kind: str, item_id: str) -> bool:
        """Restore a trashed artifact by kind and id.

        ``False`` if not there to restore.
        """
        restore, _ = self._trash_dispatch(kind)
        return restore(item_id)

    def erase_trashed(self, kind: str, item_id: str) -> bool:
        """Permanently erase a trashed artifact by kind and id.

        ``False`` if not there.
        """
        _, erase = self._trash_dispatch(kind)
        return erase(item_id)

    def expired_trashed_derivations(self, older_than_days: int) -> list[str]:
        """Names of derivations trashed more than ``older_than_days`` ago.

        Derivations are not in :meth:`purge_trash` because erasing one needs the
        served app's runtime hooks (the live registry, the sidecar), which this
        module cannot reach. This is the query half; the app's maintenance tick
        pairs it with :meth:`erase_derivation` and the runtime cleanup.
        """
        if older_than_days <= 0:
            return []
        cutoff = _now() - timedelta(days=older_than_days)
        with self._session() as session:
            return list(
                session.exec(
                    select(Derivation.name).where(
                        col(Derivation.deleted_at).is_not(None),
                        col(Derivation.deleted_at) < cutoff,
                    )
                )
            )

    def purge_trash(self, older_than_days: int) -> int:
        """Permanently erase everything trashed more than ``older_than_days`` ago.

        Follows :meth:`prune_audit`'s shape: ``older_than_days <= 0`` disables the
        sweep and returns 0. Delegates to :meth:`erase_trashed`, the same cascade
        a manual "delete forever" uses, so the retention window and immediate
        erasure are provably the same hard delete on different schedules. Each
        erase re-checks that its row still exists, so a notebook swept up early
        as part of its folder's subtree is skipped harmlessly when its own turn
        comes around in the ``notebook`` pass.
        """
        if older_than_days <= 0:
            return 0
        cutoff = _now() - timedelta(days=older_than_days)
        purged = 0
        for kind, model, id_column in (
            ("notebook", Notebook, Notebook.id),
            ("folder", NotebookFolder, NotebookFolder.id),
            ("dashboard", Dashboard, Dashboard.id),
            ("saved_query", SavedQuery, SavedQuery.id),
            ("metric", MetricRow, MetricRow.name),
            ("feature_view", FeatureViewRow, FeatureViewRow.name),
        ):
            with self._session() as session:
                expired = list(
                    session.exec(
                        select(id_column).where(
                            col(model.deleted_at).is_not(None),
                            col(model.deleted_at) < cutoff,
                        )
                    )
                )
            for item_id in expired:
                if self.erase_trashed(kind, item_id):
                    purged += 1
        return purged


def _row_to_job(row: JobRow) -> Job:
    return Job(
        id=row.id,
        key=row.key,
        label=row.label,
        state=cast(JobState, row.state),
        progress=row.progress,
        result=None if row.result_json is None else json.loads(row.result_json),
        error=row.error,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


class DbJobStore:
    """A database-backed :class:`~elbi.JobStore`: jobs survive a restart.

    Writes are serialized on a shared lock so several worker threads updating job state
    do not trip SQLite's single-writer locking; reads go straight to the database. Each
    operation returns the immutable :class:`~elbi.Job`, so a reader never sees a
    half-written row.
    """

    def __init__(
        self, engine: Engine, lock: threading.Lock, owner: Store | None = None
    ) -> None:
        self._engine = engine
        self._lock = lock
        # The engine belongs to the Store that handed it over, and that Store disposes
        # the pool once it is unreachable. Holding it keeps the engine alive for a job
        # store that outlives the expression that made it -- `open_store(...)
        # .job_store()` drops the Store immediately otherwise, and every later query
        # opens a connection with no owner left to close it.
        self._owner = owner

    def create(self, job: Job) -> None:
        """Persist a newly submitted job."""
        with self._lock, Session(self._engine) as session:
            session.add(_job_to_row(job))
            session.commit()

    def get(self, job_id: str) -> Job | None:
        """The job of this id, or ``None``."""
        with Session(self._engine) as session:
            row = session.get(JobRow, job_id)
            return None if row is None else _row_to_job(row)

    def find_by_key(self, key: str) -> Job | None:
        """The most recent job submitted under ``key`` (for dedupe)."""
        with Session(self._engine) as session:
            stmt = (
                select(JobRow)
                .where(JobRow.key == key)
                .order_by(JobRow.created_at.desc())  # type: ignore[attr-defined]
            )
            row = session.exec(stmt).first()
            return None if row is None else _row_to_job(row)

    def update(self, job_id: str, **changes: Any) -> Job:
        """Apply ``changes`` to the stored job and return the new value."""
        with self._lock, Session(self._engine) as session:
            row = session.get(JobRow, job_id)
            if row is None:
                raise KeyError(job_id)
            for name, value in changes.items():
                if name == "result":
                    row.result_json = None if value is None else json.dumps(value)
                else:
                    setattr(row, name, value)
            session.add(row)
            session.commit()
            session.refresh(row)
            return _row_to_job(row)

    def list(self) -> list[Job]:
        """Every job, newest first."""
        with Session(self._engine) as session:
            stmt = select(JobRow).order_by(JobRow.created_at.desc())  # type: ignore[attr-defined]
            return [_row_to_job(row) for row in session.exec(stmt)]


def _job_to_row(job: Job) -> JobRow:
    return JobRow(
        id=job.id,
        key=job.key,
        label=job.label,
        state=job.state,
        progress=job.progress,
        result_json=None if job.result is None else json.dumps(job.result),
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def open_store(
    db_uri: str | None = None, *, store_class: type[Store] | None = None
) -> Store:
    """Open (and, on first use, create) the store named by ``db_uri`` or ``DB_URI``.

    ``store_class`` builds the store, so an extension that adds tables and queries of
    its own can serve them from the one engine rather than opening a second against the
    same file. Omitted, an installed extension supplying one decides, and otherwise it
    is :class:`Store`.

    Resolved before the schema is created, because loading the extension is what
    registers its tables on the metadata that ``migrate`` then creates.
    """
    from .extensions import store_class as extension_store_class

    if store_class is None:
        store_class = extension_store_class() or Store
    uri = db_uri or os.environ.get("DB_URI") or DEFAULT_DB_URI
    url = sqlalchemy_url(uri)
    prefix = "sqlite:///"
    if url.startswith(prefix):  # ensure the SQLite file's directory exists
        path = url[len(prefix) :]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    _enable_sqlite_wal(engine)
    migrate(engine)
    return store_class(engine)


#: Advisory-lock key for the schema migration. The value is arbitrary; it only has to
#: stay stable across releases so every process contends for the same lock.
_MIGRATION_LOCK_KEY = 8231907445120733


def migrate(engine: Engine) -> None:
    """Bring the schema up to date. Safe to run concurrently and repeatedly.

    Every process that opens the store runs this, so replicas booting together would
    otherwise race: each inspects a table, each sees the same column missing, and each
    issues the same ``ALTER TABLE``. One wins and the others error. On Postgres an
    advisory lock serializes them: a loser waits, then re-inspects and finds nothing
    left to do.

    The lock is transaction-scoped, not session-scoped, so ``COMMIT`` releases it. A
    session-scoped lock is never released when a connection pooler in transaction mode
    hands the connection back between statements, which wedges every later boot.

    All of it runs on one connection inside one transaction: the lock has to be held
    across the inspect-then-alter sequence to close the race, and Postgres DDL is
    transactional, so a failure part-way leaves no half-migrated schema.

    SQLite needs no lock. It is single-node, and its own write lock plus the busy
    timeout from ``_enable_sqlite_wal`` already serialize writers.
    """
    with engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": _MIGRATION_LOCK_KEY}
            )
        SQLModel.metadata.create_all(conn)
        _migrate_conversation_updated_at(conn)
        _migrate_add_column(conn, "conversation", "profile", "VARCHAR")
        _migrate_add_column(conn, "conversation", "prompt_tokens", "INTEGER")
        _migrate_add_column(conn, "conversation", "completion_tokens", "INTEGER")
        _migrate_add_column(conn, "conversation", "cost", "FLOAT")
        _migrate_add_column(conn, "conversation", "summary", "VARCHAR")
        _migrate_add_column(conn, "conversation", "summary_turns", "INTEGER")
        _migrate_add_column(conn, "message", "feedback", "VARCHAR")
        _migrate_add_column(conn, "llmprofile", "reasoning_effort", "VARCHAR")
        _migrate_add_column(conn, "notebook", "lock_json", "VARCHAR")
        _migrate_add_column(conn, "notebook", "folder_id", "VARCHAR")
        _migrate_add_column(conn, "notebook", "compute_profile", "VARCHAR")
        _migrate_add_column(conn, "compute_usage", "kind", "VARCHAR")
        _migrate_add_column(conn, "retrainpolicy", "source_kind", "VARCHAR")
        _migrate_add_column(conn, "retrainpolicy", "engine", "VARCHAR")
        _migrate_add_column(conn, "retrainpolicy", "groups", "VARCHAR")
        _migrate_add_column(conn, "feature_view", "contract_json", "VARCHAR")
        _migrate_add_column(conn, "orchestration_run", "parent_run_id", "VARCHAR")
        _migrate_add_column(conn, "asset_run", "logs", "VARCHAR")
        _migrate_add_column(conn, "asset_run", "checks_json", "VARCHAR")
        _migrate_add_column(conn, "orchestration_run", "workflow_id", "VARCHAR")
        _migrate_add_column(conn, "orchestration_run", "workflow_steps_json", "VARCHAR")
        _migrate_add_column(conn, "notebook", "copied_from", "VARCHAR")
        _migrate_add_column(conn, "dashboard", "copied_from", "VARCHAR")
        _migrate_add_column(conn, "saved_query", "copied_from", "VARCHAR")
        _migrate_add_column(conn, "metric", "copied_from", "VARCHAR")
        for _trash_table in (
            "notebook",
            "notebook_folder",
            "dashboard",
            "saved_query",
            "metric",
            "feature_view",
            "derivation",
        ):
            _migrate_add_column(conn, _trash_table, "deleted_at", "TIMESTAMP")
        _migrate_rebuild_budget(conn)
        for _stale_table, _stale_column in _undeclared_columns(conn):
            _migrate_retire_column(conn, _stale_table, _stale_column)


def _undeclared_columns(conn: Connection) -> list[tuple[str, str]]:
    """Columns the database has that this version's models no longer declare.

    ``create_all`` only ever adds, so a column dropped from a model survives in a
    database an earlier version made. At best it is dead weight; one that is NOT NULL
    rejects every insert the current code writes, because that code no longer supplies
    it, which is a store that opens and then cannot be written to.

    Derived by comparing the live schema against the declared one rather than read from
    a list, so a column nobody remembered to write down is found too. Only tables the
    models declare are considered, and within them the models are the whole truth:
    anything else in the database belongs to whoever put it there and is left alone.
    """
    inspector = inspect(conn)
    present = set(inspector.get_table_names())
    stale: list[tuple[str, str]] = []
    for name, model in SQLModel.metadata.tables.items():
        if name not in present:
            continue
        declared = {column.name for column in model.columns}
        stale.extend(
            (name, column["name"])
            for column in inspector.get_columns(name)
            if column["name"] not in declared
        )
    return stale


def _migrate_rebuild_budget(conn: Connection) -> None:
    """Rebuild the budget table when it is still keyed by subject, carrying the cap.

    A cap used to be held per subject, so the table was keyed by one; there is a single
    caller now, and the key is fixed. That is a change of primary key rather than a
    dropped column, which no ``ALTER`` expresses -- SQLite refuses to drop a key column
    at all -- so the table is rebuilt.

    The instance-wide row carries over. It is the one whose meaning survives: a cap on
    the whole deployment is a cap on its one caller, where a cap on a particular user is
    a cap on somebody who no longer exists to this schema.
    """
    inspector = inspect(conn)
    if "budget" not in inspector.get_table_names():
        return
    if "subject_type" not in {c["name"] for c in inspector.get_columns("budget")}:
        return
    carried = conn.execute(
        text(
            "SELECT max_budget, window, spend FROM budget"
            " WHERE subject_type = 'global' LIMIT 1"
        )
    ).first()
    table = SQLModel.metadata.tables["budget"]
    conn.execute(text("DROP TABLE budget"))
    table.create(conn)
    if carried is not None:
        conn.execute(
            insert(table).values(
                id=BUDGET_ID,
                max_budget=carried[0],
                window=carried[1],
                window_start=_now(),
                spend=carried[2],
            )
        )


def _migrate_retire_column(conn: Connection, table: str, column: str) -> None:
    """Remove a column this version's models no longer declare.

    ``create_all`` only ever adds, so a column left behind survives in a database made
    by an earlier version. A nullable leftover is merely dead weight, but one declared
    NOT NULL without a default rejects every insert the current code writes, because
    that code no longer supplies it -- which is a store that opens and then cannot be
    written to.

    SQLite refuses to drop a column an index depends on, so its indexes go first.
    Postgres drops them with the column, and asking for them there is a no-op, so the
    lookup is skipped rather than guarded.

    Names go through the dialect's own quoting, for the reason
    :func:`_migrate_add_column` explains.
    """
    inspector = inspect(conn)
    if table not in inspector.get_table_names():
        return
    if column not in {c["name"] for c in inspector.get_columns(table)}:
        return
    if column in (inspector.get_pk_constraint(table).get("constrained_columns") or ()):
        # Part of the key, which is a change of identity rather than a dropped field.
        # No ``ALTER`` expresses it and SQLite refuses outright, so the table is rebuilt
        # from the model and the rows carried across on the columns they share.
        _migrate_rebuild_table(conn, table)
        return
    quote = conn.dialect.identifier_preparer.quote
    if conn.dialect.name == "sqlite":
        for index in inspector.get_indexes(table):
            if index["name"] and column in (index["column_names"] or ()):
                conn.execute(text(f"DROP INDEX {quote(index['name'])}"))
    conn.execute(text(f"ALTER TABLE {quote(table)} DROP COLUMN {quote(column)}"))


def _migrate_rebuild_table(conn: Connection, table: str) -> None:
    """Replace a table with the model's definition, carrying the rows it can.

    Only the columns both shapes share come across; a row whose identity depended on a
    column the model dropped cannot be carried, since the new key would not distinguish
    it from its siblings. Those rows are left behind rather than collapsed onto one
    another, which would invent a row nobody wrote.
    """
    model = SQLModel.metadata.tables.get(table)
    if model is None:
        return
    quote = conn.dialect.identifier_preparer.quote
    existing = Table(table, MetaData(), autoload_with=conn)
    shared = [c.name for c in model.columns if c.name in existing.columns]
    staging = model.to_metadata(MetaData(), name=f"{table}__rebuilt")
    staging.drop(conn, checkfirst=True)
    staging.create(conn)
    if shared:
        carried = select(*(existing.columns[c] for c in shared))
        keys = [c.name for c in model.primary_key.columns if c.name in shared]
        if keys:
            # One row per new key: rows that differed only by a dropped key column are
            # no longer distinguishable, and inserting them all would collide.
            carried = carried.group_by(*(existing.columns[k] for k in keys))
        conn.execute(insert(staging).from_select(shared, carried))
    existing.drop(conn)
    conn.execute(text(f"ALTER TABLE {quote(staging.name)} RENAME TO {quote(table)}"))


def _migrate_add_column(
    conn: Connection, table: str, column: str, sql_type: str, default: str | None = None
) -> None:
    """Add a nullable ``column`` to ``table`` on a database created before it.

    ``create_all`` does not alter an existing table, so a store opened on an older
    database is missing the column and every query on it would fail. Add it when absent.

    ``default`` is SQL, and both fills existing rows and applies to later inserts that
    omit the column. Needed where the model's own default is load-bearing: a column the
    code compares against a value cannot arrive as NULL on rows that predate it, or
    every such comparison quietly answers false.

    Takes a connection rather than an engine so the inspect-then-alter pair runs inside
    the caller's locked transaction; inspecting on a separate connection would reopen
    the race the lock exists to close.

    Names go through the dialect's own quoting, which is what lets this touch a table
    called ``user``: a reserved word in Postgres, where the statement fails unquoted,
    while SQLite needs no quotes and does not want them. The dialect knows which.
    """
    inspector = inspect(conn)
    if table not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns(table)}
    if column in columns:
        return
    quote = conn.dialect.identifier_preparer.quote
    clause = f"ALTER TABLE {quote(table)} ADD COLUMN {quote(column)} {sql_type}"
    if default is not None:
        clause += f" DEFAULT {default}"
    conn.execute(text(clause))


def _migrate_conversation_updated_at(conn: Connection) -> None:
    """Add ``Conversation.updated_at`` to a database created before the column existed.

    ``create_all`` does not alter an existing table, so a store opened on an older
    database would be missing the column and every query on it would fail. Add it when
    absent and backfill from ``created_at``, so the recent-activity ordering has a value
    for every existing conversation.
    """
    inspector = inspect(conn)
    if "conversation" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("conversation")}
    if "updated_at" in columns:
        return
    conn.execute(text("ALTER TABLE conversation ADD COLUMN updated_at TIMESTAMP"))
    conn.execute(text("UPDATE conversation SET updated_at = created_at"))


def _enable_sqlite_wal(engine: Engine) -> None:
    """Put a SQLite engine in WAL mode with a busy timeout (no effect on Postgres).

    Under the default rollback journal a writer blocks all readers; WAL lets readers run
    while a writer holds the database, which is what the app does (a request reads while
    a background-job write is in flight). The busy timeout makes a writer wait for a
    contended lock instead of failing at once with "database is locked".
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()
