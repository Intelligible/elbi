"""Pydantic response models for the JSON API.

Each model declares the field names a route puts on the wire, so the OpenAPI schema
carries them and FastAPI validates what the handler returns.

``alias_generator=to_camel`` serializes every field to camelCase, matching the API's
convention (see :mod:`elbi.casing`); ``populate_by_name`` keeps the Python side writing
snake_case.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class Wire(BaseModel):
    """Base for a camelCased response model: camelCase out, snake_case in Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class Bare(BaseModel):
    """Base for a response model serving both camelCased and exempt routes.

    Its fields are single words, which the two conventions spell alike, so it carries no
    alias generator.
    """

    model_config = ConfigDict(extra="forbid")


class Ok(Bare):
    """The answer of a route whose only result is that it worked."""

    ok: bool = True


class Created(Bare):
    """The id of a newly created object."""

    id: str


class Budget(Wire):
    """The spend cap and what has been spent against it."""

    max_budget: float
    window: str
    spend: float


class MlflowSettings(Wire):
    """Where run tracking points, and whether this deployment may change it."""

    tracking_uri: str
    source: str
    editable: bool = True


class WebhookSettings(Wire):
    """The webhook endpoint, and whether a signing secret is set."""

    url: str
    secret_set: bool


class DerivationSummary(Wire):
    """A derivation as it appears in a list."""

    name: str
    question: str
    verdict: str | None = None
    data_hash: str | None = None
    created_at: str
    origin: str | None = None


class DerivationDetail(DerivationSummary):
    """A derivation in full: its source, claim, output and attestation."""

    conversation_id: str | None = None
    source: str
    claim: dict[str, Any] | None = None
    serve: dict[str, Any] | None = None
    narrative: str
    #: None when an extension withholds it; see ``withhold_rendering``.
    rendered: str | None = None
    attestation: dict[str, Any] | None = None
    assumptions: list[str] = []


class Job(Wire):
    """A background job, its state and its result."""

    id: str
    label: str
    state: str
    progress: str
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class DataSource(Wire):
    """A registered data source. The secret is reported as set, never returned."""

    id: str
    name: str
    kind: str
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    secret_set: bool
    secret_env: str | None = None
    extra: str | None = None


class Install(Wire):
    """How this copy was installed, and the command that upgrades it."""

    kind: str
    command: str
    note: str = ""


class Version(Wire):
    """What is running here."""

    version: str
    package: str
    install: Install


class ComputeProfiles(Wire):
    """The compute menu.

    A profile's own keys vary with the ``hidden`` set an administrator configures, so
    they are carried through unmodelled.
    """

    default: str
    max_cost_per_hour: float | None = None
    profiles: list[dict[str, Any]]


class ComputeSession(Wire):
    """One finished compute session and what it cost."""

    ended_at: str
    kind: str
    notebook_id: str | None = None
    profile: str
    profile_version: str
    seconds: float
    cost: float


class ComputeUsage(Wire):
    """Compute spend over a window, by kind and by session.

    ``alert`` is the over-limit message, or None while spend is under the limit.
    """

    since: str
    spend: float
    by_kind: dict[str, float]
    limit: float | None = None
    alert: str | None = None
    sessions: list[ComputeSession]


class NotificationItem(Wire):
    """One notification and its read state."""

    id: str
    at: str
    event_type: str
    title: str
    body: str
    target_type: str
    target_id: str
    verdict: str | None = None
    read_at: str | None = None


class NotificationList(Wire):
    """Notifications, newest first, with the unread count."""

    items: list[NotificationItem]
    unread: int


class NotificationPref(Wire):
    """One event type's delivery switches."""

    event_type: str
    in_app: bool
    email: bool


class NotificationPreferences(Wire):
    """The full per-event-type matrix, and whether email can deliver at all."""

    email_available: bool
    prefs: list[NotificationPref]


class Dataset(Wire):
    """A queryable table and its size."""

    name: str
    rows: int
    columns: int
    origin: str


class GraphNode(Wire):
    """One artifact in the lineage graph."""

    id: str
    type: str
    name: str
    verdict: str | None = None
    certified: bool | None = None
    description: str | None = None
    copied_from: str | None = None


class GraphEdge(Wire):
    """A dependency between two artifacts."""

    source: str
    target: str
    kind: str


class LineageGraph(Wire):
    """The artifacts and the dependencies between them."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class Monitor(Wire):
    """A watch on a metric or derivation, and the state of its last check."""

    id: str
    name: str
    target_kind: str
    target: str
    config: dict[str, Any]
    method: str
    sensitivity: float
    min_value: float | None = None
    max_value: float | None = None
    window: int
    interval_hours: float
    enabled: bool
    last_value: float | None = None
    last_checked_at: str | None = None
    status: str


class MonitorSnapshot(Wire):
    """One observed value, and whether it read as an anomaly."""

    at: str | None = None
    value: float
    anomalous: bool
    baseline: float | None = None
    lower: float | None = None
    upper: float | None = None
    score: float | None = None
    reason: str
    source_verdict: str | None = None


class MonitorIncident(Wire):
    """A run of anomalous values, from the first to the one that closed it."""

    id: str
    opened_at: str | None = None
    closed_at: str | None = None
    peak_value: float
    peak_score: float | None = None
    reason: str
    snapshots: int
    open: bool


class MonitorHistory(Wire):
    """A monitor's observed values and the incidents they opened."""

    snapshots: list[MonitorSnapshot]
    incidents: list[MonitorIncident]


class AuditEvent(Wire):
    """One audit entry: what ran, against what, and whether it was sound."""

    id: str
    at: str
    action: str
    target_type: str | None = None
    target_id: str | None = None
    verdict: str | None = None
    data_hash: str | None = None
