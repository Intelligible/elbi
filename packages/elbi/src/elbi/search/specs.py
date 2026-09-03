"""What platform search indexes, declared column by column.

Exclusion is the default here, and that is the whole design. Nothing in this module or
the extractor reflects over a model: there is no ``model_dump``, no ``__dict__``, no
``SELECT *``. A spec names the columns it wants and the extractor selects exactly those,
so a column added to a table in a year's time cannot reach the index by being there. A
secret is not kept out by a denylist that somebody has to remember to update; it is kept
out by never having been asked for.

Two further guards sit on top, in ``tests/test_search_specs.py``. Every table in the
store is either specified here or listed in :data:`NOT_INDEXED` with a reason, so adding
one fails the suite until someone decides. And a declared column whose name looks like a
credential, or which holds JSON, is rejected unless it is acknowledged explicitly and
carries a transform, so the next ``*_json`` column is excluded by default too.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlmodel import SQLModel

from .. import db
from ..resources import ResourceType
from .schema import FIELD_BODY, FIELD_TITLE


@dataclass(frozen=True)
class TextField:
    """One column whose text is indexed, and which field it contributes to."""

    column: str
    weight: str = FIELD_BODY
    #: Applied before indexing. The only way a JSON or otherwise structured column may
    #: be used, so that using one is a deliberate, reviewable act.
    transform: Callable[[Any], str] | None = None


@dataclass(frozen=True)
class FacetField:
    """One column that constrains results rather than matching them."""

    facet: str
    column: str
    #: Applied before the value is stored, where the column is not already the facet.
    #: An incident is open or closed; the column holding that is a nullable timestamp,
    #: and without this the facet would quietly be a datetime.
    derive: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class Chunk:
    """How one entity's row becomes documents, and why it was decided that way.

    Three modes, and the reason is required because the granularity is a judgement per
    entity rather than a default:

    ``whole``
        The row is one document. Right for anything whose text is a name plus a short
        description, which is most of the corpus.
    ``field``
        One document per declared text field, so text types with different shapes do not
        dilute each other -- a derivation's Python source and its prose narrative score
        as separate documents that share a title.
    ``split``
        Long free text cut into overlapping windows. Truncation was the alternative and
        it drops everything past its limit; a window keeps it, and keeps a dense vector
        meaningful where one over a whole long document averages it away.

    ``window`` is characters, not tokens: the tokenizer belongs to the indexer.
    """

    mode: str  # whole | field | split
    reason: str
    window: int = 0
    overlap: int = 0
    #: Embed each chunk in the context of the whole row rather than on its own. A cell
    #: reading "then we filtered to Q3" says nothing alone; pooled over its notebook it
    #: keeps what the notebook was about. Needs a
    #: :class:`~elbi_core.retrieval.SpanEmbedder`; the static fallback embedder
    #: has no context to pool, so it degrades to per-chunk rather than failing.
    late: bool = False


def windows(length: int, chunk: Chunk) -> list[tuple[int, int]]:
    """Overlapping spans covering ``length`` characters, per the chunk's declaration.

    Here rather than in the extractor because it *is* what ``window`` and ``overlap``
    mean, and the extractor and the size estimate must read them the same way -- else
    the backfill's counts and the estimate they are compared against disagree by
    construction.

    A stride of zero or less never advances, so it degrades to one window: a misdeclared
    spec indexes coarsely rather than hanging.
    """
    stride = chunk.window - chunk.overlap
    if length <= chunk.window or stride <= 0:
        return [(0, max(length, 0))]
    spans = []
    start = 0
    while start < length:
        end = min(start + chunk.window, length)
        spans.append((start, end))
        if end == length:
            break
        start += stride
    return spans


#: Most of the corpus: a name and a short description, indexed as one document.
WHOLE = Chunk("whole", reason="a name and a short description")

#: Prose long enough that a single vector would average it away. 2000 characters is
#: roughly 400 tokens, comfortably inside the static embedder's useful range, and the
#: overlap keeps a sentence spanning a boundary findable from either side.
_WINDOW = 2000
_OVERLAP = 200


@dataclass(frozen=True)
class SqlSpec:
    """How one persisted entity becomes searchable documents.

    ``resource_type`` is the vocabulary this entity is named by where it differs from
    its own ``entity_type`` -- a ``notebook_folder`` is a ``folder`` everywhere outside
    the table name.
    """

    entity_type: str
    model: type[SQLModel]
    id_column: str
    text: tuple[TextField, ...]
    resource_type: ResourceType | None = None
    #: ``(parent entity type, foreign-key column)`` for a row whose lifetime comes
    #: from something else, e.g. a message from its conversation.
    parent: tuple[str, str] | None = None
    facets: tuple[FacetField, ...] = ()
    #: A route template filled from the document's ``entity_id``.
    route: str = ""
    #: How the row becomes documents. Declared for every entity, because the granularity
    #: is the decision this module exists to record.
    chunk: Chunk = WHOLE
    #: ``(column, value)`` restricting which rows are indexed at all, ``value`` of
    #: ``None`` meaning "and this column is NULL" rather than an equality. The
    #: equality case is a run's logs, worth indexing when the run failed and noise
    #: when it did not; the ``NULL`` case is every trashable entity, worth indexing
    #: only while it is live. Absent means every row.
    where: tuple[str, str | None] | None = None
    #: Set where the entity has no title of its own. A message has no name; its
    #: conversation does, and that is what labels the result. ``SearchDoc.title`` is
    #: required, so undeclared means the indexer invents one.
    title_from_parent: bool = False

    def key_type(self) -> str:
        """The kind this entity's documents are keyed under.

        Usually the entity's own. Two cases differ. An entity with a vocabulary of its
        own is keyed by that, so a ``notebook_folder`` is keyed as a ``folder``. A child
        takes whatever key its parent has, walked upward, which is what makes dropping a
        notebook drop its cells without anything about a cell saying so.
        """
        if self.resource_type is not None:
            return self.resource_type.value
        if self.parent:
            return next(s for s in SPECS if s.entity_type == self.parent[0]).key_type()
        return self.entity_type

    def declared_columns(self) -> tuple[str, ...]:
        """Every column the extractor is allowed to read for this entity."""
        columns = [self.id_column]
        columns.extend(text.column for text in self.text)
        if self.parent:
            columns.append(self.parent[1])
        columns.extend(facet.column for facet in self.facets)
        if self.where:
            columns.append(self.where[0])
        return tuple(dict.fromkeys(columns))


@dataclass(frozen=True)
class SourceSpec:
    """A document source with no table behind it: the warehouse catalog, the registry.

    Specified like a :class:`SqlSpec`, but its fields name what the producer must
    supply rather than columns on a model. No schema to check them against is exactly
    why the contract has to be written out.
    """

    entity_type: str
    describe: str
    #: Where the documents are read from, so the indexer has one place to look.
    read_from: str
    #: What fills each text field of a :class:`SearchDoc`.
    title: str
    body: str = ""
    columns: str = ""
    chunk: Chunk = WHOLE
    facets: tuple[str, ...] = ()
    route: str = ""


def _json_strings(value: Any) -> str:
    """The string leaves of a JSON blob, so structure is dropped and prose kept.

    The only sanctioned way to index a ``*_json`` column. Keys are dropped along with
    numbers and booleans: a key is schema, and schema is not what anyone searches for.
    """
    import json

    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return ""
    return " ".join(_leaves(parsed))


def _leaves(parsed: Any) -> list[str]:
    """The string leaves of a parsed blob, in document order.

    A leaf, not a word: whether a leaf is prose or a payload is a property of the whole
    leaf, and joining first would leave the filter below inspecting fragments of one.
    """
    out: list[str] = []
    # Document order: a LIFO walk reverses it, which a dense embedding can see.
    queue = deque([parsed])
    while queue:
        node = queue.popleft()
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            queue.extendleft(reversed(list(node.values())))
        elif isinstance(node, list):
            queue.extendleft(reversed(node))
    return out


#: Short enough that classifying it costs more than indexing it.
_MAX_LEAF_CHARS = 400

#: Prose puts whitespace every few characters; encoded data does not, whether it is one
#: unbroken run or wrapped at 76 columns as base64 in a notebook output usually is. One
#: space per this many characters is far below any real writing and far above any
#: encoding, so the two separate cleanly without either being counted by length.
_PROSE_SPACING = 20


def _is_payload(leaf: str) -> bool:
    """Whether a string leaf is data rather than something a person would type."""
    text = leaf.strip()
    if text.startswith("<"):
        return True  # markup: a rendered view of a result, not a description of one
    if len(text) <= _MAX_LEAF_CHARS:
        return False
    return sum(c.isspace() for c in text) * _PROSE_SPACING < len(text)


def _prose_strings(value: Any) -> str:
    """:func:`_json_strings`, minus the leaves that are payloads rather than text.

    A cell's outputs carry ``image/png`` as base64 and ``text/html`` as markup, both
    string leaves -- megabytes of tokens nobody will type. Judged by what a leaf is
    rather than how long it is: long prose is what the window exists for, and cutting it
    to a length here would be the silent truncation the specs forbid.
    """
    import json

    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return ""
    return " ".join(leaf for leaf in _leaves(parsed) if not _is_payload(leaf))


def _split(reason: str, *, late: bool = False) -> Chunk:
    return Chunk("split", reason=reason, window=_WINDOW, overlap=_OVERLAP, late=late)


def _per_field(reason: str) -> Chunk:
    return Chunk("field", reason=reason)


def _open_or_closed(closed_at: Any) -> str:
    """An incident's state, from whether it has been closed."""
    return "closed" if closed_at else "open"


SPECS: tuple[SqlSpec, ...] = (
    SqlSpec(
        entity_type="derivation",
        model=db.Derivation,
        id_column="name",
        resource_type=ResourceType.DERIVATION,
        # One document per field, not one blob: a term matching the Python keeps its
        # own length normalization instead of being diluted by the narrative beside it.
        chunk=_per_field("prose, code and structured assumptions score separately"),
        text=(
            TextField("name", weight=FIELD_TITLE),
            TextField("question"),
            TextField("source"),
            TextField("narrative"),
            TextField("assumptions_json", transform=_json_strings),
            TextField("attestation_json", transform=_json_strings),
        ),
        facets=(
            FacetField("verdict", "verdict"),
            FacetField("created_at", "created_at"),
            FacetField("data_hash", "data_hash"),
        ),
        where=("deleted_at", None),
        route="/derivations/{id}",
    ),
    SqlSpec(
        entity_type="conversation",
        model=db.Conversation,
        id_column="id",
        text=(TextField("title", weight=FIELD_TITLE), TextField("summary")),
        facets=(FacetField("created_at", "created_at"),),
        route="/conversations/{id}",
    ),
    SqlSpec(
        entity_type="message",
        model=db.Message,
        id_column="id",
        parent=("conversation", "conversation_id"),
        # Per message, not per conversation: a window would make every long
        # conversation match everything it ever discussed.
        chunk=_split("a message can be arbitrarily long prose", late=True),
        title_from_parent=True,
        text=(TextField("content"),),
        facets=(FacetField("created_at", "created_at"),),
        route="/conversations/{parent}?message={id}",
    ),
    SqlSpec(
        entity_type="notebook",
        model=db.Notebook,
        id_column="id",
        resource_type=ResourceType.NOTEBOOK,
        text=(TextField("name", weight=FIELD_TITLE),),
        facets=(FacetField("created_at", "created_at"),),
        where=("deleted_at", None),
        route="/notebooks/{id}",
    ),
    SqlSpec(
        entity_type="notebook_cell",
        model=db.NotebookCell,
        id_column="id",
        parent=("notebook", "notebook_id"),
        # ``cell_type`` is a facet, not text: nobody types "markdown", but narrowing
        # to prose cells is how you find the write-up rather than the code.
        chunk=_split(
            "cell source and its printed output are both unbounded", late=True
        ),
        title_from_parent=True,
        text=(
            TextField("source"),
            TextField("outputs_json", transform=_prose_strings),
        ),
        # ``cell_type`` wants a SearchDoc field it does not have; a schema change.
        facets=(),
        route="/notebooks/{parent}",
    ),
    SqlSpec(
        entity_type="notebook_folder",
        model=db.NotebookFolder,
        id_column="id",
        resource_type=ResourceType.FOLDER,
        text=(TextField("name", weight=FIELD_TITLE),),
        where=("deleted_at", None),
        route="/notebooks?folder={id}",
    ),
    SqlSpec(
        entity_type="dashboard",
        model=db.Dashboard,
        id_column="id",
        resource_type=ResourceType.DASHBOARD,
        text=(
            TextField("title", weight=FIELD_TITLE),
            TextField("name", weight=FIELD_TITLE),
        ),
        facets=(FacetField("status", "status"),),
        where=("deleted_at", None),
        route="/dashboards/{id}",
    ),
    SqlSpec(
        entity_type="saved_query",
        model=db.SavedQuery,
        id_column="id",
        resource_type=ResourceType.SAVED_QUERY,
        text=(TextField("name", weight=FIELD_TITLE), TextField("sql")),
        where=("deleted_at", None),
        route="/explore?query={id}",
    ),
    SqlSpec(
        entity_type="metric",
        model=db.MetricRow,
        id_column="name",
        resource_type=ResourceType.METRIC,
        # The manifest's string leaves are what a person wrote; keys and numbers go.
        text=(
            TextField("name", weight=FIELD_TITLE),
            TextField("manifest_json", transform=_json_strings),
        ),
        facets=(FacetField("created_at", "created_at"),),
        where=("deleted_at", None),
        route="/metrics?metric={id}",
    ),
    SqlSpec(
        entity_type="metric_monitor",
        model=db.MetricMonitor,
        id_column="id",
        text=(
            TextField("name", weight=FIELD_TITLE),
            TextField("target"),
            TextField("method"),
        ),
        facets=(FacetField("created_at", "created_at"),),
        route="/monitors?monitor={id}",
    ),
    SqlSpec(
        entity_type="feature_view",
        model=db.FeatureViewRow,
        id_column="name",
        resource_type=ResourceType.FEATURE_VIEW,
        text=(
            TextField("name", weight=FIELD_TITLE),
            TextField("description"),
            TextField("source"),
            TextField("contract_json", transform=_json_strings),
        ),
        where=("deleted_at", None),
        route="/features/{id}",
    ),
    SqlSpec(
        entity_type="feature_entity",
        model=db.FeatureEntity,
        id_column="name",
        text=(TextField("name", weight=FIELD_TITLE), TextField("description")),
        route="/features",
    ),
    SqlSpec(
        entity_type="training_set",
        model=db.FeatureTrainingSet,
        id_column="id",
        text=(TextField("name", weight=FIELD_TITLE), TextField("label")),
        route="/features",
    ),
    SqlSpec(
        entity_type="workflow",
        model=db.Workflow,
        id_column="id",
        text=(TextField("name", weight=FIELD_TITLE),),
        route="/orchestration?workflow={id}",
    ),
    SqlSpec(
        entity_type="asset_check",
        model=db.AssetCheck,
        id_column="id",
        text=(
            TextField("name", weight=FIELD_TITLE),
            TextField("expr"),
            TextField("asset"),
        ),
        # ``{asset}``, not ``{id}``: the page resolves a name, and ``id`` is a uuid.
        route="/orchestration?asset={asset}",
    ),
    SqlSpec(
        entity_type="materialization_schedule",
        model=db.MaterializationSchedule,
        id_column="id",
        text=(
            TextField("name", weight=FIELD_TITLE),
            TextField("selection"),
            TextField("dataset"),
        ),
        route="/orchestration",
    ),
    SqlSpec(
        entity_type="data_source",
        model=db.ExternalDataSource,
        id_column="id",
        resource_type=ResourceType.WAREHOUSE_SOURCE,
        text=(
            TextField("name", weight=FIELD_TITLE),
            TextField("description"),
            TextField("source_type"),
        ),
        facets=(FacetField("status", "status"),),
        route="/warehouse/sources/{id}",
    ),
    SqlSpec(
        entity_type="audit_event",
        model=db.AuditEvent,
        id_column="id",
        text=(
            TextField("action", weight=FIELD_TITLE),
            TextField("target_id"),
            TextField("target_type"),
        ),
        facets=(FacetField("verdict", "verdict"),),
        route="/settings/audit",
    ),
    SqlSpec(
        entity_type="llm_profile",
        model=db.LlmProfile,
        id_column="name",
        # Name only: the row also holds an API key, which the guard test forbids.
        text=(TextField("name", weight=FIELD_TITLE),),
        route="/settings/models",
    ),
    SqlSpec(
        entity_type="setting",
        model=db.Setting,
        id_column="key",
        # Key only, for the same reason: a setting's value may be a credential.
        text=(TextField("key", weight=FIELD_TITLE),),
        route="/settings",
    ),
    SqlSpec(
        # ``reason`` is what somebody types remembering an alert rather than its
        # monitor, and this is the only entity that can fill ``incident_state``.
        entity_type="monitor_incident",
        model=db.MonitorIncident,
        id_column="id",
        parent=("metric_monitor", "monitor_id"),
        title_from_parent=True,
        text=(TextField("reason"),),
        facets=(
            FacetField("incident_state", "closed_at", derive=_open_or_closed),
            FacetField("created_at", "opened_at"),
        ),
        route="/monitors?incident={id}",
    ),
    SqlSpec(
        # Thin text of its own, but ``asset_run`` scopes to it.
        entity_type="orchestration_run",
        model=db.OrchestrationRun,
        id_column="id",
        text=(TextField("cause", weight=FIELD_TITLE),),
        facets=(
            FacetField("status", "status"),
            FacetField("created_at", "started_at"),
        ),
        route="/orchestration?run={id}",
    ),
    SqlSpec(
        # Only on failure: a passing run's logs are the highest-volume, lowest-signal
        # text in the corpus.
        entity_type="asset_run",
        model=db.AssetRun,
        id_column="id",
        # From the run, not ``asset``: ``backfill`` writes that as
        # ``"{asset} [{param}={value}]"``, which is exactly what ``where`` selects.
        parent=("orchestration_run", "run_id"),
        where=("state", "failed"),
        chunk=_split("a failed run's log is long, and the error is anywhere in it"),
        text=(
            TextField("asset", weight=FIELD_TITLE),
            TextField("error"),
            TextField("logs"),
        ),
        facets=(
            FacetField("status", "state"),
            FacetField("verdict", "verdict"),
            FacetField("created_at", "created_at"),
        ),
        route="/orchestration?run={parent}",
    ),
)


#: Sources that produce documents without a table behind them.
#:
#: Two, not the four an earlier draft named. A repo-authored derivation is mirrored into
#: ``db.Derivation`` with ``origin="repo"`` by ``serve._sync_repo_derivations``, and a
#: dataset declared in the project config becomes a warehouse table through
#: ``serve._ensure_declared_sources`` -- so both are already covered above, and indexing
#: them again here would return the same derivation twice under two entity types.
SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        entity_type="warehouse_table",
        describe="a synced table, with its column names as a field",
        read_from="WarehouseService.tables() joined to WarehouseService.columns()",
        # One document per table, not per column: "datasets with a customer_id
        # column" wants *the table*, not forty rows of one.
        title="the warehouse table name",
        columns="its column names, from the persisted catalog (IP-17)",
        facets=("created_at",),
        route="/warehouse?table={id}",
    ),
    SourceSpec(
        entity_type="model",
        describe="an entry in the model registry",
        read_from="the MLflow model registry, via ModelService.registry()",
        title="the registered model name",
        body="its description and the tags a person set",
        facets=("verdict", "created_at"),
        route="/models?model={id}",
    ),
)


#: Every table deliberately left out, with the reason. The coverage guard requires each
#: table to be here or in :data:`SPECS`, so a new one cannot be missed by accident and
#: cannot be included by accident either.
NOT_INDEXED: dict[str, str] = {
    # Credentials and their hashes. Never indexed, at any level, for any caller.
    "Secret": "holds an encrypted credential",
    # Raw governed data. The values themselves, which search must never surface.
    "OnlineFeatureRow": "the feature values themselves",
    "FeatureStatisticsRow": "profiles and samples of the data",
    "FeatureDriftRow": "computed over the data",
    "FeatureExpectationRow": "computed over the data",
    "InferenceEvent": "production request and prediction payloads",
    "MetricSnapshot": "a numeric observation, filtered on rather than searched",
    "Budget": "a numeric limit, filtered on rather than searched",
    "ComputeUsage": "usage accounting, no searchable text",
    # Runs and telemetry: high volume, keyed by identifiers nobody types. Facets over
    # them are how a person reaches these, not free-text search.
    "DerivationRun": "a run of an indexed derivation",
    "JobRow": "its label is an indexed derivation's name, and nothing more",
    "MetricVersion": "a version of an indexed metric",
    "DashboardVersion": "its label is only ever 'saved' or 'published', never written",
    "DashboardSubscription": "a delivery rule with no searchable text",
    "RetrainPolicy": "settings on an indexed model, with no prose of its own",
    "PromotedQuery": "promoting authors a derivation, and that gets indexed",
    # A notification announces something already indexed under its own type -- the
    # incident, the run, the model version -- so indexing it too returns one event
    # twice, once as the artifact and once as the note about it. The inbox is the
    # surface for these, and they are pruned on a retention window besides.
    "Notification": "a delivery of an event whose artifact is indexed",
    "NotificationPreference": "per-event-type switches, with no prose of their own",
    # Both are parts of the warehouse_table document rather than documents of their own,
    # assembled in `search.sources` because it takes a join these specs cannot express.
    "ExternalDataSchema": "the warehouse_table document is assembled from it",
    "WarehouseColumn": "the warehouse_table document is assembled from it",
    "DataSource": "superseded by ExternalDataSource for connections",
}
