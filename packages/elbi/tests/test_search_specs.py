"""What search is allowed to read, enforced rather than documented.

These are the tests the security review should concentrate on. They do not check that
the current specs happen to be right; they check that the *shape* of the declaration
makes a leak require a deliberate, reviewable act. A spec names its columns and the
extractor selects exactly those, so the guards here are about what a future change can
do by accident.
"""

from __future__ import annotations

import re

import pytest
from sqlmodel import SQLModel

from elbi import db
from elbi.search.schema import (
    FIELD_BODY,
    FIELD_TITLE,
    FIELD_WEIGHTS,
)
from elbi.search.specs import (
    NOT_INDEXED,
    SOURCES,
    SPECS,
    SourceSpec,
    SqlSpec,
    _json_strings,
)


def _tables() -> dict[str, type[SQLModel]]:
    """Every persisted model in the store, by class name.

    Walks the subclass tree rather than calling ``__subclasses__()`` once, which returns
    only *direct* subclasses. A table declared through an intermediate base -- the
    ordinary ``class Base(SQLModel)`` then ``class Row(Base, table=True)`` pattern --
    would otherwise be invisible here, and the coverage guard would pass on exactly the
    "add it and say nothing" case it exists to prevent.
    """
    found: dict[str, type[SQLModel]] = {}
    seen: set[type] = set()
    queue = [SQLModel]
    while queue:
        cls = queue.pop()
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            queue.append(sub)
            if getattr(sub, "__table__", None) is not None:
                found[sub.__name__] = sub
    return found


def test_every_table_is_either_indexed_or_excluded_with_a_reason() -> None:
    """A table added later must not slip into search, or silently out of it.

    This is the guard that makes the others meaningful: without it, the way to leak a
    new table is to add it and say nothing, and the way to lose one is the same.
    """
    specified = {spec.model.__name__ for spec in SPECS}
    known = set(_tables())
    undecided = known - specified - set(NOT_INDEXED)
    assert not undecided, (
        f"these tables are neither indexed nor excluded: {sorted(undecided)}. "
        "Add a SqlSpec, or an entry in NOT_INDEXED saying why not."
    )
    stale = (specified | set(NOT_INDEXED)) - known
    assert not stale, f"these no longer exist: {sorted(stale)}"


def test_no_spec_indexes_a_forbidden_table() -> None:
    assert not {spec.model.__name__ for spec in SPECS} & FORBIDDEN_MODELS


def test_forbidden_tables_are_excluded_explicitly() -> None:
    """Absence is not enough: they have to be named, so removing them is deliberate."""
    assert set(NOT_INDEXED) >= FORBIDDEN_MODELS


def test_the_table_holding_credentials_has_no_spec_and_never_will() -> None:
    """Named, so the reason survives even if the guards are ever loosened."""
    assert db.Secret.__name__ in NOT_INDEXED
    assert "credential" in NOT_INDEXED["Secret"]


def test_nothing_is_both_indexed_and_excluded() -> None:
    assert not {spec.model.__name__ for spec in SPECS} & set(NOT_INDEXED)


def test_every_exclusion_gives_a_reason() -> None:
    """A bare name would be a decision nobody can review later."""
    for name, reason in NOT_INDEXED.items():
        assert len(reason.split()) >= 3, f"{name}: {reason!r} is not a reason"


#: Tables whose every column is off limits, whatever a spec might later ask for.
FORBIDDEN_MODELS = {"Secret"}

#: Column names that must never be indexed. Matched as a pattern rather than a list, so
#: the next column called ``*_token`` or ``*_secret`` is excluded before it exists.
FORBIDDEN_PATTERN = re.compile(
    r"(^|_)(secret|token|password|credential|api_key|private_key|key_hash)"
    r"|encrypted"
)

#: Columns that hold a credential but are named as though they do not. The pattern above
#: catches the next column called ``*_token``; these two it cannot, and both sit on
#: models that are already indexed: ``Setting.value`` is the plaintext of every stored
#: setting including credential-bearing ones, and ``Secret.value`` is a stored secret.
#: ``specs.py`` says the ``setting`` spec takes the key only because the guard would
#: reject the value. This is what makes that true.
FORBIDDEN_COLUMNS = {("Setting", "value"), ("Secret", "value")}

#: Structured columns. Excluded by default because their contents are data rather than
#: prose, and because that is where raw values and payloads live.
JSON_PATTERN = re.compile(r"_json$")

#: The only structured columns any spec may read, each also needing a transform.
#: Adding to it is the reviewable act the guard exists to force. Every one carries prose
#: a person wrote; a column not named here cannot be indexed, however transformed.
ACKNOWLEDGED_JSON: frozenset[tuple[str, str]] = frozenset(
    {
        # The author's stated assumptions and the oracle's attestation: sentences.
        ("Derivation", "assumptions_json"),
        ("Derivation", "attestation_json"),
        # A cell's printed output, across the four nbformat shapes.
        ("NotebookCell", "outputs_json"),
        # A metric definition's labels and descriptions.
        ("MetricRow", "manifest_json"),
        # A feature view's contract: the feature names somebody types.
        ("FeatureViewRow", "contract_json"),
    }
)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_no_spec_reads_a_credential_column(spec: SqlSpec) -> None:
    """The pattern, not a list: this holds for columns that do not exist yet."""
    for column in spec.declared_columns():
        assert not FORBIDDEN_PATTERN.search(column), (
            f"{spec.entity_type} declares {column!r}, which looks like a credential"
        )
        assert (spec.model.__name__, column) not in FORBIDDEN_COLUMNS, (
            f"{spec.entity_type} declares {column!r}, which holds a credential"
        )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_a_json_column_needs_acknowledging_and_a_transform(spec: SqlSpec) -> None:
    """Structured columns are where raw values and payloads live.

    Reading one has to be deliberate and has to say how the text is extracted, so a
    ``*_json`` column added to an indexed table does not quietly start being searched.
    """
    transforms = {t.column: t.transform for t in spec.text}
    transforms.update({f.column: f.derive for f in spec.facets})
    # Every declared column, not just the text ones: a facet's column is selected by the
    # extractor too, and FacetField takes a callable of its own -- so a *_json column
    # declared as a facet would otherwise reach the index raw and unacknowledged.
    for column in spec.declared_columns():
        if not JSON_PATTERN.search(column):
            continue
        pair = (spec.model.__name__, column)
        assert pair in ACKNOWLEDGED_JSON, f"{pair} reads JSON without acknowledgement"
        assert transforms.get(column) is not None, f"{pair} must declare a transform"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_every_declared_column_exists_on_its_model(spec: SqlSpec) -> None:
    """A rename in the store fails here rather than silently dropping a field."""
    columns = set(spec.model.model_fields)
    missing = set(spec.declared_columns()) - columns
    assert not missing, f"{spec.entity_type} declares missing columns: {missing}"


def test_json_extraction_keeps_prose_and_drops_structure() -> None:
    """Keys are schema; nobody searches for schema."""
    blob = '{"question": "what drives churn", "n": 42, "ok": true, "tags": ["cohort"]}'
    extracted = _json_strings(blob).split()
    assert set(extracted) == {"what", "drives", "churn", "cohort"}
    assert "question" not in extracted, "a key is not text"
    assert "42" not in extracted


def test_json_extraction_survives_a_malformed_blob() -> None:
    """A column that is not valid JSON must not break a reindex."""
    assert _json_strings("not json at all") == ""
    assert _json_strings(None) == ""
    assert _json_strings("") == ""


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_every_entity_declares_a_chunk_granularity_and_a_reason(
    spec: SqlSpec,
) -> None:
    """A default with no reason is a gap, not a decision."""
    assert spec.chunk.mode in ("whole", "field", "split")
    assert spec.chunk.reason, f"{spec.entity_type} chunks without saying why"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_a_split_declares_a_window_and_a_whole_does_not(spec: SqlSpec) -> None:
    """A window of zero would make every chunk empty; one on a whole row is a lie."""
    if spec.chunk.mode == "split":
        assert spec.chunk.window > 0
        assert 0 <= spec.chunk.overlap < spec.chunk.window
    else:
        assert spec.chunk.window == 0 and spec.chunk.overlap == 0


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_per_field_chunking_needs_more_than_one_field(spec: SqlSpec) -> None:
    """Splitting one field per document is only meaningful with fields to separate."""
    if spec.chunk.mode == "field":
        body = [t for t in spec.text if t.weight != FIELD_TITLE]
        assert len(body) > 1, f"{spec.entity_type} chunks per field but has one"


def test_no_text_is_silently_truncated() -> None:
    """Long text is windowed, never cut: a transform returns a leaf whole or not at all.

    Truncation drops everything past its limit with nothing saying so, and the window is
    the sanctioned way to handle length. Asserted by running a long value through every
    declared transform, since a transform that cuts need not say it does: a docstring is
    a claim about the code, not the code.
    """
    import json

    prose = " ".join(f"word{n}" for n in range(500))
    for spec in SPECS:
        for text in spec.text:
            if text.transform is None:
                continue
            out = text.transform(json.dumps({"note": prose}))
            assert prose in out, (
                f"{spec.entity_type}.{text.column} returned {len(out)} chars of a "
                f"{len(prose)}-char value: it cuts instead of chunking"
            )


def test_a_payload_leaf_goes_whole_or_not_at_all() -> None:
    """The outputs filter judges a leaf, not the words a leaf happens to contain.

    It used to split the joined text on spaces and filter each word, so a payload with
    whitespace in it survived in pieces: `<div>hello world</div>` left `world</div>`
    behind, and base64 wrapped at 76 columns passed line by line.
    """
    import json

    from elbi.search.specs import _prose_strings

    assert _prose_strings(json.dumps({"o": "<div>hello world</div>"})) == ""
    wrapped = "\n".join("QUJDRA" * 12 for _ in range(20))  # base64, 72 columns wide
    assert _prose_strings(json.dumps({"o": wrapped})) == ""
    assert _prose_strings(json.dumps({"o": "A" * 1000})) == ""

    # A long leaf of real prose is not a payload: length is the window's problem.
    prose = " ".join(f"word{n}" for n in range(300))
    assert _prose_strings(json.dumps({"o": prose})) == prose


def test_the_field_weights_match_the_in_memory_ranker() -> None:
    """Declared rather than imported, so they are two constants that could drift.

    This is what stops them.
    """
    from elbi_core.retrieval import Bm25Retriever

    ranker = Bm25Retriever()
    assert FIELD_WEIGHTS[FIELD_TITLE] == ranker.name_boost
    assert FIELD_WEIGHTS[FIELD_BODY] == ranker.description_boost


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.entity_type)
def test_every_source_says_where_it_reads_from(source: SourceSpec) -> None:
    """A source is a contract for code that does not exist yet.

    Nothing can check it against a model, so the one enforceable thing is that it was
    written at all: four were once declared as a name and a sentence, and none built.
    """
    assert source.read_from, f"{source.entity_type} does not say where it reads from"
    assert source.title, f"{source.entity_type} declares no title"
    assert source.describe


def test_a_source_does_not_duplicate_an_indexed_table() -> None:
    """Two producers for one entity type would index it twice.

    A repo derivation is mirrored into ``Derivation`` and a declared dataset becomes a
    warehouse table, so naming either as a source returns the same artifact twice.
    """
    overlap = {s.entity_type for s in SOURCES} & {s.entity_type for s in SPECS}
    assert not overlap, f"indexed from both a table and a source: {sorted(overlap)}"


#: Every entity IP-15 names as a document, and what became of it here. This is the
#: reconciliation the ticket asks for, kept as a test so it cannot quietly stop being
#: true: the last attempt excluded nine of these without the divergence being visible.
#:
#: A name maps to the entity type that covers it, or to ``None`` when it is deliberately
#: not a document of its own -- in which case ``NOT_INDEXED`` must say why.
PARENT_DOCUMENTS: dict[str, str | None] = {
    "Derivation": "derivation",
    "Conversation": "conversation",
    "Message": "message",
    "Notebook": "notebook",
    "NotebookCell": "notebook_cell",
    "NotebookFolder": "notebook_folder",
    "Dashboard": "dashboard",
    "SavedQuery": "saved_query",
    "MetricRow": "metric",
    "MetricMonitor": "metric_monitor",
    "FeatureEntity": "feature_entity",
    "FeatureViewRow": "feature_view",
    "FeatureTrainingSet": "training_set",
    "Workflow": "workflow",
    "MaterializationSchedule": "materialization_schedule",
    "AssetCheck": "asset_check",
    "AssetRun": "asset_run",
    "ExternalDataSource": "data_source",
    "AuditEvent": "audit_event",
    "LlmProfile": "llm_profile",
    "Setting": "setting",
    # Named as documents by the parent, and not documents of their own here. Each is
    # either a version or a run of something already indexed, or a row keyed by an
    # identifier nobody types. NOT_INDEXED carries the reason for each.
    "DerivationRun": None,
    "DashboardVersion": None,
    "DashboardSubscription": None,
    "PromotedQuery": None,
    "MetricVersion": None,
    "MonitorIncident": "monitor_incident",
    "RetrainPolicy": None,
    "OrchestrationRun": "orchestration_run",
    "ExternalDataSchema": None,
}


def test_every_parent_document_is_specified_or_excluded_with_a_reason() -> None:
    """The parent's Documents list, reconciled entity by entity.

    The coverage guard points at ``db.py``; this one points at the ticket, which is the
    other way a decision goes missing.
    """
    by_entity = {spec.entity_type: spec for spec in SPECS}
    for model_name, entity_type in PARENT_DOCUMENTS.items():
        if entity_type is None:
            assert model_name in NOT_INDEXED, (
                f"{model_name} is in the parent's Documents list, is not indexed, "
                "and NOT_INDEXED gives no reason"
            )
            continue
        assert entity_type in by_entity, (
            f"the parent lists {model_name} as a document and nothing indexes it"
        )
        assert by_entity[entity_type].model.__name__ == model_name


def test_the_reconciliation_covers_the_whole_parent_list() -> None:
    """``PARENT_DOCUMENTS`` would otherwise drift from ``SPECS``, and the
    reconciliation above would pass while covering less and less of what ships.
    """
    reconciled = {e for e in PARENT_DOCUMENTS.values() if e}
    # The two entity types with no table behind them are reconciled through SOURCES.
    from_sources = {source.entity_type for source in SOURCES}
    unreconciled = {spec.entity_type for spec in SPECS} - reconciled - from_sources
    assert not unreconciled, (
        f"indexed but not reconciled against the parent's list: {sorted(unreconciled)}"
    )


def test_the_committed_mapping_matches_the_specs() -> None:
    """``docs/search-documents.md`` is generated, and drift is a failure.

    A hand-maintained doc beside the declarations is two things that must agree with
    nothing making them agree. This is the mechanism instead.
    """
    from pathlib import Path

    from elbi.search.document_map import render

    doc = Path(__file__).resolve().parents[3] / "docs" / "search-documents.md"
    assert doc.exists(), f"{doc} is missing; regenerate it"
    assert doc.read_text(encoding="utf-8") == render(), (
        "docs/search-documents.md is stale. Regenerate it:\n"
        '  uv run python -c "from elbi.search.document_map import render; '
        "open('docs/search-documents.md','w').write(render())\""
    )


def test_a_row_that_fits_one_window_is_one_chunk() -> None:
    """The estimate counts window starts, not strides.

    Dividing the length by the stride charges a second window to a row the first one
    already held: at 2000/200, a 1900-character row came out as two. It changes no
    published number today, since nothing in the corpus falls between the window and
    one stride past it, but the estimate is what gets re-run when an assumption moves.
    """
    from elbi.search.document_map import chunks_per_row
    from elbi.search.specs import Chunk

    window = Chunk("split", reason="", window=2000, overlap=200)
    assert [chunks_per_row(window, 1, n) for n in (0, 1900, 2000)] == [1, 1, 1]
    # One character past the window opens the second; one past its stride, the third.
    assert [chunks_per_row(window, 1, n) for n in (2001, 3800, 3801)] == [2, 2, 3]


def test_every_entity_in_the_corpus_estimate_is_a_real_entity() -> None:
    """An estimate for something nothing indexes would inflate the total silently."""
    from elbi.search.document_map import CORPUS

    known = {s.entity_type for s in SPECS} | {s.entity_type for s in SOURCES}
    assert set(CORPUS) == known, (
        f"estimated but not indexed: {sorted(set(CORPUS) - known)}; "
        f"indexed but not estimated: {sorted(known - set(CORPUS))}"
    )


def test_a_chunked_entity_declares_how_its_chunks_collapse() -> None:
    """Chunking without a collapse rule turns one artifact into several results.

    A derivation matching three of its five field-documents would be three hits.
    """
    from elbi.search.schema import COLLAPSE_BY, SearchDoc

    assert COLLAPSE_BY == ("entity_type", "entity_id")
    fields = set(SearchDoc.__dataclass_fields__)
    assert set(COLLAPSE_BY) <= fields, "collapsing on a field documents do not carry"
    # The chunk index is on doc_id and deliberately not on the collapse key, which is
    # what makes the grouping possible at all.
    assert "chunk_index" not in COLLAPSE_BY


def test_a_chunk_cannot_be_built_without_the_parent_its_row_has() -> None:
    """The failure this shape exists to make impossible.

    Naming the parent on every chunk separately fails on the chunk somebody writes
    without it, and that chunk is then unreachable from the thing that contains it.
    Here a chunk has no parent field to omit: it defaults to its own row, so a chunk of
    a derivation names that derivation and a chunk of a cell names the notebook.
    """
    from elbi.search.schema import SearchDoc

    whole = SearchDoc(entity_type="derivation", entity_id="revenue", title="revenue")
    chunk = SearchDoc(
        entity_type="derivation", entity_id="revenue", title="revenue", chunk_index=3
    )

    assert (chunk.parent_type, chunk.parent_id) == (whole.parent_type, whole.parent_id)
    assert (whole.parent_type, whole.parent_id) == ("derivation", "revenue")

    # A child names what contains it, and its chunks inherit that in turn.
    cell = SearchDoc(
        entity_type="notebook_cell",
        entity_id="nb1/cell7",
        title="quarterly review",
        parent_type="notebook",
        parent_id="nb1",
        chunk_index=2,
    )
    assert (cell.parent_type, cell.parent_id) == ("notebook", "nb1")


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_an_entity_without_a_title_says_where_its_title_comes_from(
    spec: SqlSpec,
) -> None:
    """``SearchDoc.title`` is required, so a titleless entity needs an answer here.

    A message has no name and a result still has to be labelled. Left undeclared, the
    indexer invents one.
    """
    has_title = any(t.weight == FIELD_TITLE for t in spec.text)
    assert has_title or spec.title_from_parent, (
        f"{spec.entity_type} has no title field and does not say where one comes from"
    )
    if spec.title_from_parent:
        assert spec.parent is not None, (
            f"{spec.entity_type} takes its title from a parent it does not declare"
        )


def test_the_corpus_estimate_matches_the_embedder_actually_shipped() -> None:
    """The size estimate is only meaningful for the model it was computed for.

    ``EMBED_DIM`` cannot be read off the embedder -- it publishes no width, and getting
    one means loading the model, which downloads weights. So this pins the model name
    instead: swapping the embedder fails here, where the estimate is, rather than
    leaving a number that quietly describes a model nobody runs any more.
    """
    from elbi.search.document_map import EMBED_DIM, EMBED_MODEL
    from elbi_core.retrieval import OnnxEmbedder

    assert EMBED_MODEL == OnnxEmbedder.DEFAULT_MODEL, (
        f"the embedder is now {OnnxEmbedder.DEFAULT_MODEL!r}; re-check EMBED_DIM "
        f"(currently {EMBED_DIM}) and regenerate docs/search-documents.md"
    )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.entity_type)
def test_late_chunking_only_where_a_chunk_is_split_from_a_parent(
    spec: SqlSpec,
) -> None:
    """Pooling a span over its row only means something when the row is chunked.

    ``late`` on a whole-row document would ask the embedder to pool a span over itself.
    And an entity chunked out of a parent is exactly the case the flag is for, which is
    why the two that set it are the two that already take their title from one.
    """
    if spec.chunk.late:
        assert spec.chunk.mode == "split", (
            f"{spec.entity_type} is late-chunked but not split"
        )
        assert spec.title_from_parent, (
            f"{spec.entity_type} pools over a parent it does not declare"
        )


def test_the_entities_that_need_context_are_the_ones_that_get_it() -> None:
    """Named, so dropping one is a decision rather than an omission.

    A message and a notebook cell are fragments: "then we filtered to Q3" is meaningless
    without the conversation or notebook around it. A failed run's log is not -- an
    error string says what it says -- so it is split without late pooling.
    """
    late = {s.entity_type for s in SPECS if s.chunk.late}
    assert late == {"message", "notebook_cell"}
    by_entity = {s.entity_type: s for s in SPECS}
    assert by_entity["asset_run"].chunk.mode == "split"
    assert not by_entity["asset_run"].chunk.late


def test_late_chunking_has_an_embedder_that_can_do_it() -> None:
    """The flag is only meaningful if something implements the capability.

    A static embedder gives each token one fixed vector whatever surrounds it, so there
    is no context to pool. Declaring ``late`` while the only available embedder is
    static would be a spec asking for something nothing can produce.
    """
    from elbi_core.retrieval import (
        Model2VecEmbedder,
        OnnxEmbedder,
        SpanEmbedder,
    )

    assert isinstance(OnnxEmbedder(), SpanEmbedder)
    # And the fallback is honest about not being able to: the indexer checks, and
    # degrades to per-chunk rather than producing context-free vectors silently.
    assert not isinstance(Model2VecEmbedder(), SpanEmbedder)
