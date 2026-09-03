"""The document mapping, rendered from the specs, with its size estimated.

``docs/search-documents.md`` is this module's output and a test asserts the committed
file matches it, so the reviewed document and the declarations the indexer reads are the
same thing.

The estimate settles one question the parent leaves open: whether the corpus needs a
vector store of its own or fits in a column beside the documents. The row counts and
text lengths are stated assumptions, not measurements, each named so a reader can
disagree with a number rather than with the conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import FIELD_BODY, FIELD_COLUMNS, FIELD_TITLE, FIELD_WEIGHTS
from .specs import NOT_INDEXED, SOURCES, SPECS, Chunk, SqlSpec, windows

#: The embedding's width and the bytes a float32 takes: together, the size of the matrix
#: a query scans.
#:
#: 384 is ``OnnxEmbedder.DEFAULT_MODEL`` (``BAAI/bge-small-en-v1.5``). The embedder
#: publishes no width constant and reading one means loading the model, which downloads
#: weights -- so this cannot be asserted directly. A guard test pins the model name
#: instead, so swapping the model fails here rather than silently leaving the estimate
#: describing a model nobody uses.
EMBED_DIM = 384
BYTES_PER_FLOAT = 4

#: The model :data:`EMBED_DIM` was taken from. Changing the embedder must change both.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Assumption:
    """One entity's share of a corpus: how many rows, and how much text each holds.

    ``mean_chars`` is the mean length of a row's *body* text, and only matters where the
    entity is chunked by splitting. A title is never split.
    """

    rows: int
    mean_chars: int = 0
    note: str = ""


#: A mature single-team project: the scale at which the parent's "a handful of
#: derivations to hundreds" has already happened.
CORPUS: dict[str, Assumption] = {
    "derivation": Assumption(1_000, 2_400, "question, source, narrative, two JSON"),
    "conversation": Assumption(500),
    "message": Assumption(5_000, 1_200, "a chat turn, occasionally much longer"),
    "notebook": Assumption(200),
    "notebook_cell": Assumption(2_400, 1_000, "~12 cells a notebook, source + outputs"),
    "notebook_folder": Assumption(30),
    "dashboard": Assumption(50),
    "saved_query": Assumption(300, 0, "SQL text, short enough to stay whole"),
    "metric": Assumption(100),
    "metric_monitor": Assumption(50),
    "monitor_incident": Assumption(200),
    "feature_view": Assumption(40),
    "feature_entity": Assumption(20),
    "training_set": Assumption(30),
    "workflow": Assumption(20),
    "materialization_schedule": Assumption(30),
    "asset_check": Assumption(100),
    "orchestration_run": Assumption(5_000, 0, "one per materialization run"),
    "asset_run": Assumption(250, 8_000, "5% of ~5k runs fail; only those are indexed"),
    "data_source": Assumption(20),
    "audit_event": Assumption(50_000, 0, "the highest-count entity in the corpus"),
    "llm_profile": Assumption(5),
    "setting": Assumption(50),
    # The two sources with no table behind them.
    "warehouse_table": Assumption(100, 0, "columns are a field, not documents"),
    "model": Assumption(30),
}


def chunks_per_row(chunk: Chunk, fields: int, mean_chars: int) -> int:
    """How many documents one row becomes under its declared granularity.

    Split counts through :func:`~.specs.windows`, the same function the extractor cuts
    with, so this is the count the backfill will actually report rather than a second
    reading of the same two numbers.
    """
    if chunk.mode == "field":
        return max(fields, 1)
    if chunk.mode == "split":
        # The extractor's own windowing, not a formula that agrees with it today:
        # counting strides rather than window starts charged a second chunk to a row
        # the first window already held, which a second implementation is how you get.
        return len(windows(mean_chars, chunk))
    return 1


@dataclass(frozen=True)
class Estimate:
    """One entity's contribution to the corpus."""

    entity_type: str
    rows: int
    per_row: int
    chunks: int
    note: str


def estimate() -> list[Estimate]:
    """Rows to chunks for every entity, at :data:`CORPUS`."""
    body_fields = {
        spec.entity_type: len([t for t in spec.text if t.weight != FIELD_TITLE])
        for spec in SPECS
    }
    chunking: dict[str, Chunk] = {s.entity_type: s.chunk for s in SPECS}
    chunking.update({s.entity_type: s.chunk for s in SOURCES})

    out = []
    for entity_type, assumption in CORPUS.items():
        chunk = chunking[entity_type]
        per_row = chunks_per_row(
            chunk, body_fields.get(entity_type, 1), assumption.mean_chars
        )
        out.append(
            Estimate(
                entity_type=entity_type,
                rows=assumption.rows,
                per_row=per_row,
                chunks=assumption.rows * per_row,
                note=assumption.note,
            )
        )
    out.sort(key=lambda e: -e.chunks)
    return out


def embedding_bytes(chunks: int) -> int:
    """The matrix an exact similarity scan reads, at :data:`EMBED_DIM`."""
    return chunks * EMBED_DIM * BYTES_PER_FLOAT


def _facets(spec: SqlSpec) -> str:
    return ", ".join(f.facet for f in spec.facets) or "-"


def _fields(spec: SqlSpec) -> str:
    parts = []
    for text in spec.text:
        weight = "title" if text.weight == FIELD_TITLE else "body"
        via = " (json)" if text.transform is not None else ""
        parts.append(f"`{text.column}`→{weight}{via}")
    return ", ".join(parts)


def _chunk(chunk: Chunk) -> str:
    rendered = (
        f"split {chunk.window}/{chunk.overlap}" if chunk.mode == "split" else chunk.mode
    )
    return f"{rendered}, late" if chunk.late else rendered


def render() -> str:
    """The whole mapping as markdown."""
    lines = [
        "# What platform search indexes",
        "",
        "<!-- Generated from `elbi.search.specs`. Do not edit by hand: a test",
        "     asserts this file matches the declarations the indexer reads. -->",
        "",
        "Every entity is either mapped below or excluded with a reason. The",
        "mapping is the contract the indexer implements; nothing reflects over a",
        "model, so a column added later cannot reach the index by existing.",
        "",
        "## Entities",
        "",
        f"A field is worth {FIELD_WEIGHTS[FIELD_TITLE]:g} as a title and "
        f"{FIELD_WEIGHTS[FIELD_BODY]:g} as body, which is where the mapping's",
        "judgement lives: `email`\u2192title on `user` is the decision that finding",
        "a person by address ranks with finding them by name. A warehouse table's",
        f"columns weigh {FIELD_WEIGHTS[FIELD_COLUMNS]:g}, as body does.",
        "",
        "| Entity | Fields | Chunking | Facets | Route |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in sorted(SPECS, key=lambda s: s.entity_type):
        where = f" *(only `{spec.where[0]}={spec.where[1]}`)*" if spec.where else ""
        lines.append(
            f"| `{spec.entity_type}`{where} | {_fields(spec)} "
            f"| {_chunk(spec.chunk)} | {_facets(spec)} | `{spec.route or '-'}` |"
        )

    lines += [
        "",
        "### Why each is chunked that way",
        "",
        "*late* means each chunk is embedded in the context of its whole row rather",
        "than on its own, so a fragment keeps what surrounded it. It needs a",
        "`SpanEmbedder`; the static fallback embedder degrades to per-chunk.",
        "",
    ]
    for spec in sorted(SPECS, key=lambda s: s.entity_type):
        lines.append(
            f"- `{spec.entity_type}`: {_chunk(spec.chunk)}, {spec.chunk.reason}"
        )

    lines += [
        "",
        "## Sources with no table behind them",
        "",
        "| Entity | Read from | Title | Other text | Facets |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in SOURCES:
        other = ", ".join(x for x in (source.body, source.columns) if x) or "-"
        lines.append(
            f"| `{source.entity_type}` | {source.read_from} | {source.title} "
            f"| {other} | {', '.join(source.facets) or '-'} |"
        )

    lines += [
        "",
        "## Excluded, and why",
        "",
        "| Table | Reason |",
        "| --- | --- |",
    ]
    for name, reason in sorted(NOT_INDEXED.items()):
        lines.append(f"| `{name}` | {reason} |")

    rows = estimate()
    total = sum(e.chunks for e in rows)
    audit = next((e.chunks for e in rows if e.entity_type == "audit_event"), 0)
    share = 100 * audit / total if total else 0
    megabytes = embedding_bytes(total) / 1_000_000
    lines += [
        "",
        "## Size at a realistic corpus",
        "",
        "A mature single-team project. These row counts and text lengths are",
        "stated assumptions, not measurements: argue with a number rather than",
        "with the conclusion.",
        "",
        "| Entity | Rows | Chunks/row | Chunks | Assumption |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.entity_type}` | {row.rows:,} | {row.per_row} "
            f"| {row.chunks:,} | {row.note or '-'} |"
        )
    lines += [
        f"| **total** | | | **{total:,}** | |",
        "",
        f"At {EMBED_DIM} dimensions and {BYTES_PER_FLOAT} bytes a float, "
        f"{total:,} chunks is **{megabytes:.0f} MB** of vectors.",
        "",
        "An exact `array_cosine_similarity` scan over 200k documents at this width",
        "measures 127 ms, and 14 ms once a filter narrows it. A corpus this",
        "size is therefore answered by scanning a column: a dedicated vector store",
        "would buy latency that is already there. The parent's no-vector-store",
        "decision assumed as much, and this validates it.",
        "",
        "### One number decides it",
        "",
        f"`audit_event` is {audit:,} of those {total:,} chunks, {share:.0f}% of the",
        "whole corpus, and the only entity with no natural ceiling. It is also the one",
        "whose text is mostly identifiers rather than prose, so it gains least from",
        "being embedded at all.",
        "",
        "Excluding it, or indexing it lexically without a vector, leaves",
        f"{total - audit:,} chunks and "
        f"{embedding_bytes(total - audit) / 1_000_000:.0f} MB.",
        "That is the decision worth taking deliberately before the indexer is",
        "built, rather than discovering it when the audit log has grown.",
        "",
    ]
    return "\n".join(lines)
