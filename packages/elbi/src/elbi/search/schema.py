"""The shape of a searchable document, and the fields it is scored over.

The output contract of :mod:`.specs`: a spec says which columns become which field, and
a :class:`SearchDoc` is what that produces. Storing and scoring them is the indexer's,
not this module's.

The facet fields are the parent ticket's facet list in one place, so a facet is
something a document declares rather than something a query invents.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

#: Bumped when the DDL below changes. A mismatch drops the file, never migrates it.
#: 2: ``body_hash`` became ``doc_hash`` and covers every stored field rather than the
#: text alone, so a reconciliation pass notices a facet that moved without its text.
#: ``chunk_index``; ``certified``, ``staleness`` and ``drift_state`` dropped, no column
#: producing any of them.
SCHEMA_VERSION: Final = 5

#: Matching :class:`~elbi_core.retrieval.Bm25Retriever`'s defaults, so a
#: document scores the same through the index as through the in-memory ranker.
#: ``test_search_specs`` asserts the equality rather than trusting this comment.
NAME_BOOST: Final = 2.0
DESCRIPTION_BOOST: Final = 1.0

#: The boosted field: a short deliberate name, where a hit means the document is *about*
#: the term rather than merely mentioning it.
FIELD_TITLE: Final = "title"

#: Free text: a description, a question, message bodies, cell source, SQL.
FIELD_BODY: Final = "body"

#: A warehouse table's column names. Its own field so its weight can be tuned apart
#: from body text.
FIELD_COLUMNS: Final = "columns"

#: Weight per field. ``columns`` starts level with body: a table is not "about" every
#: column it happens to hold. The labeled-query set should settle it, not a guess here.
FIELD_WEIGHTS: Final = {
    FIELD_TITLE: NAME_BOOST,
    FIELD_BODY: DESCRIPTION_BOOST,
    FIELD_COLUMNS: DESCRIPTION_BOOST,
}

#: Fields a caller holding only ``browse`` may match against. A body term would leak the
#: body by inference. Columns are here because ``WarehouseService.columns`` already
#: serves them at browse -- gating them higher would let the catalog list a table that
#: search cannot find by the same column.
BROWSE_FIELDS: Final = frozenset({FIELD_TITLE, FIELD_COLUMNS})

#: What a chunked entity is worth as a result: one per ``entity_type`` and
#: ``entity_id``, scored by its best chunk.
#:
#: Without this, a derivation matching three of its five fields is three results.
#: ``doc_id`` carries the chunk index and ``entity_id`` does not, so collapsing is a
#: ``GROUP BY``. Best rather than summed, because summing rewards a document for being
#: long -- the effect BM25's length normalization exists to remove.
COLLAPSE_BY: Final = ("entity_type", "entity_id")


@dataclass(frozen=True)
class SearchDoc:
    """One indexed artifact: its text, its facets, and the row it belongs to.

    A long row becomes several documents, and a child record (a message, a notebook
    cell) is a document of its own. Both carry the key of the row they came from, so
    dropping a row drops everything it produced without a second pass to find them.
    """

    entity_type: str
    entity_id: str
    title: str
    body: str = ""
    columns: str = ""
    #: The row this document came from, as ``(kind, id)``. A top-level artifact is its
    #: own: a derivation's is its own name. A child's is what contains it, so a cell's
    #: names its notebook. Defaults to the document itself, which is right for
    #: everything not contained by something else; the extractor overrides it.
    parent_type: str = ""
    parent_id: str = ""
    #: Position within the row, for a chunked entity. ``0`` for a whole-row document.
    chunk_index: int = 0
    #: Set on a late-chunked document: the row's title and text, and this chunk's span
    #: within it. The builder pools token vectors over ``context`` instead of embedding
    #: ``body`` alone. Not stored -- read once at embed time and dropped.
    context: str = ""
    span: tuple[int, int] | None = None
    verdict: str | None = None
    status: str | None = None
    incident_state: str | None = None
    created_at: datetime | None = None
    data_hash: str | None = None
    #: Where the UI should navigate. Resolved at index time so a result needs no lookup.
    route: str = ""

    # ``certified``, ``staleness`` and ``drift_state`` are in the parent's facet list
    # and deliberately absent: no indexed model has a column for any of them, so they
    # would be three filters that are always null.

    def __post_init__(self) -> None:
        """Point an unset parent key at the document itself, or reject a half-set one.

        Defaulting rather than requiring it keeps the common case -- an artifact nothing
        contains -- from restating its own identity, and means a document can never be
        built without a key. An empty one would group with nothing and outlive the row
        it came from.

        Both halves or neither: filling one in from the document's own identity makes a
        key that names another kind's type and this one's id, and that key matches
        whatever happens to be there. A null half is the way it would arise, so it is
        caught here rather than left to depend on ids not colliding.
        """
        if bool(self.parent_type) != bool(self.parent_id):
            raise ValueError(
                f"{self.entity_type}:{self.entity_id} has half a parent key "
                f"(parent_type={self.parent_type!r}, parent_id={self.parent_id!r}); "
                "give both or neither"
            )
        if not self.parent_type:
            object.__setattr__(self, "parent_type", self.entity_type)
            object.__setattr__(self, "parent_id", self.entity_id)

    @property
    def doc_id(self) -> str:
        """``type:id#chunk``, composite so a metric and a derivation may share a name.

        The chunk index is always present, never elided for chunk 0: omitting it lets
        an id containing ``#`` collide with another row's chunk, and a derivation name
        is a primary key with no character restriction.
        """
        return f"{self.entity_type}:{self.entity_id}#{self.chunk_index}"

    def text_fields(self) -> dict[str, str]:
        """The field name to text mapping this document contributes to the index."""
        return {
            FIELD_TITLE: self.title,
            FIELD_BODY: self.body,
            FIELD_COLUMNS: self.columns,
        }


def split_doc_id(doc_id: str) -> tuple[str, str]:
    """A document id back into ``(entity_type, entity_id)``, dropping its chunk.

    The inverse of :attr:`SearchDoc.doc_id`, and exact rather than best-effort. An
    ``entity_id`` may contain either separator, but the chunk index is always appended
    so the *last* ``#`` is the one to cut on, and an entity type is an identifier so the
    *first* ``:`` is the other.
    """
    entity_type, _, entity_id = doc_id.rsplit("#", 1)[0].partition(":")
    return entity_type, entity_id


#: Every table the index owns, in an order safe to drop them in. Named once so that
#: adding a table to :func:`create_sql` cannot leave a stale one behind on a rebuild,
#: which is how a table once survived a version bump and collided with its own DDL.
TABLES: Final = (
    "search_field_stats",
    "search_term_stats",
    "search_term",
    "search_doc_field",
    "search_doc",
    "search_meta",
)


def create_sql(dim: int) -> tuple[str, ...]:
    """Every statement that builds an empty index, in order.

    ``dim`` is measured from the embedder rather than hardcoded, so swapping the model
    for a smaller one is a configuration change and not an edit here.
    """
    return (
        """
        CREATE TABLE search_meta (
            key   VARCHAR PRIMARY KEY,
            value VARCHAR
        )
        """,
        f"""
        CREATE TABLE search_doc (
            doc_id         VARCHAR PRIMARY KEY,
            entity_type    VARCHAR NOT NULL,
            entity_id      VARCHAR NOT NULL,
            chunk_index    INTEGER NOT NULL,
            parent_type    VARCHAR NOT NULL,
            parent_id      VARCHAR NOT NULL,
            title          VARCHAR NOT NULL,
            body           VARCHAR NOT NULL,
            columns        VARCHAR NOT NULL,
            doc_hash       VARCHAR NOT NULL,
            vec            FLOAT[{dim}],
            verdict        VARCHAR,
            status         VARCHAR,
            incident_state VARCHAR,
            created_at     TIMESTAMP,
            data_hash      VARCHAR,
            route          VARCHAR NOT NULL
        )
        """,
        """
        CREATE TABLE search_doc_field (
            doc_id VARCHAR NOT NULL,
            field  VARCHAR NOT NULL,
            len    INTEGER NOT NULL,
            PRIMARY KEY (doc_id, field)
        )
        """,
        """
        CREATE TABLE search_term (
            doc_id VARCHAR NOT NULL,
            field  VARCHAR NOT NULL,
            term   VARCHAR NOT NULL,
            tf     INTEGER NOT NULL,
            PRIMARY KEY (doc_id, field, term)
        )
        """,
        """
        CREATE TABLE search_term_stats (
            term VARCHAR PRIMARY KEY,
            df   BIGINT NOT NULL
        )
        """,
        """
        CREATE TABLE search_field_stats (
            field     VARCHAR PRIMARY KEY,
            total_len BIGINT NOT NULL
        )
        """,
        # Lookup is by term: a query seeks its own postings rather than scanning.
        "CREATE INDEX ix_search_term_term ON search_term (term)",
        "CREATE INDEX ix_search_doc_type ON search_doc (entity_type)",
        # COLLAPSE_BY's grouping key.
        "CREATE INDEX ix_search_doc_entity ON search_doc (entity_type, entity_id)",
        # Everything one row produced, so dropping the row drops its chunks and
        # children in one statement.
        "CREATE INDEX ix_search_doc_parent ON search_doc (parent_type, parent_id)",
    )
