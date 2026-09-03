"""The persisted search index: a DuckDB file that is a cache, not a database.

Everything here is recomputed from the store and the project registry, so the file can
be deleted at any time and the only cost is a rebuild. That is what lets a version
mismatch drop it rather than migrate it, and why no ``ALTER`` appears anywhere.

Three things about this are the first of their kind in this repository, so they are
stated rather than assumed. Nothing else opens a DuckDB file on disk, holds a
connection beyond one operation, or owns DuckDB DDL: every other use registers an Arrow
dataset into a fresh in-memory connection and throws it away. The precedents worth
following are therefore not the DuckDB ones. The file's lifecycle follows
:class:`~elbi_core.cache.LocalCacheStore`, a versioned derived artifact under
the cache directory. The locking follows ``DbJobStore``: a single writer under a lock
with lock-free readers, because DuckDB, like SQLite, permits one writer per file.

That also means one process per file. The index is local state under the project's cache
directory, rebuilt from the shared application database; it must never be put on a
volume mounted into more than one replica.

Queries are fixed, parameterized and ``LIMIT``-bounded over a small local file, which is
why they do not go through :mod:`elbi.query_runner`. That exists to isolate
user-authored warehouse SQL of unbounded cost, and routing through its two shared
subprocesses would make search latency depend on whatever query a user just started, as
well as giving up the buffer cache a long-lived connection keeps warm.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from elbi_core.text import TOKENIZER_VERSION, rrf_fuse, tokens

from . import sql as q
from .schema import SCHEMA_VERSION, TABLES, SearchDoc, create_sql, split_doc_id

logger = logging.getLogger(__name__)

#: How many candidates each half of the hybrid produces before fusion. Deep paging past
#: this re-queries with a larger window rather than silently returning less.
DEFAULT_CANDIDATES = 200

#: The widest the candidate window is allowed to grow while a page fills. The fold from
#: documents to results is unbounded in principle -- a corpus could hold one entity of
#: ten thousand chunks -- so the widening needs an end, and past this the honest answer
#: is a short page rather than a scan of everything.
MAX_SEARCH_CANDIDATES = 5_000


class SearchIndexError(RuntimeError):
    """The index could not be opened or rebuilt."""


def _is_lock_conflict(exc: Exception) -> bool:
    """Whether this open failure is another process holding the file.

    Matched on the message because DuckDB raises the same ``IOException`` for a lock it
    cannot take and a file it cannot read, and the two demand opposite responses: one
    must never delete the file, the other exists to.
    """
    return "conflicting lock" in str(exc).lower()


#: The HNSW index over the stored vectors. Named so it can be verified, dropped and
#: rebuilt by name rather than by scanning the catalog for whatever is there.
VECTOR_INDEX: Final = "ix_search_doc_vec"

#: The stored columns of ``search_doc``. Named in the ``INSERT`` rather than left to DDL
#: order, so adding a column to :func:`~.schema.create_sql` cannot shift every value one
#: place along.
DOC_COLUMNS: Final = (
    "doc_id",
    "entity_type",
    "entity_id",
    "chunk_index",
    "parent_type",
    "parent_id",
    "title",
    "body",
    "columns",
    "doc_hash",
    "vec",
    "verdict",
    "status",
    "incident_state",
    "created_at",
    "data_hash",
    "route",
)


def _beyond(doc_id: str, counts: Mapping[tuple[str, str], int]) -> bool:
    """Whether this document's chunk index is past what its entity now produces."""
    entity_type, entity_id = split_doc_id(doc_id)
    kept = counts.get((entity_type, entity_id))
    return kept is not None and int(doc_id.rsplit("#", 1)[1]) >= kept


def collapse(doc_ids: Iterable[str]) -> list[str]:
    """One document id per entity, keeping the best-ranked chunk of each.

    "Best" is first: the input is already in rank order, so a stable de-duplication is
    exactly "scored by its best chunk", with no second notion of best to disagree.
    """
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    for doc_id in doc_ids:
        key = split_doc_id(doc_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc_id)
    return out


#: What a result row needs. Everything stored except the vector and the digest, neither
#: of which a caller has any use for.
HYDRATED: Final = tuple(c for c in DOC_COLUMNS if c not in ("vec", "doc_hash"))


def _doc_row(doc: SearchDoc, vec: Any) -> tuple[Any, ...]:
    """One ``search_doc`` row, projected through :data:`DOC_COLUMNS`.

    A mapping rather than a tuple in the right order, so a renamed column raises here
    instead of writing a value into its neighbour.
    """
    values: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "entity_type": doc.entity_type,
        "entity_id": doc.entity_id,
        "chunk_index": doc.chunk_index,
        "parent_type": doc.parent_type,
        "parent_id": doc.parent_id,
        "title": doc.title,
        "body": doc.body,
        "columns": doc.columns,
        "doc_hash": _doc_hash(doc),
        "vec": vec,
        "verdict": doc.verdict,
        "status": doc.status,
        "incident_state": doc.incident_state,
        "created_at": doc.created_at,
        "data_hash": doc.data_hash,
        "route": doc.route,
    }
    return tuple(values[name] for name in DOC_COLUMNS)


def _doc_hash(doc: SearchDoc) -> str:
    """A digest of everything this document stores, so a sweep can skip what is settled.

    Every stored field, not only the text, and that is the whole point. Hashing the text
    alone answers "does this need re-embedding?", and a reconciliation pass asks a
    different question: "is the indexed row still right?" A derivation certified since
    the last pass has identical text and a new verdict, so a text-only digest calls it
    unchanged and the index goes on serving the old verdict -- against the very facet
    search exists to filter on.

    The vector is excluded because it is derived from the text rather than stored input,
    and ``doc_id`` because it is the key this digest is compared under.
    """
    digest = hashlib.sha256()
    for value in (
        doc.title,
        doc.body,
        doc.columns,
        doc.parent_type,
        doc.parent_id,
        doc.verdict,
        doc.status,
        doc.incident_state,
        doc.created_at,
        doc.data_hash,
        doc.route,
        # Not stored, but it decides the vector: editing cell 3 re-embeds cells 1 and 2.
        doc.context,
    ):
        digest.update(str(value).encode())
        digest.update(b"\x00")
    return digest.hexdigest()


class SearchIndex:
    """A hybrid index over every artifact, stored in one DuckDB file.

    Args:
        path: Where the file lives. Under the project's cache directory, beside the
            application database it derives from.
        dim: The embedding width, measured from the embedder rather than assumed, so
            swapping in a smaller model does not need an edit here.
        embedding_model: Recorded so a model change forces a rebuild; vectors from two
            models are not comparable.
    """

    def __init__(
        self, path: Path, dim: int | Callable[[], int], embedding_model: str
    ) -> None:
        self._path = path
        # A callable defers measuring the width, which would load the model.
        self._dim_source = dim
        self._dim_value: int | None = dim if isinstance(dim, int) else None
        self._model = embedding_model
        self._lock = threading.Lock()
        self._con: Any = None
        # Read once at open and updated by :meth:`set_trusted`, so refusing to serve
        # costs a boolean rather than a query on every search.
        self._trusted = True

    @property
    def _dim(self) -> int:
        """The embedding width, measured on first use and remembered."""
        if self._dim_value is None:
            source = self._dim_source
            self._dim_value = source() if callable(source) else source
        return self._dim_value

    def open(self, recreate: bool = True) -> None:
        """Open the file, recreating it when its version does not match this build.

        ``recreate=False`` refuses instead, for a reader that must not be able to
        destroy what it was asked to describe.
        """
        import duckdb

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SearchIndexError(f"cannot create {self._path.parent}") from exc
        try:
            self._con = duckdb.connect(str(self._path))
        except duckdb.Error as exc:
            if _is_lock_conflict(exc):
                # Deleting here would destroy a running server's working index.
                raise SearchIndexError(
                    f"the search index at {self._path} is held by another process"
                ) from exc
            # Nothing holds it and it still will not open: recreating is safe.
            logger.warning("search index unusable, recreating: %s", exc)
            try:
                self._path.unlink(missing_ok=True)
                self._con = duckdb.connect(str(self._path))
            except (duckdb.Error, OSError) as retry:
                raise SearchIndexError(f"cannot open {self._path}") from retry

        if not self._is_current():
            if not recreate:
                raise SearchIndexError(
                    f"the search index at {self._path} was written by another build"
                )
            self._recreate()
        self._trusted = self._read_trusted()
        self._ensure_vector_index()

    def close(self) -> None:
        """Release the file. Idempotent, so a failed startup can still shut down."""
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None

    def _cursor(self) -> Any:
        """A child cursor for a read.

        Readers take no lock: DuckDB gives a cursor a consistent snapshot, so a reindex
        in progress neither blocks a query nor shows it a half-written document. Child
        cursors share the parent's buffer cache, which is most of why a warm scan beats
        a cold one.
        """
        if self._con is None:
            raise SearchIndexError("search index is not open")
        return self._con.cursor()

    def _is_current(self) -> bool:
        """Whether the open file was written by this build, tokenizer and model."""
        try:
            rows = dict(
                self._con.execute("SELECT key, value FROM search_meta").fetchall()
            )
        except Exception:
            return False
        return (
            rows.get("schema_version") == str(SCHEMA_VERSION)
            and rows.get("tokenizer_version") == str(TOKENIZER_VERSION)
            and rows.get("embedding_model") == self._model
            and rows.get("embedding_dim") == str(self._dim)
        )

    def _recreate(self) -> None:
        """Drop everything and lay out an empty index at the current version."""
        with self._lock:
            for table in TABLES:
                self._con.execute(f"DROP TABLE IF EXISTS {table}")
            for statement in create_sql(self._dim):
                self._con.execute(statement)
            self._con.executemany(
                "INSERT INTO search_meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("tokenizer_version", str(TOKENIZER_VERSION)),
                    ("embedding_model", self._model),
                    ("embedding_dim", str(self._dim)),
                    ("n_docs", "0"),
                    ("state", "empty"),
                    ("index_trusted", "true"),
                ],
            )

    def _load_vss(self) -> bool:
        """Load ``vss`` and allow HNSW on a file-backed database.

        The flag is required: without it the binder refuses an HNSW index on anything
        but an in-memory database. DuckDB marks persistence experimental mainly over
        write-ahead-log recovery, so the index -- never the vectors -- can be left
        unusable after an unclean shutdown; :meth:`_ensure_vector_index` is the
        mitigation that makes the fast path safe.

        A load failure is not fatal: only the ANN shortcut is lost, and the same query
        answers by exact scan.
        """
        try:
            self._con.execute("INSTALL vss")
            self._con.execute("LOAD vss")
            self._con.execute("SET hnsw_enable_experimental_persistence = true")
        except Exception as exc:
            logger.info("vss unavailable, dense search will scan exactly: %s", exc)
            return False
        return True

    def _ensure_vector_index(self) -> None:
        """Create the HNSW index, or rebuild it if it is missing or unusable.

        Verified by querying *through* it rather than by asking whether it exists: a
        recovered index can be listed and still error. A rebuild reads the persisted
        vectors and re-embeds nothing.
        """
        if self._con is None or not self._load_vss():
            return
        with self._lock:
            if self._vector_index_usable():
                return
            # Over an empty file this is routine; over stored vectors it is a loss.
            if self._index_exists() or self._has_vectors():
                logger.warning(
                    "search vector index missing or unusable, rebuilding from the "
                    "persisted vectors"
                )
            try:
                self._con.execute(f"DROP INDEX IF EXISTS {VECTOR_INDEX}")
                self._con.execute(
                    f"CREATE INDEX {VECTOR_INDEX} ON search_doc "
                    "USING HNSW (vec) WITH (metric = 'cosine')"
                )
            except Exception as exc:
                logger.warning("could not build the vector index: %s", exc)

    def _index_exists(self) -> bool:
        """Whether the catalog lists the vector index, however usable it turns out."""
        listed = self._con.execute(
            "SELECT 1 FROM duckdb_indexes() WHERE index_name = ?", [VECTOR_INDEX]
        ).fetchone()
        return listed is not None

    def _has_vectors(self) -> bool:
        """Whether anything is embedded, so a rebuild would have something to read."""
        row = self._con.execute(
            "SELECT 1 FROM search_doc WHERE vec IS NOT NULL LIMIT 1"
        ).fetchone()
        return row is not None

    def _vector_index_usable(self) -> bool:
        """Whether a distance query answers through the index without erroring."""
        try:
            if not self._index_exists():
                return False
            self._con.execute(
                f"SELECT doc_id FROM search_doc "
                f"ORDER BY array_cosine_distance(vec, ?::FLOAT[{self._dim}]) LIMIT 1",
                [[0.0] * self._dim],
            ).fetchall()
        except Exception:
            return False
        return True

    def rebuild_vector_index(self) -> None:
        """Rebuild the vector index from the persisted vectors, re-embedding nothing.

        The recovery path a maintenance command calls, and also how the index is
        compacted: DuckDB tombstones an HNSW entry on delete rather than removing it, so
        a long-lived index over a churning corpus grows.
        """
        if self._con is None or not self._load_vss():
            return
        with self._lock:
            self._con.execute(f"DROP INDEX IF EXISTS {VECTOR_INDEX}")
            self._con.execute(
                f"CREATE INDEX {VECTOR_INDEX} ON search_doc "
                "USING HNSW (vec) WITH (metric = 'cosine')"
            )
        logger.info("search vector index rebuilt")

    def upsert(
        self, docs: Iterable[SearchDoc], vectors: Sequence[Any] | None = None
    ) -> int:
        """Index documents, replacing any already stored under the same id.

        Returns how many were written. The cost is a fixed number of statements over
        the changed documents' rows, not a pass over the corpus: measured at ~20 ms for
        one document against 50,000, where recreating DuckDB's ``fts`` index over the
        same corpus takes ~250 ms and grows linearly while this stays flat. That gap,
        not a raw speed, is the reason the index is maintained by hand.
        """
        materialized = list(docs)
        if not materialized:
            return 0
        vecs = list(vectors) if vectors is not None else [None] * len(materialized)

        doc_rows: list[tuple[Any, ...]] = []
        field_rows: list[tuple[Any, ...]] = []
        posting_rows: list[tuple[Any, ...]] = []
        for doc, vec in zip(materialized, vecs, strict=True):
            doc_id = doc.doc_id
            doc_rows.append(_doc_row(doc, vec))
            for name, text in doc.text_fields().items():
                field_tokens = tokens(text)
                if not field_tokens:
                    continue
                field_rows.append((doc_id, name, len(field_tokens)))
                posting_rows.extend(
                    (doc_id, name, term, count)
                    for term, count in Counter(field_tokens).items()
                )

        ids = [doc.doc_id for doc in materialized]
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._apply_stat_delta(ids, -1)
                self._replace(ids, doc_rows, field_rows, posting_rows)
                self._apply_stat_delta(ids, +1)
                self._set_doc_count()
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise
        return len(materialized)

    def _replace(
        self,
        ids: Sequence[str],
        doc_rows: Sequence[tuple[Any, ...]],
        field_rows: Sequence[tuple[Any, ...]],
        posting_rows: Sequence[tuple[Any, ...]],
    ) -> None:
        """Rewrite a whole batch in a fixed number of statements.

        Row-at-a-time DML is the wrong shape here: DuckDB pays a per-statement
        cost that dwarfs the work when the work is one row: writing a document as five
        statements measured 11 ms per document, so a cold build of fifty thousand would
        have taken ten minutes. Batched into a fixed number of statements, with the wide
        tables going through Arrow, the same build costs 0.3 ms per document.
        """
        self._con.execute(
            "DELETE FROM search_term WHERE doc_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [list(ids)],
        )
        self._con.execute(
            "DELETE FROM search_doc_field "
            "WHERE doc_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [list(ids)],
        )
        self._con.execute(
            "DELETE FROM search_doc WHERE doc_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [list(ids)],
        )
        placeholders = ", ".join("?" * len(DOC_COLUMNS))
        carried = list(doc_rows)
        self._con.executemany(
            f"INSERT INTO search_doc ({', '.join(DOC_COLUMNS)}) "
            f"VALUES ({placeholders})",
            carried,
        )
        # Arrow for the wide tables: executemany sends ~two dozen postings a row.
        self._insert_columnar(
            "search_doc_field", ("doc_id", "field", "len"), field_rows
        )
        self._insert_columnar(
            "search_term", ("doc_id", "field", "term", "tf"), posting_rows
        )

    def _insert_columnar(
        self, table: str, columns: Sequence[str], rows: Sequence[tuple[Any, ...]]
    ) -> None:
        """Append ``rows`` to ``table`` as one Arrow batch.

        Registered under a name and inserted from, rather than interpolated: the batch
        is data, and the table and column names are this module's own constants.
        """
        if not rows:
            return
        import pyarrow as pa

        batch = pa.table(
            {name: [row[i] for row in rows] for i, name in enumerate(columns)}
        )
        self._con.register("_batch", batch)
        try:
            self._con.execute(f"INSERT INTO {table} SELECT * FROM _batch")
        finally:
            self._con.unregister("_batch")

    def forget(self, doc_ids: Iterable[str]) -> int:
        """Remove documents and their postings, including anything they contain."""
        ids = list(doc_ids)
        if not ids:
            return 0
        # Every document names a parent key and a top-level row names itself, so the
        # entities being removed are excluded -- otherwise forgetting one notebook
        # cell would take the notebook and its siblings with it.
        entities = [
            f"{entity_type}:{entity_id}"
            for entity_type, entity_id in map(split_doc_id, ids)
        ]
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                children = [
                    row[0]
                    for row in self._con.execute(
                        "SELECT doc_id FROM search_doc "
                        "WHERE parent_type || ':' || parent_id "
                        "      IN (SELECT UNNEST(?::VARCHAR[])) "
                        "  AND entity_type || ':' || entity_id "
                        "      NOT IN (SELECT UNNEST(?::VARCHAR[]))",
                        [entities, entities],
                    ).fetchall()
                ]
                targets = [*ids, *children]
                self._apply_stat_delta(targets, -1)
                for table in ("search_term", "search_doc_field", "search_doc"):
                    self._con.execute(
                        f"DELETE FROM {table} "
                        "WHERE doc_id IN (SELECT UNNEST(?::VARCHAR[]))",
                        [targets],
                    )
                self._set_doc_count()
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise
        return len(targets)

    def _apply_stat_delta(self, ids: Sequence[str], sign: int) -> None:
        """Add or remove these documents' contribution to the corpus counters.

        Called with ``-1`` before a batch's rows are replaced and ``+1`` after, so the
        counters move by the difference rather than being recomputed.

        Recomputing them instead is tempting and was the first implementation: one
        aggregate over the whole term table cannot drift out of step with the rows it
        describes. It also made a single-document update cost a pass over every posting
        in the corpus, 171 ms against 50,000 documents where the delta costs 20 ms,
        which is the same O(corpus) write that rules out ``PRAGMA create_fts_index``. A
        counter that is maintained is the point; :meth:`recount` repairs it if it
        drifts.

        A document frequency counts a document once however many of its fields hold the
        term, which is what the in-memory ranker counts and why this is ``COUNT(DISTINCT
        doc_id)`` rather than a row count.
        """
        if not ids:
            return
        targets = list(ids)
        self._con.execute(
            f"""
            INSERT INTO search_term_stats (term, df)
            SELECT term, {sign} * COUNT(DISTINCT doc_id)
            FROM search_term WHERE doc_id IN (SELECT UNNEST(?::VARCHAR[]))
            GROUP BY term
            ON CONFLICT (term) DO UPDATE SET df = search_term_stats.df + excluded.df
            """,
            [targets],
        )
        self._con.execute(
            f"""
            INSERT INTO search_field_stats (field, total_len)
            SELECT field, {sign} * SUM(len)
            FROM search_doc_field WHERE doc_id IN (SELECT UNNEST(?::VARCHAR[]))
            GROUP BY field
            ON CONFLICT (field) DO UPDATE SET
                total_len = search_field_stats.total_len + excluded.total_len
            """,
            [targets],
        )
        # A term nobody holds would otherwise sit at zero and still be scored.
        self._con.execute("DELETE FROM search_term_stats WHERE df <= 0")
        self._con.execute("DELETE FROM search_field_stats WHERE total_len <= 0")

    def _set_doc_count(self) -> None:
        """Record the corpus size the scorer divides by."""
        n_docs = self._con.execute("SELECT COUNT() FROM search_doc").fetchone()[0]
        self._con.execute(
            "UPDATE search_meta SET value = ? WHERE key = 'n_docs'", [str(n_docs)]
        )

    def recount(self) -> None:
        """Rebuild the counters from the postings, whatever they currently say.

        The repair for the maintained counters above. Cheap relative to a full reindex
        (it re-embeds nothing), so a periodic sweep can run it rather than trusting that
        every write path got its arithmetic right.
        """
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._con.execute("DELETE FROM search_term_stats")
                self._con.execute(
                    """
                    INSERT INTO search_term_stats (term, df)
                    SELECT term, COUNT(DISTINCT doc_id) FROM search_term GROUP BY term
                    """
                )
                self._con.execute("DELETE FROM search_field_stats")
                self._con.execute(
                    """
                    INSERT INTO search_field_stats (field, total_len)
                    SELECT field, SUM(len) FROM search_doc_field GROUP BY field
                    """
                )
                self._set_doc_count()
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise

    def diff(self, docs: Sequence[SearchDoc]) -> tuple[list[SearchDoc], set[str]]:
        """Split ``docs`` into what actually changed, and name what has gone.

        This is what :attr:`doc_hash` is stored for. A full pass extracts everything,
        but rewriting everything would also re-*embed* everything, which is the
        expensive half and is pure waste for a document whose text is byte-identical to
        what is already indexed. Comparing digests turns a sweep over an unchanged
        corpus into a scan.

        The second return value is the other half of the same comparison: any indexed
        document that this pass did not produce no longer exists in the store. Without
        it a rebuild could only ever add, so a deleted artifact would stay findable for
        as long as the file lived.
        """
        stored = dict(
            self._cursor().execute("SELECT doc_id, doc_hash FROM search_doc").fetchall()
        )
        changed = [doc for doc in docs if stored.get(doc.doc_id) != _doc_hash(doc)]
        gone = stored.keys() - {doc.doc_id for doc in docs}
        return changed, set(gone)

    def entity_ids(self, entity_type: str) -> set[str]:
        """Every ``entity_id`` of one kind that this index holds.

        An id absent here cannot be returned by any query, however well it matches, so
        this is how a caller knows whether the index has caught up with it. Empty when
        the index is untrusted, because nothing it holds may be served -- a caller that
        read the real set would conclude it was covered and route to a reader that can
        only answer nothing, skipping the fallback that exists for exactly this.
        """
        if not self._trusted:
            return set()
        return {
            row[0]
            for row in self._cursor()
            .execute(
                "SELECT DISTINCT entity_id FROM search_doc WHERE entity_type = ?",
                [entity_type],
            )
            .fetchall()
        }

    def forget_entities(self, keys: Iterable[tuple[str, str]]) -> int:
        """Remove every document of these ``(entity_type, entity_id)`` pairs.

        By entity rather than document id because a chunked artifact is several
        documents and a caller deleting one row means all of them.
        """
        ids = [
            row[0]
            for row in self._cursor()
            .execute(
                "SELECT doc_id FROM search_doc WHERE entity_type || ':' || entity_id "
                "IN (SELECT UNNEST(?::VARCHAR[]))",
                [[f"{t}:{i}" for t, i in keys]],
            )
            .fetchall()
        ]
        return self.forget(ids) if ids else 0

    def trim_chunks(self, counts: Mapping[tuple[str, str], int]) -> int:
        """Remove the chunks an entity no longer produces. Returns how many went.

        A rewrite replaces only the doc_ids it is writing, so an entity whose text
        shrank keeps every chunk past its new end -- still indexed, still matching, with
        the body it had before the edit. The incremental path never noticed because
        ``stale`` is keyed on the entity, and a re-extracted entity is not stale.
        """
        doomed = [
            row[0]
            for row in self._cursor()
            .execute(
                "SELECT doc_id FROM search_doc "
                "WHERE entity_type || ':' || entity_id "
                "      IN (SELECT UNNEST(?::VARCHAR[])) "
                "  AND chunk_index >= ?",
                # The smallest count, not the largest: a batch holds entities of
                # different lengths, and ``_beyond`` can only judge a row the query
                # returned. Filtering at the largest hides every shrunk entity whose
                # new end falls below it, which is most of them.
                [[f"{t}:{i}" for t, i in counts], min(counts.values(), default=0)],
            )
            .fetchall()
            if _beyond(row[0], counts)
        ]
        return self.forget(doomed) if doomed else 0

    def set_trusted(self, trusted: bool) -> None:
        """Record whether this index may be served at all."""
        if not trusted:
            self._trusted = False
        with self._lock:
            self._con.execute(
                "UPDATE search_meta SET value = ? WHERE key = 'index_trusted'",
                ["true" if trusted else "false"],
            )
            self._trusted = trusted

    def trusted(self) -> bool:
        """Whether the index may be served."""
        return self._trusted

    def _read_trusted(self) -> bool:
        try:
            row = (
                self._cursor()
                .execute("SELECT value FROM search_meta WHERE key = 'index_trusted'")
                .fetchone()
            )
        except Exception:
            return False
        return row is None or row[0] != "false"

    def unembedded(self) -> set[str]:
        """Ids of indexed documents carrying no vector.

        The digest covers stored input, not the vector derived from it, so a document
        written while the embedder was failing looks settled forever without this.
        """
        return {
            row[0]
            for row in self._cursor()
            .execute("SELECT doc_id FROM search_doc WHERE vec IS NULL")
            .fetchall()
        }

    def doc_count(self) -> int:
        """How many documents are indexed."""
        return int(
            self._cursor().execute("SELECT COUNT() FROM search_doc").fetchone()[0]
        )

    def lexical(
        self,
        query: str,
        limit: int = DEFAULT_CANDIDATES,
        where_sql: str = q.UNRESTRICTED,
        where_params: Sequence[Any] = (),
    ) -> list[tuple[str, float]]:
        """BM25F over the index: ``(doc_id, score)`` pairs, best first.

        Every query term is scored. Dropping terms whose document frequency is high
        enough that they discriminate little is the usual optimization here, and it was
        tried: it makes this ranker disagree with the in-memory one, which is the single
        property the design rests on, and a generated corpus catches the disagreement
        immediately. Since the lexical half measures in single-digit milliseconds even
        with a term in a quarter of the corpus, it was buying latency that is not the
        cost centre at the price of the guarantee that matters.
        """
        if not self._trusted:
            return []
        n_docs = self._n_docs()
        statement, params = q.bm25_sql(
            sorted(set(tokens(query))),
            n_docs=n_docs,
            limit=limit,
            where_sql=where_sql,
            where_params=where_params,
        )
        return [
            (row[0], float(row[1]))
            for row in self._cursor().execute(statement, params).fetchall()
        ]

    def dense(
        self,
        vector: Sequence[float],
        limit: int = DEFAULT_CANDIDATES,
        where_sql: str = q.UNRESTRICTED,
        where_params: Sequence[Any] = (),
    ) -> list[tuple[str, float]]:
        """Cosine similarity over the stored vectors, best first."""
        if not self._trusted:
            return []
        statement, params = q.dense_sql(
            vector,
            self._dim,
            limit=limit,
            where_sql=where_sql,
            where_params=where_params,
        )
        return [
            (row[0], float(row[1]))
            for row in self._cursor().execute(statement, params).fetchall()
        ]

    def search(
        self,
        query: str,
        vector: Sequence[float] | None = None,
        limit: int = 20,
        candidates: int = DEFAULT_CANDIDATES,
        where_sql: str = q.UNRESTRICTED,
        where_params: Sequence[Any] = (),
    ) -> list[str]:
        """Fuse the lexical and dense rankings with RRF; return document ids.

        With no vector this is the lexical half alone, which is what ``search: lexical``
        selects: it constructs no embedder and so makes no outbound call.

        Chunks collapse to one result per entity after fusion, per
        :data:`~.schema.COLLAPSE_BY`: a derivation matching three of its fields is one
        result, represented by whichever chunk fused highest. After fusion rather than
        before, so "highest" means what the ranking means; ``candidates`` is wide enough
        that the fold cannot starve ``limit``.
        """
        rankings = [
            [
                doc
                for doc, _ in self.lexical(
                    query,
                    limit=candidates,
                    where_sql=where_sql,
                    where_params=where_params,
                )
            ]
        ]
        if vector is not None:
            rankings.append(
                [
                    doc
                    for doc, _ in self.dense(
                        vector,
                        limit=candidates,
                        where_sql=where_sql,
                        where_params=where_params,
                    )
                ]
            )
        return collapse(rrf_fuse(*rankings))[:limit]

    def facet_counts(
        self,
        column: str,
        query: str = "",
        vector: Sequence[float] | None = None,
        where_sql: str = q.UNRESTRICTED,
        where_params: Sequence[Any] = (),
    ) -> list[tuple[Any, int]]:
        """Count documents per value of one facet.

        Counted from the same predicate the results come from, so a facet cannot
        advertise rows the page does not show.

        A query that tokenizes to nothing counts nothing rather than everything -- a
        facet answering ``###`` with the whole corpus is a page arguing with itself.
        """
        if query and not tokens(query):
            return []
        if not self._trusted:
            return []
        statement, params = q.facet_counts_sql(
            column,
            terms=sorted(set(tokens(query))) if query else (),
            # The same vector the ranking used, so a semantically-matched result is
            # counted by the facet beside it rather than missing from every bucket.
            vector=vector,
            dim=self._dim if vector is not None else 0,
            where_sql=where_sql,
            where_params=where_params,
        )
        return [
            (row[0], int(row[1]))
            for row in self._cursor().execute(statement, params).fetchall()
        ]

    def hydrate(self, doc_ids: Sequence[str]) -> list[dict[str, Any]]:
        """The stored rows for ``doc_ids``, in the order given.

        Read as tuples and zipped against the cursor's column names rather than through
        Arrow: this is one page of results, so the columnar path buys nothing, and the
        vector column makes the conversion more trouble than the loop it replaces.
        """
        if not doc_ids or not self._trusted:
            # Untrusted means nothing here may be served, and a caller holding ids from
            # an earlier request would otherwise read the rows straight out. The gate
            # belongs on every reader, not only the one that produces the ids.
            return []
        # Named columns rather than ``*``: a page of results has no use for the vector,
        # and at 384 floats a row it is most of what ``*`` would transfer.
        cursor = self._cursor().execute(
            f"SELECT {', '.join(HYDRATED)} FROM search_doc d "
            "WHERE d.doc_id IN (SELECT UNNEST(?::VARCHAR[]))",
            [list(doc_ids)],
        )
        columns = [description[0] for description in cursor.description]
        by_id = {
            row[columns.index("doc_id")]: dict(zip(columns, row, strict=True))
            for row in cursor.fetchall()
        }
        return [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]

    def _n_docs(self) -> int:
        row = (
            self._cursor()
            .execute("SELECT value FROM search_meta WHERE key = 'n_docs'")
            .fetchone()
        )
        return int(row[0]) if row else 0
