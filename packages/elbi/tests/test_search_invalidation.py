"""Freshness: a write to the store reaches the index without a rebuild.

The index is only worth having if what you just saved is findable, so these drive the
real store and assert from the outside -- search, not queue internals. The negative
cases carry as much weight: a rolled-back write must leave nothing behind, and a write
to somebody else's store must not be picked up, because a test suite runs several stores
in one process.

The second half is the asymmetry the ticket exists for. Content may fail open, so a
dropped change costs freshness the sweep repairs. Access may not: a document left
readable under an owner it no longer has is a disclosure, so it is withheld the moment
the store changes and stays withheld until it is rewritten.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("duckdb")

from sqlmodel import Session

from elbi.db import (
    Derivation,
    SavedQuery,
    Store,
    open_store,
)
from elbi.search.build import SearchBuilder
from elbi.search.index import SearchIndex
from elbi.search.invalidation import (
    Change,
    PendingWrites,
    install,
)
from elbi.search.schema import SearchDoc
from elbi.search.sources import Sources

DIM = 4


def _raises(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("the index is unwritable")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'app.db'}")


@pytest.fixture
def index(tmp_path: Path) -> Iterator[SearchIndex]:
    ix = SearchIndex(tmp_path / "search.duckdb", dim=DIM, embedding_model="stub")
    ix.open()
    yield ix
    ix.close()


@pytest.fixture
def wired(
    store: Store, index: SearchIndex
) -> Iterator[tuple[SearchBuilder, PendingWrites]]:
    """The wiring a server sets up: listeners on the store, a builder over the index."""
    pending = PendingWrites()
    remove = install(store._engine, pending)
    yield SearchBuilder(index, Sources(store=store)), pending
    remove()


def _row(store: Store, row: object) -> None:
    """Put one row in the store, for the shapes its helpers cannot construct."""
    with Session(store._engine) as session:
        session.add(row)
        session.commit()


def _titles(index: SearchIndex, query: str) -> set[str]:
    return {
        index.hydrate([doc_id])[0]["title"] for doc_id in index.search(query, limit=20)
    }


# --- Content: a write reaches the index --------------------------------------------


def test_a_saved_derivation_is_findable_without_a_rebuild(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """The point of the whole mechanism: no rebuild, no interval, just findable."""
    builder, pending = wired
    store.save_derivation(
        Derivation(name="quarterly_churn", question="churn by cohort", source="...")
    )

    assert builder.apply_pending(pending) >= 1
    assert "quarterly_churn" in _titles(builder._index, "quarterly_churn")


def test_an_edit_replaces_the_old_content(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    builder, pending = wired
    store.save_saved_query(
        SavedQuery(id="q1", name="active subscribers", sql="select 1")
    )
    builder.apply_pending(pending)

    store.save_saved_query(
        SavedQuery(id="q1", name="churned subscribers", sql="select 2")
    )
    builder.apply_pending(pending)

    assert _titles(builder._index, "subscribers") == {"churned subscribers"}


def test_a_deleted_row_leaves_no_document(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    builder, pending = wired
    store.save_saved_query(
        SavedQuery(id="q1", name="active subscribers", sql="select 1")
    )
    builder.apply_pending(pending)
    assert _titles(builder._index, "subscribers") == {"active subscribers"}

    store.delete_saved_query("q1")

    assert builder.apply_pending(pending) >= 1
    assert _titles(builder._index, "subscribers") == set()


def test_a_write_outside_the_stores_own_path_is_captured(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """Why the listener is on the ``Session`` class rather than ``Store._write``.

    ``DbJobStore`` is the standing example -- it shares the engine and opens its own
    sessions -- and it is not the only thing that can. A hook on the store's own write
    path sees none of them, so the mechanism is asserted directly here: a session opened
    against the engine by anything at all is captured.
    """
    from sqlmodel import Session

    builder, pending = wired
    with Session(store._engine) as session:
        session.add(SavedQuery(id="q9", name="quarterly revenue", sql="select 1"))
        session.commit()

    assert builder.apply_pending(pending) >= 1
    assert "quarterly revenue" in _titles(builder._index, "quarterly revenue")


def test_a_rolled_back_write_indexes_nothing(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """Capture happens inside the transaction, publication only after it commits.

    Without the split, a failed save would leave the index advertising a row the
    database never kept -- the one staleness a search box cannot explain.
    """
    builder, pending = wired
    with pytest.raises(RuntimeError), store._write() as session:
        session.add(Derivation(name="ghost", question="q", source="s"))
        session.flush()
        raise RuntimeError("the caller changed its mind")

    assert builder.apply_pending(pending) == 0
    assert _titles(builder._index, "ghost") == set()


def test_another_store_is_not_watched(
    wired: tuple[SearchBuilder, PendingWrites], store: Store, tmp_path: Path
) -> None:
    """Listening on the Session class means the bind filter is what scopes them."""
    _builder, pending = wired
    other = open_store(f"sqlite:{tmp_path / 'other.db'}")

    other.save_derivation(Derivation(name="elsewhere", question="q", source="s"))
    assert len(pending) == 0

    store.save_derivation(Derivation(name="here", question="q", source="s"))
    assert len(pending) > 0


def test_an_idle_pass_does_no_work(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    builder, pending = wired
    store.save_derivation(Derivation(name="d", question="q", source="s"))
    builder.apply_pending(pending)

    assert builder.apply_pending(pending) == 0


# --- The queue ----------------------------------------------------------------------


def test_the_queue_is_bounded() -> None:
    pending = PendingWrites(limit=2)
    pending.add([Change("derivation", f"d{i}") for i in range(10)])

    changes, _retitled, _kinds, dropped = pending.drain()
    assert len(changes) == 2
    assert dropped == 8


def test_a_repeat_is_not_counted_as_dropped() -> None:
    """A duplicate is already owed, so reporting it as a backlog would be a lie."""
    pending = PendingWrites(limit=1)
    pending.add([Change("derivation", "d")] * 5)

    changes, _retitled, _kinds, dropped = pending.drain()
    assert len(changes) == 1
    assert dropped == 0


# --- Access: withheld the moment the store changes ---------------------------------


def test_an_edit_that_deletes_and_reinserts_comes_back(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """Several kinds are edited as delete-then-insert across two separate commits.

    The drain must bring the document back rather than read the removal event as a
    verdict and forget it. Getting this wrong deletes an artifact from search every
    time someone edits it.
    """
    builder, pending = wired
    store.save_saved_query(
        SavedQuery(id="q1", name="active subscribers", sql="select 1")
    )
    builder.apply_pending(pending)

    store.save_saved_query(
        SavedQuery(id="q1", name="active subscribers", sql="select 2")
    )
    builder.apply_pending(pending)

    assert _titles(builder._index, "subscribers") == {"active subscribers"}


def test_renaming_a_parent_refreshes_the_children_it_labels(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """A message, a cell and an incident have no name of their own.

    Each is labelled by what contains it, so renaming the parent leaves every one of
    them advertising the old name -- and the entity the reader searched for comes back
    under a title that no longer exists anywhere.
    """
    builder, pending = wired
    notebook = store.create_notebook("quarterly review")
    store.add_cell(notebook, source="compute_margin_by_region()")
    builder.build()

    store.update_notebook(notebook, name="annual review")
    builder.apply_pending(pending)

    assert _titles(builder._index, "review") == {"annual review"}


def test_a_shrinking_entity_drops_the_chunks_it_no_longer_produces(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """A rewrite replaces only the doc_ids it writes, so the tail survived the edit.

    The incremental path could not see it: ``stale`` is keyed on the entity, and a
    re-extracted entity is never stale. So a cleared secret stayed searchable in the
    chunks past the new end.
    """
    builder, pending = wired
    notebook = store.create_notebook("quarterly")
    cell = store.add_cell(notebook, source="hunter2-correct-horse " * 400)
    builder.build()
    before = builder._index.doc_count()
    assert _titles(builder._index, "hunter2")

    store.update_cell(cell.id, source="cleared")
    builder.apply_pending(pending)

    assert builder._index.doc_count() < before
    assert _titles(builder._index, "hunter2") == set()


def test_removing_the_listeners_stops_the_capture(
    store: Store, index: SearchIndex
) -> None:
    """``install`` is per store; without a remover a test suite accumulates them all."""
    pending = PendingWrites()
    remove = install(store._engine, pending)
    store.save_derivation(Derivation(name="watched", question="q", source="s"))
    assert len(pending) > 0
    pending.drain()

    remove()
    store.save_derivation(Derivation(name="unwatched", question="q", source="s"))

    assert len(pending) == 0


# --- Recovery ----------------------------------------------------------------------


def test_a_vector_rebuild_re_embeds_nothing(index: SearchIndex, store: Store) -> None:
    """AC 7's last clause: the vectors are column data and were never at risk."""

    class _Counting:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: Any) -> list[list[float]]:
            self.calls += 1
            return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    embedder = _Counting()
    store.save_derivation(Derivation(name="churn_risk", question="q", source="s"))
    builder = SearchBuilder(index, Sources(store=store), embedder)
    builder.build()
    after_first = embedder.calls
    assert after_first > 0

    index.rebuild_vector_index()

    assert embedder.calls == after_first
    stored = (
        index._cursor()
        .execute("SELECT COUNT(*) FROM search_doc WHERE vec IS NOT NULL")
        .fetchone()
    )
    assert stored[0] > 0


def test_the_cli_does_not_wipe_a_lexical_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``status`` is read-only, and naming the model wrong made it destructive.

    A server on ``search: lexical`` records the embedding model as ``"lexical"``, and an
    index whose recorded model does not match is dropped rather than migrated -- so a
    command that guessed the ONNX name deleted the corpus it was asked to describe.
    """
    from elbi.search.document_map import EMBED_DIM

    path = tmp_path / "search.duckdb"
    written = SearchIndex(path, dim=EMBED_DIM, embedding_model="lexical")
    written.open()
    written.upsert([SearchDoc(entity_type="metric", entity_id="m", title="revenue")])
    assert written.doc_count() == 1
    written.close()

    reopened = SearchIndex(path, dim=EMBED_DIM, embedding_model="lexical")
    reopened.open()
    try:
        assert reopened.doc_count() == 1
    finally:
        reopened.close()

    # And the case that bit: the same file opened under the embedding model's name.
    mismatched = SearchIndex(
        path, dim=EMBED_DIM, embedding_model="BAAI/bge-small-en-v1.5"
    )
    mismatched.open()
    try:
        assert mismatched.doc_count() == 0, "a model change drops the file, by design"
    finally:
        mismatched.close()


# --- What the review found ----------------------------------------------------------


def test_a_rebuild_that_cannot_see_a_kind_does_not_delete_it(
    store: Store, index: SearchIndex
) -> None:
    """ "I did not look" is not "there are none", and a build deletes the unseen.

    The reindex command has no tracking server, so it cannot speak for models. Without
    saying so it read every indexed model as deleted -- the documented recovery command
    emptying the corpus it was run to repair.
    """
    from elbi.search.sources import MODEL

    index.upsert(
        [SearchDoc(entity_type=MODEL, entity_id="churn_clf", title="churn_clf")]
    )
    report = SearchBuilder(
        index, Sources(store=store, unavailable=frozenset({MODEL}))
    ).build()

    assert report.removed == 0
    models = (
        index._cursor()
        .execute("SELECT COUNT(*) FROM search_doc WHERE entity_type = 'model'")
        .fetchone()
    )
    assert models[0] == 1


def test_a_reader_cannot_recreate_the_index(tmp_path: Path) -> None:
    """A reader that rebuilds destroys the subject it was asked to report on."""
    from elbi.search.index import SearchIndexError

    path = tmp_path / "search.duckdb"
    written = SearchIndex(path, dim=DIM, embedding_model="stub")
    written.open()
    written.upsert([SearchDoc(entity_type="metric", entity_id="m", title="revenue")])
    written.close()

    reader = SearchIndex(path, dim=DIM, embedding_model="another-model")
    with pytest.raises(SearchIndexError, match="another build"):
        reader.open(recreate=False)

    survivor = SearchIndex(path, dim=DIM, embedding_model="stub")
    survivor.open()
    try:
        assert survivor.doc_count() == 1
    finally:
        survivor.close()


def test_a_rebuild_restores_a_distrusted_index(
    store: Store, index: SearchIndex
) -> None:
    """Distrust must not be a one-way door needing a human to notice."""
    store.save_derivation(Derivation(name="churn_risk", question="q", source="s"))
    builder = SearchBuilder(index, Sources(store=store))
    builder.build()
    index.set_trusted(False)
    assert index.lexical("churn_risk") == []

    builder.build()

    assert index.trusted()
    assert index.lexical("churn_risk")


def test_an_untrusted_index_reports_no_coverage(index: SearchIndex) -> None:
    """Otherwise the retriever believes it is covered and routes to a mute reader.

    ``WhenCaughtUp`` asks whether the index holds the corpus, then trusts the answer. An
    ungated reply meant an untrusted index passed the coverage check and the in-memory
    fallback -- which exists for exactly this -- was never reached.
    """
    index.upsert(
        [SearchDoc(entity_type="derivation", entity_id="churn_risk", title="churn")]
    )
    assert index.entity_ids("derivation") == {"churn_risk"}

    index.set_trusted(False)

    assert index.entity_ids("derivation") == set()


def test_a_failed_drain_keeps_what_it_could_not_apply(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """``drain`` empties the queue first, so a later failure would lose the batch."""
    builder, pending = wired
    store.save_derivation(Derivation(name="churn_risk", question="q", source="s"))
    assert len(pending) > 0

    builder._index.upsert = _raises  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        builder.apply_pending(pending)

    assert len(pending) > 0, "the batch must survive a failure to apply it"


def test_a_shrunk_entity_is_trimmed_beside_a_longer_one(
    wired: tuple[SearchBuilder, PendingWrites], store: Store
) -> None:
    """One drain carries entities of different lengths, and each has its own end.

    The query that finds trimmable chunks filtered at the largest count in the batch, so
    a shrunk entity whose new end fell below it was never returned to be judged. One
    entity per test could not see it: with a single count the largest and the smallest
    are the same number.
    """
    builder, pending = wired
    short = store.create_notebook("short")
    short_cell = store.add_cell(short, source="hunter2-correct-horse " * 400)
    long = store.create_notebook("long")
    store.add_cell(long, source="unrelated padding " * 4000)
    builder.build()
    assert _titles(builder._index, "hunter2")

    # Both in one drain, the long one still long: its count is what the batch filters
    # at, and the short one's surviving chunks all sit below it.
    store.update_cell(short_cell.id, source="cleared")
    builder.apply_pending(pending)

    assert _titles(builder._index, "hunter2") == set(), (
        "the shrunk entity kept the chunks past its new end while a longer entity "
        "shared its batch"
    )


def test_an_index_that_cannot_record_distrust_still_stops_serving(
    index: SearchIndex,
) -> None:
    """The one failure that most warrants stopping is the one that used to be ignored.

    ``set_trusted`` wrote the row and then assigned the flag, so a raising write left
    the process serving on. Its caller is a fallback whose reason for running is that
    the index misbehaved, and it only logs.
    """

    class Unwritable:
        """The connection, except that recording trust raises."""

        def __init__(self, con: Any) -> None:
            self._con = con

        def execute(self, sql: str, *args: object) -> Any:
            if "index_trusted" in sql:
                raise RuntimeError("cannot write")
            return self._con.execute(sql, *args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._con, name)

    original = index._con
    index._con = Unwritable(original)
    try:
        with pytest.raises(RuntimeError):
            index.set_trusted(False)
    finally:
        index._con = original

    assert not index.trusted(), "it could not write it down, so it served on"


def test_a_drain_that_outlives_the_join_gets_no_second_writer(
    store: Store, index: SearchIndex
) -> None:
    """The join has a timeout, so it can return with the worker still inside a write.

    ``stop`` then ran its own final pass and the caller closed the index next, which is
    two writers against a file one of them is about to close. Shutdown cannot block for
    ever, so the answer is to leave the work for the next build and say so.
    """
    import threading

    from elbi.search.build import Drain

    builder = SearchBuilder(index, Sources(store=store))
    pending = PendingWrites()
    release = threading.Event()
    applied: list[int] = []

    def slow(queue: PendingWrites) -> int:
        applied.append(1)
        release.wait(5)
        return 0

    builder.apply_pending = slow  # type: ignore[method-assign]
    drain = Drain(builder, pending, interval=0.01)
    drain.start()
    try:
        pending.add([Change("notebook", "nb-slow")])
        for _ in range(200):
            if applied:
                break
            time.sleep(0.01)
        assert applied, "the drain never picked the write up"

        # It is inside apply_pending and will not leave before the join gives up.
        drain._thread.join = lambda timeout=None: None  # type: ignore[union-attr]
        drain.stop()

        assert len(applied) == 1, "it started a second pass beside the running one"
    finally:
        release.set()
