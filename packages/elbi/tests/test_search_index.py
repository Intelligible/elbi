"""The corpus index: what it stores, what it ranks, and that anything fills it at all.

The last attempt shipped an indexer wired to every consumer with nothing calling a
build. Search returned nothing and the suite still passed, because every test wrote its
own documents first. So the first section never calls ``upsert``: it asks whether the
path a running server takes leaves the index non-empty.

The rest covers the parent's acceptance criteria -- the corpus restriction applied as
a filter in the query rather than after it, vectors surviving a restart, a lost vector
index rebuilt from them, and a lexical config that ranks with no embedder.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("duckdb")

from sqlmodel import Session

from elbi.db import LlmProfile, Store, open_store
from elbi.search.build import LEXICAL_ONLY, SearchBuilder
from elbi.search.extract import extract
from elbi.search.index import (
    VECTOR_INDEX,
    SearchIndex,
    SearchIndexError,
    collapse,
)
from elbi.search.retrievers import (
    IndexedBm25Retriever,
    WhenCaughtUp,
    indexed_retriever,
)
from elbi.search.schema import SearchDoc, split_doc_id
from elbi.search.sources import Sources
from elbi.search.specs import SPECS
from elbi_core.derivation import Derivation
from elbi_core.retrieval import Bm25Retriever

DIM = 8


class _StubEmbedder:
    """One dimension per letter of the alphabet's start.

    Enough structure that similar texts land near each other, with no model and no
    network, so the vector path runs offline.
    """

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        out = []
        for text in texts:
            vector = [0.0] * DIM
            for character in text.lower():
                position = ord(character) - ord("a")
                if 0 <= position < DIM:
                    vector[position] += 1.0
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            out.append([v / norm for v in vector])
        return out


def _index(tmp_path: Path, *, model: str = "stub") -> SearchIndex:
    index = SearchIndex(tmp_path / "search.duckdb", dim=DIM, embedding_model=model)
    index.open()
    return index


def _store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'app.db'}")


def _derivation(name: str, description: str | None = None) -> Derivation:
    return Derivation(name=name, compute=lambda ctx: None, description=description)


# --- Something calls a build ----------------------------------------------------


def test_the_maintenance_tick_is_what_fills_the_index(tmp_path: Path) -> None:
    """Through the server's own path: no test-authored documents anywhere.

    Writes a conversation through the store, runs the tick the scheduler runs, and asks
    the index whether anything is in it, so a build nobody calls fails here.
    """
    store = _store(tmp_path)
    store.create_conversation("Why did revenue fall in Q3?")
    index = _index(tmp_path)
    builder = SearchBuilder(index, Sources(store=store))

    assert index.doc_count() == 0
    builder.build()
    assert index.doc_count() > 0


def test_the_app_wires_the_tick_to_a_builder(tmp_path: Path) -> None:
    """``create_app`` exposes the tick, and the tick reaches the builder it was given.

    Against the app rather than the builder: the defect was in the wiring, not either
    piece -- both worked and nothing joined them.
    """
    from elbi.app import create_app

    store = _store(tmp_path)
    store.create_conversation("a question worth finding")
    index = _index(tmp_path)
    app = create_app(
        load_datasets=dict,
        store=store,
        search_builder=SearchBuilder(index, Sources(store=store)),
    )

    app.state.search_tick()
    assert index.doc_count() > 0


def test_a_second_build_writes_nothing(tmp_path: Path) -> None:
    """Idempotent: the maintenance pass runs repeatedly and must stay cheap."""
    store = _store(tmp_path)
    store.create_conversation("Why did revenue fall in Q3?")
    builder = SearchBuilder(_index(tmp_path), Sources(store=store))

    first = builder.build()
    second = builder.build()
    assert sum(first.indexed.values()) > 0
    assert sum(second.indexed.values()) == 0
    assert second.total == first.total


def test_a_deleted_artifact_stops_being_findable(tmp_path: Path) -> None:
    """A rebuild that only ever inserts leaves deleted rows in the index forever."""
    store = _store(tmp_path)
    conversation = store.create_conversation("Why did revenue fall in Q3?")
    index = _index(tmp_path)
    builder = SearchBuilder(index, Sources(store=store))
    builder.build()
    before = index.doc_count()

    store.delete_conversation(conversation)
    report = builder.build()

    assert report.removed > 0
    assert index.doc_count() < before
    assert index.lexical("revenue") == []


# --- Vectors ---------------------------------------------------------------------


def test_vectors_survive_a_restart_without_re_embedding(tmp_path: Path) -> None:
    """Boot must not re-embed: the vectors are ordinary column data."""
    index = _index(tmp_path)
    index.upsert(
        [SearchDoc(entity_type="metric", entity_id="churn", title="churn risk")],
        [[1.0] + [0.0] * (DIM - 1)],
    )
    index.close()

    reopened = _index(tmp_path)
    stored = (
        reopened._cursor()
        .execute("SELECT vec FROM search_doc WHERE doc_id = 'metric:churn#0'")
        .fetchone()
    )
    assert stored[0][0] == pytest.approx(1.0)
    assert reopened.dense([1.0] + [0.0] * (DIM - 1), limit=5)


def test_a_lost_vector_index_is_rebuilt_from_the_vectors(tmp_path: Path) -> None:
    """The unclean-shutdown safeguard, exercised by taking the index away."""
    import duckdb

    index = _index(tmp_path)
    index.upsert(
        [SearchDoc(entity_type="metric", entity_id="churn", title="churn risk")],
        [[1.0] + [0.0] * (DIM - 1)],
    )
    if not index._index_exists():
        pytest.skip("vss is not installable in this environment")
    index.close()

    connection = duckdb.connect(str(tmp_path / "search.duckdb"))
    connection.execute("LOAD vss")
    connection.execute("SET hnsw_enable_experimental_persistence = true")
    connection.execute(f"DROP INDEX {VECTOR_INDEX}")
    connection.close()

    reopened = _index(tmp_path)
    assert reopened._index_exists()
    stored = (
        reopened._cursor()
        .execute("SELECT vec FROM search_doc WHERE doc_id = 'metric:churn#0'")
        .fetchone()
    )
    assert stored[0][0] == pytest.approx(1.0)


def test_lexical_configuration_indexes_and_ranks_with_no_embedder(
    tmp_path: Path,
) -> None:
    """``search: lexical``: no model, no vectors, and still a ranking."""
    store = _store(tmp_path)
    store.create_conversation("Why did revenue fall in Q3?")
    index = _index(tmp_path, model="lexical")
    report = SearchBuilder(index, Sources(store=store), embedder=None).build()

    assert report.embedded == 0
    embedded = (
        index._cursor()
        .execute("SELECT COUNT(*) FROM search_doc WHERE vec IS NOT NULL")
        .fetchone()
    )
    assert embedded[0] == 0
    assert index.lexical("revenue")


def test_the_highest_count_entity_is_indexed_without_a_vector(tmp_path: Path) -> None:
    """``audit_event`` is most of the corpus and gains least from being embedded."""
    store = _store(tmp_path)
    store.record_audit("delete_dashboard", target_type="dashboard", target_id="revenue")
    index = _index(tmp_path)
    report = SearchBuilder(index, Sources(store=store), _StubEmbedder()).build()

    assert "audit_event" in LEXICAL_ONLY
    assert report.total["audit_event"] > 0
    rows = (
        index._cursor()
        .execute(
            "SELECT COUNT(*) FROM search_doc "
            "WHERE entity_type = 'audit_event' AND vec IS NOT NULL"
        )
        .fetchone()
    )
    assert rows[0] == 0
    assert index.lexical("dashboard")


# --- The retrieval seam -----------------------------------------------------------


def test_the_registry_and_the_index_rank_the_same_query_alike(tmp_path: Path) -> None:
    """One index, two consumers: the parent's claim, asserted rather than stated.

    A registry pointed at the index must return what the index returns, or
    ``search_derivations`` and platform search are two rankers agreeing by coincidence.
    """
    from elbi_core.registry import Registry

    index = _index(tmp_path)
    corpus = [
        _derivation("revenue_by_region", "Total revenue grouped by sales region."),
        _derivation("churn_risk", "Per-customer churn risk scores."),
    ]
    index.upsert(
        [
            SearchDoc(
                entity_type="derivation",
                entity_id=d.name,
                title=d.name,
                body=d.description or "",
            )
            for d in corpus
        ]
    )

    registry = Registry(retriever=indexed_retriever(index))
    for derivation in corpus:
        registry.register(derivation)

    through_registry = [d.name for d in registry.search("revenue", limit=5)]
    through_index = [
        split_doc_id(doc_id)[1] for doc_id in index.search("revenue", limit=5)
    ]
    assert through_registry == through_index


def test_an_unbuilt_index_falls_back_to_the_in_memory_ranker(tmp_path: Path) -> None:
    """The failure mode this guards: wired to an index nothing has filled yet."""
    index = _index(tmp_path)
    corpus = [_derivation("revenue_by_region", "Total revenue by region.")]
    retriever = WhenCaughtUp(index, indexed_retriever(index), Bm25Retriever())

    assert [d.name for d in retriever.search("revenue", corpus, limit=5)] == [
        "revenue_by_region"
    ]


def test_a_derivation_the_index_has_not_seen_is_still_findable(tmp_path: Path) -> None:
    """The index refreshes hourly; a just-certified derivation must not wait for it.

    Deferring to the index unconditionally means certifying a derivation and being told
    by the agent that it does not exist, for up to a maintenance interval.
    """
    index = _index(tmp_path)
    # A non-empty index, of a kind that says nothing about derivation coverage --
    # audit events alone used to latch the retriever onto the index forever.
    index.upsert(
        [SearchDoc(entity_type="audit_event", entity_id="a1", title="deleted")]
    )
    index.upsert(
        [SearchDoc(entity_type="derivation", entity_id="old_one", title="old_one")]
    )

    fresh = _derivation("revenue_by_region", "Total revenue by region.")
    retriever = WhenCaughtUp(index, indexed_retriever(index), Bm25Retriever())

    assert [d.name for d in retriever.search("revenue", [fresh], limit=5)] == [
        "revenue_by_region"
    ]


def test_an_unreadable_index_ranks_in_memory_rather_than_raising(
    tmp_path: Path,
) -> None:
    """A broken index must cost ranking quality, not the request."""
    index = _index(tmp_path)
    corpus = [_derivation("revenue_by_region", "Total revenue by region.")]
    index.upsert(
        [
            SearchDoc(entity_type="derivation", entity_id=d.name, title=d.name)
            for d in corpus
        ]
    )
    retriever = WhenCaughtUp(index, indexed_retriever(index), Bm25Retriever())
    index.close()  # what the lifespan does while a request is still in flight

    assert [d.name for d in retriever.search("revenue", corpus, limit=5)] == [
        "revenue_by_region"
    ]


def test_only_the_eligible_derivations_are_ranked(tmp_path: Path) -> None:
    """The corpus restriction is a filter in the query, not a post-filter.

    ``search_derivations`` passes only served, certified derivations, on the stated
    invariant that a proposed one can never crowd out a real match.
    """
    index = _index(tmp_path)
    index.upsert(
        [
            SearchDoc(
                entity_type="derivation",
                entity_id=f"draft_{i}",
                title="revenue by region",
                body="revenue revenue revenue",
            )
            for i in range(50)
        ]
        + [
            SearchDoc(
                entity_type="derivation",
                entity_id="certified_one",
                title="revenue",
                body="revenue by region",
            )
        ]
    )
    eligible = [_derivation("certified_one", "revenue by region")]
    retriever = IndexedBm25Retriever(index, candidates=5)

    # With a window of 5 over 51 documents, a post-filter would return nothing.
    assert [d.name for d in retriever.search("revenue", eligible, limit=5)] == [
        "certified_one"
    ]


def test_a_retriever_abstains_rather_than_returning_everything(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.upsert(
        [SearchDoc(entity_type="derivation", entity_id="churn", title="churn")]
    )
    retriever = indexed_retriever(index)
    assert retriever.search("photosynthesis", [_derivation("churn")], limit=5) == []


# --- Chunking -------------------------------------------------------------------


def test_a_chunked_entity_is_one_result(tmp_path: Path) -> None:
    """:data:`COLLAPSE_BY`: a derivation matching three fields is not three results."""
    index = _index(tmp_path)
    index.upsert(
        [
            SearchDoc(
                entity_type="derivation",
                entity_id="churn_risk",
                title="churn risk",
                body="churn",
                chunk_index=chunk,
            )
            for chunk in range(3)
        ]
    )
    assert index.search("churn", limit=5) == ["derivation:churn_risk#0"]


def test_collapse_keeps_the_best_chunk_of_each_entity() -> None:
    assert collapse(["a:x#2", "a:x#0", "b:y#0", "a:x#1"]) == ["a:x#2", "b:y#0"]


@pytest.mark.parametrize(
    ("doc_id", "expected"),
    [
        ("derivation:churn#0", ("derivation", "churn")),
        # A name may contain either separator; the chunk suffix is what disambiguates.
        ("derivation:a#b#3", ("derivation", "a#b")),
        ("derivation:a:b#0", ("derivation", "a:b")),
    ],
)
def test_a_document_id_splits_back_exactly(
    doc_id: str, expected: tuple[str, str]
) -> None:
    assert split_doc_id(doc_id) == expected


# --- Late chunking ----------------------------------------------------------------


class _SpanEmbedder(_StubEmbedder):
    """A span embedder that records what it was asked to pool over."""

    def __init__(self) -> None:
        self.contexts: list[str] = []

    def embed_spans(
        self, text: str, spans: Sequence[tuple[int, int]]
    ) -> list[Sequence[float]]:
        self.contexts.append(text)
        # Deliberately not text[start:end]: a chunk embedded in context must differ from
        # the same chunk embedded alone, or "late" is a label rather than a behaviour.
        return self.embed([f"{text} {text[start:end]}" for start, end in spans])


def test_a_late_chunk_is_embedded_in_the_context_of_its_row(tmp_path: Path) -> None:
    embedder = _SpanEmbedder()
    builder = SearchBuilder(_index(tmp_path), Sources(store=_store(tmp_path)), embedder)
    whole = "the notebook is about churn. then we filtered to Q3."
    docs = [
        SearchDoc(
            entity_type="notebook_cell",
            entity_id=f"n/{i}",
            title="churn analysis",
            body=whole[start:end],
            chunk_index=i,
            context=whole,
            span=(start, end),
        )
        for i, (start, end) in enumerate([(0, 30), (30, len(whole))])
    ]
    vectors = builder._embed(docs)

    assert embedder.contexts == [whole]  # one pass over the parent, not one per chunk
    assert all(v is not None for v in vectors)
    assert vectors[0] != vectors[1]


def test_a_static_embedder_degrades_to_per_chunk(tmp_path: Path) -> None:
    """``Model2VecEmbedder`` has no context to pool: it must fall back, not fail."""
    embedder = _StubEmbedder()
    builder = SearchBuilder(_index(tmp_path), Sources(store=_store(tmp_path)), embedder)
    docs = [
        SearchDoc(
            entity_type="notebook_cell",
            entity_id="n/0",
            title="churn analysis",
            body="then we filtered to Q3",
            context="the notebook is about churn. then we filtered to Q3",
            span=(28, 50),
        )
    ]
    assert builder._embed(docs)[0] is not None


def test_editing_one_late_chunk_re_embeds_its_siblings(tmp_path: Path) -> None:
    """A sibling's vector depends on the parent, so a changed parent invalidates it."""
    index = _index(tmp_path)
    unchanged = SearchDoc(
        entity_type="notebook_cell",
        entity_id="n/0",
        title="analysis",
        body="first cell",
        context="first cell. second cell",
        span=(0, 10),
    )
    index.upsert([unchanged], [[1.0] + [0.0] * (DIM - 1)])

    moved = SearchDoc(**{**unchanged.__dict__, "context": "first cell. third cell"})
    changed, _ = index.diff([moved])
    assert [d.doc_id for d in changed] == [moved.doc_id]


# --- Counts against the estimate --------------------------------------------------


def test_the_build_reports_counts_against_the_estimate(tmp_path: Path) -> None:
    """The parent's no-vector-store decision rests on an estimate; this checks it."""
    from elbi.search.build import counts_against_estimate

    store = _store(tmp_path)
    store.create_conversation("Why did revenue fall in Q3?")
    report = SearchBuilder(_index(tmp_path), Sources(store=store)).build()

    rows = dict[str, tuple[int, int]]()
    for name, actual, expected in counts_against_estimate(report):
        rows[name] = (actual, expected)
    assert rows["conversation"][0] > 0
    assert rows["conversation"][1] > 0
    # Every entity the mapping estimates is accounted for, even at zero rows.
    assert all(isinstance(value, tuple) for value in rows.values())


def test_the_registry_source_is_skipped_without_a_service(tmp_path: Path) -> None:
    """An installation with no tracking server still has a searchable corpus."""
    from elbi.search.sources import registry_models

    assert list(registry_models(Sources(store=_store(tmp_path)))) == []


class _Broken:
    """A model service whose registry cannot be reached."""

    def registry(self) -> Any:
        raise RuntimeError("tracking server is down")


def test_an_unreachable_registry_does_not_fail_the_build(tmp_path: Path) -> None:
    """The rest of the corpus still indexes; only the models are left unknown."""
    from elbi.search.extract import extract_all

    store = _store(tmp_path)
    store.create_conversation("Why did revenue fall in Q3?")
    extraction = extract_all(Sources(store=store, models=_Broken()))

    assert extraction.incomplete == frozenset({"model"})
    assert any(doc.entity_type == "conversation" for doc in extraction.docs)


def test_an_unreachable_registry_does_not_delete_the_models(tmp_path: Path) -> None:
    """ "Produced nothing" and "could not look" look alike to a diff, and must not be.

    A restarting tracking server used to empty the model corpus: the source swallowed
    its failure, yielded nothing, and every indexed model landed in the deletion set.
    """
    index = _index(tmp_path)
    index.upsert(
        [SearchDoc(entity_type="model", entity_id="churn_clf", title="churn_clf")]
    )
    store = _store(tmp_path)
    report = SearchBuilder(index, Sources(store=store, models=_Broken())).build()

    assert report.removed == 0
    models = (
        index._cursor()
        .execute("SELECT COUNT(*) FROM search_doc WHERE entity_type = 'model'")
        .fetchone()
    )
    assert models[0] == 1
    assert index.lexical("churn_clf")


def test_a_colliding_document_id_does_not_abort_the_build(tmp_path: Path) -> None:
    """A duplicate id is the primary key, so it aborted the whole batch.

    Two warehouse sources of the same type syncing a same-named table produce one
    ``doc_id`` twice, rolling the transaction back every pass -- so nothing was ever
    indexed and platform search stayed empty.
    """
    from elbi.search.build import _deduplicated

    twins = [
        SearchDoc(
            entity_type="warehouse_table",
            entity_id="postgres__users",
            title="postgres__users",
            route=f"/warehouse?table=postgres__users&source={source}",
        )
        for source in ("prod", "staging")
    ]
    kept = _deduplicated(
        [*twins, SearchDoc(entity_type="metric", entity_id="m", title="m")]
    )

    assert len(kept) == 2
    index = _index(tmp_path)
    assert index.upsert(kept) == 2
    assert index.lexical("postgres__users")


def test_a_document_left_unembedded_is_embedded_next_time(tmp_path: Path) -> None:
    """The digest covers text, not the vector, so a failed embed looked settled."""

    class _Flaky:
        def __init__(self) -> None:
            self.healthy = False

        def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
            if not self.healthy:
                raise RuntimeError("model not loaded")
            return _StubEmbedder().embed(texts)

    index = _index(tmp_path)
    store = _store(tmp_path)
    store.create_conversation("Why did revenue fall in Q3?")
    embedder = _Flaky()
    builder = SearchBuilder(index, Sources(store=store), embedder)

    first = builder.build()
    assert first.embedded == 0
    assert sum(first.indexed.values()) > 0  # still indexed, still findable lexically

    embedder.healthy = True
    second = builder.build()
    assert second.embedded > 0
    stored = (
        index._cursor()
        .execute("SELECT COUNT(*) FROM search_doc WHERE vec IS NOT NULL")
        .fetchone()
    )
    assert stored[0] > 0


def test_forgetting_a_parent_forgets_what_it_contains(tmp_path: Path) -> None:
    """A parent key names a row, not one of its documents; the join never matched."""
    index = _index(tmp_path)
    index.upsert(
        [
            SearchDoc(entity_type="conversation", entity_id="c1", title="revenue talk"),
            SearchDoc(
                entity_type="message",
                entity_id="c1/m1",
                title="revenue talk",
                body="revenue dropped in the east",
                parent_type="conversation",
                parent_id="c1",
            ),
        ]
    )
    removed = index.forget(["conversation:c1#0"])

    assert removed == 2
    assert index.doc_count() == 0
    assert index.lexical("revenue") == []


# --- The boot path ----------------------------------------------------------------


def test_booting_a_server_downloads_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the index must not resolve the embedder.

    The DDL needs the vector width at open, and measuring it means loading the model --
    which would put a weight download in the boot path of every server, including a
    sealed one configured never to embed. The width is declared instead, and this fails
    if it ever goes back to being measured.
    """
    pytest.importorskip("deltalake")
    from elbi_core.retrieval import OnnxEmbedder

    def _refuse(self: OnnxEmbedder) -> None:
        raise AssertionError("the embedder was loaded while the server was starting")

    monkeypatch.setattr(OnnxEmbedder, "_load", _refuse)
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'wh'}")

    root = tmp_path / "proj"
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text("customer_id,amount\nc1,100\n")
    (root / "elbi.yaml").write_text(
        "project: t\nsources:\n  - name: sales\n    type: csv\n"
        "    path: ./fixtures/sales.csv\n"
    )

    from elbi.serve import build

    app = build(root, with_mcp=False)
    assert app.state.search_tick is not None


# --- The two rankers agree --------------------------------------------------------
#
# The property the whole design rests on: BM25F in SQL must order a corpus exactly as
# the in-memory ranker does, or a query is answered differently depending on whether an
# index happens to exist. A hand-written example proves nothing here -- the failure mode
# is a corpus shape nobody thought of -- so this is generated.


def _agree(pairs: list[tuple[str, str]], query: str) -> None:
    """Index ``pairs`` as derivations and assert both rankers order them alike.

    Each call gets its own file: a function-scoped ``tmp_path`` is created once for the
    whole property, so sharing one would let every generated corpus accumulate into the
    index the next example queries.
    """
    corpus = [_derivation(name, description) for name, description in pairs]
    index = _index(Path(tempfile.mkdtemp()))
    index.upsert(
        [
            SearchDoc(
                entity_type="derivation",
                entity_id=d.name,
                title=d.name,
                body=d.description or "",
            )
            for d in corpus
        ]
    )
    in_memory = [
        d.name for d in Bm25Retriever().search(query, corpus, limit=len(corpus))
    ]
    ranked = index.lexical(query, limit=len(corpus))
    through_sql = [split_doc_id(doc_id)[1] for doc_id, _ in ranked]

    if in_memory != through_sql:
        # Only a tie may differ, and only in the last few bits: both sort by score then
        # by name, so anything else is a real disagreement in the scoring itself.
        scores = {split_doc_id(doc_id)[1]: score for doc_id, score in ranked}
        assert set(in_memory) == set(through_sql), (in_memory, through_sql)
        for left, right in zip(in_memory, through_sql, strict=True):
            assert scores[left] == pytest.approx(scores[right], abs=1e-9), (
                in_memory,
                through_sql,
                scores,
            )
    index.close()


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.text(alphabet="abcde_", min_size=1, max_size=10),
            st.text(alphabet="abcde ", min_size=0, max_size=24),
        ),
        min_size=1,
        max_size=6,
        unique_by=lambda pair: pair[0],
    ),
    st.text(alphabet="abcde ", min_size=1, max_size=8),
)
def test_the_sql_ranker_agrees_with_the_in_memory_one(
    pairs: list[tuple[str, str]], query: str
) -> None:
    _agree(pairs, query)


def test_a_rare_term_outranks_a_common_one_through_the_index() -> None:
    """The IDF case, pinned by name: it is what a generated corpus rarely isolates."""
    _agree(
        [("aaaone", "alpha"), ("aaatwo", "alpha"), ("zzztarget", "beta")], "alpha beta"
    )


def test_the_focused_match_outranks_the_padded_one_through_the_index() -> None:
    """Per-field length normalization -- the property ``fts`` cannot express."""
    _agree(
        [("zfocused", "term"), ("apadded", "term filler filler filler filler")], "term"
    )


def test_a_held_index_is_never_deleted(tmp_path: Path) -> None:
    """A second server must not destroy the running one's corpus.

    DuckDB raises the same error for a lock it cannot take and a file it cannot read,
    and recreate-on-open treated both as corruption -- so a boot racing another server's
    lock deleted a *working* index and re-embedded the whole corpus.
    """
    index = _index(tmp_path)
    index.upsert([SearchDoc(entity_type="metric", entity_id="churn", title="churn")])
    index.close()

    # The lock is per process: DuckDB hands a second connection in the *same* process
    # the same database instance, so the conflict only exists across processes -- which
    # is why an in-process test would pass while the real case destroyed data.
    path = tmp_path / "search.duckdb"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            # Bound to a name: an unassigned connection is freed at once, which
            # closes it, and the contender then opens the file with no conflict.
            "import duckdb, sys; con = duckdb.connect(sys.argv[1]); "
            "print('held', flush=True); sys.stdin.readline(); con.close()",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        contender = SearchIndex(path, dim=DIM, embedding_model="stub")
        with pytest.raises(SearchIndexError, match="another process"):
            contender.open()
    finally:
        holder.communicate("\n", timeout=10)

    # The point of the test: the documents are still there.
    reopened = SearchIndex(path, dim=DIM, embedding_model="stub")
    reopened.open()
    assert reopened.doc_count() == 1
    reopened.close()


def test_an_unreadable_index_is_still_recreated(tmp_path: Path) -> None:
    """The case the recreate path exists for, kept working: garbage nobody holds."""
    path = tmp_path / "search.duckdb"
    path.write_bytes(b"not a database, just bytes")
    index = SearchIndex(path, dim=DIM, embedding_model="stub")
    index.open()

    assert index.doc_count() == 0
    index.upsert([SearchDoc(entity_type="metric", entity_id="churn", title="churn")])
    assert index.doc_count() == 1


def test_a_rebuild_drops_every_table_the_schema_creates() -> None:
    """A table the drop list forgets survives a version bump and collides with its DDL.

    Deriving one list from the other is the fix; this is what keeps them derived.
    """
    import re

    from elbi.search.schema import TABLES, create_sql

    created = set(re.findall(r"CREATE TABLE (\w+)", " ".join(create_sql(4))))

    assert set(TABLES) == created


def test_a_single_column_entity_still_indexes(tmp_path: Path) -> None:
    """``llm_profile`` and ``setting`` each declare exactly one column.

    ``sqlmodel.select`` is a *scalar* select when handed one column and ``Session.exec``
    unwraps it to bare strings, so every attribute read came back ``None`` and the row
    was skipped -- all three absent from search, with no error and no log line.
    """
    store = _store(tmp_path)
    store.set_config("theme", "dark")
    store.save_profile(LlmProfile(name="cheap", model="haiku"))
    index = _index(tmp_path)
    report = SearchBuilder(index, Sources(store=store)).build()

    assert report.total["setting"] > 0
    assert report.total["llm_profile"] > 0
    assert index.lexical("theme")


def test_an_unwritable_cache_directory_does_not_stop_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Search degrades; the deployment does not go down over it.

    ``open()`` raised ``SearchIndexError`` only for a lock conflict, so an ``OSError``
    from the mkdir escaped ``serve.build`` and the process never started.
    """
    from elbi.search.index import SearchIndex as _Index

    def _refuse(self: Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", _refuse)
    index = _Index(tmp_path / "nope" / "search.duckdb", dim=DIM, embedding_model="stub")
    with pytest.raises(SearchIndexError, match="cannot create"):
        index.open()


def test_a_late_chunk_is_pooled_over_its_title_too(tmp_path: Path) -> None:
    """A cell's only human label is its notebook's name, so its vector must carry it.

    Every other document embeds ``title body``; a late chunk is pooled over a context,
    and that context was the body alone -- so a semantic query naming the notebook could
    not reach its cells.
    """
    store = _store(tmp_path)
    conversation = store.create_conversation("Why did revenue fall in Q3?")
    store.add_message(conversation, "user", content="we filtered to the east region")
    spec = next(s for s in SPECS if s.entity_type == "message")

    with Session(store._engine) as session:
        docs = list(extract(spec, session))

    assert docs
    for doc in docs:
        assert doc.context.startswith(doc.title)
        # The span still selects the chunk itself: the title reaches it through
        # attention, not by being concatenated onto the text that gets embedded.
        assert doc.span is not None
        assert doc.context[doc.span[0] : doc.span[1]] == doc.body


def test_an_unchunked_entity_carries_no_context(tmp_path: Path) -> None:
    """Only late-chunked kinds pay for a context; the rest embed ``title body``."""
    store = _store(tmp_path)
    store.create_conversation("Why did revenue fall in Q3?")
    spec = next(s for s in SPECS if s.entity_type == "conversation")

    with Session(store._engine) as session:
        docs = list(extract(spec, session))

    assert docs
    assert all(doc.context == "" and doc.span is None for doc in docs)


def test_the_two_rankers_diverge_once_an_entity_is_chunked(tmp_path: Path) -> None:
    """The parity claim holds for one-document entities, and only for those.

    ``_agree`` writes one document per derivation, so it proves the unchunked case and
    says nothing about the chunked one. Scoring an artifact by its best chunk is a
    different function -- this asserts the boundary rather than leaving a reader to
    assume there isn't one.
    """
    corpus = [
        _derivation("churn_risk", "revenue by region and the customer churn score"),
        _derivation("price_effects", "median price for each region"),
    ]
    index = _index(tmp_path)
    # The same artifacts, one document per field, every chunk carrying the title.
    index.upsert(
        [
            SearchDoc(
                entity_type="derivation",
                entity_id=d.name,
                title=d.name,
                body=part,
                chunk_index=position,
            )
            for d in corpus
            for position, part in enumerate((d.description or "").split(" and "))
        ]
    )

    in_memory = [d.name for d in Bm25Retriever().search("region", corpus, limit=5)]
    through_sql: list[str] = []
    for doc_id in index.search("region", limit=5):
        name = split_doc_id(doc_id)[1]
        if name not in through_sql:
            through_sql.append(name)

    # Both find both artifacts; the claim under test is only that neither invents or
    # drops one, not that a chunked corpus orders identically.
    assert set(in_memory) == set(through_sql) == {d.name for d in corpus}
