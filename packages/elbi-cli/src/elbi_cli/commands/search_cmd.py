"""``elbi search``: inspect and rebuild the platform search index.

Recovery for when the index and the store disagree. Invalidation keeps them together
while a server is running, and the hourly sweep repairs whatever the queue dropped, but
neither helps an index that was lost, distrusted, or written by an older build. This is
the deliberate rebuild, run by hand.

DuckDB gives a database file to one process at a time, so these commands require the
server to be stopped. That is worth saying in the error rather than passing DuckDB's own
wording through, which reads like corruption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from elbi_core.errors import ElbiError

from .._console import arrow, fail, ok
from ..project import load_project

search_app = typer.Typer(
    name="search",
    help="Inspect and rebuild the platform search index.",
    no_args_is_help=True,
)


@dataclass
class _Opened:
    """Everything a command needs, resolved the way the server resolves it."""

    store: Any
    index: Any
    embedder: Any
    sources: Any


def _open(directory: Path, *, recreate: bool = True) -> _Opened:
    """The store, index, embedder and sources for this project.

    Every one of these is built the way ``serve.py`` builds it, and that is the point
    rather than tidiness: a rebuild deletes what a pass did not produce, so each
    dependency this command resolves differently from the server is a way to empty the
    index it was run to repair.

    Imported here rather than at module scope: the index needs the warehouse extra, and
    ``elbi --help`` must work without it.
    """
    from elbi.db import open_store
    from elbi.env import env
    from elbi.search.document_map import EMBED_DIM, EMBED_MODEL
    from elbi.search.index import SearchIndex, SearchIndexError
    from elbi.search.sources import MODEL, Sources
    from elbi_core.retrieval import OnnxEmbedder

    project = load_project(directory.resolve())
    # ``DB_URI`` first, exactly as serve.py:243 does. Hardcoding the SQLite fallback
    # points a Postgres deployment at a fresh empty database, and a rebuild from an
    # empty store deletes every document in the index.
    store = open_store(env("DB_URI") or f"sqlite:{project.cache_dir / 'app.db'}")
    lexical = project.config.search == "lexical"
    # The same model name the server records. A mismatch reads as a model change, and a
    # model change drops the file -- which made ``status`` destructive.
    index = SearchIndex(
        project.cache_dir / "search.duckdb",
        dim=EMBED_DIM,
        embedding_model="lexical" if lexical else EMBED_MODEL,
    )
    try:
        index.open(recreate=recreate)
    except SearchIndexError as exc:
        fail(f"{exc}.")
        arrow("stop the server, or run 'elbi search reindex' to rebuild")
        raise typer.Exit(code=1) from exc
    embedder = None if lexical else OnnxEmbedder()
    # This command has no tracking server, and "I did not look" is not "there are none".
    # Without saying so, a rebuild reads every indexed model as deleted.
    return _Opened(
        store=store,
        index=index,
        embedder=embedder,
        sources=Sources(store=store, unavailable=frozenset({MODEL})),
    )


@search_app.command("status")
def status(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
) -> None:
    """Show what the index holds, and whether it may be served."""
    try:
        # Never recreate from here: this command reports on the index, and one that
        # rebuilds when it does not recognise the file destroys what it was asked about.
        opened = _open(directory, recreate=False)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    index = opened.index
    try:
        arrow(f"documents: {index.doc_count()}")
        if not index.trusted():
            fail("this index is not served: a build left it in an unknown state")
            arrow("run 'elbi search reindex' to rebuild it")
        else:
            ok("the index is current enough to serve")
    finally:
        index.close()


@search_app.command("reindex")
def reindex(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
    vectors_only: bool = typer.Option(
        False,
        "--vectors-only",
        help="Rebuild only the vector index, from the embeddings already stored.",
    ),
) -> None:
    """Rebuild the index from the store.

    A full pass re-reads every document, so it repairs whatever diverged. Embeddings
    are reused where the text has not changed, so this is not the cost of a first build.
    """
    try:
        opened = _open(directory)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    index = opened.index
    try:
        if vectors_only:
            # The vectors are ordinary column data and were never at risk; only the
            # index over them is experimental, so this re-embeds nothing.
            index.rebuild_vector_index()
            ok("vector index rebuilt from the stored embeddings")
            return

        from elbi.search.build import SearchBuilder

        builder = SearchBuilder(index, opened.sources, opened.embedder)
        report = builder.build()
        arrow(f"{sum(report.total.values())} documents over {len(report.total)} types")
        ok(f"{sum(report.indexed.values())} written, {report.removed} removed")
        if report.total.get("model") is None:
            arrow("models were not reindexed: this command has no tracking server")
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    finally:
        index.close()
