"""Notebook data access: bounded server-side queries, and datasets fetched on demand.

A notebook used to receive every bound dataset as a list of row dicts, serialised to
JSON and parsed before the first cell ran. That is roughly 30-50x the memory of the
equivalent columnar bytes, paid whether a cell touches the dataset or not, so the
kernel's memory limit, not the data, decided what could be analysed.

Two things replace it, both routed through one callable the kernel supplies:

``sql(...)``
    Runs against the warehouse where the engine already is. The scan, the join and the
    aggregation happen server-side over the Delta tables; the result crosses into the
    kernel whole, bounded only by the host's own ceiling.

    The bound is on *rendering*, not on the data: what falls over at scale is a browser
    drawing a table, so a result renders its first ``DISPLAY_ROWS`` and says how many it
    has. This is the division Databricks draws with ``display`` and pandas with
    ``display.max_rows`` -- a capped view over a whole object. Capping the data instead
    hands a fraction of a table to somebody who then fits a model to it.

``data['name']``
    The original affordance, kept because it is pleasant for small data and is what
    the notebook docs describe, but now fetched on first access rather than up front,
    and it warns when a fetch is large enough that the caller probably wanted ``sql``.

The split is deliberately visible. A lazy handle whose materialisation is invisible only
moves the memory failure later; making the crossing explicit is what turns a limit into
something a person can reason about.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Protocol

#: Sentinel for "every row the host will part with", which is what both paths ask for.
#: The host applies its own ceiling; this only declines to add a second, smaller one.
_UNLIMITED = 0x7FFFFFFF

#: Rows a result *renders*. The bound belongs here rather than on the data, because the
#: thing that falls over at scale is the browser drawing a table -- which is why
#: Databricks' ``display`` caps rows while the DataFrame behind it stays whole, and why
#: pandas' ``display.max_rows`` truncates a repr and never the object.
DISPLAY_ROWS = 1000

#: Fetching more than this many rows into the kernel as dicts earns a warning: it is the
#: point where ``sql`` with an aggregate is almost always what was wanted. Not a cap:
#: refusing the fetch would break the documented ``data['x']`` contract.
LARGE_FETCH_ROWS = 100_000


class QueryFn(Protocol):
    """Runs a request server-side and returns the reply the kernel host sent back.

    Exactly one of ``sql`` or ``table`` is set. ``table`` exists so the ``data['x']``
    path never builds SQL in the kernel: the host resolves it against the tables it has
    registered, so a dataset name (which is user-controlled) cannot become a query
    fragment. Interpolating names into SQL is the mistake behind CVE-2026-42811 and
    CVE-2026-42810 in Apache Polaris, and it is avoidable here by not doing it.
    """

    def __call__(
        self, *, sql: str = "", table: str = "", limit: int
    ) -> dict[str, Any]: ...


class NotebookQueryError(RuntimeError):
    """A query failed server-side. Carries the engine's message, not a traceback."""


class QueryResult:
    """A query result, and the explicit step that turns it into a frame.

    ``truncated`` says the host's ceiling was reached, so a mean over these rows
    describes a prefix rather than the table. Reaching it is rare: it means the result
    exceeded what the deployment will hand any caller.
    """

    __slots__ = ("_sql", "columns", "rows", "truncated")

    def __init__(
        self,
        *,
        columns: list[str],
        rows: list[dict[str, Any]],
        truncated: bool,
        sql: str = "",
    ) -> None:
        self.columns = columns
        self.rows = rows
        self.truncated = truncated
        self._sql = sql

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def __repr__(self) -> str:
        shape = f"{len(self.rows)} rows x {len(self.columns)} columns"
        return f"<QueryResult {shape}{', truncated' if self.truncated else ''}>"

    def to_pandas(self) -> Any:
        """A DataFrame of the returned rows. Requires pandas in the kernel."""
        import pandas

        return pandas.DataFrame(self.rows, columns=self.columns or None)

    #: What a notebook reflex reaches for; the same thing.
    df = to_pandas

    def to_arrow(self) -> Any:
        """An Arrow table of the returned rows. Requires pyarrow in the kernel."""
        import pyarrow

        columns = self.columns or sorted({k for row in self.rows for k in row})
        return pyarrow.table({c: [row.get(c) for row in self.rows] for c in columns})

    def to_polars(self) -> Any:
        """A Polars DataFrame of the returned rows. Requires polars in the kernel.

        Via Arrow rather than the row dicts, which is the whole point of Polars being an
        Arrow-native frame.
        """
        import polars

        return polars.from_arrow(self.to_arrow())

    def _repr_html_(self) -> str:
        """Render as a table, with truncation stated rather than implied."""
        if not self.rows:
            return "<em>no rows</em>"
        cols = self.columns or sorted({k for row in self.rows for k in row})
        head = "".join(f"<th>{_escape(c)}</th>" for c in cols)
        body = "".join(
            "<tr>" + "".join(f"<td>{_escape(row.get(c))}</td>" for c in cols) + "</tr>"
            for row in self.rows[:DISPLAY_ROWS]
        )
        notes = []
        if len(self.rows) > DISPLAY_ROWS:
            notes.append(f"showing {DISPLAY_ROWS:,} of {len(self.rows):,} rows")
        if self.truncated:
            notes.append(
                "the query matched more than the host will return, so this is a "
                "preview: aggregate in SQL rather than computing over these rows"
            )
        table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        return table + (f"<em>{'; '.join(notes)}</em>" if notes else "")


def _escape(value: object) -> str:
    from html import escape

    return escape("" if value is None else str(value))


def run_query(
    query_fn: QueryFn,
    *,
    sql: str = "",
    table: str = "",
    limit: int = _UNLIMITED,
) -> QueryResult:
    """Execute a query or a whole-table read server-side and wrap the reply.

    Raises :class:`NotebookQueryError` on a query error so a mistyped column reads as a
    normal cell error rather than a protocol failure.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if bool(sql) == bool(table):
        raise ValueError("pass exactly one of sql or table")
    reply = query_fn(sql=sql, table=table, limit=limit)
    error = reply.get("error")
    if error:
        raise NotebookQueryError(str(error))
    rows = list(reply.get("rows") or [])
    return QueryResult(
        columns=list(reply.get("columns") or []),
        rows=rows,
        truncated=bool(reply.get("truncated")),
        sql=sql,
    )


class LazyData(Mapping[str, list[dict[str, Any]]]):
    """``data['name']``: the bound datasets, fetched on first access and then cached.

    Reads as the plain mapping it replaced: ``data['orders']`` is a list of row dicts,
    ``list(data)`` names what is bound, ``len(data)`` counts it. What changed is when
    the rows arrive, and that a large fetch says so.
    """

    def __init__(
        self,
        query_fn: QueryFn,
        names: tuple[str, ...] = (),
        *,
        large_fetch_rows: int = LARGE_FETCH_ROWS,
        on_materialise: Callable[[], None] | None = None,
    ) -> None:
        self._query_fn = query_fn
        self._names = tuple(names)
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._large_fetch_rows = large_fetch_rows
        # Called the first time rows actually cross into this process, so a notebook can
        # say which side of the boundary a cell is on. Only on a real fetch: reading a
        # cached table again is not a second crossing.
        self._on_materialise = on_materialise

    def __getitem__(self, name: str) -> list[dict[str, Any]]:
        if name in self._cache:
            return self._cache[name]
        if self._names and name not in self._names:
            raise KeyError(name)
        # No limit: the documented contract is "the dataset", and silently returning a
        # slice of it would be worse than the memory cost. The warning is the honest
        # middle ground.
        if self._on_materialise is not None:
            self._on_materialise()
        result = run_query(self._query_fn, table=name, limit=_UNLIMITED)
        if len(result.rows) >= self._large_fetch_rows:
            warnings.warn(
                f"data[{name!r}] materialised {len(result.rows):,} rows into the "
                "kernel. Prefer sql(...) with an aggregate so the work runs where the "
                "data is and only the result crosses.",
                stacklevel=2,
            )
        self._cache[name] = result.rows
        return result.rows

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        loaded = sorted(self._cache)
        return (
            f"<data: {len(self._names)} dataset(s) available"
            f"{', loaded: ' + ', '.join(loaded) if loaded else ''}>"
        )
