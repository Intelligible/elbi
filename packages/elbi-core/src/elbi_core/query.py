"""Run read-only SQL over bound datasets in-process, via DuckDB.

The local-data companion to :mod:`elbi.sql` (which reads from an external
database): this runs one ad-hoc query over data already bound in a project. Each
named dataset -- a ``list[dict]``, an Arrow table, or a :class:`~pathlib.Path` to a
Parquet/CSV/JSON file scanned in place -- is registered as a DuckDB view, and the
query sees exactly those views.

It backs the exploration surface's SQL workbench, where a person queries bound data
directly. It is deliberately outside the certified-derivation path: a human exploring
is trusted and never gated; promoting a query to a certified derivation is a separate,
opt-in step.

DuckDB is an optional extra (``pip install "elbi[duckdb]"``). When every source
is in memory, external access is disabled after registration, so a query cannot read or
write arbitrary files or URLs -- it reaches the registered datasets and nothing else. A
filesystem source keeps external access on, since its view rescans the file on demand.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .compute.arrow import is_arrow_source, rows_to_arrow
from .data import Row
from .errors import DataBindingError

#: Default cap on rows returned to the caller (protects memory and the UI).
DEFAULT_MAX_ROWS = 10_000

#: A dataset name that is a plain identifier. A view name cannot be a bound parameter,
#: so validating it keeps registration to safe names; the query itself never
#: string-interpolates a name (DuckDB's reader API quotes it).
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class QueryResult:
    """The outcome of an ad-hoc query: its columns and bounded, JSON-safe rows.

    ``truncated`` is true when the query produced more than ``max_rows`` rows and the
    result was cut to the cap, so the caller can tell a full result from a clipped one.
    """

    columns: list[str]
    rows: list[Row]
    truncated: bool


def query_datasets(
    sources: Mapping[str, Any],
    sql: str,
    *,
    params: Sequence[Any] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> QueryResult:
    """Run read-only ``sql`` over ``sources`` and return bounded, JSON-safe rows.

    ``sources`` maps a dataset name (a plain identifier) to its data: a ``list[dict]``,
    an Arrow table, or a :class:`~pathlib.Path` to a Parquet/CSV/JSON file. Each becomes
    a DuckDB view of that name, queryable by the SQL. ``params`` are bound to ``?``
    placeholders in ``sql`` (values only; never identifiers), the injection-safe way to
    embed filter values.

    Raises:
        DataBindingError: if DuckDB is not installed, a source name is not a valid
            identifier, a file cannot be scanned, or the query fails.
    """
    duckdb = _require_duckdb()
    con = duckdb.connect()
    try:
        scans_files = False
        for name, data in sources.items():
            scans_files |= _register(con, name, data)
        # With only in-memory sources, lock the query to the registered views: no
        # filesystem or URL access. A file-backed view must rescan its file, so it
        # cannot be locked down this way.
        if not scans_files:
            con.execute("SET enable_external_access=false")
        try:
            result = con.execute(sql, list(params)) if params else con.execute(sql)
        except Exception as exc:  # a user SQL error, surfaced for the workbench
            raise DataBindingError(f"query failed: {exc}") from exc
        columns = [descriptor[0] for descriptor in result.description or []]
        fetched = result.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [
            _json_safe(dict(zip(columns, values, strict=True)))
            for values in fetched[:max_rows]
        ]
        return QueryResult(columns=columns, rows=rows, truncated=truncated)
    finally:
        con.close()


def _register(con: Any, name: str, data: Any) -> bool:
    """Register ``data`` as DuckDB view ``name``; return whether it scans a file."""
    if not _NAME.match(name):
        raise DataBindingError(
            f"invalid dataset name {name!r}; expected a plain identifier "
            "(letters, digits, underscore; not starting with a digit)"
        )
    if isinstance(data, Path):
        _register_file(con, name, data)
        return True
    table = data if is_arrow_source(data) else rows_to_arrow(_as_rows(name, data))
    con.register(name, table)
    return False


def _register_file(con: Any, name: str, path: Path) -> None:
    """Register a data file as a view via DuckDB's reader API.

    The path and view name pass through DuckDB's typed reader calls rather than an
    interpolated SQL string, so neither can be a query-injection vector.
    """
    readers = {
        ".parquet": con.read_parquet,
        ".csv": con.read_csv,
        ".json": con.read_json,
        ".ndjson": con.read_json,
        ".jsonl": con.read_json,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise DataBindingError(f"cannot scan {path.suffix!r} as a dataset: {path}")
    reader(str(path)).create_view(name, replace=True)


def _as_rows(name: str, data: Any) -> list[Row]:
    """Coerce a source to ``list[dict]`` rows, or reject it with a clear error."""
    if isinstance(data, (list, tuple)) and all(isinstance(row, dict) for row in data):
        return [dict(row) for row in data]
    raise DataBindingError(
        f"dataset {name!r} must be a list of row dicts, an Arrow table, or a file path"
    )


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise DataBindingError(
            'querying bound datasets needs the duckdb extra: pip install "elbi[duckdb]"'
        ) from exc
    return duckdb


def _json_safe(row: Row) -> Row:
    return {key: _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value
