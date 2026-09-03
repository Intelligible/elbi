"""Load a dataset from a SQL database into a :class:`~elbi.data.Table`.

SQL support is optional (``pip install "elbi[sql]"`` for the core, plus a
driver extra such as ``[postgres]``, ``[mysql]``, ``[mssql]``). It uses
SQLAlchemy Core (no ORM), so one code path serves every engine via a connection
URL.

The posture follows the industry standard for read-only analytics access:

* **Governance lives at the source.** Connect with a SELECT-only database role;
  the framework does not re-implement access control.
* **Bounded results.** A row cap protects the warehouse and the agent's context.
* **Short-lived connections.** ``NullPool``: the runner is not a long-running
  pooled service.
* **Statement timeouts** are set per dialect on a best-effort basis.
* **JSON-safe rows.** ``datetime``/``date`` become ISO strings; ``Decimal``
  becomes a string (lossless).
"""

from __future__ import annotations

import datetime as _dt
import re
from decimal import Decimal
from typing import Any

from .data import Row, Table
from .errors import DataBindingError

#: Default maximum rows pulled from a SQL source.
DEFAULT_MAX_ROWS = 10_000
#: Default statement timeout in seconds.
DEFAULT_TIMEOUT = 30

# Table identifiers cannot be passed as bound parameters, so a ``table:`` binding
# is validated against this allow-list (schema-qualified names permitted).
_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def load_sql(
    connection_url: str,
    *,
    query: str | None = None,
    table: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout: int = DEFAULT_TIMEOUT,
    params: dict[str, Any] | None = None,
) -> Table:
    """Execute a read-only query against ``connection_url`` and return a Table.

    Exactly one of ``query`` or ``table`` must be given. ``params`` are passed as
    bound parameters (safe for values; never use them for identifiers).

    Raises:
        DataBindingError: if SQLAlchemy is not installed, the binding is
            malformed, or the query fails.
    """
    if (query is None) == (table is None):
        raise DataBindingError("a SQL binding needs exactly one of 'query' or 'table'")

    sqlalchemy = _import_sqlalchemy()
    statement = _statement(sqlalchemy, query, table)

    url = _safe_url(sqlalchemy, connection_url)
    try:
        engine = sqlalchemy.create_engine(
            url,
            poolclass=sqlalchemy.pool.NullPool,
            connect_args=_connect_args(url.get_backend_name(), timeout),
        )
    except Exception as exc:
        raise DataBindingError(f"SQL connection failed: {exc}") from exc

    try:
        with engine.connect() as conn:
            conn = _read_only(conn, url.get_backend_name())
            result = conn.execute(statement, params or {})
            rows: list[Row] = [
                _json_safe(dict(m)) for m in result.mappings().fetchmany(max_rows)
            ]
    except Exception as exc:
        raise DataBindingError(f"SQL query failed: {exc}") from exc
    finally:
        engine.dispose()
    return Table(rows=rows)


def _import_sqlalchemy() -> Any:
    try:
        import sqlalchemy
        import sqlalchemy.pool
    except ImportError as exc:
        raise DataBindingError(
            "SQL bindings require the 'sql' extra: pip install \"elbi[sql]\" "
            "(plus a driver extra such as [postgres], [mysql], or [mssql])"
        ) from exc
    return sqlalchemy


def _statement(sqlalchemy: Any, query: str | None, table: str | None) -> Any:
    if query is not None:
        return sqlalchemy.text(query)
    # table is guaranteed non-None by load_sql's xor check; validate the
    # identifier against an allow-list (identifiers cannot be bound parameters).
    if table is None or not _TABLE_NAME.match(table):
        raise DataBindingError(
            f"invalid table name {table!r}; expected an identifier like 'schema.table'"
        )
    return sqlalchemy.text(f"SELECT * FROM {table}")  # noqa: S608 - allow-listed


def _safe_url(sqlalchemy: Any, connection_url: str) -> Any:
    try:
        return sqlalchemy.make_url(connection_url)
    except Exception as exc:
        # Never echo the raw URL; it may contain a password.
        raise DataBindingError("invalid SQL connection URL") from exc


def _read_only(conn: Any, backend: str) -> Any:
    """Apply a read-only transaction where the dialect supports it."""
    if backend == "postgresql":
        return conn.execution_options(postgresql_readonly=True)
    return conn


def _connect_args(backend: str, timeout: int) -> dict[str, Any]:
    """Best-effort per-dialect statement/connect timeout (driver kwargs differ)."""
    if backend == "postgresql":
        return {"options": f"-c statement_timeout={timeout * 1000}"}
    if backend in {"mysql", "mariadb"}:
        return {"connect_timeout": timeout}  # pymysql
    if backend in {"mssql", "sqlite"}:
        return {"timeout": timeout}  # pyodbc / sqlite3
    return {}


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
