"""Resolve a registered data source to a connection URL, and test it.

The runtime seam is unchanged: this produces a connection URL string that flows into
``elbi.sql.load_sql`` (SELECT-only, row-capped, ``NullPool`` + dispose).
A per-source engine is built on demand, tested with a ``SELECT 1``, and a masked view
never leaks the password.
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine, pool, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from . import crypto
from .db import DataSource
from .wire import DataSource as WireDataSource

#: A connection ``kind`` mapped to a SQLAlchemy driver (psycopg v3 for Postgres, to
#: match the app store). Drivers beyond sqlite/postgres need their extra installed.
_DRIVERS = {
    "postgres": "postgresql+psycopg",
    "postgresql": "postgresql+psycopg",
    "mysql": "mysql+pymysql",
    "sqlite": "sqlite",
    "mssql": "mssql+pyodbc",
}

#: Shown in place of a stored password so it never leaves the server. An update that
#: sends the mask back means "keep the stored secret".
PASSWORD_MASK = "•" * 8


def _password(source: DataSource) -> str | None:
    if source.secret_env:
        return os.environ.get(source.secret_env)
    if source.secret:
        return crypto.decrypt(source.secret)
    return None


def connection_url(source: DataSource) -> str:
    """The runtime connection URL for a source, secret resolved.

    Raises ``ValueError`` for an unsupported ``kind``. For SQLite the ``database`` is a
    file path.
    """
    driver = _DRIVERS.get(source.kind.lower())
    if driver is None:
        raise ValueError(f"unsupported data source kind: {source.kind!r}")
    if driver == "sqlite":
        return f"sqlite:///{source.database or ':memory:'}"
    return URL.create(
        driver,
        username=source.username,
        password=_password(source),
        host=source.host,
        port=source.port,
        database=source.database,
    ).render_as_string(hide_password=False)


def _without_secret(source: DataSource, message: str) -> str:
    """``message`` with the source's password masked out."""
    secret = _password(source)
    return message.replace(secret, "***") if secret else message


def test_connection(source: DataSource) -> dict[str, Any]:
    """Open a throwaway connection and run ``SELECT 1`` to validate the config.

    The driver's own message is reported so a failure names the host and cause, with
    the password masked: a dialect is free to quote the URL it was handed.
    """
    try:
        engine = create_engine(connection_url(source), poolclass=pool.NullPool)
    except Exception as exc:  # bad URL / missing driver / unsupported kind
        return {"ok": False, "error": _without_secret(source, str(exc))}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except SQLAlchemyError as exc:
        return {"ok": False, "error": _without_secret(source, str(exc))}
    finally:
        engine.dispose()


def load_rows(
    source: DataSource,
    *,
    query: str | None = None,
    table: str | None = None,
    max_rows: int = 1000,
) -> list[dict[str, Any]]:
    """Read rows from a source via the runtime's own ``load_sql`` (read-only, capped).

    This is the seam to the analysis runtime: a registered source becomes rows the same
    way a project dataset does (URL in, ``list[dict]`` out) with no runtime changes.
    """
    from elbi_core.sql import load_sql

    return load_sql(
        connection_url(source), query=query, table=table, max_rows=max_rows
    ).rows


def catalog(source: DataSource, *, max_tables: int = 500) -> dict[str, Any]:
    """List a source's tables and views with their columns, via SQLAlchemy reflection.

    Reflection is dialect-agnostic (no hand-rolled ``information_schema``), so one path
    serves every engine. Feeds the exploration surface's schema browser.

    Raises:
        SQLAlchemyError: if the source cannot be reflected (bad config, no access).
    """
    from sqlalchemy import create_engine, inspect, pool

    engine = create_engine(connection_url(source), poolclass=pool.NullPool)
    try:
        inspector = inspect(engine)
        names = [*inspector.get_table_names(), *inspector.get_view_names()]
        tables: list[dict[str, Any]] = []
        for name in sorted(names)[:max_tables]:
            columns = [
                {"name": column["name"], "type": str(column["type"])}
                for column in inspector.get_columns(name)
            ]
            tables.append({"name": name, "columns": columns})
        return {"tables": tables}
    finally:
        engine.dispose()


def view(source: DataSource) -> WireDataSource:
    """The wire form of a source: its config with the secret masked, never returned."""
    return WireDataSource(
        id=source.id,
        name=source.name,
        kind=source.kind,
        host=source.host,
        port=source.port,
        database=source.database,
        username=source.username,
        secret_set=bool(source.secret) or bool(source.secret_env),
        secret_env=source.secret_env,
        extra=source.extra,
    )
