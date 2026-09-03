"""Tests for the SQL dataset loader (exercised against SQLite)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from elbi_core import DataBindingError
from elbi_core.sql import _connect_args, _json_safe_value, _read_only, load_sql

sa = pytest.importorskip("sqlalchemy")


@pytest.fixture
def db(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 'sales.db'}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE sales (id INTEGER, region TEXT, amount REAL)")
        )
        conn.execute(
            sa.text("INSERT INTO sales VALUES (1, 'west', 100.0), (2, 'east', 5.0)")
        )
    engine.dispose()
    return url


def test_load_via_query(db: str) -> None:
    table = load_sql(db, query="SELECT id, region FROM sales ORDER BY id")
    assert table.rows == [
        {"id": 1, "region": "west"},
        {"id": 2, "region": "east"},
    ]


def test_load_via_table(db: str) -> None:
    table = load_sql(db, table="sales")
    assert len(table.rows) == 2
    assert {"id", "region", "amount"} <= table.rows[0].keys()


def test_schema_qualified_table_name_allowed(db: str) -> None:
    # SQLite's default schema is 'main'; a dotted name must pass validation.
    table = load_sql(db, table="main.sales")
    assert len(table.rows) == 2


def test_row_cap_limits_results(db: str) -> None:
    assert len(load_sql(db, table="sales", max_rows=1).rows) == 1


def test_bound_params_are_safe(db: str) -> None:
    table = load_sql(
        db, query="SELECT id FROM sales WHERE region = :r", params={"r": "west"}
    )
    assert [r["id"] for r in table.rows] == [1]


def test_query_xor_table_required(db: str) -> None:
    with pytest.raises(DataBindingError, match="exactly one"):
        load_sql(db)
    with pytest.raises(DataBindingError, match="exactly one"):
        load_sql(db, query="SELECT 1", table="sales")


def test_a_table_name_cannot_carry_sql(db: str) -> None:
    """The name is quoted as an identifier, so the appended statement never runs."""
    with pytest.raises(DataBindingError, match="SQL query failed"):
        load_sql(db, table="sales; DROP TABLE sales")
    assert len(load_sql(db, table="sales").rows) == 2


def test_empty_and_over_qualified_table_names_rejected(db: str) -> None:
    for name in ("", ".", "a.b.c.d"):
        with pytest.raises(DataBindingError, match="invalid table name"):
            load_sql(db, table=name)


def test_a_reserved_word_table_name_resolves(tmp_path: Path) -> None:
    """``order`` is a keyword; unquoted it is a syntax error rather than a table."""
    url = f"sqlite:///{tmp_path / 'kw.db'}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text('CREATE TABLE "order" (id INTEGER)'))
        conn.execute(sa.text('INSERT INTO "order" VALUES (1)'))
        conn.execute(sa.text('CREATE TABLE "MixedCase" (id INTEGER)'))
        conn.execute(sa.text('INSERT INTO "MixedCase" VALUES (2)'))
    engine.dispose()
    assert load_sql(url, table="order").rows == [{"id": 1}]
    assert load_sql(url, table="MixedCase").rows == [{"id": 2}]


def test_invalid_connection_url() -> None:
    with pytest.raises(DataBindingError, match="invalid SQL connection URL"):
        load_sql("not-a-url", table="sales")


def test_unknown_dialect_is_wrapped() -> None:
    with pytest.raises(DataBindingError, match=r"SQL (connection|query) failed"):
        load_sql("nodialect://host/db", table="sales")


def test_query_error_is_wrapped(db: str) -> None:
    with pytest.raises(DataBindingError, match="SQL query failed"):
        load_sql(db, query="SELECT * FROM does_not_exist")


@pytest.mark.parametrize(
    ("backend", "expected_key"),
    [
        ("postgresql", "options"),
        ("mysql", "connect_timeout"),
        ("mariadb", "connect_timeout"),
        ("mssql", "timeout"),
        ("sqlite", "timeout"),
    ],
)
def test_connect_args_per_dialect(backend: str, expected_key: str) -> None:
    assert expected_key in _connect_args(backend, timeout=15)


def test_connect_args_unknown_dialect_is_empty() -> None:
    assert _connect_args("snowflake", timeout=15) == {}


def test_postgres_statement_timeout_is_in_milliseconds() -> None:
    assert _connect_args("postgresql", timeout=15)["options"] == (
        "-c statement_timeout=15000"
    )


class _FakeConn:
    def __init__(self) -> None:
        self.options: dict[str, Any] | None = None
        self.statements: list[str] = []

    def execution_options(self, **kwargs: Any) -> _FakeConn:
        self.options = kwargs
        return self

    def exec_driver_sql(self, statement: str) -> None:
        self.statements.append(statement)


def test_read_only_applies_for_postgres() -> None:
    conn = _FakeConn()
    returned = _read_only(conn, "postgresql")
    assert returned is conn
    assert conn.options == {"postgresql_readonly": True}


def test_read_only_noop_for_dialects_without_one() -> None:
    conn = _FakeConn()
    assert _read_only(conn, "snowflake") is conn
    assert conn.options is None
    assert conn.statements == []


def test_read_only_sets_query_only_for_sqlite() -> None:
    conn = _FakeConn()
    assert _read_only(conn, "sqlite") is conn
    assert conn.statements == ["PRAGMA query_only = ON"]


def test_mysql_connects_read_only() -> None:
    """Set on connect: a transaction already open would stay read-write."""
    args = _connect_args("mysql", timeout=15)
    assert args["init_command"] == "SET SESSION TRANSACTION READ ONLY"


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE sales",
        "CREATE TABLE evil (x INTEGER)",
        "INSERT INTO sales VALUES (3, 'north', 1.0)",
        "UPDATE sales SET amount = 0",
        "DELETE FROM sales",
    ],
)
def test_a_write_never_reaches_the_database(db: str, statement: str) -> None:
    """Row-returning is checked after execution, so a write has to be refused first.

    SQLite commits DDL outside the transaction, which is how a dropped table used to
    survive the error the caller saw.
    """
    with pytest.raises(DataBindingError, match="SQL query failed"):
        load_sql(db, query=statement)

    engine = sa.create_engine(db)
    with engine.connect() as conn:
        names = {
            row[0]
            for row in conn.execute(
                sa.text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        }
        rows = conn.execute(sa.text("SELECT id, amount FROM sales")).fetchall()
    engine.dispose()
    assert names == {"sales"}
    assert rows == [(1, 100.0), (2, 5.0)]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime.date(2024, 1, 2), "2024-01-02"),
        (datetime.datetime(2024, 1, 2, 3, 4, 5), "2024-01-02T03:04:05"),
        (datetime.time(3, 4, 5), "03:04:05"),
        (Decimal("1.50"), "1.50"),
        (b"hi", "hi"),
        (42, 42),
        ("x", "x"),
        (None, None),
    ],
)
def test_json_safe_value(value: Any, expected: Any) -> None:
    assert _json_safe_value(value) == expected


def test_import_sqlalchemy_without_the_sql_extra_raises_a_clear_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from elbi_core.sql import _import_sqlalchemy

    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    with pytest.raises(DataBindingError, match=r"elbi\[sql\]"):
        _import_sqlalchemy()
