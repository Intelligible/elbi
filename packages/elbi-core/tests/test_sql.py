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


def test_bad_table_name_rejected(db: str) -> None:
    with pytest.raises(DataBindingError, match="invalid table name"):
        load_sql(db, table="sales; DROP TABLE sales")


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

    def execution_options(self, **kwargs: Any) -> _FakeConn:
        self.options = kwargs
        return self


def test_read_only_applies_for_postgres() -> None:
    conn = _FakeConn()
    returned = _read_only(conn, "postgresql")
    assert returned is conn
    assert conn.options == {"postgresql_readonly": True}


def test_read_only_noop_for_other_engines() -> None:
    conn = _FakeConn()
    assert _read_only(conn, "sqlite") is conn
    assert conn.options is None


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
