"""Tests for ad-hoc SQL over bound datasets (:mod:`elbi.query`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import QueryResult, query_datasets
from elbi_core.errors import DataBindingError

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")


SALES = [
    {"region": "west", "amount": 10},
    {"region": "west", "amount": 30},
    {"region": "east", "amount": 20},
]


def test_query_aggregates_rows() -> None:
    result = query_datasets(
        {"sales": SALES},
        "SELECT region, sum(amount) AS total FROM sales "
        "GROUP BY region ORDER BY region",
    )
    assert isinstance(result, QueryResult)
    assert result.columns == ["region", "total"]
    assert result.rows == [
        {"region": "east", "total": 20},
        {"region": "west", "total": 40},
    ]
    assert result.truncated is False


def test_query_joins_across_datasets() -> None:
    regions = [
        {"region": "west", "manager": "ana"},
        {"region": "east", "manager": "bo"},
    ]
    result = query_datasets(
        {"sales": SALES, "regions": regions},
        "SELECT r.manager, sum(s.amount) AS total FROM sales s "
        "JOIN regions r ON r.region = s.region GROUP BY r.manager ORDER BY r.manager",
    )
    assert result.rows == [
        {"manager": "ana", "total": 40},
        {"manager": "bo", "total": 20},
    ]


def test_row_cap_sets_truncated() -> None:
    rows = [{"n": i} for i in range(10)]
    result = query_datasets({"nums": rows}, "SELECT n FROM nums ORDER BY n", max_rows=3)
    assert result.truncated is True
    assert result.rows == [{"n": 0}, {"n": 1}, {"n": 2}]


def test_exact_cap_is_not_truncated() -> None:
    rows = [{"n": i} for i in range(3)]
    result = query_datasets({"nums": rows}, "SELECT n FROM nums", max_rows=3)
    assert result.truncated is False
    assert len(result.rows) == 3


def test_datetimes_and_decimals_are_json_safe() -> None:
    result = query_datasets(
        {"sales": SALES},
        "SELECT DATE '2026-07-06' AS d, CAST(1.5 AS DECIMAL(4, 2)) AS x LIMIT 1",
    )
    row = result.rows[0]
    assert row["d"] == "2026-07-06"
    assert row["x"] == "1.50"


def test_external_access_is_disabled_for_in_memory_sources(tmp_path: Path) -> None:
    """A query cannot reach the filesystem when every source is in memory."""
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(DataBindingError, match="query failed"):
        query_datasets(
            {"sales": SALES},
            f"SELECT * FROM read_csv_auto('{secret}')",
        )


def test_can_scan_a_file_source(tmp_path: Path) -> None:
    data = tmp_path / "nums.csv"
    data.write_text("n\n1\n2\n3\n", encoding="utf-8")
    result = query_datasets({"nums": data}, "SELECT sum(n) AS total FROM nums")
    assert result.rows == [{"total": 6}]


def test_invalid_dataset_name_is_rejected() -> None:
    with pytest.raises(DataBindingError, match="invalid dataset name"):
        query_datasets({"bad name": SALES}, "SELECT 1")


def test_bad_sql_raises_binding_error() -> None:
    with pytest.raises(DataBindingError, match="query failed"):
        query_datasets({"sales": SALES}, "SELECT * FROM nonexistent_table")


def test_non_row_source_is_rejected() -> None:
    with pytest.raises(DataBindingError, match="must be a list of row dicts"):
        query_datasets({"sales": "not rows"}, "SELECT 1")
