"""Tests for tabular data loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import DataBindingError
from elbi_core.data import Table, load_table


def test_table_columns_first_seen_order() -> None:
    table = Table(rows=[{"a": 1, "b": 2}, {"b": 3, "c": 4}])
    assert table.columns == ["a", "b", "c"]
    assert len(table) == 2
    assert list(table) == table.rows


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_text("id,risk\na,0.9\nb,0.1\n", encoding="utf-8")
    table = load_table(path)
    assert table.rows == [{"id": "a", "risk": "0.9"}, {"id": "b", "risk": "0.1"}]


def test_load_json_array(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
    assert load_table(path).rows == [{"a": 1}, {"a": 2}]


def test_load_json_object(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert load_table(path).rows == [{"a": 1}]


def test_load_json_scalar_rejected(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text("42", encoding="utf-8")
    with pytest.raises(DataBindingError):
        load_table(path)


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert load_table(path).rows == [{"a": 1}, {"a": 2}]


def test_load_jsonl_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DataBindingError, match="invalid JSON line"):
        load_table(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataBindingError, match="not found"):
        load_table(tmp_path / "nope.csv")


def test_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "d.xlsx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(DataBindingError, match="unsupported"):
        load_table(path)


def test_load_json_array_with_non_object_element(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text('[{"a": 1}, 5]', encoding="utf-8")
    with pytest.raises(DataBindingError, match="expected an object"):
        load_table(path)


def test_load_parquet(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "d.parquet"
    table = pa.table({"id": ["a", "b"], "amount": [100, 5]})
    pq.write_table(table, path)

    loaded = load_table(path)
    assert loaded.rows == [
        {"id": "a", "amount": 100},
        {"id": "b", "amount": 5},
    ]


def test_load_parquet_without_the_data_extra_raises_a_clear_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    path = tmp_path / "data.parquet"
    path.write_bytes(b"")  # the import fails before the file is read; content is moot
    # None entries make `import pyarrow.parquet` raise, simulating the extra absent.
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(DataBindingError, match=r"elbi\[data\]"):
        load_table(path)
