"""Fields an API page carries nothing in, and what the warehouse does with them.

A connector infers each batch's schema from the JSON it just received, so the schema
describes that page rather than the resource. A field the page had nothing in infers as
Arrow ``null`` (None in every row) or as an empty struct (``{}``), and Delta can store
neither. Both are dropped: the page says nothing about the field's real type, and
guessing one poisons the table for the page that does know.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("deltalake")
pytest.importorskip("pyarrow")

import pyarrow as pa

from elbi.warehouse import storage


@pytest.fixture(autouse=True)
def warehouse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")


def _schema(table: str) -> pa.Schema:
    from deltalake import DeltaTable

    return DeltaTable(storage.table_uri(table)).to_pyarrow_table().schema


def _rows(table: str) -> pa.Table:
    from deltalake import DeltaTable

    return DeltaTable(storage.table_uri(table)).to_pyarrow_table()


def test_a_column_null_for_the_whole_page_no_longer_fails_the_write() -> None:
    page = pa.Table.from_pylist(
        [{"id": "cus_1", "description": None}, {"id": "cus_2", "description": None}]
    )
    assert pa.types.is_null(page.schema.field("description").type), (
        "precondition: from_pylist infers the null type here"
    )

    storage.write_arrow("customers", page, mode="overwrite")

    assert _schema("customers").names == ["id"]
    assert _rows("customers").num_rows == 2


def test_an_empty_struct_is_dropped_because_parquet_cannot_write_one() -> None:
    # Stripe sends `metadata: {}` on any object carrying none.
    page = pa.Table.from_pylist([{"id": "cus_1", "metadata": {}}])

    storage.write_arrow("customers", page, mode="overwrite")

    assert "metadata" not in _schema("customers").names


def test_nulls_nested_in_a_struct_are_dropped_and_the_parent_survives() -> None:
    # A nested null is rejected exactly as a top-level one; the error only gains an
    # "External error:" per level of nesting, which is how this was first missed.
    page = pa.Table.from_pylist(
        [{"id": "in_1", "automatic_tax": {"enabled": False, "liability": None}}]
    )

    storage.write_arrow("invoices", page, mode="overwrite")

    assert _schema("invoices").field("automatic_tax").type == pa.struct(
        [pa.field("enabled", pa.bool_())]
    )


def test_a_struct_emptied_by_dropping_is_itself_dropped() -> None:
    page = pa.Table.from_pylist([{"id": "in_1", "tax": {"liability": None}}])

    storage.write_arrow("invoices", page, mode="overwrite")

    assert "tax" not in _schema("invoices").names


def test_a_later_page_supplies_the_real_type_and_earlier_rows_read_as_null() -> None:
    # The invoices failure in production: page 1 had automatic_tax.liability = null,
    # page 2 had it as a struct. sync.py writes page 1 overwrite, the rest append+merge.
    # Casting the null to string made page 2 unmergeable ("Unsupported CAST from
    # Struct(...) to Struct(...)"); dropping lets the real type arrive intact.
    page_1 = pa.Table.from_pylist(
        [{"id": "in_1", "automatic_tax": {"enabled": False, "liability": None}}]
    )
    page_2 = pa.Table.from_pylist(
        [
            {
                "id": "in_2",
                "automatic_tax": {"enabled": True, "liability": {"type": "self"}},
            }
        ]
    )

    storage.write_arrow("invoices", page_1, mode="overwrite")
    storage.write_arrow("invoices", page_2, mode="append")

    liability = _schema("invoices").field("automatic_tax").type.field("liability")
    assert liability.type == pa.struct([pa.field("type", pa.string())])
    values = {
        row["id"]: row["automatic_tax"]["liability"]
        for row in _rows("invoices").to_pylist()
    }
    assert values == {"in_1": None, "in_2": {"type": "self"}}


def test_the_reverse_order_works_too() -> None:
    # A real value first and the empty page second must merge just as cleanly, since
    # page order depends on the account's data, not on anything we control.
    page_1 = pa.Table.from_pylist([{"id": "in_1", "liability": {"type": "self"}}])
    page_2 = pa.Table.from_pylist([{"id": "in_2", "liability": None}])

    storage.write_arrow("invoices", page_1, mode="overwrite")
    storage.write_arrow("invoices", page_2, mode="append")

    assert _schema("invoices").field("liability").type == pa.struct(
        [pa.field("type", pa.string())]
    )


def test_a_list_of_nulls_is_dropped() -> None:
    page = pa.Table.from_pylist([{"id": "cus_1", "preferred_locales": []}])

    storage.write_arrow("customers", page, mode="overwrite")

    assert "preferred_locales" not in _schema("customers").names


def test_a_page_with_nothing_unwritable_is_passed_through_untouched() -> None:
    page = pa.Table.from_pylist(
        [{"id": "cus_1", "description": "present", "balance": 0}]
    )

    storage.write_arrow("customers", page, mode="overwrite")

    assert _schema("customers").names == ["id", "description", "balance"]
    assert _rows("customers").column("description").to_pylist() == ["present"]


def test_dropping_a_field_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    # A column that disappears with no trace becomes someone's bug report later.
    page = pa.Table.from_pylist([{"id": "cus_1", "description": None, "metadata": {}}])

    with caplog.at_level("INFO", logger="elbi.warehouse.storage"):
        storage.write_arrow("customers", page, mode="overwrite")

    assert "description" in caplog.text
    assert "metadata" in caplog.text
