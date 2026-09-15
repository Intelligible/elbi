"""Storing a list of structs as JSON text, for shapes Delta cannot evolve.

A list of structs is where per-page schema inference stops being survivable. Delta
merges a struct that gains a field and a list that gains one, but a list whose element
struct diverges between pages *while also* gaining a list-typed field cannot be cast.
Stripe's `invoices` does exactly that, so the column is stored as text instead --
opt-in, because it trades column-level access for a sync that cannot break on shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("deltalake")
pytest.importorskip("pyarrow")

import pyarrow as pa

from elbi.warehouse import storage

#: Two invoices whose line items genuinely differ: one line came from a subscription,
#: the other from an invoice item, and only the second carries `taxes`. This is the
#: shape that fails in production, reduced to its two necessary conditions.
_PAGE_1 = [
    {
        "id": "in_1",
        "total": 100,
        "lines": {
            "data": [
                {
                    "id": "il_1",
                    "parent": {
                        "subscription_item_details": {"subscription": "sub_1"},
                        "type": "subscription_item_details",
                    },
                }
            ]
        },
    }
]
_PAGE_2 = [
    {
        "id": "in_2",
        "total": 250,
        "lines": {
            "data": [
                {
                    "id": "il_2",
                    "parent": {
                        "invoice_item_details": {"invoice_item": "ii_1"},
                        "type": "invoice_item_details",
                    },
                    "taxes": [{"amount": 10, "tax_behavior": "exclusive"}],
                }
            ]
        },
    }
]


@pytest.fixture
def warehouse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")


@pytest.fixture
def json_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAREHOUSE_JSON_NESTED_LISTS", "1")


def _written(table: str) -> pa.Table:
    from deltalake import DeltaTable

    return DeltaTable(storage.table_uri(table)).to_pyarrow_table()


def test_without_the_flag_the_divergent_shape_still_fails(warehouse: None) -> None:
    # The failure this exists to fix, so a regression here is visible rather than
    # silently "fixed" by something else.
    storage.write_arrow("invoices", pa.Table.from_pylist(_PAGE_1), mode="overwrite")

    with pytest.raises(Exception, match="Unsupported CAST"):
        storage.write_arrow("invoices", pa.Table.from_pylist(_PAGE_2), mode="append")


def test_with_the_flag_both_pages_sync(warehouse: None, json_lists: None) -> None:
    storage.write_arrow("invoices", pa.Table.from_pylist(_PAGE_1), mode="overwrite")
    storage.write_arrow("invoices", pa.Table.from_pylist(_PAGE_2), mode="append")

    assert _written("invoices").num_rows == 2


def test_the_encoded_column_keeps_every_field(
    warehouse: None, json_lists: None
) -> None:
    # Encoding must be lossless: the point is to store the shape faithfully as text,
    # not to discard the parts that would not fit a column.
    storage.write_arrow("invoices", pa.Table.from_pylist(_PAGE_1), mode="overwrite")
    storage.write_arrow("invoices", pa.Table.from_pylist(_PAGE_2), mode="append")

    rows = {
        row["id"]: json.loads(row["lines"]) for row in _written("invoices").to_pylist()
    }

    assert rows["in_1"]["data"][0]["parent"]["type"] == "subscription_item_details"
    assert rows["in_2"]["data"][0]["parent"]["type"] == "invoice_item_details"
    assert rows["in_2"]["data"][0]["taxes"] == [
        {"amount": 10, "tax_behavior": "exclusive"}
    ]


def test_only_columns_holding_a_struct_list_are_encoded(
    warehouse: None, json_lists: None
) -> None:
    storage.write_arrow("invoices", pa.Table.from_pylist(_PAGE_1), mode="overwrite")

    schema = _written("invoices").schema
    assert schema.field("lines").type == pa.string()
    # Scalars and plain structs are untouched -- this is not "JSON-ify everything".
    assert schema.field("total").type == pa.int64()


def test_a_plain_struct_column_is_left_typed(warehouse: None, json_lists: None) -> None:
    page = [{"id": "cus_1", "address": {"city": "Madison", "country": "US"}}]

    storage.write_arrow("customers", pa.Table.from_pylist(page), mode="overwrite")

    assert pa.types.is_struct(_written("customers").schema.field("address").type)


def test_a_list_of_scalars_is_left_typed(warehouse: None, json_lists: None) -> None:
    page = [{"id": "cus_1", "preferred_locales": ["en-GB", "en-US"]}]

    storage.write_arrow("customers", pa.Table.from_pylist(page), mode="overwrite")

    assert _written("customers").schema.field("preferred_locales").type == pa.list_(
        pa.string()
    )


def test_the_flag_is_off_by_default(warehouse: None) -> None:
    page = [{"id": "in_1", "lines": {"data": [{"id": "il_1", "amount": 1}]}}]

    storage.write_arrow("invoices", pa.Table.from_pylist(page), mode="overwrite")

    assert pa.types.is_struct(_written("invoices").schema.field("lines").type)


def test_a_null_inside_an_encoded_column_survives_as_json_null(
    warehouse: None, json_lists: None
) -> None:
    # Encoding runs before the null-dropping pass, because JSON represents a null
    # perfectly well and dropping first would lose the key from the text.
    page = [{"id": "in_1", "lines": {"data": [{"id": "il_1", "description": None}]}}]

    storage.write_arrow("invoices", pa.Table.from_pylist(page), mode="overwrite")

    line = json.loads(_written("invoices").to_pylist()[0]["lines"])["data"][0]
    assert "description" in line
    assert line["description"] is None
