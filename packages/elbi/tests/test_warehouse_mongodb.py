"""The MongoDB connector: collections as tables, over a real server.

Almost everything here runs against MongoDB in Docker, because the three problems this
connector solves are all things a fake cannot pose. Schema inference is an aggregation
the server evaluates. The index rule depends on what the server reports as an index. And
the type handling only matters for documents a real driver decoded into real BSON.

The tests skip when no server is reachable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

pytest.importorskip("pyarrow")
pymongo = pytest.importorskip("pymongo")

from elbi.warehouse.sources.base import SourceInputs  # noqa: E402
from elbi.warehouse.sources.mongodb import (  # noqa: E402
    EXTRA,
    _cursor_value,
    _flatten,
    _incremental_fields,
)
from elbi.warehouse.sources.registry import SourceRegistry  # noqa: E402

URI = "mongodb://127.0.0.1:17017"
DATABASE = "elbi_live_test"


@pytest.fixture
def mongo() -> Any:
    """A database seeded for one test, dropped on the way in rather than the way out."""
    client = pymongo.MongoClient(URI, serverSelectionTimeoutMS=1500)
    try:
        client.admin.command("ping")
    except Exception:
        client.close()
        pytest.skip(f"no MongoDB at {URI}; start one to run the live MongoDB tests")
    client.drop_database(DATABASE)
    database = client[DATABASE]
    database.events.insert_many(
        [
            {
                "user": "ada",
                "amount": 10,
                "at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "tags": ["a", "b"],
                "meta": {"k": 1},
            },
            {
                "user": "bo",
                "amount": 20,
                "at": datetime(2026, 2, 1, tzinfo=timezone.utc),
            },
            {
                "user": "cy",
                "amount": "thirty",
                "at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            },
        ]
    )
    database.events.create_index("at")
    # Compound, so `amount` is indexed but is not the leading key.
    database.events.create_index([("user", 1), ("amount", 1)])
    database.profiles.insert_many([{"name": f"p{i}"} for i in range(3)])
    yield database
    client.drop_database(DATABASE)
    client.close()


def _config(**over: Any) -> dict[str, Any]:
    return {"connection_string": URI, "database": DATABASE, **over}


def _rows(schema: str, **kw: Any) -> list[dict[str, Any]]:
    source = SourceRegistry.get("mongodb")
    tables = list(source.extract(SourceInputs(config=_config(), schema=schema, **kw)))
    return [row for table in tables for row in table.to_pylist()]


def _schema(schema: str) -> Any:
    source = SourceRegistry.get("mongodb")
    tables = list(source.extract(SourceInputs(config=_config(), schema=schema)))
    return tables[0].schema


# --- catalog ----------------------------------------------------------------------


def test_mongodb_is_catalogued_as_a_database() -> None:
    config = SourceRegistry.get("mongodb").config
    assert config.label == "MongoDB"
    assert config.category == "Databases"
    names = [f.name for f in config.fields]
    # Its own two, then whatever the tunnel appends; the order is what the form renders.
    assert names[:2] == ["connection_string", "database"]
    assert "ssh_host" in names


def test_the_connection_string_is_stored_as_a_secret() -> None:
    """It carries the password, so it must never be a plain text field."""
    fields = {f.name: f for f in SourceRegistry.get("mongodb").config.fields}
    assert fields["connection_string"].type == "password"
    assert fields["database"].required is False


# --- live: listing and discovery --------------------------------------------------


def test_live_every_collection_is_offered_as_a_table(mongo: Any) -> None:
    names = [s.name for s in SourceRegistry.get("mongodb").schemas(_config())]
    assert names == ["events", "profiles"]


def test_live_a_cursor_is_offered_only_on_a_field_that_leads_an_index(
    mongo: Any,
) -> None:
    """The rule that keeps incremental sync from being slower than a full refresh.

    ``at`` leads its own index and is offered. ``amount`` sits second in a compound
    index, which does not accelerate ``amount > cursor`` at all, so offering it would
    hand the user a sync that rescans the collection every time.
    """
    schemas = {
        s.name: s.incremental_fields
        for s in SourceRegistry.get("mongodb").schemas(_config())
    }
    assert schemas["events"] == ["_id", "at"]
    assert "amount" not in schemas["events"]
    assert "user" not in schemas["events"]


def test_live_a_collection_with_only_its_id_index_still_offers_that(mongo: Any) -> None:
    schemas = {
        s.name: s.incremental_fields
        for s in SourceRegistry.get("mongodb").schemas(_config())
    }
    assert schemas["profiles"] == ["_id"]


def test_live_validate_reports_a_server_that_is_not_there() -> None:
    ok, errors = SourceRegistry.get("mongodb").validate(
        {"connection_string": "mongodb://127.0.0.1:1/", "database": "x"}
    )
    assert not ok
    assert "Could not connect" in " ".join(errors)


def test_live_a_connection_string_with_no_database_says_so() -> None:
    ok, errors = SourceRegistry.get("mongodb").validate({"connection_string": URI})
    assert not ok
    assert "Database field" in " ".join(errors)


# --- live: types ------------------------------------------------------------------


def test_live_a_consistently_typed_field_keeps_its_type(mongo: Any) -> None:
    types = {f.name: str(f.type) for f in _schema("events")}
    assert types["at"] == "timestamp[ms, tz=timezone.utc]"
    assert types["user"] == "string"


def test_live_a_field_holding_two_types_becomes_text_rather_than_failing(
    mongo: Any,
) -> None:
    """Arrow columns hold one type; MongoDB fields do not have to. Text keeps both."""
    assert str(_schema("events").field("amount").type) == "string"
    amounts = sorted(r["amount"] for r in _rows("events"))
    assert amounts == ["10", "20", "thirty"]


def test_live_documents_and_arrays_are_kept_as_json_not_dropped(mongo: Any) -> None:
    row = next(r for r in _rows("events") if r["user"] == "ada")
    assert row["tags"] == '["a", "b"]'
    assert row["meta"] == '{"k": 1}'


def test_live_an_object_id_arrives_as_its_hex_string(mongo: Any) -> None:
    rows = _rows("events")
    assert all(len(r["_id"]) == 24 for r in rows)


def test_live_a_field_missing_from_a_document_is_null_not_absent(mongo: Any) -> None:
    """Every row in a batch needs the same keys, or the batches will not concatenate."""
    row = next(r for r in _rows("events") if r["user"] == "bo")
    assert row["tags"] is None
    assert set(row) == {"_id", "amount", "at", "meta", "tags", "user", EXTRA}


def test_live_a_field_introduced_after_the_sample_window_is_still_carried(
    mongo: Any,
) -> None:
    """The data-loss bug this column exists for.

    Inference reads the first documents in the collection. A field added to recent
    records is not in that sample, and before the catch-all it was dropped with nothing
    to show it had ever been there.
    """
    from elbi.warehouse.sources.mongodb import _SAMPLE

    mongo.wide.insert_many([{"n": i} for i in range(_SAMPLE + 5)])
    mongo.wide.update_one({"n": _SAMPLE + 4}, {"$set": {"late": "kept"}})

    rows = _rows("wide")
    carried = [r for r in rows if r[EXTRA]]
    assert len(carried) == 1
    assert carried[0][EXTRA] == '{"late": "kept"}'


def test_live_a_document_with_nothing_extra_leaves_the_catch_all_null(
    mongo: Any,
) -> None:
    assert all(r[EXTRA] is None for r in _rows("events"))


# --- live: incremental ------------------------------------------------------------


def test_live_a_cursor_returns_only_documents_past_it(mongo: Any) -> None:
    rows = _rows(
        "events",
        incremental_field="at",
        incremental_since=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )
    assert sorted(r["user"] for r in rows) == ["bo", "cy"]


def test_live_a_cursor_past_everything_returns_nothing(mongo: Any) -> None:
    rows = _rows(
        "events",
        incremental_field="at",
        incremental_since=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    assert rows == []


def test_live_an_object_id_cursor_survives_the_round_trip_as_text(mongo: Any) -> None:
    """It is written to the warehouse as hex, so it comes back as hex and must convert.

    Compared as a string it would match no document, and the sync would quietly return
    nothing for ever after -- a failure with no error to notice.
    """
    first = sorted(r["_id"] for r in _rows("events"))[0]
    rows = _rows("events", incremental_field="_id", incremental_since=first)
    assert len(rows) == 2
    assert first not in {r["_id"] for r in rows}


# --- the pieces, without a server -------------------------------------------------


def test_only_orderable_indexed_fields_are_offered_as_cursors() -> None:
    fields = {
        "at": {"date"},
        "seq": {"long"},
        "name": {"string"},
        "flag": {"bool"},
        "mixed": {"long", "string"},
        "unindexed": {"date"},
    }
    indexed = {"at", "seq", "name", "flag", "mixed"}
    assert _incremental_fields(fields, indexed) == ["at", "seq"]


def test_a_field_that_is_sometimes_null_is_still_orderable() -> None:
    """Nulls are not a second type for this purpose; every optional field has them."""
    assert _incremental_fields({"at": {"date", "null"}}, {"at"}) == []


def test_an_object_id_cursor_is_converted_back_but_only_for_the_id_field() -> None:
    from bson import ObjectId

    hex_id = "6a95d1b1847e1d4ee37defa1"
    assert _cursor_value("_id", hex_id) == ObjectId(hex_id)
    assert _cursor_value("other", hex_id) == hex_id


def test_a_cursor_that_is_not_a_valid_object_id_is_left_alone() -> None:
    """Better to compare it as written than to raise inside a scheduled sync."""
    assert _cursor_value("_id", "not-an-object-id") == "not-an-object-id"


def test_binary_becomes_hex_rather_than_breaking_the_column() -> None:
    assert _flatten(b"\x00\xff") == "00ff"


def test_a_decimal_keeps_its_precision_as_text() -> None:
    from bson import Decimal128

    assert _flatten(Decimal128("1.10")) == "1.10"
