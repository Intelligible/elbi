"""Elasticsearch and DynamoDB: the two sources with no schema to read.

Both are driven against real services in Docker, because what is worth testing is what a
real server does with real documents -- Elasticsearch's mapping and its paging,
DynamoDB's tagged types and 38-digit numbers. Both skip when nothing is listening.

The pieces that need no service are tested directly at the bottom.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from elbi.warehouse.sources.base import SourceInputs
from elbi.warehouse.sources.registry import SourceRegistry

ES_URL = "http://127.0.0.1:19200"
DDB_ENDPOINT = "http://127.0.0.1:18000"


def _rows(
    source_type: str, config: dict[str, Any], schema: str, **kw: Any
) -> list[dict]:
    source = SourceRegistry.get(source_type)
    batches = source.extract(SourceInputs(config=config, schema=schema, **kw))
    return [row for batch in batches for row in batch.to_pylist()]


def _arrow_schema(source_type: str, config: dict[str, Any], schema: str) -> Any:
    source = SourceRegistry.get(source_type)
    return next(iter(source.extract(SourceInputs(config=config, schema=schema)))).schema


# --- Elasticsearch ----------------------------------------------------------------


def _es(method: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{ES_URL}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read() or b"{}")


ES_CONFIG: dict[str, Any] = {
    "url": ES_URL,
    "username": "elastic",
    "password": "",
    "verify_certs": False,
}


@pytest.fixture
def elastic() -> str:
    """An index with a date, an integer, a nested object and a custom-format date."""
    try:
        _es("GET", "/_cluster/health")
    except Exception:
        pytest.skip(f"no Elasticsearch at {ES_URL}; start one for the live tests")
    index = "elbi_live_orders"
    with suppress(urllib.error.HTTPError):
        # Absent on the first run, which is not a failure worth reporting.
        _es("DELETE", f"/{index}")
    _es(
        "PUT",
        f"/{index}",
        {
            "mappings": {
                "properties": {
                    "sku": {"type": "keyword"},
                    "qty": {"type": "integer"},
                    "at": {"type": "date"},
                    "legacy_day": {"type": "date", "format": "yyyyMMdd"},
                    "buyer": {"properties": {"name": {"type": "keyword"}}},
                    "lines": {"type": "nested"},
                }
            }
        },
    )
    for number, (sku, qty, at) in enumerate(
        [("a", 1, "2026-01-01"), ("b", 2, "2026-02-01"), ("c", 3, "2026-03-01")],
        start=1,
    ):
        _es(
            "PUT",
            f"/{index}/_doc/{number}?refresh=true",
            {
                "sku": sku,
                "qty": qty,
                "at": at,
                "legacy_day": "20260101",
                "buyer": {"name": f"n{number}"},
                "lines": [{"item": "x"}],
            },
        )
    yield index
    with _quiet():
        _es("DELETE", f"/{index}")


class _quiet:
    """Swallow a teardown failure: the index is disposable and the test already ran."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


def test_elasticsearch_is_catalogued_as_a_database() -> None:
    config = SourceRegistry.get("elasticsearch").config
    assert config.label == "Elasticsearch"
    assert config.category == "Databases"


def test_elasticsearch_needs_one_of_the_two_credentials() -> None:
    ok, errors = SourceRegistry.get("elasticsearch").validate({"url": ES_URL})
    assert not ok
    assert "API key or a username" in " ".join(errors)


def test_live_es_offers_every_index_but_not_the_system_ones(elastic: str) -> None:
    """A dotted index is Kibana's own state or the cluster's; nobody analyses those."""
    names = [s.name for s in SourceRegistry.get("elasticsearch").schemas(ES_CONFIG)]
    assert elastic in names
    assert not any(name.startswith(".") for name in names)


def test_live_es_offers_dates_and_integers_as_cursors(elastic: str) -> None:
    schemas = {
        s.name: s.incremental_fields
        for s in SourceRegistry.get("elasticsearch").schemas(ES_CONFIG)
    }
    assert schemas[elastic] == ["at", "qty"]
    # keyword orders lexically, which has nothing to do with when a document arrived.
    assert "sku" not in schemas[elastic]


def test_live_es_flattens_an_object_into_a_dotted_column(elastic: str) -> None:
    rows = _rows("elasticsearch", ES_CONFIG, elastic)
    assert sorted(r["buyer.name"] for r in rows) == ["n1", "n2", "n3"]


def test_live_es_keeps_a_nested_field_whole_as_json(elastic: str) -> None:
    """Its children repeat per document, so they cannot be one column each."""
    rows = _rows("elasticsearch", ES_CONFIG, elastic)
    assert json.loads(rows[0]["lines"]) == [{"item": "x"}]


def test_live_es_parses_a_default_format_date(elastic: str) -> None:
    types = {
        f.name: str(f.type) for f in _arrow_schema("elasticsearch", ES_CONFIG, elastic)
    }
    assert types["at"] == "timestamp[ms, tz=timezone.utc]"
    rows = _rows("elasticsearch", ES_CONFIG, elastic)
    assert rows[0]["at"] == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_live_es_keeps_a_custom_format_date_verbatim(elastic: str) -> None:
    """Elasticsearch takes any Joda pattern, so guessing at one would lose it."""
    types = {
        f.name: str(f.type) for f in _arrow_schema("elasticsearch", ES_CONFIG, elastic)
    }
    assert types["legacy_day"] == "string"
    assert _rows("elasticsearch", ES_CONFIG, elastic)[0]["legacy_day"] == "20260101"


def test_live_es_carries_the_document_id(elastic: str) -> None:
    """It lives outside `_source`, so without this the rows have no stable identity."""
    from elbi.warehouse.sources.elasticsearch import DOC_ID

    rows = _rows("elasticsearch", ES_CONFIG, elastic)
    assert sorted(r[DOC_ID] for r in rows) == ["1", "2", "3"]


def test_live_es_honours_a_cursor(elastic: str) -> None:
    rows = _rows(
        "elasticsearch",
        ES_CONFIG,
        elastic,
        incremental_field="at",
        incremental_since="2026-01-15",
    )
    assert sorted(r["sku"] for r in rows) == ["b", "c"]


def test_live_es_pages_past_one_batch_without_repeating_a_document(
    elastic: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paging is the part that goes wrong quietly, so it is driven past one page.

    With the batch forced to two, three documents take two pages. A `search_after` walk
    that lost its place would return a document twice or skip one, and the count and the
    id set together catch both.
    """
    from elbi.warehouse.sources import elasticsearch as module
    from elbi.warehouse.sources.elasticsearch import DOC_ID

    monkeypatch.setattr(module, "_BATCH", 2)
    rows = _rows("elasticsearch", ES_CONFIG, elastic)
    assert len(rows) == 3
    assert sorted(r[DOC_ID] for r in rows) == ["1", "2", "3"]


def test_live_es_reports_a_cluster_that_is_not_there() -> None:
    ok, errors = SourceRegistry.get("elasticsearch").validate(
        {"url": "http://127.0.0.1:1", "username": "x", "password": "y"}
    )
    assert not ok
    assert "Could not connect" in " ".join(errors)


# --- DynamoDB ---------------------------------------------------------------------

DDB_CONFIG: dict[str, Any] = {
    "region": "us-east-1",
    "endpoint": DDB_ENDPOINT,
    "access_key_id": "local",
    "secret_access_key": "local",
}


@pytest.fixture
def dynamo() -> str:
    """A table holding an integral number, a fractional one, a list and a map."""
    boto3 = pytest.importorskip("boto3")
    resource = boto3.resource(
        "dynamodb",
        region_name="us-east-1",
        endpoint_url=DDB_ENDPOINT,
        aws_access_key_id="local",
        aws_secret_access_key="local",
    )
    try:
        list(resource.tables.limit(1))
    except Exception:
        pytest.skip(f"no DynamoDB at {DDB_ENDPOINT}; start one for the live tests")
    name = "elbi_live_orders"
    try:
        resource.Table(name).delete()
        resource.Table(name).wait_until_not_exists()
    except Exception:
        pass
    table = resource.create_table(
        TableName=name,
        KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    table.put_item(
        Item={
            "id": "1",
            "qty": Decimal(2),
            "price": Decimal("9.99"),
            "tags": ["a", "b"],
            "ok": True,
        }
    )
    table.put_item(
        Item={
            "id": "2",
            "qty": Decimal(5),
            "price": Decimal("1.50"),
            "meta": {"k": "v"},
        }
    )
    yield name
    with _quiet():
        resource.Table(name).delete()


def test_dynamodb_is_catalogued_as_a_database() -> None:
    config = SourceRegistry.get("dynamodb").config
    assert config.label == "DynamoDB"
    assert config.category == "Databases"


def test_live_dynamodb_lists_its_tables(dynamo: str) -> None:
    names = [s.name for s in SourceRegistry.get("dynamodb").schemas(DDB_CONFIG)]
    assert dynamo in names


def test_live_dynamodb_offers_no_cursor_and_says_so_by_offering_none(
    dynamo: str,
) -> None:
    """A filtered scan reads and costs the same as an unfiltered one, so there is no
    cheaper incremental read to offer -- and offering one anyway would be a lie about
    what it costs."""
    schemas = {
        s.name: s.incremental_fields
        for s in SourceRegistry.get("dynamodb").schemas(DDB_CONFIG)
    }
    assert schemas[dynamo] == []


def test_live_dynamodb_keeps_an_integral_number_as_an_integer(dynamo: str) -> None:
    types = {f.name: str(f.type) for f in _arrow_schema("dynamodb", DDB_CONFIG, dynamo)}
    assert types["qty"] == "int64"
    assert sorted(r["qty"] for r in _rows("dynamodb", DDB_CONFIG, dynamo)) == [2, 5]


def test_live_dynamodb_will_not_round_a_fractional_number_into_a_float(
    dynamo: str,
) -> None:
    """DynamoDB stores 38 significant digits. A silently wrong amount is worse than a
    string that has to be cast."""
    types = {f.name: str(f.type) for f in _arrow_schema("dynamodb", DDB_CONFIG, dynamo)}
    assert types["price"] == "string"
    assert sorted(r["price"] for r in _rows("dynamodb", DDB_CONFIG, dynamo)) == [
        "1.5",
        "9.99",
    ]


def test_live_dynamodb_keeps_lists_and_maps_as_json(dynamo: str) -> None:
    rows = {r["id"]: r for r in _rows("dynamodb", DDB_CONFIG, dynamo)}
    assert json.loads(rows["1"]["tags"]) == ["a", "b"]
    assert json.loads(rows["2"]["meta"]) == {"k": "v"}


def test_live_dynamodb_leaves_an_absent_attribute_null(dynamo: str) -> None:
    rows = {r["id"]: r for r in _rows("dynamodb", DDB_CONFIG, dynamo)}
    assert rows["2"]["tags"] is None
    assert rows["1"]["meta"] is None


def test_live_dynamodb_reports_an_endpoint_that_is_not_there() -> None:
    ok, errors = SourceRegistry.get("dynamodb").validate(
        {**DDB_CONFIG, "endpoint": "http://127.0.0.1:1"}
    )
    assert not ok
    assert "Could not connect" in " ".join(errors)


# --- the pieces, without a service ------------------------------------------------


def test_an_elasticsearch_mapping_flattens_to_dotted_paths() -> None:
    from elbi.warehouse.sources.elasticsearch import _DATE_CUSTOM, _leaf_fields

    fields = _leaf_fields(
        {
            "sku": {"type": "keyword"},
            "buyer": {"properties": {"name": {"type": "keyword"}}},
            "lines": {"type": "nested", "properties": {"item": {"type": "keyword"}}},
            "at": {"type": "date"},
            "legacy": {"type": "date", "format": "yyyyMMdd"},
        }
    )
    assert fields == {
        "sku": "keyword",
        "buyer.name": "keyword",
        "lines": "nested",
        "at": "date",
        "legacy": _DATE_CUSTOM,
    }


def test_a_dynamodb_integral_decimal_becomes_an_int_and_a_fraction_becomes_text() -> (
    None
):
    from elbi.warehouse.sources.dynamodb import _flatten

    assert _flatten(Decimal("7")) == 7
    assert _flatten(Decimal("7.25")) == "7.25"
    # Past int64, so a float would lose digits and an int keeps them.
    assert _flatten(Decimal("123456789012345678901234567890")) == int(
        "123456789012345678901234567890"
    )


def test_a_dynamodb_set_becomes_a_sorted_json_array() -> None:
    """Sorted, so two syncs of an unchanged item produce the same bytes."""
    from elbi.warehouse.sources.dynamodb import _flatten

    assert json.loads(_flatten({"b", "a", "c"})) == ["a", "b", "c"]


def test_a_dynamodb_binary_attribute_becomes_hex() -> None:
    from elbi.warehouse.sources.dynamodb import _flatten

    assert _flatten(b"\x00\xff") == "00ff"
