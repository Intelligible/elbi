"""DynamoDB connector: a table is a table, read by scanning it.

DynamoDB is schemaless below its key, so the fields are inferred from a sample of items
and anything the sample does not name is carried as JSON. That is the same approach the
MongoDB connector takes, deliberately: they pose the same problem, and two different
answers to it would make the two sources behave differently for no reason a user could
see.

What differs is the reading. There is no cursor here and no incremental sync, and that
is a property of DynamoDB rather than a gap. A scan is the only way to read a table
without knowing a partition key, and adding a filter to a scan does not reduce what
DynamoDB reads or charges for -- it only discards rows after the fact. Offering an
incremental option would cost exactly as much as the full refresh it appeared to
replace, so the connector says plainly that it does full refreshes and leaves it there.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

#: Items per Arrow batch. Each scan page is capped at 1 MB by DynamoDB regardless.
_BATCH = 5_000

#: Items looked at when inferring a table's fields.
_SAMPLE = 1_000

#: Anything an item holds that the inferred fields do not name, as JSON. Same reason as
#: the MongoDB connector's column of the same name: without it, an attribute that only
#: appears on newer items would be dropped with nothing to show it was ever there.
EXTRA = "_extra"


def _resource(config: dict[str, Any]) -> Any:
    """A boto3 DynamoDB resource, which deserialises items for us.

    The resource API rather than the client: the client returns DynamoDB's own tagged
    form (``{"N": "1"}``) and every caller then has to undo it, while the resource hands
    back Python values. Credentials are optional so an instance role is used when there
    is one, which is the right default anywhere this runs inside AWS.
    """
    import boto3

    kwargs: dict[str, Any] = {"region_name": str(config.get("region") or "").strip()}
    key = str(config.get("access_key_id") or "").strip()
    secret = str(config.get("secret_access_key") or "").strip()
    if key and secret:
        kwargs["aws_access_key_id"] = key
        kwargs["aws_secret_access_key"] = secret
    if endpoint := str(config.get("endpoint") or "").strip():
        kwargs["endpoint_url"] = endpoint
    return boto3.resource("dynamodb", **kwargs)


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, set):
        return json.dumps(sorted(str(v) for v in value))
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _flatten(value: Any) -> Any:
    """One attribute value, made safe to put in an Arrow column.

    Numbers arrive as ``Decimal`` because DynamoDB stores 38 significant digits, more
    than a float can hold. An integral one becomes an int, which Python sizes to fit;
    anything with a fraction becomes text rather than being rounded into a float, since
    a silently wrong amount is worse than a string that has to be cast.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else str(value)
    if isinstance(value, bytes | bytearray):
        return bytes(value).hex()
    if isinstance(value, set):
        return json.dumps(sorted(str(v) for v in value))
    if isinstance(value, dict | list):
        return json.dumps(value, default=_json_default)
    return value


def _infer(table: Any) -> dict[str, set[str]]:
    """Attribute names in a sample of the table, and the Python types seen in each."""
    response = table.scan(Limit=_SAMPLE)
    fields: dict[str, set[str]] = {}
    for item in response.get("Items", []):
        for name, value in item.items():
            fields.setdefault(name, set()).add(type(_flatten(value)).__name__)
    return fields


def _arrow_type(types: set[str]) -> pa.DataType:
    """The Arrow type for an attribute seen as exactly one Python type."""
    concrete = types - {"NoneType"}
    if len(concrete) != 1:
        return pa.string()
    return {
        "int": pa.int64(),
        "float": pa.float64(),
        "bool": pa.bool_(),
    }.get(next(iter(concrete)), pa.string())


def _arrow_schema(fields: dict[str, set[str]]) -> pa.Schema:
    """A fixed schema, so every batch of a scan has the same shape."""
    return pa.schema(
        [pa.field(name, _arrow_type(types)) for name, types in sorted(fields.items())]
        + [pa.field(EXTRA, pa.string())]
    )


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=_json_default)


def _row(
    item: dict[str, Any], fields: dict[str, set[str]], native: set[str]
) -> dict[str, Any]:
    """One item as a flat row, with unnamed attributes collected into :data:`EXTRA`."""
    row: dict[str, Any] = dict.fromkeys(fields)
    extra: dict[str, Any] = {}
    for name, value in item.items():
        flat = _flatten(value)
        if name in fields:
            row[name] = flat if name in native else _as_text(flat)
        else:
            extra[name] = flat
    row[EXTRA] = json.dumps(extra, default=_json_default) if extra else None
    return row


@SourceRegistry.register
class DynamoDbSource(SimpleSource):
    """Amazon DynamoDB, and anything that answers its API on a local endpoint."""

    supports_column_selection = True

    @property
    def source_type(self) -> str:
        """The registry key."""
        return "dynamodb"

    @property
    def config(self) -> SourceConfig:
        """A region, optional credentials, and an endpoint for a local instance."""
        return SourceConfig(
            name="dynamodb",
            label="DynamoDB",
            category="Databases",
            icon="⚡",
            caption="Sync tables from Amazon DynamoDB.",
            docs_url="https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.html",
            fields=[
                SourceField(name="region", label="Region", placeholder="us-east-1"),
                SourceField(
                    name="access_key_id",
                    label="Access key ID",
                    required=False,
                    caption="Leave empty to use the credentials the host already has, "
                    "such as an instance role.",
                ),
                SourceField(
                    name="secret_access_key",
                    label="Secret access key",
                    type="password",
                    required=False,
                ),
                SourceField(
                    name="endpoint",
                    label="Endpoint URL",
                    required=False,
                    placeholder="http://localhost:8000",
                    caption="Only for DynamoDB Local. Leave empty for AWS.",
                ),
            ],
        )

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the fields, then list the tables to prove the credentials work."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            list(_resource(config).tables.limit(1))
        except Exception as e:  # any failure boto3 raises
            return False, [f"Could not connect: {e}"]
        return True, []

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Every table in the region.

        No incremental fields are offered, for the reason in the module docstring: a
        filtered scan reads and costs the same as an unfiltered one.
        """
        resource = _resource(config)
        return [
            SourceSchema(name=table.name, incremental_fields=[])
            for table in sorted(resource.tables.all(), key=lambda t: t.name)
        ]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Scan the table, following DynamoDB's paging key to the end."""
        table = _resource(inputs.config).Table(inputs.schema)
        fields = _infer(table)
        if not fields:
            # An empty table still has to produce a table with a schema, or the write
            # would land nothing and the next sync would see no existing columns.
            fields = {}
        native = {
            name
            for name, types in fields.items()
            if len(types - {"NoneType"}) == 1
            and next(iter(types - {"NoneType"})) in ("int", "float", "bool", "str")
        }
        schema = _arrow_schema(fields)

        batch: list[dict[str, Any]] = []
        start_key: dict[str, Any] | None = None
        while True:
            kwargs: dict[str, Any] = {}
            if start_key:
                kwargs["ExclusiveStartKey"] = start_key
            response = table.scan(**kwargs)
            for item in response.get("Items", []):
                batch.append(_row(item, fields, native))
                if len(batch) >= _BATCH:
                    yield pa.Table.from_pylist(batch, schema=schema)
                    batch = []
            start_key = response.get("LastEvaluatedKey")
            if not start_key:
                # DynamoDB stops returning a key when the scan has reached the end. A
                # page can come back empty before that and still not be the end, so the
                # key is what ends the loop rather than the item count.
                break
        if batch:
            yield pa.Table.from_pylist(batch, schema=schema)
