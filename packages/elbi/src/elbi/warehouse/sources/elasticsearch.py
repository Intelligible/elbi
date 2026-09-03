"""Elasticsearch connector: an index is a table, read through a point-in-time view.

Reading an index is not the same problem as reading a table. An index is being written
to while it is read, and the two obvious ways of paging through it both go wrong: `from`
and `size` cannot go past ten thousand documents, and a plain `search_after` walks a
shifting target, so a document indexed mid-read can be returned twice or missed.

The fix is Elasticsearch's own: open a point-in-time, page through it with
`search_after`, and close it at the end. Every page then reads the same frozen view of
the index, which is what makes a sync of a live index reproducible.

Fields come from the index mapping rather than from sampling documents, because unlike
a document store Elasticsearch already knows its own schema. A field the mapping
declares as a date or an integer is offered as an incremental cursor; ordering on
anything else would not correspond to arrival.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry
from .tunnel import routed_url, tunnel_fields

#: Documents per page, and rows per Arrow batch.
_BATCH = 2_000

#: How long the frozen view is kept alive between pages. Renewed on every request, so
#: this bounds the gap between pages rather than the length of the whole read.
_KEEP_ALIVE = "5m"

#: Mapping types that order the way arrival does, so a cursor over them advances.
_ORDERABLE = frozenset({"date", "date_nanos", "long", "integer", "short", "byte"})

#: A date field whose mapping declares its own ``format``. Elasticsearch accepts any
#: pattern there, so the stored string cannot be parsed without reimplementing Joda; the
#: value is kept exactly as written instead, which loses the type and loses no data.
_DATE_CUSTOM = "date_custom"

#: Mapping types carried into Arrow as themselves. Everything else becomes JSON text:
#: `object` and `nested` hold structure, `text` is analysed rather than stored whole.
_ARROW = {
    "date": pa.timestamp("ms", tz="UTC"),
    "date_nanos": pa.timestamp("us", tz="UTC"),
    "long": pa.int64(),
    "integer": pa.int32(),
    "short": pa.int16(),
    "byte": pa.int8(),
    "double": pa.float64(),
    "float": pa.float32(),
    "half_float": pa.float32(),
    "scaled_float": pa.float64(),
    "boolean": pa.bool_(),
    "keyword": pa.string(),
    "constant_keyword": pa.string(),
    "text": pa.string(),
    "ip": pa.string(),
    "version": pa.string(),
}

#: The document's own id, carried alongside its fields. Elasticsearch keeps it outside
#: `_source`, so without this the rows would arrive with no stable identity at all.
DOC_ID = "_doc_id"


@contextmanager
def _client(config: dict[str, Any]) -> Iterator[Elasticsearch]:
    """A client for this connection, closed when the caller is done with it."""
    from elasticsearch import Elasticsearch

    kwargs: dict[str, Any] = {
        "hosts": [str(config.get("url") or "").strip().rstrip("/")],
        "request_timeout": 60,
        "verify_certs": bool(config.get("verify_certs", True)),
    }
    if api_key := str(config.get("api_key") or "").strip():
        kwargs["api_key"] = api_key
    elif user := str(config.get("username") or "").strip():
        kwargs["basic_auth"] = (user, str(config.get("password") or ""))

    client = Elasticsearch(**kwargs)
    try:
        yield client
    finally:
        client.close()


def _leaf_fields(properties: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """The mapping flattened to ``dotted.path -> type``.

    An object field has no type of its own, only children, so it is walked into. A
    ``nested`` field is left whole: its children repeat per document and cannot be one
    column each, so the whole array is carried as JSON.
    """
    fields: dict[str, str] = {}
    for name, spec in properties.items():
        path = f"{prefix}{name}"
        kind = spec.get("type")
        if kind == "nested":
            fields[path] = "nested"
        elif children := spec.get("properties"):
            fields.update(_leaf_fields(children, f"{path}."))
        elif kind in ("date", "date_nanos") and spec.get("format"):
            fields[path] = _DATE_CUSTOM
        else:
            fields[path] = kind or "object"
    return fields


def _incremental_fields(fields: dict[str, str]) -> list[str]:
    """Mapping fields whose order corresponds to arrival."""
    return sorted(name for name, kind in fields.items() if kind in _ORDERABLE)


def _dig(document: dict[str, Any], path: str) -> Any:
    """One dotted path out of a document, or ``None`` where the path does not go."""
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _cell(value: Any, kind: str) -> Any:
    """One field value, made safe for its Arrow column.

    A field mapped as a single type can still arrive as an array -- Elasticsearch allows
    any field to hold one -- so anything structured becomes JSON text rather than
    breaking the column it was supposed to fit in.
    """
    if value is None:
        return None
    if isinstance(value, dict | list):
        return json.dumps(value, default=_json_default)
    if kind in ("date", "date_nanos"):
        return _timestamp(value)
    if kind not in _ARROW:
        # A type this connector has no column for -- a custom-format date, a geo point,
        # anything Elasticsearch adds later. A string is already text and is kept as it
        # stands; encoding it would wrap the value in quotes that were never in it.
        return (
            value
            if isinstance(value, str)
            else json.dumps(value, default=_json_default)
        )
    return value


def _timestamp(value: Any) -> datetime | None:
    """A default-format date, as a datetime.

    A date field carries its value in ``_source`` exactly as it was indexed, which for
    the two formats Elasticsearch accepts by default is either an ISO-8601 string or
    epoch milliseconds. Both are handled; a value that is neither becomes null rather
    than breaking the column, and a field whose mapping declares a custom format never
    reaches here -- it is typed as text and kept verbatim.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


@SourceRegistry.register
class ElasticsearchSource(SimpleSource):
    """Elasticsearch, and the OpenSearch forks that answer the same API."""

    supports_column_selection = True

    @property
    def source_type(self) -> str:
        """The registry key."""
        return "elasticsearch"

    @property
    def config(self) -> SourceConfig:
        """A cluster URL and one of the two credentials the API accepts."""
        return SourceConfig(
            name="elasticsearch",
            label="Elasticsearch",
            category="Databases",
            icon="🔍",
            caption="Sync indices from an Elasticsearch or OpenSearch cluster.",
            docs_url="https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-up-authentication.html",
            fields=[
                SourceField(
                    name="url",
                    label="Cluster URL",
                    placeholder="https://my-cluster.es.us-east-1.aws.found.io:9243",
                    caption="Including the scheme and port.",
                ),
                SourceField(
                    name="api_key",
                    label="API key",
                    type="password",
                    required=False,
                    caption="An encoded API key. Use this or a username and password.",
                ),
                SourceField(name="username", label="Username", required=False),
                SourceField(
                    name="password", label="Password", type="password", required=False
                ),
                SourceField(
                    name="verify_certs",
                    label="Verify TLS certificate",
                    type="switch",
                    required=False,
                    default=True,
                    caption="Turn off only for a cluster using a self-signed "
                    "certificate on a network you trust.",
                ),
                *tunnel_fields(),
            ],
        )

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the fields, then ask the cluster who it is."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        if not (config.get("api_key") or config.get("username")):
            return False, ["Either an API key or a username and password is required"]
        try:
            with (
                routed_url(config, "url", 9200) as reachable,
                _client(reachable) as client,
            ):
                client.info()
        except ValueError as e:
            # Raised by the tunnel, already phrased for whoever filled in the form.
            return False, [str(e)]
        except Exception as e:  # any failure the client raises
            return False, [f"Could not connect: {e}"]
        return True, []

    def _mapping(self, client: Elasticsearch, index: str) -> dict[str, str]:
        """The index's declared fields, flattened."""
        response = client.indices.get_mapping(index=index)
        for body in response.body.values():
            return _leaf_fields(body.get("mappings", {}).get("properties", {}))
        return {}

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Every index the credential can read, with its date fields as cursors.

        Hidden and system indices are left out. They start with a dot, hold Kibana's own
        state and the cluster's internals, and nobody sets up a warehouse to analyse
        those.
        """
        out: list[SourceSchema] = []
        with routed_url(config, "url", 9200) as reachable, _client(reachable) as client:
            names = sorted(
                name
                for name in client.indices.get_alias(index="*").body
                if not name.startswith(".")
            )
            for name in names:
                try:
                    fields = self._mapping(client, name)
                except Exception:
                    fields = {}
                out.append(
                    SourceSchema(
                        name=name, incremental_fields=_incremental_fields(fields)
                    )
                )
        return out

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Page through a frozen view of the index, oldest document first."""
        with (
            routed_url(inputs.config, "url", 9200) as reachable,
            _client(reachable) as client,
        ):
            fields = self._mapping(client, inputs.schema)
            schema = _arrow_schema(fields)

            query: dict[str, Any] = {"match_all": {}}
            if inputs.incremental_field and inputs.incremental_since is not None:
                query = {
                    "range": {
                        inputs.incremental_field: {
                            "gt": _bound(inputs.incremental_since)
                        }
                    }
                }

            # Sorting by the cursor when there is one, and by the document's own
            # shard-and-position tiebreak otherwise. `_shard_doc` is the cheapest total
            # order Elasticsearch offers and is only available inside a point-in-time,
            # which is the other reason to open one.
            sort: list[Any] = (
                [{inputs.incremental_field: "asc"}, {"_shard_doc": "asc"}]
                if inputs.incremental_field
                else [{"_shard_doc": "asc"}]
            )

            pit = client.open_point_in_time(
                index=inputs.schema, keep_alive=_KEEP_ALIVE
            )["id"]
            try:
                after: list[Any] | None = None
                while True:
                    body: dict[str, Any] = {
                        "size": _BATCH,
                        "query": query,
                        "sort": sort,
                        "pit": {"id": pit, "keep_alive": _KEEP_ALIVE},
                    }
                    if after is not None:
                        body["search_after"] = after
                    response = client.search(**body)
                    hits = response["hits"]["hits"]
                    if not hits:
                        break
                    yield pa.Table.from_pylist(
                        [_row(hit, fields) for hit in hits], schema=schema
                    )
                    after = hits[-1]["sort"]
                    # The view can be re-pinned across pages, so use whatever the last
                    # response named rather than assuming the original id still applies.
                    pit = response.get("pit_id", pit)
            finally:
                # The view expires on its own after the keep-alive above, so failing to
                # close it releases the same resources a moment later. Letting that
                # failure propagate would fail a sync whose rows are already written.
                with suppress(Exception):
                    client.close_point_in_time(id=pit)


def _bound(since: Any) -> Any:
    """The stored cursor in the form a range query will accept."""
    if isinstance(since, datetime | date):
        return since.isoformat()
    return since


def _row(hit: dict[str, Any], fields: dict[str, str]) -> dict[str, Any]:
    """One hit as a flat row, with the document id alongside its fields."""
    source = hit.get("_source") or {}
    row: dict[str, Any] = {
        name: _cell(_dig(source, name), kind) for name, kind in fields.items()
    }
    row[DOC_ID] = hit.get("_id")
    return row


def _arrow_schema(fields: dict[str, str]) -> pa.Schema:
    """A fixed Arrow schema from the mapping, so every page has the same shape."""
    return pa.schema(
        [
            pa.field(name, _ARROW.get(kind, pa.string()))
            for name, kind in sorted(fields.items())
        ]
        + [pa.field(DOC_ID, pa.string())]
    )
