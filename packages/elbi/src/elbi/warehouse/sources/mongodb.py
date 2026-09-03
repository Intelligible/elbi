"""MongoDB connector: a collection is a table, read over the document protocol.

The other database connectors share one SQLAlchemy engine because they share a query
language. MongoDB shares neither, so this brings its own listing, its own schema
discovery and its own reader. What it does not bring is a different contract: a
collection is offered as a table, with the same incremental cursor the SQL sources use.

Three problems have to be solved that a relational source does not pose.

A collection has no declared schema, so one is inferred by asking the server to report
the keys and BSON types across a bounded sample. That is an aggregation the server runs
against the ``_id`` index, not a scan, so it stays cheap on a large collection.

A field can hold different types in different documents, and Arrow columns cannot. A
field whose sample shows one scalar type is carried across natively; anything else --
mixed types, embedded documents, arrays -- is written as JSON text, so no document is
dropped and nothing is silently coerced into the wrong type.

An incremental cursor is only offered on a field that leads an index. Filtering on
anything else makes MongoDB read the whole collection every sync, which is slower than
the full refresh it was meant to replace, so offering it would be a trap rather than a
feature.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.mongo_client import MongoClient

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, Source, SourceInputs
from .registry import SourceRegistry
from .tunnel import SshTunnel, routed_url, tunnel_fields

#: Anything a document holds that the inferred schema does not name, as JSON. The
#: sample below looks at the first documents in the collection, so a field introduced
#: later -- a new attribute on recent records, exactly the kind worth having -- is not
#: in it. Without somewhere to put those values they would be dropped with no sign that
#: anything was missing, which is the one outcome a sync must not have.
EXTRA = "_extra"

#: Documents read per round trip, and rows per Arrow batch.
_BATCH = 5_000

#: Documents the server looks at when inferring a collection's fields. Large enough to
#: see the optional fields that matter, small enough that the aggregation stays a read
#: of the ``_id`` index rather than a collection scan.
_SAMPLE = 1_000

#: How long the server may spend on that inference before giving up. A collection that
#: cannot answer in this budget still syncs -- the fallback below just knows less.
_SAMPLE_TIMEOUT_MS = 15_000

#: BSON types that a `>` comparison orders meaningfully, so a cursor over them advances.
#: Notably absent: string, which orders but whose ordering rarely matches arrival, and
#: bool, object and array, which do not usefully order at all.
_ORDERABLE = frozenset(
    {"date", "int", "long", "double", "decimal", "objectId", "timestamp"}
)

#: BSON types carried into Arrow as themselves. Everything else becomes JSON text.
_SCALARS = frozenset(
    {"double", "string", "bool", "date", "int", "long", "decimal", "objectId"}
)


def _tunnel_problem(config: dict[str, Any]) -> str | None:
    """Why this connection string cannot be tunnelled, if it cannot.

    A forward reaches one address. Both forms below name a *set* of servers, and the
    driver resolves the set and then connects to whatever the servers advertise
    themselves as -- which is their own hostnames, on the far side of the tunnel, where
    nothing here can reach them. The sync would appear to start and then hang, so it is
    refused up front with the reason instead.
    """
    if SshTunnel.from_config(config) is None:
        return None
    uri = str(config.get("connection_string") or "").strip()
    if uri.startswith("mongodb+srv://"):
        return (
            "a mongodb+srv:// connection string cannot be used through an SSH tunnel: "
            "it names a DNS record that expands to several servers, and the driver "
            "would connect to those directly. Use a mongodb:// string naming the one "
            "server to read, or reach the cluster over a private network instead."
        )
    host_part = uri.split("://", 1)[-1].split("/", 1)[0].rsplit("@", 1)[-1]
    if "," in host_part:
        return (
            "a connection string listing several servers cannot be used through an "
            "SSH tunnel, because a forward reaches one of them. Name the single server "
            "to read from."
        )
    return None


@contextmanager
def _reachable(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The config, with its connection string pointed at a tunnel when there is one.

    ``directConnection`` is added along with the forward. Without it the driver treats
    the address as a seed, asks the server for the rest of its replica set, and then
    connects to the hostnames that server reports -- which are not reachable here. With
    it, the driver talks to the one server at the end of the tunnel and nothing else.
    """
    problem = _tunnel_problem(config)
    if problem:
        raise ValueError(problem)
    if SshTunnel.from_config(config) is None:
        yield config
        return
    with routed_url(config, "connection_string", 27017) as routed:
        uri = str(routed["connection_string"])
        joiner = "&" if "?" in uri else "?"
        yield {**routed, "connection_string": f"{uri}{joiner}directConnection=true"}


@contextmanager
def _client(config: dict[str, Any]) -> Iterator[MongoClient[dict[str, Any]]]:
    """A client for this connection, closed when the caller is done with it.

    Closed explicitly rather than left to the garbage collector, because a MongoClient
    starts background monitor threads per server and keeps a connection pool open; a
    long-running app that dropped one per sync would accumulate both.
    """
    from pymongo import MongoClient

    client: MongoClient[dict[str, Any]] = MongoClient(
        str(config.get("connection_string") or "").strip(),
        serverSelectionTimeoutMS=10_000,
        connectTimeoutMS=10_000,
    )
    try:
        yield client
    finally:
        client.close()


def _database(
    client: MongoClient[dict[str, Any]], config: dict[str, Any]
) -> Database[dict[str, Any]]:
    """The database to read: the one named on the form, else the one in the URI.

    A connection string may carry a default database or none at all, and a user whose
    string omits it has no other way to say which they meant -- hence the override.
    """
    if named := str(config.get("database") or "").strip():
        return client[named]
    try:
        default = client.get_default_database()
    except Exception:
        # The driver raises rather than returning None when the URI carries no path.
        # Either way the answer for the caller is the same, and it is not a failure to
        # connect -- the server is reachable, it just was not told what to read.
        default = None
    if default is None:
        raise ValueError(
            "the connection string names no database, so one has to be given in the "
            "Database field"
        )
    return default


def _infer(collection: Collection[dict[str, Any]]) -> dict[str, set[str]]:
    """Each field in a sample of the collection, and the BSON types seen in it.

    The work happens on the server: ``$objectToArray`` turns each document into its
    key-value pairs, ``$unwind`` splits them into rows, and ``$group`` collects the
    distinct ``$type`` per key. Only the summary crosses the wire.
    """
    pipeline: list[dict[str, Any]] = [
        {"$limit": _SAMPLE},
        {"$project": {"pairs": {"$objectToArray": "$$ROOT"}}},
        {"$unwind": "$pairs"},
        {"$group": {"_id": "$pairs.k", "types": {"$addToSet": {"$type": "$pairs.v"}}}},
    ]
    result = collection.aggregate(pipeline, maxTimeMS=_SAMPLE_TIMEOUT_MS)
    return {row["_id"]: set(row["types"]) for row in result}


def _leading_index_keys(collection: Collection[dict[str, Any]]) -> set[str]:
    """The fields that are the first key of some index.

    Only a leading key accelerates ``field > cursor``. A field sitting second in a
    compound index reads as indexed and behaves like an unindexed one for this access
    pattern, which is exactly the case that would otherwise be offered by mistake.
    """
    keys: set[str] = set()
    for index in collection.list_indexes():
        spec = index.get("key") or {}
        leading = next(iter(spec), None)
        if leading is not None:
            keys.add(str(leading))
    return keys


def _incremental_fields(fields: dict[str, set[str]], indexed: set[str]) -> list[str]:
    """Fields that both order meaningfully and lead an index."""
    return sorted(
        name
        for name, types in fields.items()
        if name in indexed and types and types <= _ORDERABLE
    )


def _json_default(value: Any) -> str:
    """A BSON value with no JSON form, rendered as text rather than refused."""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _flatten(value: Any) -> Any:
    """One document value, made safe to put in an Arrow column.

    Scalars pass through. Anything structured becomes JSON text, which keeps the whole
    value queryable in the warehouse instead of dropping it for not fitting a column.
    """
    from bson import Binary, Decimal128, ObjectId

    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Decimal128):
        return str(value.to_decimal())
    if isinstance(value, Binary | bytes):
        return bytes(value).hex()
    if isinstance(value, dict | list):
        return json.dumps(value, default=_json_default)
    return value


def _row(
    document: dict[str, Any], fields: dict[str, set[str]], native: set[str]
) -> dict[str, Any]:
    """One document as a flat row, with every inferred field present.

    Fields missing from this document are filled with ``None`` so every row in a batch
    has the same keys; without that, Arrow would infer a different schema per batch and
    the Delta writer would be merging columns that were never actually new.

    Fields the inference never saw go to :data:`EXTRA` rather than being discarded.
    """
    row: dict[str, Any] = dict.fromkeys(fields)
    extra: dict[str, Any] = {}
    for key, value in document.items():
        flat = _flatten(value)
        if key in fields:
            row[key] = flat if key in native else _as_text(flat)
        else:
            extra[key] = flat
    row[EXTRA] = json.dumps(extra, default=_json_default) if extra else None
    return row


def _as_text(value: Any) -> str | None:
    """A value forced to text, for a field whose type is not consistent."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    return json.dumps(value, default=_json_default)


def _cursor_value(field: str, since: Any) -> Any:
    """The stored cursor, converted back to what MongoDB will compare against.

    An ``ObjectId`` is written to the warehouse as its hex string, so it comes back as
    one and has to become an ObjectId again -- compared as a string it would match
    nothing and the sync would silently return no rows.
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    if field == "_id" and isinstance(since, str):
        try:
            return ObjectId(since)
        except (InvalidId, TypeError):
            return since
    return since


@SourceRegistry.register
class MongoDbSource(SimpleSource):
    """MongoDB, and the hosted services that speak its protocol.

    One connection string covers every deployment shape MongoDB has -- a single server,
    a replica set, Atlas's ``mongodb+srv://`` -- because the driver already reads all of
    them. Asking for host, port and replica-set name separately would be re-parsing what
    the user can paste directly from their own dashboard.
    """

    supports_column_selection = True

    @property
    def source_type(self) -> str:
        """The registry key."""
        return "mongodb"

    @property
    def config(self) -> SourceConfig:
        """A connection string, and the database to read if the string omits it."""
        return SourceConfig(
            name="mongodb",
            label="MongoDB",
            category="Databases",
            icon="🍃",
            caption="Sync collections from a MongoDB database.",
            docs_url="https://www.mongodb.com/docs/manual/reference/connection-string/",
            fields=[
                SourceField(
                    name="connection_string",
                    label="Connection string",
                    type="password",
                    placeholder="mongodb+srv://user:password@cluster.mongodb.net/mydb",
                    caption="The whole string, as your provider gives it. It carries "
                    "the password, so it is stored encrypted and never shown again.",
                ),
                SourceField(
                    name="database",
                    label="Database",
                    required=False,
                    placeholder="analytics",
                    caption="Only needed if the connection string does not name one.",
                ),
                *tunnel_fields(),
            ],
        )

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the fields, then ping the server and resolve the database."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            with _reachable(config) as routed, _client(routed) as client:
                database = _database(client, routed)
                database.command("ping")
        except ValueError as e:
            # Raised above, already phrased for whoever is filling in the form.
            return False, [str(e)]
        except Exception as e:  # any failure the driver raises
            return False, [f"Could not connect: {e}"]
        return True, []

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Every readable collection, with the cursor fields it can actually use.

        Collections the credential may not read are left out rather than listed and
        failed later, and a view is skipped because it has no index of its own to
        support a cursor and no stable identity to sync.
        """
        out: list[SourceSchema] = []
        with _reachable(config) as routed, _client(routed) as client:
            database = _database(client, routed)
            for name in sorted(
                database.list_collection_names(
                    filter={"type": "collection"}, authorizedCollections=True
                )
            ):
                collection = database[name]
                try:
                    fields = _infer(collection)
                    indexed = _leading_index_keys(collection)
                except Exception:
                    # Inference is an optimisation. A collection whose sample times out,
                    # or whose indexes cannot be listed, still syncs as a full refresh.
                    fields, indexed = {}, set()
                out.append(
                    SourceSchema(
                        name=name,
                        incremental_fields=_incremental_fields(fields, indexed),
                    )
                )
        return out

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Read a collection in batches, past the cursor when syncing incrementally."""
        with _reachable(inputs.config) as routed, _client(routed) as client:
            collection = _database(client, routed)[inputs.schema]
            fields = _infer(collection)
            if not fields:
                fields = {"_id": {"objectId"}}
            native = {
                name for name, types in fields.items() if types and types <= _SCALARS
            }
            # A field seen as more than one scalar type is still not one Arrow column,
            # so only the single-typed ones stay native.
            native = {
                name
                for name in native
                if len({t for t in fields[name] if t != "null"}) <= 1
            }

            query: dict[str, Any] = {}
            if inputs.incremental_field and inputs.incremental_since is not None:
                query[inputs.incremental_field] = {
                    "$gt": _cursor_value(
                        inputs.incremental_field, inputs.incremental_since
                    )
                }

            schema = _arrow_schema(fields, native)
            batch: list[dict[str, Any]] = []
            cursor = collection.find(query, batch_size=_BATCH)
            try:
                for document in cursor:
                    batch.append(_row(document, fields, native))
                    if len(batch) >= _BATCH:
                        yield pa.Table.from_pylist(batch, schema=schema)
                        batch = []
            finally:
                cursor.close()
            if batch:
                yield pa.Table.from_pylist(batch, schema=schema)


def _arrow_schema(fields: dict[str, set[str]], native: set[str]) -> pa.Schema:
    """A fixed Arrow schema for the collection, so every batch has the same shape.

    Declared rather than inferred per batch: pyarrow reading one batch of documents that
    all happen to omit an optional field would type that column differently from the
    next batch's, and the two would not concatenate.
    """
    return pa.schema(
        [
            pa.field(name, _arrow_type(fields[name]) if name in native else pa.string())
            for name in sorted(fields)
        ]
        + [pa.field(EXTRA, pa.string())]
    )


def _arrow_type(types: set[str]) -> pa.DataType:
    """The Arrow type for a field seen as exactly one BSON scalar type."""
    concrete = {t for t in types if t != "null"}
    bson = next(iter(concrete), "string")
    return {
        "double": pa.float64(),
        "int": pa.int32(),
        "long": pa.int64(),
        "bool": pa.bool_(),
        "date": pa.timestamp("ms", tz="UTC"),
    }.get(bson, pa.string())


# Re-exported so the tests can drive the pieces the public methods compose.
__all__ = [
    "EXTRA",
    "MongoDbSource",
    "Source",
    "_cursor_value",
    "_flatten",
    "_incremental_fields",
    "_leading_index_keys",
]
