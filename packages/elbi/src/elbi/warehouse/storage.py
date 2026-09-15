"""Portable Delta Lake lakehouse storage: one config knob runs it anywhere.

``STORAGE_URI`` selects where warehouse tables live:

file:///var/lib/elbi/warehouse   # laptop: zero cloud deps (default) s3://bucket/prefix
# AWS or S3-compatible (STORAGE_ENDPOINT) gs://bucket/prefix                        #
GCP abfss://container@acct.dfs.../prefix      # Azure

Every synced source lands in an Apache **Delta Lake** table, written through delta-rs
(`deltalake`), so one code path serves a local directory on a laptop or the user's own
bucket in the cloud. Delta needs no external catalog, a table is just a directory (or
object-store prefix) of Parquet plus a transaction log, so there is nothing extra to run
locally. DuckDB, already the platform's compute backend, reads the tables directly.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..env import env

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE = f"file://{Path.home() / '.elbi' / 'warehouse'}"


def storage_uri() -> str:
    """The warehouse root: a local directory or a cloud bucket URL."""
    return (env("STORAGE_URI", _DEFAULT_STORAGE) or _DEFAULT_STORAGE).rstrip("/")


def _is_local() -> bool:
    return storage_uri().startswith("file://")


def table_uri(table: str) -> str:
    """The Delta table location for ``table``: the URI delta-rs and DuckDB both read.

    Local storage resolves to a plain filesystem path (both delta-rs and DuckDB's
    ``delta_scan`` accept it directly); cloud storage keeps the object-store URI.
    """
    root = storage_uri()
    if root.startswith("file://"):
        return str(Path(root[len("file://") :]) / table)
    return f"{root}/{table}"


def storage_options() -> dict[str, str]:
    """Object-store credentials/endpoint for delta-rs, from the environment.

    Empty for local ``file://`` storage. ``AWS_ENDPOINT_URL`` (set from
    ``STORAGE_ENDPOINT``) is what makes an S3-compatible store (MinIO locally, or
    another provider) work. ``AWS_S3_ALLOW_UNSAFE_RENAME`` is set because there is one
    writer and no locking provider, without which delta-rs refuses the rename.
    """
    if _is_local():
        return {}
    opts: dict[str, str] = {"AWS_S3_ALLOW_UNSAFE_RENAME": "true"}
    pairs = {
        "AWS_ENDPOINT_URL": "STORAGE_ENDPOINT",
        "AWS_ACCESS_KEY_ID": "STORAGE_KEY",
        "AWS_SECRET_ACCESS_KEY": "STORAGE_SECRET",
        "AWS_REGION": "STORAGE_REGION",
        "GOOGLE_SERVICE_ACCOUNT": "STORAGE_GCS_SERVICE_ACCOUNT",
        "AZURE_STORAGE_ACCOUNT_NAME": "STORAGE_AZURE_ACCOUNT",
        "AZURE_STORAGE_ACCOUNT_KEY": "STORAGE_AZURE_KEY",
    }
    for opt, var in pairs.items():
        value = env(var)
        if value:
            opts[opt] = value
    return opts


def open_readable(path: str) -> Any:
    """A seekable pyarrow stream over ``path`` -- a local path or an object-store URI.

    For reading a file the connector was pointed at, or validating an upload: seekable
    so a dialect can be sampled and then the read rewound. An object-store URI resolves
    through pyarrow's own credential discovery, the same as the reader that follows it.
    """
    import pyarrow.fs as pafs

    if "://" in path:
        filesystem, resolved = pafs.FileSystem.from_uri(path)
        return filesystem.open_input_file(resolved)
    return pafs.LocalFileSystem().open_input_file(path)


def warehouse_object(key: str) -> tuple[Any, str, str]:
    """The filesystem, path, and URI for ``<warehouse root>/<key>``.

    One code path over *whatever* the warehouse root is -- a local directory, S3, MinIO,
    GCS, or Azure -- reusing the same endpoint and credentials :func:`storage_options`
    resolves for delta-rs, rather than a second client that would miss a MinIO endpoint
    or an explicit key. The caller streams to the path, reads it back to validate, and
    deletes it on failure through the returned filesystem, so a raw upload can land
    beside the Delta tables and be read back by the connector.

    Returns ``(filesystem, path_within_filesystem, external_uri)``. On an object store,
    writing to the path performs a multipart upload underneath, so a large file is never
    buffered whole. ``external_uri`` is the value a source's ``path`` field takes.
    """
    root = storage_uri()
    filesystem, base = _warehouse_filesystem()
    path = f"{base}/{key}" if base else key
    uri = str(Path(path)) if root.startswith("file://") else f"{root}/{key}"
    return filesystem, path, uri


def _warehouse_filesystem() -> tuple[Any, str]:
    """The pyarrow filesystem for the warehouse root, and the path inside it.

    Built from :func:`storage_options` so an upload uses the same endpoint and
    credentials as every Delta write -- rather than a second client that would miss a
    MinIO endpoint or an explicit key and work only where the ambient AWS chain happens
    to. S3 with no key falls back to the standard chain (env, ``~/.aws``, or the
    instance role), which is how a pod with an assumed role authenticates.
    """
    import pyarrow.fs as pafs

    root = storage_uri()
    opts = storage_options()
    scheme, _, rest = root.partition("://")
    if scheme == "file":
        return pafs.LocalFileSystem(), root[len("file://") :]
    if scheme == "s3":
        kwargs: dict[str, Any] = {}
        if opts.get("AWS_ENDPOINT_URL"):
            endpoint = opts["AWS_ENDPOINT_URL"]
            kwargs["endpoint_override"] = endpoint
            if endpoint.startswith("http://"):
                kwargs["scheme"] = "http"
        for arg, opt in (
            ("access_key", "AWS_ACCESS_KEY_ID"),
            ("secret_key", "AWS_SECRET_ACCESS_KEY"),
            ("region", "AWS_REGION"),
        ):
            if opts.get(opt):
                kwargs[arg] = opts[opt]
        return pafs.S3FileSystem(**kwargs), rest
    if scheme in ("gs", "gcs"):
        # Application Default Credentials -- workload identity on GKE, or
        # GOOGLE_APPLICATION_CREDENTIALS pointing at a key file.
        return pafs.GcsFileSystem(), rest
    if scheme == "abfss":
        # abfss://container@account.dfs.core.windows.net/prefix
        container = rest.split("@", 1)[0]
        prefix = rest.split("/", 1)[1] if "/" in rest else ""
        return (
            pafs.AzureFileSystem(
                account_name=opts.get("AZURE_STORAGE_ACCOUNT_NAME", ""),
                account_key=opts.get("AZURE_STORAGE_ACCOUNT_KEY") or None,
            ),
            f"{container}/{prefix}".rstrip("/"),
        )
    raise ValueError(f"unsupported storage scheme for upload: {scheme}://")


def _delta_writable_type(dtype: pa.DataType) -> pa.DataType | None:
    """The Delta-writable form of ``dtype``, or ``None`` if it carries no writable data.

    A connector infers each batch's schema from the JSON it just received, so the schema
    describes *that page*, not the resource. Two inferred types cannot be written, and
    both come from a field the page happened to have nothing in:

    * ``null`` -- the field was None for every row in the batch. Delta has no Null type
      ("Invalid data type for Delta Lake: Null").
    * a struct with no fields -- the field was ``{}``, such as Stripe's ``metadata`` on
      an object carrying none. Parquet refuses it ("Parquet does not support writing
      empty structs").

    Both are dropped rather than coerced, because either way the page tells us nothing
    about the field's real type, and **guessing one is worse than having none**. Casting
    a null to string looks harmless and poisons the table: the next page that carries a
    real value of a different shape cannot merge into it, and the sync fails with
    "Unsupported CAST from Struct(...) to Struct(...)". Dropped, the field is simply
    absent until a page supplies its real type, at which point schema merge adds it --
    in either order, with earlier rows reading back as null. Nothing is lost that the
    data ever contained.

    Recursive, because a null or empty struct nested inside a struct or list is
    rejected exactly as a top-level one is; the error gains an "External error:"
    per level, which is its only outward sign. Dropping bubbles up: a struct left
    with no fields is itself unwritable, so it goes too.
    """
    if pa.types.is_null(dtype):
        return None
    if pa.types.is_struct(dtype):
        kept = [
            pa.field(field.name, writable, field.nullable)
            for field in dtype
            if (writable := _delta_writable_type(field.type)) is not None
        ]
        return pa.struct(kept) if kept else None
    if pa.types.is_large_list(dtype):
        value = _delta_writable_type(dtype.value_type)
        return pa.large_list(value) if value is not None else None
    if pa.types.is_list(dtype):
        value = _delta_writable_type(dtype.value_type)
        return pa.list_(value) if value is not None else None
    if pa.types.is_map(dtype):
        key = _delta_writable_type(dtype.key_type)
        item = _delta_writable_type(dtype.item_type)
        return pa.map_(key, item) if key is not None and item is not None else None
    return dtype


#: Encode every list-of-struct column as a JSON string instead of typed columns.
#: Off by default: it trades column-level access for a sync that cannot break on shape.
_JSON_LISTS_ENV = "WAREHOUSE_JSON_NESTED_LISTS"


def _json_nested_lists_enabled() -> bool:
    """Whether list-of-struct columns are stored as JSON text."""
    return (env(_JSON_LISTS_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


def _contains_a_struct_list(dtype: pa.DataType) -> bool:
    """Whether ``dtype`` holds a list of structs at any depth.

    The whole column is encoded when it does, not just the offending sub-field: Stripe
    nests the list one level down (``lines`` is a struct whose ``data`` is the list), so
    a rule that only looked at top-level list columns would miss the very case this
    exists for. Rebuilding one nested field inside a struct is also far more fiddly than
    encoding the column, for no gain -- a column holding an unmergeable list is not
    projectable in the part that matters either way.
    """
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return _holds_a_struct(dtype.value_type)
    if pa.types.is_struct(dtype):
        return any(_contains_a_struct_list(field.type) for field in dtype)
    return False


def _holds_a_struct(dtype: pa.DataType) -> bool:
    """Whether ``dtype`` is, or contains, a struct."""
    if pa.types.is_struct(dtype):
        return True
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return _holds_a_struct(dtype.value_type)
    return False


def _json_encode_struct_lists(data: pa.Table) -> pa.Table:
    """Replace each column holding a list of structs with its JSON text.

    A list of structs is where per-page schema inference stops being survivable. Delta
    merges a struct column that gains a field, and merges a list that gains one, but a
    list whose element struct *diverges between pages* while also gaining a list-typed
    field cannot be evolved:

        Unsupported CAST from Struct("data": List(Struct(...))) to Struct(...)

    Stripe's ``invoices`` does exactly this: a line item's ``parent`` differs depending
    on whether the line came from a subscription or an invoice item, and only some
    carry ``taxes``. No amount of type coercion reconciles two genuinely different
    shapes, so the choice is to store the shape faithfully or to store the *text*.

    This is what the managed-pipeline vendors settled on (Fivetran, Airbyte): a nested
    collection becomes one JSON column, and the consumer unpacks the parts it wants --
    DuckDB, Athena and Spark all read JSON text natively. The cost is real: no
    column-level projection or predicate pushdown into that field. That is why it is
    opt-in rather than the default, and why the trade belongs to whoever runs the
    warehouse rather than to this function.
    """
    import json

    names: list[str] = []
    columns: list[pa.Array | pa.ChunkedArray] = []
    encoded: list[str] = []
    for index, field in enumerate(data.schema):
        column = data.column(index)
        if not _contains_a_struct_list(field.type):
            names.append(field.name)
            columns.append(column)
            continue

        encoded.append(field.name)
        names.append(field.name)
        as_text = [
            None if value is None else json.dumps(value) for value in column.to_pylist()
        ]
        columns.append(pa.array(as_text, type=pa.string()))

    if not encoded:
        return data

    logger.info(
        "stored %d list-of-struct field(s) as JSON text (%s=1): %s",
        len(encoded),
        _JSON_LISTS_ENV,
        ", ".join(encoded),
    )
    return pa.Table.from_arrays(columns, names=names)


def _coerce_for_delta(data: pa.Table) -> pa.Table:
    """Drop the parts of an inferred Arrow table that Delta cannot store.

    See :func:`_delta_writable_type` for what goes and why.
    """
    if _json_nested_lists_enabled():
        # Before dropping, not after: a null inside an encoded column is representable
        # in JSON, and dropping it first would silently lose the key from the text.
        data = _json_encode_struct_lists(data)

    names: list[str] = []
    fields: list[pa.Field] = []
    dropped: list[str] = []
    for field in data.schema:
        writable = _delta_writable_type(field.type)
        if writable is None:
            dropped.append(field.name)
            continue
        names.append(field.name)
        fields.append(pa.field(field.name, writable, field.nullable))

    target = pa.schema(fields)
    if target == data.schema:
        return _json_encode_struct_lists(data) if _json_nested_lists_enabled() else data

    if dropped:
        # A column vanishing with no trace is its own bug report later. Say so here,
        # where the reason is still known, rather than leaving the absence to be
        # discovered downstream by a derivation that expected the field.
        logger.info(
            "dropped %d field(s) this batch carried no value for: %s",
            len(dropped),
            ", ".join(dropped),
        )
    return data.select(names).cast(target)


def write_arrow(table: str, data: pa.Table, *, mode: str = "append") -> str:
    """Write an Arrow table to its Delta table; return the table URI.

    ``mode="overwrite"`` replaces all rows and lets the schema change (full-refresh);
    ``mode="append"`` adds rows and merges new columns into the schema (incremental).
    The returned URI locates
    the table, so a freshly-synced table is immediately queryable.
    """
    from deltalake import write_deltalake

    uri = table_uri(table)
    if _is_local():
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    overwrite = mode == "overwrite"
    write_deltalake(
        uri,
        _coerce_for_delta(data),
        mode="overwrite" if overwrite else "append",
        schema_mode="overwrite" if overwrite else "merge",
        storage_options=storage_options() or None,
    )
    return uri


def table_location(table: str) -> str | None:
    """The Delta table URI if it exists, else ``None``."""
    from deltalake import DeltaTable

    uri = table_uri(table)
    if DeltaTable.is_deltatable(uri, storage_options=storage_options() or None):
        return uri
    return None


def drop_table(table: str) -> None:
    """Remove a table from the warehouse (its Parquet files and transaction log).

    Local storage is deleted directly; a cloud prefix is left for the operator to remove
    (delta-rs has no drop primitive), logged so it isn't silently orphaned.
    """
    if _is_local():
        shutil.rmtree(table_uri(table), ignore_errors=True)
        return
    logger.warning(
        "Warehouse table %r dropped from the catalog; remove its objects at %s "
        "manually (object stores have no delete-prefix primitive here).",
        table,
        table_uri(table),
    )
