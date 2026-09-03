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
        data,
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
