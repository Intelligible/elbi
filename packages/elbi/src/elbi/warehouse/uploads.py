"""Storing uploaded CSV / Parquet files for the file connector.

The CSV / Parquet source reads a path or object-store URL. This lets a user upload a
local file instead: the bytes are streamed into the warehouse storage root (under an
``_uploads/`` prefix) and the stored URI is handed back to be used as the source's
``path``. Re-syncs re-read that stored file, so the upload is the durable source of
truth, not a one-shot import.

The upload goes to whatever the warehouse root is -- a local directory or the object
store behind it (S3, MinIO, GCS, Azure) -- through the same filesystem and credentials
the warehouse uses for Delta writes. So the browser (or the CLI) sends the bytes and the
server writes them, which is the server-side proxy pattern Databricks and Superset use:
it works whether or not the client can reach the object store, the common case for a
private or air-gapped deployment.
"""

from __future__ import annotations

import contextlib
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

from ..env import env
from .service import WarehouseError
from .storage import storage_uri, warehouse_object


class BinaryReader(Protocol):
    """A blocking binary stream (e.g. an upload's spooled temp file)."""

    def read(self, size: int = ..., /) -> bytes:
        """Read up to ``size`` bytes; an empty result signals end of stream."""
        ...


#: Bytes read from the stream per write, so a large upload never lands in memory whole.
_CHUNK = 1024 * 1024

#: File types the connector can read, so an upload of anything else is refused up front
#: rather than failing later at sync time.
_SUPPORTED_SUFFIXES = (".csv", ".parquet")

#: The default upload ceiling; the operator can raise it (see _max_upload_bytes).
_DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _max_upload_bytes() -> int:
    """The upload size ceiling, in bytes, from ``WAREHOUSE_UPLOAD_MAX_BYTES``.

    A cap is protection, not metering, so it stays even though the operator owns the
    hardware: on a shared deployment one unbounded upload holds a worker and its
    bandwidth for the whole transfer, and the reverse proxy in front (nginx defaults to
    1 MiB) refuses an oversized body before it even arrives. Past the cap the right move
    is to land the data in the bucket and point a source at it -- the same thing
    Databricks' 5 GiB UI limit steers you to. Configurable because it is the operator's
    infrastructure, which is how Superset (``UPLOAD_MAX_BYTES``) and the rest do it.
    """
    raw = (env("WAREHOUSE_UPLOAD_MAX_BYTES") or "").strip()
    if not raw:
        return _DEFAULT_MAX_UPLOAD_BYTES
    if not raw.isdigit() or int(raw) == 0:
        raise WarehouseError(
            "WAREHOUSE_UPLOAD_MAX_BYTES must be a positive integer number of bytes"
        )
    return int(raw)


def _human(size: int) -> str:
    """A byte count as the nearest whole unit, for a limit message a person reads."""
    for unit in ("GiB", "MiB", "KiB"):
        step = {"GiB": 1024**3, "MiB": 1024**2, "KiB": 1024}[unit]
        if size >= step:
            return f"{size // step} {unit}"
    return f"{size} bytes"


def _safe_name(filename: str) -> str:
    """A filesystem-safe basename, preserving a supported extension."""
    base = Path(filename or "").name
    suffix = Path(base).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise WarehouseError(
            f"unsupported file type {suffix or '(none)'!r}; upload a "
            f"{' or '.join(_SUPPORTED_SUFFIXES)} file"
        )
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("_") or "upload"
    return f"{stem}{suffix}"


def save_upload(filename: str, reader: BinaryReader) -> str:
    """Write an uploaded file into the warehouse and return its stored URI.

    Streams the upload to the warehouse root -- a local directory or an object
    store -- a chunk at a time, never buffering it whole; on an object store pyarrow
    uploads it in parts underneath. It lands under a unique ``_uploads/`` key so two
    uploads of the same name cannot collide, the size cap is enforced while writing, and
    a partial, empty, or unreadable upload is deleted rather than left to be taken for
    a good one. The returned URI is what a CSV/Parquet source's ``path`` field takes.
    """
    safe = _safe_name(filename)
    key = f"_uploads/{uuid.uuid4().hex}/{safe}"
    filesystem, path, uri = warehouse_object(key)
    if storage_uri().startswith("file://"):
        # Local pyarrow does not create parent directories; an object store needs none.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    limit = _max_upload_bytes()
    written = 0
    try:
        with filesystem.open_output_stream(path) as out:
            while chunk := reader.read(_CHUNK):
                written += len(chunk)
                if written > limit:
                    raise WarehouseError(
                        f"file exceeds the {_human(limit)} upload limit"
                    )
                out.write(chunk)
        if written == 0:
            raise WarehouseError("the uploaded file is empty")
        # Validate by actually reading it back, not by trusting the extension or the
        # client's content type (both spoofable): confirm the connector's own reader can
        # parse it, so a mislabeled or corrupt file is rejected now with a clear message
        # rather than failing later at sync time.
        _validate_readable(filesystem, path, safe)
    except Exception:
        # Never leave a partial, empty, or unreadable upload behind. A cleanup that
        # cannot find the object (nothing was flushed) must not mask the real error.
        with contextlib.suppress(OSError):
            filesystem.delete_file(path)
        raise
    return uri


def _validate_readable(filesystem: Any, path: str, name: str) -> None:
    """Confirm the stored object parses as the CSV/Parquet it claims to be, or raise.

    Cheap by design: Parquet is validated from its footer alone and CSV by reading
    only the first block, so validating a multi-gigabyte upload does not re-read the
    whole file. Reads through the filesystem, so it works on an object store as well as
    on local disk.
    """
    suffix = Path(name).suffix.lower()
    try:
        with filesystem.open_input_file(path) as source:
            if suffix == ".parquet":
                import pyarrow.parquet as pq

                pq.read_metadata(source)
            else:
                from . import csv_dialect

                # The same dialect detection the connector uses at sync, so a file that
                # validates is one the sync can read.
                with csv_dialect.open_reader(source) as reader:
                    reader.read_next_batch()
    except StopIteration:
        raise WarehouseError("the CSV file has no data rows") from None
    except WarehouseError:
        raise
    except Exception as exc:
        raise WarehouseError(
            f"the file could not be read as {suffix.lstrip('.').upper()}: {exc}"
        ) from exc
