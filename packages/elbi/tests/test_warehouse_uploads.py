"""Uploading a CSV/Parquet file into the warehouse, on local disk and object storage.

The gap this closes: a browser upload wrote only to ``file://`` storage, so the feature
was dead on every cloud (object-store) deployment. The upload now streams through the
same filesystem and credentials the warehouse uses for Delta writes, so it works on S3,
MinIO, GCS, and Azure as well as a local directory -- the server-side proxy pattern
Databricks and Superset both use.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pyarrow.fs as pafs
import pytest

from elbi.warehouse import storage, uploads
from elbi.warehouse.service import WarehouseError


@pytest.fixture
def local(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path}/warehouse")
    monkeypatch.delenv("WAREHOUSE_UPLOAD_MAX_BYTES", raising=False)
    return str(tmp_path)


# -- the filesystem mapping: right backend, right path, from storage_options -----
@pytest.mark.parametrize(
    ("uri", "expected_fs", "expected_base"),
    [
        ("s3://bucket/prefix", pafs.S3FileSystem, "bucket/prefix"),
        ("gs://bucket/prefix", pafs.GcsFileSystem, "bucket/prefix"),
        (
            "abfss://container@acct.dfs.core.windows.net/prefix",
            pafs.AzureFileSystem,
            "container/prefix",
        ),
    ],
)
def test_the_filesystem_matches_the_storage_scheme(
    uri: str,
    expected_fs: type,
    expected_base: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upload writes through the backend the warehouse root names -- so it lands
    beside the Delta tables rather than through a second client that guesses."""
    monkeypatch.setenv("STORAGE_URI", uri)
    # Azure needs an account to construct; the others build from the ambient chain.
    monkeypatch.setenv("STORAGE_AZURE_ACCOUNT", "acct")
    filesystem, base = storage._warehouse_filesystem()
    assert isinstance(filesystem, expected_fs)
    assert base == expected_base


def test_a_minio_endpoint_and_key_reach_the_s3_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trap this guards: a naive client would miss the MinIO endpoint and explicit
    key and work only where the ambient AWS chain happens to. Construction has to accept
    them without error."""
    monkeypatch.setenv("STORAGE_URI", "s3://bucket/prefix")
    monkeypatch.setenv("STORAGE_ENDPOINT", "http://minio.local:9000")
    monkeypatch.setenv("STORAGE_KEY", "minioadmin")
    monkeypatch.setenv("STORAGE_SECRET", "minioadmin")
    monkeypatch.setenv("STORAGE_REGION", "us-east-1")
    filesystem, base = storage._warehouse_filesystem()
    assert isinstance(filesystem, pafs.S3FileSystem)
    assert base == "bucket/prefix"


# -- the local path --------------------------------------------------------------
def test_a_csv_is_stored_and_returned_by_uri(local: str) -> None:
    uri = uploads.save_upload("data.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"))
    assert uri.endswith("data.csv")
    assert Path(uri).read_text() == "a,b\n1,2\n3,4\n"


def test_an_empty_upload_is_refused(local: str) -> None:
    with pytest.raises(WarehouseError, match="empty"):
        uploads.save_upload("empty.csv", io.BytesIO(b""))


def test_a_mislabeled_file_is_refused_by_reading_it(local: str) -> None:
    """The extension says parquet; the bytes are not. Caught now, not at sync time."""
    with pytest.raises(WarehouseError, match="could not be read"):
        uploads.save_upload("fake.parquet", io.BytesIO(b"not parquet"))


def test_an_unsupported_type_is_refused(local: str) -> None:
    with pytest.raises(WarehouseError, match="unsupported file type"):
        uploads.save_upload("notes.txt", io.BytesIO(b"hello"))


# -- the size cap ----------------------------------------------------------------
def test_the_cap_is_enforced_while_writing(
    local: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WAREHOUSE_UPLOAD_MAX_BYTES", str(16))
    with pytest.raises(WarehouseError, match="upload limit"):
        uploads.save_upload("big.csv", io.BytesIO(b"a,b\n" + b"1,2\n" * 100))


def test_a_malformed_cap_is_a_clear_error_not_a_silent_default(
    local: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WAREHOUSE_UPLOAD_MAX_BYTES", "lots")
    with pytest.raises(WarehouseError, match="positive integer"):
        uploads.save_upload("data.csv", io.BytesIO(b"a,b\n1,2\n"))


# -- the object-store path, with an in-memory store that has S3 semantics ---------
class _Blob(io.BytesIO):
    """A writable stream that publishes its bytes to the store when closed."""

    def __init__(self, store: dict[str, bytes], path: str) -> None:
        super().__init__()
        self._store = store
        self._path = path

    def close(self) -> None:
        if not self.closed:
            self._store[self._path] = self.getvalue()
        super().close()


class _FakeObjectStore:
    """An object store: a flat key space with no directories, the way S3 behaves.

    The point of the fake is fidelity to the trait that broke the old code -- an object
    store needs no parent directory -- which ``pyarrow``'s hierarchical mock does not
    model. ``open_input_file`` returns a plain seekable stream the CSV and Parquet
    readers accept.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def open_output_stream(self, path: str) -> _Blob:
        return _Blob(self.objects, path)

    def open_input_file(self, path: str) -> io.BytesIO:
        return io.BytesIO(self.objects[path])

    def delete_file(self, path: str) -> None:
        self.objects.pop(path, None)


def _use_object_store(monkeypatch: pytest.MonkeyPatch) -> _FakeObjectStore:
    fake = _FakeObjectStore()
    monkeypatch.setenv("STORAGE_URI", "s3://bucket/warehouse")
    monkeypatch.delenv("WAREHOUSE_UPLOAD_MAX_BYTES", raising=False)
    monkeypatch.setattr(
        storage, "_warehouse_filesystem", lambda: (fake, "bucket/warehouse")
    )
    return fake


def test_an_upload_streams_to_object_storage_and_returns_the_s3_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_object_store(monkeypatch)
    uri = uploads.save_upload("data.csv", io.BytesIO(b"a,b\n1,2\n"))
    assert uri.startswith("s3://bucket/warehouse/_uploads/")
    assert uri.endswith("/data.csv")
    # the bytes actually landed in the store, under the key the URI names
    key = uri[len("s3://") :]
    assert fake.objects[key] == b"a,b\n1,2\n"


def test_a_bad_object_upload_is_deleted_not_left_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mislabeled upload must not linger in the bucket for a later sync."""
    fake = _use_object_store(monkeypatch)
    with pytest.raises(WarehouseError, match="could not be read"):
        uploads.save_upload("fake.parquet", io.BytesIO(b"not parquet"))
    assert fake.objects == {}  # cleaned up
