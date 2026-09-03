"""The object-storage connectors: S3, Google Cloud Storage, Azure Blob and SFTP.

Two layers. The first drives the shared engine over a local directory, so globbing,
grouping, format detection, provenance and the incremental cursor are covered on every
run with no services involved. The second drives the real S3 and SFTP connectors against
MinIO and an SFTP server in Docker, because the credential and protocol handling is
exactly the part a local filesystem cannot exercise; those skip when the containers are
absent.

The Docker halves are what caught the bug worth remembering: ``_root`` stripped the
leading slash from the container, which is right for a bucket name and destroys an
absolute SFTP path.
"""

from __future__ import annotations

import io
import json
import socket
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("fsspec")

from elbi.warehouse.config import SourceField
from elbi.warehouse.sources.base import SourceInputs
from elbi.warehouse.sources.object_store import (
    FILE_MODIFIED,
    FILE_PATH,
    _after,
    _modified,
    _ObjectStoreSource,
    format_for,
)
from elbi.warehouse.sources.registry import SourceRegistry

MINIO_ENDPOINT = "http://127.0.0.1:19000"
MINIO_KEY = "elbitest"
MINIO_SECRET = "elbitest123"
SFTP_PORT = 12222


class _LocalStore(_ObjectStoreSource):
    """The shared engine over a plain directory, driven without any service."""

    protocol = "file"
    name = "local_test_store"
    label = "Local directory"
    icon = "L"
    container_is_path = True

    def _credential_fields(self) -> list[SourceField]:
        return []

    def _storage_options(self, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {}


@pytest.fixture
def lake(tmp_path: Path) -> Path:
    """Laid out the way a lakehouse is: folders as tables, plus a loose file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "events").mkdir()
    (tmp_path / "users").mkdir()
    (tmp_path / "orders").mkdir()
    (tmp_path / "events" / "day1.csv").write_text("id,amount\n1,10\n2,20\n")
    (tmp_path / "events" / "day2.csv").write_text("id,amount\n3,30\n")
    (tmp_path / "users" / "u.jsonl").write_text(
        '{"id":1,"name":"ada"}\n{"id":2,"name":"bo"}\n'
    )
    pq.write_table(
        pa.table({"sku": ["a", "b", "c"], "qty": [1, 2, 3]}),
        tmp_path / "orders" / "part-0.parquet",
    )
    (tmp_path / "top.csv").write_text("x\n7\n")
    return tmp_path


def _config(lake: Path, **over: Any) -> dict[str, Any]:
    return {"container": str(lake), "pattern": "**", "format": "auto", **over}


def _rows(source: Any, config: dict[str, Any], schema: str, **kw: Any) -> list[dict]:
    tables = list(source.extract(SourceInputs(config=config, schema=schema, **kw)))
    return [row for table in tables for row in table.to_pylist()]


# --- the shared engine ------------------------------------------------------------


def test_a_folder_becomes_a_table_and_loose_files_become_root(lake: Path) -> None:
    names = [s.name for s in _LocalStore().schemas(_config(lake))]
    assert names == ["events", "orders", "root", "users"]


def test_every_table_offers_the_file_time_as_its_cursor(lake: Path) -> None:
    for schema in _LocalStore().schemas(_config(lake)):
        assert schema.incremental_fields == [FILE_MODIFIED]


@pytest.mark.parametrize(
    ("schema", "rows", "column"),
    [("events", 3, "amount"), ("users", 2, "name"), ("orders", 3, "sku")],
)
def test_each_format_is_read_by_its_extension(
    lake: Path, schema: str, rows: int, column: str
) -> None:
    """csv, jsonl and parquet, chosen per file, as a mixed container needs."""
    got = _rows(_LocalStore(), _config(lake), schema)
    assert len(got) == rows
    assert column in got[0]


def test_the_files_of_one_table_are_read_together(lake: Path) -> None:
    got = _rows(_LocalStore(), _config(lake), "events")
    assert sorted(r["id"] for r in got) == [1, 2, 3]
    assert {r[FILE_PATH] for r in got} == {"events/day1.csv", "events/day2.csv"}


def test_rows_carry_the_file_they_came_from(lake: Path) -> None:
    """Provenance is the point: a wrong row has to be traceable back to one file."""
    got = _rows(_LocalStore(), _config(lake), "orders")
    assert all(r[FILE_PATH] == "orders/part-0.parquet" for r in got)
    assert all(isinstance(r[FILE_MODIFIED], datetime) for r in got)


def test_the_pattern_narrows_what_is_synced(lake: Path) -> None:
    names = [s.name for s in _LocalStore().schemas(_config(lake, pattern="**/*.jsonl"))]
    assert names == ["users"]


def test_the_prefix_reroots_the_container(lake: Path) -> None:
    """With the prefix at a folder, its files are loose, so they land in root."""
    source = _LocalStore()
    config = _config(lake, prefix="events")
    assert [s.name for s in source.schemas(config)] == ["root"]
    assert len(_rows(source, config, "root")) == 3


def test_a_chosen_format_overrides_the_extension(lake: Path) -> None:
    """A file whose name does not say what it is still has to be readable."""
    (lake / "events" / "day3.data").write_text("id,amount\n4,40\n")
    got = _rows(
        _LocalStore(), _config(lake, pattern="events/*", format="csv"), "events"
    )
    assert sorted(r["id"] for r in got) == [1, 2, 3, 4]


def test_a_file_with_no_readable_format_is_skipped_not_fatal(lake: Path) -> None:
    (lake / "events" / "notes.rtf").write_text("not data")
    got = _rows(_LocalStore(), _config(lake), "events")
    assert sorted(r["id"] for r in got) == [1, 2, 3]


def test_the_empty_folder_markers_a_console_leaves_behind_are_ignored(
    lake: Path,
) -> None:
    """A zero-byte key is how a store fakes an empty folder; it holds no rows."""
    (lake / "events" / "placeholder.csv").write_bytes(b"")
    assert len(_rows(_LocalStore(), _config(lake), "events")) == 3


# --- the incremental cursor -------------------------------------------------------


def test_a_cursor_in_the_future_yields_nothing(lake: Path) -> None:
    got = _rows(
        _LocalStore(),
        _config(lake),
        "events",
        incremental_field=FILE_MODIFIED,
        incremental_since=datetime.now(timezone.utc),
    )
    assert got == []


def test_a_cursor_in_the_past_yields_everything(lake: Path) -> None:
    got = _rows(
        _LocalStore(),
        _config(lake),
        "events",
        incremental_field=FILE_MODIFIED,
        incremental_since=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )
    assert len(got) == 3


def test_only_files_newer_than_the_cursor_are_re_read(lake: Path) -> None:
    """What the design exists for: one new file, one file's worth of rows."""
    import os
    import time

    source = _LocalStore()
    config = _config(lake)
    old = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    for name in ("day1.csv", "day2.csv"):
        os.utime(lake / "events" / name, (old, old))
    cursor = datetime(2021, 1, 1, tzinfo=timezone.utc)

    assert (
        _rows(
            source,
            config,
            "events",
            incremental_field=FILE_MODIFIED,
            incremental_since=cursor,
        )
        == []
    )

    (lake / "events" / "day3.csv").write_text("id,amount\n9,90\n")
    time.sleep(0.01)
    fresh = _rows(
        source,
        config,
        "events",
        incremental_field=FILE_MODIFIED,
        incremental_since=cursor,
    )
    assert [r["id"] for r in fresh] == [9]


def test_a_cursor_that_came_back_as_a_string_still_orders_correctly() -> None:
    """It round-trips through the database, so it need not return a datetime."""
    modified = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert _after(modified, "2026-01-01T00:00:00Z") is True
    assert _after(modified, "2026-12-01T00:00:00+00:00") is False


def test_an_unreadable_cursor_re_reads_rather_than_drops() -> None:
    """Duplicating a file is recoverable; silently skipping one is not."""
    modified = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert _after(modified, "not a date") is True
    assert _after(modified, None) is True
    assert _after(modified, 12345) is True


def test_a_naive_cursor_is_read_as_utc_not_rejected() -> None:
    modified = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert _after(modified, datetime(2026, 1, 1)) is True
    assert _after(modified, datetime(2026, 12, 1)) is False


@pytest.mark.parametrize(
    "info",
    [
        {"LastModified": datetime(2026, 5, 4, tzinfo=timezone.utc)},
        {"last_modified": datetime(2026, 5, 4, tzinfo=timezone.utc)},
        {"mtime": datetime(2026, 5, 4, tzinfo=timezone.utc).timestamp()},
        {"modification_time": "2026-05-04T00:00:00Z"},
    ],
)
def test_each_providers_timestamp_key_is_understood(info: dict[str, Any]) -> None:
    """Providers disagree on both the key and the type; the cursor cannot."""
    assert _modified(info) == datetime(2026, 5, 4, tzinfo=timezone.utc)


def test_a_file_with_no_timestamp_reads_as_always_new() -> None:
    """Epoch, so it syncs every time -- the opposite failure would never sync it."""
    assert _modified({"size": 1}) == datetime.fromtimestamp(0, tz=timezone.utc)


def test_a_naive_provider_timestamp_is_read_as_utc() -> None:
    assert _modified({"mtime": datetime(2026, 5, 4)}) == datetime(
        2026, 5, 4, tzinfo=timezone.utc
    )


# --- format detection -------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.csv", "csv"),
        ("a.tsv", "csv"),
        ("a.jsonl", "jsonl"),
        ("a.ndjson", "jsonl"),
        ("a.json", "jsonl"),
        ("a.parquet", "parquet"),
        ("a.pq", "parquet"),
        ("nested/dir/a.csv", "csv"),
        ("a.csv.gz", "csv"),
        ("a.parquet.zst", "parquet"),
        ("a.rtf", None),
        ("noextension", None),
    ],
)
def test_the_extension_decides_the_format(path: str, expected: str | None) -> None:
    assert format_for(path) == expected


# --- the provider catalog entries -------------------------------------------------


@pytest.mark.parametrize(
    ("source_type", "label"),
    [
        ("s3", "Amazon S3"),
        ("gcs", "Google Cloud Storage"),
        ("azure_blob", "Azure Blob Storage"),
        ("sftp", "SFTP"),
    ],
)
def test_every_provider_is_catalogued_under_file_storage(
    source_type: str, label: str
) -> None:
    config = SourceRegistry.get(source_type).config
    assert config.label == label
    assert config.category == "File storage"
    assert {"container", "prefix", "pattern", "format"} <= {
        f.name for f in config.fields
    }


def test_a_bucket_name_loses_a_pasted_slash_but_a_path_keeps_its_root() -> None:
    """The bug the Docker tests caught: one of these is a name, the other an address."""
    s3 = SourceRegistry.get("s3")
    sftp = SourceRegistry.get("sftp")
    assert s3._root({"container": "/lake/"}) == "lake"  # type: ignore[attr-defined]
    assert sftp._root({"container": "/exports/"}) == "/exports"  # type: ignore[attr-defined]


def test_s3_sends_an_endpoint_only_when_one_is_given() -> None:
    """An endpoint is how the connector reaches MinIO or R2; empty means AWS itself."""
    s3 = SourceRegistry.get("s3")
    plain = s3._storage_options({"access_key_id": "k", "secret_access_key": "s"})  # type: ignore[attr-defined]
    assert plain == {"key": "k", "secret": "s"}
    routed = s3._storage_options(  # type: ignore[attr-defined]
        {"access_key_id": "k", "secret_access_key": "s", "endpoint": "http://m:9000"}
    )
    assert routed["client_kwargs"] == {"endpoint_url": "http://m:9000"}


def test_s3_without_keys_defers_to_whatever_the_host_already_has() -> None:
    """An instance role beats a pasted key, so empty must not mean anonymous."""
    assert SourceRegistry.get("s3")._storage_options({}) == {}  # type: ignore[attr-defined]


def test_gcs_takes_the_service_account_key_as_its_token() -> None:
    key = {"type": "service_account", "project_id": "p"}
    options = SourceRegistry.get("gcs")._storage_options({"key_file": json.dumps(key)})  # type: ignore[attr-defined]
    assert options == {"token": key}


def test_a_malformed_gcs_key_says_so_rather_than_failing_deeper_in() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        SourceRegistry.get("gcs")._storage_options({"key_file": "{oops"})  # type: ignore[attr-defined]


def test_azure_accepts_either_credential_and_prefers_the_account_key() -> None:
    azure = SourceRegistry.get("azure_blob")
    both = azure._storage_options(  # type: ignore[attr-defined]
        {"account_name": "acct", "account_key": "k", "sas_token": "s"}
    )
    assert both == {"account_name": "acct", "account_key": "k"}
    sas_only = azure._storage_options({"account_name": "acct", "sas_token": "s"})  # type: ignore[attr-defined]
    assert sas_only == {"account_name": "acct", "sas_token": "s"}


def test_azure_with_neither_credential_is_refused_before_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ok, errors = SourceRegistry.get("azure_blob").validate(
        {"container": "c", "account_name": "acct"}
    )
    assert not ok
    assert "account key or a SAS token" in " ".join(errors)


def test_sftp_with_neither_credential_is_refused_before_a_connection() -> None:
    ok, errors = SourceRegistry.get("sftp").validate(
        {"container": "/exports", "host": "h", "username": "u"}
    )
    assert not ok
    assert "password or a private key" in " ".join(errors)


def test_sftp_never_falls_back_to_the_hosts_own_ssh_credentials() -> None:
    """A connector authenticates as what the user configured, not as whoever runs it.

    Left to its defaults paramiko would also try the host's ssh-agent and the private
    keys in its home directory, so a source with a wrong password could still connect --
    as the server's own account. The agent socket it opens for that is never closed
    either, which is how this was found.
    """
    options = SourceRegistry.get("sftp")._storage_options(  # type: ignore[attr-defined]
        {"host": "h", "username": "u", "password": "p"}
    )
    assert options["allow_agent"] is False
    assert options["look_for_keys"] is False


def test_an_unreadable_private_key_reports_what_each_loader_said() -> None:
    """ "Could not be read" leaves nothing to act on; the reasons say which."""
    from elbi.warehouse.sources.object_store import _private_key

    with pytest.raises(ValueError, match="Ed25519") as raised:
        _private_key("-----BEGIN OPENSSH PRIVATE KEY-----\nnonsense\n")
    assert "Ed25519Key:" in str(raised.value)


# --- against real services --------------------------------------------------------


def _minio_client() -> Any:
    boto3 = pytest.importorskip("boto3")
    client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_KEY,
        aws_secret_access_key=MINIO_SECRET,
        region_name="us-east-1",
    )
    try:
        client.list_buckets()
    except Exception:
        pytest.skip(f"no MinIO at {MINIO_ENDPOINT}; start one to run the live S3 tests")
    return client


@pytest.fixture
def minio_bucket() -> Any:
    """A bucket holding one folder per format, seeded fresh for this test."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    client = _minio_client()
    bucket = "elbi-live-test"
    try:
        client.create_bucket(Bucket=bucket)
    except Exception:
        existing = client.list_objects_v2(Bucket=bucket).get("Contents", [])
        for obj in existing:
            client.delete_object(Bucket=bucket, Key=obj["Key"])
    client.put_object(
        Bucket=bucket, Key="events/day1.csv", Body=b"id,amount\n1,10\n2,20\n"
    )
    client.put_object(Bucket=bucket, Key="events/day2.csv", Body=b"id,amount\n3,30\n")
    client.put_object(
        Bucket=bucket,
        Key="users/u.jsonl",
        Body=b'{"id":1,"name":"ada"}\n{"id":2,"name":"bo"}\n',
    )
    buf = io.BytesIO()
    pq.write_table(pa.table({"sku": ["a", "b"], "qty": [1, 2]}), buf)
    client.put_object(Bucket=bucket, Key="orders/part-0.parquet", Body=buf.getvalue())
    client.put_object(Bucket=bucket, Key="top.csv", Body=b"x\n7\n")
    return bucket


def _s3_config(bucket: str) -> dict[str, Any]:
    return {
        "container": bucket,
        "pattern": "**",
        "format": "auto",
        "access_key_id": MINIO_KEY,
        "secret_access_key": MINIO_SECRET,
        "region": "us-east-1",
        "endpoint": MINIO_ENDPOINT,
    }


def test_live_s3_lists_groups_and_reads_every_format(minio_bucket: str) -> None:
    """The whole path over a real S3 API: signing, listing, three readers."""
    source = SourceRegistry.get("s3")
    config = _s3_config(minio_bucket)
    assert source.validate(config) == (True, [])
    assert [s.name for s in source.schemas(config)] == [
        "events",
        "orders",
        "root",
        "users",
    ]
    assert sorted(r["id"] for r in _rows(source, config, "events")) == [1, 2, 3]
    assert sorted(r["name"] for r in _rows(source, config, "users")) == ["ada", "bo"]
    assert sorted(r["sku"] for r in _rows(source, config, "orders")) == ["a", "b"]
    assert [r["x"] for r in _rows(source, config, "root")] == [7]


def test_live_s3_rejects_a_wrong_secret(minio_bucket: str) -> None:
    ok, errors = SourceRegistry.get("s3").validate(
        {**_s3_config(minio_bucket), "secret_access_key": "wrong"}
    )
    assert not ok
    assert errors


def test_live_s3_incremental_picks_up_only_the_new_object(minio_bucket: str) -> None:
    source = SourceRegistry.get("s3")
    config = _s3_config(minio_bucket)
    before = datetime.now(timezone.utc)
    assert (
        _rows(
            source,
            config,
            "events",
            incremental_field=FILE_MODIFIED,
            incremental_since=before,
        )
        == []
    )
    _minio_client().put_object(
        Bucket=minio_bucket, Key="events/day3.csv", Body=b"id,amount\n9,90\n"
    )
    fresh = _rows(
        source,
        config,
        "events",
        incremental_field=FILE_MODIFIED,
        incremental_since=before,
    )
    assert [r["id"] for r in fresh] == [9]


@pytest.fixture
def sftp_folder() -> str:
    """A folder on the SFTP server, seeded fresh for this test."""
    paramiko = pytest.importorskip("paramiko")

    # Checked with a plain socket first, and closed. paramiko's Transport constructor
    # opens its own socket and connects; when nothing is listening it raises with that
    # socket still open, and nothing is left holding a reference to close it. The
    # garbage collector reports it later, during whichever unrelated test happens to be
    # running -- which is how a leak here becomes a failure somewhere else entirely.
    probe = socket.socket()
    probe.settimeout(3)
    try:
        probe.connect(("127.0.0.1", SFTP_PORT))
    except OSError:
        pytest.skip(f"no SFTP server on port {SFTP_PORT}; start one for the live tests")
    finally:
        probe.close()

    transport = paramiko.Transport(("127.0.0.1", SFTP_PORT))
    try:
        transport.connect(username="elbi", password="elbipass")
    except Exception:
        transport.close()
        pytest.skip(f"no SFTP server on port {SFTP_PORT}; start one for the live tests")
    client = paramiko.SFTPClient.from_transport(transport)
    assert client is not None
    base = "/exports"
    for path in (f"{base}/sales",):
        with suppress(OSError):
            # Already there from an earlier run, which is fine.
            client.mkdir(path)
    for name in client.listdir(f"{base}/sales"):
        client.remove(f"{base}/sales/{name}")
    client.putfo(io.BytesIO(b"region,total\neu,5\nus,9\n"), f"{base}/sales/q1.csv")
    client.putfo(io.BytesIO(b"region,total\napac,4\n"), f"{base}/sales/q2.csv")
    client.close()
    transport.close()
    return base


def test_live_sftp_reads_an_absolute_folder_path(sftp_folder: str) -> None:
    """The regression test for the stripped leading slash."""
    source = SourceRegistry.get("sftp")
    config = {
        "container": sftp_folder,
        "pattern": "**",
        "format": "auto",
        "host": "127.0.0.1",
        "port": SFTP_PORT,
        "username": "elbi",
        "password": "elbipass",
    }
    assert source.validate(config) == (True, [])
    assert "sales" in [s.name for s in source.schemas(config)]
    rows = _rows(source, config, "sales")
    assert sorted(r["region"] for r in rows) == ["apac", "eu", "us"]


def test_live_sftp_rejects_a_wrong_password(sftp_folder: str) -> None:
    ok, _ = SourceRegistry.get("sftp").validate(
        {
            "container": sftp_folder,
            "host": "127.0.0.1",
            "port": SFTP_PORT,
            "username": "elbi",
            "password": "nope",
        }
    )
    assert not ok
