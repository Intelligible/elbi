"""Object-storage connectors: S3, Google Cloud Storage, Azure Blob, and SFTP.

One shared engine behind a connector per provider. The engine lists a container by glob,
groups what it finds into tables, reads each file by its format, and appends two columns
recording where the row came from and when that file last changed. A provider subclass
supplies only the two things that genuinely differ: the fsspec protocol it speaks, and
the credentials it asks for.

Every provider is reached through fsspec rather than its own SDK, which is what lets the
listing, grouping, format handling and incremental logic be written once. It also
means a new provider is a subclass and a filesystem package, not another copy of this
file.

Two decisions here are worth recording.

Files are read in batches and yielded per file rather than concatenated, so a container
holding more data than memory still syncs. The cost is that two files with different
columns arrive as differently-shaped batches; the Delta writer merges those into one
schema, which is the same thing it does for an incremental append.

The incremental cursor is the file's modification time, not a column inside the data.
Object storage has no transaction log to read, so the only thing that reliably orders
arrivals is the store's own timestamp. A file rewritten in place is therefore picked up
again, and a file backdated on upload is missed -- both are properties of the store, not
of this code, and no connector reading a bucket can do better.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, Source, SourceInputs
from .registry import SourceRegistry

#: Rows carry the file they came from and when that file last changed. The second is the
#: incremental cursor; the first is what makes a wrong row traceable back to a file.
FILE_PATH = "_file_path"
FILE_MODIFIED = "_file_modified"

#: Read in slices rather than whole files, so one large object cannot exhaust memory.
_BATCH = 50_000

_FORMATS = ("auto", "csv", "jsonl", "parquet")

_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".json": "jsonl",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
}

#: Compression fsspec unwraps transparently, so the format is decided by the extension
#: underneath. Listed rather than stripped blindly, because an unknown trailing suffix
#: should stay part of the name and fail loudly instead of being guessed at.
_COMPRESSION = (".gz", ".bz2", ".zst", ".zstd", ".xz")


def _table_name(value: str) -> str:
    """A schema name safe to carry into a Delta table name."""
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "root"


def format_for(path: str) -> str | None:
    """The format implied by ``path``'s extension, or ``None`` if it implies none."""
    name = path
    for suffix in _COMPRESSION:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    _, _, ext = name.lower().rpartition(".")
    return _EXTENSIONS.get(f".{ext}") if ext else None


class _ObjectStoreSource(SimpleSource):
    """Shared object-store engine; a provider subclass fixes protocol and secrets."""

    #: The fsspec protocol this provider speaks (``s3``, ``gcs``, ``abfs``, ``sftp``).
    protocol: str = ""
    name: str = ""
    label: str = ""
    icon: str = ""
    #: What the provider calls the thing that holds files, for the form's first field.
    container_label = "Bucket"
    container_placeholder = "my-bucket"
    #: Whether the container is a filesystem path rather than a bucket name. A bucket
    #: name never begins with a slash, so a leading one there is a paste artefact worth
    #: dropping; on a path that slash is the root, and dropping it breaks the address.
    container_is_path = False
    #: Whether this provider's filesystem holds a connection that has to be closed. The
    #: HTTP-backed stores pool their own sockets and are safe to cache and share; an SSH
    #: session is neither, and fsspec's SFTP filesystem offers no close of its own, so
    #: one is opened per operation here and shut down below.
    holds_connection = False

    @property
    def source_type(self) -> str:
        """The registry key."""
        return self.name

    def _credential_fields(self) -> list[SourceField]:
        """The provider's own credential inputs."""
        raise NotImplementedError

    def _storage_options(self, config: dict[str, Any]) -> dict[str, Any]:
        """Options passed to fsspec, built from the saved credentials."""
        raise NotImplementedError

    @property
    def caption(self) -> str:
        """The one-line description shown on the catalog tile."""
        return f"Sync CSV, JSON and Parquet files from {self.label}."

    @property
    def docs_url(self) -> str:
        """Where to read about the credentials this provider wants."""
        return ""

    @property
    def config(self) -> SourceConfig:
        """A container, what to match inside it, and the provider's credentials."""
        return SourceConfig(
            name=self.name,
            label=self.label,
            category="File storage",
            icon=self.icon,
            caption=self.caption,
            docs_url=self.docs_url,
            fields=[
                SourceField(
                    name="container",
                    label=self.container_label,
                    placeholder=self.container_placeholder,
                ),
                SourceField(
                    name="prefix",
                    label="Prefix",
                    required=False,
                    placeholder="exports/",
                    caption="Limit the sync to one folder. Leave empty for the whole "
                    f"{self.container_label.lower()}.",
                ),
                SourceField(
                    name="pattern",
                    label="File pattern",
                    required=False,
                    default="**",
                    placeholder="**",
                    caption="A glob matched against the path below the prefix. `**` "
                    "takes everything; `*.parquet` takes one format; "
                    "`events/*.csv` takes one folder.",
                ),
                SourceField(
                    name="format",
                    label="Format",
                    type="select",
                    required=False,
                    default="auto",
                    options=[{"value": f, "label": f} for f in _FORMATS],
                    caption="`auto` reads each file according to its extension, which "
                    "is what a mixed folder needs. Set a format to read files "
                    "whose names do not say what they are.",
                ),
                *self._credential_fields(),
            ],
        )

    @contextmanager
    def _open(self, config: dict[str, Any]) -> Iterator[AbstractFileSystem]:
        """The provider's filesystem, closed afterwards if it holds a connection.

        fsspec caches filesystem instances, which is what makes the HTTP-backed stores
        cheap to re-enter. A connection-holding provider opts out of that cache, because
        a shared instance is one nothing can decide it is finished with.
        """
        import fsspec

        fs = fsspec.filesystem(
            self.protocol,
            skip_instance_cache=self.holds_connection,
            **self._storage_options(config),
        )
        try:
            yield fs
        finally:
            if self.holds_connection:
                client = getattr(fs, "client", None)
                if client is not None:
                    client.close()

    def _root(self, config: dict[str, Any]) -> str:
        """The container path files are addressed under."""
        raw = str(config.get("container", "")).strip().rstrip("/")
        return raw if self.container_is_path else raw.lstrip("/")

    def _base(self, config: dict[str, Any]) -> str:
        """The container plus prefix: everything a listing is relative to."""
        prefix = str(config.get("prefix") or "").strip().strip("/")
        root = self._root(config)
        return posixpath.join(root, prefix) if prefix else root

    def _listing(
        self, fs: AbstractFileSystem, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Every file matching the pattern, with its path relative to the base.

        Sorted by relative path so a table's files are read in a stable order and two
        runs over an unchanged container produce the same rows in the same sequence.

        The cache is dropped first. fsspec keeps directory listings per filesystem and
        reuses the instance across calls, so without this an incremental sync in a
        long-lived process keeps answering from the listing it took the first time --
        never seeing a new file, and raising FileNotFoundError for one it has since been
        told about by other means.
        """
        fs.invalidate_cache()
        base = self._base(config)
        pattern = str(config.get("pattern") or "**").strip().lstrip("/")
        matches = fs.glob(f"{base}/{pattern}" if base else pattern, detail=True)
        if not isinstance(matches, dict):  # pragma: no cover - fsspec always details
            matches = {path: fs.info(path) for path in matches}

        files: list[dict[str, Any]] = []
        for path, info in matches.items():
            if info.get("type") != "file" or not info.get("size", 1):
                # Directories, and the zero-byte markers some consoles create to make an
                # empty "folder" appear, both list as entries with nothing to read.
                continue
            relative = path[len(base) :].lstrip("/") if base else path
            files.append(
                {
                    "path": path,
                    "relative": relative,
                    "modified": _modified(info),
                }
            )
        files.sort(key=lambda f: f["relative"])
        return files

    def _group(self, relative: str) -> str:
        """The table a file belongs to: its first folder below the prefix, else root.

        A lakehouse-shaped container (``events/…``, ``users/…``) becomes one table per
        folder, which is what the layout already means. A flat container becomes one
        table, because there is nothing in the paths to divide it by.
        """
        head, _, tail = relative.partition("/")
        return _table_name(head) if tail else "root"

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the fields, then list the container to prove the credentials work."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            with self._open(config) as fs:
                fs.ls(self._base(config), detail=False)
        except Exception as e:  # any failure the provider's client raises
            return False, [f"Could not list {self._base(config)!r}: {e}"]
        return True, []

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Group the matching files into tables, one per folder below the prefix."""
        with self._open(config) as fs:
            groups = {self._group(f["relative"]) for f in self._listing(fs, config)}
        return [
            SourceSchema(name=name, incremental_fields=[FILE_MODIFIED])
            for name in sorted(groups)
        ]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Read the chosen table's files, newest-modified first cursor honoured."""
        chosen = str(inputs.config.get("format") or "auto")
        since = inputs.incremental_since if inputs.incremental_field else None

        with self._open(inputs.config) as fs:
            for file in self._listing(fs, inputs.config):
                if self._group(file["relative"]) != inputs.schema:
                    continue
                if since is not None and not _after(file["modified"], since):
                    continue
                fmt = chosen if chosen != "auto" else format_for(file["relative"])
                if fmt is None or fmt == "auto":
                    if inputs.logger is not None:
                        inputs.logger.info(
                            "skipping %s: no format in the name, and none chosen",
                            file["relative"],
                        )
                    continue
                for batch in _read(fs, file["path"], fmt):
                    yield _stamp(batch, file)


def _modified(info: dict[str, Any]) -> datetime:
    """The file's modification time as an aware timezone.utc datetime.

    Providers disagree on the key and the type -- S3 gives ``LastModified`` as a
    datetime, others give ``mtime``, and some give a plain epoch number -- so this
    normalises rather than trusting any one of them. A file with no usable timestamp is
    dated at the epoch, which makes it always-new instead of silently unsyncable.
    """
    for key in ("LastModified", "last_modified", "mtime", "modification_time"):
        value = info.get(key)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, int | float):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(0, tz=timezone.utc)


def _after(modified: datetime, since: Any) -> bool:
    """Whether ``modified`` is past the stored cursor, whatever type that came back as.

    The cursor round-trips through the app's database, so it can return as a naive
    datetime or a string even though it left here aware. An unreadable cursor is treated
    as no cursor, which re-reads a file rather than dropping one.
    """
    cursor: datetime | None = None
    if isinstance(since, str):
        try:
            cursor = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            return True
    elif isinstance(since, datetime):
        cursor = since
    if cursor is None:
        return True
    if cursor.tzinfo is None:
        cursor = cursor.replace(tzinfo=timezone.utc)
    return modified > cursor


def _stamp(batch: pa.Table, file: dict[str, Any]) -> pa.Table:
    """Append the provenance columns to a batch."""
    rows = batch.num_rows
    return batch.append_column(
        FILE_PATH, pa.array([file["relative"]] * rows, pa.string())
    ).append_column(
        FILE_MODIFIED,
        pa.array([file["modified"]] * rows, pa.timestamp("us", tz="UTC")),
    )


def _read(fs: AbstractFileSystem, path: str, fmt: str) -> Iterator[pa.Table]:
    """Read one file as Arrow, in slices bounded by ``_BATCH`` rows."""
    if fmt == "parquet":
        import pyarrow.parquet as pq

        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=_BATCH):
                yield pa.Table.from_batches([batch])
        return

    if fmt == "csv":
        import pyarrow.csv as pcsv

        with fs.open(path, "rb") as handle:
            reader = pcsv.open_csv(
                handle, read_options=pcsv.ReadOptions(block_size=1 << 24)
            )
            try:
                for batch in reader:
                    yield pa.Table.from_batches([batch])
            finally:
                reader.close()
        return

    if fmt == "jsonl":
        import pyarrow.json as pjson

        # read_json takes the whole file: its reader has no batch interface, and the
        # block_size below bounds the parse buffer rather than the result. A file too
        # large for memory has to be split in the store, which is how the format is
        # meant to be used anyway.
        with fs.open(path, "rb") as handle:
            table = pjson.read_json(
                handle, read_options=pjson.ReadOptions(block_size=1 << 24)
            )
        for start in range(0, table.num_rows, _BATCH):
            yield table.slice(start, _BATCH)
        return

    raise ValueError(f"unsupported format: {fmt}")


@SourceRegistry.register
class S3Source(_ObjectStoreSource):
    """Amazon S3, and anything that speaks its API.

    The endpoint field is what makes the second half true: pointed at MinIO, Cloudflare
    R2, Backblaze B2 or any other S3-compatible store, the rest of the connector is
    unchanged. Left empty it addresses AWS itself.

    Credentials are optional because a public bucket needs none, and because a copy
    running on EC2, ECS or EKS should use the role it was given rather than a key pasted
    into a form. Empty keys therefore mean "use the ambient credentials", and only if
    there are none does boto3 fall back to unsigned requests.
    """

    protocol = "s3"
    name = "s3"
    label = "Amazon S3"
    icon = "🪣"
    container_label = "Bucket"
    container_placeholder = "my-data-bucket"

    @property
    def docs_url(self) -> str:
        """AWS's own credentials guidance."""
        return "https://docs.aws.amazon.com/sdkref/latest/guide/access.html"

    @property
    def caption(self) -> str:
        """Names the S3-compatible case, which is why the endpoint field exists."""
        return "Sync CSV, JSON and Parquet files from S3, or any S3-compatible store."

    def _credential_fields(self) -> list[SourceField]:
        return [
            SourceField(
                name="access_key_id",
                label="Access key ID",
                required=False,
                caption="Leave empty to use the credentials the host already has (an "
                "instance role, or the environment), or for a public bucket.",
            ),
            SourceField(
                name="secret_access_key",
                label="Secret access key",
                type="password",
                required=False,
            ),
            SourceField(
                name="region",
                label="Region",
                required=False,
                placeholder="us-east-1",
            ),
            SourceField(
                name="endpoint",
                label="Endpoint URL",
                required=False,
                placeholder="https://play.min.io",
                caption="Only for an S3-compatible store such as MinIO, R2 or B2. "
                "Leave empty for AWS.",
            ),
        ]

    def _storage_options(self, config: dict[str, Any]) -> dict[str, Any]:
        key = str(config.get("access_key_id") or "").strip()
        secret = str(config.get("secret_access_key") or "").strip()
        client: dict[str, Any] = {}
        if endpoint := str(config.get("endpoint") or "").strip():
            client["endpoint_url"] = endpoint
        if region := str(config.get("region") or "").strip():
            client["region_name"] = region
        options: dict[str, Any] = {}
        if key and secret:
            options["key"] = key
            options["secret"] = secret
        if client:
            options["client_kwargs"] = client
        return options


@SourceRegistry.register
class GcsSource(_ObjectStoreSource):
    """Google Cloud Storage, authenticated with a service-account key.

    The same key field the BigQuery connector uses, for the same reason: a key pasted
    into the form works identically wherever the app runs, where anything ambient only
    works on Google's own infrastructure. Left empty, the ambient credentials are used,
    which is what a copy running on GKE or Cloud Run should do.
    """

    protocol = "gcs"
    name = "gcs"
    label = "Google Cloud Storage"
    icon = "🗄️"
    container_label = "Bucket"
    container_placeholder = "my-gcs-bucket"

    @property
    def docs_url(self) -> str:
        """Google's service-account documentation."""
        return "https://cloud.google.com/iam/docs/service-account-overview"

    def _credential_fields(self) -> list[SourceField]:
        return [
            SourceField(
                name="key_file",
                label="Google Cloud JSON key file",
                type="textarea",
                secret=True,
                required=False,
                placeholder='{"type": "service_account", ...}',
                caption="A service-account key with read access to the bucket. Leave "
                "empty to use the credentials the host already has.",
            ),
        ]

    def _storage_options(self, config: dict[str, Any]) -> dict[str, Any]:
        raw = str(config.get("key_file") or "").strip()
        if not raw:
            return {}
        try:
            token = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"the key file is not valid JSON: {e}") from e
        return {"token": token}


@SourceRegistry.register
class AzureBlobSource(_ObjectStoreSource):
    """Azure Blob Storage, by account key or shared-access signature.

    Two credentials rather than one because Azure issues two kinds and neither
    substitutes for the other: an account key opens the whole account, and a SAS token
    is scoped and expiring. A deployment that should not hold the account key uses the
    SAS token, so refusing to accept it would force the wrong choice.
    """

    protocol = "abfs"
    name = "azure_blob"
    label = "Azure Blob Storage"
    icon = "📦"
    container_label = "Container"
    container_placeholder = "my-container"

    @property
    def docs_url(self) -> str:
        """Azure's authorisation documentation."""
        return "https://learn.microsoft.com/azure/storage/common/authorize-data-access"

    def _credential_fields(self) -> list[SourceField]:
        return [
            SourceField(
                name="account_name",
                label="Storage account name",
                placeholder="mystorageaccount",
            ),
            SourceField(
                name="account_key",
                label="Account key",
                type="password",
                required=False,
                caption="Either this or a SAS token. The account key opens the whole "
                "account; a SAS token can be scoped to this container.",
            ),
            SourceField(
                name="sas_token",
                label="SAS token",
                type="password",
                required=False,
            ),
        ]

    def _storage_options(self, config: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {
            "account_name": str(config.get("account_name") or "").strip()
        }
        if key := str(config.get("account_key") or "").strip():
            options["account_key"] = key
        elif sas := str(config.get("sas_token") or "").strip():
            options["sas_token"] = sas
        return options

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Require one of the two credentials before trying to list the container."""
        fields_ok, field_errors = Source.validate(self, config)
        if not fields_ok:
            return False, field_errors
        if not (config.get("account_key") or config.get("sas_token")):
            return False, ["Either an account key or a SAS token is required"]
        return super().validate(config)


@SourceRegistry.register
class SftpSource(_ObjectStoreSource):
    """A directory on an SFTP server, by password or private key.

    Not object storage, but the same problem: files under a path, matched by a glob and
    read by format. Sharing the engine means SFTP gets incremental sync and Parquet
    support for free, which a connector written on its own would have had to repeat.
    """

    protocol = "sftp"
    name = "sftp"
    label = "SFTP"
    icon = "🔐"
    container_label = "Folder path"
    container_placeholder = "/exports"
    container_is_path = True
    holds_connection = True

    @property
    def docs_url(self) -> str:
        """No vendor to point at; the fields are the whole story."""
        return ""

    @property
    def caption(self) -> str:
        """Names the transport rather than a provider."""
        return "Sync CSV, JSON and Parquet files from a folder on an SFTP server."

    def _credential_fields(self) -> list[SourceField]:
        return [
            SourceField(name="host", label="Host", placeholder="sftp.example.com"),
            SourceField(name="port", label="Port", type="number", required=False),
            SourceField(name="username", label="Username"),
            SourceField(
                name="password",
                label="Password",
                type="password",
                required=False,
                caption="Either this or a private key.",
            ),
            SourceField(
                name="private_key",
                label="Private key",
                type="textarea",
                secret=True,
                required=False,
                placeholder="-----BEGIN OPENSSH PRIVATE KEY-----",
            ),
        ]

    def _storage_options(self, config: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {
            "host": str(config.get("host") or "").strip(),
            "username": str(config.get("username") or "").strip(),
            "port": int(config.get("port") or 22),
            # Use the credential this source was configured with and nothing else.
            # Left on, paramiko would also try the host's ssh-agent and the private keys
            # in its home directory, so a connector could authenticate as whoever
            # happens to be running the server -- and the agent socket it opens for that
            # is never closed. Both are wrong for a credential the user typed in.
            "allow_agent": False,
            "look_for_keys": False,
        }
        if password := str(config.get("password") or "").strip():
            options["password"] = password
        if raw_key := str(config.get("private_key") or "").strip():
            options["pkey"] = _private_key(raw_key)
        return options

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Require one of the two credentials before trying to list the folder."""
        fields_ok, field_errors = Source.validate(self, config)
        if not fields_ok:
            return False, field_errors
        if not (config.get("password") or config.get("private_key")):
            return False, ["Either a password or a private key is required"]
        return super().validate(config)


def _private_key(raw: str) -> Any:
    """A pasted private key as a paramiko key object.

    The key type is not asked for, because the header already says which it is and a
    mismatched answer would be one more thing to get wrong. Each loader is tried and the
    first that accepts the material wins. DSA is not among them: paramiko dropped it,
    and OpenSSH refused it by default long before that.

    When none accepts it, every loader's own complaint is carried into the error. One
    of them is the real reason -- an encrypted key, a truncated paste -- and discarding
    them all would leave the user with "could not be read" and nothing to act on.
    """
    import io

    import paramiko

    loaders = (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey)
    refusals: list[str] = []
    for loader in loaders:
        try:
            return loader.from_private_key(io.StringIO(raw))
        except Exception as e:
            refusals.append(f"{loader.__name__}: {e}")
    raise ValueError(
        "the private key could not be read as Ed25519, ECDSA or RSA; check that the "
        "whole file was pasted, including its header and footer ("
        + "; ".join(refusals)
        + ")"
    )
