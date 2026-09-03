"""Project configuration and local data bindings.

Two files, two responsibilities:

* ``elbi.yaml``, the *committed*, logical project wiring: which
  derivations directory to load and which datasets exist. No credentials.
* ``elbi.dev.yaml``, *local-only* (gitignored) bindings from a dataset
  name to a concrete source: a file path, or ``env(VAR)`` for a connection string
  the developer is authorized to use.

The split is the governance line: the laptop never holds governed data unless the
source already grants this developer access.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import versioning
from .compute import ComputeBackend
from .data import Table, load_table
from .errors import ConfigError, DataBindingError
from .sandbox import ComputeProfileError, ComputeProfiles

PROJECT_FILENAME = "elbi.yaml"
DEV_BINDINGS_FILENAME = "elbi.dev.yaml"

#: File formats a compute engine can scan without loading fully (out of core).
_SCANNABLE_SUFFIXES = {".parquet", ".csv", ".json", ".ndjson", ".jsonl"}

_ENV_REF = re.compile(r"^env\(([A-Za-z_][A-Za-z0-9_]*)\)$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


#: The analytical role a column plays. These are column-intrinsic and
#: domain-general (a semantic-layer staple): they describe what a column *is*, not
#: how to answer any particular question.
_COLUMN_ROLES = ("dimension", "measure", "weight", "identifier", "time")


@dataclass(frozen=True)
class ColumnSpec:
    """Author-declared meaning for one dataset column (its semantic model).

    Every field is optional context describing the *data*, the way a dbt or
    Snowflake semantic model does: what the column means, its type and unit, the
    analytical role it plays, alternate names, and a few example values. It never
    describes a question or its answer.
    """

    name: str
    description: str | None = None
    type: str | None = None
    unit: str | None = None
    role: str | None = None
    synonyms: tuple[str, ...] = ()
    sample_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset's name plus its optional semantic model (per-column meaning)."""

    name: str
    description: str | None = None
    columns: tuple[ColumnSpec, ...] = ()


@dataclass(frozen=True)
class TargetSpec:
    """One deployment this project is meant to talk to.

    The host lives in the committed file and the credential does not, which is the split
    Databricks recommends for the same reason: it makes the file portable, because a
    checkout says *which* deployment it belongs to while every machine keeps its own way
    of authenticating to it.

    Without this a repo says nothing about where it belongs, and one environment
    variable serves a personal install and a company deployment alike, so a stale value
    pushes personal work into production with nothing able to notice.
    """

    name: str
    host: str
    default: bool = False


@dataclass(frozen=True)
class SourceSpec:
    """A data source the project pre-declares in the warehouse.

    The deployed app treats the warehouse as the only source of data: a project names
    its data here (a connector ``type`` and its ``config``) and the app ensures the
    source exists and is synced at load, landing a warehouse table named ``name``. Core
    carries this spec inertly (like ``sandbox``/``egress``); only the app acts on it.
    ``sync`` is the auto-sync cadence (``manual`` by default).
    """

    name: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)
    sync: str = "manual"


@dataclass(frozen=True)
class ProjectConfig:
    """The committed project wiring (``elbi.yaml``)."""

    project: str
    derivations_dir: str = "derivations"
    #: Each declared dataset with its optional semantic model (per-column meaning).
    #: A dataset declared as a bare name has an empty model.
    datasets: tuple[DatasetSpec, ...] = ()
    #: Warehouse sources the app ensures + syncs at load (the app's data declaration).
    #: Core carries these inertly; the deployed app is what acts on them.
    sources: tuple[SourceSpec, ...] = ()
    #: The deployments this project may be applied to. Empty means unconstrained, the
    #: case for a laptop and for every repo predating this.
    targets: tuple[TargetSpec, ...] = ()
    #: Optional project-specific guidance for the authoring agent, appended to the
    #: server's default analysis guidance and surfaced over MCP.
    ai_context: str | None = None
    #: Retrieval mode for ``search_derivations``: ``"hybrid"`` (the default: BM25 fused
    #: with semantic embeddings via RRF, matching on both words and meaning) or
    #: ``"lexical"`` (BM25 alone, a purely lexical, offline ranker).
    search: str = "hybrid"
    #: Where sandboxed code runs: ``"subprocess"`` (a hardened host subprocess, the
    #: dependency-free default), ``"docker"`` (an isolated Linux container, which needs
    #: a Docker daemon but gives a real filesystem boundary and manylinux wheels, so
    #: native libraries like OpenMP resolve), or ``"kubernetes"`` (a pod per session,
    #: sized by its compute profile and isolated by the cluster, the deployed one).
    sandbox: str = "subprocess"
    #: Outbound network policy for the docker exploration sandbox: ``"full"`` (default),
    #: ``"none"`` (offline), or a list of allowlisted hosts (enforced by a filtering
    #: proxy, so an analysis with your data cannot exfiltrate to arbitrary hosts).
    egress: str | tuple[str, ...] = "full"
    #: Container image for the docker sandbox; ``None`` uses the slim default. Point it
    #: at a data-science image (``just sandbox-image`` builds one) so the common stack
    #: and a build toolchain are present and the agent skips per-session installs.
    sandbox_image: str | None = None
    #: The sized shapes a sandbox may run as, as a list of profile mappings. Empty means
    #: one implicit profile at the built-in defaults, so a project that does not care
    #: about sizing writes nothing. See :mod:`elbi_core.sandbox`.  A profile's limits
    #: are enforced by the container runtime's cgroup, which is the only place a memory
    #: cap can be enforced honestly: ``RLIMIT_AS`` counts file mappings and virtual
    #: address space, so it fails spuriously in numpy code long before real memory is
    #: exhausted. The ``subprocess`` sandbox therefore enforces nothing and is bounded
    #: only by the host, which is why it is the single-machine tier rather than the
    #: deployed one.
    compute_profiles: tuple[Mapping[str, Any], ...] = ()
    #: Which profile a session gets when it does not ask for one. Defaults to the first.
    default_compute_profile: str | None = None

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """The declared dataset names, in order."""
        return tuple(spec.name for spec in self.datasets)

    def spec_for(self, name: str) -> DatasetSpec | None:
        """The semantic model declared for dataset ``name``, if any."""
        return next((s for s in self.datasets if s.name == name), None)

    @classmethod
    def load(cls, path: Path) -> ProjectConfig:
        """Load and validate a project config from ``path``."""
        data = _read_yaml_mapping(path)

        project = data.get("project")
        if not isinstance(project, str) or not project:
            raise ConfigError(f"{path}: 'project' must be a non-empty string")

        derivations_dir = data.get("derivations_dir", "derivations")
        if not isinstance(derivations_dir, str) or not derivations_dir:
            raise ConfigError(f"{path}: 'derivations_dir' must be a non-empty string")

        datasets = _parse_datasets(data.get("datasets", []), path)
        sources = _parse_sources(data.get("sources", []), path)

        ai_context = data.get("ai_context")
        if ai_context is not None and not isinstance(ai_context, str):
            raise ConfigError(f"{path}: 'ai_context' must be a string")

        search = data.get("search", "hybrid")
        if search not in ("lexical", "hybrid"):
            raise ConfigError(
                f"{path}: 'search' must be 'lexical' or 'hybrid', not {search!r}"
            )

        sandbox = data.get("sandbox", "subprocess")
        if sandbox not in ("subprocess", "docker", "kubernetes"):
            raise ConfigError(
                f"{path}: 'sandbox' must be 'subprocess', 'docker' or 'kubernetes', "
                f"not {sandbox!r}"
            )

        egress = _parse_egress(data.get("egress", "full"), path)

        sandbox_image = data.get("sandbox_image")
        if sandbox_image is not None and (
            not isinstance(sandbox_image, str) or not sandbox_image.strip()
        ):
            raise ConfigError(f"{path}: 'sandbox_image' must be a non-empty string")

        profiles = _parse_compute_profiles(data.get("compute_profiles", []), path)
        default_profile = data.get("default_compute_profile")
        if default_profile is not None and not isinstance(default_profile, str):
            raise ConfigError(f"{path}: 'default_compute_profile' must be a string")

        return cls(
            project=project,
            derivations_dir=derivations_dir,
            datasets=datasets,
            sources=sources,
            targets=_parse_targets(data.get("targets"), path),
            ai_context=ai_context.strip() if ai_context else None,
            search=search,
            sandbox=sandbox,
            egress=egress,
            sandbox_image=sandbox_image.strip() if sandbox_image else None,
            compute_profiles=profiles,
            default_compute_profile=default_profile,
        )


def _parse_targets(raw: object, path: Path) -> tuple[TargetSpec, ...]:
    """Validate ``targets``: a mapping of name to at least a host.

    At most one may be the default. Databricks allows "one and only one", and the reason
    holds: with two, which one a bare command means depends on iteration order, and the
    whole point of naming a host in the file is that it cannot be ambiguous.
    """
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path}: 'targets' must be a mapping of name to settings, "
            "for example {dev: {host: https://..., default: true}}"
        )
    specs: list[TargetSpec] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: target {name!r} must be a mapping")
        host = entry.get("host")
        if not isinstance(host, str) or not host.strip():
            raise ConfigError(f"{path}: target {name!r} needs a 'host'")
        flag = entry.get("default", False)
        if not isinstance(flag, bool):
            raise ConfigError(
                f"{path}: target {name!r}: 'default' must be true or false"
            )
        specs.append(
            TargetSpec(name=str(name), host=host.strip().rstrip("/"), default=flag)
        )
    defaults = [spec.name for spec in specs if spec.default]
    if len(defaults) > 1:
        raise ConfigError(
            f"{path}: only one target may be the default, but "
            f"{', '.join(sorted(defaults))} all are"
        )
    return tuple(specs)


def _parse_compute_profiles(raw: object, path: Path) -> tuple[Mapping[str, Any], ...]:
    """Validate ``compute_profiles`` eagerly, and keep it as plain data.

    Parsed here so a bad profile fails when the project loads rather than when someone
    opens a notebook, but stored as mappings: the config layer stays a description of
    what was written, and :class:`~elbi_core.sandbox.ComputeProfiles` is what
    resolves the menu (a deployment can add to it from Helm).
    """
    if not isinstance(raw, (list, tuple)):
        raise ConfigError(f"{path}: 'compute_profiles' must be a list of profiles")
    if not raw:
        return ()
    try:
        ComputeProfiles.from_config(raw, context=f"{path}: compute_profiles")
    except ComputeProfileError as exc:
        raise ConfigError(str(exc)) from exc
    return tuple(dict(item) for item in raw)


def _parse_egress(value: Any, path: Path) -> str | tuple[str, ...]:
    """Validate the ``egress`` setting: ``"full"``, ``"none"``, or a list of hosts."""
    if isinstance(value, str):
        if value not in ("full", "none"):
            raise ConfigError(
                f"{path}: 'egress' string must be 'full' or 'none', not {value!r}"
            )
        return value
    if isinstance(value, list):
        if not value or not all(isinstance(h, str) and h.strip() for h in value):
            raise ConfigError(
                f"{path}: 'egress' list must be one or more non-empty hostnames"
            )
        return tuple(h.strip() for h in value)
    raise ConfigError(f"{path}: 'egress' must be 'full', 'none', or a list of hosts")


_SQL_KEYS = {"connection", "query", "table", "freshness", "max_rows", "timeout"}


@dataclass(frozen=True)
class _SqlSpec:
    connection: str
    query: str | None
    table: str | None
    freshness: str | None
    max_rows: int
    timeout: int


def _run_sql(
    spec: _SqlSpec, *, query: str | None = None, table: str | None = None
) -> Table:
    from . import sql

    return sql.load_sql(
        spec.connection,
        query=query,
        table=table,
        max_rows=spec.max_rows,
        timeout=spec.timeout,
    )


@dataclass(frozen=True)
class DataBindings:
    """Local dataset bindings (``elbi.dev.yaml``).

    A binding value is either a **string** (a file path or ``env(VAR)``) or a
    **mapping** describing a SQL source (``connection`` + ``query``/``table``).
    """

    bindings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> DataBindings:
        """Load bindings from ``path``; an absent file yields empty bindings."""
        if not path.exists():
            return cls(bindings={})
        data = _read_yaml_mapping(path)
        raw = data.get("data", {})
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: 'data' must be a mapping of dataset to source")
        bindings: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, str):
                bindings[str(key)] = value
            elif isinstance(value, dict):
                bindings[str(key)] = dict(value)
            else:
                raise ConfigError(
                    f"{path}: binding for {key!r} must be a string (file) or a "
                    "mapping (SQL source)"
                )
        return cls(bindings=bindings)

    def resolve(self, dataset: str, base_dir: Path) -> Path:
        """Resolve a *file* dataset binding to a concrete path.

        ``env(VAR)`` bindings are read from the environment; bare paths are
        resolved relative to ``base_dir``.

        Raises:
            DataBindingError: if the dataset is unbound, is a SQL binding, or an
                ``env`` var is unset.
        """
        source = self._require(dataset)
        if not isinstance(source, str):
            raise DataBindingError(
                f"dataset {dataset!r} is a SQL binding, not a file path"
            )
        candidate = Path(_resolve_env(source, dataset))
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return candidate

    def load_dataset(self, dataset: str, base_dir: Path) -> Table:
        """Load a dataset into a :class:`~elbi.data.Table`.

        Dispatches on the binding shape: a string is a file; a mapping is a SQL
        source.
        """
        source = self._require(dataset)
        if isinstance(source, str):
            return load_table(self.resolve(dataset, base_dir))
        return self._load_sql(dataset, source)

    def load_backend(self, dataset: str, base_dir: Path) -> ComputeBackend:
        """Bind a dataset to a compute backend for the trusted analysis tools.

        The scale path: when a compute engine (DuckDB) is installed and the binding is a
        columnar/text file, the file is bound as an engine handle that the profiler,
        contract checker, and oracle push down to and scan without loading it into
        memory. Otherwise the rows are materialized into the reference backend.
        Unlike :meth:`load_dataset` this never returns raw rows to a caller; it is for
        the first-party tools that operate through the compute seam, not for a
        derivation's ``ctx.input(...).rows``.
        """
        from .compute import backend_for

        source = self._require(dataset)
        if isinstance(source, str):
            path = self.resolve(dataset, base_dir)
            if path.suffix.lower() in _SCANNABLE_SUFFIXES:
                try:
                    return backend_for(path)  # an engine scan, out of core
                except TypeError:
                    pass  # no engine installed; fall back to loading the rows
            return backend_for(load_table(path).rows)
        return backend_for(self._load_sql(dataset, source).rows)

    def version(self, dataset: str, base_dir: Path) -> tuple[str, Table | None]:
        """Return a content fingerprint for a dataset, for cache invalidation.

        The fingerprint changes whenever the underlying data changes, so a
        derivation keyed on it recomputes exactly when its inputs change. The
        second element is the loaded :class:`Table` when computing the version
        required loading it (so the caller can reuse it instead of loading twice),
        else ``None``.

        * **File** → SHA-256 of the file's bytes (cheap; no parse).
        * **SQL with a ``freshness`` query** → hash of that probe's result (cheap;
          avoids pulling the full dataset on a cache hit).
        * **SQL without ``freshness``** → hash of the full result set (sound, but
          requires the read); the loaded table is returned for reuse.
        """
        source = self._require(dataset)
        if isinstance(source, str):
            return versioning.hash_file(self.resolve(dataset, base_dir)), None
        spec = self._parse_sql(dataset, source)
        if spec.freshness is not None:
            probe = _run_sql(spec, query=spec.freshness)
            return versioning.hash_json(probe.rows), None
        table = _run_sql(spec, query=spec.query, table=spec.table)
        return versioning.hash_json(table.rows), table

    def _require(self, dataset: str) -> Any:
        if dataset not in self.bindings:
            known = ", ".join(sorted(self.bindings)) or "(none)"
            raise DataBindingError(
                f"dataset {dataset!r} has no local binding; bound datasets: {known}. "
                f"Add it to {DEV_BINDINGS_FILENAME}."
            )
        return self.bindings[dataset]

    def _load_sql(self, dataset: str, source: dict[str, Any]) -> Table:
        spec = self._parse_sql(dataset, source)
        return _run_sql(spec, query=spec.query, table=spec.table)

    def _parse_sql(self, dataset: str, source: dict[str, Any]) -> _SqlSpec:
        from . import sql

        unknown = set(source) - _SQL_KEYS
        if unknown:
            raise ConfigError(
                f"dataset {dataset!r}: unknown SQL binding key(s) {sorted(unknown)}; "
                f"allowed: {sorted(_SQL_KEYS)}"
            )
        connection = source.get("connection")
        if not isinstance(connection, str) or not connection:
            raise ConfigError(
                f"dataset {dataset!r}: SQL binding needs a 'connection' string"
            )
        for key in ("query", "table", "freshness"):
            if source.get(key) is not None and not isinstance(source[key], str):
                raise ConfigError(f"dataset {dataset!r}: {key!r} must be a string")
        max_rows = source.get("max_rows", sql.DEFAULT_MAX_ROWS)
        timeout = source.get("timeout", sql.DEFAULT_TIMEOUT)
        if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
            raise ConfigError(f"dataset {dataset!r}: 'max_rows' must be a positive int")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
            raise ConfigError(f"dataset {dataset!r}: 'timeout' must be a positive int")
        return _SqlSpec(
            connection=_resolve_env(connection, dataset),
            query=source.get("query"),
            table=source.get("table"),
            freshness=source.get("freshness"),
            max_rows=max_rows,
            timeout=timeout,
        )


def _resolve_env(value: str, dataset: str) -> str:
    """Resolve an ``env(VAR)`` reference, else return the value unchanged."""
    match = _ENV_REF.match(value)
    if not match:
        return value
    var = match.group(1)
    resolved = os.environ.get(var)
    if resolved is None:
        raise DataBindingError(
            f"dataset {dataset!r} binds to env({var}), but {var} is not set"
        )
    return resolved


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return loaded


def _parse_datasets(raw: Any, path: Path) -> tuple[DatasetSpec, ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: 'datasets' must be a list")
    specs = tuple(_parse_dataset(entry, path) for entry in raw)
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ConfigError(f"{path}: dataset {spec.name!r} is declared twice")
        seen.add(spec.name)
    return specs


#: A source/table name must be a valid Delta directory and DuckDB identifier.
_SOURCE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_sources(raw: Any, path: Path) -> tuple[SourceSpec, ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: 'sources' must be a list")
    specs = tuple(_parse_source(entry, path) for entry in raw)
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ConfigError(f"{path}: source {spec.name!r} is declared twice")
        seen.add(spec.name)
    return specs


def _parse_source(entry: Any, path: Path) -> SourceSpec:
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        raise ConfigError(f"{path}: each source must be a mapping with a 'name'")
    name = entry["name"]
    if not _SOURCE_NAME_RE.match(name):
        raise ConfigError(
            f"{path}: source name {name!r} must be a letter/underscore followed by "
            "letters, digits, or underscores (it becomes a warehouse table name)"
        )
    source_type = entry.get("type")
    if not isinstance(source_type, str) or not source_type:
        raise ConfigError(f"{path}: source {name!r}: 'type' must be a non-empty string")
    config = entry.get("config", {})
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: source {name!r}: 'config' must be a mapping")
    config = dict(config)
    # CSV/Parquet shorthand: a top-level `path:` folds into the connector config, and
    # the table is named after the source so it lands unprefixed as `name`.
    if "path" in entry:
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise ConfigError(f"{path}: source {name!r}: 'path' must be a string")
        config.setdefault("path", entry["path"])
    if source_type == "csv":
        config.setdefault("table_name", name)
    sync = entry.get("sync", "manual")
    if not isinstance(sync, str) or not sync:
        raise ConfigError(f"{path}: source {name!r}: 'sync' must be a string")
    return SourceSpec(name=name, type=source_type, config=config, sync=sync)


def _parse_dataset(entry: Any, path: Path) -> DatasetSpec:
    if isinstance(entry, str):
        return DatasetSpec(name=_dataset_name(entry, path))
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        raise ConfigError(
            f"{path}: each dataset must be a name or a mapping with a 'name'"
        )
    name = _dataset_name(entry["name"], path)
    description = entry.get("description")
    if description is not None and not isinstance(description, str):
        raise ConfigError(f"{path}: dataset {name!r}: 'description' must be a string")
    raw_columns = entry.get("columns", [])
    if not isinstance(raw_columns, list):
        raise ConfigError(f"{path}: dataset {name!r}: 'columns' must be a list")
    columns = tuple(_parse_column(col, name, path) for col in raw_columns)
    return DatasetSpec(
        name=name,
        description=description.strip() if description else None,
        columns=columns,
    )


def _parse_column(entry: Any, dataset: str, path: Path) -> ColumnSpec:
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        raise ConfigError(
            f"{path}: dataset {dataset!r}: each column must be a mapping with a 'name'"
        )
    name = entry["name"]
    for field_name in ("description", "type", "unit", "role"):
        value = entry.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ConfigError(
                f"{path}: dataset {dataset!r} column {name!r}: "
                f"{field_name!r} must be a string"
            )
    role = entry.get("role")
    if role is not None and role not in _COLUMN_ROLES:
        raise ConfigError(
            f"{path}: dataset {dataset!r} column {name!r}: role {role!r} must be "
            f"one of {', '.join(_COLUMN_ROLES)}"
        )
    return ColumnSpec(
        name=name,
        description=_clean(entry.get("description")),
        type=_clean(entry.get("type")),
        unit=_clean(entry.get("unit")),
        role=role,
        synonyms=_str_tuple(entry.get("synonyms"), dataset, name, "synonyms", path),
        sample_values=_str_tuple(
            entry.get("sample_values"), dataset, name, "sample_values", path
        ),
    )


def _dataset_name(name: str, path: Path) -> str:
    if not _NAME.match(name):
        raise ConfigError(f"{path}: dataset name {name!r} must match ^[a-z][a-z0-9_]*$")
    return name


def _clean(value: str | None) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _str_tuple(
    value: Any, dataset: str, column: str, field_name: str, path: Path
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(
            f"{path}: dataset {dataset!r} column {column!r}: "
            f"{field_name!r} must be a list of strings"
        )
    return tuple(value)
