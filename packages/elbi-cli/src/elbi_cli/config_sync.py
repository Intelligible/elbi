"""Analytics-as-code: sync a project repo to a running app, and pull it back.

Every configurable artifact (metrics, dashboards, feature views, monitors, schedules,
workflows, data-quality checks, models, saved queries, notebooks) lives as a file in the
repo under a fixed folder, and this engine moves those files to and from a running app
over its HTTP API. The app's own routes do the validation and gating, so ``sync`` is a
thin, uniform file<->API mapper.

Each artifact type is a :class:`Surface`: where its files live, how one object maps to
and from a file, and how to list/create/update/delete it against the app. The engine
verbs are ``validate`` (parse + JSON-Schema check), ``plan`` (diff repo vs app),
``sync`` (apply repo -> app, upserting by name), and ``pull`` (app -> repo).

Objects are keyed by ``name``: the file's stem is the object's name, which is how a
uuid-keyed object (a notebook, a saved query) maps to a stable file: ``sync`` matches an
existing object by name and updates it, so re-running ``sync`` is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import yaml

#: The fixed repo layout. Derivations are plain ``.py`` under ``derivations/`` and are
#: loaded by the project itself, so they are not pushed over the API here.
DEFAULT_FOLDERS = (
    "metrics",
    "dashboards",
    "features",
    "monitors",
    "schedules",
    "workflows",
    "checks",
    "models",
    "queries",
    "notebooks",
    "certificates",
)


class SyncError(Exception):
    """A repo/app sync failure with a caller-facing message."""


#: Where the record of the last pull lives. A checkout's own bookkeeping, not the
#: project: it describes what *this* copy saw, so it is per-machine and gitignored, the
#: way Terraform keeps its working directory beside the configuration rather than in it.
STATE_PATH = Path(".elbi") / "state.json"


@dataclass(frozen=True)
class Change:
    """One planned change for a surface, or one reason a change cannot be made.

    ``create``/``update``/``delete``/``unchanged`` describe what a sync would do.
    ``drift`` and ``conflict`` describe what it found instead, and exist only because
    the repo remembers what it last pulled:

    * ``drift`` -- changed in the app since the pull, untouched here. Terraform uses the
      same word for a resource altered outside the tool, and the same remedy: pull, to
      take the new state as the baseline.
    * ``conflict`` -- changed in both, differently. Pushing would silently discard the
      other change. That is the lost-update problem HTTP answers with ``If-Match`` and
      a 412; here the answer is to refuse and say which objects.
    """

    surface: str
    name: str
    action: str


def _fingerprint(surface: Surface, body: dict[str, Any]) -> str:
    """A stable digest of an object, taken from the form it is written to disk in.

    Serialized first so that the comparison is exactly what a reader of the repo would
    see: two bodies that produce the same file are the same object, whatever their key
    order was in transit.
    """
    return hashlib.sha256(surface.serialize(body).encode("utf-8")).hexdigest()


def read_state(root: Path) -> dict[str, dict[str, str]]:
    """What the last pull into ``root`` saw, as ``surface -> name -> fingerprint``.

    An absent or unreadable file is an empty baseline, which degrades to the old
    behaviour rather than refusing to work: a repo predating this, or one assembled by
    hand, still syncs.
    """
    path = root / STATE_PATH
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    surfaces_ = loaded.get("surfaces") if isinstance(loaded, dict) else None
    if not isinstance(surfaces_, dict):
        return {}
    return {
        str(name): {str(k): str(v) for k, v in table.items()}
        for name, table in surfaces_.items()
        if isinstance(table, dict)
    }


def write_state(root: Path, state: Mapping[str, Mapping[str, str]]) -> None:
    """Record what this copy now has, replacing any previous record."""
    path = root / STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"surfaces": {k: dict(v) for k, v in state.items()}}, indent=2)
        + "\n",
        encoding="utf-8",
    )


@dataclass
class Surface:
    """One artifact type and how it maps between a file and the app's API."""

    name: str
    folder: str
    ext: str
    #: All objects on the app, as ``name -> canonical dict`` (the file's content model).
    remote: Callable[[httpx.Client], dict[str, dict[str, Any]]]
    #: The app-side id for each object name (for update/delete); name itself when the
    #: object is name-keyed on the app.
    remote_ids: Callable[[httpx.Client], dict[str, str]]
    #: Serialize a canonical dict to file text, and parse file text back.
    serialize: Callable[[dict[str, Any]], str]
    deserialize: Callable[[str], dict[str, Any]]
    #: Create or update one object; ``existing_id`` is its app id if already present.
    push: Callable[[httpx.Client, str, dict[str, Any], str | None], None]
    #: Delete one object by its app id.
    delete: Callable[[httpx.Client, str], None]
    #: Optional JSON Schema for the file body (for validate + editor tooling).
    schema: dict[str, Any] | None = None
    #: Split one parsed file into several objects, for a format that groups them (a
    #: metrics file listing several metrics). Given the file stem and its parsed body,
    #: returns ``name -> body``. Absent means one file is one object.
    explode: Callable[[str, dict[str, Any]], dict[str, dict[str, Any]]] | None = None
    #: Names of other objects *in this surface* that ``body`` refers to, so pushes run
    #: in dependency order. Absent means the surface has no internal references.
    references: Callable[[dict[str, Any]], set[str]] | None = None

    def manages(self, root: Path) -> bool:
        """Whether this repo declares this artifact type at all.

        An absent folder is not an empty one: it means the repo says nothing about this
        type, so pruning must leave it alone. An *empty* folder does say something --
        "there should be none of these" -- and pruning honours that.
        """
        return (root / self.folder).is_dir()

    def files(self, root: Path) -> dict[str, str]:
        """Local objects as ``name -> file text`` (empty when the folder is absent)."""
        directory = root / self.folder
        if not directory.is_dir():
            return {}
        out: dict[str, str] = {}
        for path in sorted(directory.glob(f"*{self.ext}")):
            out[path.stem] = path.read_text(encoding="utf-8")
        return out

    def local(self, root: Path) -> dict[str, dict[str, Any]]:
        """Local objects as ``name -> canonical dict``, in dependency order."""
        parsed: dict[str, dict[str, Any]] = {}
        for name, text in self.files(root).items():
            try:
                body = self.deserialize(text)
            except Exception as exc:  # a malformed file is a validation error
                raise SyncError(
                    f"{self.folder}/{name}{self.ext}: could not parse: {exc}"
                ) from exc
            if self.explode is None:
                parsed[name] = body
                continue
            try:
                parsed.update(self.explode(name, body))
            except SyncError as exc:
                raise SyncError(f"{self.folder}/{name}{self.ext}: {exc}") from exc
        return self.ordered(parsed)

    def ordered(
        self, objects: Mapping[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """``objects`` with anything referenced placed before whatever refers to it.

        A ratio metric names the two metrics it divides, and the app rejects one whose
        inputs it has not seen yet -- so pushing in filename order fails whenever the
        alphabet disagrees with the dependency, which for ``x_rate`` before ``x_total``
        it usually does. A cycle keeps its original order: the app's own validation is
        the right place to reject it, and it says so far better than a sort could.
        """
        if self.references is None:
            return dict(objects)
        pending = dict(objects)
        out: dict[str, dict[str, Any]] = {}
        while pending:
            free = [
                name
                for name, body in pending.items()
                if not (self.references(body) & pending.keys() - {name})
            ]
            if not free:  # a cycle: emit the rest as they came
                out.update(pending)
                break
            for name in free:
                out[name] = pending.pop(name)
        return out


# -- serialization helpers ---------------------------------------------------------


def _yaml_dump(data: dict[str, Any]) -> str:
    """Canonical YAML: block style, keys unsorted (spec order), trailing newline."""
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, indent=2)


def _yaml_load(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise SyncError("expected a mapping at the top level")
    return data


#: Per-cell keys a run writes under ``metadata.elbi`` (see
#: ``Store.save_cell_result``): which side of the data boundary the cell landed on, and
#: the queries it pushed down, with their timings.
_RUN_TELEMETRY = ("data_mode", "queries")


def _strip_notebook(ipynb: dict[str, Any]) -> dict[str, Any]:
    """A notebook with everything a run produced removed, for clean diffs.

    The telemetry matters beyond tidiness, because the fingerprint is taken over this
    form: query timings differ between runs, so keeping it would make a notebook that
    has merely *run* look changed in the app, which ``sync`` refuses as a conflict.
    Authored metadata (a ``parameters`` tag) is kept; only what a run wrote is dropped.
    """
    cells = []
    for cell in ipynb.get("cells", []):
        cell = dict(cell)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.pop("id", None)
        metadata = cell.get("metadata")
        ours_in = metadata.get("elbi") if isinstance(metadata, dict) else None
        if isinstance(metadata, dict) and isinstance(ours_in, dict):
            ours = {
                key: value
                for key, value in ours_in.items()
                if key not in _RUN_TELEMETRY
            }
            metadata = dict(metadata)
            # Dropped entirely when a run wrote all of it, so a notebook that has run
            # and one that has not serialise identically.
            if ours:
                metadata["elbi"] = ours
            else:
                metadata.pop("elbi")
            cell["metadata"] = metadata
        cells.append(cell)
    return {**ipynb, "cells": cells}


# -- HTTP helpers ------------------------------------------------------------------


def _get(client: httpx.Client, path: str) -> Any:
    resp = client.get(path)
    _raise_for(resp, path)
    return resp.json()


def _post(client: httpx.Client, path: str, body: dict[str, Any]) -> Any:
    resp = client.post(path, json=body)
    _raise_for(resp, path)
    return resp.json() if resp.content else None


def _put(client: httpx.Client, path: str, body: dict[str, Any]) -> Any:
    resp = client.put(path, json=body)
    _raise_for(resp, path)
    return resp.json() if resp.content else None


def _delete(client: httpx.Client, path: str) -> None:
    resp = client.delete(path)
    _raise_for(resp, path)


def _raise_for(resp: httpx.Response, path: str) -> None:
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text[:200]
        raise SyncError(f"{path} -> HTTP {resp.status_code}: {detail}")


# -- file-body JSON Schemas --------------------------------------------------------
#
# A file omits the manifest boilerplate the app supplies (``specVersion``/``kind`` and
# the object's ``name``, which is the file stem), so the schema a repo file is checked
# against is the app's bundled manifest schema with those keys relaxed from required.
# Surfaces the app has no bundled schema for carry a hand-written schema of their body.


def _metric_file_schema() -> dict[str, Any]:
    """One metric, or a ``metrics:`` list of them.

    The list is the shape every semantic layer writes (dbt groups a metric with the ones
    it divides, which is how anyone reads a ratio), and the shape this project's own
    MetricSet manifest and docs use. A lone metric per file stays valid because that is
    what ``pull`` writes.
    """
    from elbi_core.metrics.spec import load_metric_schema

    full = load_metric_schema()
    defs = dict(full["$defs"])
    metric = dict(defs["metric"])
    metric["required"] = [r for r in metric.get("required", []) if r != "name"]
    defs["metric"] = metric
    grouped = {
        "type": "object",
        "required": ["metrics"],
        "properties": {
            "specVersion": {"type": "string"},
            "kind": {"const": "MetricSet"},
            "metrics": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "allOf": [
                        {"$ref": "#/$defs/metric"},
                        {"required": ["name"]},
                    ]
                },
            },
        },
    }
    return {
        "$schema": full["$schema"],
        "$defs": defs,
        "oneOf": [grouped, {"$ref": "#/$defs/metric"}],
    }


def _explode_metrics(stem: str, body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One metrics file as ``name -> metric``, whichever of the two shapes it uses."""
    listed = body.get("metrics")
    if not isinstance(listed, list):
        return {stem: body}
    out: dict[str, dict[str, Any]] = {}
    for entry in listed:
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            raise SyncError("every metric in a 'metrics:' list needs a name")
        name = str(entry["name"]).strip()
        out[name] = {k: v for k, v in entry.items() if k != "name"}
    return out


def _metric_references(body: dict[str, Any]) -> set[str]:
    """The metrics this one is defined in terms of.

    A ratio names two, a cumulative one, and a derived expression any number -- and the
    app rejects a metric whose inputs it has not been given yet, so these decide the
    order they are pushed in. Expression operands are read as identifiers; a name that
    is not a metric simply never matches anything and is ignored.
    """
    refs = {
        str(body[key]).strip()
        for key in ("numerator", "denominator", "inputMetric", "input_metric")
        if str(body.get(key) or "").strip()
    }
    for key in ("inputMetrics", "input_metrics"):
        listed = body.get(key)
        if isinstance(listed, list):
            refs |= {str(m).strip() for m in listed if str(m).strip()}
    expr = str(body.get("expr") or "")
    refs |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    return {r for r in refs if r}


def _dashboard_file_schema() -> dict[str, Any]:
    """A DashboardSpec with ``name`` relaxed (the file stem supplies it)."""
    from elbi_core.dashboard.spec import load_dashboard_schema

    full = dict(load_dashboard_schema())
    full["required"] = [r for r in full.get("required", []) if r != "name"]
    return full


def _feature_store_file_schema() -> dict[str, Any]:
    """The feature-store manifest with ``specVersion``/``kind`` relaxed."""
    from elbi_core.features.spec import load_feature_store_schema

    full = dict(load_feature_store_schema())
    full["required"] = [
        r for r in full.get("required", []) if r not in ("specVersion", "kind")
    ]
    return full


_QUERIES_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["sql"],
    "additionalProperties": False,
    "properties": {
        "sql": {"type": "string", "minLength": 1},
        "source_id": {"type": ["string", "null"]},
    },
}

_NOTEBOOKS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["cells", "nbformat"],
    "properties": {
        "cells": {"type": "array", "items": {"type": "object"}},
        "metadata": {"type": "object"},
        "nbformat": {"type": "integer"},
        "nbformat_minor": {"type": "integer"},
    },
}

_CERTIFICATES_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema", "certificate", "signature"],
    "properties": {
        "schema": {"type": "string"},
        "certificate": {"type": "object"},
        "signature": {
            "type": "object",
            "required": ["algorithm", "value"],
            "properties": {
                "algorithm": {"type": "string"},
                "value": {"type": "string"},
            },
        },
    },
}

_MONITORS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["target_kind", "target"],
    "additionalProperties": False,
    "properties": {
        "target_kind": {"enum": ["metric", "derivation"]},
        "target": {"type": "string", "minLength": 1},
        "method": {"enum": ["mad", "zscore"]},
        "sensitivity": {"type": "number", "exclusiveMinimum": 0},
        "window": {"type": "integer", "minimum": 2},
        "interval_hours": {"type": "number", "exclusiveMinimum": 0},
        "min_value": {"type": ["number", "null"]},
        "max_value": {"type": ["number", "null"]},
        "config": {"type": "object"},
    },
}

_SCHEDULES_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selection": {"type": "string"},
        # 'cron' fires on the cron expression; 'on_data_change' is a sensor that fires
        # when ``dataset``'s content hash moves (the value the engine checks for).
        "mode": {"enum": ["cron", "on_data_change"]},
        "cron": {"type": "string"},
        "dataset": {"type": ["string", "null"]},
        "enabled": {"type": "boolean"},
    },
}

_WORKFLOWS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "selection": {"enum": ["stale", "all"]},
                    "assets": {"type": "array", "items": {"type": "string"}},
                    "includeDownstream": {"type": "boolean"},
                    "dependsOn": {"type": "array", "items": {"type": "string"}},
                    "runIf": {
                        "enum": [
                            "all_success",
                            "at_least_one_success",
                            "none_failed",
                            "all_done",
                            "at_least_one_failed",
                            "all_failed",
                        ]
                    },
                },
            },
        },
    },
}

_CHECKS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["asset", "name", "expr"],
    "properties": {
        "asset": {"type": "string"},
        "name": {"type": "string"},
        "expr": {"type": "string"},
        "severity": {"enum": ["warn", "error"]},
        "enabled": {"type": "boolean"},
    },
}

_MODELS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["dataset", "target"],
    "additionalProperties": False,
    "properties": {
        "source_kind": {"enum": ["dataset", "derivation", "training_set"]},
        "dataset": {"type": "string", "minLength": 1},
        "target": {"type": "string", "minLength": 1},
        "features": {"type": "array", "items": {"type": "string"}},
        "task": {"type": "string"},
        "engine": {"type": "string"},
        "time_budget": {"type": "number", "exclusiveMinimum": 0},
        "metric": {"type": ["string", "null"]},
        "ensemble": {"type": "boolean"},
        "mode": {"enum": ["on_data_change", "interval"]},
        "interval_hours": {"type": "number", "exclusiveMinimum": 0},
        "time_col": {"type": ["string", "null"]},
        "horizon": {"type": ["integer", "null"]},
        # The column naming a row's entity: kept out of the features, and kept whole
        # across the holdout so a repeated entity is not scored against itself.
        "groups": {"type": ["string", "null"]},
        "enabled": {"type": "boolean"},
    },
}


# -- surface definitions -----------------------------------------------------------


def _metrics_surface() -> Surface:
    """Metrics: MetricSet manifests as ``metrics/<name>.yaml``, keyed by name."""

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        out = {}
        for view in _get(client, "/api/metrics"):
            manifest = {k: v for k, v in view.items() if k != "sourceCertified"}
            out[manifest["name"]] = manifest
        return out

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        _post(client, "/api/metrics", {**body, "name": name})

    return Surface(
        name="metrics",
        folder="metrics",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {n: n for n in remote(c)},
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, name: _delete(c, f"/api/metrics/{name}"),
        schema=_metric_file_schema(),
        explode=_explode_metrics,
        references=_metric_references,
    )


def _queries_surface() -> Surface:
    """Saved explore queries as ``queries/<name>.sql`` with a source header line."""

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        # The comparable body is exactly what a file round-trips to (sql + source); the
        # name is the map key / file stem, not part of the body.
        out = {}
        for view in _get(client, "/api/explore/queries"):
            source = view.get("source_id") or None
            out[view["name"]] = {
                "sql": (view.get("sql") or "").strip(),
                "source_id": source,
            }
        return out

    def serialize(body: dict[str, Any]) -> str:
        source = body.get("source_id") or "warehouse"
        return f"-- source: {source}\n{body['sql'].rstrip()}\n"

    def deserialize(text: str) -> dict[str, Any]:
        source: str | None = None
        lines = text.splitlines()
        if lines and lines[0].startswith("-- source:"):
            source = lines[0].split(":", 1)[1].strip()
            lines = lines[1:]
        if source in (None, "", "warehouse"):
            source = None
        return {"sql": "\n".join(lines).strip(), "source_id": source}

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        payload = {"name": name, "sql": body["sql"], "source_id": body.get("source_id")}
        if existing is not None:
            payload["id"] = existing
        _post(client, "/api/explore/queries", payload)

    return Surface(
        name="queries",
        folder="queries",
        ext=".sql",
        remote=remote,
        remote_ids=lambda c: {
            v["name"]: v["id"] for v in _get(c, "/api/explore/queries")
        },
        serialize=serialize,
        deserialize=deserialize,
        push=push,
        delete=lambda c, qid: _delete(c, f"/api/explore/queries/{qid}"),
        schema=_QUERIES_SCHEMA,
    )


def _notebooks_surface() -> Surface:
    """Notebooks as ``notebooks/<name>.ipynb`` (outputs stripped), keyed by name.

    A notebook that already exists is updated in place. Replacing it by deleting and
    re-importing -- which is what this used to do -- changed its id out from under
    anything referring to it and reset its history.
    """

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        out = {}
        for nb in _get(client, "/api/notebooks"):
            ipynb = _get(client, f"/api/notebooks/{nb['id']}/export")
            out[nb["name"]] = _strip_notebook(ipynb)
        return out

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        if existing is not None:
            _put(
                client,
                f"/api/notebooks/{existing}/ipynb",
                {"ipynb": body, "name": name},
            )
            return
        _post(client, "/api/notebooks/import", {"name": name, "ipynb": body})

    return Surface(
        name="notebooks",
        folder="notebooks",
        ext=".ipynb",
        remote=remote,
        remote_ids=lambda c: {nb["name"]: nb["id"] for nb in _get(c, "/api/notebooks")},
        serialize=lambda body: json.dumps(body, indent=1, ensure_ascii=False) + "\n",
        deserialize=lambda text: json.loads(text),
        push=push,
        delete=lambda c, nid: _delete(c, f"/api/notebooks/{nid}"),
        schema=_NOTEBOOKS_SCHEMA,
    )


def _dashboards_surface() -> Surface:
    """Dashboards: DashboardSpec manifests as ``dashboards/<name>.yaml``, by name."""

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        out = {}
        for d in _get(client, "/api/dashboards"):
            detail = _get(client, f"/api/dashboards/{d['id']}")
            out[d["name"]] = detail["spec"]  # the draft DashboardSpec manifest
        return out

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        manifest = {**body, "name": name}
        if existing is not None:
            _put(client, f"/api/dashboards/{existing}", manifest)
        else:
            _post(client, "/api/dashboards", manifest)

    return Surface(
        name="dashboards",
        folder="dashboards",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {d["name"]: d["id"] for d in _get(c, "/api/dashboards")},
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, did: _delete(c, f"/api/dashboards/{did}"),
        schema=_dashboard_file_schema(),
    )


def _schedules_surface() -> Surface:
    """Orchestration schedules as ``schedules/<name>.yaml``, upserted by name."""
    _FIELDS = ("selection", "mode", "cron", "dataset", "enabled")

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        out = {}
        for s in _get(client, "/api/orchestration/schedules"):
            out[s["name"]] = {k: s.get(k) for k in _FIELDS}
        return out

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        payload = {"name": name, **{k: body.get(k) for k in _FIELDS if k in body}}
        _post(client, "/api/orchestration/schedules", payload)

    return Surface(
        name="schedules",
        folder="schedules",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {
            s["name"]: s["id"] for s in _get(c, "/api/orchestration/schedules")
        },
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, sid: _delete(c, f"/api/orchestration/schedules/{sid}"),
        schema=_SCHEDULES_SCHEMA,
    )


def _workflows_surface() -> Surface:
    """Procedural workflows as ``workflows/<name>.yaml`` (a DAG of gated steps)."""

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        return {
            w["name"]: {"steps": w["steps"]}
            for w in _get(client, "/api/orchestration/workflows")
        }

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        payload = {"id": existing, "name": name, "steps": body.get("steps", [])}
        _post(client, "/api/orchestration/workflows", payload)

    return Surface(
        name="workflows",
        folder="workflows",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {
            w["name"]: w["id"] for w in _get(c, "/api/orchestration/workflows")
        },
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, wid: _delete(c, f"/api/orchestration/workflows/{wid}"),
        schema=_WORKFLOWS_SCHEMA,
    )


def _checks_surface() -> Surface:
    """Data-quality checks as ``checks/<asset>.<name>.yaml`` (keyed by asset + name)."""
    _FIELDS = ("asset", "name", "expr", "severity", "enabled")

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        out = {}
        for c in _get(client, "/api/orchestration/checks"):
            out[f"{c['asset']}.{c['name']}"] = {k: c.get(k) for k in _FIELDS}
        return out

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        payload = {"id": existing, **{k: body[k] for k in _FIELDS if k in body}}
        _post(client, "/api/orchestration/checks", payload)

    return Surface(
        name="checks",
        folder="checks",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {
            f"{x['asset']}.{x['name']}": x["id"]
            for x in _get(c, "/api/orchestration/checks")
        },
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, cid: _delete(c, f"/api/orchestration/checks/{cid}"),
        schema=_CHECKS_SCHEMA,
    )


def _features_surface() -> Surface:
    """The whole feature store as one file ``features/store.yaml`` (entities + views).

    Entities and views are two object types that reference each other by name, so the
    store is one logical object: ``sync`` writes every entity, then every view (which
    reference entities). Its file body uses the manifest's camelCase keys.
    """

    def _entity_manifest(row: dict[str, Any]) -> dict[str, Any]:
        out = {
            "name": row["name"],
            "joinKey": row.get("join_key") or row.get("joinKey"),
        }
        value_type = row.get("value_type") or row.get("valueType")
        if value_type and value_type != "string":
            out["valueType"] = value_type
        return out

    def _view_manifest(view: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": view["name"],
            "entities": list(view.get("entities", [])),
            "source": view["source"],
        }
        if view.get("timestamp_field"):
            out["timestampField"] = view["timestamp_field"]
        if view.get("ttl_seconds") is not None:
            out["ttlSeconds"] = view["ttl_seconds"]
        if view.get("features"):
            # Each feature already arrives as the spec's own object ({name, dtype,
            # description}), with explicit nulls for whatever is unset. Wrapping it
            # again in {"name": ...} produced {"name": {"name": ...}}, which `sync`
            # rejects -- features[].name must be a string. Drop the nulls instead,
            # which is what Feature.to_manifest does, so pull writes what push accepts.
            out["features"] = [
                {"name": f}
                if isinstance(f, str)
                else {k: v for k, v in f.items() if v is not None}
                for f in view["features"]
            ]
        if view.get("description"):
            out["description"] = view["description"]
        return out

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        entities = [_entity_manifest(e) for e in _get(client, "/api/features/entities")]
        views = [_view_manifest(v) for v in _get(client, "/api/features/views")]
        if not entities and not views:
            return {}
        return {"store": {"entities": entities, "featureViews": views}}

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        for entity in body.get("entities", []):
            _post(
                client,
                "/api/features/entities",
                {
                    "name": entity["name"],
                    "join_key": entity.get("joinKey", entity["name"]),
                    "value_type": entity.get("valueType", "string"),
                },
            )
        for view in body.get("featureViews", []):
            _post(client, "/api/features/views", view)

    return Surface(
        name="features",
        folder="features",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {"store": "store"} if remote(c) else {},
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, _id: None,  # the store is one object; prune is a no-op
        schema=_feature_store_file_schema(),
    )


def _monitors_surface() -> Surface:
    """Anomaly monitors as ``monitors/<name>.yaml``, keyed by name.

    A monitor watches a certified metric or derivation, so the app gates its target on
    create. The app has no update route, so ``sync`` replaces a monitor of the same name
    (delete then create) to stay idempotent. The file carries only the monitor's
    definition; runtime state (last value, incidents) stays app-side.
    """
    _FIELDS = (
        "target_kind",
        "target",
        "method",
        "sensitivity",
        "window",
        "interval_hours",
        "min_value",
        "max_value",
        "config",
    )

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        out = {}
        for v in _get(client, "/api/monitors"):
            out[v["name"]] = {
                "target_kind": v["targetKind"],
                "target": v["target"],
                "method": v["method"],
                "sensitivity": v["sensitivity"],
                "window": v["window"],
                "interval_hours": v["intervalHours"],
                "min_value": v.get("minValue"),
                "max_value": v.get("maxValue"),
                "config": v.get("config") or {},
            }
        return out

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        payload = {"name": name, **{k: body.get(k) for k in _FIELDS}}
        if existing is not None:
            # Updated rather than replaced: deleting a monitor takes its snapshots and
            # incidents with it, and those snapshots are the baseline anomaly detection
            # compares against, so re-creating one to change a threshold would silently
            # reset the detector.
            _put(client, f"/api/monitors/{existing}", payload)
            return
        _post(client, "/api/monitors", payload)

    return Surface(
        name="monitors",
        folder="monitors",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {v["name"]: v["id"] for v in _get(c, "/api/monitors")},
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, mid: _delete(c, f"/api/monitors/{mid}"),
        schema=_MONITORS_SCHEMA,
    )


def _models_surface() -> Surface:
    """Model retrain policies as ``models/<name>.yaml``, keyed by model name.

    The file is a model's continuous-training policy (the training spec the app
    resubmits on a schedule or on data change). ``source_kind`` names what ``dataset``
    refers to: a warehouse ``dataset``, a certified ``derivation``, or a materialized
    ``training_set``.
    """
    _FIELDS = (
        "source_kind",
        "dataset",
        "target",
        "features",
        "task",
        "engine",
        "time_budget",
        "metric",
        "ensemble",
        "mode",
        "interval_hours",
        "time_col",
        "horizon",
        "groups",
        "enabled",
    )

    def _policies(client: httpx.Client) -> list[dict[str, Any]]:
        # _get returns parsed JSON (Any); this endpoint yields a list of policy objects.
        return cast(
            "list[dict[str, Any]]", _get(client, "/api/registry/retrain-policies")
        )

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        # Omit what the policy does not set rather than writing `null`: source_kind is
        # an enum and time_budget/interval_hours are numbers, none of which admit null,
        # so emitting one makes a file this repo's own `plan` reports as invalid.
        return {
            p["model"]: {k: v for k in _FIELDS if (v := p.get(k)) is not None}
            for p in _policies(client)
        }

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        # The app's training routes name the source by its kind (`dataset`/`derivation`/
        # `training_set`), so translate the stored `source_kind` back into that key.
        kind = body.get("source_kind") or "dataset"
        payload = {k: body.get(k) for k in _FIELDS if k != "source_kind"}
        payload.pop("dataset", None)
        payload[kind] = body.get("dataset")
        _put(client, f"/api/registry/models/{name}/retrain", payload)

    return Surface(
        name="models",
        folder="models",
        ext=".yaml",
        remote=remote,
        remote_ids=lambda c: {p["model"]: p["model"] for p in _policies(c)},
        serialize=_yaml_dump,
        deserialize=_yaml_load,
        push=push,
        delete=lambda c, name: _delete(c, f"/api/registry/models/{name}/retrain"),
        schema=_MODELS_SCHEMA,
    )


def _certificates_surface() -> Surface:
    """Signed verification certificates as ``certificates/<name>.json``, keyed by name.

    Derived, read-only state: ``pull`` writes each certified derivation's certificate,
    and ``sync`` pushes it back through the app's verify gate, which rejects a tampered
    file (HTTP 400) or one that no longer matches the derivation (409). There is nothing
    to create app-side, so a derivation with no verification is skipped and prune is a
    no-op. This is what makes the certificate round-trip through pull/sync: an edited
    ``certificates/<name>.json`` fails the sync loudly rather than being applied.
    """

    def remote(client: httpx.Client) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in _get(client, "/api/derivations"):
            name = row["name"]
            path = f"/api/certificates/{name}"
            resp = client.get(path)
            if resp.status_code == 404:
                continue  # the derivation has no verification to certify
            _raise_for(resp, path)
            out[name] = resp.json()
        return out

    def push(
        client: httpx.Client, name: str, body: dict[str, Any], existing: str | None
    ) -> None:
        # The app verifies the submitted certificate: a tampered file 400s and a stale
        # one 409s, so a corrupted certificates/<name>.json fails sync rather than
        # mutating anything (the certificate is derived from the stored run).
        _post(client, f"/api/certificates/{name}", body)

    return Surface(
        name="certificates",
        folder="certificates",
        ext=".json",
        remote=remote,
        remote_ids=lambda c: {
            row["name"]: row["name"] for row in _get(c, "/api/derivations")
        },
        serialize=lambda body: json.dumps(body, indent=2, sort_keys=True) + "\n",
        deserialize=lambda text: cast("dict[str, Any]", json.loads(text)),
        push=push,
        delete=lambda c, _id: None,  # derived state; prune is a no-op
        schema=_CERTIFICATES_SCHEMA,
    )


#: Every surface the engine knows, in a deterministic apply order (referenced-first).
def surfaces() -> list[Surface]:
    """The registered surfaces, in dependency order (metrics before dashboards)."""
    return [
        _metrics_surface(),
        _dashboards_surface(),
        _features_surface(),
        _monitors_surface(),
        _schedules_surface(),
        _workflows_surface(),
        _checks_surface(),
        _models_surface(),
        _queries_surface(),
        _notebooks_surface(),
        _certificates_surface(),
    ]


def schemas() -> dict[str, dict[str, Any]]:
    """Each surface's file-body JSON Schema, keyed by folder name."""
    return {s.folder: s.schema for s in surfaces() if s.schema is not None}


def reload_project(client: httpx.Client) -> dict[str, Any]:
    """Ask the app to re-read derivation source files and declared source data.

    ``sync`` pushes the declarative artifacts (metrics, dashboards, ...) over the API,
    but a derivation is Python and a source is data: both live on the app's disk and
    take effect when the app reads them. This calls the app's reload route so those take
    effect without a restart, completing the repo -> live picture. Returns the app's
    reconciliation summary (added/reloaded/removed derivations, and the source names).
    """
    resp = client.post("/api/project/reload")
    _raise_for(resp, "/api/project/reload")
    return resp.json() if resp.content else {}


# -- engine verbs ------------------------------------------------------------------


def validate(root: Path, surfaces_: list[Surface] | None = None) -> list[str]:
    """Parse every repo file and JSON-Schema-check it; return readable problems."""
    problems: list[str] = []
    for surface in surfaces_ or surfaces():
        try:
            parsed = surface.local(root)
        except SyncError as exc:
            problems.append(str(exc))
            continue
        if surface.schema is not None:
            problems.extend(_schema_problems(surface, parsed))
    return problems


def _schema_problems(surface: Surface, parsed: dict[str, dict[str, Any]]) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return []
    problems = []
    validator = jsonschema.Draft202012Validator(surface.schema or {})
    for name, body in parsed.items():
        for err in validator.iter_errors(body):
            loc = "/".join(str(p) for p in err.path) or "(root)"
            where = f"{surface.folder}/{name}{surface.ext}"
            problems.append(f"{where}: {loc}: {err.message}")
    return problems


def classify(
    surface: Surface,
    local: Mapping[str, dict[str, Any]],
    remote: Mapping[str, dict[str, Any]],
    base: Mapping[str, str],
    *,
    prune: bool = False,
) -> list[Change]:
    """Compare three states for one surface: the repo, the app, and the last pull.

    Two states can only say "these differ". Three say *who* changed, which is the whole
    reason the baseline is recorded. An object differing because somebody edited it in
    the app is drift and should be pulled; one differing because both sides changed is a
    conflict and must not be pushed over.

    With no baseline (a repo assembled by hand, or one predating the state file) this
    degrades to the two-way comparison it replaced. Anything present on both sides and
    differing is reported as an update, which is the old behaviour and the safe reading
    when there is nothing to attribute a change to.
    """
    changes: list[Change] = []
    for name, body in local.items():
        here = _fingerprint(surface, body)
        was = base.get(name)
        if name not in remote:
            # Gone from the app. Deleting it there while the repo still has it is not
            # the same as never pushing it, but both want the same act: push it back.
            changes.append(Change(surface.name, name, "create"))
            continue
        there = _fingerprint(surface, remote[name])
        if here == there:
            changes.append(Change(surface.name, name, "unchanged"))
        elif was is None:
            changes.append(Change(surface.name, name, "update"))
        elif here == was:
            changes.append(Change(surface.name, name, "drift"))
        elif there == was:
            changes.append(Change(surface.name, name, "update"))
        else:
            changes.append(Change(surface.name, name, "conflict"))
    if prune:
        # Only when the caller will actually delete. A plan is a promise about what
        # happens next, so listing removals that `sync` leaves alone would overstate the
        # damage -- and a plan nobody trusts is one nobody reads.
        for name in remote:
            if name not in local:
                changes.append(Change(surface.name, name, "delete"))
    return changes


def plan(
    client: httpx.Client,
    root: Path,
    surfaces_: list[Surface] | None = None,
    *,
    prune: bool = False,
) -> list[Change]:
    """Diff repo against the app; return the changes ``sync`` would make.

    Reads the record of the last pull, so a change made in the app is reported as drift
    rather than as something this repo is about to overwrite. ``prune`` decides whether
    removals are among those changes: a plan states what the matching ``sync`` does, not
    what some other invocation might.
    """
    state = read_state(root)
    changes: list[Change] = []
    for surface in surfaces_ or surfaces():
        changes.extend(
            classify(
                surface,
                surface.local(root),
                surface.remote(client),
                state.get(surface.name, {}),
                prune=prune and surface.manages(root),
            )
        )
    return changes


def sync(
    client: httpx.Client,
    root: Path,
    *,
    prune: bool = False,
    force: bool = False,
    surfaces_: list[Surface] | None = None,
) -> list[Change]:
    """Apply repo -> app: create/update every local object; delete extras when pruning.

    Returns the changes applied. ``prune=False`` (default) never deletes app objects the
    repo omits, so a partial repo is safe; ``prune=True`` makes the app match the repo.

    Refuses outright if anything changed both here and in the app since the last pull.
    That is the lost-update problem, and overwriting silently is the one outcome nobody
    can recover from: the other change is simply gone, with nothing recording that it
    existed. ``force`` overwrites anyway, for somebody who has looked and decided.
    """
    state = read_state(root)
    applied: list[Change] = []
    conflicts: list[Change] = []
    plans: list[
        tuple[Surface, dict[str, dict[str, Any]], dict[str, str], list[Change], bool]
    ]
    plans = []
    for surface in surfaces_ or surfaces():
        local = surface.local(root)
        ids = surface.remote_ids(client)
        remote = surface.remote(client)
        prunes = prune and surface.manages(root)
        found = classify(
            surface, local, remote, state.get(surface.name, {}), prune=prunes
        )
        conflicts.extend(c for c in found if c.action == "conflict")
        plans.append((surface, local, ids, found, prunes))

    # Nothing is applied until every surface has been classified. A partial sync that
    # stopped at the first conflict would leave the deployment in a state neither the
    # repo nor the app describes, and whoever fixed it would have to work out how far it
    # got.
    if conflicts and not force:
        listed = ", ".join(f"{c.surface}/{c.name}" for c in conflicts)
        raise SyncError(
            f"changed here and in the app since the last pull: {listed}. "
            "Pull to take the app's version, resolve the file by hand, or pass force "
            "to overwrite the app."
        )

    for surface, local, ids, found, prunes in plans:
        actions = {change.name: change.action for change in found}
        for name, body in local.items():
            if actions.get(name) in ("unchanged", "drift"):
                # Drift is left alone deliberately: this repo did not change the object,
                # so pushing its older copy would undo somebody's edit in the app for no
                # reason. `plan` reports it and `pull` accepts it.
                continue
            action = "update" if name in ids else "create"
            surface.push(client, name, body, ids.get(name))
            applied.append(Change(surface.name, name, action))
        if prunes:
            for name, obj_id in ids.items():
                if name not in local:
                    surface.delete(client, obj_id)
                    applied.append(Change(surface.name, name, "delete"))

    # The baseline moves to what was just pushed, so an immediately following sync is a
    # no-op rather than reporting everything as changed again.
    if applied:
        _record(client, root, surfaces_)
    return applied


def pull(
    client: httpx.Client, root: Path, surfaces_: list[Surface] | None = None
) -> list[Change]:
    """Write the app's objects to the repo as canonical files; return what changed."""
    state = read_state(root)
    written: list[Change] = []
    for surface in surfaces_ or surfaces():
        remote = surface.remote(client)
        directory = root / surface.folder
        directory.mkdir(parents=True, exist_ok=True)
        existing_files = surface.local(root)
        for name, body in remote.items():
            action = "update" if name in existing_files else "create"
            path = directory / f"{name}{surface.ext}"
            path.write_text(surface.serialize(body), encoding="utf-8")
            written.append(Change(surface.name, name, action))
        # Deleted in the app, and this copy is known to have had it: drop the file.
        # Without the baseline a pull could not tell "deleted upstream" from "authored
        # here and not yet pushed", so it left both, and the next sync put the deleted
        # object back.
        # Only what the baseline knows this copy had. A file authored here and not yet
        # pushed is absent from the baseline, so it is never considered, which is what
        # keeps unpushed work from being read as "deleted upstream".
        for name in state.get(surface.name, {}):
            if name in remote:
                continue
            path = directory / f"{name}{surface.ext}"
            if path.exists():
                path.unlink()
                written.append(Change(surface.name, name, "delete"))
    _record(client, root, surfaces_)
    return written


def _record(
    client: httpx.Client, root: Path, surfaces_: list[Surface] | None = None
) -> None:
    """Write the baseline: what the app holds right now, as this copy last saw it."""
    state: dict[str, dict[str, str]] = {}
    for surface in surfaces_ or surfaces():
        state[surface.name] = {
            name: _fingerprint(surface, body)
            for name, body in surface.remote(client).items()
        }
    # A subset run records only those surfaces, so a `--surface metrics` pull does not
    # claim the rest were seen. Anything already recorded for another surface survives.
    if surfaces_ is not None:
        merged = read_state(root)
        merged.update(state)
        state = merged
    write_state(root, state)
