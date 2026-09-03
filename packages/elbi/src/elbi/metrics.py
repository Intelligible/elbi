"""The metrics service: define, query, and interchange semantic-layer metrics.

Assembles a :class:`~elbi.metrics.MetricSet` from the metric rows in the store,
gates a simple metric's source on certification (verified numbers), and resolves a
metric query by running its source derivation and aggregating with the DuckDB engine.
Import and export speak the OSI standard, so metrics move between elbi and any
OSI-aware tool.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from elbi_core import MetricSet, compile_metric, from_osi, resolve_metric, to_osi
from elbi_core.errors import ElbiError, SpecValidationError
from elbi_core.metrics import Filter, Metric
from elbi_core.versioning import hash_json

from .db import Store
from .duplicate import copy_identifier, copy_label

Rows = list[dict[str, Any]]


class MetricServiceError(Exception):
    """A metric operation failed for a reason worth showing the caller."""


class MetricService:
    """Define, list, query, and interchange metrics over a loaded project."""

    def __init__(
        self,
        store: Store,
        is_certified: Callable[[str], bool],
        load_source: Callable[[str], Rows],
    ) -> None:
        self._store = store
        # is_certified gates a metric's source at definition time; load_source runs a
        # certified source derivation to its rows for resolution.
        self._is_certified = is_certified
        self._load_source = load_source

    def define(self, manifest: dict[str, Any], author: str = "human") -> dict[str, Any]:
        """Validate a metric against the set, gate its source, and persist it.

        Also appends a definition version (unless the definition is unchanged), so the
        metric carries a change log. ``author`` records who made the change (a human, or
        ``"agent"`` on the agent path).

        Raises:
            MetricServiceError: if the metric is malformed, a ratio operand is unknown,
                or a simple metric's source derivation is not certified.
        """
        try:
            metric = Metric.from_manifest(manifest)
        except (KeyError, TypeError) as exc:
            raise MetricServiceError(f"malformed metric: {exc}") from exc
        others = tuple(m for m in self._set().metrics if m.name != metric.name)
        try:
            MetricSet(metrics=(*others, metric)).validate()
        except SpecValidationError as exc:
            raise MetricServiceError(str(exc)) from exc
        certified = metric.source is None or self._is_certified(metric.source)
        if metric.type == "simple" and not certified:
            raise MetricServiceError(
                f"source derivation {metric.source!r} is not certified; a metric "
                "serves only verified numbers"
            )
        canonical = metric.to_manifest()
        self._store.upsert_metric(
            name=metric.name, manifest_json=json.dumps(canonical), source=metric.source
        )
        self._store.append_metric_version(
            name=metric.name,
            manifest_json=json.dumps(canonical),
            definition_hash=hash_json(canonical),
            author=author,
            verdict="sound" if certified else None,
        )
        return self._view(metric)

    def duplicate(self, name: str) -> dict[str, Any]:
        """Copy a metric definition into a new one owned by the caller.

        A duplicate is a new metric, not a new version of an old one: it gets its own
        name and its own version history starting at 1.

        Raises:
            MetricServiceError: if the metric does not exist, or if the copy cannot be
                defined -- notably when its source derivation is no longer certified,
                since a copy carries no verdict over and has to earn its own.
        """
        row = self._store.get_metric(name)
        if row is None:
            raise MetricServiceError(f"metric {name!r} not found")
        manifest: dict[str, Any] = json.loads(row.manifest_json)
        # A metric's name is its primary key, so the free-name check has to clear the
        # whole namespace rather than a subset of it.
        taken = {metric.name for metric in self._store.list_metrics()}
        manifest["name"] = copy_identifier(name, taken)
        if manifest.get("label"):
            manifest["label"] = copy_label(str(manifest["label"]), ())
        # Through ``define`` rather than the store directly: it validates against
        # the whole metric set, holds the source to the certified gate, and records
        # the first version. ``copied_from`` follows separately because a metric
        # manifest admits no properties beyond the spec's own.
        view = self.define(manifest)
        self._store.set_metric_copied_from(str(manifest["name"]), name)
        return {**view, "copiedFrom": name}

    def list_metrics(self) -> list[dict[str, Any]]:
        """The caller's metrics, each with its source's certification state."""
        return [self._view(metric) for metric in self._set().metrics]

    def get(self, name: str) -> dict[str, Any] | None:
        """A metric's view, or ``None`` when there is no such metric."""
        row = self._store.get_metric(name)
        if row is None:
            return None
        view = self._view(Metric.from_manifest(json.loads(row.manifest_json)))
        return {**view, "copiedFrom": row.copied_from}

    def delete(self, name: str, permanent: bool = False) -> bool:
        """Move a metric to trash, or erase it immediately with ``permanent``.

        Returns whether it existed.
        """
        if permanent:
            return self._store.erase_metric(name)
        return self._store.trash_metric(name)

    def query(
        self,
        name: str,
        group_by: Sequence[str] = (),
        grain: str | None = None,
        filters: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Resolve a metric to grouped, aggregated rows over its certified source.

        Raises:
            MetricServiceError: if the metric or a referenced column is unknown, a
                dimension is not declared, or a source derivation is not certified.
        """
        metric_set = self._set()
        parsed = [Filter.from_manifest(f) for f in filters]
        try:
            result = resolve_metric(
                metric_set,
                name,
                self._load_source,
                group_by=list(group_by),
                grain=grain,
                filters=parsed,
            )
        except ElbiError as exc:
            raise MetricServiceError(str(exc)) from exc
        return {"columns": result.columns, "rows": result.rows}

    def overview(self) -> list[dict[str, Any]]:
        """Each metric's current value and (for time-series metrics) a small trend.

        Powers the catalog's sparkline-and-current-value rows. Best-effort per metric: a
        metric whose source is uncertified or errs yields ``value: None`` rather than
        failing the whole overview.
        """
        out: list[dict[str, Any]] = []
        metric_set = self._set()
        for metric in metric_set.metrics:
            time_column = (
                metric.time_dimension.column if metric.time_dimension else None
            )
            entry: dict[str, Any] = {"name": metric.name, "value": None, "series": []}
            try:
                result = resolve_metric(
                    metric_set,
                    metric.name,
                    self._load_source,
                    group_by=[time_column] if time_column else [],
                    grain=metric.time_dimension.grain
                    if metric.time_dimension
                    else None,
                )
            except ElbiError:
                out.append(entry)
                continue
            rows = result.rows
            values = [
                r[metric.name]
                for r in rows
                if isinstance(r.get(metric.name), int | float)
            ]
            if values:
                entry["value"] = values[-1]
            if time_column:
                entry["series"] = [
                    {"t": str(r.get(time_column)), "v": r.get(metric.name)}
                    for r in rows
                    if isinstance(r.get(metric.name), int | float)
                ]
            out.append(entry)
        return out

    def compile_sql(
        self,
        name: str,
        group_by: Sequence[str] = (),
        grain: str | None = None,
        filters: Sequence[dict[str, Any]] = (),
    ) -> str:
        """The SQL a metric query compiles to: the generated-SQL trust affordance."""
        try:
            return compile_metric(
                self._set(),
                name,
                self._load_source,
                group_by=list(group_by),
                grain=grain,
                filters=[Filter.from_manifest(f) for f in filters],
            )
        except ElbiError as exc:
            raise MetricServiceError(str(exc)) from exc

    def history(self, name: str) -> list[dict[str, Any]]:
        """A metric's definition change log, newest version first.

        Read under the metric's owner rather than the caller: versions are keyed by
        owner, so a grantee asking under their own id would see an empty log.
        """
        row = self._store.get_metric(name)
        if row is None:
            return []
        return [
            {
                "version": version.version,
                "created_at": version.created_at.isoformat(),
                "author": version.author,
                "verdict": version.verdict,
                "change_summary": version.change_summary,
                "definition": json.loads(version.manifest_json),
            }
            for version in self._store.list_metric_versions(name)
        ]

    def export_osi(self) -> dict[str, Any]:
        """Export the metric set as an OSI semantic-model document.

        Raises:
            MetricServiceError: if the set cannot be expressed as conformant OSI. The
                ordinary case is having defined nothing yet: OSI requires a model to
                carry at least one dataset, so an empty set has no valid document, and
                saying that is more use than emitting something no reader accepts.
        """
        try:
            return to_osi(self._set())
        except ElbiError as exc:
            raise MetricServiceError(f"nothing to export as OSI: {exc}") from exc

    def import_osi(
        self, document: dict[str, Any], source_for: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Import an OSI document, defining each metric (subject to the same gates)."""
        try:
            imported = from_osi(document, source_for=source_for)
        except ElbiError as exc:
            raise MetricServiceError(str(exc)) from exc
        return [self.define(metric.to_manifest()) for metric in imported.metrics]

    def _set(self) -> MetricSet:
        return MetricSet(
            metrics=tuple(
                Metric.from_manifest(json.loads(row.manifest_json))
                for row in self._store.list_metrics()
            )
        )

    def _view(self, metric: Metric) -> dict[str, Any]:
        """The wire form of a metric: its manifest plus its source's certified state."""
        certified = metric.source is None or self._is_certified(metric.source)
        return {**metric.to_manifest(), "sourceCertified": certified}
