"""The feature-store capabilities the chat agent drives, as model-shaped text.

Wraps :class:`~elbi.features.FeatureStoreService` so the agent operates the same feature
store a human does: discover feature views, author one, materialize the online store,
and retrieve features online or point-in-time. Serving is gated: a view's source
derivation must be certified, so the gate, not the agent, authorizes a served feature
value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .features import FeatureStoreError, FeatureStoreService

#: Cap a rendered feature sample so a large retrieval stays readable in context.
_MAX_ROWS = 20


class FeatureStoreAgent:
    """Bind a :class:`FeatureStoreService` into the agent's ``fs_*`` capabilities."""

    def __init__(self, service: FeatureStoreService) -> None:
        self._service = service

    def list(self) -> str:
        """List the feature views, their keys, and whether their source is certified."""
        catalog = self._service.catalog()
        if not catalog:
            return "No feature views yet. Define one with define_feature_view."
        lines = []
        for view in catalog:
            keys = ", ".join(view["join_keys"])
            feats = (
                ", ".join(f["name"] for f in view["features"]) or "(all source columns)"
            )
            mark = "certified" if view["certified"] else "UNCERTIFIED source"
            lines.append(
                f"- {view['name']} [{keys}] ← {view['source']} "
                f"({mark}); features: {feats}"
            )
        return "Feature views:\n" + "\n".join(lines)

    def define(
        self,
        name: str,
        entities: Sequence[str],
        source: str,
        features: Sequence[str] | None,
        timestamp_field: str | None,
        ttl_seconds: int | None,
    ) -> str:
        """Define (or replace) a feature view over a certified derivation."""
        manifest: dict[str, Any] = {
            "name": name,
            "entities": list(entities),
            "source": source,
        }
        if features:
            manifest["features"] = [{"name": f} for f in features]
        if timestamp_field:
            manifest["timestampField"] = timestamp_field
        if ttl_seconds:
            manifest["ttlSeconds"] = ttl_seconds
        try:
            self._service.define_feature_view(manifest)
        except FeatureStoreError as exc:
            return f"Not defined: {exc}"
        return (
            f"Defined feature view '{name}' over {source}. "
            "Materialize it with materialize_features, then read it with "
            "get_online_features / get_historical_features."
        )

    def materialize(self, feature_views: Sequence[str] | None) -> str:
        """Refresh the online store with the latest value per entity."""
        try:
            written = self._service.materialize(
                feature_views=list(feature_views) if feature_views else None,
                require_certified=True,
            )
        except FeatureStoreError as exc:
            return f"Not materialized: {exc}"
        parts = ", ".join(f"{view}: {count} keys" for view, count in written.items())
        return f"Materialized {parts or '(nothing)'}."

    def get_online(
        self, features: Sequence[str], entity_rows: Sequence[Mapping[str, Any]]
    ) -> str:
        """Read the latest materialized features for entity rows."""
        rows = self._service.get_online([dict(r) for r in entity_rows], list(features))
        return _render(rows)

    def get_historical(
        self, features: Sequence[str], entity_df: Sequence[Mapping[str, Any]]
    ) -> str:
        """Point-in-time join of features onto an entity dataframe (no leakage)."""
        try:
            rows = self._service.get_historical(
                [dict(r) for r in entity_df], list(features), require_certified=True
            )
        except FeatureStoreError as exc:
            return f"Retrieval failed: {exc}"
        return _render(rows)

    def statistics(self, feature_view: str, set_baseline: bool) -> str:
        """Profile a feature view's columns now, optionally as the drift baseline."""
        try:
            snapshot = self._service.snapshot_statistics(
                feature_view, set_baseline=set_baseline
            )
        except FeatureStoreError as exc:
            return f"Not profiled: {exc}"
        lines = [
            f"- {f['name']}: {f['completeness'] * 100:.0f}% complete, "
            f"{f['distinct']} distinct, range [{f['minimum']}, {f['maximum']}]"
            for f in snapshot["features"]
        ]
        base = " (baseline set)" if snapshot["is_baseline"] else ""
        return (
            f"Profiled '{feature_view}' over {snapshot['row_count']} rows{base}:\n"
            + "\n".join(lines)
        )

    def drift(self, feature_view: str) -> str:
        """Check a feature view's current values for drift against its baseline."""
        try:
            result = self._service.drift(feature_view)
        except FeatureStoreError as exc:
            return f"Drift check failed: {exc}"
        drifted = [c["column"] for c in result["columns"] if c["drifted"]]
        verdict = "DATASET DRIFT" if result["dataset_drift"] else "no dataset drift"
        detail = f"; drifted: {', '.join(drifted)}" if drifted else ""
        return (
            f"{feature_view}: {verdict}, {result['n_drifted']}/{result['n_columns']} "
            f"features moved over {result['n_current_rows']} rows{detail}."
        )

    def check_expectations(self, feature_view: str) -> str:
        """Verify a feature view's values against its attached data contract."""
        try:
            result = self._service.verify_expectations(feature_view)
        except FeatureStoreError as exc:
            return f"Expectation check failed: {exc}"
        failed = [c["name"] for c in result["clauses"] if c["verdict"] != "sound"]
        detail = f"; failing: {', '.join(failed)}" if failed else ""
        return (
            f"{feature_view}: contract {result['verdict'].upper()}, "
            f"{result['n_violated']}/{result['n_clauses']} clauses not sound over "
            f"{result['row_count']} rows{detail}."
        )

    def create_training_set(
        self,
        name: str,
        features: Sequence[str],
        entity_df: Sequence[Mapping[str, Any]],
        label: str | None,
    ) -> str:
        """Materialize a leakage-free point-in-time join into a named training set."""
        try:
            result = self._service.create_training_set(
                name,
                list(features),
                [dict(r) for r in entity_df],
                label=label,
                require_certified=True,
            )
        except FeatureStoreError as exc:
            return f"Not created: {exc}"
        cols = ", ".join(result["columns"]) or "(none)"
        return (
            f"Created training set '{name}': {result['row_count']} rows, "
            f"columns: {cols}. Train on it with train_model (source '{name}', "
            f"kind training_set)."
        )


def _render(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "No rows."
    shown = [dict(r) for r in rows[:_MAX_ROWS]]
    suffix = f"\n…({len(rows) - _MAX_ROWS} more rows)" if len(rows) > _MAX_ROWS else ""
    return json.dumps(shown, default=str) + suffix
