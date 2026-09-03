"""The feature-store service: registry, materialization, and retrieval.

Assembles a :class:`~elbi.features.FeatureStore` from the registry rows in the store,
runs a feature view's source derivation through the project runner (only when certified:
verified features), materializes the latest values into a DB-backed online store, and
serves point-in-time historical joins and online lookups.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from elbi_core import FeatureStore, Runner, profile_columns
from elbi_core.errors import SpecValidationError
from elbi_core.features import (
    FeatureView,
    get_historical_features,
    get_online_features,
    materialize,
)
from elbi_core.features.retrieval import Key
from elbi_core.quality import DataContract, verify_contract
from elbi_core.quality import suggest_contract as suggest_data_contract

from .db import Store

#: Cap the reference row sample retained on a baseline snapshot (mirrors the model
#: monitor's ``reference.csv``): enough to characterize a distribution, bounded storage.
_BASELINE_SAMPLE = 2000


class FeatureStoreError(Exception):
    """A feature-store operation failed for a reason worth showing the caller."""


class DbOnlineStore:
    """An :class:`~elbi.features.OnlineStore` backed by the app store."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def upsert(
        self, feature_view: str, records: Sequence[tuple[Key, dict[str, Any]]]
    ) -> None:
        """Write each key's feature values for a view."""
        self._store.online_upsert(
            feature_view, [(_key_str(key), values) for key, values in records]
        )

    def lookup(
        self, feature_view: str, keys: Sequence[Key]
    ) -> list[dict[str, Any] | None]:
        """Read each key's stored feature values, or ``None`` when absent."""
        return self._store.online_lookup(feature_view, [_key_str(key) for key in keys])


class FeatureStoreService:
    """Registry, materialization, and retrieval over a loaded project."""

    def __init__(
        self,
        store: Store,
        make_runner: Callable[[], Runner],
        is_certified: Callable[[str], bool],
    ) -> None:
        self._store = store
        self._make_runner = make_runner
        self._is_certified = is_certified

    # -- registry ----------------------------------------------------------------
    def define_entity(
        self,
        name: str,
        join_key: str,
        value_type: str = "string",
        description: str | None = None,
    ) -> None:
        """Register (or replace) an entity."""
        self._store.upsert_feature_entity(
            name=name, join_key=join_key, value_type=value_type, description=description
        )

    def define_feature_view(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Validate a feature-view manifest against the registry and persist it."""
        candidate = self._spec().to_manifest()
        candidate["featureViews"] = [
            view
            for view in candidate.get("featureViews", [])
            if view["name"] != manifest.get("name")
        ]
        candidate["featureViews"].append(manifest)
        try:
            FeatureStore.from_manifest(candidate)
        except SpecValidationError as exc:
            raise FeatureStoreError(str(exc)) from exc
        self._store.upsert_feature_view(
            name=str(manifest["name"]),
            entities=list(manifest["entities"]),
            source=str(manifest["source"]),
            timestamp_field=manifest.get("timestampField"),
            ttl_seconds=manifest.get("ttlSeconds"),
            features=list(manifest.get("features", [])),
            description=manifest.get("description"),
        )
        return self.get_feature_view(str(manifest["name"]))

    def list_entities(self) -> list[dict[str, Any]]:
        """The registered entities."""
        return [
            {
                "name": row.name,
                "join_key": row.join_key,
                "value_type": row.value_type,
                "description": row.description,
            }
            for row in self._store.list_feature_entities()
        ]

    def catalog(self) -> list[dict[str, Any]]:
        """The registered feature views for discovery, with certification and freshness.

        Feature-view names are globally unique, so the online-store freshness for every
        view is fetched in one grouped query and matched by name.
        """
        store = self._spec()
        freshness = self._store.online_stats_all()
        catalog: list[dict[str, Any]] = []
        for view in store.feature_views:
            n_keys, last = freshness.get(view.name, (0, None))
            catalog.append(
                {
                    "name": view.name,
                    "entities": list(view.entities),
                    "join_keys": list(store.join_keys(view)),
                    "source": view.source,
                    "certified": self._is_certified(view.source),
                    "timestamp_field": view.timestamp_field,
                    "ttl_seconds": int(view.ttl.total_seconds()) if view.ttl else None,
                    "features": [_feature_dict(f) for f in view.features],
                    "description": view.description,
                    "n_online_keys": n_keys,
                    "last_materialized_at": last.isoformat() if last else None,
                }
            )
        return catalog

    def get_feature_view(self, name: str) -> dict[str, Any]:
        """A single feature view's registry entry."""
        row = self._store.get_feature_view(name)
        if row is None:
            raise FeatureStoreError(f"feature view {name!r} not found")
        for entry in self.catalog():
            if entry["name"] == name:
                return entry
        raise FeatureStoreError(f"feature view {name!r} not found")

    def view_detail(self, name: str) -> dict[str, Any]:
        """A feature view's registry entry plus its monitoring history, in one payload.

        Composes the catalog entry (schema, certification, freshness) with the recorded
        statistics, drift, and expectation checks and whether a contract is attached, so
        the detail page loads the whole view in a single request.
        """
        entry = self.get_feature_view(name)
        contract = self.get_contract(name)
        return {
            **entry,
            "has_contract": contract is not None,
            "statistics": self.statistics(name),
            "drift": self.drift_history(name),
            "expectations": self.expectations_history(name),
        }

    def delete_feature_view(self, name: str, permanent: bool = False) -> None:
        """Move a feature view to trash, or erase it immediately with ``permanent``."""
        if self._store.get_feature_view(name) is None:
            raise FeatureStoreError(f"feature view {name!r} not found")
        if permanent:
            self._store.erase_feature_view(name)
        else:
            self._store.trash_feature_view(name)

    # -- materialization & retrieval ---------------------------------------------
    def materialize(
        self,
        feature_views: Sequence[str] | None = None,
        require_certified: bool = False,
    ) -> dict[str, int]:
        """Refresh the online store with the latest value per entity for each view.

        ``require_certified`` (the agent's guardrail) refuses a view whose source is not
        a certified derivation; the trusted human path leaves it off.
        """
        store = self._spec()
        self._resolve_views(store, feature_views, require_certified=require_certified)
        return materialize(
            store,
            DbOnlineStore(self._store),
            run=self._run,
            feature_views=feature_views,
        )

    def get_online(
        self, entity_rows: Sequence[dict[str, Any]], features: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Read the latest materialized features for entity rows."""
        store = self._spec()
        return get_online_features(
            store, entity_rows, features, online=DbOnlineStore(self._store)
        )

    def get_historical(
        self,
        entity_df: Sequence[dict[str, Any]],
        features: Sequence[str],
        require_certified: bool = False,
    ) -> list[dict[str, Any]]:
        """Point-in-time join of features onto an entity dataframe.

        ``require_certified`` (the agent's guardrail) refuses a view whose source is not
        a certified derivation; the trusted human path leaves it off.
        """
        store = self._spec()
        view_names = [f.split(":")[0] for f in features]
        self._resolve_views(store, view_names, require_certified=require_certified)
        try:
            return get_historical_features(store, entity_df, features, run=self._run)
        except ValueError as exc:
            raise FeatureStoreError(str(exc)) from exc

    # -- monitoring: statistics & drift ------------------------------------------
    def snapshot_statistics(
        self, view_name: str, set_baseline: bool = False
    ) -> dict[str, Any]:
        """Profile a view's feature columns now and record the snapshot.

        With ``set_baseline`` the snapshot also retains a capped row sample as the
        reference a later :meth:`drift` check compares against.
        """
        store = self._spec()
        view = self._require_view(store, view_name)
        rows = self._run(view.source)
        features = store.exposed_features(view, list(rows[0]) if rows else [])
        projected = _project(rows, features)
        profiles = [p.to_dict() for p in profile_columns(projected)]
        self._store.add_feature_statistics(
            view_name,
            row_count=len(rows),
            profiles=profiles,
            is_baseline=set_baseline,
            sample=projected[:_BASELINE_SAMPLE] if set_baseline else None,
        )
        return {
            "feature_view": view_name,
            "row_count": len(rows),
            "is_baseline": set_baseline,
            "features": profiles,
        }

    def statistics(self, view_name: str, limit: int = 50) -> list[dict[str, Any]]:
        """A view's profile snapshots, newest first."""
        self._require_view(self._spec(), view_name)
        return [
            {
                "id": row.id,
                "at": row.at.isoformat(),
                "row_count": row.row_count,
                "is_baseline": row.is_baseline,
                "features": json.loads(row.profiles_json),
            }
            for row in self._store.list_feature_statistics(view_name, limit)
        ]

    def drift(self, view_name: str) -> dict[str, Any]:
        """Compare a view's current values against its baseline snapshot.

        Reuses the same Evidently data-drift engine the model monitor uses, so a
        feature view and a model report drift the same way. Requires a baseline
        (:meth:`snapshot_statistics` with ``set_baseline``) and the ``ml`` extra.
        """
        store = self._spec()
        view = self._require_view(store, view_name)
        baseline = self._store.get_feature_baseline(view_name)
        if baseline is None or not baseline.sample_json:
            raise FeatureStoreError(
                f"feature view {view_name!r} has no baseline; snapshot one with "
                "set_baseline first"
            )
        reference = json.loads(baseline.sample_json)
        current = self._run(view.source)
        features = store.exposed_features(view, list(current[0]) if current else [])
        report = _data_drift(reference, _project(current, features), features)
        summary = report.summary()
        self._store.add_feature_drift(
            view_name,
            n_columns=summary["n_columns"],
            n_drifted=summary["n_drifted"],
            share_drifted=summary["share_drifted"],
            dataset_drift=summary["dataset_drift"],
            n_current_rows=len(current),
            report=summary,
        )
        return {"feature_view": view_name, "n_current_rows": len(current), **summary}

    def drift_history(self, view_name: str, limit: int = 50) -> list[dict[str, Any]]:
        """A view's recorded drift checks, newest first."""
        self._require_view(self._spec(), view_name)
        return [
            {
                "id": row.id,
                "at": row.at.isoformat(),
                "n_columns": row.n_columns,
                "n_drifted": row.n_drifted,
                "share_drifted": row.share_drifted,
                "dataset_drift": row.dataset_drift,
                "n_current_rows": row.n_current_rows,
                "columns": json.loads(row.report_json).get("columns", []),
            }
            for row in self._store.list_feature_drift(view_name, limit)
        ]

    # -- training sets -----------------------------------------------------------
    def create_training_set(
        self,
        name: str,
        features: Sequence[str],
        entity_df: Sequence[dict[str, Any]],
        label: str | None = None,
        require_certified: bool = False,
    ) -> dict[str, Any]:
        """Materialize a point-in-time join into a named, reusable training set.

        The join is leakage-free (each feature is the latest value at or before the
        entity row's timestamp); persisting the result makes it a first-class training
        source a model can train on, the same way it trains on a bound dataset.
        """
        store = self._spec()
        view_names = [f.split(":")[0] for f in features]
        self._resolve_views(store, view_names, require_certified=require_certified)
        try:
            rows = get_historical_features(
                store, list(entity_df), list(features), run=self._run
            )
        except ValueError as exc:
            raise FeatureStoreError(str(exc)) from exc
        self._store.upsert_training_set(
            name=name, features=list(features), label=label, rows=rows
        )
        return {
            "name": name,
            "row_count": len(rows),
            "features": list(features),
            "label": label,
            "columns": list(rows[0]) if rows else [],
        }

    def list_training_sets(self) -> list[dict[str, Any]]:
        """The materialized training sets, newest first."""
        return [
            {
                "name": row.name,
                "created_at": row.created_at.isoformat(),
                "row_count": row.row_count,
                "features": json.loads(row.features_json),
                "label": row.label,
            }
            for row in self._store.list_training_sets()
        ]

    def get_training_set(self, name: str, sample: int = 20) -> dict[str, Any]:
        """A training set's metadata and a small row sample."""
        row = self._store.get_training_set(name)
        if row is None:
            raise FeatureStoreError(f"training set {name!r} not found")
        rows = json.loads(row.rows_json)
        return {
            "name": row.name,
            "created_at": row.created_at.isoformat(),
            "row_count": row.row_count,
            "features": json.loads(row.features_json),
            "label": row.label,
            "columns": list(rows[0]) if rows else [],
            "sample": rows[:sample],
        }

    def delete_training_set(self, name: str) -> None:
        """Remove a materialized training set."""
        if self._store.get_training_set(name) is None:
            raise FeatureStoreError(f"training set {name!r} not found")
        self._store.delete_training_set(name)

    # -- expectations: data contracts on feature values --------------------------
    def set_contract(
        self, view_name: str, manifest: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Attach a data contract to a view (or clear it with an empty manifest)."""
        self._require_view(self._spec(), view_name)
        if not manifest:
            self._store.set_feature_view_contract(view_name, None)
            return None
        try:
            contract = DataContract.from_manifest(manifest)
        except SpecValidationError as exc:
            raise FeatureStoreError(str(exc)) from exc
        stored = contract.to_manifest()
        self._store.set_feature_view_contract(view_name, json.dumps(stored))
        return stored

    def get_contract(self, view_name: str) -> dict[str, Any] | None:
        """A view's attached data contract, or ``None`` when it has none."""
        row = self._store.get_feature_view(view_name)
        if row is None:
            raise FeatureStoreError(f"feature view {view_name!r} not found")
        return json.loads(row.contract_json) if row.contract_json else None

    def suggest_contract(self, view_name: str) -> dict[str, Any]:
        """Profile a view's feature columns and propose a contract to start from."""
        store = self._spec()
        view = self._require_view(store, view_name)
        rows = self._run(view.source)
        features = store.exposed_features(view, list(rows[0]) if rows else [])
        return suggest_data_contract(_project(rows, features)).to_manifest()

    def verify_expectations(self, view_name: str) -> dict[str, Any]:
        """Check a view's current feature values against its attached data contract.

        Reuses the platform's contract engine (the same three-valued verdict and located
        violations the verification oracle speaks), so a feature view's data quality is
        held to the same bar as everything else.
        """
        store = self._spec()
        view = self._require_view(store, view_name)
        row = self._store.get_feature_view(view_name)
        if row is None or not row.contract_json:
            raise FeatureStoreError(
                f"feature view {view_name!r} has no data contract; set one first"
            )
        contract = DataContract.from_manifest(json.loads(row.contract_json))
        rows = self._run(view.source)
        features = store.exposed_features(view, list(rows[0]) if rows else [])
        report = verify_contract(_project(rows, features), contract)
        attestation = report.attestation()
        n_violated = len(report.violated_clauses)
        self._store.add_feature_expectation(
            view_name,
            verdict=report.verdict,
            n_clauses=len(report.clauses),
            n_violated=n_violated,
            row_count=report.row_count,
            report=attestation,
        )
        return {
            "feature_view": view_name,
            "verdict": report.verdict,
            "n_clauses": len(report.clauses),
            "n_violated": n_violated,
            "row_count": report.row_count,
            "clauses": attestation["clauses"],
            "violations": attestation["violations"],
        }

    def expectations_history(
        self, view_name: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """A view's recorded contract checks, newest first."""
        self._require_view(self._spec(), view_name)
        return [
            {
                "id": r.id,
                "at": r.at.isoformat(),
                "verdict": r.verdict,
                "n_clauses": r.n_clauses,
                "n_violated": r.n_violated,
                "row_count": r.row_count,
            }
            for r in self._store.list_feature_expectations(view_name, limit)
        ]

    # -- internals ---------------------------------------------------------------
    def _require_view(self, store: FeatureStore, name: str) -> FeatureView:
        view = store.feature_view(name)
        if view is None:
            raise FeatureStoreError(f"unknown feature view {name!r}")
        return view

    def _spec(self) -> FeatureStore:
        # Entities are resolved project-wide rather than per caller. Every read of a
        # feature view rebuilds this spec and cross-checks that its entities exist.
        entities = [
            {
                "name": row.name,
                "joinKey": row.join_key,
                "valueType": row.value_type,
                **({"description": row.description} if row.description else {}),
            }
            for row in self._store.list_feature_entities()
        ]
        rows = self._store.list_feature_views()
        views = [_view_manifest(row) for row in rows]
        return FeatureStore.from_manifest(
            {
                "specVersion": "1.0",
                "kind": "FeatureStore",
                "entities": entities,
                "featureViews": views,
            }
        )

    def _resolve_views(
        self,
        store: FeatureStore,
        view_names: Sequence[str] | None,
        require_certified: bool,
    ) -> None:
        """Validate that the named views exist, and (for the agent) are certified."""
        names = (
            list(view_names)
            if view_names is not None
            else [v.name for v in store.feature_views]
        )
        uncertified = []
        for name in names:
            view = store.feature_view(name)
            if view is None:
                raise FeatureStoreError(f"unknown feature view {name!r}")
            if require_certified and not self._is_certified(view.source):
                uncertified.append(f"{name} (source {view.source})")
        if uncertified:
            raise FeatureStoreError(
                "feature views bind uncertified derivations: " + ", ".join(uncertified)
            )

    def _run(self, name: str) -> list[dict[str, Any]]:
        artifact = self._make_runner().run(name)
        value = artifact.value
        if not (isinstance(value, list) and all(isinstance(r, dict) for r in value)):
            raise FeatureStoreError(f"derivation {name!r} did not produce feature rows")
        return list(value)


def _feature_dict(feature: Any) -> dict[str, Any]:
    """A view's declared feature as ``{name, dtype, description}`` for the registry."""
    return {
        "name": feature.name,
        "dtype": feature.dtype,
        "description": feature.description,
    }


def _project(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Restrict each row to a view's feature ``columns`` for profiling and drift."""
    return [{column: row.get(column) for column in columns} for row in rows]


def _data_drift(
    reference: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Any:
    """Run the shared drift engine, mapping its absence/complaints to our error type."""
    from elbi_core.errors import ModelError

    try:
        from elbi_core.ml.drift import data_drift
    except ImportError as exc:  # the ml extra is not installed
        raise FeatureStoreError(
            "drift monitoring requires the 'ml' extra: pip install 'elbi-app[ml]'"
        ) from exc
    try:
        return data_drift(list(reference), list(current), columns=list(columns))
    except ModelError as exc:  # too few rows, no shared columns, evidently missing
        raise FeatureStoreError(str(exc)) from exc


def _view_manifest(row: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "name": row.name,
        "entities": json.loads(row.entities_json),
        "source": row.source,
        "features": json.loads(row.features_json),
    }
    if row.timestamp_field:
        manifest["timestampField"] = row.timestamp_field
    if row.ttl_seconds:
        manifest["ttlSeconds"] = row.ttl_seconds
    if row.description:
        manifest["description"] = row.description
    return manifest


def _key_str(key: Key) -> str:
    return json.dumps(list(key), default=str)
