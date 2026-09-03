"""The feature-store spec: entities and feature views over certified derivations.

An entity names a join key. A feature view groups feature columns produced by a
certified derivation, keyed by one or more entities and (optionally) an event-time
column that enables point-in-time joins. The store is the collection of both, validated
against the bundled Feature Store Spec schema on construction.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema

from ..errors import SpecValidationError

#: The Feature Store Spec version this SDK implements (MAJOR.MINOR).
FEATURE_STORE_SPEC_VERSION = "1.0"

#: Reserved event-timestamp column on an entity dataframe for point-in-time retrieval.
EVENT_TIMESTAMP = "event_timestamp"

VALUE_TYPES = ("string", "integer", "float", "boolean")


@dataclass(frozen=True)
class Entity:
    """A join key that features are keyed by."""

    name: str
    join_key: str
    value_type: str = "string"
    description: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``entity`` object."""
        out: dict[str, Any] = {"name": self.name, "joinKey": self.join_key}
        if self.value_type != "string":
            out["valueType"] = self.value_type
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Entity:
        """Parse a spec ``entity`` object."""
        return cls(
            name=str(data["name"]),
            join_key=str(data["joinKey"]),
            value_type=str(data.get("valueType", "string")),
            description=_opt_str(data.get("description")),
        )


@dataclass(frozen=True)
class Feature:
    """A single feature column exposed by a view."""

    name: str
    dtype: str | None = None
    description: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``feature`` object."""
        out: dict[str, Any] = {"name": self.name}
        if self.dtype is not None:
            out["dtype"] = self.dtype
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Feature:
        """Parse a spec ``feature`` object."""
        return cls(
            name=str(data["name"]),
            dtype=_opt_str(data.get("dtype")),
            description=_opt_str(data.get("description")),
        )


@dataclass(frozen=True)
class FeatureView:
    """A group of features sourced from a certified derivation, keyed by entities.

    ``source`` is the derivation whose rows carry the join keys, the optional
    ``timestamp_field`` event time, and the feature columns. ``ttl`` bounds how far a
    point-in-time join looks back from each entity timestamp. ``features`` restricts the
    exposed columns; when empty, every non-key, non-timestamp column is exposed.
    """

    name: str
    entities: tuple[str, ...]
    source: str
    timestamp_field: str | None = None
    ttl: timedelta | None = None
    features: tuple[Feature, ...] = ()
    description: str | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The explicitly declared feature column names."""
        return tuple(f.name for f in self.features)

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a spec ``featureView`` object."""
        out: dict[str, Any] = {
            "name": self.name,
            "entities": list(self.entities),
            "source": self.source,
        }
        if self.timestamp_field is not None:
            out["timestampField"] = self.timestamp_field
        if self.ttl is not None:
            out["ttlSeconds"] = int(self.ttl.total_seconds())
        if self.features:
            out["features"] = [f.to_manifest() for f in self.features]
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> FeatureView:
        """Parse a spec ``featureView`` object."""
        ttl_seconds = data.get("ttlSeconds")
        return cls(
            name=str(data["name"]),
            entities=tuple(str(e) for e in data["entities"]),
            source=str(data["source"]),
            timestamp_field=_opt_str(data.get("timestampField")),
            ttl=timedelta(seconds=int(ttl_seconds)) if ttl_seconds else None,
            features=tuple(Feature.from_manifest(f) for f in data.get("features", ())),
            description=_opt_str(data.get("description")),
        )


@dataclass(frozen=True)
class FeatureStore:
    """A registry of entities and feature views.

    Construct from a validated manifest with :meth:`from_manifest`, or build
    programmatically and call :meth:`validate`.
    """

    entities: tuple[Entity, ...] = ()
    feature_views: tuple[FeatureView, ...] = ()

    def entity(self, name: str) -> Entity | None:
        """The entity named ``name``, or ``None``."""
        return next((e for e in self.entities if e.name == name), None)

    def feature_view(self, name: str) -> FeatureView | None:
        """The feature view named ``name``, or ``None``."""
        return next((v for v in self.feature_views if v.name == name), None)

    def join_keys(self, view: FeatureView) -> tuple[str, ...]:
        """The join-key columns a view is keyed by, from its entities in order."""
        keys: list[str] = []
        for entity_name in view.entities:
            entity = self.entity(entity_name)
            if entity is not None and entity.join_key not in keys:
                keys.append(entity.join_key)
        return tuple(keys)

    def key_value_types(self, view: FeatureView) -> tuple[str, ...]:
        """The declared value type of each join key, aligned with :meth:`join_keys`.

        Retrieval coerces keys to these types so a lookup with ``grade="7"`` matches a
        materialized ``grade=7``; without it the tuple keys differ by type and miss.
        """
        types: dict[str, str] = {}
        for entity_name in view.entities:
            entity = self.entity(entity_name)
            if entity is not None and entity.join_key not in types:
                types[entity.join_key] = entity.value_type
        return tuple(types.get(k, "string") for k in self.join_keys(view))

    def exposed_features(self, view: FeatureView, columns: Sequence[str]) -> list[str]:
        """The feature columns a view exposes given its source's ``columns``.

        Declared features when present, else every source column that is not a join key
        or the timestamp field.
        """
        if view.features:
            return list(view.feature_names)
        reserved = {*self.join_keys(view), view.timestamp_field}
        return [c for c in columns if c not in reserved]

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to a full, spec-conformant FeatureStore manifest."""
        out: dict[str, Any] = {
            "specVersion": FEATURE_STORE_SPEC_VERSION,
            "kind": "FeatureStore",
        }
        if self.entities:
            out["entities"] = [e.to_manifest() for e in self.entities]
        if self.feature_views:
            out["featureViews"] = [v.to_manifest() for v in self.feature_views]
        return out

    def validate(self) -> None:
        """Validate this store against the spec.

        Raises:
            SpecValidationError: if the store is not conformant.
        """
        validate_feature_store(self.to_manifest())

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> FeatureStore:
        """Parse and validate a FeatureStore manifest.

        Raises:
            SpecValidationError: if the manifest does not conform to the spec.
        """
        validate_feature_store(data)
        return cls(
            entities=tuple(Entity.from_manifest(e) for e in data.get("entities", ())),
            feature_views=tuple(
                FeatureView.from_manifest(v) for v in data.get("featureViews", ())
            ),
        )


@lru_cache(maxsize=1)
def load_feature_store_schema() -> dict[str, Any]:
    """Return the bundled Feature Store JSON Schema as a dict."""
    resource = files("elbi_core") / "spec" / "feature_store.schema.json"
    schema: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return schema


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = load_feature_store_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_feature_store(manifest: dict[str, Any]) -> None:
    """Validate a FeatureStore manifest against the spec and its invariants.

    Checks JSON-Schema conformance, then the cross-references the schema cannot express:
    unique names, and every feature view's entities resolving to a defined entity.

    Raises:
        SpecValidationError: with one message per violation.
    """
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(e.path))
    messages = [_format_error(error) for error in errors]
    if not messages:
        messages.extend(_reference_errors(manifest))
    if messages:
        raise SpecValidationError(messages)


def is_valid_feature_store(manifest: dict[str, Any]) -> bool:
    """Return whether a manifest conforms, without raising."""
    try:
        validate_feature_store(manifest)
    except SpecValidationError:
        return False
    return True


def _reference_errors(manifest: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    entities = manifest.get("entities", [])
    views = manifest.get("featureViews", [])
    messages.extend(_duplicate_errors("entity name", entities, "name"))
    messages.extend(_duplicate_errors("feature view name", views, "name"))
    defined = {str(e["name"]) for e in entities}
    for view in views:
        for entity_name in view["entities"]:
            if str(entity_name) not in defined:
                messages.append(
                    f"featureViews/{view['name']}: unknown entity {entity_name!r}"
                )
    return messages


def _duplicate_errors(label: str, items: list[Any], key: str) -> list[str]:
    seen: set[str] = set()
    messages: list[str] = []
    for item in items:
        value = str(item.get(key, ""))
        if value in seen:
            messages.append(f"duplicate {label}: {value!r}")
        seen.add(value)
    return messages


def _format_error(error: jsonschema.ValidationError) -> str:
    location = "/".join(str(part) for part in error.path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
