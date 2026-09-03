"""Tests for the FeatureStore spec: manifest round-tripping and validation.

A valid manifest must round-trip through the dataclasses unchanged, and the checks the
JSON Schema cannot express (unique names and feature views referencing defined entities)
must be enforced. Each negative case reverts one rule and asserts the violation.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from elbi_core import FeatureStore
from elbi_core.errors import SpecValidationError
from elbi_core.features import is_valid_feature_store, validate_feature_store


def _manifest() -> dict:
    return {
        "specVersion": "1.0",
        "kind": "FeatureStore",
        "entities": [
            {"name": "user", "joinKey": "user_id", "valueType": "string"},
        ],
        "featureViews": [
            {
                "name": "user_stats",
                "entities": ["user"],
                "source": "user_stats_src",
                "timestampField": "event_timestamp",
                "ttlSeconds": 86400,
                "features": [{"name": "clicks", "dtype": "integer"}],
            }
        ],
    }


def test_manifest_round_trips() -> None:
    store = FeatureStore.from_manifest(_manifest())
    view = store.feature_view("user_stats")
    assert view is not None
    assert store.join_keys(view) == ("user_id",)
    assert view.ttl == timedelta(days=1)
    assert view.feature_names == ("clicks",)
    assert FeatureStore.from_manifest(store.to_manifest()) == store


def test_exposed_features_defaults_to_non_key_non_timestamp_columns() -> None:
    manifest = _manifest()
    del manifest["featureViews"][0]["features"]
    store = FeatureStore.from_manifest(manifest)
    view = store.feature_view("user_stats")
    assert view is not None
    columns = ["user_id", "event_timestamp", "clicks", "views"]
    assert store.exposed_features(view, columns) == ["clicks", "views"]


def test_accepts_the_example() -> None:
    assert is_valid_feature_store(_manifest())


def test_feature_view_referencing_unknown_entity_rejected() -> None:
    manifest = _manifest()
    manifest["featureViews"][0]["entities"] = ["ghost"]
    with pytest.raises(SpecValidationError) as err:
        validate_feature_store(manifest)
    assert any("unknown entity 'ghost'" in m for m in err.value.messages)


def test_duplicate_entity_name_rejected() -> None:
    manifest = _manifest()
    manifest["entities"].append({"name": "user", "joinKey": "uid"})
    with pytest.raises(SpecValidationError) as err:
        validate_feature_store(manifest)
    assert any("duplicate entity name" in m for m in err.value.messages)


def test_duplicate_feature_view_name_rejected() -> None:
    manifest = _manifest()
    manifest["featureViews"].append(dict(manifest["featureViews"][0]))
    with pytest.raises(SpecValidationError) as err:
        validate_feature_store(manifest)
    assert any("duplicate feature view name" in m for m in err.value.messages)


def test_schema_rejects_unknown_field() -> None:
    manifest = _manifest()
    manifest["featureViews"][0]["unexpected"] = True
    assert not is_valid_feature_store(manifest)


def test_feature_view_requires_at_least_one_entity() -> None:
    manifest = _manifest()
    manifest["featureViews"][0]["entities"] = []
    assert not is_valid_feature_store(manifest)
