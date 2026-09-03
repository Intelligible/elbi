"""Feature store: entities and feature views over certified derivations.

Defines the spec (:class:`FeatureStore`, :class:`FeatureView`, :class:`Entity`,
:class:`Feature`), validation against the bundled Feature Store Spec schema, and the
:mod:`retrieval` layer (point-in-time historical joins, online lookup, and
materialization) that reads feature values by running a view's source derivation.
"""

from __future__ import annotations

from .retrieval import (
    InMemoryOnlineStore,
    OnlineStore,
    get_historical_features,
    get_online_features,
    materialize,
)
from .spec import (
    EVENT_TIMESTAMP,
    FEATURE_STORE_SPEC_VERSION,
    Entity,
    Feature,
    FeatureStore,
    FeatureView,
    is_valid_feature_store,
    load_feature_store_schema,
    validate_feature_store,
)

__all__ = [
    "EVENT_TIMESTAMP",
    "FEATURE_STORE_SPEC_VERSION",
    "Entity",
    "Feature",
    "FeatureStore",
    "FeatureView",
    "InMemoryOnlineStore",
    "OnlineStore",
    "get_historical_features",
    "get_online_features",
    "is_valid_feature_store",
    "load_feature_store_schema",
    "materialize",
    "validate_feature_store",
]
