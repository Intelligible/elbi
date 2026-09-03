"""Feature retrieval: point-in-time historical joins, online lookup, materialization.

The offline path (:func:`get_historical_features`) joins each entity row to the latest
feature values at or before that row's event timestamp, the point-in-time-correct join
that keeps future values out of training data. The online path
(:func:`get_online_features`) reads the latest materialized values by key.
:func:`materialize` refreshes the online store from a view's source derivation. The same
derivation feeds both paths, so there is no train/serve skew.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol

from .spec import EVENT_TIMESTAMP, FeatureStore, FeatureView

#: Runs a derivation by name and returns its rows.
RunFn = Callable[[str], list[dict[str, Any]]]

#: An entity's join-key values, in the view's key order.
Key = tuple[Any, ...]


class OnlineStore(Protocol):
    """A low-latency key-value store of the latest feature values per entity."""

    def upsert(
        self, feature_view: str, records: Sequence[tuple[Key, dict[str, Any]]]
    ) -> None:
        """Write (key, feature values) records for a view, replacing prior values."""
        ...

    def lookup(
        self, feature_view: str, keys: Sequence[Key]
    ) -> list[dict[str, Any] | None]:
        """Return each key's stored feature values, or ``None`` when absent."""
        ...


class InMemoryOnlineStore:
    """An :class:`OnlineStore` backed by a dict, for local use and tests."""

    def __init__(self) -> None:
        self._data: dict[str, dict[Key, dict[str, Any]]] = {}

    def upsert(
        self, feature_view: str, records: Sequence[tuple[Key, dict[str, Any]]]
    ) -> None:
        """Store each key's feature values for a view, replacing prior values."""
        view = self._data.setdefault(feature_view, {})
        for key, values in records:
            view[key] = dict(values)

    def lookup(
        self, feature_view: str, keys: Sequence[Key]
    ) -> list[dict[str, Any] | None]:
        """Return each key's stored feature values, or ``None`` when absent."""
        view = self._data.get(feature_view, {})
        return [view.get(key) for key in keys]


def get_historical_features(
    store: FeatureStore,
    entity_df: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    run: RunFn,
) -> list[dict[str, Any]]:
    """Point-in-time join of ``features`` onto ``entity_df``.

    ``features`` are ``"view:feature"`` references. Each entity row supplies the join
    keys and, for a timestamped view, an ``event_timestamp``; the value joined is the
    latest source row for that key at or before the entity timestamp, within the view's
    TTL. Missing matches yield ``None``. The entity rows are returned augmented with the
    requested feature columns, in input order.
    """
    result = [dict(row) for row in entity_df]
    for view_name, wanted in _group_by_view(features).items():
        view = _require_view(store, view_name)
        keys = store.join_keys(view)
        types = store.key_value_types(view)
        source = run(view.source)
        index = _index_source(source, keys, types, view.timestamp_field)
        for entity_row, out in zip(entity_df, result, strict=True):
            key = _key(entity_row, keys, types)
            match = _as_of(index.get(key, []), entity_row, view)
            for feature in wanted:
                out[feature] = match.get(feature) if match else None
    return result


def get_online_features(
    store: FeatureStore,
    entity_rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    online: OnlineStore,
) -> list[dict[str, Any]]:
    """Read the latest materialized ``features`` for ``entity_rows`` from ``online``.

    ``features`` are ``"view:feature"`` references. Each entity row supplies the join
    keys; the returned rows carry the requested feature columns (``None`` when a key has
    no materialized value).
    """
    result = [dict(row) for row in entity_rows]
    for view_name, wanted in _group_by_view(features).items():
        view = _require_view(store, view_name)
        keys = store.join_keys(view)
        types = store.key_value_types(view)
        lookups = online.lookup(
            view.name, [_key(row, keys, types) for row in entity_rows]
        )
        for values, out in zip(lookups, result, strict=True):
            for feature in wanted:
                out[feature] = values.get(feature) if values else None
    return result


def materialize(
    store: FeatureStore,
    online: OnlineStore,
    *,
    run: RunFn,
    feature_views: Sequence[str] | None = None,
) -> dict[str, int]:
    """Refresh the online store with the latest value per entity for each view.

    Runs each view's source derivation, keeps the newest row per key (by the view's
    timestamp, or the last-seen row when untimestamped), and upserts the exposed feature
    columns. Returns the number of keys written per view.
    """
    names = (
        feature_views
        if feature_views is not None
        else [v.name for v in store.feature_views]
    )
    written: dict[str, int] = {}
    for view_name in names:
        view = _require_view(store, view_name)
        keys = store.join_keys(view)
        types = store.key_value_types(view)
        source = run(view.source)
        columns = list(source[0]) if source else []
        exposed = store.exposed_features(view, columns)
        latest = _latest_per_key(source, keys, types, view.timestamp_field)
        records = [
            (key, {feature: row.get(feature) for feature in exposed})
            for key, row in latest.items()
        ]
        online.upsert(view.name, records)
        written[view.name] = len(records)
    return written


def _group_by_view(features: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for reference in features:
        view, _, feature = reference.partition(":")
        if not feature:
            raise ValueError(f"feature reference {reference!r} must be 'view:feature'")
        grouped.setdefault(view, []).append(feature)
    return grouped


def _require_view(store: FeatureStore, name: str) -> FeatureView:
    view = store.feature_view(name)
    if view is None:
        raise KeyError(f"unknown feature view {name!r}")
    return view


def _key(
    row: dict[str, Any], join_keys: Sequence[str], key_types: Sequence[str]
) -> Key:
    return tuple(
        _coerce_key(row.get(k), t) for k, t in zip(join_keys, key_types, strict=True)
    )


def _coerce_key(value: Any, value_type: str) -> Any:
    """Normalize a join-key value to its entity's declared type.

    The online store keys on tuples, so a materialized ``grade=7`` and a lookup
    ``grade="7"`` must resolve to the same value or the lookup misses. Coerce to the
    declared type, leaving uncoercible or null values untouched (a mismatch there is a
    data problem, not something to silently paper over).
    """
    if value is None:
        return None
    try:
        if value_type == "integer":
            return int(value)
        if value_type == "float":
            return float(value)
        if value_type == "boolean":
            return _to_bool(value)
        return str(value)
    except (TypeError, ValueError):
        return value


def _to_bool(value: Any) -> bool:
    """Coerce a boolean join key, accepting the usual string/number spellings."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return bool(value)


def _index_source(
    source: Sequence[dict[str, Any]],
    keys: Sequence[str],
    key_types: Sequence[str],
    timestamp_field: str | None,
) -> dict[Key, list[tuple[Any, dict[str, Any]]]]:
    """Group source rows by key, each sorted ascending by comparable timestamp."""
    index: dict[Key, list[tuple[Any, dict[str, Any]]]] = {}
    for row in source:
        ts = _comparable(row.get(timestamp_field)) if timestamp_field else 0
        index.setdefault(_key(row, keys, key_types), []).append((ts, row))
    for entries in index.values():
        entries.sort(key=lambda pair: pair[0])
    return index


def _as_of(
    entries: Sequence[tuple[Any, dict[str, Any]]],
    entity_row: dict[str, Any],
    view: FeatureView,
) -> dict[str, Any] | None:
    """The latest source row at or before the entity timestamp, within TTL."""
    if not entries:
        return None
    if view.timestamp_field is None:
        return entries[-1][1]
    cutoff = _comparable(entity_row.get(EVENT_TIMESTAMP))
    stamps = [ts for ts, _ in entries]
    position = bisect.bisect_right(stamps, cutoff)
    if position == 0:
        return None
    ts, row = entries[position - 1]
    if view.ttl is not None and _before_ttl(ts, cutoff, view.ttl):
        return None
    return row


def _latest_per_key(
    source: Sequence[dict[str, Any]],
    keys: Sequence[str],
    key_types: Sequence[str],
    timestamp_field: str | None,
) -> dict[Key, dict[str, Any]]:
    latest: dict[Key, dict[str, Any]] = {}
    best: dict[Key, Any] = {}
    for row in source:
        key = _key(row, keys, key_types)
        if timestamp_field is None:
            latest[key] = row
            continue
        ts = _comparable(row.get(timestamp_field))
        if key not in best or ts >= best[key]:
            best[key] = ts
            latest[key] = row
    return latest


def _comparable(value: Any) -> Any:
    """Coerce a timestamp to a sortable value: datetime, float, or string."""
    if isinstance(value, (int, float, datetime)):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _before_ttl(feature_ts: Any, entity_ts: Any, ttl: timedelta) -> bool:
    """Whether a feature timestamp falls before the entity timestamp minus the TTL.

    Raises:
        ValueError: if the two timestamps are not both datetimes or both numeric, so
            no TTL window can be measured. A view that declares a TTL but whose
            timestamp column does not parse to a datetime or number is misconfigured;
            failing here beats silently serving values the TTL was meant to exclude.
    """
    if isinstance(entity_ts, datetime) and isinstance(feature_ts, datetime):
        return feature_ts < entity_ts - ttl
    if _is_number(entity_ts) and _is_number(feature_ts):
        return bool(feature_ts < entity_ts - ttl.total_seconds())
    raise ValueError(
        f"cannot enforce a TTL on non-temporal timestamps ({entity_ts!r}, "
        f"{feature_ts!r}); a view with a TTL needs a datetime or numeric "
        "timestamp column"
    )


def _is_number(value: Any) -> bool:
    """Whether a value is numeric for TTL arithmetic (bools are not numbers here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
