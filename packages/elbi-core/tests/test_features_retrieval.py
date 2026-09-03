"""Tests for feature retrieval: point-in-time joins, online lookup, materialization.

The point-in-time cases are the ones that matter: a historical join must return the
latest feature value at or before each entity's event timestamp and never a later one
(the anti-leakage guarantee), and must respect the view's TTL. The online cases pin that
materialization keeps the newest value per key and lookup reads it back.
"""

from __future__ import annotations

import pytest

from elbi_core.features import (
    FeatureStore,
    InMemoryOnlineStore,
    get_historical_features,
    get_online_features,
    materialize,
)

# Feature history for two users; user "a" has three snapshots over January.
SOURCE = [
    {"user_id": "a", "event_timestamp": "2024-01-01", "clicks": 1},
    {"user_id": "a", "event_timestamp": "2024-01-10", "clicks": 5},
    {"user_id": "a", "event_timestamp": "2024-01-20", "clicks": 9},
    {"user_id": "b", "event_timestamp": "2024-01-05", "clicks": 2},
]


def _run(name: str) -> list[dict]:
    assert name == "user_stats_src"
    return [dict(r) for r in SOURCE]


def _store(ttl_seconds: int | None = None, timestamped: bool = True) -> FeatureStore:
    view: dict = {
        "name": "user_stats",
        "entities": ["user"],
        "source": "user_stats_src",
        "features": [{"name": "clicks"}],
    }
    if timestamped:
        view["timestampField"] = "event_timestamp"
    if ttl_seconds is not None:
        view["ttlSeconds"] = ttl_seconds
    return FeatureStore.from_manifest(
        {
            "specVersion": "1.0",
            "kind": "FeatureStore",
            "entities": [{"name": "user", "joinKey": "user_id"}],
            "featureViews": [view],
        }
    )


def test_point_in_time_join_excludes_future_values() -> None:
    store = _store()
    entity_df = [
        {"user_id": "a", "event_timestamp": "2024-01-15"},
        {"user_id": "a", "event_timestamp": "2024-01-02"},
        {"user_id": "b", "event_timestamp": "2024-01-01"},
    ]
    out = get_historical_features(store, entity_df, ["user_stats:clicks"], run=_run)
    # As-of 2024-01-15 the latest snapshot is 2024-01-10 (clicks=5); the 2024-01-20
    # value is in the future and must not leak.
    assert out[0]["clicks"] == 5
    # As-of 2024-01-02 only the 2024-01-01 snapshot exists.
    assert out[1]["clicks"] == 1
    # No snapshot for "b" at or before 2024-01-01.
    assert out[2]["clicks"] is None


def test_ttl_excludes_stale_values() -> None:
    store = _store(ttl_seconds=7 * 86400)
    entity_df = [{"user_id": "a", "event_timestamp": "2024-01-09"}]
    out = get_historical_features(store, entity_df, ["user_stats:clicks"], run=_run)
    # The only snapshot at or before 2024-01-09 is 2024-01-01, which is >7 days old.
    assert out[0]["clicks"] is None


def test_ttl_keeps_recent_values() -> None:
    store = _store(ttl_seconds=7 * 86400)
    entity_df = [{"user_id": "a", "event_timestamp": "2024-01-13"}]
    out = get_historical_features(store, entity_df, ["user_stats:clicks"], run=_run)
    # 2024-01-10 is within 7 days of 2024-01-13.
    assert out[0]["clicks"] == 5


def test_ttl_honored_with_numeric_timestamps() -> None:
    # Epoch-style integer timestamps: the TTL window is measured in seconds.
    store = _store(ttl_seconds=7 * 86400)
    source = [
        {"user_id": "a", "event_timestamp": 0, "clicks": 1},
        {"user_id": "a", "event_timestamp": 10 * 86400, "clicks": 5},
    ]
    entity_df = [{"user_id": "a", "event_timestamp": 8 * 86400}]
    out = get_historical_features(
        store, entity_df, ["user_stats:clicks"], run=lambda _: source
    )
    # The latest snapshot at or before day 8 is day 0, which is >7 days stale.
    assert out[0]["clicks"] is None


def test_ttl_raises_on_non_temporal_timestamps() -> None:
    # A view that declares a TTL but whose timestamp column parses to neither a datetime
    # nor a number cannot express a window; enforcing must fail loudly, not silently
    # serve values the TTL was meant to exclude.
    store = _store(ttl_seconds=7 * 86400)
    source = [{"user_id": "a", "event_timestamp": "week-1", "clicks": 1}]
    entity_df = [{"user_id": "a", "event_timestamp": "week-2"}]
    with pytest.raises(ValueError, match="TTL"):
        get_historical_features(
            store, entity_df, ["user_stats:clicks"], run=lambda _: source
        )


def test_non_temporal_timestamps_without_ttl_do_not_raise() -> None:
    # Without a TTL there is no window to measure, so non-temporal timestamps join by
    # lexicographic order without error.
    store = _store()
    source = [{"user_id": "a", "event_timestamp": "week-1", "clicks": 1}]
    entity_df = [{"user_id": "a", "event_timestamp": "week-2"}]
    out = get_historical_features(
        store, entity_df, ["user_stats:clicks"], run=lambda _: source
    )
    assert out[0]["clicks"] == 1


def test_untimestamped_view_joins_current_value() -> None:
    store = _store(timestamped=False)
    entity_df = [{"user_id": "a"}, {"user_id": "b"}, {"user_id": "c"}]
    out = get_historical_features(store, entity_df, ["user_stats:clicks"], run=_run)
    assert out[0]["clicks"] == 9  # last row seen for "a"
    assert out[1]["clicks"] == 2
    assert out[2]["clicks"] is None


def test_materialize_keeps_latest_and_online_reads_it_back() -> None:
    store = _store()
    online = InMemoryOnlineStore()
    written = materialize(store, online, run=_run)
    assert written == {"user_stats": 2}
    out = get_online_features(
        store,
        [{"user_id": "a"}, {"user_id": "b"}, {"user_id": "c"}],
        ["user_stats:clicks"],
        online=online,
    )
    assert out[0]["clicks"] == 9  # newest snapshot for "a"
    assert out[1]["clicks"] == 2
    assert out[2]["clicks"] is None


def _typed_store(value_type: str) -> FeatureStore:
    return FeatureStore.from_manifest(
        {
            "specVersion": "1.0",
            "kind": "FeatureStore",
            "entities": [
                {"name": "student", "joinKey": "grade", "valueType": value_type}
            ],
            "featureViews": [
                {
                    "name": "grade_stats",
                    "entities": ["student"],
                    "source": "grade_src",
                    "features": [{"name": "pass_rate"}],
                }
            ],
        }
    )


def test_online_lookup_normalizes_key_type() -> None:
    # The source materializes an integer grade; the caller looks up with the string
    # "7". Without coercion to the entity's declared integer type the tuple keys differ
    # ((7,) vs ("7",)) and the lookup misses.
    store = _typed_store("integer")
    online = InMemoryOnlineStore()
    materialize(store, online, run=lambda _: [{"grade": 7, "pass_rate": 0.8}])
    out = get_online_features(
        store, [{"grade": "7"}], ["grade_stats:pass_rate"], online=online
    )
    assert out[0]["pass_rate"] == 0.8


def test_historical_join_normalizes_key_type() -> None:
    # The offline path must coerce identically, so a string entity key joins an integer
    # source key.
    store = _typed_store("integer")
    out = get_historical_features(
        store,
        [{"grade": "7"}],
        ["grade_stats:pass_rate"],
        run=lambda _: [{"grade": 7, "pass_rate": 0.8}],
    )
    assert out[0]["pass_rate"] == 0.8


def test_online_and_offline_agree_on_the_latest_row() -> None:
    # The same derivation feeds both paths, so the newest online value equals the
    # historical value joined at "now": no train/serve skew.
    store = _store()
    online = InMemoryOnlineStore()
    materialize(store, online, run=_run)
    online_out = get_online_features(
        store, [{"user_id": "a"}], ["user_stats:clicks"], online=online
    )
    offline_out = get_historical_features(
        store,
        [{"user_id": "a", "event_timestamp": "2024-12-31"}],
        ["user_stats:clicks"],
        run=_run,
    )
    assert online_out[0]["clicks"] == offline_out[0]["clicks"] == 9
