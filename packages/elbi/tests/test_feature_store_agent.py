"""The chat agent driving the feature store through the runtime, as a human would.

Binds a FeatureStoreAgent onto a Workspace and exercises the runtime dispatch: define a
view, materialize, and retrieve online and point-in-time. The certification gate lives
here (the agent's guardrail): the agent refuses to materialize a view whose source is
not certified, while the human API (tested elsewhere) is trusted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from elbi.db import open_store
from elbi.feature_store_agent import FeatureStoreAgent
from elbi.features import FeatureStoreService
from elbi_agent import Workspace
from elbi_agent.llm import ToolCall
from elbi_agent.runtime import _dispatch_features
from elbi_core import Artifact, Context, Registry, Runner, derivation, serve
from elbi_core.registry import use_registry

SOURCE = [
    {"user_id": "a", "event_timestamp": "2024-01-01", "clicks": 1},
    {"user_id": "a", "event_timestamp": "2024-01-20", "clicks": 9},
    {"user_id": "b", "event_timestamp": "2024-01-05", "clicks": 2},
]


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def user_stats_src(ctx: Context) -> Artifact:
            return Artifact.table([dict(r) for r in SOURCE])

    registry.register(
        replace(registry.get("user_stats_src"), name="draft_src", status="proposed")
    )
    return registry


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Workspace, FeatureStoreService]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()
    service = FeatureStoreService(
        store=store,
        make_runner=lambda: Runner(registry),
        is_certified=lambda name: any(
            d.name == name and d.is_certified for d in registry
        ),
    )
    service.define_entity(name="user", join_key="user_id")
    agent = FeatureStoreAgent(service)
    ws = Workspace(datasets={})
    ws.fs_list = agent.list
    ws.fs_define = agent.define
    ws.fs_materialize = agent.materialize
    ws.fs_online = agent.get_online
    ws.fs_historical = agent.get_historical
    ws.fs_statistics = agent.statistics
    ws.fs_drift = agent.drift
    ws.fs_expectations = agent.check_expectations
    ws.fs_training_set = agent.create_training_set
    return ws, service


def _call(ws: Workspace, tool: str, **args: object) -> str:
    return _dispatch_features(ToolCall(id="1", name=tool, arguments=dict(args)), ws)


def _define(ws: Workspace, source: str = "user_stats_src") -> str:
    return _call(
        ws,
        "define_feature_view",
        name="user_stats",
        entities=["user"],
        source=source,
        features=["clicks"],
        timestamp_field="event_timestamp",
    )


def test_agent_defines_materializes_and_retrieves(
    workspace: tuple[Workspace, FeatureStoreService],
) -> None:
    ws, _ = workspace
    assert "No feature views yet" in _call(ws, "list_feature_views")
    assert "Defined feature view" in _define(ws)
    assert "user_stats" in _call(ws, "list_feature_views")
    assert "2 keys" in _call(ws, "materialize_features")

    online = _call(
        ws,
        "get_online_features",
        features=["user_stats:clicks"],
        entity_rows=[{"user_id": "a"}],
    )
    assert '"clicks": 9' in online  # newest snapshot

    historical = _call(
        ws,
        "get_historical_features",
        features=["user_stats:clicks"],
        entity_df=[{"user_id": "a", "event_timestamp": "2024-01-10"}],
    )
    assert '"clicks": 1' in historical  # as-of 2024-01-10, not the future value 9


def test_agent_refuses_uncertified_source(
    workspace: tuple[Workspace, FeatureStoreService],
) -> None:
    ws, _ = workspace
    _define(ws, source="draft_src")  # a proposed derivation
    out = _call(ws, "materialize_features")
    assert "Not materialized" in out and "uncertified" in out


def test_agent_profiles_a_view(
    workspace: tuple[Workspace, FeatureStoreService],
) -> None:
    ws, _ = workspace
    _define(ws)
    out = _call(ws, "profile_feature_view", feature_view="user_stats")
    assert "Profiled 'user_stats'" in out
    assert "clicks:" in out


@pytest.mark.filterwarnings("ignore")
def test_agent_reports_drift_needs_a_baseline(
    workspace: tuple[Workspace, FeatureStoreService],
) -> None:
    ws, _ = workspace
    _define(ws)
    out = _call(ws, "check_feature_drift", feature_view="user_stats")
    assert "Drift check failed" in out and "baseline" in out


def test_agent_checks_expectations(
    workspace: tuple[Workspace, FeatureStoreService],
) -> None:
    ws, service = workspace
    _define(ws)
    # Without a contract the tool says so; with one it reports the verdict.
    assert "no data contract" in _call(
        ws, "check_feature_expectations", feature_view="user_stats"
    )
    service.set_contract(
        "user_stats",
        {
            "specVersion": "1.0",
            "kind": "DataContract",
            "fields": [{"name": "clicks", "type": "integer"}],
        },
    )
    out = _call(ws, "check_feature_expectations", feature_view="user_stats")
    assert "contract SOUND" in out


def test_agent_creates_a_training_set(
    workspace: tuple[Workspace, FeatureStoreService],
) -> None:
    ws, service = workspace
    _define(ws)
    out = _call(
        ws,
        "create_training_set",
        name="churn_ts",
        features=["user_stats:clicks"],
        entity_df=[{"user_id": "a", "event_timestamp": "2024-01-15", "label": "1"}],
        label="label",
    )
    assert "Created training set 'churn_ts'" in out
    assert service.get_training_set("churn_ts")["row_count"] == 1


def test_bare_workspace_declines() -> None:
    out = _dispatch_features(
        ToolCall(id="1", name="list_feature_views", arguments={}),
        Workspace(datasets={}),
    )
    assert "not available" in out
    # The monitoring tools decline on a bare workspace too.
    assert "not available" in _dispatch_features(
        ToolCall(id="2", name="check_feature_drift", arguments={}),
        Workspace(datasets={}),
    )
