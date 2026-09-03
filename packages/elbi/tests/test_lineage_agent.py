"""The chat agent driving lineage/catalog through the runtime.

Binds a LineageAgent onto a Workspace and exercises the runtime dispatch: search the
catalog, trace a node's lineage, and run impact analysis by bare name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi.db import open_store
from elbi.lineage import LineageService
from elbi.lineage_agent import LineageAgent
from elbi_agent import Workspace
from elbi_agent.llm import ToolCall
from elbi_agent.runtime import _dispatch_lineage
from elbi_core import Artifact, Context, Dataset, Registry, derivation, serve
from elbi_core.registry import use_registry


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table([{"x": 1}])

    return registry


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()
    service = LineageService(
        store=store,
        registry_provider=lambda: registry,
        dataset_names=lambda: ["sales"],
    )
    agent = LineageAgent(service)
    ws = Workspace(datasets={})
    ws.lin_catalog = agent.catalog
    ws.lin_lineage = agent.lineage
    ws.lin_impact = agent.impact
    return ws


def _call(ws: Workspace, tool: str, **args: object) -> str:
    return _dispatch_lineage(ToolCall(id="1", name=tool, arguments=dict(args)), ws)


def test_search_catalog(workspace: Workspace) -> None:
    out = _call(workspace, "search_catalog", query="revenue")
    assert "derivation: revenue" in out


def test_trace_lineage_by_bare_name(workspace: Workspace) -> None:
    out = _call(workspace, "trace_lineage", node="revenue")
    assert "Feeds from: dataset sales" in out


def test_impact_analysis_by_bare_name(workspace: Workspace) -> None:
    out = _call(workspace, "impact_analysis", node="sales")
    assert "affects" in out and "revenue" in out


@pytest.mark.parametrize(
    ("tool", "expected"),
    [("trace_lineage", "Feeds from: dataset sales"), ("impact_analysis", "affects")],
)
def test_a_lineage_question_builds_the_graph_once(
    tmp_path: Path, tool: str, expected: str
) -> None:
    """One tool call, one graph.

    Tracing asks three questions and impact analysis two, each previously through a
    service method that built its own graph. ``_graph`` calls ``registry_provider`` once
    per build, so the provider counts builds.
    """
    store = open_store(f"sqlite:{tmp_path / 'agent.db'}")
    registry = _registry()
    builds = 0

    def counting_provider() -> Registry:
        nonlocal builds
        builds += 1
        return registry

    ws = Workspace(datasets={})
    agent = LineageAgent(
        LineageService(
            store=store,
            registry_provider=counting_provider,
            dataset_names=lambda: ["sales"],
        )
    )
    ws.lin_lineage = agent.lineage
    ws.lin_impact = agent.impact

    out = _call(ws, tool, node="revenue" if tool == "trace_lineage" else "sales")

    assert expected in out
    assert builds == 1


def test_bare_workspace_declines() -> None:
    out = _dispatch_lineage(
        ToolCall(id="1", name="search_catalog", arguments={"query": ""}),
        Workspace(datasets={}),
    )
    assert "not available" in out
