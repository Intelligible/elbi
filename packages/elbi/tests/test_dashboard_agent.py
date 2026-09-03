"""The chat agent driving dashboards through the runtime, as a human would.

Binds a DashboardAgent onto a Workspace and exercises the runtime's dashboard tool
dispatch: seeing the certified derivations it can bind, authoring a dashboard, reading
it back, and hitting the same certification gate the UI enforces on publish. A bare
workspace (no store) declines the tools rather than erroring.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from elbi.dashboard_agent import DashboardAgent
from elbi.dashboards import DashboardService
from elbi.db import open_store
from elbi_agent import Workspace
from elbi_agent.llm import ToolCall
from elbi_agent.runtime import _dispatch_dashboard
from elbi_core import (
    Artifact,
    Context,
    Registry,
    Runner,
    derivation,
    param,
    serve,
)
from elbi_core.registry import use_registry


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table([{"region": "west", "revenue": 10}])

        @derivation(
            params={"n": param.integer(required=False, default=1)}, serve=serve.table()
        )
        def regions(ctx: Context) -> Artifact:
            return Artifact.table([{"region": "west"}])

    registry.register(replace(registry.get("regions"), name="draft", status="proposed"))
    return registry


def _service(tmp_path: Path) -> DashboardService:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()
    return DashboardService(
        store=store,
        make_runner=lambda: Runner(registry),
        certified_catalog=lambda: [
            {
                "name": d.name,
                "title": d.description or d.name,
                "params": d.to_manifest().get("params", {}),
                "served": d.is_served,
            }
            for d in registry
            if d.is_certified
        ],
    )


def _ws(tmp_path: Path) -> Workspace:
    agent = DashboardAgent(_service(tmp_path))
    ws = Workspace(datasets={})
    ws.dash_list = agent.list
    ws.dash_sources = agent.sources
    ws.dash_read = agent.read
    ws.dash_write = agent.write
    ws.dash_publish = agent.publish
    return ws


def _call(ws: Workspace, name: str, **args: object) -> str:
    return _dispatch_dashboard(ToolCall(id="1", name=name, arguments=dict(args)), ws)


def _id_from(write_output: str) -> str:
    """Recover the dashboard id the agent surfaces in a write_dashboard result."""
    match = re.search(r"id ([0-9a-f-]+)", write_output)
    assert match is not None, write_output
    return match.group(1)


def _spec(bind: str = "revenue") -> dict:
    return {
        "specVersion": "1.0",
        "kind": "Dashboard",
        "name": "sales",
        "title": "Sales",
        "pages": [
            {
                "name": "main",
                "widgets": [
                    {
                        "id": "t",
                        "type": "table",
                        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                        "bind": {"derivation": bind},
                    }
                ],
            }
        ],
    }


def test_sources_lists_only_certified(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    out = _call(ws, "dashboard_sources")
    assert "revenue" in out and "regions" in out
    assert "draft" not in out  # a proposed derivation cannot back a widget


def test_author_read_and_publish(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    assert "No dashboards yet" in _call(ws, "list_dashboards")

    created = _call(ws, "write_dashboard", spec=_spec())
    assert "Saved" in created

    listed = _call(ws, "list_dashboards")
    assert "Sales" in listed

    # The agent reads a dashboard the way it reads a notebook, to edit it.
    dashboard_id = _id_from(created)
    read = _call(ws, "read_dashboard", dashboard_id=dashboard_id)
    assert "table" in read and "revenue" in read

    published = _call(ws, "publish_dashboard", dashboard_id=dashboard_id)
    assert "Published" in published


def test_publish_gate_refuses_uncertified(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    created = _call(ws, "write_dashboard", spec=_spec(bind="draft"))
    dashboard_id = _id_from(created)
    out = _call(ws, "publish_dashboard", dashboard_id=dashboard_id)
    assert "Not published" in out and "uncertified" in out


def test_invalid_spec_is_returned_for_self_correction(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    bad = _spec()
    bad["pages"] = []
    out = _call(ws, "write_dashboard", spec=bad)
    assert "Invalid dashboard" in out


def test_bare_workspace_declines(tmp_path: Path) -> None:
    out = _dispatch_dashboard(
        ToolCall(id="1", name="list_dashboards", arguments={}), Workspace(datasets={})
    )
    assert "not available" in out


@pytest.mark.parametrize(
    "name",
    ["list_dashboards", "dashboard_sources", "read_dashboard", "write_dashboard"],
)
def test_tools_are_advertised(name: str) -> None:
    from elbi_agent.runtime import _TOOLSPECS

    assert name in {spec.name for spec in _TOOLSPECS}
