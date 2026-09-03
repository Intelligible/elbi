"""The chat agent driving orchestration through the runtime.

Binds an OrchestrationAgent onto a Workspace and exercises the runtime dispatch: report
stale assets, then materialize them in dependency order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi.db import open_store
from elbi.orchestration import OrchestrationService
from elbi.orchestration_agent import OrchestrationAgent
from elbi_agent import Workspace
from elbi_agent.llm import ToolCall
from elbi_agent.runtime import _dispatch_orchestration
from elbi_core import (
    Artifact,
    Context,
    Dataset,
    Registry,
    Runner,
    derivation,
    serve,
)
from elbi_core.config import DataBindings
from elbi_core.registry import use_registry


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "sales.csv").write_text("amount\n10\n20\n", encoding="utf-8")
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("sales").rows)

    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    service = OrchestrationService(
        store=store,
        registry_provider=lambda: registry,
        make_runner=lambda: Runner(
            registry,
            bindings=DataBindings(bindings={"sales": "sales.csv"}),
            base_dir=tmp_path,
        ),
        dataset_names=lambda: ["sales"],
    )
    agent = OrchestrationAgent(service)
    ws = Workspace(datasets={})
    ws.orch_status = agent.status
    ws.orch_materialize = agent.materialize
    return ws


def _call(ws: Workspace, tool: str, **args: object) -> str:
    return _dispatch_orchestration(
        ToolCall(id="1", name=tool, arguments=dict(args)), ws
    )


def test_status_reports_stale(workspace: Workspace) -> None:
    out = _call(workspace, "asset_status")
    assert "revenue" in out
    assert "need materializing" in out


def test_materialize_runs_stale(workspace: Workspace) -> None:
    out = _call(workspace, "materialize_assets", selection="stale", assets="")
    assert "materialized revenue" in out
    # A second call finds it fresh.
    again = _call(workspace, "materialize_assets", selection="stale", assets="")
    assert "materialized" not in again or "revenue" not in again


def test_bare_workspace_declines() -> None:
    out = _dispatch_orchestration(
        ToolCall(id="1", name="asset_status", arguments={}), Workspace(datasets={})
    )
    assert "not available" in out
