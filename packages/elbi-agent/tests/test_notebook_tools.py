"""The notebook tools route to the workspace's injected capabilities, or decline.

These pin the agent-side contract: a notebook tool call reaches the bound capability
with the right arguments, argument mistakes come back as actionable text (not a crash),
and a workspace without the capability declines with a clear reason. The capabilities
themselves (the real notebook operations) are tested app-side; here the stubs record
what they were handed.
"""

from __future__ import annotations

from typing import Any

from elbi_agent import ToolCall, Workspace
from elbi_agent.runtime import _NO_NOTEBOOK, _dispatch, _RunState


def _run(call: ToolCall, workspace: Workspace) -> str:
    output, result = _dispatch(call, workspace, 0, _RunState())
    assert result is None  # notebook tools never terminate the loop
    return output


def test_write_notebook_routes_with_its_arguments() -> None:
    seen: dict[str, Any] = {}

    def nb_write(notebook_id, name, cells, deps):  # type: ignore[no-untyped-def]
        seen.update(notebook_id=notebook_id, name=name, cells=cells, deps=deps)
        return "saved id abc"

    workspace = Workspace(datasets={}, nb_write=nb_write)
    call = ToolCall(
        id="c1",
        name="write_notebook",
        arguments={
            "name": "Demo",
            "cells": [{"cell_type": "code", "source": "x = 1"}],
            "deps": ["pandas"],
        },
    )
    assert _run(call, workspace) == "saved id abc"
    assert seen["notebook_id"] is None  # no id -> create
    assert seen["name"] == "Demo"
    assert seen["cells"] == [{"cell_type": "code", "source": "x = 1"}]
    assert seen["deps"] == ["pandas"]


def test_run_notebook_routes_cell_ids() -> None:
    seen: dict[str, Any] = {}

    def nb_run(notebook_id, cell_ids):  # type: ignore[no-untyped-def]
        seen.update(notebook_id=notebook_id, cell_ids=cell_ids)
        return "ran"

    workspace = Workspace(datasets={}, nb_run=nb_run)
    call = ToolCall(id="c2", name="run_notebook", arguments={"notebook_id": "nb1"})
    assert _run(call, workspace) == "ran"
    assert seen == {"notebook_id": "nb1", "cell_ids": None}  # omitted -> run all


def test_schedule_notebook_routes_arguments() -> None:
    seen: dict[str, Any] = {}

    def nb_schedule(notebook_id, mode, interval_hours, dataset):  # type: ignore[no-untyped-def]
        seen.update(
            notebook_id=notebook_id,
            mode=mode,
            interval_hours=interval_hours,
            dataset=dataset,
        )
        return "scheduled"

    workspace = Workspace(datasets={}, nb_schedule=nb_schedule)
    call = ToolCall(
        id="c3",
        name="schedule_notebook",
        arguments={"notebook_id": "nb1", "mode": "interval", "interval_hours": 6},
    )
    assert _run(call, workspace) == "scheduled"
    assert seen["mode"] == "interval"
    assert seen["interval_hours"] == 6.0
    assert seen["dataset"] is None


def test_notebook_tool_declines_without_capability() -> None:
    workspace = Workspace(datasets={})  # a bare workspace has no notebook capabilities
    call = ToolCall(id="c4", name="list_notebooks", arguments={})
    assert _run(call, workspace) == _NO_NOTEBOOK


def test_write_notebook_without_cells_gives_actionable_error() -> None:
    def nb_write(*_a: Any, **_k: Any) -> str:
        raise AssertionError("should not be called without cells")

    workspace = Workspace(datasets={}, nb_write=nb_write)
    call = ToolCall(id="c5", name="write_notebook", arguments={"name": "Demo"})
    assert "cells" in _run(call, workspace)
