"""Tests for `elbi cache`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.project import load_project


def test_status_empty_then_populated(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    result = runner.invoke(app, ["cache", "status", "-C", str(project)])
    assert result.exit_code == 0
    assert "cached entries: 0" in result.output

    # Populate the cache by running a derivation through the project's runner.
    load_project(project).make_runner().run("churn_risk")
    result = runner.invoke(app, ["cache", "status", "-C", str(project)])
    assert "cached entries: 1" in result.output


def test_clear(scaffold: Callable[[str], Path], runner: CliRunner) -> None:
    project = scaffold("standard")
    load_project(project).make_runner().run("churn_risk")
    result = runner.invoke(app, ["cache", "clear", "-C", str(project)])
    assert result.exit_code == 0
    assert "cleared" in result.output
    status = runner.invoke(app, ["cache", "status", "-C", str(project)])
    assert "cached entries: 0" in status.output


def test_clear_by_tag(scaffold: Callable[[str], Path], runner: CliRunner) -> None:
    project = scaffold("standard")
    load_project(project).make_runner().run("churn_risk")
    result = runner.invoke(app, ["cache", "clear", "-C", str(project), "--tag", "nope"])
    assert result.exit_code == 0
    assert "invalidated 0" in result.output


def test_gc_runs(scaffold: Callable[[str], Path], runner: CliRunner) -> None:
    project = scaffold("standard")
    load_project(project).make_runner().run("churn_risk")
    result = runner.invoke(app, ["cache", "gc", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "reclaimed" in result.output


def test_gc_with_max_age_evicts(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    load_project(project).make_runner().run("churn_risk")
    result = runner.invoke(
        app, ["cache", "gc", "-C", str(project), "--max-age-days", "0"]
    )
    assert result.exit_code == 0, result.output
    assert "evicted 1" in result.output  # the one entry is older than 0 days
    status = runner.invoke(app, ["cache", "status", "-C", str(project)])
    assert "cached entries: 0" in status.output


def test_gc_missing_project(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["cache", "gc", "-C", str(tmp_path)])
    assert result.exit_code == 1


def test_status_missing_project(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["cache", "status", "-C", str(tmp_path)])
    assert result.exit_code == 1


def test_clear_missing_project(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["cache", "clear", "-C", str(tmp_path)])
    assert result.exit_code == 1
