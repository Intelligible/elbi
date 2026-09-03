"""Tests for `elbi certify` and `elbi derivations`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.project import load_project

_PROPOSED = '''\
from elbi_core import Context, derivation, serve


@derivation(name="forecast", serve=serve.text(), origin="agent", status="proposed")
def forecast(ctx: Context) -> str:
    """A proposed forecast."""
    return "soon"
'''


def _add_proposed(project: Path) -> None:
    (project / "derivations" / "forecast.py").write_text(_PROPOSED, encoding="utf-8")


def test_derivations_lists_status(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    _add_proposed(project)
    result = runner.invoke(app, ["derivations", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "forecast" in result.output
    assert "PROPOSED" in result.output
    assert "churn_risk" in result.output


def test_certify_promotes_proposed(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    _add_proposed(project)

    # Before: discovered but proposed.
    assert not load_project(project).registry.get("forecast").is_certified

    result = runner.invoke(app, ["certify", "forecast", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "certified forecast" in result.output

    # After: the override is applied on reload.
    assert load_project(project).registry.get("forecast").is_certified


def test_certify_already_certified_is_noop(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")  # churn_risk is human/certified
    result = runner.invoke(app, ["certify", "churn_risk", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "already certified" in result.output


def test_certify_unknown_name_fails(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    result = runner.invoke(app, ["certify", "nope", "-C", str(project)])
    assert result.exit_code == 1
    assert "no derivation named" in result.output


def test_certify_missing_project_fails(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["certify", "x", "-C", str(tmp_path)])
    assert result.exit_code == 1
    assert "elbi.yaml" in result.output


def test_derivations_empty_project(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")  # no derivations dir
    result = runner.invoke(app, ["derivations", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "no derivations discovered" in result.output


def test_derivations_missing_project_fails(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["derivations", "-C", str(tmp_path)])
    assert result.exit_code == 1
    assert "elbi.yaml" in result.output
