"""Tests for `elbi validate`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

import elbi_cli.commands.validate_cmd as validate_cmd
from elbi_cli.app import app
from elbi_core import SpecValidationError


def test_validate_scaffolded_project_passes(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "conform to the spec" in result.output
    assert "churn_risk" in result.output


def test_validate_missing_project(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["validate", "-C", str(tmp_path)])
    assert result.exit_code == 1
    assert "elbi.yaml" in result.output


def test_validate_warns_on_undeclared_dataset(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    # Remove the dataset declaration so churn_risk's input is undeclared.
    (project / "elbi.yaml").write_text(
        "project: acme\nderivations_dir: derivations\ndatasets: []\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", "-C", str(project)])
    assert result.exit_code == 0
    assert "not declared" in result.output


def test_validate_fails_on_broken_derivation(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    (project / "derivations" / "broken.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["validate", "-C", str(project)])
    assert result.exit_code == 1


def test_validate_reports_spec_failure_per_derivation(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A derivation that imported cleanly but whose manifest fails re-validation
    # (e.g. a future spec tightening) must be reported and exit non-zero.
    project = scaffold("standard")

    def boom(_manifest: dict) -> None:
        raise SpecValidationError(["forced failure"])

    monkeypatch.setattr(validate_cmd, "validate_manifest", boom)
    result = runner.invoke(app, ["validate", "-C", str(project)])
    assert result.exit_code == 1
    assert "failed validation" in result.output
