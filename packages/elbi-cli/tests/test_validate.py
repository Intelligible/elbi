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
    # Remove the dataset declaration so churn_risk's input is undeclared, and its
    # binding with it, so the project's wiring still agrees.
    (project / "elbi.yaml").write_text(
        "project: acme\nderivations_dir: derivations\ndatasets: []\n",
        encoding="utf-8",
    )
    (project / "elbi.dev.yaml").write_text("data: {}\n", encoding="utf-8")
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


def test_validate_fails_on_declared_dataset_without_binding(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    (project / "elbi.dev.yaml").write_text(
        "data:\n  other: ./fixtures/sales.csv\n", encoding="utf-8"
    )
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 1, result.output
    assert (
        "dataset 'sales' is declared in elbi.yaml but has no binding in elbi.dev.yaml"
        in result.output
    )


def test_validate_fails_on_binding_for_undeclared_dataset(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    (project / "elbi.dev.yaml").write_text(
        "data:\n  other: ./fixtures/sales.csv\n", encoding="utf-8"
    )
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 1, result.output
    assert "elbi.dev.yaml binds 'other', which elbi.yaml does not declare" in (
        result.output
    )


def test_validate_fails_on_binding_to_missing_file(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    (project / "elbi.dev.yaml").write_text(
        "data:\n  sales: ./fixtures/missing.csv\n", encoding="utf-8"
    )
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 1, result.output
    assert "fixtures/missing.csv" in result.output
    assert "does not exist" in result.output


def test_validate_skips_wiring_check_without_dev_bindings(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    (project / "elbi.dev.yaml").unlink()
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "skipped" in result.output


def test_validate_minimal_template_passes(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    assert not (project / "elbi.dev.yaml").exists()
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output


def test_validate_reports_malformed_dev_bindings(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    (project / "elbi.dev.yaml").write_text("data: 3\n", encoding="utf-8")
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 1, result.output
    assert "'data' must be a mapping" in result.output


def test_validate_reports_unreadable_dev_bindings(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    (project / "elbi.dev.yaml").unlink()
    (project / "elbi.dev.yaml").mkdir()
    result = runner.invoke(
        app, ["validate", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 1, result.output
    assert "cannot read" in result.output
