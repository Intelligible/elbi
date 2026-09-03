"""Tests for `elbi init`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from elbi_cli.app import app


def test_init_standard_creates_expected_files(scaffold: Callable[[str], Path]) -> None:
    project = scaffold("standard")
    for expected in (
        "elbi.yaml",
        "elbi.dev.yaml",
        "pyproject.toml",
        "derivations/churn_risk.py",
        "fixtures/sales.csv",
        "tests/test_derivations.py",
        ".github/workflows/test.yml",
        ".gitignore",
        "README.md",
        "AGENTS.md",
        "CLAUDE.md",
    ):
        assert (project / expected).exists(), f"missing {expected}"
    # The dev bindings file must be gitignored.
    assert "elbi.dev.yaml" in (project / ".gitignore").read_text()


def test_init_yaml_surfaces_lexical_search_option(
    scaffold: Callable[[str], Path],
) -> None:
    from elbi_core.config import ProjectConfig

    project = scaffold("standard")
    config_text = (project / "elbi.yaml").read_text(encoding="utf-8")
    # The lexical opt-out is discoverable from the generated config itself.
    assert "search: lexical" in config_text
    # It is commented out, so the default (hybrid) loads.
    assert ProjectConfig.load(project / "elbi.yaml").search == "hybrid"


def test_init_minimal_is_sparse(scaffold: Callable[[str], Path]) -> None:
    project = scaffold("minimal")
    assert (project / "elbi.yaml").exists()
    assert (project / "AGENTS.md").exists()  # the agent guide ships with every template
    assert not (project / "derivations").exists()
    assert not (project / "pyproject.toml").exists()


def test_init_agent_guide_bridges_and_stays_accurate(
    scaffold: Callable[[str], Path],
) -> None:
    from elbi_cli.config_sync import DEFAULT_FOLDERS

    project = scaffold("standard")
    agents = (project / "AGENTS.md").read_text(encoding="utf-8")
    claude = (project / "CLAUDE.md").read_text(encoding="utf-8")

    assert claude.strip() == "@AGENTS.md"  # Claude Code needs the import bridge
    # Accuracy guard: a renamed/added sync surface the guide omits fails here.
    for folder in DEFAULT_FOLDERS:
        assert folder in agents, f"AGENTS.md omits sync surface {folder!r}"
    assert "acme" in agents  # project name substituted
    assert "elbi run" in agents
    assert "never" in agents.lower()


def test_init_rejects_bad_name(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "Bad Name"])
    assert result.exit_code == 1
    assert "invalid project name" in result.output


def test_init_refuses_non_empty_dir(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme" / "keep.txt").write_text("x", encoding="utf-8")
    result = runner.invoke(app, ["init", "acme"])
    assert result.exit_code == 1
    assert "not empty" in result.output


def test_init_force_into_non_empty_dir(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme" / "keep.txt").write_text("x", encoding="utf-8")
    result = runner.invoke(app, ["init", "acme", "--force"])
    assert result.exit_code == 0
    assert (tmp_path / "acme" / "elbi.yaml").exists()


def test_init_offers_nothing_that_would_reach_a_remote_service(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scaffolding is entirely local, and the options say so.

    `init` writes a project directory and nothing else, so an option implying it had
    also registered that project somewhere would be a promise nothing keeps.
    """
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "acme", "--link", "acme-prod"]).exit_code == 2
    assert runner.invoke(app, ["init", "acme"]).exit_code == 0
    assert (tmp_path / "acme" / "elbi.yaml").exists()


def _checkout(root: Path) -> Path:
    """A directory shaped like an elbi checkout."""
    for name in ("elbi-core", "elbi-cli"):
        (root / "packages" / name).mkdir(parents=True)
    return root


def test_init_default_pins_no_local_path(scaffold: Callable[[str], Path]) -> None:
    # Absolute paths in a shared pyproject.toml resolve only on one machine, so
    # the default must depend on the registry alone.
    pyproject = (scaffold("standard") / "pyproject.toml").read_text(encoding="utf-8")
    assert "tool.uv.sources" not in pyproject
    assert "elbi-core" in pyproject


def test_init_editable_redirects_to_the_checkout(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout(tmp_path / "checkout")
    monkeypatch.setenv("ELBI_GIT_REPO_DIR", str(checkout))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "acme", "--editable"])

    assert result.exit_code == 0, result.output
    pyproject = (tmp_path / "acme" / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.uv.sources]" in pyproject
    for name in ("elbi-core", "elbi-cli"):
        expected = f'{name} = {{ path = "{checkout / "packages" / name}"'
        assert expected in pyproject, f"{name} not redirected: {pyproject}"
        assert "editable = true" in pyproject


def test_init_editable_without_the_env_var_fails(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ELBI_GIT_REPO_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "acme", "--editable"])

    assert result.exit_code == 1
    assert "ELBI_GIT_REPO_DIR" in result.output
    # Failing closed means writing nothing, not a project pinned to a bad path.
    assert not (tmp_path / "acme" / "pyproject.toml").exists()


def test_init_editable_rejects_a_directory_that_is_not_a_checkout(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELBI_GIT_REPO_DIR", str(tmp_path / "elsewhere"))
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "acme", "--editable"])

    assert result.exit_code == 1
    assert "not an elbi checkout" in result.output
