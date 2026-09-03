"""Shared fixtures for the CLI test suite."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from elbi_cli.app import app


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the CLI's console from wrapping, so assertions on its output are stable.

    Rich picks a width from the environment and hard-wraps to it, mid-phrase. A test
    looking for "not empty" fails when the break lands between the two words, which
    makes the assertion depend on how wide the terminal happened to be -- passing when
    run directly and failing under a runner that reports a different width.

    Wide enough that nothing wraps, rather than merely wider: these messages quote
    temporary paths, which are long and vary in length per run, so any width a phrase
    can straddle is a width some run will break on.
    """
    monkeypatch.setenv("COLUMNS", "1000")
    monkeypatch.setenv("TERM", "dumb")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def scaffold(tmp_path: Path, runner: CliRunner) -> Iterator[Callable[[str], Path]]:
    """Scaffold a project of the given template under tmp_path and return its dir."""

    def _scaffold(template: str = "standard") -> Path:
        result = runner.invoke(
            app, ["init", "acme", "--template", template], catch_exceptions=False
        )
        assert result.exit_code == 0, result.output
        return tmp_path / "acme"

    # Run init with tmp_path as the working directory.
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield _scaffold
    finally:
        os.chdir(cwd)
