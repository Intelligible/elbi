"""Tests for ``elbi history``.

The command reads the project's certified-run log; these seed it directly and drive the
command through the CLI, so the wiring, project loading, and rendering are exercised the
way a user runs it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.project import CACHE_DIRNAME, RUN_LOG_FILENAME
from elbi_core.tracking import CertifiedRun, JsonlRunLog


def _seed(project: Path) -> None:
    log = JsonlRunLog(project / CACHE_DIRNAME / RUN_LOG_FILENAME)
    # Oldest first: v1 baseline, then v2 with the data refreshed (a new input version).
    log.append(
        CertifiedRun(
            name="wages",
            derivation_version="v1aaaaaaaaaa",
            verdict="sound",
            created_at="2026-07-03T01:00:00+00:00",
            estimate=0.40,
            estimate_label="pp",
            claim={"x": "edu", "y": "inc"},
            code_version="code1",
            input_versions={"d": "d1"},
        )
    )
    log.append(
        CertifiedRun(
            name="wages",
            derivation_version="v2bbbbbbbbbb",
            verdict="sound",
            created_at="2026-07-03T02:00:00+00:00",
            estimate=0.42,
            estimate_label="pp",
            claim={"x": "edu", "y": "inc"},
            code_version="code1",
            input_versions={"d": "d2"},
        )
    )


def test_history_shows_versions_newest_first_and_what_changed(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    _seed(project)
    result = runner.invoke(app, ["history", "wages", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "2 version" in result.output
    # Both versions present, newest first, and the move attributed to the data refresh.
    assert "v2bbbbbbbbbb" in result.output and "v1aaaaaaaaaa" in result.output
    assert result.output.index("v2bbbbbbbbbb") < result.output.index("v1aaaaaaaaaa")
    assert "changed: data" in result.output


def test_history_empty_is_reported(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    result = runner.invoke(app, ["history", "wages", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "no certified versions" in result.output
