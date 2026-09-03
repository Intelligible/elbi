"""Tests for the `elbi-app` commands that run and exit: `diagnostics` and `migrate`.

Neither had a test. `diagnostics` printed a key its report had stopped carrying and
raised ``KeyError`` for everyone who ran it, which a type checker cannot see because
the report is a plain dict. It is also the command a person reaches for once something
else is already broken, so that is the worst place for a crash to sit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from elbi.cli import app

runner = CliRunner()


def test_text_report_prints_every_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_URI", "sqlite:///diagnostics.db")
    result = runner.invoke(app, ["diagnostics"])
    assert result.exit_code == 0, result.output
    for section in ("store:", "warehouse:", "configured:", "environment variables"):
        assert section in result.output


def test_json_report_is_parseable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_URI", "sqlite:///diagnostics.db")
    result = runner.invoke(app, ["diagnostics", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert {"store", "warehouse", "configured", "environment"} <= report.keys()


def test_the_text_report_prints_only_keys_the_report_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``configured`` key the text branch reads has to exist.

    Reading the JSON and the text output in the same test is what ties them together:
    a key dropped from the report but left in the f-string fails here rather than in
    front of a user.
    """
    monkeypatch.setenv("DB_URI", "sqlite:///diagnostics.db")
    report = json.loads(runner.invoke(app, ["diagnostics", "--json"]).output)
    configured = report["configured"]
    assert {"llm_model", "smtp", "proxy", "custom_ca"} <= configured.keys()
    assert runner.invoke(app, ["diagnostics"]).exit_code == 0


def test_passwords_never_reach_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_URI", "postgresql://elbi:hunter2@db.internal:5432/elbi")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")
    output = runner.invoke(app, ["diagnostics", "--json"]).output
    assert "hunter2" not in output
    assert "sk-secret-value" not in output
    assert "db.internal" in output


def test_migrate_builds_the_schema_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`migrate` exists so a deployment can fail on the schema step, not in a pod.

    It is the same call `serve` makes, run alone, so the test is that the wiring reaches
    it and reports success rather than that migrations work.
    """
    db = tmp_path / "store.db"
    monkeypatch.setenv("DB_URI", f"sqlite:///{db}")
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'wh'}")
    result = runner.invoke(app, ["migrate"])
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output
    assert db.exists() and db.stat().st_size > 0
