"""Tests for `elbi cache`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.commands import cache_cmd
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


def test_clear_by_derivation_removes_only_its_entries(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    load_project(project).make_runner().run("churn_risk")
    other = runner.invoke(
        app, ["cache", "clear", "-C", str(project), "--derivation", "other"]
    )
    assert "invalidated 0" in other.output
    result = runner.invoke(
        app, ["cache", "clear", "-C", str(project), "--derivation", "churn_risk"]
    )
    assert result.exit_code == 0, result.output
    assert "invalidated 1" in result.output
    status = runner.invoke(app, ["cache", "status", "-C", str(project)])
    assert "cached entries: 0" in status.output


def _remote(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    seen: dict[str, object],
) -> None:
    def fake(url: str, token: str | None, **kwargs: object) -> httpx.Client:
        seen["url"], seen["token"] = url, token
        return httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://app"
        )

    monkeypatch.setattr(cache_cmd, "client_for", fake)


def test_clear_with_a_url_invalidates_on_the_app_and_leaves_the_local_cache(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = scaffold("standard")
    load_project(project).make_runner().run("churn_risk")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"removed": 3})

    _remote(monkeypatch, handler, seen)
    result = runner.invoke(
        app,
        [
            "cache", "clear", "-C", str(project),
            "--derivation", "churn_risk", "--tag", "finance",
            "--url", "http://app.example", "--token", "k",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen == {
        "url": "http://app.example",
        "token": "k",
        "path": "/api/cache/invalidate",
        "body": {"tag": "finance", "derivation": "churn_risk"},
    }
    assert "invalidated 3" in result.output
    status = runner.invoke(app, ["cache", "status", "-C", str(project)])
    assert "cached entries: 1" in status.output


def test_a_remote_clear_without_a_tag_or_derivation_sends_nothing(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = scaffold("standard")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["called"] = True
        return httpx.Response(200, json={"removed": 0})

    _remote(monkeypatch, handler, seen)
    result = runner.invoke(
        app, ["cache", "clear", "-C", str(project), "--url", "http://app.example"]
    )
    assert result.exit_code == 2
    assert "needs --tag or --derivation" in result.output
    assert seen == {}


def test_a_refused_remote_clear_fails_with_the_status(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = scaffold("standard")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    _remote(monkeypatch, handler, {})
    result = runner.invoke(
        app,
        ["cache", "clear", "-C", str(project), "--tag", "x", "--url", "http://a"],
    )
    assert result.exit_code == 1
    assert "401" in result.output


def test_an_unreachable_app_fails_cleanly(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = scaffold("standard")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _remote(monkeypatch, handler, {})
    result = runner.invoke(
        app,
        ["cache", "clear", "-C", str(project), "--tag", "x", "--url", "http://a"],
    )
    assert result.exit_code == 1
    assert "connection refused" in result.output
