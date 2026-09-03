"""Tests for `elbi promote`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.authored import AuthoredStore
from elbi_core import Registry, certify, propose, serve


def _seed(project: Path) -> AuthoredStore:
    store = AuthoredStore(project / ".elbi" / "authored")
    reg = Registry()
    store.save(
        certify(
            propose(
                "agent_metric",
                'def agent_metric(ctx):\n    "Authored."\n    return [{"a": 1}]\n',
                serve=serve.json(),
                registry=reg,
            ),
            registry=reg,
        )
    )
    return store


def test_promote_writes_source_and_clears_quarantine(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    store = _seed(project)
    result = runner.invoke(
        app, ["promote", "agent_metric", "-C", str(project)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    target = project / "derivations" / "agent_metric.py"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "AI-generated" in text and "@derivation" in text
    assert store.read("agent_metric") is None  # moved out of quarantine


def test_promote_unknown_name_fails(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    result = runner.invoke(app, ["promote", "nope", "-C", str(project)])
    assert result.exit_code == 1
    assert "no agent-authored derivation" in result.output


def test_promote_refuses_existing_then_force(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    _seed(project)
    (project / "derivations").mkdir(exist_ok=True)
    (project / "derivations" / "agent_metric.py").write_text("# existing\n")

    blocked = runner.invoke(app, ["promote", "agent_metric", "-C", str(project)])
    assert blocked.exit_code == 1
    assert "already exists" in " ".join(blocked.output.split())  # unwrap rich newlines

    forced = runner.invoke(
        app,
        ["promote", "agent_metric", "-C", str(project), "--force"],
        catch_exceptions=False,
    )
    assert forced.exit_code == 0, forced.output
    assert "AI-generated" in (project / "derivations" / "agent_metric.py").read_text(
        encoding="utf-8"
    )


def test_promote_dry_run_writes_nothing(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    store = _seed(project)
    result = runner.invoke(
        app,
        ["promote", "agent_metric", "-C", str(project), "--dry-run"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "@derivation" in result.output
    assert not (project / "derivations" / "agent_metric.py").exists()
    assert store.read("agent_metric") is not None  # still quarantined
