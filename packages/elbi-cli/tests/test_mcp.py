"""Tests for `elbi mcp` (server stubbed so it does not block)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

import elbi_cli.commands.mcp_cmd as mcp_cmd
from elbi_cli.app import app
from elbi_cli.authored import AuthoredStore
from elbi_core import Registry, certify, propose, serve


class _FakeServer:
    def __init__(self) -> None:
        self.ran_transport: str | None = None
        # Transport settings reach run(), not build_server(), so the fake records them.
        self.ran_kwargs: dict[str, object] = {}

    def run(self, transport: str, **kwargs: object) -> None:
        self.ran_transport = transport
        self.ran_kwargs = kwargs


def test_mcp_starts_and_reports_url(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = scaffold("standard")
    captured: dict[str, object] = {}

    served = _FakeServer()

    def fake_build_server(
        registry: Registry, make_runner: object, **kwargs: object
    ) -> _FakeServer:
        captured.update(kwargs)
        captured["count"] = len(registry)
        return served

    monkeypatch.setattr(mcp_cmd, "build_server", fake_build_server)

    result = runner.invoke(
        app, ["mcp", "-C", str(project), "--port", "9999"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9999/mcp" in result.output
    assert "discovered 1 derivation" in result.output
    assert captured["count"] == 1
    # --port has to reach the transport, which now takes it at run() rather than build.
    assert served.ran_kwargs["port"] == 9999
    assert served.ran_kwargs["host"] == "127.0.0.1"
    assert served.ran_kwargs["streamable_http_path"] == "/mcp"


def test_mcp_reports_reloaded_authored_derivations(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = scaffold("minimal")
    store = AuthoredStore(project / ".elbi" / "authored")
    built = Registry()
    store.save(
        certify(
            propose(
                "agent_metric",
                'def agent_metric(ctx):\n    "x"\n    return [{"a": 1}]\n',
                serve=serve.json(),
                registry=built,
            ),
            registry=built,
        )
    )
    monkeypatch.setattr(mcp_cmd, "build_server", lambda *a, **k: _FakeServer())

    result = runner.invoke(app, ["mcp", "-C", str(project)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "reloaded 1 agent-authored derivation" in result.output


def test_mcp_activates_analysis_guidance(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = scaffold("standard")
    captured: dict[str, object] = {}

    def fake_build_server(
        registry: Registry, make_runner: object, **kwargs: object
    ) -> _FakeServer:
        captured.update(kwargs)
        return _FakeServer()

    monkeypatch.setattr(mcp_cmd, "build_server", fake_build_server)
    result = runner.invoke(app, ["mcp", "-C", str(project)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "analysis guidance active" in result.output
    assert "careful data scientist" in str(captured["instructions"])


def test_mcp_wires_chained_audit_and_issuer(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi_core import ChainedJsonlAuditSink
    from elbi_core.certificate import CertificateIssuer

    project = scaffold("standard")
    captured: dict[str, object] = {}

    def fake_build_server(
        registry: Registry, make_runner: object, **kwargs: object
    ) -> _FakeServer:
        captured.update(kwargs)
        return _FakeServer()

    monkeypatch.setattr(mcp_cmd, "build_server", fake_build_server)
    result = runner.invoke(app, ["mcp", "-C", str(project)], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    audit = captured["audit"]
    assert isinstance(audit, ChainedJsonlAuditSink)
    assert audit.path == project / ".elbi" / "audit.jsonl"
    assert isinstance(captured["issuer"], CertificateIssuer)


def test_mcp_missing_project(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["mcp", "-C", str(tmp_path)])
    assert result.exit_code == 1
    assert "elbi.yaml" in result.output


def test_mcp_reports_unbound_when_no_dev_yaml(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The minimal template has no elbi.dev.yaml.
    project = scaffold("minimal")
    monkeypatch.setattr(mcp_cmd, "build_server", lambda *a, **k: _FakeServer())

    result = runner.invoke(app, ["mcp", "-C", str(project)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "datasets are unbound" in result.output


def _set_lexical(project: Path) -> None:
    config = project / "elbi.yaml"
    config.write_text(config.read_text() + "\nsearch: lexical\n", encoding="utf-8")


def test_mcp_default_installs_hybrid_retriever(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi_core import HybridRetriever

    project = scaffold("standard")  # default search is hybrid

    captured: dict[str, object] = {}

    def fake_build_server(
        registry: Registry, make_runner: object, **kwargs: object
    ) -> _FakeServer:
        captured["retriever"] = registry.retriever
        return _FakeServer()

    monkeypatch.setattr(mcp_cmd, "build_server", fake_build_server)

    result = runner.invoke(app, ["mcp", "-C", str(project)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "search: hybrid" in result.output
    assert "set search: hybrid" not in result.output  # no hint when already default
    # Hybrid is the default with no extra to install, and the embedding model stays
    # unloaded until the first search, so standing the server up needs no model.
    assert isinstance(captured["retriever"], HybridRetriever)


def test_mcp_lexical_mode_hints_at_hybrid(
    scaffold: Callable[[str], Path],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi_core import Bm25Retriever

    project = scaffold("standard")
    _set_lexical(project)

    captured: dict[str, object] = {}

    def fake_build_server(
        registry: Registry, make_runner: object, **kwargs: object
    ) -> _FakeServer:
        captured["retriever"] = registry.retriever
        return _FakeServer()

    monkeypatch.setattr(mcp_cmd, "build_server", fake_build_server)

    result = runner.invoke(app, ["mcp", "-C", str(project)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "search: lexical (BM25)" in result.output
    assert "set search: hybrid" in result.output  # discoverability hint
    assert isinstance(captured["retriever"], Bm25Retriever)
