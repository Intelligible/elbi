"""Tests for ``elbi certificate`` (issue / verify / chain / public-key)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_core import AuditEvent, ChainedJsonlAuditSink, verify_certificate
from elbi_core.tracking import CertifiedRun, JsonlRunLog


def _seed_run(project: Path) -> None:
    run = CertifiedRun(
        name="eff",
        derivation_version="v0123456789abc",
        verdict="sound",
        created_at="2026-07-20T00:00:00+00:00",
        data_hash="d" * 64,
        estimate=2.0,
        estimate_label="slope",
        claim={"x": "x", "y": "y"},
        checks=(("effect", "sound", "holds"),),
        code_version="cv1",
        input_versions={"d": "1"},
        question="does x move y?",
    )
    JsonlRunLog(project / ".elbi" / "runs.jsonl").append(run)


def test_issue_then_verify_round_trip(
    scaffold: Callable[[str], Path], runner: CliRunner, tmp_path: Path
) -> None:
    project = scaffold("minimal")
    _seed_run(project)
    out = tmp_path / "cert.json"

    issued = runner.invoke(
        app, ["certificate", "issue", "eff", "-C", str(project), "-o", str(out)]
    )
    assert issued.exit_code == 0, issued.output
    assert out.exists()

    verified = runner.invoke(app, ["certificate", "verify", str(out)])
    assert verified.exit_code == 0, verified.output
    assert "verified" in verified.output


def test_verify_detects_tampering(
    scaffold: Callable[[str], Path], runner: CliRunner, tmp_path: Path
) -> None:
    project = scaffold("minimal")
    _seed_run(project)
    out = tmp_path / "cert.json"
    runner.invoke(
        app, ["certificate", "issue", "eff", "-C", str(project), "-o", str(out)]
    )

    document = json.loads(out.read_text())
    document["certificate"]["verdict"] = "unsound"
    out.write_text(json.dumps(document), encoding="utf-8")

    result = runner.invoke(app, ["certificate", "verify", str(out)])
    assert result.exit_code == 1
    assert "does not verify" in result.output or "signature" in result.output


def test_issue_selects_the_newest_run_and_honors_version_and_issuer(
    scaffold: Callable[[str], Path], runner: CliRunner, tmp_path: Path
) -> None:
    project = scaffold("minimal")
    log = JsonlRunLog(project / ".elbi" / "runs.jsonl")
    older = CertifiedRun(
        name="eff",
        derivation_version="v1_older_abcdef",
        verdict="sound",
        created_at="2026-01-01T00:00:00+00:00",
        claim={"x": "x", "y": "y"},
    )
    newer = CertifiedRun(
        name="eff",
        derivation_version="v2_newer_abcdef",
        verdict="sound",
        created_at="2026-06-01T00:00:00+00:00",
        claim={"x": "x", "y": "y"},
    )
    log.append(older)  # appended oldest-first; runs() reverses to newest-first
    log.append(newer)

    # No --version: picks the newest appended run.
    out = tmp_path / "newest.json"
    runner.invoke(
        app, ["certificate", "issue", "eff", "-C", str(project), "-o", str(out)]
    )
    newest_doc = verify_certificate(json.loads(out.read_text()))
    assert newest_doc.derivation_version == "v2_newer_abcdef"

    # --version filters to the matching prefix, not the newest.
    out2 = tmp_path / "older.json"
    runner.invoke(
        app,
        [
            "certificate",
            "issue",
            "eff",
            "-C",
            str(project),
            "--version",
            "v1",
            "-o",
            str(out2),
        ],
    )
    older_doc = verify_certificate(json.loads(out2.read_text()))
    assert older_doc.derivation_version == "v1_older_abcdef"

    # --issuer overrides the default (the OS user) in the issued document.
    out3 = tmp_path / "custom_issuer.json"
    runner.invoke(
        app,
        [
            "certificate",
            "issue",
            "eff",
            "-C",
            str(project),
            "--issuer",
            "custom@example.com",
            "-o",
            str(out3),
        ],
    )
    custom_doc = verify_certificate(json.loads(out3.read_text()))
    assert custom_doc.issuer == "custom@example.com"


def test_issue_and_public_key_reject_a_corrupted_key_file(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    _seed_run(project)
    key_dir = project / ".elbi"
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / "certificate.key").write_bytes(b"\x00" * 5)  # not the required 32 bytes

    issued = runner.invoke(app, ["certificate", "issue", "eff", "-C", str(project)])
    assert issued.exit_code == 1
    assert "must be 32 bytes" in issued.output

    printed = runner.invoke(app, ["certificate", "public-key", "-C", str(project)])
    assert printed.exit_code == 1
    assert "must be 32 bytes" in printed.output


def test_verify_rejects_an_unreadable_certificate_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "cert.json"
    out.write_text("{not valid json", encoding="utf-8")

    result = runner.invoke(app, ["certificate", "verify", str(out)])
    assert result.exit_code == 1
    assert "could not read certificate" in result.output


def test_verify_warns_only_when_no_public_key_is_pinned(
    scaffold: Callable[[str], Path], runner: CliRunner, tmp_path: Path
) -> None:
    project = scaffold("minimal")
    _seed_run(project)
    out = tmp_path / "cert.json"
    runner.invoke(
        app, ["certificate", "issue", "eff", "-C", str(project), "-o", str(out)]
    )
    pub = project / ".elbi" / "certificate.key.pub"

    with_key = runner.invoke(
        app, ["certificate", "verify", str(out), "--public-key", str(pub)]
    )
    assert with_key.exit_code == 0, with_key.output
    assert "self-consistency only" not in with_key.output

    without_key = runner.invoke(app, ["certificate", "verify", str(out)])
    assert without_key.exit_code == 0, without_key.output
    assert "self-consistency only" in without_key.output


def test_verify_with_public_key_file(
    scaffold: Callable[[str], Path], runner: CliRunner, tmp_path: Path
) -> None:
    project = scaffold("minimal")
    _seed_run(project)
    out = tmp_path / "cert.json"
    runner.invoke(
        app, ["certificate", "issue", "eff", "-C", str(project), "-o", str(out)]
    )
    pub = project / ".elbi" / "certificate.key.pub"

    ok = runner.invoke(
        app, ["certificate", "verify", str(out), "--public-key", str(pub)]
    )
    assert ok.exit_code == 0, ok.output

    bad = runner.invoke(
        app, ["certificate", "verify", str(out), "--public-key", "AAAAnotthekey"]
    )
    assert bad.exit_code == 1


def test_issue_unknown_derivation_fails(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    result = runner.invoke(app, ["certificate", "issue", "nope", "-C", str(project)])
    assert result.exit_code == 1
    assert "no certified run" in result.output


def test_public_key_prints_stable_key(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    first = runner.invoke(app, ["certificate", "public-key", "-C", str(project)])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, ["certificate", "public-key", "-C", str(project)])
    assert first.output.strip() == second.output.strip()


def test_chain_ok_then_broken(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("minimal")
    audit_path = project / ".elbi" / "audit.jsonl"
    sink = ChainedJsonlAuditSink(audit_path)
    for i in range(3):
        sink.record(AuditEvent(f"d{i}", float(i), "allow", "ok"))

    ok = runner.invoke(app, ["certificate", "chain", "-C", str(project)])
    assert ok.exit_code == 0, ok.output
    assert "intact" in ok.output

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"allow"', '"deny"')
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    broken = runner.invoke(app, ["certificate", "chain", "-C", str(project)])
    assert broken.exit_code == 1
    assert "broken" in broken.output


def test_chain_without_project_fails(runner: CliRunner, tmp_path: Path) -> None:
    # Outside a project, chain reports the load error and exits 1, rather than letting
    # the ConfigError trace out of the command.
    result = runner.invoke(app, ["certificate", "chain", "-C", str(tmp_path)])
    assert result.exit_code == 1
    # The handler prints the load error; a leaked ConfigError would not appear here.
    assert "elbi.yaml" in result.output


def test_public_key_without_project_fails(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["certificate", "public-key", "-C", str(tmp_path)])
    assert result.exit_code == 1
    assert "elbi.yaml" in result.output


def test_chain_explicit_path_needs_no_project(
    runner: CliRunner, tmp_path: Path
) -> None:
    # --path verifies a bare log anywhere, with no project load: the offline
    # audit-verification path an external anchor would use.
    audit_path = tmp_path / "audit.jsonl"
    sink = ChainedJsonlAuditSink(audit_path)
    sink.record(AuditEvent("d", 0.0, "allow", "ok"))

    result = runner.invoke(app, ["certificate", "chain", "--path", str(audit_path)])
    assert result.exit_code == 0, result.output
    assert "intact" in result.output


def test_chain_empty_log_is_intact_without_head(
    runner: CliRunner, tmp_path: Path
) -> None:
    # An absent (or empty) log is a valid empty chain: intact, with no head to anchor.
    result = runner.invoke(
        app, ["certificate", "chain", "--path", str(tmp_path / "none.jsonl")]
    )
    assert result.exit_code == 0, result.output
    assert "intact" in result.output
    assert "head" not in result.output
