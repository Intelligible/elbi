"""`elbi upload`: stream a local file to the deployment's warehouse.

The command is a thin client of the same server endpoint the browser uses -- the
server-side proxy. It streams the file (httpx loads one chunk at a time), and turns the
server's answers into messages a person can act on: a path on success, "sign in again"
on a lapsed session, and the server's own reason on a refusal.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.commands import config_cmd


def _mock_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setattr(
        config_cmd,
        "client_for",
        lambda url, token, **kwargs: httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://app"
        ),
    )


def test_a_csv_is_streamed_to_the_upload_endpoint(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["multipart"] = request.headers["content-type"].startswith("multipart/")
        seen["carried_bytes"] = b"a,b\n1,2\n" in request.content
        return httpx.Response(
            200,
            json={
                "path": "s3://b/warehouse/_uploads/x/data.csv",
                "filename": "data.csv",
            },
        )

    _mock_client(monkeypatch, handler)
    result = runner.invoke(app, ["upload", str(csv), "--url", "http://app"])
    assert result.exit_code == 0, result.output
    assert seen == {
        "path": "/api/warehouse/uploads",
        "multipart": True,
        "carried_bytes": True,
    }
    assert "s3://b/warehouse/_uploads/x/data.csv" in result.output


def test_an_unsupported_type_is_refused_before_any_request(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    _mock_client(monkeypatch, handler)
    result = runner.invoke(app, ["upload", str(txt), "--url", "http://app"])
    assert result.exit_code == 2
    assert "unsupported file type" in result.output
    assert not called  # rejected client-side, no request made


def test_a_refused_upload_says_so_rather_than_raising(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(403, json={"detail": "forbidden"}),
    )
    result = runner.invoke(app, ["upload", str(csv), "--url", "http://app"])
    assert result.exit_code == 1
    assert "refused" in result.output.lower()


def test_a_bad_file_surfaces_the_servers_reason(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            400, json={"detail": "the uploaded file is empty"}
        ),
    )
    result = runner.invoke(app, ["upload", str(csv), "--url", "http://app"])
    assert result.exit_code == 1
    assert "empty" in result.output


def test_as_table_registers_the_upload_and_loads_it(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging a file and making it a table were two steps, one of them CLI-less.

    Every warehouse CLI stages then loads; this one had no way to load at all, so the
    one thing a person wants from "upload my data" ended at a path they then had to
    take to the API or the UI.
    """
    csv = tmp_path / "loans.csv"
    csv.write_text("a,b\n1,2\n")
    calls: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/warehouse/uploads":
            calls.append((path, {}))
            return httpx.Response(200, json={"path": "s3://b/_uploads/x/loans.csv"})
        if path == "/api/warehouse/sources":
            import json as _json

            calls.append((path, _json.loads(request.content)))
            return httpx.Response(200, json={"id": "src-1", "name": "loans"})
        calls.append((path, {}))
        return httpx.Response(
            200,
            json={"outcomes": [{"table": "csv__loans", "rows": 2, "ok": True}]},
        )

    _mock_client(monkeypatch, handler)
    result = runner.invoke(app, ["upload", str(csv), "--as-table", "loans"])
    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == [
        "/api/warehouse/uploads",
        "/api/warehouse/sources",
        "/api/warehouse/sources/src-1/sync",
    ]
    body = calls[1][1]
    assert body["name"] == "loans" and body["source_type"] == "csv"
    assert body["config"] == {"path": "s3://b/_uploads/x/loans.csv"}
    assert "csv__loans" in result.output and "2" in result.output


def test_a_load_that_fails_is_not_reported_as_success(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "loans.csv"
    csv.write_text("a,b\n1,2\n")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/warehouse/uploads":
            return httpx.Response(200, json={"path": "s3://b/_uploads/x/loans.csv"})
        if request.url.path == "/api/warehouse/sources":
            return httpx.Response(200, json={"id": "src-1", "name": "loans"})
        return httpx.Response(
            200,
            json={
                "outcomes": [
                    {
                        "table": "csv__loans",
                        "rows": 0,
                        "ok": False,
                        "error": "bad utf-8",
                    }
                ]
            },
        )

    _mock_client(monkeypatch, handler)
    result = runner.invoke(app, ["upload", str(csv), "--as-table", "loans"])
    assert result.exit_code == 1
    assert "bad utf-8" in result.output
