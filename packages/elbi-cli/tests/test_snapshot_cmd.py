"""`elbi snapshot`: a dashboard export rendered as one self-contained page."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.commands import snapshot_cmd
from elbi_cli.snapshot import build, render


def _document(**overrides: object) -> dict[str, object]:
    spec = {
        "title": "Revenue",
        "description": "What we bill.",
        "pages": [
            {
                "name": "main",
                "title": "Main",
                "widgets": [
                    {
                        "id": "mrr",
                        "type": "metric",
                        "viz": {"field": "mrr", "agg": "sum"},
                    },
                    {"id": "note", "type": "text", "content": "Static **words**."},
                ],
            }
        ],
    }
    document: dict[str, object] = {
        "schema": "elbi.export/v1",
        "id": "d1",
        "name": "revenue",
        "title": "Revenue",
        "status": "published",
        "version": 3,
        "created_at": "2026-09-23T15:00:00+00:00",
        "updated_at": "2026-09-22T10:00:00+00:00",
        "spec": {**spec, "title": "Draft title"},
        "published_spec": spec,
        "versions": [{"version": 1, "spec": {"secret": "old draft"}}],
        "values": {
            "main": [
                {
                    "widget_id": "mrr",
                    "kind": "table",
                    "derivation": "revenue_by_product",
                    "data_version": "abc123",
                    "error": None,
                    "value": [{"mrr": 300.0}],
                }
            ]
        },
    }
    document.update(overrides)
    return document


def _payload(page_html: str) -> dict[str, object]:
    match = re.search(
        r'<script id="snapshot-data" type="application/json">(.*?)</script>',
        page_html,
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_the_published_spec_and_values_reach_the_page() -> None:
    payload = build(_document())
    widget = payload["pages"][0]["widgets"][0]
    assert payload["title"] == "Revenue"
    assert widget["data"]["value"] == [{"mrr": 300.0}]
    assert widget["data"]["derivation"] == "revenue_by_product"
    assert "data" not in payload["pages"][0]["widgets"][1]


def test_version_history_never_reaches_the_page() -> None:
    page = render(_document())
    assert "old draft" not in page
    assert "Draft title" not in page


def test_a_never_published_dashboard_renders_its_draft() -> None:
    payload = build(_document(status="draft", published_spec=None))
    assert payload["status"] == "draft"
    assert payload["pages"][0]["name"] == "main"


def test_values_cannot_close_the_data_script() -> None:
    document = _document()
    document["values"]["main"][0]["value"] = [
        {"mrr": "</script><script>alert(1)</script>"}
    ]
    page = render(document)
    assert "</script><script>alert(1)" not in page
    assert _payload(page)["pages"][0]["widgets"][0]["data"]["value"][0][
        "mrr"
    ].startswith("</script>")


def test_the_title_is_escaped() -> None:
    page = render(_document(title="<b>Revenue</b>"))
    assert "<title>&lt;b&gt;Revenue&lt;/b&gt;</title>" in page


def _mock_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    monkeypatch.setattr(
        snapshot_cmd,
        "client_for",
        lambda url, token, **kwargs: httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://app"
        ),
    )


def test_snapshot_writes_the_page_by_dashboard_name(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/dashboards":
            return httpx.Response(200, json=[{"id": "d1", "name": "revenue"}])
        assert request.url.path == "/api/exports/dashboards/d1"
        return httpx.Response(200, json=_document())

    _mock_client(monkeypatch, handler)
    out = tmp_path / "page.html"
    result = runner.invoke(
        app, ["snapshot", "revenue", "-o", str(out), "-C", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert _payload(out.read_text())["name"] == "revenue"


def test_an_unknown_dashboard_names_the_ones_that_exist(
    tmp_path: Path, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(200, json=[{"id": "d1", "name": "revenue"}]),
    )
    result = runner.invoke(app, ["snapshot", "nope", "-C", str(tmp_path)])
    assert result.exit_code == 1
    assert "revenue" in result.output
