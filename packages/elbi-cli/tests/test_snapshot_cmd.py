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
                        "bind": {"metric": "mrr"},
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
                    # A metric tile as DashboardService.resolve returns it: the metric's
                    # name in `derivation`, no data version, and the metric's format.
                    "widget_id": "mrr",
                    "kind": "table",
                    "derivation": "mrr",
                    "data_version": None,
                    "error": None,
                    "value": [{"mrr": 300.0}],
                    "format": {"kind": "currency", "currency": "USD", "precision": 0},
                }
            ]
        },
    }
    document.update(overrides)
    return document


def _data_block(page_html: str) -> str:
    match = re.search(
        r'<script id="snapshot-data" type="application/json">(.*?)</script>',
        page_html,
        re.S,
    )
    assert match is not None
    return match.group(1)


def _not_json(constant: str) -> object:
    raise AssertionError(f"{constant} is not JSON; the page's JSON.parse rejects it")


def _payload(page_html: str) -> dict[str, object]:
    # Strict, as the browser is: Python's json.loads would otherwise accept a bare NaN.
    return json.loads(_data_block(page_html), parse_constant=_not_json)


def test_the_published_spec_and_values_reach_the_page() -> None:
    payload = build(_document())
    widget = payload["pages"][0]["widgets"][0]
    assert payload["title"] == "Revenue"
    assert widget["data"]["value"] == [{"mrr": 300.0}]
    assert widget["data"]["derivation"] == "mrr"
    # A metric tile reads `bind.metric` and shows it in the metric's format.
    assert widget["bind"] == {"metric": "mrr"}
    assert widget["data"]["format"] == {
        "kind": "currency",
        "currency": "USD",
        "precision": 0,
    }
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


def test_the_data_block_holds_nothing_html_or_javascript_reads_specially() -> None:
    raw = "a<b>c&d\u2028e\u2029f"
    document = _document()
    document["values"]["main"][0]["value"] = [{"mrr": raw}]
    page = render(document)
    for char in "<>&\u2028\u2029":
        assert char not in _data_block(page)
    assert _payload(page)["pages"][0]["widgets"][0]["data"]["value"] == [{"mrr": raw}]


def test_a_missing_number_shows_as_empty_instead_of_blanking_the_page() -> None:
    document = _document()
    document["values"]["main"][0]["value"] = [
        {"mrr": float("nan")},
        {"mrr": float("inf")},
        {"mrr": 5.0},
    ]
    payload = _payload(render(document))
    assert payload["pages"][0]["widgets"][0]["data"]["value"] == [
        {"mrr": None},
        {"mrr": None},
        {"mrr": 5.0},
    ]


def test_the_page_is_a_standards_mode_document_in_english() -> None:
    # Without a doctype a browser lays the page out in quirks mode; without lang a
    # screen reader has to guess the language (WCAG 3.1.1).
    assert render(_document()).startswith('<!doctype html>\n<html lang="en">\n')


def test_the_title_is_escaped() -> None:
    page = render(_document(title="<b>Revenue</b>"))
    assert "<title>&lt;b&gt;Revenue&lt;/b&gt;</title>" in page


def test_a_title_naming_a_placeholder_stays_literal() -> None:
    page = render(_document(title="__DATA__ and __TITLE__"))
    assert "<title>__DATA__ and __TITLE__</title>" in page
    assert _payload(page)["title"] == "__DATA__ and __TITLE__"


def test_the_page_loads_nothing() -> None:
    page = render(_document())
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]*)"', page)
    assert policy is not None
    assert "default-src 'none'" in policy.group(1)
    # The SVG namespace is an identifier, never fetched; any other URL would be.
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", page))
    assert urls == {"http://www.w3.org/2000/svg"}
    for loader in ("<link", "@import", "url(", " src="):
        assert loader not in page


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
