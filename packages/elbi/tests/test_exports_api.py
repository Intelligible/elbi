"""Tests for the /api/exports/* data-portability routes (IP-31).

Every acceptance criterion gets one test:

* AC-1 / AC-1b: a certified derivation's export carries source, claim, verdict,
  attestation, and result history; an *uncertified* one still exports, with
  ``certificate: null`` rather than a 404.
* AC-2: a dashboard export has both its definition and its current resolved values; a
  metric export has both its manifest and its version history.
* AC-5: a planted secret never appears in any export document.

Plus the notebook-outputs flag (G7 in the IP-31 plan).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.dashboards import DashboardService
from elbi.db import Derivation, Secret, open_store
from elbi.derivation_html import render as render_derivation_html
from elbi.metrics import MetricService
from elbi.monitoring import MonitorService
from elbi.notebooks import NotebookService
from elbi_cli.commands.export_cmd import _write_record
from elbi_core import (
    Artifact,
    Context,
    Registry,
    Runner,
    derivation,
    load_issuer,
    serve,
)
from elbi_core.registry import use_registry
from elbi_core.sandbox import ComputeProfile, ComputeProfiles
from elbi_core.tracking import CertifiedRun

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
pytest.importorskip("cryptography")


_ATTESTATION = {
    "schema": "elbi.verification/v1",
    "verdict": "sound",
    "claim": {"x": "x", "y": "y"},
    "estimate": 2.0,
    "estimate_label": "+2.00 in y per +1 unit of x",
    "adjusted_for": ["region"],
    "checks": [{"name": "effect", "verdict": "sound", "detail": "holds"}],
    "skipped": [],
    "data_hash": "d" * 64,
}

_SECRET_SENTINEL = "sk-live-DO-NOT-LEAK-9f3a7c21"


def _registry() -> Registry:
    """One certified derivation, ``revenue`` -- real enough for a dashboard to bind."""
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table([{"region": "west", "amount": 10}])

    return registry


def _save_derivation(store: Any, *, name: str, certified: bool) -> None:
    store.save_derivation(
        Derivation(
            name=name,
            question=f"does x move y for {name!r}?",
            source=f"def {name}(ctx): ...",
            verdict="sound" if certified else None,
            data_hash="d" * 64,
            claim_json=json.dumps({"x": "x", "y": "y"}),
            attestation_json=json.dumps(_ATTESTATION) if certified else None,
        )
    )


@pytest.fixture
def parts(tmp_path: Path) -> tuple[Any, Any]:
    """The store and a fully-wired app -- shared by every test in this module."""
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()
    notebook_service = NotebookService(
        store=store,
        load_datasets=lambda: {"d": []},
        profiles=ComputeProfiles(
            profiles=(ComputeProfile(name="test", max_runtime=15),), default="test"
        ),
    )
    dashboard_service = DashboardService(
        store=store,
        make_runner=lambda: Runner(registry),
        certified_catalog=lambda: [
            {"name": "revenue", "title": "revenue", "params": {}, "served": True}
        ],
    )
    metric_service = MetricService(
        store=store,
        is_certified=lambda name: name == "revenue",
        load_source=lambda name: [{"region": "west", "amount": 10}],
    )
    monitor_service = MonitorService(
        store=store,
        read_value=lambda monitor: 1.0,
        source_certified=lambda kind, target: True,
    )
    issuer = load_issuer(tmp_path, issuer="acme-corp")
    app = create_app(
        load_datasets=lambda: {"d": []},
        client=MagicMock(),
        store=store,
        notebook_service=notebook_service,
        dashboard_service=dashboard_service,
        metric_service=metric_service,
        monitor_service=monitor_service,
        certificate_issuer=issuer,
    )
    return store, app


@pytest.fixture
def client(parts: tuple[Any, Any]) -> Iterator[TestClient]:
    _, app = parts
    with TestClient(app) as http:
        yield http


# -- AC-1 / AC-1b: derivation export -------------------------------------------------


def test_certified_derivation_export_has_the_full_record(
    parts: tuple[Any, Any],
) -> None:
    store, app = parts
    _save_derivation(store, name="eff", certified=True)
    with TestClient(app) as http:
        resp = http.get("/api/exports/derivations/eff")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema"] == "elbi.export/v1"
    assert body["source"]
    assert body["claim"] == {"x": "x", "y": "y"}
    assert body["verdict"] == "sound"
    assert body["attestation"]["verdict"] == "sound"
    assert isinstance(body["history"], list)
    assert body["certificate"] is not None
    assert body["certificate"]["certificate"]["verdict"] == "sound"


def test_uncertified_derivation_exports_with_a_null_certificate(
    parts: tuple[Any, Any],
) -> None:
    """G8: an uncertified derivation exports (200), not the download route's 404."""
    store, app = parts
    _save_derivation(store, name="draft", certified=False)
    with TestClient(app) as http:
        resp = http.get("/api/exports/derivations/draft")
    assert resp.status_code == 200, resp.text
    assert resp.json()["certificate"] is None


# -- derivation output as HTML ----------------------------------------------------


def test_derivation_output_exports_as_an_html_page(parts: tuple[Any, Any]) -> None:
    store, app = parts
    store.save_derivation(
        Derivation(
            name="posture",
            question="why it costs what it does",
            source="def posture(ctx): ...",
            verdict="sound",
            serve_json=json.dumps({"format": "markdown", "title": "Posture cost"}),
            narrative="Most of it was a **one-off** scan.",
            rendered="# Posture\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
        )
    )
    with TestClient(app) as http:
        resp = http.get("/api/exports/derivations/posture/html")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/html")
    assert 'filename="posture.html"' in resp.headers["content-disposition"]
    page = resp.text
    assert "<title>Posture cost</title>" in page
    assert "<h1>Posture</h1>" in page
    assert "<td>1</td>" in page, "GFM tables render, as they do in the app"
    assert "<strong>one-off</strong>" in page
    assert "Verified output" in page
    assert "default-src 'none'" in page
    assert "script-src" not in page


def test_derivation_html_export_snapshots_the_whole_page(
    parts: tuple[Any, Any],
) -> None:
    """Every section the derivation page shows travels, not just its output."""
    store, app = parts
    _save_derivation(store, name="eff", certified=True)
    store.append_run(
        CertifiedRun(
            name="eff",
            derivation_version="v1abcdef",
            verdict="sound",
            created_at="2026-10-01T00:00:00+00:00",
            estimate=2.0,
            estimate_label="in y per unit x",
            data_hash="d" * 64,
            claim={"x": "x", "y": "y"},
        )
    )
    with TestClient(app) as http:
        page = http.get("/api/exports/derivations/eff/html").text
    assert "does x move y for &#x27;eff&#x27;?" in page, "the question"
    assert "d" * 64 in page, "the data hash"
    assert "Verification · 1 checks run by the oracle" in page
    assert "<b>effect</b>: holds" in page
    assert "Result history · 1 version" in page
    assert "2 in y per unit x" in page
    assert "first certified version" in page
    assert "&quot;x&quot;: &quot;x&quot;" in page, "the claim"
    assert "def eff(ctx): ..." in page, "the source"


def test_derivation_html_export_without_output_still_exports(
    parts: tuple[Any, Any],
) -> None:
    """A page snapshot needs no output: it shows what the page shows, output or not."""
    store, app = parts
    _save_derivation(store, name="bare", certified=False)
    with TestClient(app) as http:
        resp = http.get("/api/exports/derivations/bare/html")
        assert http.get("/api/exports/derivations/nope/html").status_code == 404
    assert resp.status_code == 200
    assert "def bare(ctx): ..." in resp.text
    assert ">Output<" not in resp.text
    assert "Not checked by the oracle." in resp.text, "as the claim panel says"


def test_derivation_html_export_honours_withholding(parts: tuple[Any, Any]) -> None:
    """A withheld output is left out of the page, as the detail view leaves it out."""
    store, app = parts
    store.save_derivation(
        Derivation(name="secret", source="def secret(ctx): ...", rendered="SECRET ROWS")
    )
    app.state.withhold_rendering = lambda name: name == "secret"
    with TestClient(app) as http:
        page = http.get("/api/exports/derivations/secret/html").text
    assert "SECRET ROWS" not in page
    assert "def secret(ctx): ..." in page


def test_derivation_html_export_sanitizes_raw_html(parts: tuple[Any, Any]) -> None:
    store, app = parts
    meta = '<meta http-equiv="refresh" content="0;url=https://example.com">'
    store.save_derivation(
        Derivation(
            name="notes",
            source="def notes(ctx): ...",
            serve_json=json.dumps({"format": "table"}),
            rendered=(
                "| note | n |\n|---|---:|\n"
                f"| {meta} | 1 |\n| <style>.meta{{x:y}}</style> | 2 |\n"
                "| due <b>soon</b> | 3 |"
            ),
        )
    )
    with TestClient(app) as http:
        page = http.get("/api/exports/derivations/notes/html").text
    assert 'http-equiv="refresh"' not in page, "a synced cell must not navigate"
    assert ".meta{x:y}" not in page, "output must not restyle the verdict line"
    assert "due <b>soon</b>" in page, "safe markup renders, as it does in the app"
    assert '<td style="text-align:right">3</td>' in page, "column alignment survives"


def test_derivation_html_dates_the_export_for_people() -> None:
    central = timezone(timedelta(hours=-5))
    page = render_derivation_html(
        name="d",
        title="T",
        output="x",
        exported_at=datetime(2026, 10, 7, 14, 17, 44, 849144, tzinfo=central),
    )
    want = '<time datetime="2026-10-07T19:17:44+00:00">7 Oct 2026, 19:17 UTC</time>'
    assert want in page


@pytest.mark.parametrize(
    ("verdict", "status"),
    [
        ("unsound", '<p class="meta">Not sound. Exported'),
        ("INCONCLUSIVE", '<p class="meta">Inconclusive. Exported'),
        ("sound", '<p class="meta">Verified. Exported'),
        ("<odd>", '<p class="meta">&lt;odd&gt;. Exported'),
        (None, '<p class="meta">Exported'),
    ],
)
def test_derivation_html_names_the_verdict_as_the_app_badge(
    verdict: str | None, status: str
) -> None:
    page = render_derivation_html(
        name="d",
        title="T",
        output="x",
        verdict=verdict,
        exported_at=datetime(2026, 10, 7, tzinfo=timezone.utc),
    )
    assert status in page


# -- AC-2: dashboard + metric export --------------------------------------------------

_DASHBOARD = {
    "specVersion": "1.0",
    "kind": "Dashboard",
    "name": "sales",
    "title": "Sales",
    "pages": [
        {
            "name": "main",
            "widgets": [
                {
                    "id": "w1",
                    "type": "table",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "bind": {"derivation": "revenue"},
                }
            ],
        }
    ],
}

_METRIC = {
    "name": "revenue_metric",
    "type": "simple",
    "source": "revenue",
    "measure": {"agg": "sum", "column": "amount"},
}


def test_dashboard_export_has_definition_and_current_values(
    client: TestClient,
) -> None:
    created = client.post("/api/dashboards", json=_DASHBOARD)
    assert created.status_code == 200, created.text
    dashboard_id = created.json()["id"]
    resp = client.get(f"/api/exports/dashboards/{dashboard_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema"] == "elbi.export/v1"
    assert body["spec"]["name"] == "sales"
    assert "main" in body["values"]
    assert isinstance(body["versions"], list) and body["versions"]


def test_dashboard_snapshot_is_a_downloadable_page(client: TestClient) -> None:
    created = client.post("/api/dashboards", json=_DASHBOARD)
    dashboard_id = created.json()["id"]
    resp = client.get(f"/api/exports/dashboards/{dashboard_id}/snapshot")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.headers["content-disposition"] == (
        'attachment; filename="sales-snapshot.html"'
    )
    assert '<script id="snapshot-data"' in resp.text


def test_an_unknown_dashboard_has_no_snapshot(client: TestClient) -> None:
    assert client.get("/api/exports/dashboards/nope/snapshot").status_code == 404


def test_metric_export_has_manifest_and_history(client: TestClient) -> None:
    created = client.post("/api/metrics", json=_METRIC)
    assert created.status_code == 200, created.text
    resp = client.get("/api/exports/metrics/revenue_metric")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["measure"] == {"agg": "sum", "column": "amount"}
    assert isinstance(body["versions"], list) and body["versions"]


def test_a_downloaded_record_is_the_file_the_cli_archive_writes(
    tmp_path: Path, client: TestClient
) -> None:
    """One record, two delivery routes, one set of bytes: browser and archive agree."""
    client.post("/api/metrics", json=_METRIC)
    resp = client.get("/api/exports/metrics/revenue_metric")

    assert resp.headers["content-disposition"] == (
        'attachment; filename="revenue_metric-record.json"'
    )
    written = tmp_path / "record.json"
    _write_record(written, resp.json())
    assert resp.text == written.read_text(encoding="utf-8")


def test_export_envelope_timestamps_the_export_not_the_object(
    parts: tuple[Any, Any],
) -> None:
    """``created_at`` is when this document was produced, as docs/exports.md says.

    ``_derivation_summary`` carries a ``created_at`` of its own, so spreading it into
    the envelope ahead of the envelope's own key silently replaced the export's
    timestamp with the derivation's -- identical to what the detail route returns.
    """
    store, app = parts
    _save_derivation(store, name="eff", certified=True)
    with TestClient(app) as http:
        detail = http.get("/api/derivations/eff").json()
        export = http.get("/api/exports/derivations/eff").json()

    assert export["created_at"] != detail["createdAt"]
    stamped = datetime.fromisoformat(export["created_at"])
    assert abs((datetime.now(timezone.utc) - stamped).total_seconds()) < 60


# -- AC-5: no secret or credential value in any export --------------------------------


def test_no_export_leaks_a_planted_secret(parts: tuple[Any, Any]) -> None:
    store, app = parts
    store.save_secret(Secret(name="warehouse_password", value=_SECRET_SENTINEL))
    _save_derivation(store, name="eff", certified=True)
    with TestClient(app) as http:
        client_ = http
        client_.post("/api/dashboards", json=_DASHBOARD)
        client_.post("/api/metrics", json=_METRIC)
        notebook_id = client_.post("/api/notebooks", json={}).json()["id"]
        documents = [
            client_.get("/api/exports/derivations/eff").text,
            client_.get("/api/exports/metrics/revenue_metric").text,
            client_.get(f"/api/notebooks/{notebook_id}/export").text,
        ]
    for document in documents:
        assert _SECRET_SENTINEL not in document


# -- G7: notebook outputs are a deployment decision, not a caller's -------------------


def test_a_notebook_export_strips_its_outputs_by_default(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.delenv("NOTEBOOK_EXPORT_OUTPUTS", raising=False)
    notebook_id = client.post("/api/notebooks", json={}).json()["id"]
    export = client.get(f"/api/notebooks/{notebook_id}/export").json()
    for cell in export["cells"]:
        assert cell.get("outputs", []) == []
