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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.dashboards import DashboardService
from elbi.db import Derivation, Secret, open_store
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
