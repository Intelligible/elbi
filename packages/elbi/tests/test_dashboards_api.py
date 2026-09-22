"""Integration tests for the dashboard API over a real store and project runner.

These drive the HTTP surface with a TestClient (the boundary the browser uses), so they
cover the wiring end to end: creating and saving a spec (with validation), the publish
gate that refuses an uncertified binding, resolving a page's widgets to rows through the
project runner, dynamic filter options, and subscriptions. The LLM client is a stub,
since dashboards never call it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.dashboards import DashboardService
from elbi.db import open_store
from elbi_core import (
    Artifact,
    Context,
    Registry,
    Runner,
    derivation,
    param,
    serve,
)
from elbi_core.registry import use_registry

DATASETS: dict[str, list[dict[str, Any]]] = {}


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(
            params={"region": param.string(required=False, default="all")},
            serve=serve.table(),
        )
        def revenue(ctx: Context) -> Artifact:
            region = ctx.param("region")
            rows = [
                {"region": "west", "revenue": 10},
                {"region": "east", "revenue": 20},
            ]
            if region != "all":
                rows = [r for r in rows if r["region"] == region]
            return Artifact.table(rows)

        @derivation(serve=serve.table())
        def regions(ctx: Context) -> Artifact:
            return Artifact.table([{"region": "west"}, {"region": "east"}])

    # A proposed (uncertified) derivation, to exercise the publish gate.
    registry.register(replace(registry.get("regions"), name="draft", status="proposed"))
    return registry


def _dashboard(bind: str = "revenue") -> dict[str, Any]:
    return {
        "specVersion": "1.0",
        "kind": "Dashboard",
        "name": "sales",
        "title": "Sales",
        "variables": [
            {
                "name": "region",
                "type": "string",
                "default": "all",
                "options": {"derivation": "regions", "column": "region"},
            }
        ],
        "pages": [
            {
                "name": "main",
                "widgets": [
                    {
                        "id": "revenue_table",
                        "type": "table",
                        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                        "bind": {
                            "derivation": bind,
                            "params": {"region": "$region"},
                        },
                    }
                ],
            }
        ],
    }


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()

    def certified_catalog() -> list[dict[str, Any]]:
        return [
            {
                "name": d.name,
                "title": d.description or d.name,
                "params": d.to_manifest().get("params", {}),
                "served": d.is_served,
            }
            for d in registry
            if d.is_certified
        ]

    service = DashboardService(
        store=store,
        make_runner=lambda: Runner(registry),
        certified_catalog=certified_catalog,
        resolve_metric=lambda name, group_by, grain, filters: [
            {"region": "west", name: 40},
            {"region": "east", name: 20},
        ],
        metric_exists=lambda name: name == "region_revenue",
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        dashboard_service=service,
    )
    with TestClient(app) as http:
        yield http


def _create(http: TestClient, bind: str = "revenue") -> str:
    response = http.post("/api/dashboards", json=_dashboard(bind))
    assert response.status_code == 200, response.text
    dashboard_id: str = response.json()["id"]
    return dashboard_id


def test_create_get_and_list(client: TestClient) -> None:
    dashboard_id = _create(client)
    got = client.get(f"/api/dashboards/{dashboard_id}").json()
    assert got["name"] == "sales"
    assert got["status"] == "draft"
    assert got["spec"]["pages"][0]["widgets"][0]["id"] == "revenue_table"

    listing = client.get("/api/dashboards").json()
    assert [d["id"] for d in listing] == [dashboard_id]


def test_invalid_manifest_is_rejected(client: TestClient) -> None:
    bad = _dashboard()
    bad["pages"] = []  # a dashboard must have at least one page
    assert client.post("/api/dashboards", json=bad).status_code == 400


def test_save_bumps_version(client: TestClient) -> None:
    dashboard_id = _create(client)
    spec = _dashboard()
    spec["title"] = "Renamed"
    saved = client.put(f"/api/dashboards/{dashboard_id}", json={"spec": spec}).json()
    assert saved["title"] == "Renamed"
    assert saved["version"] == 2
    versions = client.get(f"/api/dashboards/{dashboard_id}/versions").json()
    assert len(versions) == 2


def test_publish_gate_accepts_certified_bindings(client: TestClient) -> None:
    dashboard_id = _create(client, bind="revenue")
    response = client.post(f"/api/dashboards/{dashboard_id}/publish")
    assert response.status_code == 200, response.text
    assert client.get(f"/api/dashboards/{dashboard_id}").json()["status"] == "published"


def test_human_publish_allows_uncertified_binding(client: TestClient) -> None:
    # Certification is an agent guardrail, not a human one: a human publishing over the
    # HTTP path is trusted and may bind an as-yet-uncertified derivation.
    dashboard_id = _create(client, bind="draft")  # a proposed derivation
    response = client.post(f"/api/dashboards/{dashboard_id}/publish")
    assert response.status_code == 200, response.text
    assert client.get(f"/api/dashboards/{dashboard_id}").json()["status"] == "published"


def test_publish_gate_rejects_unknown_binding(client: TestClient) -> None:
    dashboard_id = _create(client, bind="revenue")
    spec = _dashboard(bind="ghost")
    client.put(f"/api/dashboards/{dashboard_id}", json={"spec": spec})
    response = client.post(f"/api/dashboards/{dashboard_id}/publish")
    assert response.status_code == 409
    assert "unknown" in response.json()["detail"]


def test_resolve_page_runs_the_bound_derivation(client: TestClient) -> None:
    dashboard_id = _create(client)
    response = client.post(
        f"/api/dashboards/{dashboard_id}/pages/main/data",
        json={"variables": {"region": "west"}},
    )
    assert response.status_code == 200, response.text
    widgets = response.json()["widgets"]
    assert len(widgets) == 1
    assert widgets[0]["widgetId"] == "revenue_table"
    assert widgets[0]["value"] == [{"region": "west", "revenue": 10}]
    assert widgets[0]["error"] is None


def test_resolve_page_uses_defaults_when_unset(client: TestClient) -> None:
    dashboard_id = _create(client)
    response = client.post(
        f"/api/dashboards/{dashboard_id}/pages/main/data", json={"variables": {}}
    )
    rows = response.json()["widgets"][0]["value"]
    assert {r["region"] for r in rows} == {"west", "east"}


def test_dynamic_variable_options(client: TestClient) -> None:
    dashboard_id = _create(client)
    response = client.get(f"/api/dashboards/{dashboard_id}/variables/region/options")
    assert response.status_code == 200
    values = [o["value"] for o in response.json()["options"]]
    assert values == ["west", "east"]


def test_catalog_lists_certified_derivations(client: TestClient) -> None:
    names = {entry["name"] for entry in client.get("/api/dashboards/catalog").json()}
    assert "revenue" in names and "regions" in names
    assert "draft" not in names  # proposed derivations are not bindable


def test_subscriptions_lifecycle(client: TestClient) -> None:
    dashboard_id = _create(client)
    created = client.post(
        f"/api/dashboards/{dashboard_id}/subscriptions",
        json={"page": "main", "cron": "0 9 * * *", "recipients": ["a@b.co"]},
    )
    assert created.status_code == 200
    subscription_id = created.json()["id"]
    listing = client.get(f"/api/dashboards/{dashboard_id}/subscriptions").json()
    assert [s["id"] for s in listing] == [subscription_id]
    deleted = client.delete(
        f"/api/dashboards/{dashboard_id}/subscriptions/{subscription_id}"
    )
    assert deleted.status_code == 200
    assert client.get(f"/api/dashboards/{dashboard_id}/subscriptions").json() == []


def test_delete_dashboard(client: TestClient) -> None:
    dashboard_id = _create(client)
    assert client.delete(f"/api/dashboards/{dashboard_id}").status_code == 200
    assert client.get(f"/api/dashboards/{dashboard_id}").status_code == 404


def _metric_dashboard(metric: str = "region_revenue") -> dict[str, Any]:
    return {
        "specVersion": "1.0",
        "kind": "Dashboard",
        "name": "metric_board",
        "title": "Metric board",
        "pages": [
            {
                "name": "main",
                "widgets": [
                    {
                        "id": "m",
                        "type": "table",
                        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                        "bind": {"metric": metric, "groupBy": ["region"]},
                    }
                ],
            }
        ],
    }


def test_metric_bound_dashboard_publishes_and_resolves(client: TestClient) -> None:
    created = client.post("/api/dashboards", json=_metric_dashboard()).json()
    dashboard_id = created["id"]
    # The publish gate passes: a metric tile's source is certified by construction.
    published = client.post(f"/api/dashboards/{dashboard_id}/publish")
    assert published.status_code == 200, published.text
    # The tile resolves through the metric resolver to rows.
    resolved = client.post(
        f"/api/dashboards/{dashboard_id}/pages/main/data", json={"variables": {}}
    )
    assert resolved.status_code == 200
    widget = resolved.json()["widgets"][0]
    assert widget["kind"] == "table"
    assert {r["region"] for r in widget["value"]} == {"west", "east"}


def test_publish_refuses_unknown_metric(client: TestClient) -> None:
    created = client.post("/api/dashboards", json=_metric_dashboard("nope")).json()
    published = client.post(f"/api/dashboards/{created['id']}/publish")
    assert published.status_code == 409


def test_the_dashboard_schema_is_served_for_an_editor(client: TestClient) -> None:
    """An editor checks a spec against the same document the server validates with.

    A copy of the rules kept in the client drifts, and then it reports an error the
    save accepts, or accepts one the save refuses.
    """
    schema = client.get("/api/dashboards/schema").json()

    assert schema["required"] == ["specVersion", "kind", "name", "pages"]
    assert "widget" in schema["$defs"]
    assert schema["$defs"]["widget"]["required"] == ["id", "type", "gridPos"]
