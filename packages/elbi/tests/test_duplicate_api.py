"""Acceptance tests for duplicating notebooks, dashboards, saved queries and metrics.

Driven through the HTTP surface, because most of what a duplicate has to get right is
about what does *not* follow the copy: a schedule, a subscription, a verdict, the
original's history. None of that is observable from the service layer alone.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.dashboards import DashboardService
from elbi.db import NotebookCell, SavedQuery, Store, open_store
from elbi.explore import ExploreService
from elbi.lineage import LineageService
from elbi.metrics import MetricService
from elbi.notebooks import NotebookService
from elbi_core import Registry, Runner
from elbi_core.sandbox import ComputeProfile, ComputeProfiles

BOB = {"X-Test-User": "bob"}
CAROL = {"X-Test-User": "carol"}

_DASHBOARD_SPEC: dict[str, Any] = {
    "specVersion": "1.0",
    "kind": "Dashboard",
    "name": "overview",
    "title": "Overview",
    "pages": [
        {
            "name": "main",
            "widgets": [
                {
                    "id": "t",
                    "type": "table",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "bind": {"derivation": "revenue"},
                }
            ],
            "columns": 24,
        }
    ],
    "variables": [{"name": "region", "type": "string", "default": "west"}],
}

_METRIC: dict[str, Any] = {
    "name": "revenue",
    "type": "simple",
    "label": "Revenue",
    "source": "rev_deriv",
    "measure": {"column": "amt", "agg": "sum"},
}


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'app.db'}")


# -- AC 1: a notebook copy has the same cells and no execution history -------------
@pytest.fixture
def client(store: Store) -> Iterator[TestClient]:
    notebooks = NotebookService(
        store=store,
        load_datasets=lambda: {},
        profiles=ComputeProfiles(
            profiles=(ComputeProfile(name="test", max_runtime=15),), default="test"
        ),
    )
    app = create_app(
        load_datasets=lambda: {},
        client=MagicMock(),
        store=store,
        notebook_service=notebooks,
        dashboard_service=DashboardService(
            store=store,
            make_runner=lambda: MagicMock(spec=Runner),
            certified_catalog=lambda: [{"name": "revenue"}],
        ),
        explore_service=ExploreService(
            store=store,
            dataset_names=list,
            resolve_source=lambda name: [],
            read_schema=lambda name: ([], 0),
        ),
        metric_service=MetricService(
            store=store, is_certified=lambda name: True, load_source=lambda name: []
        ),
        lineage_service=LineageService(
            store=store, registry_provider=Registry, dataset_names=list
        ),
    )
    with TestClient(app) as http:
        yield http


def test_notebook_copy_has_the_same_cells_and_no_outputs(
    client: TestClient, store: Store
) -> None:
    notebook_id = store.create_notebook("Analysis")
    store.replace_cells(
        notebook_id,
        [
            NotebookCell(
                id="c1",
                notebook_id=notebook_id,
                position=0,
                cell_type="markdown",
                source="# Heading",
            ),
            NotebookCell(
                id="c2",
                notebook_id=notebook_id,
                position=1,
                cell_type="code",
                source="x = 1",
                outputs_json='[{"output_type": "stream", "text": "1"}]',
                execution_count=7,
            ),
        ],
    )
    store.update_notebook(notebook_id, lock_json='["pandas==2.0.0"]')

    response = client.post(f"/api/notebooks/{notebook_id}/duplicate")
    assert response.status_code == 200
    copy_id = response.json()["id"]

    cells = store.get_cells(copy_id)
    assert [(c.position, c.cell_type, c.source) for c in cells] == [
        (0, "markdown", "# Heading"),
        (1, "code", "x = 1"),
    ]
    assert [c.outputs_json for c in cells] == ["[]", "[]"]
    assert [c.execution_count for c in cells] == [None, None]
    # Cell ids are globally keyed, so a copy must not reuse the source's.
    assert {c.id for c in cells}.isdisjoint({"c1", "c2"})
    # The environment lock rides along, so the copy resolves the same versions.
    assert store.get_notebook(copy_id).lock_json == '["pandas==2.0.0"]'  # type: ignore[union-attr]


# -- AC 2: a dashboard copy is reproduced and unpublished --------------------------
def test_dashboard_copy_reproduces_the_spec_and_is_unpublished(
    client: TestClient, store: Store
) -> None:
    dashboard_id = store.create_dashboard(
        name="overview",
        title="Overview",
        spec_json=json.dumps(_DASHBOARD_SPEC),
    )
    store.publish_dashboard(
        dashboard_id
    )  # the source is published; the copy must not be

    response = client.post(f"/api/dashboards/{dashboard_id}/duplicate")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "draft"
    assert body["publishedSpec"] is None
    assert body["version"] == 1
    spec = body["spec"]
    assert [w["id"] for w in spec["pages"][0]["widgets"]] == ["t"]
    assert spec["pages"][0]["widgets"][0]["gridPos"] == {
        "x": 0,
        "y": 0,
        "w": 12,
        "h": 8,
    }
    assert [v["name"] for v in spec["variables"]] == ["region"]
    # The row's name and the spec's name are one identifier and must stay in step.
    row = store.get_dashboard(body["id"])
    assert row is not None and json.loads(row.spec_json)["name"] == row.name


def test_dashboard_copy_carries_no_scheduled_delivery(
    client: TestClient, store: Store
) -> None:
    """A copy must not mail people who subscribed to the original."""
    dashboard_id = store.create_dashboard(
        name="overview",
        title="Overview",
        spec_json=json.dumps(_DASHBOARD_SPEC),
    )
    store.create_dashboard_subscription(
        dashboard_id=dashboard_id,
        page="main",
        cron="0 9 * * *",
        recipients=["ops@example.com"],
        variable_state={},
        fmt="png",
        channel="email",
    )

    copy_id = client.post(f"/api/dashboards/{dashboard_id}/duplicate").json()["id"]

    assert store.list_dashboard_subscriptions(dashboard_id) != []
    assert store.list_dashboard_subscriptions(copy_id) == []


# -- AC 4: the copy is its own lineage node with no edge to the original -----------
def test_lineage_shows_a_copy_as_its_own_node_with_no_edge(
    client: TestClient, store: Store
) -> None:
    dashboard_id = store.create_dashboard(
        name="overview",
        title="Overview",
        spec_json=json.dumps(_DASHBOARD_SPEC),
    )
    copy = client.post(f"/api/dashboards/{dashboard_id}/duplicate").json()

    graph = client.get("/api/lineage/graph").json()
    source_node = "dashboard:overview"
    copy_node = f"dashboard:{copy['name']}"
    ids = {n["id"] for n in graph["nodes"]}
    assert {source_node, copy_node} <= ids

    node = next(n for n in graph["nodes"] if n["id"] == copy_node)
    assert node["copiedFrom"] == dashboard_id
    # Provenance is an attribute, never a dependency: impact analysis over one must not
    # reach the other in either direction.
    for edge in graph["edges"]:
        assert {edge["source"], edge["target"]} != {source_node, copy_node}


# -- AC 5: a name collision generates a distinct name rather than erroring ---------
def test_repeated_duplicates_get_distinct_names(
    client: TestClient, store: Store
) -> None:
    notebook_id = store.create_notebook("Analysis")
    dashboard_id = store.create_dashboard(
        name="overview",
        title="Overview",
        spec_json=json.dumps(_DASHBOARD_SPEC),
    )

    nb_names, dash_names = [], []
    for _ in range(3):
        nb = client.post(f"/api/notebooks/{notebook_id}/duplicate")
        dash = client.post(f"/api/dashboards/{dashboard_id}/duplicate")
        assert nb.status_code == 200 and dash.status_code == 200
        nb_names.append(store.get_notebook(nb.json()["id"]).name)  # type: ignore[union-attr]
        dash_names.append(dash.json()["name"])

    assert len(set(nb_names)) == 3
    assert len(set(dash_names)) == 3
    # A dashboard name is a spec identifier, so every generated one must still be valid.
    assert all(re.match(r"^[a-z][a-z0-9_]*$", n) for n in dash_names)


# -- saved queries -----------------------------------------------------------------
def test_a_saved_query_copy_is_a_new_row_not_an_update(
    client: TestClient, store: Store
) -> None:
    """A fresh key: saving over the id would rewrite the original, not copy it."""
    store.save_saved_query(SavedQuery(name="Revenue", sql="select 1"))
    source = store.list_saved_queries()[0]

    body = client.post(f"/api/explore/queries/{source.id}/duplicate").json()

    assert body["id"] != source.id
    assert body["name"] != source.name
    assert body["sql"] == "select 1"
    assert body["copiedFrom"] == source.id
    assert store.get_saved_query(body["id"]) is not None
    # The original survives untouched.
    assert store.get_saved_query(source.id) is not None


# -- metrics: the name is a primary key, so the collision check is a safety gate ---


def test_duplicating_something_that_does_not_exist_is_a_404(client: TestClient) -> None:
    assert client.post("/api/notebooks/nope/duplicate").status_code == 404
    assert client.post("/api/dashboards/nope/duplicate").status_code == 404
    assert client.post("/api/explore/queries/nope/duplicate").status_code == 404
    assert client.post("/api/metrics/nope/duplicate").status_code == 404


# -- where a notebook copy lands -----------------------------------------------------


def test_a_copy_stays_in_a_folder_the_caller_owns(
    client: TestClient, store: Store
) -> None:
    """The other half of the rule: your own folder is where you expect the copy."""
    folder_id = store.create_folder("Mine")
    notebook_id = store.create_notebook("Analysis", folder_id=folder_id)

    copy_id = client.post(f"/api/notebooks/{notebook_id}/duplicate").json()["id"]

    assert store.get_notebook(copy_id).folder_id == folder_id  # type: ignore[union-attr]


def test_a_metric_without_a_label_duplicates(client: TestClient, store: Store) -> None:
    unlabelled = {k: v for k, v in _METRIC.items() if k != "label"}
    client.post("/api/metrics", json=unlabelled)

    response = client.post("/api/metrics/revenue/duplicate")

    assert response.status_code == 200
    assert response.json()["name"] == "revenue_copy"
    assert store.get_metric("revenue_copy") is not None


def test_duplicating_a_metric_whose_source_lost_certification_is_refused(
    store: Store,
) -> None:
    """A copy earns its own verdict rather than inheriting one, so this is a 400."""
    notebooks = NotebookService(
        store=store,
        load_datasets=lambda: {},
        profiles=ComputeProfiles(
            profiles=(ComputeProfile(name="test", max_runtime=15),), default="test"
        ),
    )
    certified = {"value": True}
    app = create_app(
        load_datasets=lambda: {},
        client=MagicMock(),
        store=store,
        notebook_service=notebooks,
        metric_service=MetricService(
            store=store,
            is_certified=lambda name: certified["value"],
            load_source=lambda name: [],
        ),
    )
    with TestClient(app) as http:
        assert http.post("/api/metrics", json=_METRIC).status_code == 200
        certified["value"] = False  # the source lost its verdict after the fact
        response = http.post("/api/metrics/revenue/duplicate")

    assert response.status_code == 400
