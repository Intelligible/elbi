"""Integration tests for the lineage + catalog API over a real store and project.

Drive the HTTP surface: build the cross-artifact graph (a dataset feeding a derivation
that a dashboard and a feature view depend on), then check impact analysis names the
downstream artifacts and the catalog is searchable across types.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.dashboards import DashboardService
from elbi.db import open_store
from elbi.features import FeatureStoreService
from elbi.lineage import LineageService
from elbi_core import (
    Artifact,
    Context,
    Dataset,
    Registry,
    Runner,
    derivation,
    serve,
)
from elbi_core.registry import use_registry

DATASETS: dict[str, list[dict[str, Any]]] = {}


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table([{"region": "west", "revenue": 10}])

    return registry


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()

    def make_runner() -> Runner:
        return Runner(registry)

    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        dashboard_service=DashboardService(
            store=store,
            make_runner=make_runner,
            certified_catalog=lambda: [{"name": "revenue"}],
        ),
        feature_store_service=FeatureStoreService(
            store=store,
            make_runner=make_runner,
            is_certified=lambda name: True,
        ),
        lineage_service=LineageService(
            store=store,
            registry_provider=lambda: registry,
            dataset_names=lambda: ["sales"],
        ),
    )
    with TestClient(app) as http:
        # A dashboard and a feature view that both depend on `revenue`.
        http.post(
            "/api/dashboards",
            json={
                "specVersion": "1.0",
                "kind": "Dashboard",
                "name": "overview",
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
                    }
                ],
            },
        )
        http.post(
            "/api/features/entities", json={"name": "region", "join_key": "region"}
        )
        http.post(
            "/api/features/views",
            json={
                "name": "region_revenue",
                "entities": ["region"],
                "source": "revenue",
                "features": [{"name": "revenue"}],
            },
        )
        yield http


def test_graph_spans_dataset_derivation_dashboard_and_feature_view(
    client: TestClient,
) -> None:
    graph = client.get("/api/lineage/graph").json()
    ids = {n["id"] for n in graph["nodes"]}
    assert {
        "dataset:sales",
        "derivation:revenue",
        "dashboard:overview",
        "feature_view:region_revenue",
        "entity:region",
    } <= ids
    edges = {(e["source"], e["target"]) for e in graph["edges"]}
    assert ("dataset:sales", "derivation:revenue") in edges
    assert ("derivation:revenue", "dashboard:overview") in edges
    assert ("derivation:revenue", "feature_view:region_revenue") in edges


def test_impact_names_downstream_artifacts(client: TestClient) -> None:
    impact = client.get("/api/lineage/impact", params={"node": "dataset:sales"}).json()
    # Changing the source dataset affects the derivation, dashboard, and feature view.
    assert impact["affected"]["derivation"] == ["revenue"]
    assert impact["affected"]["dashboard"] == ["overview"]
    assert impact["affected"]["featureView"] == ["region_revenue"]
    assert impact["count"] == 3  # revenue, overview, region_revenue


def test_impact_on_unknown_node_is_404(client: TestClient) -> None:
    assert (
        client.get("/api/lineage/impact", params={"node": "dataset:ghost"}).status_code
        == 404
    )


def test_provenance_names_upstream_artifacts(client: TestClient) -> None:
    """The mirror of impact, and what the share dialog asks before handing one out."""
    feeds = client.get(
        "/api/lineage/provenance", params={"node": "dashboard:overview"}
    ).json()
    assert feeds["feedsFrom"]["derivation"] == ["revenue"]
    assert feeds["feedsFrom"]["dataset"] == ["sales"]
    assert feeds["count"] == 2


def test_provenance_on_unknown_node_is_404(client: TestClient) -> None:
    assert (
        client.get(
            "/api/lineage/provenance", params={"node": "dashboard:ghost"}
        ).status_code
        == 404
    )


def test_catalog_is_searchable_across_types(client: TestClient) -> None:
    everything = client.get("/api/catalog").json()
    types = {n["type"] for n in everything}
    assert {"dataset", "derivation", "dashboard", "feature_view"} <= types

    filtered = client.get("/api/catalog", params={"q": "revenue"}).json()
    names = {n["name"] for n in filtered}
    assert "revenue" in names and "overview" not in names


def test_subgraph_focuses_on_a_node(client: TestClient) -> None:
    sub = client.get(
        "/api/lineage/subgraph", params={"node": "derivation:revenue"}
    ).json()
    assert sub["focus"] == "derivation:revenue"
    ids = {n["id"] for n in sub["nodes"]}
    assert "dataset:sales" in ids and "dashboard:overview" in ids


def test_impact_on_feature_view_reaches_training_sets_and_models(
    tmp_path: Path,
) -> None:
    # A feature view feeds a training set (a frozen point-in-time join), which a model
    # trains on. Impact analysis on the view must reach both, so "which models consume
    # this feature view" is answerable: the edge chain that was previously broken.
    store = open_store(f"sqlite:{tmp_path / 'lin.db'}")
    registry = _registry()
    store.upsert_feature_entity(name="region", join_key="region")
    store.upsert_feature_view(
        name="region_revenue",
        entities=["region"],
        source="revenue",
        timestamp_field=None,
        ttl_seconds=None,
        features=[{"name": "revenue"}],
    )
    store.upsert_training_set(
        name="churn_2024",
        features=["region_revenue:revenue"],
        label="churn",
        rows=[{"revenue": 10, "churn": 1}],
    )
    models = [
        # Trained on the training set: the normal feature-store path.
        {
            "name": "churn_model",
            "verdict": "sound",
            "source": None,
            "source_kind": "training_set",
            "dataset": "churn_2024",
        },
        # Trained directly on the derivation: the path that already worked.
        {
            "name": "revenue_model",
            "verdict": "sound",
            "source": "revenue",
            "source_kind": "derivation",
            "dataset": "revenue",
        },
    ]
    service = LineageService(
        store=store,
        registry_provider=lambda: registry,
        dataset_names=lambda: ["sales"],
        models_provider=lambda: models,
    )

    edges = {(e["source"], e["target"]): e["kind"] for e in service.graph()["edges"]}
    assert edges[("feature_view:region_revenue", "training_set:churn_2024")] == "joins"
    assert edges[("training_set:churn_2024", "model:churn_model")] == "trains"
    assert edges[("derivation:revenue", "model:revenue_model")] == "trains"

    impact = service.impact("feature_view:region_revenue")
    assert impact["affected"]["training_set"] == ["churn_2024"]
    assert impact["affected"]["model"] == ["churn_model"]


def test_metrics_and_monitors_are_nodes_with_inherited_verdict(tmp_path: Path) -> None:
    # A metric aggregating a certified derivation, and monitors watching that metric and
    # the derivation directly, must appear as first-class nodes downstream of it, each
    # inheriting the derivation's oracle verdict, so impact analysis reaches them.
    from elbi.db import Derivation as DerivationRow
    from elbi.db import MetricMonitor

    store = open_store(f"sqlite:{tmp_path / 'lin.db'}")
    registry = _registry()  # a certified `revenue` derivation over `sales`
    store.save_derivation(
        DerivationRow(name="revenue", verdict="sound", origin="repo", source="...")
    )
    store.upsert_metric(name="revenue_by_region", manifest_json="{}", source="revenue")
    store.save_metric_monitor(
        MetricMonitor(
            name="rev_watch", target_kind="metric", target="revenue_by_region"
        )
    )
    store.save_metric_monitor(
        MetricMonitor(name="deriv_watch", target_kind="derivation", target="revenue")
    )
    service = LineageService(
        store=store, registry_provider=lambda: registry, dataset_names=lambda: ["sales"]
    )

    graph = service.graph()
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = {(e["source"], e["target"]): e["kind"] for e in graph["edges"]}

    assert {"metric:revenue_by_region", "monitor:rev_watch", "monitor:deriv_watch"} <= (
        nodes.keys()
    )
    assert edges[("derivation:revenue", "metric:revenue_by_region")] == "aggregates"
    assert edges[("metric:revenue_by_region", "monitor:rev_watch")] == "watches"
    assert edges[("derivation:revenue", "monitor:deriv_watch")] == "watches"
    # Each downstream node inherits the certified derivation's verdict.
    assert nodes["metric:revenue_by_region"]["verdict"] == "sound"
    assert nodes["monitor:rev_watch"]["verdict"] == "sound"  # via the metric's source
    assert nodes["monitor:deriv_watch"]["verdict"] == "sound"

    # Impact analysis on the derivation reaches the metric and both monitors.
    impact = service.impact("derivation:revenue")
    assert impact["affected"]["metric"] == ["revenue_by_region"]
    assert sorted(impact["affected"]["monitor"]) == ["deriv_watch", "rev_watch"]


def _counting_service(tmp_path: Path) -> tuple[LineageService, Callable[[], int]]:
    """A service over a many-node project, and a count of the graphs it has built.

    ``_graph`` calls ``registry_provider`` once per build, so it counts builds.
    """
    store = open_store(f"sqlite:{tmp_path / 'count.db'}")
    registry = _registry()
    for i in range(8):
        store.upsert_metric(name=f"metric_{i}", manifest_json="{}", source="revenue")
    builds = 0

    def counting_provider() -> Registry:
        nonlocal builds
        builds += 1
        return registry

    service = LineageService(
        store=store,
        registry_provider=counting_provider,
        dataset_names=lambda: ["sales"],
    )
    return service, lambda: builds


def test_catalog_builds_the_graph_once(tmp_path: Path) -> None:
    """Listing the catalog must not rebuild the graph per row.

    The direct-upstream count was read from a graph built inside the loop, so listing N
    artifacts built the whole graph N + 1 times. Quadratic, on the request that lists
    everything, and invisible to every other test because the answer stayed correct.
    """
    service, builds = _counting_service(tmp_path)

    records = service.catalog()

    assert len(records) > 3, "the fixture must be big enough for N + 1 to differ from 1"
    assert builds() == 1


def test_a_snapshot_answers_many_questions_from_one_graph(tmp_path: Path) -> None:
    """Each service method builds its own graph; a snapshot builds one for all of them.

    Tracing a node -- resolve it, ask what feeds it, ask what it feeds -- is three
    builds through the service and one through a view.
    """
    service, builds = _counting_service(tmp_path)

    view = service.snapshot()
    node = view.resolve("revenue")
    assert node == "derivation:revenue"
    assert view.provenance(node)["feeds_from"]["dataset"] == ["sales"]
    assert view.impact(node)["count"] == 8  # the eight metrics
    assert view.catalog()

    assert builds() == 1


def test_a_snapshot_is_a_snapshot(tmp_path: Path) -> None:
    """A view answers from the graph it was built with, not from the store.

    A view's answers only agree with each other if it is fixed at the moment it was
    taken, so a write landing mid-operation cannot change the second answer.
    """
    service, builds = _counting_service(tmp_path)
    store = open_store(f"sqlite:{tmp_path / 'count.db'}")

    view = service.snapshot()
    catalog_before = view.catalog()
    impact_before = view.impact("derivation:revenue")

    store.upsert_metric(name="metric_99", manifest_json="{}", source="revenue")

    assert view.catalog() == catalog_before
    assert view.impact("derivation:revenue") == impact_before
    assert builds() == 1

    # ...and a view taken after the write does see it, so the fixture really did write.
    assert service.snapshot().impact("derivation:revenue")["count"] == 9
