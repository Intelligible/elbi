"""A byte-level pin on the catalog's output, so a performance fix cannot change it.

Nothing about the records may move:
``/api/catalog`` feeds the catalog page, whose ``CatalogRecord``
(``web/src/lib/lineage.ts``) names every field including ``upstream``.

The fixture holds one artifact of every node type, so each ``_add_*`` branch
contributes. Record order is ``LineageGraph._nodes`` insertion order, deterministic for
a fixed build order, which is why comparing serialized bytes is fair rather than flaky.
One of each type keeps it that way, since ``list_dashboards`` and friends sort on a
timestamp that two rows written in the same test would tie on.

``_GOLDEN`` was generated against the pre-IP-16 implementation and committed before it
was touched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.dashboards import DashboardService
from elbi.db import Derivation as DerivationRow
from elbi.db import MetricMonitor, Store, open_store
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

_DASHBOARD_SPEC = {
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
}

_MODELS: list[dict[str, Any]] = [
    {
        "name": "churn_model",
        "verdict": "sound",
        "source": None,
        "source_kind": "training_set",
        "dataset": "churn_2024",
    }
]

#: The catalog exactly as it read before IP-16 touched how it is built, updated for
#: IP-30's addition of "copied_from" to every node.
_GOLDEN: list[dict[str, Any]] = [
    {
        "id": "dataset:sales",
        "type": "dataset",
        "name": "sales",
        "verdict": None,
        "certified": None,
        "description": None,
        "copied_from": None,
        "upstream": 0,
    },
    {
        "id": "derivation:revenue",
        "type": "derivation",
        "name": "revenue",
        "verdict": "sound",
        "certified": True,
        "description": "Revenue by region.",
        "copied_from": None,
        "upstream": 1,
    },
    {
        "id": "derivation:margin_trend",
        "type": "derivation",
        "name": "margin_trend",
        "verdict": None,
        "certified": True,
        "description": "Retention across cohorts.",
        "copied_from": None,
        "upstream": 1,
    },
    {
        "id": "dashboard:overview",
        "type": "dashboard",
        "name": "overview",
        "verdict": None,
        "certified": None,
        "description": "Overview",
        "copied_from": None,
        "upstream": 1,
    },
    {
        "id": "feature_view:region_revenue",
        "type": "feature_view",
        "name": "region_revenue",
        "verdict": None,
        "certified": None,
        "description": None,
        "copied_from": None,
        "upstream": 2,
    },
    {
        "id": "entity:region",
        "type": "entity",
        "name": "region",
        "verdict": None,
        "certified": None,
        "description": None,
        "copied_from": None,
        "upstream": 0,
    },
    {
        "id": "training_set:churn_2024",
        "type": "training_set",
        "name": "churn_2024",
        "verdict": None,
        "certified": None,
        "description": None,
        "copied_from": None,
        "upstream": 1,
    },
    {
        "id": "model:churn_model",
        "type": "model",
        "name": "churn_model",
        "verdict": "sound",
        "certified": None,
        "description": None,
        "copied_from": None,
        "upstream": 1,
    },
    {
        "id": "metric:revenue_by_region",
        "type": "metric",
        "name": "revenue_by_region",
        "verdict": "sound",
        "certified": None,
        "description": None,
        "copied_from": None,
        "upstream": 1,
    },
    {
        "id": "monitor:rev_watch",
        "type": "monitor",
        "name": "rev_watch",
        "verdict": "sound",
        "certified": None,
        "description": None,
        "copied_from": None,
        "upstream": 1,
    },
]


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            "Revenue by region."
            return Artifact.table([{"region": "west", "revenue": 10}])

        # Holds the fixture's only "retention", and holds it in the description rather
        # than the name or type.
        @derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
        def margin_trend(ctx: Context) -> Artifact:
            "Retention across cohorts."
            return Artifact.table([{"cohort": "q1", "margin": 3}])

    return registry


def _populate(store: Store) -> None:
    """One artifact of every type the graph knows how to add."""
    store.save_derivation(
        DerivationRow(name="revenue", verdict="sound", origin="repo", source="...")
    )
    store.create_dashboard(
        name="overview", title="Overview", spec_json=json.dumps(_DASHBOARD_SPEC)
    )
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
    store.upsert_metric(name="revenue_by_region", manifest_json="{}", source="revenue")
    store.save_metric_monitor(
        MetricMonitor(
            name="rev_watch", target_kind="metric", target="revenue_by_region"
        )
    )


@pytest.fixture
def registry() -> Registry:
    return _registry()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with open_store(f"sqlite:{tmp_path / 'app.db'}") as opened:
        _populate(opened)
        yield opened


@pytest.fixture
def service(store: Store, registry: Registry) -> LineageService:
    return LineageService(
        store=store,
        registry_provider=lambda: registry,
        dataset_names=lambda: ["sales"],
        models_provider=lambda: _MODELS,
    )


def test_catalog_records_are_unchanged(service: LineageService) -> None:
    """Every record, every field, in the order the graph produced them."""
    assert json.dumps(service.catalog(), indent=4) == json.dumps(_GOLDEN, indent=4)


def _golden_subset(*ids: str) -> list[dict[str, Any]]:
    """The golden records with these ids, in golden order.

    Named rather than re-derived: an expectation computed by reimplementing the filter
    shares whatever bug the reimplementation has.
    """
    wanted = set(ids)
    subset = [r for r in _GOLDEN if r["id"] in wanted]
    assert len(subset) == len(wanted), "a named id is missing from the golden"
    return subset


def test_a_filtered_catalog_is_unchanged(service: LineageService) -> None:
    """The substring filter selects the same rows, with the same counts."""
    expected = _golden_subset(
        "derivation:revenue",
        "feature_view:region_revenue",
        "metric:revenue_by_region",
    )
    assert json.dumps(service.catalog("revenue")) == json.dumps(expected)


def test_the_filter_still_searches_descriptions(service: LineageService) -> None:
    """A record reachable only by its description is still reachable.

    Name and type are load-bearing in every other assertion here, which leaves
    description as the one field a regression could drop with the file still green.
    """
    assert service.catalog("retention") == _golden_subset("derivation:margin_trend")


def test_the_api_returns_the_same_bytes(
    store: Store, registry: Registry, service: LineageService
) -> None:
    """And the wire format the catalog page parses is byte-for-byte what it was.

    Through the app rather than the service, so the camelCase layer is covered.
    ``copied_from`` is the first multi-word field the catalog carries, so its wire form
    is ``copiedFrom``. The expectation is spelled out here rather than run through
    ``casing.camelize``: a test that pins the wire format must not build what it
    expects with the code it is pinning.
    """

    def make_runner() -> Runner:
        return Runner(registry)

    app = create_app(
        load_datasets=lambda: {},
        client=MagicMock(),
        store=store,
        dashboard_service=DashboardService(
            store=store,
            make_runner=make_runner,
            certified_catalog=lambda: [{"name": "revenue"}],
        ),
        feature_store_service=FeatureStoreService(
            store=store, make_runner=make_runner, is_certified=lambda name: True
        ),
        lineage_service=service,
    )
    with TestClient(app) as http:
        # ``json.dumps`` defaults match because the camelCase middleware re-serializes
        # this route (casing.py); FastAPI's own JSONResponse writes compact separators.
        # Exempting /api/catalog there would break this on formatting, not content.
        expected = [
            {
                ("copiedFrom" if key == "copied_from" else key): value
                for key, value in record.items()
            }
            for record in _GOLDEN
        ]
        assert http.get("/api/catalog").content == json.dumps(expected).encode()
