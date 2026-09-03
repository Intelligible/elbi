"""Integration tests for the feature-store API over a real store and project runner.

Drive the HTTP surface with a TestClient: register an entity and a feature view,
materialize the online store, read features online and point-in-time offline, and hit
the certification gate that refuses to serve a view bound to an uncertified derivation.
The LLM client is a stub; the feature store never calls it.
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
from elbi.db import open_store
from elbi.features import FeatureStoreService
from elbi_core import Artifact, Context, Registry, Runner, derivation, serve
from elbi_core.registry import use_registry

DATASETS: dict[str, list[dict[str, Any]]] = {}

SOURCE = [
    {"user_id": "a", "event_timestamp": "2024-01-01", "clicks": 1},
    {"user_id": "a", "event_timestamp": "2024-01-10", "clicks": 5},
    {"user_id": "a", "event_timestamp": "2024-01-20", "clicks": 9},
    {"user_id": "b", "event_timestamp": "2024-01-05", "clicks": 2},
]


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def user_stats_src(ctx: Context) -> Artifact:
            return Artifact.table([dict(r) for r in SOURCE])

    registry.register(
        replace(registry.get("user_stats_src"), name="draft_src", status="proposed")
    )
    return registry


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    registry = _registry()
    service = FeatureStoreService(
        store=store,
        make_runner=lambda: Runner(registry),
        is_certified=lambda name: any(
            d.name == name and d.is_certified for d in registry
        ),
    )
    app = create_app(
        load_datasets=lambda: DATASETS,
        client=MagicMock(),
        store=store,
        feature_store_service=service,
    )
    with TestClient(app) as http:
        yield http


def _register(http: TestClient, source: str = "user_stats_src") -> None:
    assert (
        http.post(
            "/api/features/entities", json={"name": "user", "join_key": "user_id"}
        ).status_code
        == 200
    )
    response = http.post(
        "/api/features/views",
        json={
            "name": "user_stats",
            "entities": ["user"],
            "source": source,
            "timestampField": "event_timestamp",
            "features": [{"name": "clicks"}],
        },
    )
    assert response.status_code == 200, response.text


def test_register_materialize_and_online(client: TestClient) -> None:
    _register(client)
    written = client.post("/api/features/materialize", json={}).json()
    assert written["written"] == {"userStats": 2}  # one latest row each for "a", "b"
    online = client.post(
        "/api/features/online",
        json={
            "entity_rows": [{"user_id": "a"}, {"user_id": "b"}, {"user_id": "c"}],
            "features": ["user_stats:clicks"],
        },
    ).json()["rows"]
    assert online[0]["clicks"] == 9  # newest snapshot for "a"
    assert online[1]["clicks"] == 2
    assert online[2]["clicks"] is None


def test_point_in_time_historical(client: TestClient) -> None:
    _register(client)
    rows = client.post(
        "/api/features/historical",
        json={
            "entity_df": [{"user_id": "a", "event_timestamp": "2024-01-15"}],
            "features": ["user_stats:clicks"],
        },
    ).json()["rows"]
    # As-of 2024-01-15 the latest value is 5; the 2024-01-20 value must not leak.
    assert rows[0]["clicks"] == 5


def test_reading_one_view_returns_its_registry_entry(client: TestClient) -> None:
    """The plain GET, as against the `/detail` view the page reads.

    Only `/detail` was exercised, so the route that returns the stored definition (what
    `pull` and the SDK read) had no test, and neither did its 404.
    """
    _register(client)
    view = client.get("/api/features/views/user_stats").json()
    assert view["name"] == "user_stats"
    assert view["entities"] == ["user"]
    assert view["source"] == "user_stats_src"
    assert client.get("/api/features/views/absent").status_code == 404


def test_a_training_set_is_materialized_and_read_back(client: TestClient) -> None:
    """Create through the API rather than the service, which is the untested half.

    The point-in-time join underneath has its own tests; this covers the route that
    names the set, and the listing that then finds it.
    """
    _register(client)
    created = client.post(
        "/api/features/training-sets",
        json={
            "name": "clicks_at_jan_15",
            "features": ["user_stats:clicks"],
            "entity_df": [{"user_id": "a", "event_timestamp": "2024-01-15"}],
        },
    )
    assert created.status_code == 200, created.text
    assert created.json()["name"] == "clicks_at_jan_15"

    listed = {t["name"] for t in client.get("/api/features/training-sets").json()}
    assert "clicks_at_jan_15" in listed
    fetched = client.get("/api/features/training-sets/clicks_at_jan_15").json()
    assert fetched["rowCount"] == 1
    assert fetched["sample"][0]["clicks"] == 5  # the as-of value, not the later 9

    # The name is a handle, not a version: writing it again replaces the rows.
    again = client.post(
        "/api/features/training-sets",
        json={
            "name": "clicks_at_jan_15",
            "features": ["user_stats:clicks"],
            "entity_df": [
                {"user_id": "a", "event_timestamp": "2024-01-15"},
                {"user_id": "b", "event_timestamp": "2024-01-15"},
            ],
        },
    )
    assert again.status_code == 200, again.text
    assert (
        client.get("/api/features/training-sets/clicks_at_jan_15").json()["rowCount"]
        == 2
    )

    unknown = client.post(
        "/api/features/training-sets",
        json={
            "name": "nope",
            "features": ["absent_view:clicks"],
            "entity_df": [{"user_id": "a", "event_timestamp": "2024-01-15"}],
        },
    )
    assert unknown.status_code == 409


def test_catalog_lists_views_with_certification(client: TestClient) -> None:
    _register(client)
    catalog = client.get("/api/features/views").json()
    assert len(catalog) == 1
    assert catalog[0]["name"] == "user_stats"
    assert catalog[0]["certified"] is True
    assert catalog[0]["joinKeys"] == ["user_id"]


def test_feature_view_detail_reports_schema_and_freshness(client: TestClient) -> None:
    client.post("/api/features/entities", json={"name": "user", "join_key": "user_id"})
    client.post(
        "/api/features/views",
        json={
            "name": "user_stats",
            "entities": ["user"],
            "source": "user_stats_src",
            "timestampField": "event_timestamp",
            "features": [
                {
                    "name": "clicks",
                    "dtype": "integer",
                    "description": "clicks in window",
                }
            ],
        },
    )
    # Before materialization there is no online data, so no freshness.
    detail = client.get("/api/features/views/user_stats/detail").json()
    assert detail["nOnlineKeys"] == 0
    assert detail["lastMaterializedAt"] is None
    assert detail["hasContract"] is False
    assert detail["statistics"] == []
    # Per-feature dtype and description are surfaced, not dropped.
    assert detail["features"] == [
        {"name": "clicks", "dtype": "integer", "description": "clicks in window"}
    ]
    # After materialization the freshness reflects the online store.
    client.post("/api/features/materialize", json={})
    detail = client.get("/api/features/views/user_stats/detail").json()
    assert detail["nOnlineKeys"] == 2
    assert detail["lastMaterializedAt"] is not None


def test_catalog_reports_freshness_after_materialize(client: TestClient) -> None:
    _register(client)
    assert client.get("/api/features/views").json()[0]["nOnlineKeys"] == 0
    client.post("/api/features/materialize", json={})
    catalog = client.get("/api/features/views").json()
    assert catalog[0]["nOnlineKeys"] == 2
    assert catalog[0]["lastMaterializedAt"] is not None


def test_human_can_materialize_uncertified_source(client: TestClient) -> None:
    # The human API is trusted, like Databricks: certification is the agent's
    # guardrail, not a requirement imposed on a person.
    _register(client, source="draft_src")  # a proposed derivation
    response = client.post("/api/features/materialize", json={})
    assert response.status_code == 200, response.text
    assert response.json()["written"] == {"userStats": 2}


def test_unknown_entity_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/features/views",
        json={"name": "bad", "entities": ["ghost"], "source": "user_stats_src"},
    )
    assert response.status_code == 400
    assert "unknown entity" in response.json()["detail"]


def test_delete_feature_view(client: TestClient) -> None:
    _register(client)
    assert client.delete("/api/features/views/user_stats").status_code == 200
    assert client.get("/api/features/views").json() == []


def test_defining_an_entity_without_a_name_is_a_400_naming_it(
    client: TestClient,
) -> None:
    """The route parses its own body, so the missing-field check is its own to make."""
    response = client.post("/api/features/entities", json={})
    assert response.status_code == 400
    assert "name" in response.json()["detail"]
