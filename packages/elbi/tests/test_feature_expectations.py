"""Data-contract expectations on feature views, over the HTTP surface.

A feature view can carry a data contract, the platform's existing quality bar, and its
values are checked against it with the same three-valued verdict the oracle speaks. The
source derivation reads a mutable list so a test can make good data pass and then break
it, proving the check actually distinguishes conforming values from violations.
"""

from __future__ import annotations

from collections.abc import Iterator
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

_ROWS: list[dict[str, Any]] = []

CONTRACT = {
    "specVersion": "1.0",
    "kind": "DataContract",
    "fields": [
        {
            "name": "clicks",
            "type": "integer",
            "constraints": {"required": True, "minimum": 0, "maximum": 100},
        }
    ],
}


def _seed(clicks: list[int]) -> None:
    _ROWS.clear()
    _ROWS.extend(
        {"user_id": f"u{i}", "event_timestamp": "2024-01-01", "clicks": value}
        for i, value in enumerate(clicks)
    )


def _registry() -> Registry:
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def user_stats_src(ctx: Context) -> Artifact:
            return Artifact.table([dict(r) for r in _ROWS])

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


def _register(http: TestClient) -> None:
    http.post("/api/features/entities", json={"name": "user", "join_key": "user_id"})
    assert (
        http.post(
            "/api/features/views",
            json={
                "name": "user_stats",
                "entities": ["user"],
                "source": "user_stats_src",
                "timestampField": "event_timestamp",
                "features": [{"name": "clicks"}],
            },
        ).status_code
        == 200
    )


def test_set_get_and_clear_a_contract(client: TestClient) -> None:
    _seed([1, 2, 3])
    _register(client)
    put = client.put(
        "/api/features/views/user_stats/contract", json={"contract": CONTRACT}
    )
    assert put.status_code == 200
    got = client.get("/api/features/views/user_stats/contract").json()["contract"]
    assert got["fields"][0]["name"] == "clicks"
    # An empty manifest clears it.
    client.put("/api/features/views/user_stats/contract", json={"contract": None})
    cleared = client.get("/api/features/views/user_stats/contract").json()
    assert cleared["contract"] is None


def test_expectations_pass_on_conforming_values(client: TestClient) -> None:
    _seed(list(range(1, 41)))  # all within [0, 100]
    _register(client)
    client.put("/api/features/views/user_stats/contract", json={"contract": CONTRACT})
    result = client.post(
        "/api/features/views/user_stats/expectations-check", json={}
    ).json()
    assert result["verdict"] == "sound"
    assert result["nViolated"] == 0
    history = client.get("/api/features/views/user_stats/expectations").json()["checks"]
    assert history[0]["verdict"] == "sound"


def test_expectations_fail_on_violating_values(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    client.put("/api/features/views/user_stats/contract", json={"contract": CONTRACT})
    _seed([99999, 88888, 77777, *range(1, 38)])  # exceed the maximum of 100
    result = client.post(
        "/api/features/views/user_stats/expectations-check", json={}
    ).json()
    assert result["verdict"] == "unsound"
    assert result["nViolated"] >= 1
    assert any(c["verdict"] != "sound" for c in result["clauses"])
    history = client.get("/api/features/views/user_stats/expectations").json()["checks"]
    assert history[0]["verdict"] == "unsound"


def test_suggest_proposes_a_contract(client: TestClient) -> None:
    _seed(list(range(1, 41)))
    _register(client)
    proposed = client.post(
        "/api/features/views/user_stats/contract/suggest", json={}
    ).json()["contract"]
    assert proposed["kind"] == "DataContract"
    assert "clicks" in [f["name"] for f in proposed["fields"]]


def test_verify_without_a_contract_is_a_409(client: TestClient) -> None:
    _seed([1, 2, 3])
    _register(client)
    response = client.post("/api/features/views/user_stats/expectations-check", json={})
    assert response.status_code == 409
    assert "contract" in response.json()["detail"]


def test_invalid_contract_is_rejected(client: TestClient) -> None:
    _seed([1, 2, 3])
    _register(client)
    bad = {"specVersion": "1.0", "kind": "DataContract", "fields": [{"name": "x"}]}
    response = client.put(
        "/api/features/views/user_stats/contract", json={"contract": bad}
    )
    assert response.status_code == 400
