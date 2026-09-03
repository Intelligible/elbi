"""Integration tests for the metrics API over a real store and certified source.

Drive the HTTP surface with a TestClient: define a metric over a certified derivation,
query it by dimension and time grain, hit the certified-source gate, and round-trip the
set through OSI. The LLM client is a stub; metrics never call it.
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
from elbi.metrics import MetricService

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

ORDERS = [
    {"region": "west", "day": "2026-01-01", "amount": 10, "converted": 1},
    {"region": "west", "day": "2026-01-02", "amount": 30, "converted": 0},
    {"region": "east", "day": "2026-02-01", "amount": 20, "converted": 1},
]

CERTIFIED = {"orders"}  # which source derivations are certified


def _load_source(name: str) -> list[dict[str, Any]]:
    if name not in CERTIFIED:
        raise KeyError(name)
    return [dict(r) for r in ORDERS]


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    service = MetricService(
        store=store,
        is_certified=lambda name: name in CERTIFIED,
        load_source=_load_source,
    )
    app = create_app(
        load_datasets=dict,
        client=MagicMock(),
        store=store,
        metric_service=service,
    )
    with TestClient(app) as http:
        yield http


REVENUE = {
    "name": "revenue",
    "type": "simple",
    "source": "orders",
    "measure": {"agg": "sum", "column": "amount"},
    "dimensions": ["region"],
    "timeDimension": {"column": "day", "grain": "month"},
}


def test_define_and_get_metric(client: TestClient) -> None:
    created = client.post("/api/metrics", json=REVENUE)
    assert created.status_code == 200
    assert created.json()["sourceCertified"] is True
    fetched = client.get("/api/metrics/revenue").json()
    assert fetched["measure"] == {"agg": "sum", "column": "amount"}
    assert [m["name"] for m in client.get("/api/metrics").json()] == ["revenue"]


def test_query_metric_by_dimension(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    response = client.post("/api/metrics/revenue/query", json={"group_by": ["region"]})
    assert response.status_code == 200
    body = response.json()
    assert body["columns"] == ["region", "revenue"]
    totals = {row["region"]: row["revenue"] for row in body["rows"]}
    assert totals == {"east": 20, "west": 40}


def test_query_metric_by_month_grain(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    response = client.post(
        "/api/metrics/revenue/query", json={"group_by": ["day"], "grain": "month"}
    )
    totals = {row["day"][:7]: row["revenue"] for row in response.json()["rows"]}
    assert totals == {"2026-01": 40, "2026-02": 20}


def test_define_rejects_uncertified_source(client: TestClient) -> None:
    response = client.post(
        "/api/metrics",
        json={
            "name": "draft_revenue",
            "type": "simple",
            "source": "draft_orders",
            "measure": {"agg": "sum", "column": "amount"},
        },
    )
    assert response.status_code == 400
    assert "not certified" in response.json()["detail"]


def test_query_rejects_undeclared_dimension(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    response = client.post("/api/metrics/revenue/query", json={"group_by": ["amount"]})
    assert response.status_code == 400
    assert "cannot group" in response.json()["detail"]


def test_ratio_metric_end_to_end(client: TestClient) -> None:
    client.post(
        "/api/metrics",
        json={
            "name": "conversions",
            "type": "simple",
            "source": "orders",
            "measure": {"agg": "sum", "column": "converted"},
            "dimensions": ["region"],
        },
    )
    client.post(
        "/api/metrics",
        json={
            "name": "order_count",
            "type": "simple",
            "source": "orders",
            "measure": {"agg": "count"},
            "dimensions": ["region"],
        },
    )
    assert (
        client.post(
            "/api/metrics",
            json={
                "name": "conversion_rate",
                "type": "ratio",
                "numerator": "conversions",
                "denominator": "order_count",
                "dimensions": ["region"],
            },
        ).status_code
        == 200
    )
    response = client.post(
        "/api/metrics/conversion_rate/query", json={"group_by": ["region"]}
    )
    rates = {row["region"]: row["conversion_rate"] for row in response.json()["rows"]}
    assert rates["east"] == 1.0
    assert rates["west"] == 0.5


def test_osi_export_and_import_roundtrip(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    document = client.get("/api/metrics/osi").json()
    assert document["semantic_model"][0]["datasets"][0]["name"] == "orders"

    # Re-import into a fresh store via a second client shares the class-level state, so
    # just confirm import accepts the exported document and returns the metric.
    imported = client.post("/api/metrics/osi", json={"document": document})
    assert imported.status_code == 200
    assert any(m["name"] == "revenue" for m in imported.json()["imported"])


def test_osi_export_of_an_empty_catalog_explains_itself(client: TestClient) -> None:
    """OSI needs at least one dataset, so an empty catalog is a 400, not a 500."""
    response = client.get("/api/metrics/osi")
    assert response.status_code == 400
    assert "nothing to export" in response.json()["detail"]


def test_osi_import_rejects_an_exported_record(client: TestClient) -> None:
    """A record is definition plus evidence, and import says so by name."""
    client.post("/api/metrics", json=REVENUE)
    record = client.get("/api/exports/metrics/revenue").json()
    assert record["schema"].startswith("elbi.export/")

    response = client.post("/api/metrics/osi", json={"document": record})
    assert response.status_code == 400
    assert "not an OSI document" in response.json()["detail"]


def test_delete_metric(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    assert client.delete("/api/metrics/revenue").status_code == 200
    assert client.get("/api/metrics/revenue").status_code == 404


def test_format_round_trips(client: TestClient) -> None:
    spec = {
        **REVENUE,
        "format": {"kind": "currency", "currency": "USD", "precision": 2},
    }
    assert client.post("/api/metrics", json=spec).status_code == 200
    fetched = client.get("/api/metrics/revenue").json()
    assert fetched["format"] == {"kind": "currency", "currency": "USD", "precision": 2}


def test_overview_gives_current_value_and_trend(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    overview = client.get("/api/metrics/overview").json()
    entry = next(e for e in overview if e["name"] == "revenue")
    assert entry["value"] == 20  # latest month (2026-02) revenue
    assert [p["v"] for p in entry["series"]] == [40, 20]  # monthly trend


def test_compile_returns_generated_sql(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    sql = client.post(
        "/api/metrics/revenue/compile", json={"group_by": ["region"]}
    ).json()["sql"]
    assert "SELECT" in sql and "GROUP BY" in sql and "sum" in sql.lower()


def test_history_appends_only_on_change(client: TestClient) -> None:
    client.post("/api/metrics", json=REVENUE)
    client.post("/api/metrics", json=REVENUE)  # identical re-save: no new version
    assert len(client.get("/api/metrics/revenue/history").json()) == 1
    client.post("/api/metrics", json={**REVENUE, "description": "Total revenue."})
    history = client.get("/api/metrics/revenue/history").json()
    assert [v["version"] for v in history] == [2, 1]  # newest first
    assert history[0]["verdict"] == "sound"


class _PlanClient:
    """A stub LLM client that returns a fixed metric-query plan as JSON."""

    def __init__(self, plan: str) -> None:
        self._plan = plan

    def step(self, transcript: object, tools: object) -> object:
        from elbi_agent import Step

        return Step(text=self._plan)


def _ask_client(plan: str, tmp_path: Path) -> TestClient:
    store = open_store(f"sqlite:{tmp_path / 'ask.db'}")
    service = MetricService(
        store=store,
        is_certified=lambda name: name in CERTIFIED,
        load_source=_load_source,
    )
    app = create_app(
        load_datasets=dict,
        client=_PlanClient(plan),
        store=store,
        metric_service=service,
    )
    http = TestClient(app)
    http.post("/api/metrics", json=REVENUE)
    return http


def test_ask_resolves_nl_to_a_metric_query(tmp_path: Path) -> None:
    plan = (
        '{"metric_name": "revenue", "group_by": ["region"], "grain": null, '
        '"filters": [{"column": "region", "op": "eq", "value": "west"}], '
        '"explanation": "Revenue in the west."}'
    )
    with _ask_client(plan, tmp_path) as http:
        body = http.post(
            "/api/metrics/ask", json={"question": "revenue in the west"}
        ).json()
    assert body["resolvedQuery"]["metricName"] == "revenue"
    assert body["rows"] == [{"region": "west", "revenue": 40}]
    assert body["explanation"] == "Revenue in the west."


def test_ask_rejects_a_hallucinated_dimension(tmp_path: Path) -> None:
    # The model picks a dimension the metric does not declare; the endpoint must decline
    # rather than run it (the text-to-semantic-query safety guarantee).
    plan = '{"metric_name": "revenue", "group_by": ["not_a_dimension"]}'
    with _ask_client(plan, tmp_path) as http:
        response = http.post("/api/metrics/ask", json={"question": "by nonsense"})
    assert response.status_code == 400
    assert "not_a_dimension" in response.json()["detail"]
