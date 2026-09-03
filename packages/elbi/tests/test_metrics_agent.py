"""Tests for the chat agent's metrics capabilities (MetricsAgent over MetricService)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from elbi.db import open_store
from elbi.metrics import MetricService
from elbi.metrics_agent import MetricsAgent

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

ORDERS = [
    {"region": "west", "amount": 10},
    {"region": "west", "amount": 30},
    {"region": "east", "amount": 20},
]


def _agent(tmp_path: Path) -> MetricsAgent:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    service = MetricService(
        store=store,
        is_certified=lambda name: name == "orders",
        load_source=lambda name: [dict(r) for r in ORDERS],
    )
    return MetricsAgent(service)


REVENUE: dict[str, Any] = {
    "name": "revenue",
    "type": "simple",
    "source": "orders",
    "measure": {"agg": "sum", "column": "amount"},
    "dimensions": ["region"],
}


def test_agent_defines_lists_and_queries(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert "Defined metric 'revenue'" in agent.define(REVENUE)
    assert "revenue" in agent.list_metrics()
    rendered = agent.query("revenue", ["region"])
    assert "| region | revenue |" in rendered
    assert "| east | 20 |" in rendered
    assert "| west | 40 |" in rendered


def test_agent_define_rejects_uncertified_source(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    message = agent.define({**REVENUE, "name": "bad", "source": "not_certified"})
    assert "Could not define" in message
    assert "not certified" in message


def test_agent_query_reports_bad_dimension(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.define(REVENUE)
    assert "cannot group" in agent.query("revenue", ["amount"])


def test_agent_lists_nothing_when_empty(tmp_path: Path) -> None:
    assert "No metrics" in _agent(tmp_path).list_metrics()
