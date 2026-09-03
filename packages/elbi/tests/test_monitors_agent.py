"""Tests for the chat agent's monitoring capabilities (MonitorsAgent)."""

from __future__ import annotations

from pathlib import Path

from elbi.db import open_store
from elbi.monitoring import MonitorService
from elbi.monitoring_agent import MonitorsAgent


def _agent(tmp_path: Path) -> MonitorsAgent:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    service = MonitorService(
        store=store,
        read_value=lambda monitor: 1.0,
        source_certified=lambda kind, target: target == "revenue",
    )
    return MonitorsAgent(service)


def test_agent_creates_and_lists(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    message = agent.create(
        {"target_kind": "metric", "target": "revenue", "name": "rev"}
    )
    assert "Monitoring metric 'revenue'" in message
    listing = agent.list_monitors()
    assert "rev watches metric 'revenue'" in listing


def test_agent_rejects_uncertified_target(tmp_path: Path) -> None:
    message = _agent(tmp_path).create({"target_kind": "metric", "target": "nope"})
    assert "Could not create" in message
    assert "not certified" in message


def test_agent_lists_nothing_when_empty(tmp_path: Path) -> None:
    assert "No monitors" in _agent(tmp_path).list_monitors()
