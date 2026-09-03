"""Tests for the local lifecycle (certification) sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_cli.lifecycle import LifecycleStore
from elbi_core.errors import ConfigError


def test_absent_sidecar_is_empty(tmp_path: Path) -> None:
    store = LifecycleStore(tmp_path / "lifecycle.yaml")
    assert store.statuses() == {}


def test_set_status_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "lifecycle.yaml"
    store = LifecycleStore(path)
    store.set_status("revenue", "certified")
    assert path.exists()
    assert LifecycleStore(path).statuses() == {"revenue": "certified"}


def test_set_status_merges(tmp_path: Path) -> None:
    store = LifecycleStore(tmp_path / "lifecycle.yaml")
    store.set_status("a", "certified")
    store.set_status("b", "proposed")
    assert store.statuses() == {"a": "certified", "b": "proposed"}


def test_set_status_rejects_unknown_status(tmp_path: Path) -> None:
    store = LifecycleStore(tmp_path / "lifecycle.yaml")
    with pytest.raises(ConfigError, match="status must be one of"):
        store.set_status("a", "approved")


def test_statuses_rejects_unknown_value(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.yaml"
    path.write_text("a: approved\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be one of"):
        LifecycleStore(path).statuses()


def test_statuses_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        LifecycleStore(path).statuses()
