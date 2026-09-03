"""Tests for asset materialization: staleness and dependency-ordered runs.

The cases that matter: staleness is computed from the content version versus the last
materialized version; a run materializes in the given order, skips assets already fresh,
records per-asset outcomes, and retries a transient failure before giving up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from elbi_core import (
    Artifact,
    Context,
    Dataset,
    Registry,
    Runner,
    asset_status,
    derivation,
    lineage_from_registry,
    materialize_assets,
)
from elbi_core.config import DataBindings
from elbi_core.orchestration.run import MaterializationStep
from elbi_core.registry import use_registry


def _project(tmp_path: Path) -> tuple[Runner, Registry]:
    (tmp_path / "sales.csv").write_text("amount\n10\n20\n", encoding="utf-8")
    registry = Registry()
    with use_registry(registry):

        @derivation(inputs={"sales": Dataset("sales")})
        def clean_sales(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("sales").rows)

        @derivation(inputs={"rows": clean_sales})
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table(ctx.input("rows").value)

    runner = Runner(
        registry,
        bindings=DataBindings(bindings={"sales": "sales.csv"}),
        base_dir=tmp_path,
    )
    return runner, registry


def test_asset_status_never_materialized(tmp_path: Path) -> None:
    runner, _ = _project(tmp_path)
    status = asset_status(runner, ["clean_sales", "revenue"], lambda _: None)
    assert status == {"clean_sales": "never", "revenue": "never"}


def test_asset_status_materialized_vs_stale(tmp_path: Path) -> None:
    runner, _ = _project(tmp_path)
    current = runner.data_version("revenue")
    # Last materialized at the current version → fresh; at an old version → stale.
    assert asset_status(runner, ["revenue"], lambda _: current) == {
        "revenue": "materialized"
    }
    assert asset_status(runner, ["revenue"], lambda _: "old") == {"revenue": "stale"}


def test_materialize_runs_and_records_versions(tmp_path: Path) -> None:
    runner, _ = _project(tmp_path)
    result = materialize_assets(
        runner, ["clean_sales", "revenue"], last_version=lambda _: None
    )
    assert result.ok
    assert [s.state for s in result.steps] == ["succeeded", "succeeded"]
    assert all(s.data_version for s in result.steps)


def test_materialize_skips_fresh_assets(tmp_path: Path) -> None:
    runner, _ = _project(tmp_path)
    fresh = {name: runner.data_version(name) for name in ("clean_sales", "revenue")}
    result = materialize_assets(
        runner, ["clean_sales", "revenue"], last_version=fresh.get
    )
    assert [s.state for s in result.steps] == ["skipped", "skipped"]


def test_materialize_retries_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = _project(tmp_path)
    original = runner.run
    calls = {"n": 0}

    def flaky(name: str, *args: Any, **kwargs: Any) -> Artifact:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(runner, "run", flaky)
    result = materialize_assets(
        runner,
        ["clean_sales"],
        last_version=lambda _: None,
        max_retries=1,
        sleep=lambda _: None,
    )
    assert result.ok
    assert result.steps[0].attempts == 2


def test_materialize_records_failure_after_exhausting_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = _project(tmp_path)

    def always_fail(name: str, *args: Any, **kwargs: Any) -> Artifact:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "run", always_fail)
    result = materialize_assets(
        runner,
        ["clean_sales"],
        last_version=lambda _: None,
        max_retries=2,
        sleep=lambda _: None,
    )
    assert not result.ok
    assert result.failed == ["clean_sales"]
    assert result.steps[0].attempts == 3


def test_on_step_callback_fires_per_asset(tmp_path: Path) -> None:
    runner, _ = _project(tmp_path)
    seen: list[MaterializationStep] = []
    materialize_assets(
        runner,
        ["clean_sales", "revenue"],
        last_version=lambda _: None,
        on_step=seen.append,
    )
    assert [s.asset for s in seen] == ["clean_sales", "revenue"]


def test_order_comes_from_the_lineage_graph(tmp_path: Path) -> None:
    _, registry = _project(tmp_path)
    graph = lineage_from_registry(registry)
    order = [
        nid.split(":", 1)[1]
        for nid in graph.topo_order()
        if nid.startswith("derivation:")
    ]
    assert order == ["clean_sales", "revenue"]
