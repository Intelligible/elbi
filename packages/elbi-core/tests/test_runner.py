"""Tests for the local Runner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from elbi_core import (
    Artifact,
    Context,
    CycleError,
    Dataset,
    Derivation,
    Registry,
    Runner,
    serve,
)
from elbi_core.config import DataBindings


@pytest.fixture
def sales_bindings(tmp_path: Path) -> tuple[DataBindings, Path]:
    (tmp_path / "sales.csv").write_text(
        "customer_id,amount\nc1,100\nc2,5\n", encoding="utf-8"
    )
    bindings = DataBindings(bindings={"sales": "sales.csv"})
    return bindings, tmp_path


def test_run_dataset_backed_derivation(
    registry: Registry, sales_bindings: tuple[DataBindings, Path]
) -> None:
    bindings, base = sales_bindings

    @derivation_in(registry)
    def churn_risk(ctx: Context) -> Artifact:
        rows = ctx.input("sales").rows
        return Artifact.table([{"customer_id": r["customer_id"]} for r in rows])

    runner = Runner(registry, bindings=bindings, base_dir=base)
    artifact = runner.run("churn_risk")
    assert artifact.value == [{"customer_id": "c1"}, {"customer_id": "c2"}]


def test_serve_renders_through_contract(
    registry: Registry, sales_bindings: tuple[DataBindings, Path]
) -> None:
    bindings, base = sales_bindings
    contract = serve.table(title="Risk", columns=["customer_id"])

    @derivation_in(registry, contract=contract)
    def churn_risk(ctx: Context) -> Artifact:
        rows = ctx.input("sales").rows
        return Artifact.table([{"customer_id": r["customer_id"]} for r in rows])

    runner = Runner(registry, bindings=bindings, base_dir=base)
    rendered = runner.serve("churn_risk")
    assert rendered.startswith("# Risk")
    assert "| customer_id |" in rendered


def test_composed_derivation_and_caching(registry: Registry) -> None:
    from elbi_core import derivation

    calls: list[str] = []

    @derivation(serve=serve.json(), registry=registry)
    def base_value(ctx: Context) -> Artifact:
        calls.append("base")
        return Artifact.json(21)

    @derivation(inputs={"b": base_value}, serve=serve.json(), registry=registry)
    def doubled(ctx: Context) -> Artifact:
        return Artifact.json(ctx.input("b").value * 2)

    @derivation(
        inputs={"b": base_value, "d": doubled}, serve=serve.json(), registry=registry
    )
    def total(ctx: Context) -> Artifact:
        return Artifact.json(ctx.input("b").value + ctx.input("d").value)

    runner = Runner(registry)
    assert runner.run("total").value == 63
    # base_value is shared by two consumers but computed once.
    assert calls == ["base"]


def test_cycle_detection(registry: Registry) -> None:
    from elbi_core import derivation

    @derivation(serve=serve.text(), registry=registry)
    def a(ctx: Context) -> str:
        return "a"

    @derivation(inputs={"a": a}, serve=serve.text(), registry=registry)
    def b(ctx: Context) -> str:
        return "b"

    # Force a cycle a -> b -> a by mutating the frozen dataclass (test-only).
    object.__setattr__(a, "inputs", {"b": b})

    runner = Runner(registry)
    with pytest.raises(CycleError, match="cycle"):
        runner.run("a")


# --- helpers ---------------------------------------------------------------


def derivation_in(
    registry: Registry, *, contract: object | None = None
) -> Callable[..., Derivation]:
    """Decorator factory binding a derivation to ``registry`` with a sales input."""
    from elbi_core import derivation

    return derivation(
        inputs={"sales": Dataset("sales")},
        serve=contract or serve.table(),
        registry=registry,
    )
