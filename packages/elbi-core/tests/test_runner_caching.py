"""Tests for the Runner's incremental caching, early cutoff, and policies."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import (
    Artifact,
    Context,
    Dataset,
    DerivationError,
    Registry,
    Runner,
    SemanticModel,
    cache,
    derivation,
    param,
    serve,
)
from elbi_core.cache import LocalCacheStore
from elbi_core.config import DataBindings
from elbi_core.metrics.osi import OSI_VERSION
from elbi_core.runner import _fresh, _output_version


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def sales(tmp_path: Path) -> tuple[Path, DataBindings]:
    path = tmp_path / "sales.csv"
    _write(path, "id,price\na,1\nb,2\n")
    return tmp_path, DataBindings(bindings={"sales": "sales.csv"})


def _model(metric_name: str = "total_amount") -> SemanticModel:
    return SemanticModel.from_osi(
        {
            "version": OSI_VERSION,
            "semantic_model": [
                {
                    "name": "sales_semantics",
                    "datasets": [{"name": "sales", "source": "sales", "fields": []}],
                    "metrics": [
                        {
                            "name": metric_name,
                            "expression": {
                                "dialects": [
                                    {
                                        "dialect": "ANSI_SQL",
                                        "expression": "SUM(amount)",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        }
    )


def test_semantic_model_input_is_reused_when_unchanged() -> None:
    registry = Registry()
    calls: list[int] = []

    @derivation(inputs={"model": _model()}, serve=serve.json(), registry=registry)
    def gate_count(ctx: Context) -> Artifact:
        calls.append(1)
        return Artifact.json(len(ctx.input("model").metrics.metrics))

    runner = Runner(registry)
    assert runner.run("gate_count").value == 1
    assert runner.run("gate_count").value == 1
    assert len(calls) == 1


def test_editing_a_definition_changes_the_data_version() -> None:
    # The document's content is the input version, so renaming a metric must
    # invalidate anything derived from it.
    def version_for(model: SemanticModel) -> str:
        registry = Registry()

        @derivation(
            name="gate_version",
            inputs={"model": model},
            serve=serve.json(),
            registry=registry,
        )
        def gate_version(ctx: Context) -> Artifact:
            return Artifact.json(1)

        return Runner(registry).data_version("gate_version")

    assert version_for(_model()) != version_for(_model("renamed_total"))


def test_reuse_within_session(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()
    calls: list[int] = []

    @derivation(inputs={"s": Dataset("sales")}, serve=serve.json(), registry=registry)
    def total(ctx: Context) -> Artifact:
        calls.append(1)
        return Artifact.json(sum(int(r["price"]) for r in ctx.input("s").rows))

    runner = Runner(registry, bindings=bindings, base_dir=base)
    assert runner.run("total").value == 3
    assert runner.run("total").value == 3
    assert len(calls) == 1  # computed once, reused


def test_recompute_when_input_changes(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()
    calls: list[int] = []

    @derivation(inputs={"s": Dataset("sales")}, serve=serve.json(), registry=registry)
    def total(ctx: Context) -> Artifact:
        calls.append(1)
        return Artifact.json(sum(int(r["price"]) for r in ctx.input("s").rows))

    runner = Runner(registry, bindings=bindings, base_dir=base)
    assert runner.run("total").value == 3
    _write(base / "sales.csv", "id,price\na,10\nb,20\n")  # input changed
    assert runner.run("total").value == 30
    assert len(calls) == 2  # recomputed because the file content changed


def test_cache_never_always_recomputes(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()
    calls: list[int] = []

    @derivation(
        inputs={"s": Dataset("sales")},
        serve=serve.json(),
        cache=cache.never(),
        registry=registry,
    )
    def total(ctx: Context) -> Artifact:
        calls.append(1)
        return Artifact.json(len(ctx.input("s").rows))

    runner = Runner(registry, bindings=bindings, base_dir=base)
    runner.run("total")
    runner.run("total")
    assert len(calls) == 2  # never cached


def test_persistent_store_across_runners(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()
    calls: list[int] = []

    @derivation(inputs={"s": Dataset("sales")}, serve=serve.json(), registry=registry)
    def total(ctx: Context) -> Artifact:
        calls.append(1)
        return Artifact.json(len(ctx.input("s").rows))

    store = LocalCacheStore(base / "cache")
    Runner(registry, bindings=bindings, base_dir=base, store=store).run("total")
    # A brand-new runner with the same on-disk store reuses the result.
    Runner(registry, bindings=bindings, base_dir=base, store=store).run("total")
    assert len(calls) == 1


def test_early_cutoff_skips_downstream(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()
    ids_calls: list[int] = []
    count_calls: list[int] = []

    @derivation(inputs={"s": Dataset("sales")}, registry=registry)  # internal
    def ids(ctx: Context) -> Artifact:
        ids_calls.append(1)
        return Artifact.table([{"id": r["id"]} for r in ctx.input("s").rows])

    @derivation(inputs={"rows": ids}, serve=serve.json(), registry=registry)
    def count_ids(ctx: Context) -> Artifact:
        count_calls.append(1)
        return Artifact.json(len(ctx.input("rows").value))

    runner = Runner(registry, bindings=bindings, base_dir=base)
    assert runner.run("count_ids").value == 2

    # Change only the price column. `ids` (which projects only id) recomputes
    # because the file changed, but produces the SAME output → early cutoff →
    # count_ids is NOT recomputed.
    _write(base / "sales.csv", "id,price\na,99\nb,88\n")
    assert runner.run("count_ids").value == 2
    assert len(ids_calls) == 2  # upstream recomputed
    assert len(count_calls) == 1  # downstream cut off (output unchanged)


def test_ttl_expiry_forces_recompute(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()
    calls: list[int] = []
    now = [0.0]

    @derivation(
        inputs={"s": Dataset("sales")},
        serve=serve.json(),
        cache=cache.auto(ttl=10),
        registry=registry,
    )
    def total(ctx: Context) -> Artifact:
        calls.append(1)
        return Artifact.json(len(ctx.input("s").rows))

    runner = Runner(registry, bindings=bindings, base_dir=base, clock=lambda: now[0])
    runner.run("total")
    now[0] = 5.0
    runner.run("total")  # within ttl → hit
    assert len(calls) == 1
    now[0] = 20.0
    runner.run("total")  # ttl expired → recompute
    assert len(calls) == 2


def test_internal_derivation_cannot_be_served(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()

    @derivation(inputs={"s": Dataset("sales")}, registry=registry)
    def internal(ctx: Context) -> Artifact:
        return Artifact.table(ctx.input("s").rows)

    runner = Runner(registry, bindings=bindings, base_dir=base)
    assert len(runner.run("internal").value) == 2  # runnable
    with pytest.raises(DerivationError, match="internal"):
        runner.serve("internal")  # not servable


def test_data_version_changes_with_input(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()

    @derivation(inputs={"s": Dataset("sales")}, serve=serve.json(), registry=registry)
    def total(ctx: Context) -> Artifact:
        return Artifact.json(len(ctx.input("s").rows))

    runner = Runner(registry, bindings=bindings, base_dir=base)
    v1 = runner.data_version("total")
    _write(base / "sales.csv", "id,price\na,1\n")
    assert runner.data_version("total") != v1


def test_output_version_helper() -> None:
    artifact = Artifact.json({"a": 1})
    deterministic = _output_version(artifact, cache.auto(), "dv")
    nondeterministic = _output_version(artifact, cache.auto(deterministic=False), "dv")
    assert deterministic != "dv"  # hash of output
    assert nondeterministic == "dv"  # falls back to data version
    # Opaque objects have no stable content hash, so they fall back too.
    assert _output_version(Artifact.opaque(object()), cache.auto(), "dv") == "dv"


def test_opaque_model_trained_once_and_reused(
    sales: tuple[Path, DataBindings],
) -> None:
    base, bindings = sales
    registry = Registry()
    train_calls: list[int] = []

    @derivation(inputs={"s": Dataset("sales")}, registry=registry)  # internal model
    def model(ctx: Context) -> Artifact:
        train_calls.append(1)
        bias = sum(int(r["price"]) for r in ctx.input("s").rows)
        return Artifact.opaque({"bias": bias})

    @derivation(
        inputs={"m": model},
        params={"x": param.number()},
        serve=serve.json(),
        registry=registry,
    )
    def predict(ctx: Context) -> Artifact:
        return Artifact.json(ctx.input("m").value["bias"] + ctx.param("x"))

    store = LocalCacheStore(base / "cache")
    first = Runner(registry, bindings=bindings, base_dir=base, store=store)
    assert first.run("predict", {"x": 1.0}).value == 4.0
    # A fresh runner reuses the pickled opaque model and the cached prediction.
    second = Runner(registry, bindings=bindings, base_dir=base, store=store)
    assert second.run("predict", {"x": 1.0}).value == 4.0
    assert len(train_calls) == 1


def test_unpicklable_output_warns_but_runs(sales: tuple[Path, DataBindings]) -> None:
    base, bindings = sales
    registry = Registry()

    @derivation(serve=serve.json(), registry=registry)
    def opaque(ctx: Context) -> Artifact:
        return Artifact.json(lambda: 1)  # lambdas can't be pickled

    runner = Runner(
        registry, bindings=bindings, base_dir=base, store=LocalCacheStore(base / "c")
    )
    with pytest.warns(UserWarning, match="could not persist cache"):
        result = runner.run("opaque")
    assert callable(result.value)  # the run still returns the value


def test_fresh_helper() -> None:
    from elbi_core.cache import CachedResult

    none_ttl = CachedResult(Artifact.text("x"), "ov", computed_at=0.0, ttl=None)
    assert _fresh(none_ttl, now=1000.0) is True
    with_ttl = CachedResult(Artifact.text("x"), "ov", computed_at=0.0, ttl=10.0)
    assert _fresh(with_ttl, now=5.0) is True
    assert _fresh(with_ttl, now=20.0) is False
