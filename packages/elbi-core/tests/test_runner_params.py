"""Tests for parameterized derivations end to end through the Runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import (
    Artifact,
    Context,
    Dataset,
    ParamError,
    Registry,
    Runner,
    derivation,
    param,
    serve,
)
from elbi_core.config import DataBindings


@pytest.fixture
def houses(tmp_path: Path) -> tuple[Registry, DataBindings, Path]:
    (tmp_path / "houses.csv").write_text(
        "zipcode,price\n98001,100\n98001,300\n98002,200\n", encoding="utf-8"
    )
    registry = Registry()

    @derivation(
        inputs={"houses": Dataset("houses")},
        params={
            "zipcode": param.string(description="Zip to filter"),
            "max_price": param.integer(required=False, default=0),
        },
        serve=serve.table(),
        registry=registry,
    )
    def houses_in_zip(ctx: Context) -> Artifact:
        zipcode = ctx.param("zipcode")
        max_price = ctx.param("max_price")
        rows = [r for r in ctx.input("houses").rows if r["zipcode"] == zipcode]
        if max_price:
            rows = [r for r in rows if float(r["price"]) <= max_price]
        return Artifact.table(rows)

    return registry, DataBindings(bindings={"houses": "houses.csv"}), tmp_path


def _runner(houses: tuple[Registry, DataBindings, Path]) -> Runner:
    registry, bindings, base = houses
    return Runner(registry, bindings=bindings, base_dir=base)


def test_manifest_includes_params(houses: tuple[Registry, DataBindings, Path]) -> None:
    registry, _, _ = houses
    manifest = registry.get("houses_in_zip").to_manifest()
    assert manifest["params"]["zipcode"] == {
        "type": "string",
        "description": "Zip to filter",
    }
    assert manifest["params"]["max_price"] == {
        "type": "integer",
        "required": False,
        "default": 0,
    }


def test_bad_param_coercion_raises(houses: tuple[Registry, DataBindings, Path]) -> None:
    with pytest.raises(ParamError, match="cannot coerce"):
        _runner(houses).run("houses_in_zip", {"zipcode": "98001", "max_price": "abc"})


def test_run_with_params_filters(houses: tuple[Registry, DataBindings, Path]) -> None:
    artifact = _runner(houses).run("houses_in_zip", {"zipcode": "98001"})
    assert [r["price"] for r in artifact.value] == ["100", "300"]


def test_params_are_coerced(houses: tuple[Registry, DataBindings, Path]) -> None:
    # max_price arrives as a string (as it would over the wire) and is coerced.
    artifact = _runner(houses).run(
        "houses_in_zip", {"zipcode": "98001", "max_price": "150"}
    )
    assert [r["price"] for r in artifact.value] == ["100"]


def test_missing_required_param_raises(
    houses: tuple[Registry, DataBindings, Path],
) -> None:
    with pytest.raises(ParamError, match="missing required parameter 'zipcode'"):
        _runner(houses).run("houses_in_zip", {})


def test_unknown_param_raises(houses: tuple[Registry, DataBindings, Path]) -> None:
    with pytest.raises(ParamError, match="unknown parameter"):
        _runner(houses).run("houses_in_zip", {"zipcode": "98001", "bogus": 1})


def test_default_applied_when_omitted(
    houses: tuple[Registry, DataBindings, Path],
) -> None:
    # max_price defaults to 0 (falsy → no price filter).
    artifact = _runner(houses).run("houses_in_zip", {"zipcode": "98002"})
    assert [r["price"] for r in artifact.value] == ["200"]


def test_cache_keyed_by_params(houses: tuple[Registry, DataBindings, Path]) -> None:
    runner = _runner(houses)
    a = runner.run("houses_in_zip", {"zipcode": "98001"})
    b = runner.run("houses_in_zip", {"zipcode": "98002"})
    assert a is not b
    assert runner.run("houses_in_zip", {"zipcode": "98001"}) is a
