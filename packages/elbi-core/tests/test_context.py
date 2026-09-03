"""Tests for the Context passed to compute functions."""

from __future__ import annotations

import pytest

from elbi_core import MissingInputError, ParamError
from elbi_core.context import Context


def test_input_returns_resolved_value() -> None:
    ctx = Context({"sales": [1, 2, 3]})
    assert ctx.input("sales") == [1, 2, 3]


def test_input_missing_raises_with_available() -> None:
    ctx = Context({"sales": 1})
    with pytest.raises(MissingInputError, match="available inputs: sales"):
        ctx.input("nope")


def test_inputs_property_is_a_copy() -> None:
    ctx = Context({"a": 1})
    snapshot = ctx.inputs
    snapshot["a"] = 999  # mutating the view must not affect the context
    assert ctx.input("a") == 1


def test_param_returns_value() -> None:
    ctx = Context({}, {"zipcode": "98103"})
    assert ctx.param("zipcode") == "98103"


def test_param_missing_raises_with_available() -> None:
    ctx = Context({}, {"zipcode": "x"})
    with pytest.raises(ParamError, match="declared params: zipcode"):
        ctx.param("nope")


def test_param_missing_with_none_declared() -> None:
    ctx = Context({})
    with pytest.raises(ParamError, match=r"declared params: \(none\)"):
        ctx.param("anything")


def test_params_property_is_a_copy() -> None:
    ctx = Context({}, {"a": 1})
    snapshot = ctx.params
    snapshot["a"] = 999
    assert ctx.param("a") == 1
