"""Tests for typed parameters."""

from __future__ import annotations

import pytest

from elbi_core import Param, param


def test_builders_set_type_and_options() -> None:
    p = param.string(description="zip", required=False, default="x")
    assert (p.type, p.description, p.required, p.default) == (
        "string",
        "zip",
        False,
        "x",
    )
    assert param.integer().type == "integer"
    assert param.number().type == "number"
    assert param.boolean().type == "boolean"


def test_python_type_mapping() -> None:
    assert param.string().python_type() is str
    assert param.integer().python_type() is int
    assert param.number().python_type() is float
    assert param.boolean().python_type() is bool


def test_coerce_casts_strings() -> None:
    assert param.integer().coerce("42") == 42
    assert param.number().coerce("3.5") == 3.5
    assert param.string().coerce(10) == "10"


@pytest.mark.parametrize("truthy", ["true", "True", "1", "yes", "y"])
def test_coerce_bool_truthy_strings(truthy: str) -> None:
    assert param.boolean().coerce(truthy) is True


@pytest.mark.parametrize("falsy", ["false", "False", "0", "no", "n"])
def test_coerce_bool_falsy_strings(falsy: str) -> None:
    assert param.boolean().coerce(falsy) is False


def test_coerce_bool_passthrough_and_invalid() -> None:
    assert param.boolean().coerce(True) is True
    with pytest.raises(ValueError, match="boolean"):
        param.boolean().coerce("maybe")
    with pytest.raises(ValueError, match="boolean"):
        param.boolean().coerce(3.5)


def test_coerce_bool_not_treated_as_int() -> None:
    # A bool must not silently satisfy an integer param.
    assert param.integer().coerce(True) == 1  # explicit cast, not pass-through


def test_coerce_invalid_raises() -> None:
    with pytest.raises(ValueError, match="cannot coerce"):
        param.integer().coerce("not-a-number")


def test_to_manifest_omits_defaults() -> None:
    assert Param("string").to_manifest() == {"type": "string"}
    full = Param("integer", description="d", required=False, default=5).to_manifest()
    assert full == {
        "type": "integer",
        "description": "d",
        "required": False,
        "default": 5,
    }


def test_structured_builders_set_type_and_items() -> None:
    obj = param.object(description="a record")
    assert (obj.type, obj.items) == ("object", None)
    arr = param.array(items="object", required=False)
    assert (arr.type, arr.items, arr.required) == ("array", "object", False)
    assert param.array().items is None


def test_structured_python_type_and_annotation() -> None:
    assert param.object().python_type() is dict
    assert param.array().python_type() is list
    assert param.object().annotation() is dict
    assert param.array(items="number").annotation() == list[float]
    assert param.array(items="object").annotation() == list[dict]
    assert param.array().annotation() is list


def test_coerce_object_requires_mapping() -> None:
    assert param.object().coerce({"a": 1}) == {"a": 1}
    with pytest.raises(ValueError, match="object"):
        param.object().coerce([1, 2])


def test_coerce_array_casts_elements_by_items() -> None:
    assert param.array(items="number").coerce(["1.5", 2]) == [1.5, 2.0]
    assert param.array(items="object").coerce([{"a": 1}]) == [{"a": 1}]
    # Without items, elements pass through untouched.
    assert param.array().coerce([1, "x"]) == [1, "x"]


def test_coerce_array_rejects_non_sequence_and_strings() -> None:
    with pytest.raises(ValueError, match="array"):
        param.array().coerce("abc")  # a string is not an array
    with pytest.raises(ValueError, match="array"):
        param.array().coerce(5)


def test_structured_to_manifest_emits_items() -> None:
    assert param.object().to_manifest() == {"type": "object"}
    assert param.array(items="string").to_manifest() == {
        "type": "array",
        "items": "string",
    }


def test_items_only_valid_on_array() -> None:
    with pytest.raises(ValueError, match="items is only valid"):
        Param("string", items="number")
