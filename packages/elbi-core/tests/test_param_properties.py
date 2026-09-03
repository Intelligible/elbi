"""Property tests for parameter coercion across the full type space.

Generated values cover scalars and nested object/array shapes, so coercion is
checked against inputs example-based tests would not enumerate.
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from elbi_core import param

_json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=15,
)


@given(value=st.dictionaries(st.text(), _json_values))
@settings(deadline=None)
def test_object_coerce_returns_equal_mapping(value: dict) -> None:
    assert param.object().coerce(value) == value


@given(value=st.lists(_json_values))
@settings(deadline=None)
def test_untyped_array_passes_elements_through(value: list) -> None:
    assert param.array().coerce(value) == value


@given(value=st.lists(st.integers()))
@settings(deadline=None)
def test_typed_array_coerces_each_element(value: list) -> None:
    assert param.array(items="integer").coerce(value) == value


@given(value=st.integers() | st.text() | st.booleans() | st.none())
@settings(deadline=None)
def test_object_coerce_rejects_non_mappings(value: object) -> None:
    with pytest.raises(ValueError, match="object"):
        param.object().coerce(value)
