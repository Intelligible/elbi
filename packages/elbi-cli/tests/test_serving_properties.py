"""Property tests for the serving cache key (the unhashable-dict bug class).

The key must stay hashable and content-stable for every shape an MCP client can
send, including the nested object/array values that broke the original
items-tuple key.
"""

from __future__ import annotations

import copy

import hypothesis.strategies as st
from hypothesis import example, given, settings

from elbi_cli.serving import _params_key

# The space of JSON-shaped param values: scalars plus nested arrays/objects.
_json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=15,
)
_param_maps = st.dictionaries(st.text(min_size=1), _json_values)


@given(params=_param_maps)
@example(params={"filter": {"status": "active"}})  # the dict value that crashed
@example(params={"ids": [1, 2, 3]})  # and the list-valued variant
@settings(deadline=None)
def test_params_key_is_hashable(params: dict) -> None:
    hash(_params_key(params))  # must never raise "unhashable type: dict"


@given(params=_param_maps)
@settings(deadline=None)
def test_params_key_depends_only_on_content(params: dict) -> None:
    # Two structurally equal inputs must produce the same key, so caching keys on
    # content and not on object identity.
    assert _params_key(params) == _params_key(copy.deepcopy(params))
