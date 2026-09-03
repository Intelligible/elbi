"""Tests for content fingerprints, especially dependency-aware code_version."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from hypothesis import given, settings
from hypothesis import strategies as st

from elbi_core import versioning
from elbi_core.versioning import (
    _distribution_version,
    _is_user_function,
    _opaque,
    code_version,
)

# Module-level functions for callee / cycle tests.


def _h1() -> int:
    return 1


def _h2() -> int:
    return 1 + 1  # genuinely different logic from _h1


def _consumer() -> int:
    return PLUGGED()  # references a module global, swapped in tests


PLUGGED = _h1

# Module-level state for the blind-spot tests below (swapped in place per test).

HELPERS = ModuleType("_fake_helpers")
HELPERS.transform = _h1  # type: ignore[attr-defined]

THRESHOLD = 0.5

LOOKUP = [1, 2, 3]


def _attribute_consumer() -> int:
    return HELPERS.transform()  # a call through a module attribute, not a bare name


def _threshold_consumer(x: float) -> bool:
    return x > THRESHOLD  # branches on a non-callable module global


def _lookup_consumer() -> int:
    return sum(LOOKUP)  # reads a mutable container global by value


def _make_mutating_consumer() -> Callable[[], int]:
    log: list[str] = []  # a closure nonlocal the returned function mutates as it runs

    def consumer() -> int:
        log.append("ran")
        return len(log)

    return consumer


class _UserModel:
    def score(self) -> int:
        return 1


def _class_consumer() -> int:
    return _UserModel().score()  # constructs a user class and calls its method


def _recurse_a() -> int:
    return _recurse_b()


def _recurse_b() -> int:
    return _recurse_a()


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{path.stem}_{id(path)}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- basic fingerprints ----------------------------------------------------


def test_hash_helpers() -> None:
    assert versioning.hash_text("a") == versioning.hash_text("a")
    assert versioning.hash_json({"a": 1, "b": 2}) == versioning.hash_json(
        {"b": 2, "a": 1}
    )


def test_hash_file(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert versioning.hash_file(p) == versioning.hash_bytes(b"hello")


def test_derivation_version_is_input_order_independent() -> None:
    a = versioning.derivation_version(
        code="c", params="p", input_versions={"x": "1", "y": "2"}
    )
    b = versioning.derivation_version(
        code="c", params="p", input_versions={"y": "2", "x": "1"}
    )
    assert a == b


# --- code_version: own logic ----------------------------------------------


def test_code_version_deterministic() -> None:
    assert code_version(_h1) == code_version(_h1)


def test_code_version_distinguishes_logic() -> None:
    assert code_version(_h1) != code_version(_h2)


def test_code_version_ignores_comments_and_whitespace(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    b.write_text("def f(x):\n    # a comment\n    return x   +   1\n", encoding="utf-8")
    assert code_version(_load(a).f) == code_version(_load(b).f)


def test_code_version_detects_logic_change(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    b.write_text("def f(x):\n    return x + 2\n", encoding="utf-8")
    assert code_version(_load(a).f) != code_version(_load(b).f)


# --- code_version: dependencies (the fix for the callee blind spot) --------


def test_code_version_tracks_called_helper() -> None:
    g = _consumer.__globals__
    g["PLUGGED"] = _h1
    v1 = code_version(_consumer)
    g["PLUGGED"] = _h2  # the helper it calls changed
    v2 = code_version(_consumer)
    g["PLUGGED"] = _h1  # restore
    assert v1 != v2


def test_code_version_cycle_terminates() -> None:
    assert code_version(_recurse_a)  # mutually recursive helpers must not hang


def test_code_version_tracks_attribute_called_helper() -> None:
    # A helper called as ``module.func()``, not a bare name, must still be tracked.
    HELPERS.transform = _h1  # type: ignore[attr-defined]
    v1 = code_version(_attribute_consumer)
    HELPERS.transform = _h2  # type: ignore[attr-defined]  # its body changed
    v2 = code_version(_attribute_consumer)
    HELPERS.transform = _h1  # type: ignore[attr-defined]  # restore
    assert v1 != v2


def test_code_version_tracks_class_method_of_referenced_class() -> None:
    # A referenced user class whose method body changes must bust the version.
    original = _UserModel.score
    v1 = code_version(_class_consumer)
    _UserModel.score = lambda self: 2  # type: ignore[assignment,method-assign]
    try:
        v2 = code_version(_class_consumer)
    finally:
        _UserModel.score = original  # type: ignore[method-assign]
    assert v1 != v2


# --- code_version: runtime-mutable globals (blind spot #2) -----------------


def test_code_version_tracks_scalar_global() -> None:
    g = _threshold_consumer.__globals__
    g["THRESHOLD"] = 0.5
    v1 = code_version(_threshold_consumer)
    g["THRESHOLD"] = 0.9  # the constant it branches on changed
    v2 = code_version(_threshold_consumer)
    g["THRESHOLD"] = 0.5  # restore
    assert v1 != v2


def test_code_version_ignores_mutated_closure_nonlocal() -> None:
    # A closure nonlocal the compute mutates must not shift the version between calls,
    # or a derivation could never hit its own cache within a run (see runner caching).
    consumer = _make_mutating_consumer()
    v1 = code_version(consumer)
    consumer()  # mutates the captured `log`
    consumer()
    v2 = code_version(consumer)
    assert v1 == v2


def test_code_version_global_container_is_content_addressed() -> None:
    g = _lookup_consumer.__globals__
    g["LOOKUP"] = [1, 2, 3]
    v1 = code_version(_lookup_consumer)
    g["LOOKUP"] = [1, 2, 3]  # a distinct object with equal content
    v2 = code_version(_lookup_consumer)
    g["LOOKUP"] = [1, 2, 4]  # content changed
    v3 = code_version(_lookup_consumer)
    g["LOOKUP"] = [1, 2, 3]  # restore
    assert v1 == v2  # content-addressed, not identity: cache still hits
    assert v1 != v3


# --- code_version / _opaque: library version drift (blind spot #3) ---------


def test_distribution_version_tags_third_party() -> None:
    assert _distribution_version("yaml")  # PyYAML is installed → a version string
    assert _distribution_version("json") is None  # stdlib carries no distribution
    assert _distribution_version("builtins") is None
    assert _distribution_version("_nonexistent_local_module") is None


def test_opaque_tags_third_party_with_version() -> None:
    import yaml  # a third-party package under site-packages

    tag = _opaque(yaml.dump)
    assert "@" in tag  # carries an installed-version marker


def test_opaque_leaves_stdlib_unversioned() -> None:
    tag = _opaque(json.dumps)
    assert tag.endswith("dumps")
    assert "@" not in tag


def test_code_version_handles_nested_code_objects() -> None:
    def with_comprehension() -> list[int]:
        return [x * 2 for x in range(3)]

    assert code_version(with_comprehension)  # co_consts code-object path


def test_code_version_no_source_falls_back() -> None:
    namespace: dict[str, object] = {}
    exec("def f(x=5):\n    return x + 1", namespace)
    fn = namespace["f"]
    assert code_version(fn) == code_version(fn)  # deterministic via bytecode fallback


# --- helpers ---------------------------------------------------------------


def test_is_user_function() -> None:
    import yaml  # a third-party package installed under site-packages

    assert _is_user_function(_h1) is True
    assert _is_user_function(json.dumps) is False  # stdlib
    assert _is_user_function(yaml.dump) is False  # third-party (site-packages)
    assert _is_user_function(len) is False  # builtin (not a FunctionType)
    namespace: dict[str, object] = {}
    exec("def f(): return 1", namespace)
    assert _is_user_function(namespace["f"]) is False  # <string> source


def test_opaque_identifies_by_name() -> None:
    assert _opaque(json.dumps).endswith("dumps")


def test_source_version_ignores_formatting() -> None:
    a = versioning.source_version("def f(ctx):\n    return 1\n")
    b = versioning.source_version("def f(ctx):  # comment\n    return 1\n")
    assert a == b


def test_source_version_changes_with_logic() -> None:
    a = versioning.source_version("def f(ctx):\n    return 1\n")
    b = versioning.source_version("def f(ctx):\n    return 2\n")
    assert a != b


def test_hash_json_is_hash_of_canonical_json() -> None:
    value = {"b": 1, "a": [3, {"z": 2, "y": 1}]}
    assert versioning.hash_json(value) == versioning.hash_text(
        versioning.canonical_json(value)
    )


# JSON-native values only (no floats, so round-tripping is exact).
_JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=15,
)


@settings(deadline=None, max_examples=100)
@given(value=_JSON)
def test_canonical_json_is_idempotent(value: object) -> None:
    once = versioning.canonical_json(value)
    assert versioning.canonical_json(json.loads(once)) == once


@settings(deadline=None, max_examples=100)
@given(data=st.dictionaries(st.text(), st.integers()))
def test_canonical_json_independent_of_key_order(data: dict[str, int]) -> None:
    reordered = dict(reversed(list(data.items())))
    assert versioning.canonical_json(reordered) == versioning.canonical_json(data)
