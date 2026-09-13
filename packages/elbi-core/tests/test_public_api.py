"""Guard the public API surface."""

from __future__ import annotations

import elbi_core


def test_version_is_exposed() -> None:
    assert isinstance(elbi_core.__version__, str)


def test_all_names_are_importable() -> None:
    for name in elbi_core.__all__:
        assert hasattr(elbi_core, name), f"{name} missing from package"


def test_serve_namespace_has_builders() -> None:
    for builder in ("table", "markdown", "json", "text", "components"):
        assert callable(getattr(elbi_core.serve, builder))


def test_registry_alias_is_a_registry() -> None:
    assert isinstance(elbi_core.registry, elbi_core.Registry)
