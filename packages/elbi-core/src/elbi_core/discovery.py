"""Discover derivations by importing a project's modules.

Importing a module runs its ``@derivation`` decorators, which register the
derivations. :func:`discover` imports every top-level ``.py`` file in a directory
and returns the derivations that became registered as a result.

The directory is loaded as a synthetic package so that derivation modules can
import one another with relative imports (``from .churn_risk import churn_risk``)
to compose. Each call uses a fresh package name, so repeated discovery of
different directories never collides in ``sys.modules``.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

from .errors import DerivationError
from .registry import Registry, default_registry, use_registry

_PACKAGE_COUNTER = itertools.count()


def discover(directory: Path, *, registry: Registry | None = None) -> list[str]:
    """Import every top-level module under ``directory`` to populate ``registry``.

    Args:
        directory: A directory of derivation modules (e.g. ``derivations/``).
        registry: Registry to populate. Defaults to the shared
            :data:`~elbi.registry.default_registry`.

    Returns:
        The names registered while importing, sorted.

    Raises:
        DerivationError: if ``directory`` does not exist, or a module fails to
            import.
    """
    target = registry if registry is not None else default_registry
    if not directory.exists() or not directory.is_dir():
        raise DerivationError(f"derivations directory not found: {directory}")

    package_name = f"_elbi_derivations_{next(_PACKAGE_COUNTER)}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package

    before = set(target.names())
    try:
        with use_registry(target):
            for module_path in sorted(directory.glob("*.py")):
                if module_path.name.startswith("_"):
                    continue
                _import_module(package_name, module_path)
    finally:
        # The synthetic package stays in sys.modules so already-imported sibling
        # references remain valid; it is namespaced per call so it never clashes.
        pass

    after = set(target.names())
    return sorted(after - before)


def _import_module(package_name: str, path: Path) -> None:
    module_name = f"{package_name}.{path.stem}"
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise DerivationError(f"could not load derivation module: {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise DerivationError(f"failed to import {path}: {exc}") from exc
