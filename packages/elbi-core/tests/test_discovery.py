"""Tests for derivation discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import DerivationError, Registry
from elbi_core.discovery import discover

MODULE = """
from elbi_core import Artifact, Context, derivation, serve


@derivation(name="{name}", serve=serve.text())
def {name}(ctx: Context) -> Artifact:
    return Artifact.text("{name}")
"""


def test_discover_imports_and_registers(tmp_path: Path) -> None:
    registry = Registry()
    derivations_dir = tmp_path / "derivations"
    derivations_dir.mkdir()
    for name in ("alpha", "beta"):
        (derivations_dir / f"{name}.py").write_text(
            MODULE.format(name=name), encoding="utf-8"
        )
    # A private module is skipped.
    (derivations_dir / "_helpers.py").write_text("X = 1\n", encoding="utf-8")

    found = discover(derivations_dir, registry=registry)
    assert found == ["alpha", "beta"]
    assert registry.names() == ["alpha", "beta"]


def test_discover_handles_sibling_imported_before_loop_reaches_it(
    tmp_path: Path,
) -> None:
    # A consumer module that sorts first imports a provider that sorts later via a
    # relative import; by the time the loop reaches the provider file it is already
    # in sys.modules and must be skipped (not re-imported).
    registry = Registry()
    derivations_dir = tmp_path / "derivations"
    derivations_dir.mkdir()
    (derivations_dir / "a_consumer.py").write_text(
        "from .z_provider import provider\n"
        "from elbi_core import Artifact, Context, derivation, serve\n\n"
        "@derivation(name='consumer', inputs={'p': provider}, serve=serve.text())\n"
        "def consumer(ctx: Context) -> Artifact:\n"
        "    return Artifact.text('c')\n",
        encoding="utf-8",
    )
    (derivations_dir / "z_provider.py").write_text(
        "from elbi_core import Artifact, Context, derivation, serve\n\n"
        "@derivation(name='provider', serve=serve.text())\n"
        "def provider(ctx: Context) -> Artifact:\n"
        "    return Artifact.text('p')\n",
        encoding="utf-8",
    )
    found = discover(derivations_dir, registry=registry)
    assert found == ["consumer", "provider"]


def test_discover_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(DerivationError, match="not found"):
        discover(tmp_path / "absent", registry=Registry())


def test_discover_surfaces_import_error(tmp_path: Path) -> None:
    derivations_dir = tmp_path / "derivations"
    derivations_dir.mkdir()
    (derivations_dir / "broken.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8"
    )
    with pytest.raises(DerivationError, match="failed to import"):
        discover(derivations_dir, registry=Registry())
