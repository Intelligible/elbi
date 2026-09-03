"""Shared fixtures for the SDK test suite."""

from __future__ import annotations

import pytest

from elbi_core import Registry


@pytest.fixture
def registry() -> Registry:
    """A fresh, isolated registry (never the global default)."""
    return Registry()
