"""Shared test fixtures for the app package."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from prometheus_client import REGISTRY


@pytest.fixture(autouse=True)
def _reset_prometheus_registry() -> Iterator[None]:
    """Clear the global Prometheus registry between tests.

    ``create_app`` registers metrics on ``prometheus_client``'s default registry; the
    suite builds the app many times, so without this reset the second build raises
    "Duplicated timeseries in CollectorRegistry". Clearing before each test keeps them
    independent.
    """
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    yield
