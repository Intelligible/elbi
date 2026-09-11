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


@pytest.fixture(autouse=True)
def _no_spa_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep booting a server out of npm.

    ``web/dist`` is not checked in, so the first test to call ``serve.build`` runs
    ``npm ci`` and ``npm run build``. Under ``--dist=loadfile`` the files that boot a
    server land on different workers, which then race on the same build directory and
    one of them hits the timeout.
    """
    from elbi import serve

    monkeypatch.setattr(serve, "_ensure_spa_built", lambda *_: None)
