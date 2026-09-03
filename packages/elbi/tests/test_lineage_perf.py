"""The catalog's latency baseline, so later search work has a number to hold itself to.

Absolute timings are recorded, not asserted -- a CI runner is noisier than the effect,
so a millisecond ceiling would flake. The assertions are the two properties that broke:
one graph build per request, and roughly linear growth rather than quadratic.

Measured on an Apple M-series laptop: ~4 ms for 175 records, ~19 ms for 1525. The same
175-record catalog took ~709 ms across 176 builds before this ticket.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from elbi.db import Derivation as DerivationRow
from elbi.db import MetricMonitor, Store, open_store
from elbi.lineage import LineageService
from elbi_core import Artifact, Context, Dataset, Registry, derivation, serve
from elbi_core.registry import use_registry

#: Derivations per size; the growth ratio is taken between them.
_SIZES = (100, 1000)

#: Source tables to spread the derivations over, so no one dataset is a huge fan-out hub
#: (which is its own unrelated cost).
_SOURCES = 20

#: Timed repeats per size. The minimum is the run least disturbed by other work.
_REPEATS = 5

_DASHBOARD_SPEC = {
    "specVersion": "1.0",
    "kind": "Dashboard",
    "name": "d",
    "pages": [
        {
            "name": "main",
            "widgets": [
                {
                    "id": "t",
                    "type": "table",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "bind": {"derivation": "deriv_0"},
                }
            ],
        }
    ],
}


def _dataset_names() -> list[str]:
    return [f"source_{i}" for i in range(_SOURCES)]


def _registry(n: int) -> Registry:
    """``n`` derivations, each consuming one of ``_SOURCES`` datasets."""
    registry = Registry()
    with use_registry(registry):
        for i in range(n):

            def body(ctx: Context) -> Artifact:
                return Artifact.table([{"v": 1}])

            body.__name__ = f"deriv_{i}"
            derivation(
                inputs={"src": Dataset(f"source_{i % _SOURCES}")}, serve=serve.table()
            )(body)
    return registry


def _populate(store: Store, n: int) -> None:
    """The stored half of a project that size, so every ``_add_*`` branch contributes.

    The parts that scale with the project scale here too, so the measurement covers the
    whole build and not just the registry walk.
    """
    for i in range(n):
        store.save_derivation(
            DerivationRow(
                name=f"deriv_{i}", verdict="sound", origin="repo", source="..."
            )
        )
    for i in range(n // 4):
        store.upsert_metric(name=f"metric_{i}", manifest_json="{}", source=f"deriv_{i}")
        store.save_metric_monitor(
            MetricMonitor(name=f"mon_{i}", target_kind="metric", target=f"metric_{i}")
        )
    for i in range(5):
        store.create_dashboard(
            name=f"dash_{i}", title=f"D{i}", spec_json=json.dumps(_DASHBOARD_SPEC)
        )


def _service(tmp_path: Path, n: int) -> tuple[LineageService, Callable[[], int]]:
    """A service over a project of ``n`` derivations, and a count of graphs built."""
    store = open_store(f"sqlite:{tmp_path / f'bench_{n}.db'}")
    registry = _registry(n)
    _populate(store, n)
    builds = 0

    def counting_provider() -> Registry:
        nonlocal builds
        builds += 1
        return registry

    service = LineageService(
        store=store,
        registry_provider=counting_provider,
        dataset_names=_dataset_names,
    )
    return service, lambda: builds


def _measure(tmp_path: Path, n: int) -> dict[str, Any]:
    service, builds = _service(tmp_path, n)
    service.catalog()  # warm the connection pool and any first-call lazies
    before = builds()

    timings = []
    for _ in range(_REPEATS):
        start = time.perf_counter()
        records = service.catalog()
        timings.append(time.perf_counter() - start)

    seconds = min(timings)
    return {
        "derivations": n,
        "records": len(records),
        "seconds": seconds,
        "per_record_us": seconds / len(records) * 1e6,
        "builds_per_catalog": (builds() - before) / _REPEATS,
    }


@pytest.mark.slow
def test_catalog_latency_baseline(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    """Record the latency, and hold the two properties that make it hold up."""
    results = [_measure(tmp_path, n) for n in _SIZES]

    for result in results:
        record_property(f"catalog_{result['derivations']}", result)
        # The recorded baseline, visible under `pytest -s`.
        print(
            f"catalog: {result['derivations']:5d} derivations "
            f"-> {result['records']:5d} records "
            f"in {result['seconds'] * 1000:8.3f} ms "
            f"({result['per_record_us']:6.2f} us/record)"
        )

    out = os.environ.get("LINEAGE_BENCH_OUT")
    if out:
        Path(out).write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "measurements": results,
                },
                indent=2,
            )
        )

    small, large = results
    # The regression that matters: it was one build per row, and nothing noticed because
    # the answer stayed correct.
    assert small["builds_per_catalog"] == 1
    assert large["builds_per_catalog"] == 1

    # Over this ~8.7x span in records, linear is ~8.7x and quadratic ~76x. The bound
    # sits between them, with room for a noisy runner.
    growth = large["records"] / small["records"]
    assert large["seconds"] < small["seconds"] * growth * 3, (
        f"catalog latency grew {large['seconds'] / small['seconds']:.1f}x over a "
        f"{growth:.1f}x corpus -- superlinear, so something is rebuilding per row"
    )
