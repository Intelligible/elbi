"""A sync that replaces a declared source's rows reaches the derivations reading it.

A project-declared file source is synced into the warehouse at load. Re-syncing it over
the API (after the file changed underneath) must change what a derivation computes, and
its data version, without restarting the app: otherwise the derivation cache and
orchestration both keep serving the rows the app booted with.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

_TERMINAL = {"succeeded", "failed", "cancelled"}

_SALES_COUNT = """\
from __future__ import annotations

from elbi_core import Artifact, Context, Dataset, derivation, serve


@derivation(inputs={"sales": Dataset("sales")}, serve=serve.table())
def sales_count(ctx: Context) -> Artifact:
    \"\"\"How many sales rows there are.\"\"\"
    return Artifact.table([{"sales_rows": f"n={len(ctx.input('sales').rows)}"}])
"""


def _write_sales(path: Path, rows: int) -> None:
    ids = [f"c{i}" for i in range(rows)]
    amounts = [i * 10 for i in range(rows)]
    if path.suffix == ".parquet":
        pq.write_table(pa.table({"customer_id": ids, "amount": amounts}), path)
    else:
        lines = ["customer_id,amount", *(f"{c},{a}" for c, a in zip(ids, amounts))]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _project(root: Path, data_file: str) -> Path:
    (root / "landing").mkdir(parents=True)
    _write_sales(root / "landing" / data_file, 3)
    (root / "elbi.yaml").write_text(
        "project: t\n"
        "sources:\n"
        "  - name: sales\n"
        "    type: csv\n"
        f"    path: ./landing/{data_file}\n",
        encoding="utf-8",
    )
    (root / "derivations").mkdir()
    (root / "derivations" / "sales_count.py").write_text(_SALES_COUNT, encoding="utf-8")
    return root


def _rendered(http: TestClient) -> str:
    detail = http.get("/api/derivations/sales_count")
    assert detail.status_code == 200, detail.text
    rendered = detail.json()["rendered"]
    assert isinstance(rendered, str)
    return rendered


def _status(http: TestClient) -> str:
    status = http.get("/api/orchestration/status").json()
    return next(s["status"] for s in status if s["asset"] == "sales_count")


def _materialize_stale(http: TestClient) -> list[str]:
    run_id = http.post(
        "/api/orchestration/materialize", json={"selection": "stale"}
    ).json()["runId"]
    for _ in range(500):
        detail = http.get(f"/api/orchestration/runs/{run_id}").json()
        if detail["status"] in _TERMINAL:
            break
        time.sleep(0.02)
    else:  # pragma: no cover - only trips if a run wedges 'running'
        raise AssertionError(f"run {run_id} did not finish")
    assert detail["status"] == "succeeded", detail
    return [s["asset"] for s in detail["steps"] if s["state"] == "succeeded"]


@pytest.mark.parametrize("data_file", ["sales.parquet", "sales.csv"])
def test_api_sync_of_a_declared_source_reaches_its_derivations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_file: str
) -> None:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build

    project = _project(tmp_path / "proj", data_file)
    app = build(project, with_mcp=False)
    with TestClient(app) as http:
        assert "n=3" in _rendered(http)
        assert _materialize_stale(http) == ["sales_count"]
        assert _status(http) == "materialized"

        # The file is re-extracted with more rows and the source re-synced over the API.
        _write_sales(project / "landing" / data_file, 7)
        sources = http.get("/api/warehouse/sources").json()
        source_id = next(s["id"] for s in sources if s["name"] == "sales")
        synced = http.post(f"/api/warehouse/sources/{source_id}/sync")
        assert synced.status_code == 200, synced.text
        assert [o["rows"] for o in synced.json()["outcomes"]] == [7]

        # The derivation's data version moved, so orchestration sees it stale ...
        assert _status(http) == "stale"
        # ... and the derivation computes over the synced rows, not the boot-time ones.
        assert "n=7" in _rendered(http)
        # Materializing the stale asset settles it.
        assert _materialize_stale(http) == ["sales_count"]
        assert _status(http) == "materialized"
