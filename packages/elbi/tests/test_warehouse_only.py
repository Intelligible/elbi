"""The warehouse is the only source of data: a project's declared sources are ensured
and synced at load, and every surface reads from the warehouse: with no path back to the
original file at run time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

_SALES = "customer_id,amount\nc1,100\nc2,5\nc3,40\nc4,250\n"


def _project(root: Path) -> Path:
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(_SALES, encoding="utf-8")
    (root / "elbi.yaml").write_text(
        "project: t\n"
        "sources:\n"
        "  - name: sales\n"
        "    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )
    return root


def test_declared_source_is_served_from_the_warehouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build
    from elbi.warehouse import storage

    project = _project(tmp_path / "proj")
    app = build(project, with_mcp=False)
    with TestClient(app) as http:
        # The declaration became a warehouse Delta table named after the source.
        assert storage.table_location("sales") is not None

        datasets = {d["name"]: d for d in http.get("/api/datasets").json()}
        assert "sales" in datasets
        assert datasets["sales"]["origin"] == "warehouse"
        assert datasets["sales"]["rows"] == 4

        query = http.post(
            "/api/explore/query",
            json={
                "sql": "SELECT customer_id FROM sales ORDER BY amount DESC LIMIT 1",
                "source_id": "warehouse",
            },
        )
        assert query.status_code == 200, query.text
        assert query.json()["rows"] == [{"customer_id": "c4"}]

        # NO ALTERNATE PATH: remove the source file entirely; every surface still works,
        # because the data lives in the warehouse, not the file.
        (project / "fixtures" / "sales.csv").unlink()
        assert not (project / "fixtures" / "sales.csv").exists()

        again = {d["name"]: d for d in http.get("/api/datasets").json()}
        assert again["sales"]["rows"] == 4  # still served from the warehouse

        recount = http.post(
            "/api/explore/query",
            json={"sql": "SELECT count(*) AS n FROM sales", "source_id": "warehouse"},
        )
        assert recount.json()["rows"] == [{"n": 4}]


def test_promote_warehouse_query_authors_a_certified_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warehouse query promotes to a certified derivation.

    ``warehouse`` is a synthetic source id, so promotion must recognize it rather than
    route it to external-source resolution ("data source 'warehouse' not found").
    """
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build

    project = _project(tmp_path / "proj")
    app = build(project, with_mcp=False)
    with TestClient(app) as http:
        promoted = http.post(
            "/api/explore/promote",
            json={
                "name": "top_customer",
                "sql": "SELECT customer_id FROM sales ORDER BY amount DESC LIMIT 1",
                "source_id": "warehouse",
            },
        )
        assert promoted.status_code == 200, promoted.text
        body = promoted.json()
        assert body["ok"] is True, body
        assert body["certified"] is True
        # Certified against the warehouse data, not a fabricated answer.
        assert "c4" in body["rendered"]

        # It lands beside every other certified artifact in the derivations catalog.
        names = {d["name"] for d in http.get("/api/derivations").json()}
        assert "top_customer" in names


def test_promote_warehouse_query_failure_registers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing warehouse query reports the error and leaves nothing registered.

    ``WarehouseError`` is a plain ``Exception``; unwrapped it escapes the promote guard,
    500s, and leaves a broken 'certified' derivation behind.
    """
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build

    project = _project(tmp_path / "proj")
    app = build(project, with_mcp=False)
    with TestClient(app) as http:
        promoted = http.post(
            "/api/explore/promote",
            json={
                "name": "broken_view",
                "sql": "SELECT nonexistent_col FROM sales",
                "source_id": "warehouse",
            },
        )
        assert promoted.status_code == 200, promoted.text
        assert promoted.json()["ok"] is False
        # Nothing half-registered: the failed promotion is absent from the catalog.
        names = {d["name"] for d in http.get("/api/derivations").json()}
        assert "broken_view" not in names


def test_promote_warehouse_query_over_row_cap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-cap warehouse query is rejected, not silently clipped.

    ``warehouse_service.query`` fetches one past the cap and flags truncation;
    certifying only the first N rows would stamp a partial (and, without ORDER BY,
    arbitrary) answer as verified, so promotion must fail loudly instead.
    """
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi import serve as serve_mod
    from elbi.serve import build

    # sales has 4 rows; cap at 2 so the query exceeds the promote limit.
    monkeypatch.setattr(serve_mod, "_EXTERNAL_PROMOTE_MAX_ROWS", 2)
    project = _project(tmp_path / "proj")
    app = build(project, with_mcp=False)
    with TestClient(app) as http:
        promoted = http.post(
            "/api/explore/promote",
            json={
                "name": "too_many",
                "sql": "SELECT customer_id FROM sales",
                "source_id": "warehouse",
            },
        )
        assert promoted.status_code == 200, promoted.text
        body = promoted.json()
        assert body["ok"] is False
        assert "rows" in body["error"].lower()
        names = {d["name"] for d in http.get("/api/derivations").json()}
        assert "too_many" not in names


def _project_with_dev_binding_only(root: Path) -> Path:
    """A project with no `sources:` at all -- only a dataset + a dev.yaml file
    binding, the shape `elbi init` scaffolds.
    """
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(_SALES, encoding="utf-8")
    (root / "elbi.yaml").write_text(
        "project: t\ndatasets:\n  - sales\n", encoding="utf-8"
    )
    (root / "elbi.dev.yaml").write_text(
        "data:\n  sales: ./fixtures/sales.csv\n", encoding="utf-8"
    )
    return root


def test_dev_binding_dataset_is_derived_into_the_warehouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset with no `sources:` entry, only a dev.yaml file binding, still
    lands in the warehouse -- a project should not have to declare the same path
    twice to make `elbi dev` and `elbi serve` both work.
    """
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build
    from elbi.warehouse import storage

    project = _project_with_dev_binding_only(tmp_path / "proj")
    app = build(project, with_mcp=False)
    with TestClient(app) as http:
        assert storage.table_location("sales") is not None
        datasets = {d["name"]: d for d in http.get("/api/datasets").json()}
        assert datasets["sales"]["rows"] == 4


def test_explicit_source_wins_over_a_dev_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit `sources:` entry is never overridden by a same-named dev.yaml
    binding, even one pointing at a different file.
    """
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.serve import build
    from elbi.warehouse import storage

    root = tmp_path / "proj"
    (root / "fixtures").mkdir(parents=True)
    (root / "fixtures" / "sales.csv").write_text(_SALES, encoding="utf-8")
    (root / "fixtures" / "sales_dev_only.csv").write_text(
        "customer_id,amount\nc9,999\n", encoding="utf-8"
    )
    (root / "elbi.yaml").write_text(
        "project: t\n"
        "sources:\n"
        "  - name: sales\n"
        "    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )
    (root / "elbi.dev.yaml").write_text(
        "data:\n  sales: ./fixtures/sales_dev_only.csv\n", encoding="utf-8"
    )
    app = build(root, with_mcp=False)
    with TestClient(app) as http:
        assert storage.table_location("sales") is not None
        datasets = {d["name"]: d for d in http.get("/api/datasets").json()}
        # 4 rows (the declared source), not 1 (the dev-only binding it could have
        # been mistaken for).
        assert datasets["sales"]["rows"] == 4
