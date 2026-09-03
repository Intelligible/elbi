"""Integration tests for the data-warehouse API and its Explore integration.

Drive the HTTP surface with a TestClient over a real store and a local Delta Lake
warehouse (``STORAGE_URI`` under tmp_path, zero cloud): read the connector catalog,
create a CSV source and sync it, list the synced tables, query them through the Explore
workbench (DuckDB over Delta), run an incremental SQLite sync, toggle a schema, and
delete a source. Also assert the surface degrades to a 503 install hint without the
``warehouse`` extra wired.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.db import open_store
from elbi.explore import ExploreService
from elbi.warehouse.service import WarehouseError, WarehouseService

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")


class _FakeClient:
    """A minimal LLMClient stub; the warehouse and explore paths never call it."""

    def step(self, transcript: Any, tools: Any) -> Any:
        return SimpleNamespace(text="")


@pytest.fixture
def people_csv(tmp_path: Path) -> str:
    path = tmp_path / "people.csv"
    path.write_text(
        "id,name,city,score\n1,Ada,London,9.5\n2,Alan,Manchester,8.0\n3,Grace,NYC,9.9\n"
    )
    return str(path)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    warehouse = WarehouseService(store)
    explore = ExploreService(
        store=store,
        dataset_names=list,
        resolve_source=lambda name: [],
        read_schema=lambda name: ([], 0),
        warehouse=warehouse,
    )
    app = create_app(
        load_datasets=lambda: {},
        client=_FakeClient(),
        store=store,
        warehouse_service=warehouse,
        explore_service=explore,
    )
    with TestClient(app) as http:
        yield http


def _create(client: TestClient, name: str, config: dict[str, Any], kind: str = "csv"):
    response = client.post(
        "/api/warehouse/sources",
        json={"source_type": kind, "name": name, "config": config},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_catalog_lists_every_connector(client: TestClient) -> None:
    catalog = client.get("/api/warehouse/catalog").json()
    names = {s["name"] for s in catalog["sources"]}
    # Databases (per-dialect), files, and the SaaS connectors.
    assert {
        "csv",
        "postgres",
        "mysql",
        "mssql",
        "sqlite",
        "snowflake",
        "bigquery",
        "stripe",
        "hubspot",
        "salesforce",
        "shopify",
        "zendesk",
        "custom",
        # SaaS REST connectors (dlt-backed)
        "chargebee",
        "mailchimp",
        "klaviyo",
        "sendgrid",
        "braze",
        "pipedrive",
        "front",
        "vercel",
        "airtable",
        "mixpanel",
    } <= names
    assert "Databases" in catalog["categories"]
    # Nothing is a "coming soon" placeholder anymore: every tile is a real connector.
    assert all(not s["comingSoon"] for s in catalog["sources"])
    # Each connector carries its own form schema, so the wizard renders generically.
    pg = next(s for s in catalog["sources"] if s["name"] == "postgres")
    assert [f["name"] for f in pg["fields"]][:2] == ["host", "port"]


def test_saas_connector_field_schemas(client: TestClient) -> None:
    sources = client.get("/api/warehouse/catalog").json()["sources"]
    catalog = {s["name"]: [f["name"] for f in s["fields"]] for s in sources}
    # The field names each vendor's own API and docs use.
    assert catalog["zendesk"] == ["subdomain", "email_address", "api_key"]
    assert catalog["shopify"] == [
        "shopify_store_id",
        "shopify_client_id",
        "shopify_client_secret",
    ]
    assert "account_id" in catalog["snowflake"]


def test_description_and_table_prefix(client: TestClient, people_csv: str) -> None:
    source = _create(
        client,
        "people",
        {"path": people_csv},
    )
    # Default prefix is the source type.
    assert source["prefix"] == "csv"
    assert source["schemas"][0]["table"] == "csv__people"

    # A custom prefix renames the landed tables; description is stored.
    response = client.post(
        "/api/warehouse/sources",
        json={
            "source_type": "csv",
            "name": "billing",
            "config": {"path": people_csv},
            "description": "Finance export",
            "prefix": "finance",
        },
    )
    assert response.status_code == 200
    detail = response.json()
    assert detail["description"] == "Finance export"
    assert detail["schemas"][0]["table"] == "finance__people"

    # An invalid prefix is rejected.
    bad = client.post(
        "/api/warehouse/sources",
        json={
            "source_type": "csv",
            "name": "bad",
            "config": {"path": people_csv},
            "prefix": "1-bad",
        },
    )
    assert bad.status_code == 400


def test_notify_me_records_interest(client: TestClient) -> None:
    assert client.post(
        "/api/warehouse/interest", json={"source_type": "snowflake"}
    ).json() == {"ok": True}


def test_create_discovers_schema_then_sync_lands_rows(
    client: TestClient, people_csv: str
) -> None:
    source = _create(client, "people", {"path": people_csv})
    assert source["schemaCount"] == 1
    schema = source["schemas"][0]
    assert schema["table"] == "csv__people"
    assert schema["shouldSync"] is True

    result = client.post(f"/api/warehouse/sources/{source['id']}/sync").json()
    assert result["outcomes"] == [
        {"table": "csv__people", "rows": 3, "ok": True, "error": None}
    ]
    assert result["source"]["status"] == "idle"

    tables = client.get("/api/warehouse/tables").json()
    assert tables == [
        {
            "table": "csv__people",
            "source": "people",
            "sourceType": "csv",
            "rows": 3,
            "location": tables[0]["location"],
        }
    ]
    assert tables[0]["location"]


def test_upload_file_stores_it_and_drives_a_source(client: TestClient) -> None:
    # Uploading a local CSV stores it in the warehouse and returns a path; that path is
    # exactly what the CSV source's `path` takes, so an uploaded file syncs like any
    # other. The filename is sanitized and the file lands under the uploads prefix.
    content = b"id,name\n1,Ada\n2,Grace\n"
    up = client.post(
        "/api/warehouse/uploads",
        files={"file": ("people upload.csv", content, "text/csv")},
    )
    assert up.status_code == 200, up.text
    path = up.json()["path"]
    assert path.endswith("people_upload.csv") and "_uploads" in path

    source = _create(client, "uploaded", {"path": path})
    result = client.post(f"/api/warehouse/sources/{source['id']}/sync").json()
    assert result["outcomes"][0]["rows"] == 2 and result["outcomes"][0]["ok"]


def test_upload_rejects_mislabeled_and_unsupported_files(client: TestClient) -> None:
    # Validation is by content and by a whitelist, not by the client's say-so: a file
    # that is not really Parquet is rejected (400) rather than stored to fail at sync,
    # and an unsupported extension never gets written.
    bad = client.post(
        "/api/warehouse/uploads",
        files={"file": ("fake.parquet", b"not parquet", "application/octet-stream")},
    )
    assert bad.status_code == 400 and "PARQUET" in bad.json()["detail"]

    exe = client.post(
        "/api/warehouse/uploads",
        files={"file": ("x.exe", b"MZ\x00", "application/octet-stream")},
    )
    assert exe.status_code == 400 and "unsupported file type" in exe.json()["detail"]

    empty = client.post(
        "/api/warehouse/uploads",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert empty.status_code == 400


def test_create_source_with_empty_prefix_lands_unprefixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, people_csv: str
) -> None:
    # A project-declared source uses prefix="" so its Delta table takes the source's
    # own name (no connector-typed prefix); the table reads back by that plain name.
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    from elbi.warehouse import storage

    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    warehouse = WarehouseService(store)
    source = warehouse.create_source(
        "csv", "sales", {"path": people_csv, "table_name": "sales"}, prefix=""
    )
    warehouse.sync_source(source.id)
    assert {t["table"] for t in warehouse.tables()} == {"sales"}  # no csv__ prefix
    assert storage.table_location("sales") is not None
    _cols, rows, _trunc = warehouse.read_any("sales", max_rows=100)
    assert len(rows) == 3


def test_synced_table_is_queryable_through_explore(
    client: TestClient, people_csv: str
) -> None:
    source = _create(client, "people", {"path": people_csv})
    client.post(f"/api/warehouse/sources/{source['id']}/sync")

    # The warehouse shows up as an Explore source, with a typed catalog.
    sources = client.get("/api/explore/sources").json()
    assert {"id": "warehouse", "name": "Data warehouse", "kind": "warehouse"} in sources
    catalog = client.get("/api/explore/catalog?source_id=warehouse").json()
    table = next(t for t in catalog["tables"] if t["name"] == "csv__people")
    assert [c["name"] for c in table["columns"]] == ["id", "name", "city", "score"]

    # And its rows are queryable via DuckDB over Delta.
    query = client.post(
        "/api/explore/query",
        json={
            "source_id": "warehouse",
            "sql": "SELECT city, score FROM csv__people ORDER BY score DESC LIMIT 1",
        },
    ).json()
    assert query["rows"] == [{"city": "NYC", "score": 9.9}]


def test_read_table_makes_a_synced_table_available_as_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, people_csv: str
) -> None:
    # read_table is how a connected source's data becomes first-class everywhere
    # (notebook kernels, model frames, the chat agent) with no promotion: read a table
    # by name, with no SQL-string interpolation of the identifier.
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    warehouse = WarehouseService(store)
    source = warehouse.create_source("csv", "people", {"path": people_csv})
    warehouse.sync_source(source.id)

    columns, rows, truncated = warehouse.read_table("csv__people", max_rows=100)
    assert columns == ["id", "name", "city", "score"]
    assert len(rows) == 3
    assert {r["city"] for r in rows} == {"London", "Manchester", "NYC"}
    assert truncated is False

    # The cap is honoured and flags a clipped read.
    _cols, capped, clipped = warehouse.read_table("csv__people", max_rows=2)
    assert len(capped) == 2
    assert clipped is True

    # An unknown table is an error, not a silent empty result.
    with pytest.raises(WarehouseError):
        warehouse.read_table("does_not_exist", max_rows=10)


def test_full_refresh_resync_does_not_duplicate(
    client: TestClient, people_csv: str
) -> None:
    source = _create(client, "people", {"path": people_csv})
    client.post(f"/api/warehouse/sources/{source['id']}/sync")
    client.post(f"/api/warehouse/sources/{source['id']}/sync")
    tables = client.get("/api/warehouse/tables").json()
    assert tables[0]["rows"] == 3  # overwrite, not append


def test_incremental_sync_appends_past_cursor(
    client: TestClient, tmp_path: Path
) -> None:
    db = tmp_path / "source.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany(
            "INSERT INTO events (id, name) VALUES (?, ?)",
            [(1, "a"), (2, "b"), (3, "c")],
        )
        conn.commit()

        source = _create(client, "evt", {"database": str(db)}, "sqlite")
        schema = next(s for s in source["schemas"] if s["name"] == "events")
        assert "id" in schema["incrementalFields"]

        patched = client.patch(
            f"/api/warehouse/schemas/{schema['id']}",
            json={"sync_type": "incremental", "incremental_field": "id"},
        ).json()
        assert patched["syncType"] == "incremental"

        first = client.post(f"/api/warehouse/sources/{source['id']}/sync").json()
        assert first["outcomes"][0]["rows"] == 3

        conn.executemany(
            "INSERT INTO events (id, name) VALUES (?, ?)", [(4, "d"), (5, "e")]
        )
        conn.commit()
    second = client.post(f"/api/warehouse/sources/{source['id']}/sync").json()
    assert second["outcomes"][0]["rows"] == 2  # only the two new rows

    total = client.post(
        "/api/explore/query",
        json={
            "source_id": "warehouse",
            "sql": "SELECT count(*) AS n FROM sqlite__events",
        },
    ).json()
    assert total["rows"] == [{"n": 5}]


def test_toggle_schema_off_excludes_it_from_sync(
    client: TestClient, people_csv: str
) -> None:
    source = _create(client, "people", {"path": people_csv})
    schema = source["schemas"][0]
    off = client.patch(
        f"/api/warehouse/schemas/{schema['id']}", json={"should_sync": False}
    ).json()
    assert off["shouldSync"] is False
    result = client.post(f"/api/warehouse/sources/{source['id']}/sync").json()
    assert result["outcomes"] == []  # nothing enabled to sync


def test_delete_source_removes_it_and_its_table(
    client: TestClient, people_csv: str
) -> None:
    source = _create(client, "people", {"path": people_csv})
    client.post(f"/api/warehouse/sources/{source['id']}/sync")
    deleted = client.delete(f"/api/warehouse/sources/{source['id']}").json()
    assert deleted == {"ok": True}
    assert client.get("/api/warehouse/sources").json() == []
    assert client.get("/api/warehouse/tables").json() == []
    assert client.get(f"/api/warehouse/sources/{source['id']}").status_code == 404


def test_bad_connection_is_a_400(client: TestClient, tmp_path: Path) -> None:
    # A networked engine with no host fails validation before anything is persisted.
    response = client.post(
        "/api/warehouse/sources",
        json={
            "source_type": "postgres",
            "name": "bad",
            "config": {"database": "analytics"},
        },
    )
    assert response.status_code == 400
    assert "host" in response.json()["detail"]
    assert client.get("/api/warehouse/sources").json() == []


def test_create_defaults_to_daily_and_can_change_cadence(
    client: TestClient, people_csv: str
) -> None:
    source = _create(client, "people", {"path": people_csv})
    assert source["syncFrequency"] == "day"
    patched = client.patch(
        f"/api/warehouse/sources/{source['id']}", json={"sync_frequency": "6hour"}
    )
    assert patched.status_code == 200
    assert patched.json()["syncFrequency"] == "6hour"
    # An unknown cadence is rejected.
    bad = client.patch(
        f"/api/warehouse/sources/{source['id']}", json={"sync_frequency": "yearly"}
    )
    assert bad.status_code == 400


def test_scheduler_tick_syncs_a_due_source(client: TestClient, people_csv: str) -> None:
    # A never-synced source on any auto cadence is immediately due for its first sync.
    _create(client, "people", {"path": people_csv})
    client.app.state.warehouse_sync_tick()
    tables = client.get("/api/warehouse/tables").json()
    assert tables and tables[0]["table"] == "csv__people"
    assert tables[0]["rows"] == 3


def test_scheduler_tick_skips_manual_sources(
    client: TestClient, people_csv: str
) -> None:
    source = _create(client, "people", {"path": people_csv})
    client.patch(
        f"/api/warehouse/sources/{source['id']}", json={"sync_frequency": "manual"}
    )
    client.app.state.warehouse_sync_tick()
    # manual = never auto-synced
    assert client.get("/api/warehouse/tables").json() == []


def test_surface_degrades_without_the_extra(tmp_path: Path) -> None:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    app = create_app(
        load_datasets=lambda: {},
        client=_FakeClient(),
        store=store,
        warehouse_service=None,
    )
    with TestClient(app) as http:
        response = http.get("/api/warehouse/catalog")
    assert response.status_code == 503
    assert "warehouse" in response.json()["detail"]
