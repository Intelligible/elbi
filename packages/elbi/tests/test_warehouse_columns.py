"""The persisted column catalog: what a column search reads instead of a Delta table.

Four things it has to be: recorded when a table is written, reconciled when its schema
changes, present for declared datasets as much as connector ones, and readable without
going near the table files.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("deltalake")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

import pyarrow as pa
from sqlalchemy.exc import IntegrityError

from elbi.db import (
    Store,
    WarehouseColumn,
    open_store,
)
from elbi.warehouse import storage
from elbi.warehouse.service import WarehouseService

_PEOPLE = "id,email,created_at\n1,a@b.c,2026-01-01\n2,c@d.e,2026-01-02\n"


@pytest.fixture
def warehouse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    yield tmp_path


def _service(tmp_path: Path) -> tuple[WarehouseService, Store]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    return WarehouseService(store), store


def _csv_source(
    service: WarehouseService, tmp_path: Path, name: str, body: str
) -> tuple[str, Path]:
    """A synced CSV source named ``name``; returns its id and the backing file."""
    path = tmp_path / f"{name}.csv"
    path.write_text(body, encoding="utf-8")
    source = service.create_source("csv", name, {"path": str(path)}, prefix="")
    service.sync_source(source.id)
    return source.id, path


def _named(columns: list[dict[str, Any]]) -> list[str]:
    return [c["name"] for c in columns]


# --- Recorded at sync time -------------------------------------------------------


def test_a_sync_records_every_column(warehouse: Path) -> None:
    """Given a synced source, every table's columns are persisted with name and type."""
    service, _store = _service(warehouse)
    _csv_source(service, warehouse, "people", _PEOPLE)

    columns = service.columns()

    assert _named(columns) == ["id", "email", "created_at"]
    assert [c["type"] for c in columns] == ["long", "string", "date"]
    assert [c["ordinal"] for c in columns] == [0, 1, 2]
    assert {c["table"] for c in columns} == {"people"}


def test_the_recorded_type_is_the_written_name_not_a_repr(warehouse: Path) -> None:
    """``string``, not the ``PrimitiveType("string")`` a Delta type stringifies to."""
    service, _store = _service(warehouse)
    _csv_source(service, warehouse, "people", _PEOPLE)

    assert all("(" not in c["type"] for c in service.columns())


def test_a_reschema_adds_and_removes_without_duplicating(warehouse: Path) -> None:
    """Given a schema change on re-sync, the columns follow it and never double up."""
    service, store = _service(warehouse)
    source_id, path = _csv_source(service, warehouse, "people", _PEOPLE)

    # `email` goes, `country` arrives.
    path.write_text("id,created_at,country\n1,2026-01-01,NZ\n", encoding="utf-8")
    service.sync_source(source_id)

    assert _named(service.columns()) == ["id", "created_at", "country"]
    # Asserted on the raw rows too: a duplicate would be invisible in the names above if
    # the read path happened to de-duplicate.
    rows = store.list_warehouse_columns(["people"])
    assert len(rows) == 3
    assert len({r.name for r in rows}) == 3


def test_a_declared_project_dataset_gets_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project-bound dataset's columns arrive through the same path as a connector's.

    Not a second mechanism: a declaration becomes an ordinary CSV source and syncs
    through ``sync_source`` like any other. This asserts that equivalence.
    """
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    db = tmp_path / "app.db"
    monkeypatch.setenv("DB_URI", f"sqlite:{db}")
    from elbi.serve import build

    project = tmp_path / "proj"
    (project / "fixtures").mkdir(parents=True)
    (project / "fixtures" / "sales.csv").write_text(
        "customer_id,amount\nc1,100\nc2,5\n", encoding="utf-8"
    )
    (project / "elbi.yaml").write_text(
        "project: t\n"
        "sources:\n"
        "  - name: sales\n"
        "    type: csv\n"
        "    path: ./fixtures/sales.csv\n",
        encoding="utf-8",
    )

    app = build(project, with_mcp=False)
    with TestClient(app):
        pass

    store = open_store(f"sqlite:{db}")
    recorded = store.list_warehouse_columns(["sales"])
    assert [c.name for c in recorded] == ["customer_id", "amount"]
    # The parent ticket's query class, answerable at last.
    assert any(c.name == "customer_id" for c in recorded)


# --- Read without touching the table ---------------------------------------------


def test_reading_columns_never_opens_the_table(warehouse: Path) -> None:
    """The columns outlive the table files, because they are not read from them.

    Had the read introspected, there would be nothing left to introspect.
    """
    service, _store = _service(warehouse)
    _csv_source(service, warehouse, "people", _PEOPLE)
    storage.drop_table("people")
    assert storage.table_location("people") is None

    # The schema rows say the table is synced; only the files are gone.
    assert _named(service.columns()) == ["id", "email", "created_at"]


def test_reading_columns_constructs_no_delta_table(
    warehouse: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And says so directly: constructing a DeltaTable during the read is a failure."""
    service, _store = _service(warehouse)
    _csv_source(service, warehouse, "people", _PEOPLE)

    import deltalake

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the column read path opened a Delta table")

    monkeypatch.setattr(deltalake, "DeltaTable", _forbidden)

    assert _named(service.columns()) == ["id", "email", "created_at"]


# --- The incremental trap ---------------------------------------------------------


class _BatchSource:
    """A connector that yields exactly the Arrow tables a test hands it."""

    def __init__(self, batches: list[pa.Table]) -> None:
        self._batches = batches

    def extract(self, inputs: object) -> Iterator[pa.Table]:
        yield from self._batches


def test_an_incremental_sync_keeps_a_column_its_batch_omits(warehouse: Path) -> None:
    """The column set is the table's, not the batch's.

    ``append`` pairs with ``schema_mode="merge"``, so a batch carrying nothing for a
    nullable column would otherwise delete it from the catalog while it stays perfectly
    selectable -- and search would quietly stop finding the table by it.
    """
    from elbi.warehouse import sync as sync_module

    full = pa.table({"id": [1], "email": ["a@b.c"], "created_at": ["2026-01-01"]})
    narrower = pa.table({"id": [2], "created_at": ["2026-01-02"]})

    first = sync_module.run_sync(
        _BatchSource([full]),  # type: ignore[arg-type]
        {},
        "people",
        table="people",
        sync_type="full_refresh",
    )
    assert first.columns is not None
    assert [c.name for c in first.columns] == ["id", "email", "created_at"]

    second = sync_module.run_sync(
        _BatchSource([narrower]),  # type: ignore[arg-type]
        {},
        "people",
        table="people",
        sync_type="incremental",
        incremental_field="id",
    )

    assert second.columns is not None
    assert [c.name for c in second.columns] == ["id", "email", "created_at"]


def test_a_column_comment_survives_into_the_catalog(warehouse: Path) -> None:
    """A described column keeps its description, rather than losing it on the way in.

    No connector describes its columns today, so without this the field would be an
    always-None one whose docstring claimed a seam that did not exist.
    """
    from elbi.warehouse import sync as sync_module

    described = pa.table(
        {"id": [1]},
        schema=pa.schema(
            [pa.field("id", pa.int64(), metadata={b"comment": b"the customer key"})]
        ),
    )

    result = sync_module.run_sync(
        _BatchSource([described]),  # type: ignore[arg-type]
        {},
        "described",
        table="described",
        sync_type="full_refresh",
    )

    assert result.columns is not None
    assert result.columns[0].description == "the customer key"


def test_a_sync_that_extracted_nothing_leaves_the_columns_alone(
    warehouse: Path,
) -> None:
    """No table to describe means no claim about its columns, not an empty one."""
    from elbi.warehouse import sync as sync_module

    result = sync_module.run_sync(
        _BatchSource([]),  # type: ignore[arg-type]
        {},
        "ghost",
        table="ghost",
        sync_type="full_refresh",
    )

    assert result.columns is None


def test_a_table_synced_before_this_gets_its_columns_filled_in(
    warehouse: Path,
) -> None:
    """A table already ``synced`` must not be left with an empty column catalog.

    Sync-time capture only helps a table that syncs again, so on an existing project the
    catalog would stay empty. Found by starting the app and seeing exactly that.
    """
    service, store = _service(warehouse)
    _csv_source(service, warehouse, "people", _PEOPLE)

    # The state of a table synced before columns were ever recorded.
    store.replace_warehouse_columns("people", "irrelevant", [])
    assert store.list_warehouse_columns(["people"]) == []

    filled = service.backfill_columns()

    assert filled == ["people"]
    assert _named(service.columns()) == ["id", "email", "created_at"]

    # Nothing left to do, so a second boot reads no transaction logs.
    assert service.backfill_columns() == []


# --- Lifecycle --------------------------------------------------------------------


def test_deleting_a_source_takes_its_columns(warehouse: Path) -> None:
    """Nothing else collects them, and the search index reads them directly."""
    service, store = _service(warehouse)
    source_id, _ = _csv_source(service, warehouse, "people", _PEOPLE)
    assert store.list_warehouse_columns(["people"])

    service.delete_source(source_id)

    assert store.list_warehouse_columns(["people"]) == []
    assert store.list_warehouse_columns() == []


def test_two_writers_cannot_leave_a_table_with_duplicate_columns(
    warehouse: Path,
) -> None:
    """The lock that serializes writers is one process's, and every replica backfills.

    Two of them can each read no columns for a table and each insert the whole set, and
    a search would then find the table twice per column. The unique constraint is what
    actually prevents that, so it is asserted directly rather than through the write
    path -- an in-process test cannot interleave two processes' transactions, but it can
    prove the guard they will hit.
    """
    service, store = _service(warehouse)
    _csv_source(service, warehouse, "people", _PEOPLE)
    recorded = store.list_warehouse_columns(["people"])

    with pytest.raises(IntegrityError), store._write() as session:
        session.add(
            WarehouseColumn(
                table="people", source_id="src", name="id", data_type="string"
            )
        )
        session.commit()

    assert store.list_warehouse_columns(["people"]) == recorded


def test_losing_that_race_keeps_the_winner_s_columns(warehouse: Path) -> None:
    """Losing must report, not raise: the caller is a sync, or a replica booting.

    ``serve`` backfills unguarded at startup, so raising here would take a replica down
    over another replica having already done its work. And the rows the loser deleted on
    its way in have to come back, or the winner's table is left empty by the loser.
    """
    service, store = _service(warehouse)
    _csv_source(service, warehouse, "people", _PEOPLE)
    recorded = store.list_warehouse_columns(["people"])

    # Stands in for the other writer's rows: whatever trips the constraint, the
    # transaction rolls back with the delete in it.
    wrote = store.replace_warehouse_columns(
        "people",
        "src",
        [
            WarehouseColumn(table="people", source_id="src", name="dup"),
            WarehouseColumn(table="people", source_id="src", name="dup"),
        ],
    )

    assert wrote is False
    assert store.list_warehouse_columns(["people"]) == recorded
