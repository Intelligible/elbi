"""A sync either lands whole or not at all, and a run that died does not stick.

A full refresh that overwrote the table with its first batch and appended the rest
would, failing on its second batch, leave a table holding only the first: it reads as
complete, and every later run that fails the same way keeps it that way. A run whose
process dies would leave its source ``syncing`` for good, which the scheduler skips.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("deltalake")
pytest.importorskip("pyarrow")

import pyarrow as pa

import elbi.db
from elbi.db import Store, open_store
from elbi.warehouse import service as service_module
from elbi.warehouse import storage
from elbi.warehouse.config import SourceConfig, SourceSchema
from elbi.warehouse.service import WarehouseService
from elbi.warehouse.sources.base import Source, SourceInputs

#: A page of the run, or the error it fails with when the run reaches it.
Page = pa.Table | Exception | Callable[[], None]


class PagedConnector(Source):
    """A connector that yields the pages a test sets, failing where one is an error."""

    pages: ClassVar[list[Page]] = []

    @property
    def source_type(self) -> str:
        return "paged"

    @property
    def config(self) -> SourceConfig:
        return SourceConfig(name="paged", label="Paged", category="Testing", fields=[])

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        return [SourceSchema(name="invoices", incremental_fields=["id"])]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        for page in PagedConnector.pages:
            if isinstance(page, Exception):
                raise page
            if isinstance(page, pa.Table):
                yield page
            else:
                page()


@pytest.fixture
def service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[WarehouseService, Store]]:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    monkeypatch.setattr(
        WarehouseService, "_connector", lambda self, t: PagedConnector()
    )
    PagedConnector.pages = []
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    yield WarehouseService(store), store


def _invoices(*ids: int) -> pa.Table:
    return pa.table({"id": list(ids), "amount": [i * 100 for i in ids]})


def _source(svc: WarehouseService, *, sync_type: str = "full_refresh") -> str:
    source = svc.create_source("paged", "stripe", {}, prefix="stripe")
    if sync_type == "incremental":
        (schema,) = svc.list_schemas(source.id)
        svc.update_schema(schema.id, sync_type="incremental", incremental_field="id")
    return source.id


def _sync(svc: WarehouseService, source_id: str, *pages: Page) -> None:
    PagedConnector.pages = list(pages)
    svc.sync_source(source_id)


def _rows() -> list[dict[str, Any]]:
    from deltalake import DeltaTable

    location = storage.table_location("stripe__invoices")
    assert location is not None
    rows = DeltaTable(location).to_pyarrow_table().to_pylist()
    return sorted(rows, key=lambda row: row["id"])


def _staging_leftovers(tmp_path: Path) -> list[Path]:
    staging = tmp_path / "warehouse" / "_staging"
    return list(staging.iterdir()) if staging.exists() else []


# --- A run lands whole or not at all ---------------------------------------------


def test_a_full_refresh_that_fails_part_way_keeps_the_previous_table(
    service: tuple[WarehouseService, Store],
) -> None:
    """The incident: page one overwrote, page two failed, and 100 of 233 rows stayed."""
    svc, store = service
    source_id = _source(svc)
    _sync(svc, source_id, _invoices(1, 2), _invoices(3))
    before = _rows()

    _sync(svc, source_id, _invoices(4), RuntimeError("Unsupported CAST on page 2"))

    assert _rows() == before
    assert len(before) == 3
    source = svc.get_source(source_id)
    assert source.status == "error"
    assert "Unsupported CAST" in (source.last_error or "")
    (schema,) = store.list_external_schemas(source_id)
    assert schema.status == "error"
    assert schema.row_count == 3, "the recorded count still describes the table"


def test_a_full_refresh_that_succeeds_replaces_the_table(
    service: tuple[WarehouseService, Store],
) -> None:
    svc, store = service
    source_id = _source(svc)
    _sync(svc, source_id, _invoices(1, 2), _invoices(3))

    _sync(svc, source_id, _invoices(7), _invoices(8, 9))

    assert [row["id"] for row in _rows()] == [7, 8, 9]
    assert svc.get_source(source_id).status == "idle"
    (schema,) = store.list_external_schemas(source_id)
    assert schema.row_count == 3


def test_a_failed_first_full_refresh_creates_no_table(
    service: tuple[WarehouseService, Store],
) -> None:
    """With nothing before it, a failed run must not leave a partial table either."""
    svc, _store = service
    source_id = _source(svc)

    _sync(svc, source_id, _invoices(1), RuntimeError("boom"))

    assert storage.table_location("stripe__invoices") is None


def test_staging_is_cleaned_up_whether_the_run_fails_or_succeeds(
    service: tuple[WarehouseService, Store], tmp_path: Path
) -> None:
    svc, _store = service
    source_id = _source(svc)

    _sync(svc, source_id, _invoices(1), RuntimeError("boom"))
    assert _staging_leftovers(tmp_path) == []

    _sync(svc, source_id, _invoices(1), _invoices(2))
    assert _staging_leftovers(tmp_path) == []


def test_an_incremental_run_that_fails_part_way_appends_nothing(
    service: tuple[WarehouseService, Store],
) -> None:
    """Its cursor stays put, so the retry re-reads the same rows: none may be kept.

    Appended batch by batch, the batches before the failure would stay in the table
    while the cursor did not move, and the retry would append them a second time.
    """
    svc, store = service
    source_id = _source(svc, sync_type="incremental")
    _sync(svc, source_id, _invoices(1, 2))

    _sync(svc, source_id, _invoices(3), RuntimeError("timeout"))
    assert [row["id"] for row in _rows()] == [1, 2]
    (schema,) = store.list_external_schemas(source_id)
    assert schema.cursor == "2"

    _sync(svc, source_id, _invoices(3), _invoices(4))
    assert [row["id"] for row in _rows()] == [1, 2, 3, 4]


# --- A run that died does not stick ----------------------------------------------


def _left_syncing(svc: WarehouseService, store: Store, *, frequency: str) -> str:
    """A source whose run started now and whose process then died."""
    source_id = _source(svc)
    svc.set_sync_frequency(source_id, frequency)
    store.update_external_source(source_id, status="syncing")
    return source_id


def test_a_source_left_syncing_by_a_dead_run_is_synced_again(
    service: tuple[WarehouseService, Store],
) -> None:
    svc, store = service
    source_id = _left_syncing(svc, store, frequency="day")
    PagedConnector.pages = [_invoices(1)]

    synced = svc.sync_due(datetime.now(timezone.utc) + timedelta(days=2))

    assert synced == [source_id]
    assert svc.get_source(source_id).status == "idle"
    assert [row["id"] for row in _rows()] == [1]


def test_an_interrupted_run_reads_as_failed_with_a_reason(
    service: tuple[WarehouseService, Store],
) -> None:
    """A manual source is never scheduled, so the recovery itself must be visible."""
    svc, store = service
    source_id = _left_syncing(svc, store, frequency="manual")

    recovered = svc.recover_interrupted_syncs(
        datetime.now(timezone.utc) + timedelta(hours=1)
    )

    assert recovered == [source_id]
    source = svc.get_source(source_id)
    assert source.status == "error"
    assert "interrupted" in (source.last_error or "").lower()


def test_a_run_that_is_still_going_is_left_alone(
    service: tuple[WarehouseService, Store],
) -> None:
    svc, store = service
    source_id = _left_syncing(svc, store, frequency="day")

    synced = svc.sync_due(datetime.now(timezone.utc) + timedelta(minutes=5))

    assert synced == []
    assert svc.get_source(source_id).status == "syncing"


def test_a_long_run_keeps_itself_alive(
    service: tuple[WarehouseService, Store], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run older than the stale threshold is not taken for dead while it works."""
    svc, _store = service
    source_id = _source(svc)
    clock = [datetime.now(timezone.utc)]
    monkeypatch.setattr(elbi.db, "_now", lambda: clock[0])
    monkeypatch.setattr(service_module, "HEARTBEAT_SECONDS", 0.0)
    seen: list[str] = []

    def an_hour_passes() -> None:
        clock[0] += timedelta(hours=1)

    def check_from_another_tick() -> None:
        svc.recover_interrupted_syncs(clock[0] + timedelta(minutes=5))
        seen.append(svc.get_source(source_id).status)

    _sync(
        svc,
        source_id,
        an_hour_passes,
        _invoices(1),
        check_from_another_tick,
        _invoices(2),
    )

    assert seen == ["syncing"]
    assert svc.get_source(source_id).status == "idle"
    assert [row["id"] for row in _rows()] == [1, 2]
