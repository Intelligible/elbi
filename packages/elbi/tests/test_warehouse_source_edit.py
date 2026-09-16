"""Editing a warehouse source in place, instead of deleting and rebuilding it.

Without this, correcting a typo in a manifest costs every secret the source held, and
destroys the warehouse tables it produced on the way out.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("deltalake")
pytest.importorskip("pyarrow")

from elbi.db import Store, open_store
from elbi.warehouse.config import SourceSchema
from elbi.warehouse.service import WarehouseError, WarehouseService
from elbi.warehouse.sources.base import Source


@pytest.fixture(autouse=True)
def warehouse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("STORAGE_URI", f"file://{tmp_path / 'warehouse'}")
    monkeypatch.setenv("APP_SECRET_KEY", "test-key-for-encrypting-configs")
    yield


class FakeConnector(Source):
    """A connector whose resources and validity are set by the test."""

    resources: ClassVar[tuple[str, ...]] = ("alpha",)
    accept: ClassVar[bool] = True
    seen: ClassVar[list[dict]] = []

    @property
    def source_type(self) -> str:
        return "fake"

    @property
    def config(self):
        from elbi.warehouse.config import SourceConfig, SourceField

        return SourceConfig(
            name="fake",
            label="Fake",
            category="Testing",
            fields=[
                SourceField(name="query", label="Query"),
                SourceField(name="token", label="Token", type="password"),
            ],
        )

    def validate(self, config):
        FakeConnector.seen.append(dict(config))
        return (True, []) if FakeConnector.accept else (False, ["nope"])

    def schemas(self, config):
        return [SourceSchema(name=name) for name in FakeConnector.resources]

    def extract(self, inputs):
        return iter(())


@pytest.fixture
def service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[WarehouseService, Store]:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    svc = WarehouseService(store)
    FakeConnector.resources = ("alpha",)
    FakeConnector.accept = True
    FakeConnector.seen = []
    monkeypatch.setattr(WarehouseService, "_connector", lambda self, t: FakeConnector())
    return svc, store


def _create(svc: WarehouseService) -> str:
    source = svc.create_source(
        "fake", "posthog", {"query": "SELECT 1", "token": "secret-abc"}
    )
    return source.id


def test_editing_the_config_keeps_a_secret_left_blank(service) -> None:
    # The whole point: fixing a query must not cost the operator their key.
    svc, _ = service
    source_id = _create(svc)

    svc.update_source(source_id, config={"query": "SELECT 2", "token": ""})

    validated = FakeConnector.seen[-1]
    assert validated["query"] == "SELECT 2"
    assert validated["token"] == "secret-abc", "a blank secret means unchanged"


def test_a_secret_left_out_entirely_is_also_kept(service) -> None:
    svc, _ = service
    source_id = _create(svc)

    svc.update_source(source_id, config={"query": "SELECT 3"})

    assert FakeConnector.seen[-1]["token"] == "secret-abc"


def test_a_supplied_secret_replaces_the_stored_one(service) -> None:
    svc, _ = service
    source_id = _create(svc)

    svc.update_source(source_id, config={"query": "SELECT 1", "token": "rotated"})

    assert FakeConnector.seen[-1]["token"] == "rotated"


def test_an_invalid_config_is_rejected_and_nothing_is_saved(service) -> None:
    svc, _ = service
    source_id = _create(svc)
    FakeConnector.accept = False

    with pytest.raises(WarehouseError, match="nope"):
        svc.update_source(source_id, config={"query": "broken"})

    FakeConnector.accept = True
    svc.update_source(source_id, config={"query": "SELECT 1"})
    # The rejected edit must not have reached storage.
    assert FakeConnector.seen[-1]["query"] == "SELECT 1"


def test_renaming_onto_an_existing_name_is_refused(service) -> None:
    svc, _ = service
    source_id = _create(svc)
    svc.create_source("fake", "other", {"query": "x", "token": "t"})

    with pytest.raises(WarehouseError, match="already exists"):
        svc.update_source(source_id, name="other")


def test_a_source_can_be_renamed_to_a_free_name(service) -> None:
    svc, _ = service
    source_id = _create(svc)

    updated = svc.update_source(source_id, name="posthog-funnel")

    assert updated.name == "posthog-funnel"


def test_a_new_resource_in_the_manifest_becomes_a_table(service) -> None:
    svc, store = service
    source_id = _create(svc)
    FakeConnector.resources = ("alpha", "beta")

    svc.update_source(source_id, config={"query": "SELECT 1"})

    names = {s.name for s in store.list_external_schemas(source_id)}
    assert names == {"alpha", "beta"}


def test_a_resource_that_disappears_is_disabled_not_deleted(service) -> None:
    # Its warehouse table holds rows the edit did not ask to destroy, and a resource
    # dropped by a typo comes back when the typo is fixed.
    svc, store = service
    source_id = _create(svc)
    FakeConnector.resources = ("beta",)

    svc.update_source(source_id, config={"query": "SELECT 1"})

    by_name = {s.name: s for s in store.list_external_schemas(source_id)}
    assert set(by_name) == {"alpha", "beta"}, "alpha must survive"
    assert by_name["alpha"].should_sync is False
    assert by_name["beta"].should_sync is True


def test_editing_the_config_does_not_wipe_existing_schemas(service) -> None:
    svc, store = service
    source_id = _create(svc)
    before = {s.id for s in store.list_external_schemas(source_id)}

    svc.update_source(source_id, config={"query": "SELECT 9"})

    after = {s.id for s in store.list_external_schemas(source_id)}
    assert before == after, "a surviving resource keeps its schema row and its id"


def test_sync_frequency_can_still_be_changed_on_its_own(service) -> None:
    # The endpoint previously did only this; it must keep working.
    svc, _ = service
    source_id = _create(svc)

    updated = svc.update_source(source_id, sync_frequency="1hour")

    assert updated.sync_frequency == "1hour"
    assert not FakeConnector.seen[1:], "no config change means no re-validation"


def test_editing_only_the_description_does_not_revalidate(service) -> None:
    svc, _ = service
    source_id = _create(svc)

    updated = svc.update_source(source_id, description="  the activation funnel  ")

    assert updated.description == "the activation funnel"
    assert not FakeConnector.seen[1:]


def test_the_stored_config_round_trips_encrypted(service) -> None:
    svc, store = service
    source_id = _create(svc)

    svc.update_source(source_id, config={"query": "SELECT 42"})

    raw = store.get_external_source(source_id).config_encrypted
    assert raw.startswith("enc:"), "an edited config must stay encrypted at rest"
    assert "secret-abc" not in raw
    assert (
        json.loads(
            json.dumps(svc._decode_config(store.get_external_source(source_id)))
        )["token"]
        == "secret-abc"
    )
