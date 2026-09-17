"""The Custom REST connection test: prove the manifest works, don't just parse it.

Structural validation lets through a manifest that cannot sync. The case this exists
for: no `paginator` declared, so dlt guesses one, and for an endpoint that answers in a
single response the guess chases a next page that never arrives. The source saves
cleanly and the sync then sits in `pending` with no error and no timeout.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")

from elbi.warehouse.sources.custom import (
    _PROBE_TIMEOUT,
    CustomSource,
    _declares_paginator,
)

_MANIFEST = {
    "client": {"base_url": "https://api.example.com", "auth": {"type": "bearer"}},
    "resources": [{"name": "things", "endpoint": {"path": "/things"}}],
}


def _config(manifest: dict | None = None, **extra) -> dict:
    return {"manifest_json": json.dumps(manifest or _MANIFEST), **extra}


def test_structural_failures_are_reported_before_any_request(monkeypatch) -> None:
    # A malformed manifest must not cost a network round trip to reject.
    called = False

    def probe(self, manifest):
        nonlocal called
        called = True
        return True, []

    monkeypatch.setattr(CustomSource, "_probe", probe)
    source = CustomSource()

    ok, _ = source.validate(_config({"client": {}, "resources": []}))

    assert not ok
    assert not called, "structural checks must short-circuit the probe"


def test_a_manifest_that_fetches_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(CustomSource, "_probe", lambda self, m: (True, []))

    ok, errors = CustomSource().validate(_config())

    assert ok and errors == []


def test_a_hanging_fetch_is_reported_rather_than_left_spinning(monkeypatch) -> None:
    # The reported bug, reproduced: the resource never yields and never raises.
    import threading

    def never_returns(manifest):
        return SimpleNamespace(
            resources={"things": iter(lambda: threading.Event().wait(60), None)}
        )

    monkeypatch.setattr(
        "elbi.warehouse.sources._dlt.rest_api_source", never_returns, raising=False
    )
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_TIMEOUT", 0.2)

    ok, errors = CustomSource()._probe(_MANIFEST)

    assert not ok
    assert "did not finish" in errors[0]


def test_the_hang_message_names_the_missing_paginator(monkeypatch) -> None:
    # Telling an operator "it timed out" without naming the likely cause sends them
    # hunting the network. The manifest is the thing they can fix.
    import threading

    def never_returns(manifest):
        return SimpleNamespace(
            resources={"things": iter(lambda: threading.Event().wait(60), None)}
        )

    monkeypatch.setattr(
        "elbi.warehouse.sources._dlt.rest_api_source", never_returns, raising=False
    )
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_TIMEOUT", 0.2)

    _, errors = CustomSource()._probe(_MANIFEST)

    assert "paginator" in errors[0]
    assert "single_page" in errors[0]


def test_a_fetch_error_is_surfaced_verbatim(monkeypatch) -> None:
    def raises(manifest):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(
        "elbi.warehouse.sources._dlt.rest_api_source", raises, raising=False
    )

    ok, errors = CustomSource()._probe(_MANIFEST)

    assert not ok
    assert "401 Unauthorized" in errors[0]


def test_declares_paginator_checks_client_and_resources() -> None:
    assert not _declares_paginator(_MANIFEST)
    assert _declares_paginator(
        {
            **_MANIFEST,
            "client": {**_MANIFEST["client"], "paginator": {"type": "single_page"}},
        }
    )
    assert _declares_paginator(
        {
            **_MANIFEST,
            "resources": [
                {
                    "name": "t",
                    "endpoint": {"path": "/t", "paginator": {"type": "single_page"}},
                }
            ],
        }
    )


def test_the_probe_timeout_is_bounded_and_sane() -> None:
    assert 5 <= _PROBE_TIMEOUT <= 60


def test_test_source_revalidates_an_existing_source(monkeypatch, tmp_path) -> None:
    """The endpoint's service half: re-run validate against a stored config.

    Validation otherwise happens once, at creation, so a key that is later rotated or a
    scope that is revoked shows up first as a failed — or hung — sync.
    """
    from elbi.warehouse.service import WarehouseService

    calls: list[dict] = []

    class FakeConnector:
        def validate(self, config):
            calls.append(config)
            return False, ["401 Unauthorized"]

    service = object.__new__(WarehouseService)
    monkeypatch.setattr(
        WarehouseService,
        "get_source",
        lambda self, sid: SimpleNamespace(source_type="custom"),
    )
    monkeypatch.setattr(WarehouseService, "_connector", lambda self, t: FakeConnector())
    monkeypatch.setattr(
        WarehouseService, "_decode_config", lambda self, s: {"manifest_json": "{}"}
    )

    ok, errors = WarehouseService.test_source(service, "abc")

    assert not ok
    assert errors == ["401 Unauthorized"]
    assert calls == [{"manifest_json": "{}"}], "it must test the stored config"
