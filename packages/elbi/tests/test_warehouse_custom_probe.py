"""The Custom REST connection test: prove the manifest works, don't just parse it.

Structural validation lets through a manifest that cannot sync. The case this exists
for: no `paginator` declared, so dlt guesses one, and for an endpoint that answers in a
single response the guess chases a next page that never arrives. The source saves
cleanly and the sync then sits in `pending` with no error and no timeout.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")

from elbi.warehouse.sources.custom import (
    _PROBE_GRACE,
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


class _Resource:
    """A stand-in for a dlt resource, with the one method the probe leans on.

    `add_limit` is how the walk is bounded, so a fake that lacks it would let the probe
    pass a test it could not pass in production.
    """

    def __init__(self, records: list[dict] | None = None) -> None:
        self.records = records or []
        self.max_time: float | None = None

    def add_limit(self, max_items=None, max_time=None):
        self.max_time = max_time
        return self

    def __iter__(self):
        yield from self.records


class _EndlessWalk(_Resource):
    """A paginator that never finds a record: requests forever unless it is stopped.

    It honours `max_time` the way dlt's own limit does — checked between pages — so a
    probe that forgets to set one leaves it running, which is the bug under test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.requests = 0
        self.finished = threading.Event()

    def __iter__(self):
        deadline = time.monotonic() + (self.max_time if self.max_time else 60.0)
        while time.monotonic() < deadline:
            self.requests += 1  # a page that came back empty
            time.sleep(0.01)
            yield from ()
        self.finished.set()


def _serving(resource, monkeypatch) -> None:
    monkeypatch.setattr(
        "elbi.warehouse.sources._dlt.rest_api_source",
        lambda manifest: SimpleNamespace(resources={"things": resource}),
        raising=False,
    )


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


class _StuckRequest(_Resource):
    """A request that hangs before any page arrives — nothing for the limit to close."""

    def __iter__(self):
        threading.Event().wait(60)
        yield from ()


def test_a_hanging_fetch_is_reported_rather_than_left_spinning(monkeypatch) -> None:
    # The reported bug, reproduced: the resource never yields and never raises.
    _serving(_StuckRequest(), monkeypatch)
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_TIMEOUT", 0.2)
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_GRACE", 0.2)

    ok, errors = CustomSource()._probe(_MANIFEST)

    assert not ok
    assert "did not finish" in errors[0]


def test_the_hang_message_names_the_missing_paginator(monkeypatch) -> None:
    # Telling an operator "it timed out" without naming the likely cause sends them
    # hunting the network. The manifest is the thing they can fix.
    _serving(_StuckRequest(), monkeypatch)
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_TIMEOUT", 0.2)
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_GRACE", 0.2)

    _, errors = CustomSource()._probe(_MANIFEST)

    assert "paginator" in errors[0]
    assert "single_page" in errors[0]


def test_an_endless_walk_stops_itself_when_the_probe_gives_up(monkeypatch) -> None:
    """The walk carries the deadline, so giving up on it actually ends it.

    Joining a worker thread bounds the wait, not the work. Left unbounded, a guessed
    paginator that never yields a record keeps asking for the next page inside a server
    process that outlives the test by months.
    """
    walk = _EndlessWalk()
    _serving(walk, monkeypatch)
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_TIMEOUT", 0.3)

    ok, errors = CustomSource()._probe(_MANIFEST)

    assert not ok
    assert "did not finish" in errors[0]
    assert walk.max_time == 0.3, "the deadline must reach the walk, not just the join"
    assert walk.finished.wait(2), "the walk must end on its own"
    requests = walk.requests
    time.sleep(0.2)
    assert walk.requests == requests, "no requests after the probe reported the hang"


def test_a_real_endless_paginator_stops_when_the_budget_runs_out(monkeypatch) -> None:
    """The same thing through dlt itself, against an endpoint that pages forever.

    The fakes above prove the probe asks for a deadline; only dlt can prove the deadline
    lands. Every page here comes back empty with a link to the next one, which is the
    shape a guessed paginator produces — the walk has to end anyway, and stop asking.
    """
    pytest.importorskip("dlt")
    import json as _json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            body = _json.dumps(
                {"items": [], "next": f"{base_url}/things?page={len(requests) + 1}"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr("elbi.warehouse.sources.custom._PROBE_TIMEOUT", 1.0)

    try:
        ok, errors = CustomSource()._probe(
            {
                "client": {
                    "base_url": base_url,
                    "paginator": {"type": "json_link", "next_url_path": "next"},
                },
                "resources": [
                    {
                        "name": "things",
                        "endpoint": {"path": "/things", "data_selector": "items"},
                    }
                ],
            }
        )

        assert not ok
        assert "did not finish" in errors[0]
        served = len(requests)
        time.sleep(0.5)
        assert len(requests) == served, "the walk must not outlive the answer"
    finally:
        server.shutdown()
        server.server_close()


def test_an_endpoint_with_no_records_is_still_reachable(monkeypatch) -> None:
    # An empty endpoint answered and the walk ended on its own; that is a connection,
    # not a hang. The budget is what separates the two, so it must not be spent.
    _serving(_Resource([]), monkeypatch)

    ok, errors = CustomSource()._probe(_MANIFEST)

    assert ok and errors == []


def test_a_fetched_record_passes(monkeypatch) -> None:
    _serving(_Resource([{"id": 1}]), monkeypatch)

    ok, errors = CustomSource()._probe(_MANIFEST)

    assert ok and errors == []


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
    # The join has to outlast the walk's own deadline, or every bounded walk is
    # reported as still running.
    assert 0 < _PROBE_GRACE <= _PROBE_TIMEOUT


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
