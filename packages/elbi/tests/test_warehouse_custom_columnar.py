"""Custom REST against APIs whose records are not a plain list of objects.

PostHog's query API answers columnar data: the field names sit once in a sibling
``columns`` list and every row is a bare array. A connector that keeps only objects
drops every row and reports a successful sync of nothing. The same API caps a query
with no LIMIT and says so with ``hasMore: true``, and landing that silently is just as
wrong. These drive a real sync, through dlt, against a local server.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("dlt")

from elbi.warehouse.sources.base import SourceInputs
from elbi.warehouse.sources.custom import CustomSource

#: Path to the body the stub answers with, set per test.
_ROUTES: dict[str, Any] = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return

    def _answer(self) -> None:
        route = _ROUTES[self.path.split("?")[0]]
        payload = route(self.path) if callable(route) else route
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._answer()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._answer()


@pytest.fixture
def stub() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    _ROUTES.clear()


def _posthog(**overrides: Any) -> dict[str, Any]:
    """A HogQL query response, trimmed to the keys that matter, in PostHog's order."""
    return {
        "columns": ["day", "event", "events"],
        "results": [
            ["2026-09-01", "analysis_failed", 3],
            ["2026-09-01", "analysis_completed", 12],
        ],
        # A list of lists too, but not records: picking it up would be its own bug.
        "types": [["day", "Date"], ["event", "String"], ["events", "UInt64"]],
        "hasMore": False,
        **overrides,
    }


def _manifest(base_url: str, endpoint: dict[str, Any], **client: Any) -> str:
    return json.dumps(
        {
            "client": {"base_url": base_url, **client},
            "resources": [
                {
                    "name": "outcomes",
                    "endpoint": {
                        "path": "/api/projects/@current/query/",
                        "method": "POST",
                        "json": {"query": {"kind": "HogQLQuery", "query": "SELECT 1"}},
                        **endpoint,
                    },
                }
            ],
        }
    )


_SINGLE_PAGE = {"paginator": {"type": "single_page"}}


def _sync(manifest_json: str, schema: str = "outcomes") -> list[dict[str, Any]]:
    inputs = SourceInputs(config={"manifest_json": manifest_json}, schema=schema)
    return [
        row for batch in CustomSource().extract(inputs) for row in batch.to_pylist()
    ]


@pytest.mark.parametrize(
    "selector", [{}, {"data_selector": "results"}], ids=["detected", "declared"]
)
def test_a_columnar_response_lands_every_row_under_its_column_names(
    stub: str, selector: dict[str, Any]
) -> None:
    _ROUTES["/api/projects/@current/query/"] = _posthog()

    rows = _sync(_manifest(stub, {**_SINGLE_PAGE, **selector}))

    assert rows == [
        {"day": "2026-09-01", "event": "analysis_failed", "events": 3},
        {"day": "2026-09-01", "event": "analysis_completed", "events": 12},
    ]


def test_columns_are_read_beside_a_nested_record_list(stub: str) -> None:
    _ROUTES["/api/projects/@current/query/"] = {
        "data": {"columns": ["id", "name"], "rows": [[1, "a"], [2, "b"]]}
    }

    rows = _sync(_manifest(stub, {**_SINGLE_PAGE, "data_selector": "data.rows"}))

    assert rows == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


def test_a_row_of_the_wrong_width_fails_the_sync(stub: str) -> None:
    payload = _posthog()
    payload["results"].append(["2026-09-02", "analysis_failed"])
    _ROUTES["/api/projects/@current/query/"] = payload

    with pytest.raises(Exception, match="row 2 of 'results' has 2 values"):
        _sync(_manifest(stub, _SINGLE_PAGE))


def test_rows_with_no_column_names_fail_the_sync(stub: str) -> None:
    payload = _posthog()
    del payload["columns"]
    _ROUTES["/api/projects/@current/query/"] = payload

    with pytest.raises(Exception, match="returned a list where a record"):
        _sync(_manifest(stub, _SINGLE_PAGE))


def test_records_that_are_not_objects_are_never_dropped_quietly(stub: str) -> None:
    _ROUTES["/api/projects/@current/query/"] = {"results": ["a", "b"]}

    with pytest.raises(Exception, match="returned a str where a record"):
        _sync(_manifest(stub, _SINGLE_PAGE))


@pytest.mark.parametrize("paginator", [_SINGLE_PAGE, {}], ids=["single", "undeclared"])
def test_a_truncated_response_with_nothing_to_fetch_the_rest_fails(
    stub: str, paginator: dict[str, Any]
) -> None:
    _ROUTES["/api/projects/@current/query/"] = _posthog(hasMore=True)

    with pytest.raises(Exception, match="hasMore: true"):
        _sync(_manifest(stub, paginator))


def test_a_has_more_flag_still_drives_a_declared_paginator(stub: str) -> None:
    """A paginator that reads ``has_more`` itself fetches the rest, unhindered."""

    def charges(path: str) -> dict[str, Any]:
        if "page=1" in path:
            return {"data": [{"id": "ch_3"}], "has_more": False}
        return {"data": [{"id": "ch_1"}, {"id": "ch_2"}], "has_more": True}

    _ROUTES["/v1/charges"] = charges
    manifest = {
        "client": {
            "base_url": stub,
            "paginator": {
                "type": "page_number",
                "base_page": 0,
                "page_param": "page",
                "has_more_path": "has_more",
                "total_path": None,
            },
        },
        "resources": [
            {
                "name": "charges",
                "endpoint": {"path": "/v1/charges", "data_selector": "data"},
            }
        ],
    }

    rows = _sync(json.dumps(manifest), schema="charges")

    assert [r["id"] for r in rows] == ["ch_1", "ch_2", "ch_3"]


def test_the_connection_test_rejects_rows_it_could_not_name(stub: str) -> None:
    payload = _posthog()
    del payload["columns"]
    _ROUTES["/api/projects/@current/query/"] = payload

    ok, errors = CustomSource().validate(
        {"manifest_json": _manifest(stub, _SINGLE_PAGE)}
    )

    assert not ok
    assert "returned a list where a record" in errors[0]


def test_the_connection_test_accepts_a_columnar_response(stub: str) -> None:
    _ROUTES["/api/projects/@current/query/"] = _posthog()

    assert CustomSource().validate(
        {"manifest_json": _manifest(stub, _SINGLE_PAGE)}
    ) == (
        True,
        [],
    )
