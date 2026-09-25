"""The PostHog connector: its catalogue entry, its paging, and its columnar reader.

PostHog answers its query endpoint columnar -- rows are arrays and the names arrive
once, beside them -- so a reader written for the object shape every other connector here
meets returns zero rows and no error. That is the failure this file exists to catch, and
it is checked both on the response shape directly and end to end against a local server
standing in for the API.

The rest is the shape of the connector against PostHog's documented behaviour: the host
is a field because a key only works against the deployment that issued it, the list
endpoints are walked by the absolute link they return, and the event query never carries
an OFFSET, which PostHog rejects with a 400 for exactly this kind of key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("pyarrow")

from elbi.warehouse.sources.base import SourceInputs
from elbi.warehouse.sources.posthog import (
    _clickhouse_time,
    _query,
    _records,
)
from elbi.warehouse.sources.registry import SourceRegistry

EVENT_COLUMNS = [
    "uuid",
    "event",
    "timestamp",
    "distinct_id",
    "person_id",
    "properties",
    "elements_chain",
]


def _source() -> Any:
    return SourceRegistry.get("posthog")


# --- the catalogue entry ----------------------------------------------------------


def test_posthog_is_catalogued_under_analytics_with_a_docs_link() -> None:
    config = _source().config
    assert config.label == "PostHog"
    assert config.category == "Analytics"
    assert config.docs_url.startswith("https://")


def test_the_api_key_is_stored_as_a_secret() -> None:
    fields = {f.name: f for f in _source().config.fields}
    assert fields["api_key"].type == "password"


def test_the_host_is_a_field_because_a_key_only_works_on_its_own_deployment() -> None:
    """US Cloud, EU Cloud and self-hosted are three different hosts for one API."""
    fields = {f.name: f for f in _source().config.fields}
    assert fields["host"].default == "https://us.posthog.com"
    assert "eu.posthog.com" in fields["host"].caption


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("", "https://us.posthog.com/api/projects/42/cohorts/"),
        ("https://eu.posthog.com", "https://eu.posthog.com/api/projects/42/cohorts/"),
        ("https://eu.posthog.com/", "https://eu.posthog.com/api/projects/42/cohorts/"),
        (" https://ph.acme.com ", "https://ph.acme.com/api/projects/42/cohorts/"),
    ],
)
def test_a_pasted_host_still_addresses_the_project(host: str, expected: str) -> None:
    """A URL copied from a browser arrives with a trailing slash and stray spaces."""
    from elbi.warehouse.sources.posthog import _url

    assert _url({"host": host, "project_id": "42"}, "cohorts/") == expected


# --- what is offered, and what carries a cursor -----------------------------------


def test_the_object_endpoints_are_offered_and_selected_by_default() -> None:
    schemas = {s.name: s for s in _source().schemas({})}
    assert set(schemas) == {
        "persons",
        "cohorts",
        "feature_flags",
        "insights",
        "experiments",
        "actions",
        "annotations",
        "surveys",
        "events",
    }
    assert all(schemas[name].default_selected for name in schemas if name != "events")


def test_raw_events_is_offered_but_never_selected_for_you() -> None:
    """It is the largest table in most projects, and PostHog rate-limits reading it."""
    events = next(s for s in _source().schemas({}) if s.name == "events")
    assert events.default_selected is False
    assert events.incremental_fields == ["timestamp"]


def test_no_object_endpoint_claims_a_cursor_it_cannot_filter_on() -> None:
    """None of them accepts a modified-since filter, so a cursor would be a fiction."""
    objects = [s for s in _source().schemas({}) if s.name != "events"]
    assert all(s.incremental_fields == [] for s in objects)


# --- the columnar response --------------------------------------------------------


def test_a_columnar_response_becomes_records() -> None:
    """The connector turns on this: results are arrays, the names arrive beside."""
    body = {
        "columns": ["event", "timestamp"],
        "results": [["pageview", "2024-05-01T00:00:00Z"], ["signup", None]],
    }
    assert _records(body) == [
        {"event": "pageview", "timestamp": "2024-05-01T00:00:00Z"},
        {"event": "signup", "timestamp": None},
    ]


def test_a_response_with_no_rows_or_no_columns_is_empty_rather_than_wrong() -> None:
    assert _records({"columns": ["a"], "results": []}) == []
    assert _records({"results": [[1, 2]]}) == []
    assert _records({}) == []


def test_a_row_with_more_values_than_columns_keeps_the_named_ones() -> None:
    """A column added on PostHog's side must not take a sync down with a ValueError."""
    body = {"columns": ["a"], "results": [[1, "extra"]]}
    assert _records(body) == [{"a": 1}]


# --- the event query --------------------------------------------------------------


def test_the_first_page_of_events_is_unfiltered_and_ordered() -> None:
    body = _query(None)
    assert body["query"]["kind"] == "HogQLQuery"
    assert "WHERE" not in body["query"]["query"]
    assert "ORDER BY timestamp" in body["query"]["query"]
    assert "values" not in body["query"]


def test_a_cursor_is_bound_as_a_value_and_never_spliced_into_the_query() -> None:
    """Otherwise a stored cursor would be read as HogQL on the way back out."""
    body = _query("2024-05-01 00:00:00.000000")
    assert "WHERE timestamp > {since}" in body["query"]["query"]
    assert body["query"]["values"] == {"since": "2024-05-01 00:00:00.000000"}
    assert "2024-05-01" not in body["query"]["query"]


def test_the_event_query_never_carries_an_offset() -> None:
    """PostHog answers 400 to an OFFSET from a personal key; keyset paging is why."""
    assert "OFFSET" not in _query(None)["query"]["query"].upper()
    assert "OFFSET" not in _query("2024-05-01 00:00:00")["query"]["query"].upper()


def test_the_query_refuses_the_cache_and_names_itself() -> None:
    """Cached, the first page of a full refresh would be the same snapshot every run."""
    body = _query(None)
    assert body["refresh"] == "force_blocking"
    assert body["name"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("2024-05-01T12:34:56.789000Z", "2024-05-01 12:34:56.789000"),
        ("2024-05-01T12:34:56.789000+00:00", "2024-05-01 12:34:56.789000"),
        ("2024-05-01T14:34:56.789000+02:00", "2024-05-01 12:34:56.789000"),
        ("2024-05-01T12:34:56", "2024-05-01 12:34:56.000000"),
    ],
)
def test_a_cursor_is_rendered_as_the_datetime_clickhouse_parses(
    given: str, expected: str
) -> None:
    """The API hands back ISO-8601; the events table is compared against neither T nor
    a zone offset, so the cursor goes through a real datetime on the way."""
    assert _clickhouse_time(given) == expected


def test_a_cursor_that_is_not_a_datetime_is_passed_through_unchanged() -> None:
    """Better a query PostHog rejects out loud than a silently rewritten filter."""
    assert _clickhouse_time("not-a-date") == "not-a-date"


# --- end to end, against a local server -------------------------------------------


QUERIES: list[dict[str, Any]] = []


class _Handler(BaseHTTPRequestHandler):
    """A stand-in PostHog: an offset-paged list endpoint and a columnar query one."""

    def log_message(self, *args: Any) -> None:
        return

    def _send(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") == "Bearer secret":
            return True
        self._send({"detail": "Incorrect authentication credentials."}, status=401)
        return False

    def do_GET(self) -> None:
        if not self._authorized():
            return
        url = urlparse(self.path)
        query = parse_qs(url.query)
        offset = int(query.get("offset", ["0"])[0])
        if url.path == "/api/projects/42/cohorts/":
            self._send({"count": 1, "next": None, "results": [{"id": 1}]})
            return
        if url.path != "/api/projects/42/persons/":
            self._send({"detail": "Endpoint not found."}, status=404)
            return
        if offset:
            self._send({"count": 3, "next": None, "results": [{"id": 3}]})
        else:
            host = self.headers.get("Host")
            self._send(
                {
                    "count": 3,
                    # Absolute, and already carrying the paging parameters, the way
                    # PostHog returns it.
                    "next": f"http://{host}/api/projects/42/persons/?limit=100&offset=2",
                    "results": [{"id": 1}, {"id": 2}],
                }
            )

    def do_POST(self) -> None:
        if not self._authorized():
            return
        length = int(self.headers.get("Content-Length") or 0)
        sent = json.loads(self.rfile.read(length) or b"{}")
        QUERIES.append(sent)
        since = sent["query"].get("values", {}).get("since")
        if since is None:
            # A full page, so the connector has to come back for the next one.
            rows = [
                [
                    f"u{n}",
                    "pageview",
                    f"2024-05-01T00:00:{n % 60:02d}.000000Z",
                    "d",
                    "p",
                    "{}",
                    "",
                ]
                for n in range(1000)
            ]
        elif since.startswith("2024-05-01"):
            rows = [
                ["u-last", "signup", "2024-05-02T00:00:00.000000Z", "d", "p", "{}", ""]
            ]
        else:
            rows = []
        self._send({"columns": EVENT_COLUMNS, "results": rows})


@pytest.fixture
def posthog() -> str:
    """A local HTTP server, so the request-to-rows path runs without a PostHog."""
    QUERIES.clear()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _config(host: str, key: str = "secret") -> dict[str, Any]:
    return {"host": host, "api_key": key, "project_id": "42"}


def _rows(host: str, schema: str, **kwargs: Any) -> list[dict[str, Any]]:
    inputs = SourceInputs(config=_config(host), schema=schema, **kwargs)
    return [row for batch in _source().extract(inputs) for row in batch.to_pylist()]


def test_a_list_endpoint_is_walked_to_the_last_page(posthog: str) -> None:
    """The second page exists only behind the link the first one returned."""
    assert [r["id"] for r in _rows(posthog, "persons")] == [1, 2, 3]


def test_events_arrive_as_records_and_not_as_arrays(posthog: str) -> None:
    rows = _rows(posthog, "events")
    assert len(rows) == 1001
    assert rows[0]["event"] == "pageview"
    assert set(rows[0]) == set(EVENT_COLUMNS)


def test_each_event_page_resumes_from_the_last_timestamp_of_the_one_before(
    posthog: str,
) -> None:
    """Keyset paging: the cursor is the data, so no page is fetched twice or skipped."""
    _rows(posthog, "events")
    sent = [q["query"].get("values", {}).get("since") for q in QUERIES]
    assert sent[0] is None
    assert sent[1] == "2024-05-01 00:00:39.000000"
    assert len(sent) == 2  # a short page ends the walk


def test_an_incremental_run_asks_only_for_what_came_after_the_cursor(
    posthog: str,
) -> None:
    rows = _rows(
        posthog,
        "events",
        incremental_field="timestamp",
        incremental_since="2024-05-01T23:59:59.000000Z",
    )
    assert [r["event"] for r in rows] == ["signup"]
    assert QUERIES[0]["query"]["values"] == {"since": "2024-05-01 23:59:59.000000"}


def test_the_bearer_token_is_sent(posthog: str) -> None:
    """The stub answers 401 without it, so rows at all is the assertion."""
    assert _rows(posthog, "persons")


def test_a_good_key_validates_against_the_project(posthog: str) -> None:
    ok, errors = _source().validate(_config(posthog))
    assert ok, errors


def test_a_bad_key_fails_at_save_time_rather_than_at_sync_time(posthog: str) -> None:
    ok, errors = _source().validate(_config(posthog, key="wrong"))
    assert not ok
    assert "Could not authenticate with PostHog" in " ".join(errors)


def test_a_missing_field_is_reported_before_anything_is_fetched() -> None:
    ok, errors = _source().validate({"host": "https://us.posthog.com"})
    assert not ok
    assert any("api_key" in e for e in errors)
