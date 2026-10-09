"""The PostHog connector: its catalogue entry, what it offers, and its paging.

The connector is declared over the shared ``rest_api`` base, so the paging under test is
dlt's, driven against a local server shaped like PostHog's list endpoints: an absolute
``next`` link that already carries its own ``limit`` and ``offset``. A walk that re-sent
its first-page parameters alongside that link would restart at page one forever.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("dlt")

from elbi.warehouse.sources.base import SourceInputs
from elbi.warehouse.sources.registry import SourceRegistry

OBJECTS = {
    "persons",
    "cohorts",
    "feature_flags",
    "insights",
    "experiments",
    "actions",
    "annotations",
    "surveys",
}
COHORTS = 250  # three pages at PostHog's default page size of 100
KEY = "phx_test"


def _source() -> Any:
    return SourceRegistry.get("posthog")


# --- the catalogue entry ----------------------------------------------------------


def test_posthog_is_catalogued_under_analytics_as_beta() -> None:
    config = _source().config
    assert config.label == "PostHog"
    assert config.category == "Analytics"
    assert config.docs_url.startswith("https://")
    assert config.release_status == "beta"


def test_the_api_key_is_stored_as_a_secret() -> None:
    fields = {f.name: f for f in _source().config.fields}
    assert fields["api_key"].is_secret


def test_the_host_is_a_field_because_a_key_only_works_on_its_own_deployment() -> None:
    """US Cloud, EU Cloud and self-hosted are three different hosts for one API."""
    fields = {f.name: f for f in _source().config.fields}
    assert fields["host"].default == "https://us.posthog.com"
    assert "eu.posthog.com" in fields["host"].caption


def test_only_the_object_tables_are_offered() -> None:
    """Raw events stay out; none of these lists accepts a modified-since filter."""
    schemas = {s.name: s for s in _source().schemas({"project_id": "42"})}
    assert set(schemas) == OBJECTS
    assert all(s.default_selected for s in schemas.values())
    assert all(s.incremental_fields == [] for s in schemas.values())


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("", "https://us.posthog.com/api/projects/42"),
        ("https://eu.posthog.com/", "https://eu.posthog.com/api/projects/42"),
        (" https://ph.acme.com ", "https://ph.acme.com/api/projects/42"),
    ],
)
def test_a_pasted_host_still_addresses_the_project(host: str, expected: str) -> None:
    """A URL copied from a browser arrives with a trailing slash and stray spaces."""
    assert _source()._base_url({"host": host, "project_id": "42"}) == expected


# --- paging, end to end -----------------------------------------------------------


@pytest.fixture
def posthog() -> Iterator[tuple[str, list[str]]]:
    """A local list endpoint shaped like PostHog's; yields its URL and the auth seen."""
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            seen.append(self.headers.get("Authorization", ""))
            if seen[-1] != f"Bearer {KEY}":
                self.send_response(401)
                self.end_headers()
                return
            url = urlparse(self.path)
            query = parse_qs(url.query)
            limit = int(query.get("limit", ["100"])[0])
            offset = int(query.get("offset", ["0"])[0])
            end = min(offset + limit, COHORTS)
            base = f"http://127.0.0.1:{self.server.server_address[1]}{url.path}"
            nxt = f"{base}?limit={limit}&offset={end}" if end < COHORTS else None
            body = {
                "count": COHORTS,
                "next": nxt,
                "previous": None,
                "results": [{"id": i, "name": f"c{i}"} for i in range(offset, end)],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def test_a_list_is_walked_by_its_next_link_once_with_the_key_on_every_page(
    posthog: tuple[str, list[str]],
) -> None:
    host, seen = posthog
    config = {"host": host, "project_id": "42", "api_key": KEY}

    tables = list(_source().extract(SourceInputs(config=config, schema="cohorts")))

    ids = [row["id"] for table in tables for row in table.to_pylist()]
    assert sorted(ids) == list(range(COHORTS))  # nothing lost, nothing twice
    assert len(seen) == 3  # one request per page: the walk never restarted
    assert seen == [f"Bearer {KEY}"] * 3
