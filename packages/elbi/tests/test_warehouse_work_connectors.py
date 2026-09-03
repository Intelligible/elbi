"""The seven work-system connectors, Google Sheets, and Oracle.

These reach services no test can stand up, so the contract under test is the manifest
each connector produces: the base URL, the auth type, the paginator, and the paths it
lists. That is where a connector is wrong in practice -- a cursor read from the wrong
field, or a selector pointing at the wrong key, fails against the live API and nowhere
else -- and the values are checked against each vendor's documented shape.

The engine those manifests run on is then driven end to end against a local server, so
the path from a declaration to fetched rows is covered once for all of them, including
the paging and the POST listing that Notion needs.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from elbi.warehouse.sources.base import SourceInputs
from elbi.warehouse.sources.registry import SourceRegistry
from elbi.warehouse.sources.rest_base import RestApiConnector

WORK_CONNECTORS = (
    "github",
    "jira",
    "notion",
    "slack",
    "sentry",
    "typeform",
    "intercom",
)


def _manifest(source_type: str, config: dict[str, Any]) -> dict[str, Any]:
    return SourceRegistry.get(source_type)._manifest(config)  # type: ignore[attr-defined]


def _resources(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["name"]: r["endpoint"] for r in manifest["resources"]}


# --- every connector, the shared expectations -------------------------------------


@pytest.mark.parametrize("source_type", WORK_CONNECTORS)
def test_every_work_connector_is_catalogued_with_a_docs_link(source_type: str) -> None:
    """The docs link is where the user goes to make the token the form is asking for."""
    config = SourceRegistry.get(source_type).config
    assert config.label
    assert config.category
    assert config.docs_url.startswith("https://")


@pytest.mark.parametrize("source_type", WORK_CONNECTORS)
def test_every_secret_is_stored_as_a_secret(source_type: str) -> None:
    """A token in a text field is a token in the clear, and in a screenshot."""
    fields = SourceRegistry.get(source_type).config.fields
    secretish = [
        f for f in fields if any(w in f.name for w in ("token", "key", "secret"))
    ]
    assert secretish, f"{source_type} declares no credential field"
    assert all(f.type == "password" for f in secretish)


@pytest.mark.parametrize("source_type", WORK_CONNECTORS)
def test_every_connector_offers_at_least_one_table(source_type: str) -> None:
    config = {
        f.name: f.default or "x" for f in SourceRegistry.get(source_type).config.fields
    }
    config["repository"] = "owner/name"
    assert SourceRegistry.get(source_type).schemas(config)


# --- per-vendor shapes ------------------------------------------------------------


def test_github_targets_one_repository_over_bearer_and_link_paging() -> None:
    manifest = _manifest("github", {"access_token": "t", "repository": "octocat/hello"})
    assert manifest["client"]["base_url"] == "https://api.github.com"
    assert manifest["client"]["auth"] == {"type": "bearer", "token": "t"}
    assert manifest["client"]["paginator"] == {"type": "header_link"}
    # Pinned, so a future breaking change does not arrive inside a scheduled sync.
    assert manifest["client"]["headers"]["X-GitHub-Api-Version"] == "2022-11-28"
    endpoints = _resources(manifest)
    assert endpoints["issues"]["path"] == "/repos/octocat/hello/issues"
    assert endpoints["pull_requests"]["path"] == "/repos/octocat/hello/pulls"


def test_github_asks_for_closed_records_too() -> None:
    """The API default hides everything closed, which is most of the history."""
    endpoints = _resources(
        _manifest("github", {"access_token": "t", "repository": "o/n"})
    )
    assert endpoints["issues"]["params"] == {"state": "all"}
    assert endpoints["pull_requests"]["params"] == {"state": "all"}


@pytest.mark.parametrize(
    "given", ["octocat", "octocat/hello/extra", "/octocat/", "", "owner/"]
)
def test_github_refuses_a_repository_that_is_not_owner_slash_name(given: str) -> None:
    ok, errors = SourceRegistry.get("github").validate(
        {"access_token": "t", "repository": given}
    )
    assert not ok
    assert errors


def test_jira_authenticates_as_the_email_and_counts_records_not_pages() -> None:
    """Atlassian's tokens go in the password field with the email as the username."""
    manifest = _manifest(
        "jira", {"domain": "acme.atlassian.net", "email": "a@b.c", "api_token": "t"}
    )
    assert manifest["client"]["base_url"] == "https://acme.atlassian.net"
    assert manifest["client"]["auth"] == {
        "type": "http_basic",
        "username": "a@b.c",
        "password": "t",
    }
    paginator = manifest["client"]["paginator"]
    assert paginator["type"] == "offset"
    assert paginator["offset_param"] == "startAt"
    assert paginator["limit_param"] == "maxResults"
    assert paginator["total_path"] == "total"


def test_jira_accepts_a_domain_pasted_with_its_scheme() -> None:
    manifest = _manifest(
        "jira",
        {"domain": "https://acme.atlassian.net/", "email": "a@b.c", "api_token": "t"},
    )
    assert manifest["client"]["base_url"] == "https://acme.atlassian.net"


def test_jira_reads_records_from_where_each_endpoint_puts_them() -> None:
    endpoints = _resources(
        _manifest("jira", {"domain": "d", "email": "e", "api_token": "t"})
    )
    assert endpoints["issues"]["data_selector"] == "issues"
    assert endpoints["projects"]["data_selector"] == "values"
    # A bare array, so naming a key would find nothing.
    assert "data_selector" not in endpoints["users"]


def test_notion_lists_content_over_post_and_names_its_api_version() -> None:
    """Notion has no GET that lists pages, and refuses a request with no version."""
    manifest = _manifest("notion", {"api_token": "t"})
    assert manifest["client"]["headers"]["Notion-Version"] == "2022-06-28"
    endpoints = _resources(manifest)
    assert endpoints["pages"]["method"] == "POST"
    assert endpoints["pages"]["json"]["filter"]["value"] == "page"
    assert endpoints["databases"]["json"]["filter"]["value"] == "database"
    # Users is an ordinary GET, so it must not have acquired a method or a body.
    assert "method" not in endpoints["users"]


def test_notion_pages_on_the_cursor_names_notion_actually_uses() -> None:
    paginator = _manifest("notion", {"api_token": "t"})["client"]["paginator"]
    assert paginator["cursor_path"] == "next_cursor"
    assert paginator["cursor_param"] == "start_cursor"


def test_slack_pages_on_the_cursor_nested_in_response_metadata() -> None:
    paginator = _manifest("slack", {"bot_token": "t"})["client"]["paginator"]
    assert paginator["cursor_path"] == "response_metadata.next_cursor"
    assert paginator["cursor_param"] == "cursor"


def test_slack_offers_messages_only_once_a_channel_is_named() -> None:
    """Otherwise the table could be selected and then fail asking for something the
    form never requested."""
    without = _resources(_manifest("slack", {"bot_token": "t"}))
    assert set(without) == {"channels", "users"}
    with_channel = _resources(
        _manifest("slack", {"bot_token": "t", "channel_id": "C1"})
    )
    assert with_channel["messages"]["params"] == {"channel": "C1"}


def test_sentry_defaults_to_sentry_io_but_accepts_a_self_hosted_host() -> None:
    hosted = _manifest("sentry", {"auth_token": "t", "organization": "acme"})
    assert hosted["client"]["base_url"] == "https://sentry.io/api/0"
    own = _manifest(
        "sentry",
        {"auth_token": "t", "organization": "acme", "host": "https://sentry.acme.com/"},
    )
    assert own["client"]["base_url"] == "https://sentry.acme.com/api/0"


def test_sentry_scopes_every_path_to_the_organisation() -> None:
    endpoints = _resources(
        _manifest("sentry", {"auth_token": "t", "organization": "acme"})
    )
    assert all("/organizations/acme/" in e["path"] for e in endpoints.values())


@pytest.mark.parametrize(
    ("region", "host"),
    [
        ("us", "https://api.intercom.io"),
        ("eu", "https://api.eu.intercom.io"),
        ("au", "https://api.au.intercom.io"),
        ("", "https://api.intercom.io"),
    ],
)
def test_intercom_addresses_the_region_the_workspace_lives_in(
    region: str, host: str
) -> None:
    """A token is valid only in its own region, so a wrong host is a hard fail."""
    manifest = _manifest("intercom", {"access_token": "t", "region": region})
    assert manifest["client"]["base_url"] == host


def test_intercom_pages_on_the_cursor_inside_its_pages_object() -> None:
    paginator = _manifest("intercom", {"access_token": "t"})["client"]["paginator"]
    assert paginator["cursor_path"] == "pages.next.starting_after"
    assert paginator["cursor_param"] == "starting_after"


def test_typeform_counts_pages_from_one() -> None:
    """Numbered from one, not zero: starting at zero silently skips the first page."""
    paginator = _manifest("typeform", {"access_token": "t"})["client"]["paginator"]
    assert paginator["type"] == "page_number"
    assert paginator["base_page"] == 1
    assert paginator["total_path"] == "page_count"


def test_typeform_offers_responses_only_once_a_form_is_named() -> None:
    without = _resources(_manifest("typeform", {"access_token": "t"}))
    assert set(without) == {"forms"}
    with_form = _resources(
        _manifest("typeform", {"access_token": "t", "form_id": "F1"})
    )
    assert with_form["responses"]["path"] == "/forms/F1/responses"


# --- the engine, end to end against a local server --------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Two endpoints: one paged by cursor over GET, one listing over POST."""

    def log_message(self, *args: Any) -> None:
        return

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if "Bearer secret" not in self.headers.get("Authorization", ""):
            self.send_response(401)
            self.end_headers()
            return
        after = "" if "after=" not in self.path else self.path.split("after=")[1]
        if not after:
            self._send({"items": [{"n": 1}, {"n": 2}], "next": {"token": "p2"}})
        elif after == "p2":
            self._send({"items": [{"n": 3}], "next": {"token": None}})
        else:  # pragma: no cover - a cursor we never issued
            self._send({"items": [], "next": {"token": None}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        sent = json.loads(self.rfile.read(length) or b"{}")
        self._send({"results": [{"echoed": sent.get("kind")}], "next": {"token": None}})


@pytest.fixture
def stub() -> str:
    """A local HTTP server, so the manifest-to-rows path runs without a vendor."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class _StubConnector(RestApiConnector):
    """A connector over the stub, declared exactly the way the real ones are."""

    key, label, category = "stub_work", "Stub", "Productivity"
    caption = "A local stub."
    docs_url = "https://example.invalid/docs"
    resources_ = (
        {"name": "paged", "path": "/things", "data_selector": "items"},
        {
            "name": "posted",
            "path": "/search",
            "data_selector": "results",
            "method": "POST",
            "json": {"kind": "page"},
        },
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return str(config["base_url"])

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["token"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        return {"type": "cursor", "cursor_path": "next.token", "cursor_param": "after"}


def _stub_rows(stub: str, schema: str) -> list[dict[str, Any]]:
    source = _StubConnector()
    inputs = SourceInputs(config={"base_url": stub, "token": "secret"}, schema=schema)
    return [row for batch in source.extract(inputs) for row in batch.to_pylist()]


def test_the_engine_follows_a_cursor_to_the_last_page(stub: str) -> None:
    """Three records across two pages: the second page is only reached by the cursor."""
    assert [r["n"] for r in _stub_rows(stub, "paged")] == [1, 2, 3]


def test_the_engine_can_list_over_post_with_a_body(stub: str) -> None:
    """What Notion needs, proven against a server that echoes the body back."""
    assert _stub_rows(stub, "posted") == [{"echoed": "page"}]


def test_the_engine_sends_the_bearer_token(stub: str) -> None:
    """The stub answers 401 without it, so rows at all is the assertion."""
    assert _stub_rows(stub, "paged")


# --- Google Sheets ----------------------------------------------------------------


def test_google_sheets_is_catalogued_under_file_storage() -> None:
    config = SourceRegistry.get("google_sheets").config
    assert config.label == "Google Sheets"
    assert config.category == "File storage"
    assert [f.name for f in config.fields] == ["spreadsheet", "key_file"]


def test_google_sheets_says_the_sheet_has_to_be_shared() -> None:
    """A valid key on an unshared sheet returns nothing, which looks like a bug."""
    fields = {f.name: f for f in SourceRegistry.get("google_sheets").config.fields}
    assert "share" in fields["key_file"].caption.lower()
    assert "client_email" in fields["key_file"].caption


@pytest.mark.parametrize(
    "given",
    [
        "https://docs.google.com/spreadsheets/d/1AbC-_xyz012345678901/edit#gid=0",
        "https://docs.google.com/spreadsheets/d/1AbC-_xyz012345678901",
        "1AbC-_xyz012345678901",
        "  1AbC-_xyz012345678901  ",
    ],
)
def test_a_spreadsheet_is_found_from_the_url_or_the_bare_id(given: str) -> None:
    """People copy the address bar, so id-only would be the wrong default."""
    from elbi.warehouse.sources.google_sheets import spreadsheet_id

    assert spreadsheet_id(given) == "1AbC-_xyz012345678901"


@pytest.mark.parametrize("given", ["", "nope", "https://example.com/", "short"])
def test_something_that_is_neither_says_so(given: str) -> None:
    from elbi.warehouse.sources.google_sheets import spreadsheet_id

    with pytest.raises(ValueError, match="spreadsheet id nor a Google Sheets URL"):
        spreadsheet_id(given)


def test_blank_and_repeated_headers_still_produce_usable_columns() -> None:
    """A blank header still names a column of data; a repeat would otherwise collide."""
    from elbi.warehouse.sources.google_sheets import _headers

    assert _headers(["Name", "", "Name", None, "Total"]) == [
        "Name",
        "column_2",
        "Name_2",
        "column_4",
        "Total",
    ]


def test_a_column_of_one_type_keeps_it_and_a_mixed_column_becomes_text() -> None:
    from elbi.warehouse.sources.google_sheets import _column_types

    names = ["whole", "fraction", "flag", "mixed"]
    rows = [[1, 1.5, True, 1], [2, 2.5, False, "two"]]
    types = {k: str(v) for k, v in _column_types(names, rows).items()}
    assert types == {
        "whole": "int64",
        "fraction": "double",
        "flag": "bool",
        "mixed": "string",
    }


def test_an_empty_column_is_text_rather_than_a_guess() -> None:
    from elbi.warehouse.sources.google_sheets import _column_types

    assert str(_column_types(["a"], [[None], [""]])["a"]) == "string"


def test_a_short_row_is_padded_because_sheets_omits_trailing_blanks() -> None:
    """A short row is a row whose last cells are empty, not one missing them."""
    from elbi.warehouse.sources.google_sheets import _column_types, _record

    names = ["a", "b", "c"]
    types = _column_types(names, [[1, "x", "y"]])
    assert _record([9], names, types) == {"a": 9, "b": None, "c": None}


def test_a_tab_name_containing_a_quote_is_escaped_into_the_range() -> None:
    """An unquoted A1 range with a stray quote addresses something else entirely."""
    from urllib.parse import unquote

    from elbi.warehouse.sources.google_sheets import _range

    assert unquote(_range("Q1 'raw'", 1, 100)) == "'Q1 ''raw'''!1:100"


def test_a_malformed_service_account_key_says_so() -> None:
    ok, errors = SourceRegistry.get("google_sheets").validate(
        {"spreadsheet": "1AbC-_xyz012345678901", "key_file": "{oops"}
    )
    assert not ok
    assert "not valid JSON" in " ".join(errors)


# --- Oracle -----------------------------------------------------------------------


def test_oracle_is_addressed_by_service_name_not_sid() -> None:
    """A SID names an instance; a service name names where to route a client."""
    source = SourceRegistry.get("oracle")
    url = source._url(  # type: ignore[attr-defined]
        {
            "host": "db.acme.com",
            "service_name": "ORCLPDB1",
            "user": "u",
            "password": "p",
        }
    )
    assert url == "oracle+oracledb://u:p@db.acme.com:1521/?service_name=ORCLPDB1"


def test_oracle_requires_the_schema_because_a_schema_is_a_user() -> None:
    """Left empty, the listing would return the data dictionary along with the data."""
    fields = {f.name: f for f in SourceRegistry.get("oracle").config.fields}
    assert fields["schema"].required is True
    assert fields["service_name"].required is True


def test_a_password_with_url_characters_survives_the_oracle_url() -> None:
    url = SourceRegistry.get("oracle")._url(  # type: ignore[attr-defined]
        {"host": "h", "service_name": "S", "user": "u", "password": "p@ss/word:1"}
    )
    assert "p%40ss%2Fword%3A1" in url
