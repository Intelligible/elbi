"""Google Sheets connector: a tab is a table, its first row the column names.

A spreadsheet is where a great deal of real analysis starts, and it is the source least
like a database: there is no schema, a column can hold anything, and the grid is ragged
because Sheets does not pad a row out to the width of the widest one.

Three things follow from that, and they are most of this file.

Values are requested unformatted, so a number arrives as a number rather than as the
string a locale happened to render it into. A column whose values are all the same JSON
type is then carried across with that type; a column mixing text and numbers becomes
text, because one Arrow column cannot be both.

Rows are padded to the header width and read in windows rather than all at once, so a
long sheet neither loses its trailing empty cells nor has to fit in memory in one piece.

Access is granted by sharing, not by the key. The service account has an email address,
and a sheet is readable only once someone shares it with that address -- which is why a
correct key can still return nothing, and why the form says so rather than leaving the
user to guess.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_API = "https://sheets.googleapis.com/v4/spreadsheets"
_SCOPES = ("https://www.googleapis.com/auth/spreadsheets.readonly",)

#: Rows per request and per Arrow batch. The API's own limit is on cells returned, so a
#: wide sheet is bounded by this at a smaller row count than a narrow one.
_WINDOW = 20_000

#: A spreadsheet id sits between `/d/` and the next slash in the URL people copy.
_ID_IN_URL = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")

#: What a bare spreadsheet id looks like, so a pasted id is not taken for a URL.
_BARE_ID = re.compile(r"^[a-zA-Z0-9-_]{20,}$")


def spreadsheet_id(value: str) -> str:
    """The spreadsheet id, from either the id itself or the URL it appears in.

    Users copy the address bar, not the id, so accepting only the id would make the
    common action the wrong one.
    """
    raw = (value or "").strip()
    if match := _ID_IN_URL.search(raw):
        return match.group(1)
    if _BARE_ID.match(raw):
        return raw
    raise ValueError(
        f"{raw!r} is neither a spreadsheet id nor a Google Sheets URL. Copy the "
        "address of the sheet from your browser."
    )


def _credentials(config: dict[str, Any]) -> Any:
    """Service-account credentials from the pasted key, scoped to reading."""
    from google.oauth2 import service_account

    raw = str(config.get("key_file") or "").strip()
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"the key file is not valid JSON: {e}") from e
    return service_account.Credentials.from_service_account_info(
        info, scopes=list(_SCOPES)
    )


def _session(config: dict[str, Any]) -> Any:
    """An HTTP session that refreshes the access token as needed."""
    from google.auth.transport.requests import AuthorizedSession

    return AuthorizedSession(_credentials(config))


def _get(session: Any, url: str, **params: Any) -> dict[str, Any]:
    """One API call, with the response's own error message kept when it fails.

    Google explains refusals in the body -- which sheet, which permission -- and the
    status line does not. Raising on the status alone would discard the useful half.
    """
    response = session.get(url, params=params, timeout=60)
    if response.status_code >= 400:
        try:
            detail = response.json()["error"]["message"]
        except Exception:
            detail = response.text[:300]
        raise ValueError(f"Google Sheets returned {response.status_code}: {detail}")
    result: dict[str, Any] = response.json()
    return result


def _headers(first_row: list[Any]) -> list[str]:
    """Column names from the first row, made usable and unique.

    A blank header becomes a positional name, because a column with no name still holds
    data. A repeated header is suffixed rather than allowed to collide, which would
    otherwise drop every column but the last of that name.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, cell in enumerate(first_row):
        name = str(cell).strip() if cell not in (None, "") else f"column_{index + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}_{count + 1}")
    return names


def _column_types(names: list[str], rows: list[list[Any]]) -> dict[str, pa.DataType]:
    """One Arrow type per column, from the values actually present.

    Decided over the first window rather than the whole sheet, which is the same bargain
    every schemaless source here makes: a cheap read that is right for consistent data,
    and text when it is not.
    """
    types: dict[str, pa.DataType] = {}
    for index, name in enumerate(names):
        seen = {
            type(row[index]).__name__
            for row in rows
            if index < len(row) and row[index] not in (None, "")
        }
        if seen == {"int"}:
            types[name] = pa.int64()
        elif seen and seen <= {"int", "float"}:
            types[name] = pa.float64()
        elif seen == {"bool"}:
            types[name] = pa.bool_()
        else:
            types[name] = pa.string()
    return types


def _cell(value: Any, kind: pa.DataType) -> Any:
    """One cell, coerced to its column's type or rendered as text."""
    if value is None or value == "":
        return None
    if kind == pa.string() and not isinstance(value, str):
        return json.dumps(value) if isinstance(value, dict | list) else str(value)
    return value


@SourceRegistry.register
class GoogleSheetsSource(SimpleSource):
    """A Google spreadsheet, one table per tab."""

    supports_column_selection = True

    @property
    def source_type(self) -> str:
        """The registry key."""
        return "google_sheets"

    @property
    def config(self) -> SourceConfig:
        """The spreadsheet, and a service-account key that has been given access."""
        return SourceConfig(
            name="google_sheets",
            label="Google Sheets",
            category="File storage",
            icon="📊",
            caption="Sync each tab of a Google spreadsheet as a table.",
            docs_url="https://developers.google.com/workspace/guides/create-credentials#service-account",
            fields=[
                SourceField(
                    name="spreadsheet",
                    label="Spreadsheet URL or ID",
                    placeholder="https://docs.google.com/spreadsheets/d/1AbC.../edit",
                    caption="Paste the address of the sheet.",
                ),
                SourceField(
                    name="key_file",
                    label="Google Cloud JSON key file",
                    type="textarea",
                    placeholder='{"type": "service_account", ...}',
                    caption="Then share the spreadsheet with the `client_email` in "
                    "that key, exactly as you would share it with a colleague. "
                    "Without that, the key is valid and the sheet is invisible.",
                ),
            ],
        )

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the fields, then open the spreadsheet the key was given."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            identifier = spreadsheet_id(str(config.get("spreadsheet") or ""))
            _get(_session(config), f"{_API}/{identifier}", fields="properties.title")
        except ValueError as e:
            # Already phrased for whoever is filling in the form.
            return False, [str(e)]
        except Exception as e:
            return False, [f"Could not open the spreadsheet: {e}"]
        return True, []

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Every tab in the spreadsheet.

        No incremental fields: a spreadsheet has no notion of when a row was added, and
        a cell can be edited in place without anything recording that it was. A full
        refresh is the only reading of a sheet that is ever correct.
        """
        identifier = spreadsheet_id(str(config.get("spreadsheet") or ""))
        payload = _get(
            _session(config), f"{_API}/{identifier}", fields="sheets.properties.title"
        )
        return [
            SourceSchema(name=sheet["properties"]["title"], incremental_fields=[])
            for sheet in payload.get("sheets", [])
        ]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Read one tab in row windows, the first row naming the columns."""
        session = _session(inputs.config)
        identifier = spreadsheet_id(str(inputs.config.get("spreadsheet") or ""))
        tab = inputs.schema

        names: list[str] = []
        types: dict[str, pa.DataType] = {}
        schema: pa.Schema | None = None
        start = 1
        while True:
            end = start + _WINDOW - 1
            payload = _get(
                session,
                f"{_API}/{identifier}/values/{_range(tab, start, end)}",
                valueRenderOption="UNFORMATTED_VALUE",
                dateTimeRenderOption="FORMATTED_STRING",
                majorDimension="ROWS",
            )
            rows: list[list[Any]] = payload.get("values", [])
            if not rows:
                break

            if not names:
                names = _headers(rows[0])
                rows = rows[1:]
                types = _column_types(names, rows)
                schema = pa.schema([pa.field(n, types[n]) for n in names])
            if rows and schema is not None:
                yield pa.Table.from_pylist(
                    [_record(row, names, types) for row in rows], schema=schema
                )
            if len(payload.get("values", [])) < _WINDOW:
                # A short window is the end of the sheet. Asking again would return an
                # empty result and cost another round trip to learn the same thing.
                break
            start = end + 1


def _range(tab: str, start: int, end: int) -> str:
    """An A1 range covering one window of rows in ``tab``.

    The tab name is quoted and its own quotes doubled, because a sheet may be called
    ``Q1 'raw'`` and an unquoted range would be parsed as something else entirely.
    """
    from urllib.parse import quote

    escaped = tab.replace("'", "''")
    return quote(f"'{escaped}'!{start}:{end}", safe="")


def _record(
    row: list[Any], names: list[str], types: dict[str, pa.DataType]
) -> dict[str, Any]:
    """One sheet row as a record, padded to the header width.

    Sheets omits trailing empty cells rather than padding, so a short row is a row whose
    last columns are blank -- not a row that is missing them.
    """
    return {
        name: _cell(row[index] if index < len(row) else None, types[name])
        for index, name in enumerate(names)
    }
