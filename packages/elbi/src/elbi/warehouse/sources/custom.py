"""Custom REST source: a manifest-driven connector for any REST API.

The user provides a JSON **manifest** in the dltHub ``RESTAPIConfig`` shape (a
``client`` with base URL/auth/pagination and a list of ``resources``), and secrets are
supplied in separate fields so they can be redacted. The manifest is executed by dlt's
``rest_api`` source, the reference implementation of that config, so any manifest
written against the published shape works here.

Auth is whatever static credential the manifest declares (bearer, api key, basic, or
OAuth2 client-credentials). There is no token-refresh table and no egress filter.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_BATCH = 10_000


def _resource_name(resource: Any) -> str | None:
    """A dlt resource entry is either a bare name or a dict with a ``name``."""
    if isinstance(resource, str):
        return resource
    if isinstance(resource, dict):
        name = resource.get("name")
        return name if isinstance(name, str) else None
    return None


#: How long a connection test may take before it is reported as hung. Generous enough
#: for a slow first page, short enough that an operator is told rather than left
#: watching a spinner. It bounds the walk itself, not only the wait on it.
_PROBE_TIMEOUT = 20.0

#: Slack on top of that, for the walk to unwind and the thread to finish after its own
#: deadline has passed. Only a walk that is stuck before any page arrives uses it.
_PROBE_GRACE = 5.0


def _declares_paginator(manifest: dict[str, Any]) -> bool:
    """Whether a paginator is declared on the client or on any resource."""
    if manifest.get("client", {}).get("paginator"):
        return True
    return any(
        isinstance(resource, dict)
        and isinstance(resource.get("endpoint"), dict)
        and resource["endpoint"].get("paginator")
        for resource in manifest.get("resources", [])
    )


#: The sibling key a columnar response names its fields in (PostHog's query API, and
#: SQL-over-HTTP APIs generally): ``{"columns": [...], "results": [[...], ...]}``.
_COLUMNS_KEY = "columns"

#: Flags an API sets when it answered with only part of the result.
_TRUNCATION_FLAGS = ("hasMore", "has_more")

#: Paginator types that never ask for a second page, or only guess at one.
_NON_PAGINATORS = frozenset({"single_page", "auto"})


def _endpoints(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every endpoint config in the manifest, defaults included."""
    defaults = manifest.get("resource_defaults")
    if isinstance(defaults, dict) and isinstance(defaults.get("endpoint"), dict):
        yield defaults["endpoint"]
    for resource in manifest.get("resources", []):
        if isinstance(resource, dict) and isinstance(resource.get("endpoint"), dict):
            yield resource["endpoint"]


def _fetches_more(manifest: dict[str, Any]) -> bool:
    """Whether any declared paginator could fetch a page past the first.

    Manifest-wide on purpose: one client session serves every resource, including the
    parents a dependent resource reads, so a ``has_more`` meant for a real paginator
    anywhere must not trip the truncation check.
    """
    declared = [manifest.get("client", {}).get("paginator")]
    declared += [endpoint.get("paginator") for endpoint in _endpoints(manifest)]
    for paginator in declared:
        kind = paginator.get("type") if isinstance(paginator, dict) else paginator
        if paginator and kind not in _NON_PAGINATORS:
            return True
    return False


def _selector_paths(manifest: dict[str, Any]) -> list[tuple[str, ...]]:
    """The declared ``data_selector`` values that are plain dotted key paths."""
    paths = []
    for endpoint in _endpoints(manifest):
        selector = endpoint.get("data_selector")
        if not isinstance(selector, str):
            continue
        keys = tuple(selector.removeprefix("$").lstrip(".").split("."))
        if all(key and key.isidentifier() for key in keys):
            paths.append(keys)
    return paths


def _name_rows(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> bool:
    """Zip a sibling ``columns`` list onto every row array found at ``paths``.

    Strict: a row of the wrong width is an error, because zipping it short would land
    values under the wrong names. Returns whether anything was rewritten.
    """
    rewritten = False
    for path in paths:
        parent: Any = payload
        for key in path[:-1]:
            parent = parent.get(key) if isinstance(parent, dict) else None
        if not isinstance(parent, dict):
            continue
        rows, columns = parent.get(path[-1]), parent.get(_COLUMNS_KEY)
        if not (isinstance(rows, list) and rows and isinstance(columns, list)):
            continue
        if not all(isinstance(name, str) for name in columns):
            continue
        if not all(isinstance(row, list) for row in rows):
            continue
        records = []
        for index, row in enumerate(rows):
            if len(row) != len(columns):
                raise ValueError(
                    f"row {index} of {path[-1]!r} has {len(row)} values but "
                    f"{_COLUMNS_KEY!r} names {len(columns)} fields"
                )
            records.append(dict(zip(columns, row, strict=True)))
        parent[path[-1]] = records
        rewritten = True
    return rewritten


def _inspecting_session(manifest: dict[str, Any]) -> Any:
    """The dlt session, with each response checked before ``rest_api`` reads it.

    dlt keeps only the record list a response holds, so anything beside it (the
    field names of a columnar answer, a flag saying the answer is partial) has to be
    read here, while the whole body is still in hand. Wrapping ``send`` runs after
    dlt's response hooks, so its error handling and redaction of failed requests stay
    untouched.
    """
    from ._dlt import detect_records_path, rest_api_session

    session = rest_api_session()
    send = session.send
    selectors = _selector_paths(manifest)
    guard = not _fetches_more(manifest)

    def inspected(request: Any, **kwargs: Any) -> Any:
        response = send(request, **kwargs)
        content = response.content or b""
        flagged = guard and any(f'"{f}"'.encode() in content for f in _TRUNCATION_FLAGS)
        if not response.ok or not (flagged or f'"{_COLUMNS_KEY}"'.encode() in content):
            return response
        try:
            payload = response.json()
        except ValueError:
            return response
        if not isinstance(payload, dict):
            return response
        for flag in _TRUNCATION_FLAGS:
            if guard and payload.get(flag) is True:
                raise ValueError(
                    f"The response says {flag}: true, so it holds only part of the "
                    "result, and no paginator is declared that would fetch the rest. "
                    "Declare one, or narrow the request so it fits in one response "
                    "(a PostHog HogQL query needs a LIMIT above its row count)."
                )
        if _name_rows(payload, [*selectors, detect_records_path(payload)]):
            response._content = json.dumps(payload).encode()
        return response

    session.send = inspected
    return session


def _inspected(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest with a client session that inspects every response."""
    client = {**manifest["client"], "session": _inspecting_session(manifest)}
    return {**manifest, "client": client}


def _as_record(record: Any, resource: str) -> dict[str, Any]:
    """The record itself, or an error: a non-object is never dropped quietly."""
    if isinstance(record, dict):
        return record
    raise ValueError(
        f"Resource {resource!r} returned a {type(record).__name__} where a record (a "
        "JSON object) was expected. A columnar response needs its field names in a "
        f"sibling {_COLUMNS_KEY!r} list; otherwise point `data_selector` at a list of "
        "objects."
    )


@SourceRegistry.register
class CustomSource(SimpleSource):
    """Sync any REST API from a dltHub-shaped manifest (a Custom REST source)."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "custom"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="custom",
            label="Custom REST source",
            category="Engineering & monitoring",
            icon="🧩",
            release_status="alpha",
            caption=(
                "Set up a source from a custom manifest. Define a REST API with a "
                "manifest in the dltHub REST shape: a client (base URL, auth, "
                "pagination) and a list of resources."
            ),
            docs_url="https://dlthub.com/docs/dlt-ecosystem/verified-sources/rest_api/",
            fields=[
                SourceField(
                    name="manifest_json",
                    label="Manifest (JSON)",
                    type="textarea",
                    placeholder='{"client": {"base_url": "https://api.example.com"}, '
                    '"resources": ["items"]}',
                ),
                # One credential per sync, selected by the manifest's client.auth.type.
                SourceField(
                    name="auth_token",
                    label="Bearer token",
                    type="password",
                    required=False,
                ),
                SourceField(
                    name="auth_api_key",
                    label="API key",
                    type="password",
                    required=False,
                ),
                SourceField(
                    name="auth_password",
                    label="Auth password",
                    type="password",
                    required=False,
                ),
                SourceField(
                    name="auth_oauth2_client_secret",
                    label="OAuth2 client secret",
                    type="password",
                    required=False,
                ),
                SourceField(
                    name="auth_oauth2_refresh_token",
                    label="OAuth2 refresh token",
                    type="password",
                    required=False,
                ),
            ],
        )

    def _assemble(self, config: dict[str, Any]) -> dict[str, Any]:
        """Parse the manifest and inject the auth secrets into ``client.auth``.

        The manifest holds the ``RESTAPIConfig`` structure with no credential values;
        the secrets live in separate ``auth_*`` fields. This rebuilds the full config
        dlt's ``rest_api`` consumes, picking the secret by the manifest's auth type.
        """
        try:
            manifest = json.loads(config["manifest_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Manifest is not valid JSON: {exc}") from exc
        if not isinstance(manifest, dict) or "client" not in manifest:
            raise ValueError("Manifest must be an object with a 'client' key")

        auth = manifest["client"].get("auth")
        if isinstance(auth, dict):
            kind = str(auth.get("type", "")).lower()
            if "bearer" in kind and config.get("auth_token"):
                auth["token"] = config["auth_token"]
            if "api_key" in kind and config.get("auth_api_key"):
                auth["api_key"] = config["auth_api_key"]
            if ("basic" in kind or kind == "http") and config.get("auth_password"):
                auth["password"] = config["auth_password"]
            if "oauth2" in kind:
                if config.get("auth_oauth2_client_secret"):
                    auth["client_secret"] = config["auth_oauth2_client_secret"]
                if config.get("auth_oauth2_refresh_token"):
                    auth["refresh_token"] = config["auth_oauth2_refresh_token"]
        return manifest

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the manifest structurally, then prove it by fetching one record.

        The structural checks alone let a manifest through that cannot sync. The case
        that motivated the probe: a manifest with no ``paginator`` is structurally
        perfect, and dlt then guesses one — for an endpoint answering in a single
        response the guess chases a next page that never arrives, and the sync sits in
        ``pending`` with no error, no timeout and nothing in the UI to act on.

        Nothing short of running the extraction catches that, because auth, the URL and
        the resource are all fine. So the probe runs the real resource through the real
        ``rest_api``, bounded by a deadline, and stops at the first record.
        """
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            manifest = self._assemble(config)
        except ValueError as exc:
            return False, [str(exc)]
        base_url = manifest["client"].get("base_url")
        if not base_url or urlparse(str(base_url)).scheme not in ("http", "https"):
            return False, ["client.base_url must be an http(s) URL"]
        if not manifest.get("resources"):
            return False, ["Manifest must define at least one resource"]
        return self._probe(manifest)

    def _probe(self, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
        """Fetch one record from the first resource, or say why it could not.

        The deadline is on the walk, not only on the wait. A guessed paginator loops
        over responses that each return promptly, so no per-request timeout ever fires,
        and a worker thread abandoned when the join expires keeps asking for the next
        page — hundreds a second, for the life of a server process that runs for months.
        So the resource carries dlt's own stop: ``add_limit(max_time=...)`` closes its
        generator at the first page past the deadline, listener or not.

        Time bounds the walk rather than ``max_items`` because ``max_items`` counts
        pages, and a walk cut short at page N with no record to show reads exactly like
        an endpoint that is simply empty. A spent budget tells those two apart.

        The thread remains as the backstop for the case the limit cannot reach: a
        request that hangs before any page arrives, so nothing flows through the pipe
        for the limit to close on. That is one stuck request, not an unbounded walk.
        """
        resources = manifest.get("resources", [])
        name = _resource_name(resources[0]) if resources else None
        if not name:
            return False, ["The first resource has no name"]

        outcome: dict[str, Any] = {}

        def run() -> None:
            started = time.monotonic()
            try:
                from ._dlt import rest_api_source

                source = rest_api_source(_inspected(manifest))
                resource = source.resources[name].add_limit(max_time=_PROBE_TIMEOUT)
                for record in resource:
                    _as_record(record, name)
                    outcome["record"] = True
                    break
            except Exception as exc:  # any failure is the operator's to see
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                outcome["elapsed"] = time.monotonic() - started

        worker = threading.Thread(target=run, daemon=True, name=f"probe-{name}")
        worker.start()
        worker.join(_PROBE_TIMEOUT + _PROBE_GRACE)

        if "error" in outcome:
            return False, [f"Could not fetch from {name!r}. {outcome['error']}"]
        if outcome.get("record"):
            return True, []
        if worker.is_alive() or outcome.get("elapsed", 0.0) >= _PROBE_TIMEOUT:
            hint = (
                " No `paginator` is declared, so dlt guessed one. Declare it explicitly"
                ' — `"paginator": {"type": "single_page"}` for an endpoint that'
                " answers in one response."
                if not _declares_paginator(manifest)
                else " Check the `paginator` settings; a wrong one loops forever."
            )
            return False, [
                f"Fetching from {name!r} did not finish within {_PROBE_TIMEOUT:.0f}s."
                + hint
            ]
        # Nothing to fetch, but the endpoint answered and the walk ended on its own.
        return True, []

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """One table per resource declared in the manifest."""
        manifest = self._assemble(config)
        names = [
            name
            for resource in manifest.get("resources", [])
            if (name := _resource_name(resource))
        ]
        return [SourceSchema(name=name) for name in names]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Run the chosen resource through dlt's ``rest_api``, batched to Arrow.

        A record that is not an object fails the sync rather than being skipped, so an
        API whose shape the manifest does not describe cannot report an empty success.
        """
        from ._dlt import rest_api_source

        manifest = self._assemble(inputs.config)
        source = rest_api_source(_inspected(manifest))
        if inputs.schema not in source.resources:
            raise ValueError(f"resource {inputs.schema!r} not found in the manifest")
        batch: list[dict[str, Any]] = []
        for record in source.resources[inputs.schema]:
            batch.append(_as_record(record, inputs.schema))
            if len(batch) >= _BATCH:
                yield pa.Table.from_pylist(batch)
                batch = []
        if batch:
            yield pa.Table.from_pylist(batch)
