"""PostHog connector: product-analytics objects and events over the PostHog API.

Syncs a project's objects -- persons, cohorts, feature flags, insights, experiments,
actions, annotations, surveys -- from the paginated REST endpoints, and, when it is
asked for, the raw event stream through HogQL. The host is a field rather than a
constant because the same API serves US Cloud, EU Cloud, and every self-hosted
deployment, and a key is only ever valid against the one that issued it.

PostHog documents ``/query`` as "not a supported export mechanism", and says third-party
connectors should use its batch exports instead. That is why ``events`` is off by
default, is filtered server-side on ``timestamp``, and is paged the way PostHog
prescribes for that endpoint rather than with ``OFFSET``, which it rejects outright.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_DEFAULT_HOST = "https://us.posthog.com"
_OBJECTS = (
    "persons",
    "cohorts",
    "feature_flags",
    "insights",
    "experiments",
    "actions",
    "annotations",
    "surveys",
)
_EVENTS = "events"
_PAGE = 100
_EVENT_PAGE = 1000
_TIMEOUT = 60.0

_EVENT_SELECT = (
    "SELECT uuid, event, timestamp, distinct_id, person_id, properties, elements_chain "
    "FROM events "
)
_EVENT_ORDER = f"ORDER BY timestamp LIMIT {_EVENT_PAGE}"
# Two constants rather than one f-string: the cursor travels in ``values`` as a bound
# constant, so a stored timestamp can never be read as HogQL.
_EVENT_QUERY = _EVENT_SELECT + _EVENT_ORDER
_EVENT_QUERY_SINCE = _EVENT_SELECT + "WHERE timestamp > {since} " + _EVENT_ORDER


def _host(config: dict[str, Any]) -> str:
    return str(config.get("host") or _DEFAULT_HOST).strip().rstrip("/")


def _url(config: dict[str, Any], path: str) -> str:
    return f"{_host(config)}/api/projects/{config['project_id']}/{path}"


def _records(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a HogQL response into records by zipping ``columns`` onto each row.

    The query endpoint answers columnar: ``results`` is a list of arrays and the names
    live once in ``columns``. Read as though the rows were objects it yields nothing at
    all, without an error, which is the one way this connector could look like it works
    and not.
    """
    columns = list(body.get("columns") or [])
    if not columns:
        # Without names there is nothing to key the values by, and a row of unnamed
        # values would land as a row of nothing at all.
        return []
    return [
        dict(zip(columns, row, strict=False))
        for row in body.get("results") or []
        if isinstance(row, list)
    ]


def _clickhouse_time(value: Any) -> str:
    """Render a cursor as the space-separated UTC datetime ClickHouse parses.

    The API hands timestamps back in ISO-8601 with a ``T`` and a zone, and comparing
    that form against the events table is not reliable, so the round trip goes through
    a real datetime rather than through string surgery.
    """
    text = str(value).strip()
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")


@SourceRegistry.register
class PostHogSource(SimpleSource):
    """Sync PostHog persons, cohorts, feature flags, insights, and events."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "posthog"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="posthog",
            label="PostHog",
            category="Analytics",
            icon="🦔",
            caption=(
                "Sync persons, cohorts, feature flags, insights, experiments, and "
                "events from PostHog Cloud or a self-hosted deployment."
            ),
            docs_url="https://posthog.com/docs/api",
            fields=[
                SourceField(
                    name="host",
                    label="Host",
                    default=_DEFAULT_HOST,
                    placeholder=_DEFAULT_HOST,
                    caption=(
                        "https://us.posthog.com or https://eu.posthog.com for Cloud, "
                        "your own URL if you self-host. A key is valid only against "
                        "the deployment that issued it."
                    ),
                ),
                SourceField(
                    name="api_key",
                    label="Personal API key",
                    type="password",
                    placeholder="phx_...",
                    caption=(
                        "Settings → Personal API keys. Scope it to this project, with "
                        "read on the resources you sync and Query Read for events."
                    ),
                ),
                SourceField(
                    name="project_id",
                    label="Project ID",
                    placeholder="12345",
                    caption="The number in the project's URL, and in project settings.",
                ),
            ],
        )

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        return {"Authorization": f"Bearer {config['api_key']}"}

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Confirm the key reaches this project with a one-record probe on cohorts.

        Against a resource inside the project rather than the project list, because a
        key that is valid but scoped to a different project is the mistake worth
        catching at save time: it looks exactly like a working key until a sync 403s.
        """
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            import httpx

            response = httpx.get(
                _url(config, "cohorts/"),
                headers=self._headers(config),
                params={"limit": 1},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return True, []
        except Exception as e:  # surface the auth/fetch error to the user, any type
            return False, [f"Could not authenticate with PostHog: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """The project's objects, plus the event stream behind an explicit choice.

        Only ``events`` offers a cursor. None of the object endpoints accepts a
        modified-since filter, so a cursor on one of them would page the whole list
        anyway and then drop the rows it could not exclude -- and those lists run to
        hundreds of rows, where a full refresh costs nothing to begin with. ``events``
        runs to hundreds of millions, and HogQL does filter it server-side.
        """
        return [SourceSchema(name=obj) for obj in _OBJECTS] + [
            SourceSchema(
                name=_EVENTS,
                incremental_fields=["timestamp"],
                default_selected=False,
            )
        ]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Yield one batch per page, from HogQL for events and REST for the rest."""
        if inputs.schema == _EVENTS:
            yield from self._extract_events(inputs)
        else:
            yield from self._extract_objects(inputs)

    def _extract_objects(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Page a list endpoint by following the absolute ``next`` link it returns."""
        import httpx

        headers = self._headers(inputs.config)
        url: str | None = _url(inputs.config, f"{inputs.schema}/")
        params: dict[str, Any] | None = {"limit": _PAGE}
        with httpx.Client(timeout=_TIMEOUT) as http:
            while url:
                response = http.get(url, headers=headers, params=params)
                response.raise_for_status()
                body = response.json()
                rows = [r for r in body.get("results") or [] if isinstance(r, dict)]
                if rows:
                    yield pa.Table.from_pylist(rows)
                # The link already carries its own limit and offset; sending ours
                # alongside would restart the walk at the first page forever.
                url = body.get("next") if rows else None
                params = None

    def _extract_events(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Walk the event stream in timestamp order, a page at a time.

        Each page resumes from the last timestamp of the one before, which is the
        keyset paging PostHog documents for this endpoint -- ``OFFSET`` is rejected
        with a 400 for personal API keys. It also means an incremental run is the same
        walk started later, so there is one code path and not two.
        """
        import httpx

        since = (
            _clickhouse_time(inputs.incremental_since)
            if inputs.incremental_field == "timestamp"
            and inputs.incremental_since is not None
            else None
        )
        headers = self._headers(inputs.config)
        url = _url(inputs.config, "query/")
        with httpx.Client(timeout=_TIMEOUT) as http:
            while True:
                response = http.post(url, headers=headers, json=_query(since))
                response.raise_for_status()
                rows = _records(response.json())
                if not rows:
                    break
                yield pa.Table.from_pylist(rows)
                if len(rows) < _EVENT_PAGE:
                    break
                since = _clickhouse_time(rows[-1]["timestamp"])


def _query(since: str | None) -> dict[str, Any]:
    """One page of events as a query body.

    ``force_blocking`` because the endpoint caches by default, and the first page of a
    full refresh is the same request every run: served from cache, a scheduled sync
    would keep landing the snapshot it landed the first time. ``name`` is what PostHog
    shows in its own query log, so a slow sync is identifiable from their side.
    """
    query: dict[str, Any] = {"kind": "HogQLQuery", "query": _EVENT_QUERY}
    if since is not None:
        query["query"] = _EVENT_QUERY_SINCE
        query["values"] = {"since": since}
    return {"query": query, "refresh": "force_blocking", "name": "elbi warehouse sync"}
