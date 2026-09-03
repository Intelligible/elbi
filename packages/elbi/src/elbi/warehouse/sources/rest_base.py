"""Shared base for token/key REST connectors, run on dlt's ``rest_api`` engine.

A connector declares its base URL, auth, and resource list (path + where the records
live); the base assembles a dltHub ``RESTAPIConfig`` and lets dlt handle auth,
pagination, and JSON extraction. Subclasses only describe the API; they write no
request or pagination code.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs

_BATCH = 10_000


class RestApiConnector(SimpleSource):
    """Declarative REST connector: describe the API, dlt does the fetching."""

    # --- per-connector metadata (set by subclasses) ---------------------------
    key: ClassVar[str] = ""
    label: ClassVar[str] = ""
    category: ClassVar[str] = ""
    icon: ClassVar[str] = ""
    caption: ClassVar[str] = ""
    docs_url: ClassVar[str] = ""
    fields_: ClassVar[tuple[SourceField, ...]] = ()
    #: Each: {"name", "path", "data_selector" (dot-path), optional "params", and for an
    #: API that lists over POST rather than GET, "method" and "json".
    resources_: ClassVar[tuple[dict[str, Any], ...]] = ()

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return self.key

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name=self.key,
            label=self.label,
            category=self.category,
            icon=self.icon,
            caption=self.caption,
            docs_url=self.docs_url,
            fields=list(self.fields_),
        )

    # --- hooks a subclass implements -------------------------------------------
    def _base_url(self, config: dict[str, Any]) -> str:
        raise NotImplementedError

    def _auth(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """A dlt auth config, e.g. ``{"type": "bearer", "token": ...}``."""
        return None

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        return {}

    def _paginator(self, config: dict[str, Any]) -> Any:
        """A dlt paginator config, or ``None`` to let dlt auto-detect."""
        return None

    def _resources(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """The syncable resources; defaults to the class list (override if dynamic)."""
        return list(self.resources_)

    # --- assembly + the Source contract ----------------------------------------
    def _manifest(self, config: dict[str, Any]) -> dict[str, Any]:
        client: dict[str, Any] = {"base_url": self._base_url(config)}
        if auth := self._auth(config):
            client["auth"] = auth
        if headers := self._headers(config):
            client["headers"] = headers
        if paginator := self._paginator(config):
            client["paginator"] = paginator
        resources = []
        for r in self._resources(config):
            endpoint: dict[str, Any] = {"path": r["path"]}
            if r.get("data_selector"):
                endpoint["data_selector"] = r["data_selector"]
            if r.get("params"):
                endpoint["params"] = r["params"]
            if method := r.get("method"):
                # Notion lists its content over POST, with the filter in the body. That
                # is a listing like any other, so it belongs here rather than in a
                # connector that has to write its own requests to reach it.
                endpoint["method"] = method
            if r.get("json") is not None:
                endpoint["json"] = r["json"]
            resources.append({"name": r["name"], "endpoint": endpoint})
        return {"client": client, "resources": resources}

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the required fields resolve into a base URL and resource list."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            self._base_url(config)
            if not self._resources(config):
                return False, ["This source has no resources to sync."]
            return True, []
        except Exception as e:  # surface a config error to the user, any type
            return False, [f"Invalid configuration: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """One table per resource."""
        return [SourceSchema(name=r["name"]) for r in self._resources(config)]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Run the chosen resource through dlt's ``rest_api``, batched to Arrow."""
        from ._dlt import rest_api_source

        source = rest_api_source(self._manifest(inputs.config))
        if inputs.schema not in source.resources:
            raise ValueError(f"resource {inputs.schema!r} not found in the source")
        batch: list[dict[str, Any]] = []
        for record in source.resources[inputs.schema]:
            if isinstance(record, dict):
                batch.append(record)
            if len(batch) >= _BATCH:
                yield pa.Table.from_pylist(batch)
                batch = []
        if batch:
            yield pa.Table.from_pylist(batch)
