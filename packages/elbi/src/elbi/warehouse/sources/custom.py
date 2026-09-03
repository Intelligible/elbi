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
        """Structurally validate the manifest and its base URL."""
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
        """Run the chosen resource through dlt's ``rest_api``, batched to Arrow."""
        from ._dlt import rest_api_source

        manifest = self._assemble(inputs.config)
        source = rest_api_source(manifest)
        if inputs.schema not in source.resources:
            raise ValueError(f"resource {inputs.schema!r} not found in the manifest")
        batch: list[dict[str, Any]] = []
        for record in source.resources[inputs.schema]:
            if isinstance(record, dict):
                batch.append(record)
            if len(batch) >= _BATCH:
                yield pa.Table.from_pylist(batch)
                batch = []
        if batch:
            yield pa.Table.from_pylist(batch)
