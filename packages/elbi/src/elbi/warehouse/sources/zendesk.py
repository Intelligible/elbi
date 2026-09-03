"""Zendesk connector: Support data over the Zendesk REST API v2.

Syncs tickets, users, and organizations as tables, paging via the ``next_page`` link.
Authenticated with an agent email + API token (Zendesk's ``email/token`` basic auth).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_RESOURCES = ("tickets", "users", "organizations")
_PAGE = 100
_TIMEOUT = 30.0


@SourceRegistry.register
class ZendeskSource(SimpleSource):
    """Sync tickets, users, and organizations from Zendesk Support."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "zendesk"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="zendesk",
            label="Zendesk",
            category="Productivity",
            icon="🎧",
            caption="Sync tickets, users, and organizations from Zendesk Support.",
            docs_url="https://developer.zendesk.com/api-reference/",
            fields=[
                SourceField(
                    name="subdomain",
                    label="Zendesk subdomain",
                    placeholder="my-company",
                    caption="The part before .zendesk.com.",
                ),
                SourceField(
                    name="email_address",
                    label="Zendesk email address",
                    placeholder="agent@example.com",
                ),
                SourceField(name="api_key", label="API key", type="password"),
            ],
        )

    def _base(self, config: dict[str, Any]) -> str:
        return f"https://{config['subdomain']}.zendesk.com/api/v2"

    def _auth(self, config: dict[str, Any]) -> tuple[str, str]:
        # Zendesk API-token auth: username is "{email}/token", password is the token.
        return (f"{config['email_address']}/token", config["api_key"])

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Confirm credentials with a probe against ``/tickets``."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            import httpx

            response = httpx.get(
                f"{self._base(config)}/tickets.json",
                auth=self._auth(config),
                params={"page[size]": 1},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return True, []
        except Exception as e:  # surface the auth/fetch error to the user, any type
            return False, [f"Could not authenticate with Zendesk: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Zendesk's common resources."""
        return [SourceSchema(name=resource) for resource in _RESOURCES]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Page through a resource via the ``next_page`` link."""
        import httpx

        auth = self._auth(inputs.config)
        url: str | None = f"{self._base(inputs.config)}/{inputs.schema}.json"
        params: dict[str, Any] | None = {"page[size]": _PAGE}
        with httpx.Client(timeout=_TIMEOUT) as http:
            while url:
                response = http.get(url, auth=auth, params=params)
                response.raise_for_status()
                body = response.json()
                rows = body.get(inputs.schema, [])
                if rows:
                    yield pa.Table.from_pylist(rows)
                url = body.get("next_page") if rows else None
                params = None
