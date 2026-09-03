"""HubSpot connector: CRM objects over the HubSpot CRM v3 API.

Syncs the common CRM objects (contacts, companies, deals, tickets) as tables, paging the
list endpoints via the ``after`` cursor. Each object's ``properties`` map is flattened
onto the row alongside its id and timestamps. Authenticated with a private-app token.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_BASE = "https://api.hubapi.com"
_OBJECTS = ("contacts", "companies", "deals", "tickets")
_PAGE = 100
_TIMEOUT = 30.0


def _flatten(obj: dict[str, Any]) -> dict[str, Any]:
    """Merge a CRM object's ``properties`` onto the top level with its id/timestamps."""
    row = dict(obj.get("properties") or {})
    row["id"] = obj.get("id")
    row["createdAt"] = obj.get("createdAt")
    row["updatedAt"] = obj.get("updatedAt")
    return row


@SourceRegistry.register
class HubSpotSource(SimpleSource):
    """Sync HubSpot CRM objects (contacts, companies, deals, tickets)."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "hubspot"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="hubspot",
            label="HubSpot",
            category="CRM",
            icon="🧲",
            caption="Sync contacts, companies, deals, and tickets from HubSpot.",
            docs_url="https://developers.hubspot.com/docs/api/crm/understanding-the-crm",
            fields=[
                SourceField(
                    name="access_token",
                    label="Private app token",
                    type="password",
                    placeholder="pat-...",
                    caption="Create a private app with CRM read scopes for its token.",
                ),
            ],
        )

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        return {"Authorization": f"Bearer {config['access_token']}"}

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Confirm the token works with a one-record probe against ``/contacts``."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            import httpx

            response = httpx.get(
                f"{_BASE}/crm/v3/objects/contacts",
                headers=self._headers(config),
                params={"limit": 1},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return True, []
        except Exception as e:  # surface the auth/fetch error to the user, any type
            return False, [f"Could not authenticate with HubSpot: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """HubSpot's common CRM objects."""
        return [SourceSchema(name=obj) for obj in _OBJECTS]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Page through a CRM object's list endpoint via the ``after`` cursor."""
        import httpx

        headers = self._headers(inputs.config)
        url = f"{_BASE}/crm/v3/objects/{inputs.schema}"
        after: str | None = None
        with httpx.Client(timeout=_TIMEOUT) as http:
            while True:
                params: dict[str, Any] = {"limit": _PAGE, "archived": "false"}
                if after:
                    params["after"] = after
                response = http.get(url, headers=headers, params=params)
                response.raise_for_status()
                body = response.json()
                rows = body.get("results", [])
                if rows:
                    yield pa.Table.from_pylist([_flatten(r) for r in rows])
                after = body.get("paging", {}).get("next", {}).get("after")
                if not after or not rows:
                    break
