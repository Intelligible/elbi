"""Salesforce connector: CRM objects over the Salesforce REST API.

Syncs the common objects (Account, Contact, Opportunity, Lead) via SOQL ``FIELDS(ALL)``
queries, paging with ``nextRecordsUrl``. Auth is an instance URL plus a bearer access
token, from an OAuth session or a connected app. That needs no brokered integration on
our side, so it works the same whether the app runs on a laptop or a server.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_API_VERSION = "v60.0"
_TIMEOUT = 30.0
# Static SOQL literals per object (FIELDS(ALL) caps at 200 rows/page). Not built from a
# variable: the object name is an allow-listed dict key, so there's no dynamic SQL.
_SOQL = {
    "Account": "SELECT FIELDS(ALL) FROM Account LIMIT 200",
    "Contact": "SELECT FIELDS(ALL) FROM Contact LIMIT 200",
    "Opportunity": "SELECT FIELDS(ALL) FROM Opportunity LIMIT 200",
    "Lead": "SELECT FIELDS(ALL) FROM Lead LIMIT 200",
}
_OBJECTS = tuple(_SOQL)


@SourceRegistry.register
class SalesforceSource(SimpleSource):
    """Sync Salesforce accounts, contacts, opportunities, and leads."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "salesforce"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="salesforce",
            label="Salesforce",
            category="CRM",
            icon="☁️",
            caption="Sync accounts, contacts, opportunities, and leads.",
            docs_url="https://developer.salesforce.com/docs/apis",
            fields=[
                SourceField(
                    name="instance_url",
                    label="Instance URL",
                    placeholder="https://my-org.my.salesforce.com",
                ),
                SourceField(name="access_token", label="Access token", type="password"),
            ],
        )

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        return {"Authorization": f"Bearer {config['access_token']}"}

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Confirm the token works against the API-version resource listing."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            import httpx

            base = config["instance_url"].rstrip("/")
            response = httpx.get(
                f"{base}/services/data/{_API_VERSION}/limits",
                headers=self._headers(config),
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return True, []
        except Exception as e:  # surface the auth/fetch error to the user, any type
            return False, [f"Could not authenticate with Salesforce: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Salesforce's common CRM objects."""
        return [SourceSchema(name=obj) for obj in _OBJECTS]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Run a ``FIELDS(ALL)`` SOQL query, paging through ``nextRecordsUrl``."""
        import httpx

        base = inputs.config["instance_url"].rstrip("/")
        headers = self._headers(inputs.config)
        url: str | None = f"{base}/services/data/{_API_VERSION}/query"
        params: dict[str, Any] | None = {"q": _SOQL[inputs.schema]}
        with httpx.Client(timeout=_TIMEOUT) as http:
            while url:
                response = http.get(url, headers=headers, params=params)
                response.raise_for_status()
                body = response.json()
                rows = [
                    {k: v for k, v in record.items() if k != "attributes"}
                    for record in body.get("records", [])
                ]
                if rows:
                    yield pa.Table.from_pylist(rows)
                nxt = body.get("nextRecordsUrl")
                url = f"{base}{nxt}" if nxt and rows else None
                params = None
