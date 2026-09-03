"""Shopify connector: store data over the Shopify Admin REST API.

Syncs orders, products, and customers as tables, paging via the cursor in the ``Link``
header (``page_info``). Authenticated with an Admin API access token scoped to a shop.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_RESOURCES = ("orders", "products", "customers")
_PAGE = 250
_TIMEOUT = 30.0
_DEFAULT_API_VERSION = "2024-01"
_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


@SourceRegistry.register
class ShopifySource(SimpleSource):
    """Sync orders, products, and customers from a Shopify store."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "shopify"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="shopify",
            label="Shopify",
            category="E-commerce",
            icon="🛍️",
            caption="Sync orders, products, and customers from a Shopify store.",
            docs_url="https://shopify.dev/docs/api/admin-rest",
            fields=[
                SourceField(
                    name="shopify_store_id",
                    label="Store id",
                    placeholder="my-store",
                    caption="The part before .myshopify.com.",
                ),
                SourceField(
                    name="shopify_client_id", label="Client ID", placeholder="client-id"
                ),
                SourceField(
                    name="shopify_client_secret",
                    label="Secret",
                    type="password",
                    placeholder="shpss_...",
                ),
            ],
        )

    def _base(self, config: dict[str, Any]) -> str:
        return f"https://{config['shopify_store_id']}.myshopify.com/admin/api/{_DEFAULT_API_VERSION}"

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        # Private-app access via the client secret as the Admin API token.
        return {"X-Shopify-Access-Token": config["shopify_client_secret"]}

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Confirm the token works with a one-record probe against ``/shop``."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            import httpx

            response = httpx.get(
                f"{self._base(config)}/shop.json",
                headers=self._headers(config),
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return True, []
        except Exception as e:  # surface the auth/fetch error to the user, any type
            return False, [f"Could not authenticate with Shopify: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Shopify's common resources."""
        return [SourceSchema(name=resource) for resource in _RESOURCES]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Page through a resource via the ``Link: rel=next`` cursor header."""
        import httpx

        headers = self._headers(inputs.config)
        url: str | None = f"{self._base(inputs.config)}/{inputs.schema}.json"
        params: dict[str, Any] | None = {"limit": _PAGE}
        with httpx.Client(timeout=_TIMEOUT) as http:
            while url:
                response = http.get(url, headers=headers, params=params)
                response.raise_for_status()
                rows = response.json().get(inputs.schema, [])
                if rows:
                    yield pa.Table.from_pylist(rows)
                match = _NEXT_LINK.search(response.headers.get("Link", ""))
                # The next-page URL already carries page_info; drop our own params.
                url = match.group(1) if match and rows else None
                params = None
