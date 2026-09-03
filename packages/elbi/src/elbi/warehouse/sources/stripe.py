"""Stripe connector: payments, billing, and customer data over the Stripe REST API.

Syncs the common Stripe resources (customers, charges, invoices, subscriptions,
products) as tables, each incremental on Stripe's ``created`` timestamp. Uses cursor
pagination (``starting_after``) to page through the list endpoints, batched into Arrow.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, SourceInputs
from .registry import SourceRegistry

_BASE = "https://api.stripe.com/v1"
_RESOURCES = ("customers", "charges", "invoices", "subscriptions", "products")
_PAGE = 100
_TIMEOUT = 30.0


@SourceRegistry.register
class StripeSource(SimpleSource):
    """Sync Stripe payments, billing, and customer objects."""

    @property
    def source_type(self) -> str:
        """The registry key for this connector."""
        return "stripe"

    @property
    def config(self) -> SourceConfig:
        """The catalog entry and connection-form schema."""
        return SourceConfig(
            name="stripe",
            label="Stripe",
            category="Payments & billing",
            icon="💳",
            caption=(
                "Connect your Stripe account with a restricted API key and your Stripe "
                "account ID to sync customers, charges, invoices, and more."
            ),
            docs_url="https://stripe.com/docs/api",
            fields=[
                SourceField(
                    name="auth_method",
                    label="Authentication type",
                    type="select",
                    default="restricted_api_key",
                    options=[
                        {"value": "restricted_api_key", "label": "Restricted API key"}
                    ],
                ),
                SourceField(
                    name="stripe_secret_key",
                    label="API key",
                    type="password",
                    placeholder="rk_live_...",
                    caption=(
                        "Create a restricted API key with the pre-defined permissions"
                    ),
                ),
                SourceField(
                    name="stripe_account_id",
                    label="Account id",
                    placeholder="stripe_account_id",
                ),
            ],
        )

    def _auth(self, config: dict[str, Any]) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {config['stripe_secret_key']}"}
        account = config.get("stripe_account_id")
        if account:
            headers["Stripe-Account"] = account  # scope to the connected account
        return headers

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Confirm the key works with a one-record probe against ``/customers``."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            import httpx

            response = httpx.get(
                f"{_BASE}/customers",
                headers=self._auth(config),
                params={"limit": 1},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return True, []
        except Exception as e:  # surface the auth/fetch error to the user, any type
            return False, [f"Could not authenticate with Stripe: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Stripe's common resources, each with ``created`` as its cursor."""
        return [
            SourceSchema(name=resource, incremental_fields=["created"])
            for resource in _RESOURCES
        ]

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Page through one resource's list endpoint, yielding a batch per page.

        Filters ``created > since`` on an incremental run (Stripe's ``created[gt]``),
        and follows cursor pagination (``starting_after``) until ``has_more`` is false.
        """
        import httpx

        params: dict[str, Any] = {"limit": _PAGE}
        since = inputs.incremental_since
        if inputs.incremental_field == "created" and since is not None:
            params["created[gt]"] = int(since)
        headers = self._auth(inputs.config)
        url = f"{_BASE}/{inputs.schema}"
        with httpx.Client(timeout=_TIMEOUT) as http:
            while True:
                response = http.get(url, headers=headers, params=params)
                response.raise_for_status()
                body = response.json()
                rows = body.get("data", [])
                if rows:
                    yield pa.Table.from_pylist(rows)
                if not body.get("has_more") or not rows:
                    break
                params["starting_after"] = rows[-1]["id"]
