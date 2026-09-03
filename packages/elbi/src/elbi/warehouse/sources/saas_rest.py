"""Ten token/key REST SaaS connectors, declared over the shared dlt ``rest_api`` base.

Each connector declares its base URL, auth, resource paths, data selectors, and
paginator, and dlt's ``rest_api`` engine does the fetching; these classes write no
request or pagination code. All are full-refresh and untested against live vendor
APIs.
"""

from __future__ import annotations

from typing import Any

from ..config import SourceField
from .registry import SourceRegistry
from .rest_base import RestApiConnector


def _pw(name: str, label: str) -> SourceField:
    return SourceField(name=name, label=label, type="password")


def _res(name: str, path: str, data_selector: str = "") -> dict[str, Any]:
    return {"name": name, "path": path, "data_selector": data_selector}


@SourceRegistry.register
class ChargebeeSource(RestApiConnector):
    """Chargebee: subscriptions and billing."""

    key, label, category = "chargebee", "Chargebee", "Payments & billing"
    caption = "Sync customers, subscriptions, invoices, and transactions."
    docs_url = "https://apidocs.chargebee.com/docs/api"
    fields_ = (
        _pw("api_key", "API key"),
        SourceField(
            name="site_name", label="Site name (subdomain)", placeholder="acme"
        ),
    )
    resources_ = tuple(
        _res(r, f"/v2/{r}", f"list[*].{s}")
        for r, s in (
            ("customers", "customer"),
            ("subscriptions", "subscription"),
            ("invoices", "invoice"),
            ("transactions", "transaction"),
            ("orders", "order"),
        )
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return f"https://{config['site_name']}.chargebee.com/api"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "http_basic", "username": config["api_key"], "password": ""}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {
            "type": "cursor",
            "cursor_path": "next_offset",
            "cursor_param": "offset",
        }


@SourceRegistry.register
class MailchimpSource(RestApiConnector):
    """Mailchimp: audiences and campaigns."""

    key, label, category = "mailchimp", "Mailchimp", "Marketing & email"
    caption = "Sync lists, campaigns, and reports from Mailchimp."
    docs_url = "https://mailchimp.com/developer/marketing/api/"
    fields_ = (_pw("api_key", "API key"),)
    resources_ = (
        _res("lists", "/lists", "lists"),
        _res("campaigns", "/campaigns", "campaigns"),
        _res("reports", "/reports", "reports"),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        # The data-center is the suffix of the API key, e.g. "...-us21".
        key = config["api_key"]
        dc = key.rsplit("-", 1)[-1] if "-" in key else "us1"
        return f"https://{dc}.api.mailchimp.com/3.0"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "http_basic", "username": "key", "password": config["api_key"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {
            "type": "offset",
            "limit": 1000,
            "offset_param": "offset",
            "limit_param": "count",
            "total_path": "total_items",
        }


@SourceRegistry.register
class KlaviyoSource(RestApiConnector):
    """Klaviyo: marketing profiles, campaigns, and events."""

    key, label, category = "klaviyo", "Klaviyo", "Marketing & email"
    caption = "Sync campaigns, lists, profiles, metrics, events, and flows."
    docs_url = "https://developers.klaviyo.com/en/reference/api_overview"
    fields_ = (_pw("api_key", "API key"),)
    resources_ = tuple(
        _res(n, f"/{n}", "data")
        for n in ("campaigns", "lists", "profiles", "metrics", "events", "flows")
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return "https://a.klaviyo.com/api"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "api_key",
            "name": "Authorization",
            "api_key": f"Klaviyo-API-Key {config['api_key']}",
            "location": "header",
        }

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        return {"revision": "2024-10-15"}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {"type": "json_link", "next_url_path": "links.next"}


@SourceRegistry.register
class SendgridSource(RestApiConnector):
    """SendGrid: email suppression data."""

    key, label, category = "sendgrid", "SendGrid", "Marketing & email"
    caption = "Sync bounces, blocks, spam reports, and invalid emails from SendGrid."
    docs_url = "https://docs.sendgrid.com/api-reference"
    fields_ = (_pw("api_key", "API key"),)
    resources_ = tuple(
        _res(n, f"/v3/suppression/{n}", "$")
        for n in ("bounces", "blocks", "spam_reports", "invalid_emails")
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return "https://api.sendgrid.com"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["api_key"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {
            "type": "offset",
            "limit": 500,
            "offset_param": "offset",
            "limit_param": "limit",
        }


@SourceRegistry.register
class BrazeSource(RestApiConnector):
    """Braze: campaigns, canvases, and segments."""

    key, label, category = "braze", "Braze", "Marketing & email"
    caption = "Sync campaigns, canvases, and segments from Braze."
    docs_url = "https://www.braze.com/docs/api/basics"
    fields_ = (
        _pw("api_key", "REST API key"),
        SourceField(
            name="url",
            label="REST endpoint URL",
            placeholder="https://rest.iad-01.braze.com",
        ),
    )
    resources_ = (
        _res("campaigns", "/campaigns/list", "campaigns"),
        _res("canvases", "/canvas/list", "canvases"),
        _res("segments", "/segments/list", "segments"),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return str(config["url"]).rstrip("/")

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["api_key"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {"type": "page_number", "base_page": 0, "page_param": "page"}


@SourceRegistry.register
class PipedriveSource(RestApiConnector):
    """Pipedrive: CRM deals, people, and organizations."""

    key, label, category = "pipedrive", "Pipedrive", "CRM"
    caption = "Sync deals, persons, organizations, and activities from Pipedrive."
    docs_url = "https://developers.pipedrive.com/docs/api/v1"
    fields_ = (
        _pw("api_token", "API token"),
        SourceField(
            name="company_domain", label="Company domain", placeholder="mycompany"
        ),
    )
    resources_ = tuple(
        _res(n, f"/{n}", "data")
        for n in ("deals", "persons", "organizations", "activities", "leads")
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return f"https://{config['company_domain']}.pipedrive.com/api/v1"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "api_key",
            "name": "api_token",
            "api_key": config["api_token"],
            "location": "query",
        }

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {
            "type": "cursor",
            "cursor_path": "additional_data.pagination.next_start",
            "cursor_param": "start",
        }


@SourceRegistry.register
class FrontSource(RestApiConnector):
    """Front: shared-inbox conversations and contacts."""

    key, label, category = "front", "Front", "Productivity"
    caption = "Sync conversations, contacts, inboxes, and tags from Front."
    docs_url = "https://dev.frontapp.com/reference/introduction"
    fields_ = (_pw("api_key", "API token"),)
    resources_ = tuple(
        _res(n, f"/{n}", "_results")
        for n in ("conversations", "contacts", "inboxes", "tags", "teammates")
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return "https://api2.frontapp.com"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["api_key"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {"type": "json_link", "next_url_path": "_pagination.next"}


@SourceRegistry.register
class VercelSource(RestApiConnector):
    """Vercel: deployments and projects."""

    key, label, category = "vercel", "Vercel", "Engineering & monitoring"
    caption = "Sync deployments, projects, and teams from Vercel."
    docs_url = "https://vercel.com/docs/rest-api"
    fields_ = (_pw("api_key", "Access token"),)
    resources_ = (
        _res("deployments", "/v6/deployments", "deployments"),
        _res("projects", "/v9/projects", "projects"),
        _res("teams", "/v2/teams", "teams"),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return "https://api.vercel.com"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["api_key"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {
            "type": "cursor",
            "cursor_path": "pagination.next",
            "cursor_param": "until",
        }


@SourceRegistry.register
class AirtableSource(RestApiConnector):
    """Airtable: records from the tables of a base."""

    key, label, category = "airtable", "Airtable", "Productivity"
    caption = "Sync records from the tables of an Airtable base."
    docs_url = "https://airtable.com/developers/web/api/introduction"
    fields_ = (
        _pw("api_key", "Personal access token"),
        SourceField(name="base_id", label="Base ID", placeholder="app..."),
        SourceField(
            name="tables",
            label="Tables (comma-separated)",
            placeholder="Contacts, Deals",
            caption="The tables in the base to sync.",
        ),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        return f"https://api.airtable.com/v0/{config['base_id']}"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["api_key"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        return {"type": "cursor", "cursor_path": "offset", "cursor_param": "offset"}

    def _resources(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        from urllib.parse import quote

        raw = str(config.get("tables", "")).split(",")
        names = [t.strip() for t in raw if t.strip()]
        return [_res(t, f"/{quote(t)}", "records") for t in names]


_MIXPANEL_REGIONS = {
    "us": "mixpanel.com",
    "eu": "eu.mixpanel.com",
    "in": "in.mixpanel.com",
}


@SourceRegistry.register
class MixpanelSource(RestApiConnector):
    """Mixpanel: user profiles via the Query API (service-account auth)."""

    key, label, category = "mixpanel", "Mixpanel", "Analytics"
    caption = "Sync user profiles (engage) from Mixpanel via a service account."
    docs_url = "https://developer.mixpanel.com/reference/overview"
    fields_ = (
        SourceField(
            name="region",
            label="Data residency region",
            type="select",
            default="us",
            options=[
                {"value": "us", "label": "US (mixpanel.com)"},
                {"value": "eu", "label": "EU (eu.mixpanel.com)"},
                {"value": "in", "label": "India (in.mixpanel.com)"},
            ],
        ),
        SourceField(name="project_id", label="Project ID"),
        SourceField(name="service_account_username", label="Service account username"),
        _pw("service_account_secret", "Service account secret"),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        host = _MIXPANEL_REGIONS.get(config.get("region", "us"), "mixpanel.com")
        return f"https://{host}/api/2.0"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "http_basic",
            "username": config["service_account_username"],
            "password": config["service_account_secret"],
        }

    def _resources(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": "engage",
                "path": "/engage",
                "data_selector": "results",
                "params": {"project_id": config["project_id"]},
            }
        ]
