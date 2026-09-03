"""Connectors for the systems a team runs its work in.

Issue trackers, documents, chat, error reporting, forms and support conversations. All
seven are declarative: each names its base URL, how it authenticates, how the API
paginates, and which paths list what. The engine in :mod:`rest_base` does the fetching,
so nothing here writes a request.

Each authenticates with a token the user creates and pastes. None of them uses OAuth,
which is deliberate rather than a shortcut: an OAuth flow needs a registered application
and a callback the vendor can reach, and this app is meant to run on a laptop or a
private network where neither exists. Every one of these APIs supports a long-lived
token as a first-class option, and that is the option a self-hosted install can use.

The paginators are the part worth checking against each vendor's documentation, because
they all disagree. GitHub and Sentry put the next page in a ``Link`` header, Jira counts
records, Typeform counts pages, and the rest return an opaque cursor under four
different names.
"""

from __future__ import annotations

from typing import Any

from ..config import SourceField
from .registry import SourceRegistry
from .rest_base import RestApiConnector


def _token(name: str, label: str, caption: str = "") -> SourceField:
    """A secret field, so the value is stored encrypted and never shown again."""
    return SourceField(name=name, label=label, type="password", caption=caption)


def _res(
    name: str, path: str, selector: str | None = None, **extra: Any
) -> dict[str, Any]:
    """One listable resource: the table name, the path, and where the records sit."""
    resource: dict[str, Any] = {"name": name, "path": path}
    if selector:
        resource["data_selector"] = selector
    resource.update(extra)
    return resource


@SourceRegistry.register
class GitHubSource(RestApiConnector):
    """GitHub: the issues, pull requests, commits and releases of one repository.

    Scoped to a repository rather than a whole account, because that is the unit people
    actually ask questions about and because it keeps the token's required scope small.

    A personal access token rather than a GitHub App: the App flow needs an installation
    and a webhook endpoint, and the API treats a token as equal for reading.
    """

    key, label, category = "github", "GitHub", "Engineering & monitoring"
    icon = "🐙"
    caption = "Sync issues, pull requests, commits and releases from a repository."
    docs_url = (
        "https://docs.github.com/rest/authentication/authenticating-to-the-rest-api"
    )
    fields_ = (
        _token(
            "access_token",
            "Personal access token",
            "A fine-grained token with read access to the repository, or a classic "
            "token with the `repo` scope.",
        ),
        SourceField(
            name="repository",
            label="Repository",
            placeholder="owner/name",
            caption="Both halves, as they appear in the repository's URL.",
        ),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        del config
        return "https://api.github.com"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["access_token"]}

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        del config
        # Pinning the API version is what keeps a future breaking change from arriving
        # unannounced in a scheduled sync.
        return {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        return {"type": "header_link"}

    def _resources(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        repository = str(config.get("repository") or "").strip().strip("/")
        return [
            # `state=all`, or the default hides everything already closed -- which for
            # issues and pull requests is most of the history worth analysing.
            _res("issues", f"/repos/{repository}/issues", params={"state": "all"}),
            _res(
                "pull_requests", f"/repos/{repository}/pulls", params={"state": "all"}
            ),
            _res("commits", f"/repos/{repository}/commits"),
            _res("releases", f"/repos/{repository}/releases"),
            _res("contributors", f"/repos/{repository}/contributors"),
            _res("labels", f"/repos/{repository}/labels"),
            _res(
                "milestones", f"/repos/{repository}/milestones", params={"state": "all"}
            ),
        ]

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check the repository reads as ``owner/name`` before anything is fetched."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        repository = str(config.get("repository") or "").strip().strip("/")
        if repository.count("/") != 1 or not all(repository.split("/")):
            return False, [
                f"{repository!r} is not a repository. Give both halves, as in "
                "'octocat/hello-world'."
            ]
        return True, []


@SourceRegistry.register
class JiraSource(RestApiConnector):
    """Jira: issues, projects and the people who work them.

    Atlassian's own API tokens authenticate over HTTP basic with the account's email
    address as the username, which reads oddly and is what their documentation
    specifies.
    """

    key, label, category = "jira", "Jira", "Productivity"
    icon = "🧭"
    caption = "Sync issues, projects and users from a Jira site."
    docs_url = "https://support.atlassian.com/atlassian-account/docs/manage-api-tokens-for-your-atlassian-account/"
    fields_ = (
        SourceField(
            name="domain",
            label="Site domain",
            placeholder="acme.atlassian.net",
            caption="The host only, with no scheme and no trailing path.",
        ),
        SourceField(
            name="email",
            label="Account email",
            placeholder="you@acme.com",
            caption="The account the API token belongs to.",
        ),
        _token("api_token", "API token"),
    )
    resources_ = (
        _res("issues", "/rest/api/3/search", "issues"),
        _res("projects", "/rest/api/3/project/search", "values"),
        _res("users", "/rest/api/3/users/search"),
        _res("fields", "/rest/api/3/field"),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        domain = str(config.get("domain") or "").strip().strip("/")
        domain = domain.removeprefix("https://").removeprefix("http://")
        return f"https://{domain}"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "http_basic",
            "username": config["email"],
            "password": config["api_token"],
        }

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        # Jira counts records rather than pages, and reports the total, so the run ends
        # when the offset reaches it instead of on an empty page.
        return {
            "type": "offset",
            "limit": 100,
            "offset_param": "startAt",
            "limit_param": "maxResults",
            "total_path": "total",
        }


@SourceRegistry.register
class NotionSource(RestApiConnector):
    """Notion: the pages and databases an integration has been given.

    Notion shares nothing by default. A page reaches this connector only after someone
    connects the integration to it from Notion's own interface, which is why a token
    that works can still return an empty sync, and why the caption says so.

    Content is listed over POST, because that is the only way Notion offers it.
    """

    key, label, category = "notion", "Notion", "Productivity"
    icon = "📝"
    caption = "Sync pages, databases and users from a Notion workspace."
    docs_url = "https://developers.notion.com/docs/create-a-notion-integration"
    fields_ = (
        _token(
            "api_token",
            "Internal integration secret",
            "From the integration's page in Notion. Each page or database also has to "
            "be connected to that integration before it can be read.",
        ),
    )
    resources_ = (
        _res(
            "pages",
            "/v1/search",
            "results",
            method="POST",
            json={"filter": {"property": "object", "value": "page"}},
        ),
        _res(
            "databases",
            "/v1/search",
            "results",
            method="POST",
            json={"filter": {"property": "object", "value": "database"}},
        ),
        _res("users", "/v1/users", "results"),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        del config
        return "https://api.notion.com"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["api_token"]}

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        del config
        # Notion rejects a request that does not name a version, so this is required
        # rather than defensive.
        return {"Notion-Version": "2022-06-28"}

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        return {
            "type": "cursor",
            "cursor_path": "next_cursor",
            "cursor_param": "start_cursor",
        }


@SourceRegistry.register
class SlackSource(RestApiConnector):
    """Slack: channels, the people in them, and what was said.

    Messages come from the channels the bot has been invited to; Slack does not let an
    app read a channel it is not in, so an empty ``messages`` table usually means the
    invitation, not the token.
    """

    key, label, category = "slack", "Slack", "Productivity"
    icon = "💬"
    caption = "Sync channels, users and messages from a Slack workspace."
    docs_url = "https://api.slack.com/authentication/token-types#bot"
    fields_ = (
        _token(
            "bot_token",
            "Bot user OAuth token",
            "Starts with `xoxb-`. Needs the channels:read, users:read and "
            "channels:history scopes.",
        ),
        SourceField(
            name="channel_id",
            label="Channel ID for messages",
            required=False,
            placeholder="C0123456789",
            caption="Only needed for the messages table. Leave empty to sync channels "
            "and users alone.",
        ),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        del config
        return "https://slack.com/api"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["bot_token"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        return {
            "type": "cursor",
            "cursor_path": "response_metadata.next_cursor",
            "cursor_param": "cursor",
        }

    def _resources(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        resources = [
            _res(
                "channels",
                "/conversations.list",
                "channels",
                params={"types": "public_channel"},
            ),
            _res("users", "/users.list", "members"),
        ]
        # Offered only once a channel is named, so the table cannot be selected and then
        # fail at sync time asking for something the form never requested.
        if channel := str(config.get("channel_id") or "").strip():
            resources.append(
                _res(
                    "messages",
                    "/conversations.history",
                    "messages",
                    params={"channel": channel},
                )
            )
        return resources


@SourceRegistry.register
class SentrySource(RestApiConnector):
    """Sentry: projects, the issues raised against them, and releases."""

    key, label, category = "sentry", "Sentry", "Engineering & monitoring"
    icon = "🚨"
    caption = "Sync projects, issues and releases from a Sentry organisation."
    docs_url = "https://docs.sentry.io/api/auth/"
    fields_ = (
        _token("auth_token", "Auth token", "An organisation token with read scopes."),
        SourceField(
            name="organization",
            label="Organisation slug",
            placeholder="my-org",
        ),
        SourceField(
            name="host",
            label="Host",
            required=False,
            placeholder="sentry.io",
            caption="Only for a self-hosted Sentry. Leave empty for sentry.io.",
        ),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        host = str(config.get("host") or "sentry.io").strip().strip("/")
        host = host.removeprefix("https://").removeprefix("http://")
        return f"https://{host}/api/0"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["auth_token"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        return {"type": "header_link"}

    def _resources(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        organization = str(config.get("organization") or "").strip().strip("/")
        return [
            _res("projects", f"/organizations/{organization}/projects/"),
            _res("releases", f"/organizations/{organization}/releases/"),
            _res("teams", f"/organizations/{organization}/teams/"),
            _res(
                "issues",
                f"/organizations/{organization}/issues/",
                params={"statsPeriod": ""},
            ),
        ]


@SourceRegistry.register
class TypeformSource(RestApiConnector):
    """Typeform: the forms in an account and the responses to them.

    Responses are per-form, so the form to read is named on the connection rather than
    discovered: an account with many forms would otherwise fan out into a request per
    form on every sync.
    """

    key, label, category = "typeform", "Typeform", "Productivity"
    icon = "🗒️"
    caption = "Sync forms and their responses from a Typeform account."
    docs_url = "https://www.typeform.com/developers/get-started/personal-access-token/"
    fields_ = (
        _token("access_token", "Personal access token"),
        SourceField(
            name="form_id",
            label="Form ID for responses",
            required=False,
            placeholder="abc123XY",
            caption="Only needed for the responses table. Leave empty to sync the "
            "list of forms alone.",
        ),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        del config
        return "https://api.typeform.com"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["access_token"]}

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        # Typeform numbers pages from one and reports how many there are.
        return {
            "type": "page_number",
            "base_page": 1,
            "page_param": "page",
            "total_path": "page_count",
        }

    def _resources(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        resources = [_res("forms", "/forms", "items")]
        if form := str(config.get("form_id") or "").strip():
            # Responses are not paged the same way as forms, so this resource opts out
            # of the paginator above rather than being counted through pages that the
            # endpoint does not report.
            resources.append(
                _res(
                    "responses",
                    f"/forms/{form}/responses",
                    "items",
                    params={"page_size": 1000},
                )
            )
        return resources


@SourceRegistry.register
class IntercomSource(RestApiConnector):
    """Intercom: contacts, companies, conversations and the team answering them."""

    key, label, category = "intercom", "Intercom", "CRM"
    icon = "🎧"
    caption = "Sync contacts, companies and conversations from Intercom."
    docs_url = "https://developers.intercom.com/docs/build-an-integration/learn-more/authentication/"
    fields_ = (
        _token("access_token", "Access token"),
        SourceField(
            name="region",
            label="Region",
            type="select",
            required=False,
            default="us",
            options=[
                {"value": "us", "label": "US (api.intercom.io)"},
                {"value": "eu", "label": "Europe (api.eu.intercom.io)"},
                {"value": "au", "label": "Australia (api.au.intercom.io)"},
            ],
            caption="Intercom hosts each region separately, and a token is only valid "
            "against the one its workspace lives in.",
        ),
    )
    resources_ = (
        _res("contacts", "/contacts", "data"),
        _res("companies", "/companies", "data"),
        _res("conversations", "/conversations", "conversations"),
        _res("admins", "/admins", "admins"),
        _res("tags", "/tags", "data"),
    )

    def _base_url(self, config: dict[str, Any]) -> str:
        region = str(config.get("region") or "us").strip().lower()
        host = {"eu": "api.eu.intercom.io", "au": "api.au.intercom.io"}.get(
            region, "api.intercom.io"
        )
        return f"https://{host}"

    def _auth(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"type": "bearer", "token": config["access_token"]}

    def _headers(self, config: dict[str, Any]) -> dict[str, str]:
        del config
        return {"Intercom-Version": "2.11", "Accept": "application/json"}

    def _paginator(self, config: dict[str, Any]) -> Any:
        del config
        return {
            "type": "cursor",
            "cursor_path": "pages.next.starting_after",
            "cursor_param": "starting_after",
        }
