"""The dashboard service: CRUD, the publish gate, and widget resolution.

A dashboard is a declarative artifact over certified derivations. This service is the
seam between the stored manifest and a live project: it validates a spec on the way in,
refuses to publish a dashboard whose bindings point at derivations that do not exist,
optionally holds every number to a certified derivation when the agent publishes (its
guardrail; humans are trusted), and resolves a page's widgets by running their
derivations through the project's runner.

It is deliberately thin. Persistence lives in the :class:`~elbi.db.Store`;
the parameter-substitution and run logic lives in
:mod:`elbi.dashboard.resolve`; this class wires the two together and enforces
the lifecycle.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from elbi_core import DashboardSpec, Runner
from elbi_core.dashboard import MetricResolver, resolve_options, resolve_page
from elbi_core.errors import SpecValidationError

from .db import Dashboard, DashboardSubscription, Store
from .duplicate import copy_identifier, copy_label


class DashboardError(Exception):
    """A dashboard operation failed for a reason worth showing the caller."""


@dataclass(frozen=True)
class PublishResult:
    """The outcome of a publish attempt: whether it published, and why not."""

    ok: bool
    uncertified: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def detail(self) -> str:
        """A human-readable summary of why a publish was refused."""
        parts: list[str] = []
        if self.missing:
            parts.append(f"unknown derivations: {', '.join(self.missing)}")
        if self.uncertified:
            parts.append(f"uncertified derivations: {', '.join(self.uncertified)}")
        return "; ".join(parts) or "ok"


class DashboardService:
    """CRUD, lifecycle, and resolution for dashboards over a loaded project.

    ``make_runner`` builds a fresh project runner per resolution (the same runner the
    chat and MCP paths use, so a widget reads the same cached, certified result an
    agent would). ``certified_catalog`` returns the derivations eligible to bind, which
    both the publish gate and the editor's picker consult.
    """

    def __init__(
        self,
        store: Store,
        make_runner: Callable[[], Runner],
        certified_catalog: Callable[[], list[dict[str, Any]]],
        resolve_metric: MetricResolver | None = None,
        metric_exists: Callable[[str], bool] | None = None,
    ) -> None:
        self._store = store
        self._make_runner = make_runner
        self._catalog = certified_catalog
        # A tile may bind a semantic-layer metric; these resolve it and confirm it
        # exists. Injected by the app (the dashboard core has no metric engine).
        self._resolve_metric = resolve_metric
        self._metric_exists = metric_exists

    # -- authoring ---------------------------------------------------------------
    def create(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Validate a manifest and store it as a new draft dashboard."""
        spec = _parse(manifest)
        dashboard_id = self._store.create_dashboard(
            name=spec.name, title=spec.title or spec.name, spec_json=_dump(spec)
        )
        return self.get(dashboard_id)

    def duplicate(self, dashboard_id: str) -> dict[str, Any]:
        """Copy a dashboard into a new draft owned by the caller.

        A duplicate is a new dashboard, not a new version: its own id, its own history
        starting at version 1, and owned by whoever asked for it however the original
        was shared. Read access to the source is enough.

        The copy is an unpublished draft whatever the original was, and carries none of
        its scheduled deliveries: nobody should start receiving a report because
        somebody else copied the thing that sent it. The *draft* spec is what gets
        copied, so what an editor sees is what they get.

        Raises:
            DashboardError: if there is no dashboard of that id.
        """
        row = self._require(dashboard_id)
        manifest = _load(row.spec_json)
        taken = {summary["name"] for summary in self._store.list_dashboards()}
        # The row's name and the spec's name are the same identifier and must stay in
        # step; it is schema-constrained (lower snake_case), so it cannot take the
        # human-readable "(copy)" suffix the title gets.
        manifest["name"] = copy_identifier(row.name, taken)
        manifest["title"] = copy_label(row.title or row.name, ())
        spec = _parse(manifest)
        new_id = self._store.create_dashboard(
            name=spec.name,
            title=spec.title or spec.name,
            spec_json=_dump(spec),
            copied_from=dashboard_id,
        )
        return self.get(new_id)

    def save(self, dashboard_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        """Validate a manifest and replace a dashboard's draft spec."""
        self._require(dashboard_id)
        spec = _parse(manifest)
        self._store.save_dashboard(
            dashboard_id, _dump(spec), title=spec.title or spec.name
        )
        return self.get(dashboard_id)

    def publish(
        self, dashboard_id: str, require_certified: bool = False
    ) -> PublishResult:
        """Publish a dashboard, gating on certified bindings only for the agent.

        ``require_certified`` (the agent's guardrail) holds every tile to a certified
        derivation, so an agent-published dashboard never surfaces an unverified number.
        The trusted human path leaves it off (humans publish like they would in any BI
        tool), but a dashboard binding a derivation that does not exist is refused
        either way, since that is a broken manifest rather than a trust decision.
        """
        row = self._require(dashboard_id)
        spec = DashboardSpec.from_manifest(_load(row.spec_json))
        certified = {entry["name"] for entry in self._catalog()}
        missing: list[str] = []
        uncertified: list[str] = []
        for name in spec.derivation_names():
            if name not in certified:
                # A derivation that exists but is not certified is uncertified; one that
                # does not exist at all is missing. The catalog holds only certified
                # ones, so distinguish via a fresh registry lookup through the runner.
                if not self._exists(name):
                    missing.append(name)
                elif require_certified:
                    uncertified.append(name)
        # A metric tile is certified-backed by construction (a simple metric will not
        # define over an uncertified source), so the gate only confirms it exists.
        for metric_name in spec.metric_names():
            if self._metric_exists is not None and not self._metric_exists(metric_name):
                missing.append(metric_name)
        if missing or uncertified:
            return PublishResult(
                ok=False, uncertified=tuple(uncertified), missing=tuple(missing)
            )
        self._store.publish_dashboard(dashboard_id)
        return PublishResult(ok=True)

    def delete(self, dashboard_id: str, permanent: bool = False) -> None:
        """Move a dashboard to trash, or erase it immediately with ``permanent``.

        Requires ``Level.MANAGE`` -- the level that also lets you share the
        dashboard -- not the ``VIEW`` every other method here checks through
        :meth:`_require`. Deleting is not viewing, so it gets its own check, and
        ``trash_dashboard`` does not make one itself.
        """
        self._require(dashboard_id)
        if permanent:
            self._store.erase_dashboard(dashboard_id)
        else:
            self._store.trash_dashboard(dashboard_id)

    # -- reads -------------------------------------------------------------------
    def list_dashboards(self) -> list[dict[str, Any]]:
        """Dashboard summaries for the owner."""
        return self._store.list_dashboards()

    def get(self, dashboard_id: str) -> dict[str, Any]:
        """A dashboard's full state: draft spec, published spec, status, version."""
        row = self._require(dashboard_id)
        return _summary(row)

    def versions(self, dashboard_id: str) -> list[dict[str, Any]]:
        """The dashboard's saved revisions, newest first."""
        self._require(dashboard_id)
        return self._store.dashboard_versions(dashboard_id)

    def catalog(self) -> list[dict[str, Any]]:
        """The certified derivations available to bind, for the editor's picker."""
        return self._catalog()

    # -- resolution --------------------------------------------------------------
    def resolve(
        self,
        dashboard_id: str,
        page: str,
        variable_state: dict[str, Any],
        published: bool = False,
    ) -> list[dict[str, Any]]:
        """Run every data-bound widget on ``page`` against the viewer's selections.

        ``published`` reads the frozen published spec (what a viewer sees); otherwise
        the working draft (what an editor previews). A widget whose derivation fails
        carries an ``error`` rather than blanking the page.
        """
        spec = self._spec(dashboard_id, published=published)
        runner = self._make_runner()
        try:
            results = resolve_page(
                runner, spec, page, variable_state, metric_resolver=self._resolve_metric
            )
        except KeyError as exc:
            raise DashboardError(f"unknown page {page!r}") from exc
        return [
            {
                "widget_id": data.widget_id,
                "derivation": data.derivation,
                "kind": data.kind,
                "value": data.value,
                "data_version": data.data_version,
                "error": data.error,
            }
            for data in results
        ]

    def options(self, dashboard_id: str, variable: str) -> list[dict[str, Any]]:
        """The selectable options for a filter control, static or derivation-backed."""
        spec = self._spec(dashboard_id, published=False)
        var = spec.variable(variable)
        if var is None:
            raise DashboardError(f"unknown variable {variable!r}")
        return resolve_options(self._make_runner(), var)

    # -- subscriptions -----------------------------------------------------------
    def subscribe(
        self,
        dashboard_id: str,
        page: str,
        cron: str,
        recipients: Sequence[str],
        variable_state: dict[str, Any],
        fmt: str = "png",
        channel: str = "email",
    ) -> str:
        """Register a scheduled snapshot delivery for a dashboard."""
        self._require(dashboard_id)
        return self._store.create_dashboard_subscription(
            dashboard_id=dashboard_id,
            page=page,
            cron=cron,
            recipients=recipients,
            variable_state=variable_state,
            fmt=fmt,
            channel=channel,
        )

    def subscriptions(self, dashboard_id: str) -> list[dict[str, Any]]:
        """A dashboard's subscriptions."""
        self._require(dashboard_id)
        return [
            {
                "id": row.id,
                "page": row.page,
                "cron": row.cron,
                "fmt": row.fmt,
                "channel": row.channel,
                "active": row.active,
            }
            for row in self._store.list_dashboard_subscriptions(dashboard_id)
        ]

    def unsubscribe(self, dashboard_id: str, subscription_id: str) -> None:
        """Remove a subscription."""
        self._require(dashboard_id)
        self._store.delete_dashboard_subscription(subscription_id)

    def render_delivery(self, subscription: DashboardSubscription) -> tuple[str, str]:
        """Render a subscription's snapshot as a (subject, body) text summary.

        Resolves the subscribed page of the *published* dashboard under the saved
        variable state, then summarizes each widget: a metric's value, a table's row
        count, or a widget's error. The scheduler delivers this over the subscription's
        channel. (A pixel-perfect image/PDF snapshot is a headless-render extension.)
        """
        view = self.get(subscription.dashboard_id)
        variables = json.loads(subscription.variable_state_json or "{}")
        widgets = self.resolve(
            subscription.dashboard_id, subscription.page, variables, published=True
        )
        lines: list[str] = []
        for widget in widgets:
            if widget["error"]:
                lines.append(f"- {widget['widget_id']}: error, {widget['error']}")
            elif isinstance(widget["value"], list):
                lines.append(f"- {widget['widget_id']}: {len(widget['value'])} rows")
            else:
                lines.append(f"- {widget['widget_id']}: {widget['value']}")
        subject = f"{view['title'] or view['name']}: {subscription.page}"
        body = f"Dashboard snapshot: {subject}\n\n" + (
            "\n".join(lines) or "No data-bound widgets on this page."
        )
        return subject, body

    # -- internals ---------------------------------------------------------------
    def _require(self, dashboard_id: str) -> Dashboard:
        row = self._store.get_dashboard(dashboard_id)
        if row is None:
            raise DashboardError(f"dashboard {dashboard_id!r} not found")
        return row

    def _spec(self, dashboard_id: str, published: bool) -> DashboardSpec:
        row = self._require(dashboard_id)
        source = row.published_spec_json if published else row.spec_json
        if source is None:
            raise DashboardError("dashboard has not been published")
        return DashboardSpec.from_manifest(_load(source))

    def _exists(self, name: str) -> bool:
        try:
            self._make_runner().data_version(name)
        except Exception:
            return False
        return True


def _parse(manifest: dict[str, Any]) -> DashboardSpec:
    try:
        return DashboardSpec.from_manifest(manifest)
    except SpecValidationError as exc:
        raise DashboardError(str(exc)) from exc


def _summary(row: Dashboard) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "title": row.title,
        "status": row.status,
        "version": row.version,
        "copied_from": row.copied_from,
        "spec": _load(row.spec_json),
        "published_spec": _load(row.published_spec_json)
        if row.published_spec_json
        else None,
        "updated_at": row.updated_at.isoformat(),
    }


def _dump(spec: DashboardSpec) -> str:
    import json

    return json.dumps(spec.to_manifest())


def _load(spec_json: str) -> dict[str, Any]:
    import json

    data: dict[str, Any] = json.loads(spec_json)
    return data
