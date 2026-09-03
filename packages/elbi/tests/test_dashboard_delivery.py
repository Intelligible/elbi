"""Scheduled-delivery rendering for dashboard subscriptions.

A subscription's snapshot is the published page resolved under its saved variable state,
summarized per widget. These pin that render (the scheduler's payload) and that it reads
the published spec, not the draft.
"""

from __future__ import annotations

from pathlib import Path

from elbi.dashboards import DashboardService
from elbi.db import open_store
from elbi_core import Artifact, Context, Registry, Runner, derivation, serve
from elbi_core.registry import use_registry


def _service(tmp_path: Path) -> DashboardService:
    registry = Registry()
    with use_registry(registry):

        @derivation(serve=serve.table())
        def revenue(ctx: Context) -> Artifact:
            return Artifact.table([{"r": 1}, {"r": 2}, {"r": 3}])

    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    return DashboardService(
        store=store,
        make_runner=lambda: Runner(registry),
        certified_catalog=lambda: [
            {"name": d.name, "title": d.name, "params": {}, "served": True}
            for d in registry
            if d.is_certified
        ],
    )


def _spec() -> dict:
    return {
        "specVersion": "1.0",
        "kind": "Dashboard",
        "name": "sales",
        "title": "Sales",
        "pages": [
            {
                "name": "main",
                "widgets": [
                    {
                        "id": "rows",
                        "type": "table",
                        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                        "bind": {"derivation": "revenue"},
                    }
                ],
            }
        ],
    }


def test_render_delivery_summarizes_published_page(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = service.create(_spec())
    service.publish(created["id"])
    subscription_id = service.subscribe(
        created["id"],
        page="main",
        cron="0 9 * * *",
        recipients=["ops@example.com"],
        variable_state={},
        fmt="csv",
        channel="email",
    )
    subs = service._store.list_dashboard_subscriptions(created["id"])
    subscription = next(s for s in subs if s.id == subscription_id)

    subject, body = service.render_delivery(subscription)
    assert "Sales" in subject
    assert "rows: 3 rows" in body
