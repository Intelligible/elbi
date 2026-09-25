"""`elbi snapshot`: one dashboard, frozen into a self-contained HTML page.

For handing a dashboard to someone who should see its numbers but not the app: the
page opens anywhere, needs nothing, and states when it was taken and which derivation
produced each figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import typer

from .._console import arrow, fail, ok, warn
from ..config_sync import SyncError
from ..http import LoginRequired, client_for
from ..snapshot import render
from .config_cmd import resolve_host


def _find(client: httpx.Client, dashboard: str) -> dict[str, Any]:
    resp = client.get("/api/dashboards")
    resp.raise_for_status()
    rows = resp.json()
    for row in rows:
        if dashboard in (row.get("name"), row.get("id")):
            return row
    known = ", ".join(sorted(str(r.get("name")) for r in rows)) or "none"
    raise SyncError(f"no dashboard named {dashboard!r} (dashboards: {known})")


def snapshot(
    dashboard: str = typer.Argument(..., help="Dashboard name or id."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Where to write the page. Defaults to <name>.html."
    ),
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Repo directory.", exists=True
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
) -> None:
    """Freeze a dashboard's current values into one self-contained HTML page.

    The page renders the published version, every page of it, at default variable
    values. It is read-only: figures do not update, and it carries no version history.
    """
    try:
        host = resolve_host(directory.resolve(), target, url)
        with client_for(host, token) as client:
            row = _find(client, dashboard)
            arrow(f"resolving {row['name']} (every widget runs, cached results reused)")
            resp = client.get(f"/api/exports/dashboards/{row['id']}")
            resp.raise_for_status()
            document = resp.json()
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except SyncError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        fail(f"could not reach the app: {exc}")
        raise typer.Exit(code=1) from exc

    if document.get("status") != "published":
        warn(f"{row['name']} has never been published; snapshotting the draft")
    errors = [
        w["widget_id"]
        for widgets in (document.get("values") or {}).values()
        for w in widgets
        if w.get("error")
    ]
    if errors:
        warn(
            f"{len(errors)} widget(s) failed and show their error: {', '.join(errors)}"
        )

    path = out or Path(f"{row['name']}.html")
    path.write_text(render(document), encoding="utf-8")
    ok(f"wrote {path}")
