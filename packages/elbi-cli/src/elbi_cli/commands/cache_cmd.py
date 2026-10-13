"""``elbi cache``: inspect and clear the derivation cache.

Every command works on the project directory's own cache. ``clear`` can instead target
a running app with ``--url`` or ``--target``, through the app's
``POST /api/cache/invalidate``, so a deployment's cache can be busted without a shell
on its host. Only a tag or a derivation can be cleared that way: emptying a whole
deployment's cache stays a decision made on the host.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import typer

from elbi_core.cache import LocalCacheStore, derivation_tag
from elbi_core.errors import ElbiError

from .._console import arrow, fail, ok
from ..config_sync import SyncError
from ..http import LoginRequired, client_for
from ..project import load_project
from .config_cmd import resolve_host

cache_app = typer.Typer(
    name="cache",
    help="Inspect and clear the local derivation cache.",
    no_args_is_help=True,
)


def _load(directory: Path) -> tuple[Path, int]:
    project = load_project(directory.resolve())
    ac = project.cache_dir / "ac"
    entries = len(list(ac.glob("*.json"))) if ac.exists() else 0
    return project.cache_dir, entries


@cache_app.command("status")
def status(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
) -> None:
    """Show where the cache lives and how many entries it holds."""
    try:
        cache_dir, entries = _load(directory)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    arrow(f"cache directory: {cache_dir}")
    arrow(f"cached entries: {entries}")


@cache_app.command("clear")
def clear(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
    tag: str | None = typer.Option(
        None, "--tag", help="Only invalidate entries carrying this tag."
    ),
    derivation: str | None = typer.Option(
        None, "--derivation", help="Only invalidate this derivation's entries."
    ),
    url: str | None = typer.Option(
        None, "--url", help="Clear a running app's cache instead of the local one."
    ),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Clear this declared target's cache."
    ),
) -> None:
    """Clear the cache, or only entries with a given tag or derivation."""
    if url is not None or target is not None:
        _clear_remote(directory, tag, derivation, url, token, target)
        return
    try:
        cache_dir, _ = _load(directory)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    store = LocalCacheStore(cache_dir)
    tags = [t for t in (tag, derivation_tag(derivation) if derivation else None) if t]
    if tags:
        for each in tags:
            removed = store.invalidate_tag(each)
            ok(f"invalidated {removed} entry(ies) tagged {each!r}")
    else:
        store.clear()
        ok("cleared the derivation cache")


def _clear_remote(
    directory: Path,
    tag: str | None,
    derivation: str | None,
    url: str | None,
    token: str | None,
    target: str | None,
) -> None:
    """Ask a running app to invalidate a tag or a derivation's entries."""
    if not tag and not derivation:
        fail("a remote clear needs --tag or --derivation")
        raise typer.Exit(code=2)
    body = {
        key: value for key, value in (("tag", tag), ("derivation", derivation)) if value
    }
    try:
        host = resolve_host(directory.resolve(), target, url)
        with client_for(host, token) as client:
            response = client.post("/api/cache/invalidate", json=body)
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except (SyncError, httpx.HTTPError) as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    if response.status_code != 200:
        fail(
            f"{host} refused the invalidation ({response.status_code}): {response.text}"
        )
        raise typer.Exit(code=1)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        # A 200 from something other than the app, most often a sign-in page sitting
        # in front of it.
        fail(f"{host} did not answer as an elbi app: {response.text[:200]!r}")
        raise typer.Exit(code=1)
    ok(f"invalidated {payload.get('removed', 0)} entry(ies) on {host}")


@cache_app.command("gc")
def gc(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
    max_age_days: float | None = typer.Option(
        None,
        "--max-age-days",
        help="Also evict cache entries older than this many days.",
    ),
) -> None:
    """Reclaim orphaned blobs, and optionally evict entries past a max age."""
    try:
        cache_dir, _ = _load(directory)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    store = LocalCacheStore(cache_dir)
    if max_age_days is not None:
        evicted = store.evict_older_than(max_age_days * 86400.0)
        arrow(f"evicted {evicted} entry(ies) older than {max_age_days} day(s)")
    reclaimed = store.gc()
    ok(f"reclaimed {reclaimed} orphaned blob(s)")
