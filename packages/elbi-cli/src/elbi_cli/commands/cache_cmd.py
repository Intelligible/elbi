"""``elbi cache``: inspect and clear the local derivation cache."""

from __future__ import annotations

from pathlib import Path

import typer

from elbi_core.cache import LocalCacheStore
from elbi_core.errors import ElbiError

from .._console import arrow, fail, ok
from ..project import load_project

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
) -> None:
    """Clear the cache, or only entries with a given tag."""
    try:
        cache_dir, _ = _load(directory)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    store = LocalCacheStore(cache_dir)
    if tag is not None:
        removed = store.invalidate_tag(tag)
        ok(f"invalidated {removed} entry(ies) tagged {tag!r}")
    else:
        store.clear()
        ok("cleared the derivation cache")


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
