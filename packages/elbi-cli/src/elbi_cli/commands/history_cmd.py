"""``elbi history``: a derivation's certified result history.

Every certification records a version: the oracle's verified estimate, its verdict, and
the content-addressed identity that makes it reproducible. Unlike a conventional
experiment tracker these numbers are certified, not self-reported. ``history`` shows a
derivation's versions over time, newest first, and, for each, what moved the estimate
from the version before it (the data, the code, or the controls).
"""

from __future__ import annotations

from pathlib import Path

import typer

from elbi_core.errors import ElbiError
from elbi_core.tracking import CertifiedRun, with_changes

from .._console import EMPTY, arrow, console, fail
from ..project import LoadedProject, load_project

_DIRECTORY = typer.Option(
    Path(), "--directory", "-C", help="Project directory.", exists=True
)


def _load(directory: Path) -> LoadedProject:
    """Load the project rooted at ``directory``, exiting non-zero on failure."""
    try:
        return load_project(directory.resolve())
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc


def _estimate(run: CertifiedRun) -> str:
    """A version's certified estimate with its unit label, or a dash if it has none."""
    if run.estimate is None:
        return EMPTY
    label = f" {run.estimate_label}" if run.estimate_label else ""
    return f"{run.estimate:+.3g}{label}"


def history(
    name: str = typer.Argument(..., help="Derivation whose result history to show."),
    directory: Path = _DIRECTORY,
) -> None:
    """Show a derivation's certified versions, newest first, and what moved each.

    Each line is one certified version: its short hash, verdict, estimate, and date,
    followed by the input dimensions that changed from the previous version (data, code,
    controls, params, claim) so a moved estimate is attributed to its cause.
    """
    project = _load(directory)
    entries = with_changes(project.run_log.runs(name=name))
    if not entries:
        arrow(f"no certified versions recorded for {name!r}")
        return
    console.print(f"[bold]{name}[/bold] [dim]· {len(entries)} version(s)[/dim]")
    for run, changed in entries:
        moved = f"  [yellow]changed: {', '.join(changed)}[/yellow]" if changed else ""
        console.print(
            f"  [bold]{run.short_version}[/bold]  "
            f"[dim]{run.verdict} · {_estimate(run)} · {run.created_at}[/dim]{moved}"
        )
