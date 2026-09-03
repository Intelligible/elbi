"""``elbi promote``: graduate an agent-authored derivation into source."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer

from elbi_core.errors import ElbiError

from .._console import arrow, console, fail, ok
from ..authored import render_module
from ..project import load_project


def promote(
    name: str = typer.Argument(..., help="Agent-authored derivation to promote."),
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing source file."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be written, without writing."
    ),
) -> None:
    """Write an agent-authored derivation into derivations/ as human-owned source.

    The promoted file is a normal, editable derivation: its origin reverts to
    human, so it runs in-process rather than the sandbox. Review and commit it; the
    quarantined copy under .elbi/authored/ is removed.
    """
    try:
        project = load_project(directory.resolve())
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    record = project.authored_store.read(name)
    if record is None:
        fail(f"no agent-authored derivation named {name!r} in .elbi/authored/")
        raise typer.Exit(code=1)

    module = render_module(record, promoted_on=date.today().isoformat())
    target = project.derivations_dir / f"{name}.py"

    if dry_run:
        arrow(f"would write {target}")
        console.print(module)
        return

    if target.exists() and not force:
        fail(f"{target} already exists (use --force to overwrite)")
        raise typer.Exit(code=1)

    project.derivations_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(module, encoding="utf-8")
    project.authored_store.remove(name)
    ok(f"promoted {name!r} to {target.relative_to(project.base_dir)}")
    console.print("Review and commit it; it is now human-owned source.")
