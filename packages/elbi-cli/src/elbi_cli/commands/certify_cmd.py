"""``elbi certify`` and ``elbi derivations``: the local review gate.

A derivation authored as ``proposed`` (typically agent-authored) is discovered
but not served. ``elbi certify <name>`` records local approval in the
lifecycle sidecar, after which ``dev`` serves it. ``elbi derivations``
lists what is discovered, with provenance and effective status, so a reviewer can
see what is awaiting certification.
"""

from __future__ import annotations

from pathlib import Path

import typer

from elbi_core.errors import ElbiError

from .._console import arrow, console, fail, ok
from ..project import load_project


def certify(
    name: str = typer.Argument(..., help="Name of the derivation to certify."),
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
) -> None:
    """Approve a proposed derivation so it will be served.

    Exits non-zero if the project cannot be loaded or the name is unknown.
    """
    base_dir = directory.resolve()
    try:
        project = load_project(base_dir)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    if name not in project.registry:
        known = ", ".join(project.registry.names()) or "(none)"
        fail(f"no derivation named {name!r}; discovered: {known}")
        raise typer.Exit(code=1)

    derivation = project.registry.get(name)
    if derivation.is_certified:
        arrow(f"{name} is already certified")
        return

    project.lifecycle.set_status(name, "certified")
    ok(f"certified {name} (was proposed); it will be served on the next `dev` run")


def derivations(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
) -> None:
    """List discovered derivations with their origin and lifecycle status."""
    base_dir = directory.resolve()
    try:
        project = load_project(base_dir)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    names = project.registry.names()
    if not names:
        arrow("no derivations discovered")
        return

    for name in names:
        derivation = project.registry.get(name)
        served = "served" if derivation.is_served else "internal"
        status = "certified" if derivation.is_certified else "PROPOSED"
        console.print(
            f"  [bold]{name}[/bold]  "
            f"[dim]{derivation.origin} · {status} · {served}[/dim]"
        )
