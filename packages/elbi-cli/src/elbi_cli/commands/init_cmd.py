"""``elbi init``: scaffold a new project, offline."""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path

import typer

from .._console import arrow, console, fail, ok
from ..scaffold import WORKSPACE_PACKAGES, render

_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_CHECKOUT_ENV = "ELBI_GIT_REPO_DIR"


class Template(str, Enum):
    """Project templates."""

    standard = "standard"
    minimal = "minimal"


def init(
    name: str = typer.Argument(..., help="Project directory name (lower kebab/snake)."),
    template: Template = typer.Option(
        Template.standard, "--template", "-t", help="Which scaffold to use."
    ),
    force: bool = typer.Option(
        False, "--force", help="Write into a non-empty directory."
    ),
    editable: bool = typer.Option(
        False,
        "--editable",
        help=(
            "Depend on a local checkout rather than the registry; its path comes "
            f"from ${_CHECKOUT_ENV}."
        ),
    ),
) -> None:
    """Scaffold a new data-context project.

    Runs fully offline: nothing is sent anywhere. Then just `elbi serve`.
    """
    if not _NAME.match(name):
        fail(f"invalid project name {name!r}; use lower-case letters, digits, - or _")
        raise typer.Exit(code=1)

    target = Path.cwd() / name
    if target.exists() and any(target.iterdir()) and not force:
        fail(f"{target} already exists and is not empty (use --force to override)")
        raise typer.Exit(code=1)

    editable_root = _checkout_root() if editable else None
    files = render(
        name, minimal=template is Template.minimal, editable_root=editable_root
    )
    for relative, content in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        arrow(f"created {name}/{relative}")

    if editable_root is not None:
        arrow(f"depending on the checkout at {editable_root}")
    ok(f"scaffolded {name} ({template.value})")
    console.print()
    console.print("Next steps:")
    console.print(f"  cd {name}")
    if template is Template.standard:
        console.print(
            "  elbi serve      # chat UI + MCP at "
            "http://localhost:7700, opens a browser"
        )
        console.print(
            "[dim]   or: elbi mcp     # MCP only, no chat UI, no app package[/dim]"
        )
        console.print(
            "[dim]  elbi test         # run every derivation and "
            "re-verify each claim[/dim]"
        )
    else:
        console.print("  # add datasets and derivations to elbi.yaml")
        console.print("  elbi validate")


def _checkout_root() -> Path:
    """The checkout named by the environment, verified to be one."""
    raw = os.environ.get(_CHECKOUT_ENV)
    if not raw:
        fail(f"--editable needs ${_CHECKOUT_ENV} set to an elbi checkout")
        raise typer.Exit(code=1)
    root = Path(raw).expanduser().resolve()
    absent = [n for n in WORKSPACE_PACKAGES if not (root / "packages" / n).is_dir()]
    if absent:
        fail(f"{root} is not an elbi checkout (no packages/{absent[0]})")
        raise typer.Exit(code=1)
    return root
