"""Optional command-line plugins, discovered at startup.

A plugin adds commands to the ``elbi`` CLI and can teach it how to authenticate against
an app it would otherwise reach anonymously. Nothing here ships enabled. A package
advertises itself by declaring an entry point in the ``elbi.cli`` group::

    [project.entry-points."elbi.cli"]
    my_plugin = "my_package.cli"

The named module supplies ``install_cli(app)``, called once with the root Typer app
before arguments are parsed.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import typer

logger = logging.getLogger("elbi")

#: The entry-point group a command-line plugin declares itself in.
GROUP = "elbi.cli"


class CliPlugin(Protocol):
    """A module that adds commands to the CLI."""

    def install_cli(self, app: typer.Typer) -> None:
        """Register commands on ``app``. Called once, before arguments are parsed."""
        ...


def install_all(app: typer.Typer) -> list[str]:
    """Install every plugin onto ``app``; return the names installed.

    A plugin that fails to load must not stop the CLI from running its own commands,
    nor vanish without trace: the failure is logged and the rest still install.
    """
    installed: list[str] = []
    for entry in sorted(entry_points(group=GROUP), key=lambda e: e.name):
        try:
            cast("CliPlugin", entry.load()).install_cli(app)
        except Exception:
            logger.exception("cli plugin %r failed to load", entry.name)
            continue
        installed.append(entry.name)
    return installed
