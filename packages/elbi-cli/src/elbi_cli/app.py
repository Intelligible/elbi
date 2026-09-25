"""The ``elbi`` CLI entry point."""

from __future__ import annotations

import typer

from . import __version__, plugins
from ._console import console
from .commands import (
    certify_cmd,
    config_cmd,
    export_cmd,
    history_cmd,
    init_cmd,
    mcp_cmd,
    promote_cmd,
    serve_cmd,
    snapshot_cmd,
    test_cmd,
    update_cmd,
    validate_cmd,
)
from .commands.cache_cmd import cache_app
from .commands.certificate_cmd import certificate_app
from .commands.run_cmd import run_app
from .commands.search_cmd import search_app

app = typer.Typer(
    name="elbi",
    help="Author data context as code and serve it to agents over MCP.",
    no_args_is_help=True,
    add_completion=True,
)

app.command(name="init")(init_cmd.init)
app.command(name="serve")(serve_cmd.serve)
app.command(name="mcp")(mcp_cmd.mcp)
app.command(name="validate")(validate_cmd.validate)
app.command(name="test")(test_cmd.test)
app.command(name="promote")(promote_cmd.promote)
app.command(name="certify")(certify_cmd.certify)
app.command(name="derivations")(certify_cmd.derivations)
app.command(name="history")(history_cmd.history)
app.command(name="sync")(config_cmd.sync)
app.command(name="pull")(config_cmd.pull)
app.command(name="plan")(config_cmd.plan)
app.command(name="schema")(config_cmd.schema)
app.command(name="upload")(config_cmd.upload)
app.command(name="update")(update_cmd.update)
app.command(name="export")(export_cmd.export)
app.command(name="import")(export_cmd.import_)
app.command(name="snapshot")(snapshot_cmd.snapshot)
app.add_typer(cache_app, name="cache")
app.add_typer(search_app, name="search")
app.add_typer(certificate_app, name="certificate")
app.add_typer(run_app, name="run")

# Anything an installed plugin adds. Last, so a plugin cannot shadow a built-in command
# by registering the same name first.
plugins.install_all(app)


def _version(value: bool) -> None:
    if value:
        console.print(f"elbi {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version,
        is_eager=True,
    ),
) -> None:
    """Elbi: the open framework CLI."""


if __name__ == "__main__":  # pragma: no cover
    app()
