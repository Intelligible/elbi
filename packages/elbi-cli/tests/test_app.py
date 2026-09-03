"""Top-level behaviour of the `elbi` command itself."""

from __future__ import annotations

from click import unstyle
from typer.testing import CliRunner

from elbi_cli import __version__
from elbi_cli.app import app


def test_version_flag(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # Unstyled for the same reason: with colour on, rich styles the numbers in a version
    # and `0.1.0` comes back as `0.1` in one span and `.0` in the next.
    assert __version__ in unstyle(result.output)


def test_no_args_shows_help(runner: CliRunner) -> None:
    # no_args_is_help prints usage and exits 2 (Click's "missing command").
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.output


def test_help_lists_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("init", "serve", "mcp", "validate"):
        assert command in result.output


def test_help_does_not_offer_commands_no_plugin_installed(runner: CliRunner) -> None:
    """A command a plugin would add is absent until that plugin is installed.

    The plugin group is the only way commands reach this CLI from outside, so an empty
    group has to mean an unchanged command list rather than stubs that fail when run.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Match the command column rather than the whole page: a description mentioning
    # "the deployment" contains "deploy" and would pass a substring check either way.
    listed = {
        line.strip("│ ").split()[0]
        for line in unstyle(result.output).splitlines()
        if line.strip().startswith("│") and len(line.strip("│ ").split()) > 1
    }
    assert listed.isdisjoint({"login", "logout", "deploy"})
