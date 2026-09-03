"""``elbi update``: ask whether a newer release exists, and say how to install it."""

from __future__ import annotations

import os
import shlex

import typer

from .._console import arrow, console, fail, ok, warn
from ..update import check


def update(
    package: str = typer.Option(
        "",
        "--package",
        help="Check a specific distribution instead of the installed one.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Run the upgrade instead of printing it. Hands this process over to the "
        "installer, so nothing of Elbi is running while its files change.",
    ),
    pre: bool = typer.Option(
        False,
        "--pre",
        help="Consider pre-releases. Off unless the installed version is itself one.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Print nothing when already up to date. Exits 1 if an update exists, "
        "which is what makes it usable in a script.",
    ),
) -> None:
    """Check whether a newer release of Elbi is available.

    Reaches the network only when you run it. There is no check at startup, on a timer,
    or inside any other command, so nothing about this install is sent anywhere unless
    this command is typed.

    The request is a GET for a public JSON file on PyPI, carrying a User-Agent that
    names this tool and its version. Nothing identifies the machine or the install.

    Prints the upgrade command by default; ``--apply`` runs it.
    """
    report = check(package or None, allow_prerelease=True if pre else None)

    if report.latest is None:
        # The reason matters: a name PyPI does not know sends the user somewhere quite
        # different from a network that timed out.
        fail(f"Could not check for a newer {report.package}: {report.error}")
        arrow(f"Nothing was changed. This install is {report.installed}.")
        raise typer.Exit(code=1)

    if not report.outdated:
        if not quiet:
            ok(f"{report.package} {report.installed} is the latest release.")
        return

    if quiet:
        console.print(f"{report.package} {report.installed} -> {report.latest}")
        raise typer.Exit(code=1)

    warn(
        f"A new release of {report.package} is available: "
        f"{report.installed} -> {report.latest}"
    )
    if report.changelog:
        # Two version numbers say that something changed, not what. This is the
        # question anybody actually has before they upgrade.
        arrow(f"What changed: {report.changelog}")
    console.print()

    if apply:
        _apply(report.install.command, report.install.note)

    if report.install.command:
        arrow("To upgrade:")
        console.print(f"    {report.install.command}", style="bold")
    else:
        arrow("Upgrade it the way you installed it.")

    if report.install.note:
        console.print()
        arrow(report.install.note)

    # Exits non-zero so a script can act on it, matching the convention of every other
    # check-style command here. A person reading the output sees no difference.
    raise typer.Exit(code=1)


def _apply(command: str, note: str) -> None:
    """Hand this process over to the upgrade command.

    ``exec`` rather than a subprocess, deliberately. The upgrade rewrites the very
    environment this interpreter is running out of, and a Python process that keeps
    going while its own files are replaced can fail on the next import it happens to
    make. Replacing the process image means nothing of Elbi is running by the time
    anything changes, and the installer's exit status becomes this command's.

    Returns only when it declined to run something; otherwise this process is gone.
    """
    if not command:
        fail("There is no single command that upgrades this install.")
        if note:
            arrow(note)
        raise typer.Exit(code=1)

    if any(character in command for character in "&|;<>"):
        # A compound command is a shell instruction. Docker's is two commands joined by
        # `&&`; running it would mean invoking a shell on the user's behalf, which is
        # not something to do without them seeing the string first.
        fail("This install upgrades with a shell command, which is not run for you.")
        arrow("Run it yourself:")
        console.print(f"    {command}", style="bold")
        raise typer.Exit(code=1)

    parts = shlex.split(command)
    arrow(f"Running: {command}")
    console.print()
    try:
        os.execvp(parts[0], parts)
    except OSError as e:
        fail(f"Could not run {parts[0]!r}: {e}")
        arrow("Run it yourself:")
        console.print(f"    {command}", style="bold")
        raise typer.Exit(code=1) from e
