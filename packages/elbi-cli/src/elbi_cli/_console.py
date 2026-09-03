"""Shared console output helpers."""

from __future__ import annotations

from rich.console import Console

console = Console()
err_console = Console(stderr=True)


#: What a table cell shows when it has no value. An em dash reads as "no value"
#: where a hyphen would read as a minus sign in a numeric column.
EMPTY = "—"


def arrow(message: str) -> None:
    """Print a progress line in the framework's ``→`` style."""
    console.print(f"[cyan]→[/cyan] {message}")


def ok(message: str) -> None:
    """Print a success line."""
    console.print(f"[green]✓[/green] {message}")


def warn(message: str) -> None:
    """Print a warning line to stderr."""
    err_console.print(f"[yellow]![/yellow] {message}")


def fail(message: str) -> None:
    """Print an error line to stderr."""
    err_console.print(f"[red]✗[/red] {message}")
