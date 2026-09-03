"""``elbi serve``: launch the web app for this project.

The app is a separate, heavier package (`elbi`: FastAPI + the web UI + warehouse). It is
imported lazily so the CLI stays lean; if it is not installed, this prints how to get
it, following the same pattern as `fastapi run` from `fastapi-cli`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .._console import fail


def serve(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(7700, help="Port to bind."),
    model: str = typer.Option(
        "", help="The LLM model to run (else a Settings profile or LLM_MODEL)."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't open a browser once the app is ready."
    ),
) -> None:
    """Serve the chat UI and the MCP endpoint for this project.

    The default way to run a project: opens a browser at the printed URL once
    ready, or point an MCP client at it. Requires the ``elbi`` app
    package -- if it is absent, prints how to install it. For a lighter,
    MCP-only server with no chat UI, use `elbi mcp` instead.
    """
    try:
        from elbi.serve import serve as run_serve
    except ImportError:
        fail(
            "the web app is not installed. Install it, then re-run:\n"
            "    pip install elbi      # or:  uv add elbi"
        )
        raise typer.Exit(code=1) from None
    run_serve(
        directory,
        host=host,
        port=port,
        model=model or None,
        open_browser=not no_browser,
    )
