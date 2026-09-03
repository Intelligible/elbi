"""``elbi mcp``: serve the project's derivations over MCP on localhost."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from elbi_core import ChainedJsonlAuditSink, load_issuer
from elbi_core.errors import ElbiError

from .._console import arrow, console, fail
from ..guidance import compose_instructions
from ..mcp_server import build_server
from ..project import default_issuer_identity, load_project

if TYPE_CHECKING:
    from ..project import LoadedProject


def mcp(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(7878, help="Port to bind."),
    path: str = typer.Option("/mcp", help="HTTP path for the MCP endpoint."),
) -> None:
    """Boot a local, MCP-only server backed by your derivations (no chat UI).

    For most projects, `elbi serve` is the simpler default: it serves
    this same MCP endpoint alongside the built-in chat UI. Reach for `mcp`
    directly when you only want the MCP surface -- e.g. driving derivations from
    an external agent (Claude Desktop, Cursor, the MCP Inspector, your own
    agent) without the app's heavier dependencies installed.
    """
    base_dir = directory.resolve()
    try:
        project = load_project(base_dir)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    arrow(
        f"discovered {len(project.discovered)} derivation(s) in "
        f"{project.config.derivations_dir}/"
    )
    if project.authored:
        arrow(
            f"reloaded {len(project.authored)} agent-authored derivation(s) "
            "from .elbi/authored/"
        )
    if project.bindings_path.exists():
        arrow(f"resolved data from {project.bindings_path.name}")
    else:
        arrow(f"no {project.bindings_path.name} found; datasets are unbound")

    _report_search(project)

    issuer = load_issuer(
        project.certificate_key_dir, issuer=default_issuer_identity(project)
    )
    server = build_server(
        project.registry,
        project.make_runner,
        enable_propose=True,
        load_dataset=project.load_dataset,
        dataset_specs=project.config.datasets,
        name=project.config.project,
        authored_store=project.authored_store,
        issuer=issuer,
        audit=ChainedJsonlAuditSink(project.audit_log_path),
        run_log=project.run_log,
        cache_dir=project.cache_dir,
        scratch_dir=project.workspaces_dir / "shared",
        backend=project.config.sandbox,
        egress=project.config.egress,
        image=project.config.sandbox_image,
        instructions=compose_instructions(project.config.ai_context),
    )
    arrow(
        "analysis guidance active"
        + (" (+ project ai_context)" if project.config.ai_context else "")
    )

    url = f"http://{host}:{port}{path}"
    arrow(f"MCP server ready at  [bold]{url}[/bold]")
    console.print("[dim]Press Ctrl+C to stop. Re-run to pick up changes.[/dim]")

    try:
        # This process owns its socket, so it keeps the transport settings.
        server.run(
            transport="streamable-http", host=host, port=port, streamable_http_path=path
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("\nstopped.")


# Cosine floor for the semantic side: under the default static-embedding model
# unrelated texts score near 0 and real matches ~0.3+, so this drops weak matches
# rather than padding results. Calibrated to the shipped model.
def _report_search(project: LoadedProject) -> None:
    """Report the retriever the project loaded (installed in ``load_project``)."""
    if project.config.search == "lexical":
        arrow("search: lexical (BM25)")
        console.print(
            "[dim]   set search: hybrid in elbi.yaml (the default) to also "
            "match on meaning[/dim]"
        )
        return
    arrow("search: hybrid (BM25 + semantic embeddings, RRF-fused)")
    console.print("[dim]   the embedding model loads on the first search[/dim]")
