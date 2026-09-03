"""The ``elbi-app`` command line: run the verifying chat app."""

from __future__ import annotations

import json as _json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import typer

app = typer.Typer(
    name="elbi-app",
    help="Run the verifying analysis chat app (and the project's MCP endpoint).",
    no_args_is_help=True,
)


#: Environment variables whose values must never be printed. Matched case-insensitively
#: as substrings, so ``LLM_API_KEY`` and ``AWS_SECRET_ACCESS_KEY`` are both covered.
_SECRET_HINTS = ("secret", "password", "passwd", "token", "key", "credential")

#: Credentials embedded in a URL. Name-based matching alone is not enough: ``DB_URI``
#: contains a password but none of the hint words, so its value has to be scrubbed on
#: shape rather than on the name it happens to be stored under.
_URL_CREDENTIALS = re.compile(
    r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<user>[^:/@\s]+):[^@/\s]+@"
)


def _scrub(value: str) -> str:
    """Remove URL-embedded passwords anywhere in ``value``.

    Applied to error text as well as environment values: driver errors sometimes echo
    the connection string they failed on, and a support archive is the wrong place for
    it to turn up.
    """
    return _URL_CREDENTIALS.sub(r"\g<scheme>\g<user>:***@", value)


def _redact_uri(uri: str) -> str:
    """Strip the password from a connection string, keeping the shape readable."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "<unparseable>"
    if not parts.hostname:
        # No authority component (a SQLite path); nothing sensitive to remove.
        return uri
    user = parts.username or ""
    auth = f"{user}:***@" if parts.password else (f"{user}@" if user else "")
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit(
        (parts.scheme, f"{auth}{parts.hostname}{port}", parts.path, "", "")
    )


def _collect_diagnostics() -> dict[str, Any]:
    """Gather support-facing facts.

    Never returns a secret value: only whether one is set.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    # "elbi" is the distribution; "elbi-app" is only the console script.
    try:
        app_version = pkg_version("elbi")
    except PackageNotFoundError:
        app_version = "unknown"

    report: dict[str, Any] = {
        "version": app_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "uid": os.getuid(),
        "gid": os.getgid(),
    }

    # Store: the most common failure, so report the dialect and server version rather
    # than a bare boolean. Connects directly instead of via open_store(): that would
    # apply migrations, and a diagnostics dump must not change what it reports on.
    from sqlalchemy import text
    from sqlmodel import create_engine

    from .db import DEFAULT_DB_URI, sqlalchemy_url

    db_uri = os.environ.get("DB_URI") or DEFAULT_DB_URI
    url = sqlalchemy_url(db_uri)
    store_info: dict[str, Any] = {"uri": _redact_uri(url)}
    try:
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                store_info["reachable"] = True
                store_info["dialect"] = conn.dialect.name
                store_info["server_version"] = ".".join(
                    str(p) for p in (conn.dialect.server_version_info or ())
                )
        finally:
            engine.dispose()
    # Deliberately broad: an unreachable store is the thing being reported, and a
    # diagnostics dump that raises is worthless precisely when it is needed.
    except Exception as exc:
        store_info["reachable"] = False
        store_info["error"] = _scrub(f"{type(exc).__name__}: {exc}")
    report["store"] = store_info

    # Warehouse: report the configured root. Remote reachability is deliberately not
    # probed: a hanging network call is the last thing a support dump should do.
    from .warehouse.storage import storage_uri

    root = storage_uri()
    warehouse: dict[str, Any] = {"storage_uri": _redact_uri(root)}
    if root.startswith("file://"):
        local = Path(root[len("file://") :])
        warehouse["local_path_exists"] = local.exists()
        # The warehouse directory is created on first write, so testing it directly
        # reports False for a fresh install. Walk up to the nearest ancestor that
        # exists: that is the one whose permissions decide whether creation will
        # succeed.
        probe = local
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        warehouse["local_path_writable"] = os.access(probe, os.W_OK)
        warehouse["durable"] = False
        warehouse["note"] = "local filesystem, not durable across container restarts"
    else:
        warehouse["durable"] = True
    report["warehouse"] = warehouse

    report["configured"] = {
        "llm_model": os.environ.get("LLM_MODEL") or None,
        "llm_base_url": os.environ.get("LLM_BASE_URL") or None,
        "llm_api_key_set": bool(os.environ.get("LLM_API_KEY")),
        "smtp": bool(os.environ.get("SMTP_HOST")),
        "proxy": {
            var: os.environ.get(var)
            for var in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
            if os.environ.get(var)
        },
        "custom_ca": {
            var: os.environ.get(var)
            for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "AWS_CA_BUNDLE")
            if os.environ.get(var)
        },
    }

    # Every variable the process can see. Knowing a variable is set is often the whole
    # answer; its value rarely is. Two passes, because either alone leaks: the name
    # check misses DB_URI, and the shape check misses a bare API key.
    report["environment"] = {
        name: (
            "<set>" if any(h in name.lower() for h in _SECRET_HINTS) else _scrub(value)
        )
        for name, value in sorted(os.environ.items())
    }
    return report


@app.command()
def diagnostics(
    as_json: bool = typer.Option(
        False, "--json", help="Emit JSON instead of text (for a support bundle)."
    ),
) -> None:
    """Print environment and connectivity facts for a support ticket.

    Secret values are never printed: sensitive variables are reported as ``<set>``, and
    connection strings have their password removed.
    """
    report = _collect_diagnostics()
    if as_json:
        typer.echo(_json.dumps(report, indent=2, sort_keys=True))
        return

    typer.echo(f"elbi {report['version']}  (python {report['python']})")
    typer.echo(
        f"platform: {report['platform']}  uid={report['uid']} gid={report['gid']}"
    )
    store = report["store"]
    status = "reachable" if store.get("reachable") else "UNREACHABLE"
    typer.echo(f"\nstore: {status}  {store['uri']}")
    if store.get("dialect"):
        typer.echo(f"  {store['dialect']} {store.get('server_version', '')}".rstrip())
    if store.get("error"):
        typer.echo(f"  error: {store['error']}")
    wh = report["warehouse"]
    typer.echo(f"\nwarehouse: {wh['storage_uri']}")
    if wh.get("note"):
        typer.echo(f"  {wh['note']}")
    cfg = report["configured"]
    model = cfg["llm_model"] or "-"
    typer.echo(f"\nconfigured: model={model} smtp={cfg['smtp']}")
    if cfg["proxy"]:
        typer.echo(f"  proxy: {cfg['proxy']}")
    if cfg["custom_ca"]:
        typer.echo(f"  custom CA: {cfg['custom_ca']}")
    typer.echo(
        f"\n{len(report['environment'])} environment variables (values redacted)"
    )


@app.command()
def migrate() -> None:
    """Apply schema migrations to the store named by ``DB_URI``, then exit.

    Serving applies migrations itself, so this is not required. It exists so a
    deployment can bring the schema forward as its own step (a Kubernetes init
    container, a Helm pre-upgrade hook, or a hand-run command before a rollout) and see
    it fail loudly there rather than in a pod that then refuses to become ready.
    """
    from .db import open_store

    open_store()
    typer.echo("Schema is up to date.")


@app.command()
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
    """Serve the chat app; opens a browser at the printed URL once ready."""
    from .serve import serve as run_serve

    run_serve(
        directory,
        host=host,
        port=port,
        model=model or None,
        open_browser=not no_browser,
    )
