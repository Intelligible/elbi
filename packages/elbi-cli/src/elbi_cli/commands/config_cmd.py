"""Analytics-as-code CLI: sync a repo to a running app, pull it back, and inspect it.

``elbi sync`` pushes the repo's declarative files (metrics, dashboards, feature
views, monitors, schedules, workflows, checks, models, saved queries, notebooks) to a
running app over its
HTTP API; ``pull`` writes the app's objects back as canonical files; ``plan`` previews
the diff; ``schema`` prints the warehouse schema so an agent can see the data, or with
``--json`` emits a JSON Schema per artifact so an agent authors valid files. The app's
own routes do validation and gating, so these commands are a thin file<->API mapper.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import typer

from .._console import arrow, console, fail, ok, warn
from ..config_sync import Change, SyncError
from ..config_sync import plan as plan_engine
from ..config_sync import pull as pull_engine
from ..config_sync import reload_project as reload_engine
from ..config_sync import schemas as schemas_engine
from ..config_sync import sync as sync_engine
from ..config_sync import validate as validate_engine
from ..http import CONFIG_TIMEOUT, DEFAULT_URL, LoginRequired, client_for

#: The running app to target; overridable per invocation or via the environment.


def resolve_host(root: Path, target: str | None, url: str | None) -> str:
    """Which deployment to talk to, and refuse if the answer is ambiguous or wrong.

    A project declaring ``targets`` says where it belongs, so a command against it is
    checked rather than trusted. That is the point: one environment variable otherwise
    serves a personal install and a company deployment alike, so a stale value pushes
    personal work into production with nothing able to notice.

    Precedence, once targets exist: a named target wins, then the default one, and an
    explicit ``--url`` or ``ELBI_URL`` must agree with it. A project declaring
    no targets is unconstrained, the case for a laptop and for every repo predating
    this.
    """
    declared: tuple[Any, ...] = ()
    try:
        from ..project import load_project

        declared = load_project(root).config.targets
    except Exception:
        # No project here, or one that will not load: not this command's error to raise.
        # `validate` and the engines report that with the context to act on it.
        declared = ()

    given = url or os.environ.get("ELBI_URL")
    if not declared:
        if target:
            raise SyncError(
                f"no target named {target!r}: this project declares none. "
                "Add a `targets:` mapping to elbi.yaml, or pass --url."
            )
        return (given or DEFAULT_URL).rstrip("/")

    by_name = {spec.name: spec for spec in declared}
    if target is not None:
        chosen = by_name.get(target)
        if chosen is None:
            raise SyncError(
                f"no target named {target!r}. This project declares "
                f"{', '.join(sorted(by_name))}."
            )
    else:
        defaults = [spec for spec in declared if spec.default]
        if not defaults:
            raise SyncError(
                "this project declares targets but none is the default, so there is "
                f"nothing to mean by default. Name one with -t, or mark one "
                f"`default: true` ({', '.join(sorted(by_name))})."
            )
        chosen = defaults[0]

    if given and given.rstrip("/") != chosen.host:
        raise SyncError(
            f"target {chosen.name!r} is {chosen.host}, but {given.rstrip('/')} was "
            "given. Refusing rather than guessing which one you meant: pass the "
            f"matching target with -t, or drop the url."
        )
    return chosen.host


def _report(changes: list[Change], verb: str) -> None:
    """Print a grouped summary of applied/planned changes (nothing when empty)."""
    active = [c for c in changes if c.action != "unchanged"]
    if not active:
        ok(f"{verb}: nothing to do, repo and app match.")
        return
    for change in active:
        arrow(f"{change.action:8} {change.surface}/{change.name}")
    # Drift and conflict are findings, not changes: reporting "3 change(s)" when two of
    # them are things sync will not touch would misstate what happens next.
    doing = [c for c in active if c.action not in ("drift", "conflict")]
    found = len(active) - len(doing)
    summary = f"{verb}: {len(doing)} change(s)."
    if found:
        summary += f" {found} left alone (changed in the app; pull to take them)."
    ok(summary)


def _report_reload(summary: dict[str, Any]) -> None:
    """Print what re-reading the project from disk changed (derivations + data)."""
    derivations = summary.get("derivations", {})
    for action in ("added", "reloaded", "removed"):
        for name in derivations.get(action, []):
            arrow(f"{action:8} derivations/{name}")
    sources = summary.get("sources", [])
    if sources:
        arrow(f"{'synced':8} sources: {', '.join(sources)}")


def sync(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Repo directory.", exists=True
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
    prune: bool = typer.Option(
        False, "--prune", help="Delete app objects the repo omits (app matches repo)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the plan without applying it."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite objects changed in the app since the last pull.",
    ),
) -> None:
    """Apply the repo's declarative files to a running app (repo -> app).

    Refuses if anything changed both here and in the app since the last pull, naming
    what did. ``--force`` overwrites those, discarding the app's version.
    """
    root = directory.resolve()
    problems = validate_engine(root)
    if problems:
        for problem in problems:
            fail(problem)
        raise typer.Exit(code=1)
    try:
        with client_for(resolve_host(root, target, url), token) as client:
            if dry_run:
                _report(plan_engine(client, root, prune=prune), "plan")
                return
            _report(sync_engine(client, root, prune=prune, force=force), "sync")
            # Derivation source files and declared source data are code and data, not
            # declarative config, so the app re-reads them from disk to go live.
            _report_reload(reload_engine(client))
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except SyncError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc


def pull(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Repo directory.", exists=True
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
) -> None:
    """Write a running app's objects back to the repo as files (app -> repo)."""
    root = directory.resolve()
    try:
        with client_for(resolve_host(root, target, url), token) as client:
            _report(pull_engine(client, root), "pull")
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except SyncError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc


def plan(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Repo directory.", exists=True
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
    prune: bool = typer.Option(
        False, "--prune", help="Preview the pruning sync, which also deletes extras."
    ),
) -> None:
    """Preview the changes ``sync`` would make (diff repo vs app).

    Mirrors the flags of the sync it predicts: removals appear only with ``--prune``,
    because only that sync performs them.
    """
    root = directory.resolve()
    problems = validate_engine(root)
    for problem in problems:
        warn(problem)
    try:
        with client_for(resolve_host(root, target, url), token) as client:
            _report(plan_engine(client, root, prune=prune), "plan")
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except SyncError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc


def schema(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Repo directory (for --json output)."
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Write each artifact's JSON Schema to .elbi/schemas/ instead.",
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
) -> None:
    """Print the warehouse schema, or emit artifact JSON Schemas with ``--json``.

    Without ``--json`` this prints the live warehouse schema (tables, columns, types) so
    an agent knows what data to build against. With ``--json`` it writes one JSON Schema
    per artifact folder to ``.elbi/schemas/``: the contract an agent validates a metric,
    dashboard, monitor, or model file against before syncing. The schemas come from the
    installed package, so ``--json`` needs no running app.
    """
    if json_out:
        out_dir = directory.resolve() / ".elbi" / "schemas"
        out_dir.mkdir(parents=True, exist_ok=True)
        for folder, spec in schemas_engine().items():
            path = out_dir / f"{folder}.json"
            path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
            arrow(f"{path.relative_to(directory.resolve())}")
        ok(f"schema: wrote {len(schemas_engine())} JSON Schema(s).")
        return
    try:
        host = resolve_host(directory.resolve(), target, url)
        with client_for(host, token) as client:
            resp = client.get("/api/explore/catalog", params={"source_id": "warehouse"})
            resp.raise_for_status()
            tables = resp.json().get("tables", [])
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        fail(f"could not reach the app: {exc}")
        raise typer.Exit(code=1) from exc
    if not tables:
        warn("The warehouse has no tables yet. Add a source and sync it first.")
        return
    for table in tables:
        cols = ", ".join(
            f"{c['name']}:{c.get('type', '?')}" for c in table.get("columns", [])
        )
        rows = table.get("rows")
        count = f" ({rows} rows)" if rows is not None else ""
        arrow(f"{table['name']}{count}: {cols}")


def upload(
    file: Path = typer.Argument(
        ...,
        help="Local CSV or Parquet file to upload.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Repo directory (for target resolution)."
    ),
    as_table: str | None = typer.Option(
        None,
        "--as-table",
        help="Also register the upload as a source of this name and load it.",
    ),
) -> None:
    """Upload a local CSV/Parquet file into the app's warehouse.

    Streams the file to the app, which stores it in warehouse storage (a local directory
    or the object store behind it) and returns the path. The file never has to be
    reachable by the app: you send it, so this works from a laptop or a dev server
    whether or not either can reach the object store.

    ``--as-table NAME`` finishes the job: it points a source of that name at the
    uploaded path and loads it, so the file lands as a queryable table in one command.
    Without it the upload is only staged, which is the two-step every warehouse CLI has
    (stage, then load) and worth keeping for a file several sources will read.
    """
    suffix = file.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        fail(f"unsupported file type {suffix or '(none)'!r}; upload a .csv or .parquet")
        raise typer.Exit(code=2)
    content_type = "text/csv" if suffix == ".csv" else "application/vnd.apache.parquet"
    try:
        host = resolve_host(directory.resolve(), target, url)
        with client_for(host, token) as client, _progress_file(file) as stream:
            resp = client.post(
                "/api/warehouse/uploads",
                files={"file": (file.name, stream, content_type)},
                # A large upload runs as long as it runs; do not cut it off.
                timeout=None,
            )
        resp.raise_for_status()
        body = resp.json()
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            fail("the app refused this upload")
        else:
            fail(_http_detail(exc.response) or f"the app refused the upload: {exc}")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        fail(f"could not reach the app: {exc}")
        raise typer.Exit(code=1) from exc
    ok(f"uploaded {file.name}")
    path = str(body.get("path", ""))
    arrow(f"path: {path}")
    if not as_table:
        console.print(
            "Point a CSV/Parquet source at that path, or pick the file in the app, "
            "to make it a table. Or pass --as-table NAME to do it here."
        )
        return
    _load_upload_as_table(host, token, as_table, path, suffix.lstrip("."))


def _load_upload_as_table(
    host: str, token: str | None, name: str, path: str, source_type: str
) -> None:
    """Register an uploaded file as a warehouse source of ``name``, and load it."""
    try:
        with client_for(host, token) as client:
            created = client.post(
                "/api/warehouse/sources",
                json={
                    "source_type": source_type,
                    "name": name,
                    "config": {"path": path},
                },
                timeout=CONFIG_TIMEOUT,
            )
            created.raise_for_status()
            source = created.json()
            synced = client.post(
                f"/api/warehouse/sources/{source['id']}/sync",
                json={},
                timeout=None,  # loading is as long as the file is
            )
            synced.raise_for_status()
            outcomes = synced.json().get("outcomes", [])
    except httpx.HTTPStatusError as exc:
        fail(_http_detail(exc.response) or f"the app refused the source: {exc}")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        fail(f"could not reach the app: {exc}")
        raise typer.Exit(code=1) from exc
    failures = [o for o in outcomes if not o.get("ok")]
    for outcome in outcomes:
        table, rows = outcome.get("table", "?"), outcome.get("rows")
        if outcome.get("ok"):
            arrow(f"table {table}: {rows:,} rows" if rows else f"table {table}")
        else:
            arrow(f"table {table}: {outcome.get('error') or 'failed'}")
    if failures:
        fail(f"source {name!r} was created but did not load")
        raise typer.Exit(code=1)
    ok(f"loaded as source {name!r}")


def _http_detail(response: httpx.Response) -> str:
    """The server's ``detail`` from an error response, or empty if there is none."""
    try:
        body = response.json()
    except ValueError:
        return ""
    return str(body.get("detail", "")) if isinstance(body, dict) else ""


class _CountingReader:
    """A read-through wrapper that reports bytes read, for an upload progress bar.

    Delegates ``fileno``/``seek``/``tell`` so httpx still measures the file with one
    ``fstat`` and streams it -- wrapping without those would make httpx read the whole
    file into memory to find its length, the opposite of the point.
    """

    def __init__(self, raw: Any, advance: Callable[[int], None]) -> None:
        self._raw = raw
        self._advance = advance

    def read(self, size: int = -1) -> bytes:
        chunk: bytes = self._raw.read(size)
        if chunk:
            self._advance(len(chunk))
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        return int(self._raw.seek(offset, whence))

    def tell(self) -> int:
        return int(self._raw.tell())

    def fileno(self) -> int:
        return int(self._raw.fileno())

    def close(self) -> None:
        self._raw.close()


@contextlib.contextmanager
def _progress_file(path: Path) -> Iterator[Any]:
    """The file open for reading, wrapped to show an upload progress bar on a terminal.

    The bar renders only when stderr is a terminal, so a script or CI run stays quiet;
    the byte counting is harmless either way. Progress goes to stderr so the printed
    path stays on stdout, clean for a pipe.
    """
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    from rich.progress import Progress as RichProgress

    size = path.stat().st_size
    with (
        path.open("rb") as raw,
        RichProgress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=Console(stderr=True),
            disable=not sys.stderr.isatty(),
        ) as progress,
    ):
        task = progress.add_task(f"uploading {path.name}", total=size or None)
        yield _CountingReader(raw, lambda n: progress.advance(task, n))
