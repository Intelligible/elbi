"""Data-portability workspace export/import: one archive, both directions (IP-31).

``export`` writes one zip combining what ``pull`` already extracts -- metrics,
dashboards, feature views, monitors, schedules, workflows, checks, models, saved
queries, notebooks, certificates: the *definition* half -- with the derivation source
files from this checkout and a ``records/`` tree of the *evidence* config_sync has no
place for: a derivation's verdict, attestation, and result history; a dashboard's
current resolved values; a metric's version history.

``import`` reverses it: extract, then hand the definitions to the existing ``sync``
engine unchanged. Records are written to disk for inspection but never inserted --
they document a proof this instance's own oracle never produced, so a certificate is
verified against what this instance can certify, not replayed as if it had. See
docs/exports.md for the archive layout and the version contract.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import httpx
import typer

from .._console import arrow, fail, ok, warn
from ..config_sync import SyncError, surfaces
from ..config_sync import pull as pull_engine
from ..config_sync import sync as sync_engine
from ..http import LoginRequired, client_for
from .config_cmd import _report, resolve_host

#: The archive format's version this CLI writes and reads. A major bump (a shape
#: change import cannot degrade gracefully from) is refused by name; an addition
#: within a major is read by an older import as simply absent.
_SCHEMA_PREFIX = "elbi.export/v"
_SUPPORTED_MAJOR = 1


def _get(client: httpx.Client, path: str) -> Any:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json()


def _detail(resp: httpx.Response) -> str:
    """The server's own explanation for a failed response, else its status code.

    Never raises: it runs only where something already went wrong, and a body that is
    not JSON (a proxy's error page, an empty 502) must not replace the diagnosis with a
    traceback about parsing it.
    """
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}"
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return f"HTTP {resp.status_code}"


def _write_record(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _zip_dir(root: Path, out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root))


def _export(client: httpx.Client, root: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        arrow("pulling definitions (metrics, dashboards, ..., certificates)")
        pull_engine(client, stage)
        # pull records what it wrote in .elbi/state.json: content hashes for the
        # deployment it pulled FROM. Shipping that in the archive makes `import` diff a
        # fresh target against the source's state, skip surfaces the target never had,
        # and report "nothing to do" for objects that are simply missing. Sync state is
        # per-target; an archive is the one thing that crosses targets.
        shutil.rmtree(stage / ".elbi", ignore_errors=True)

        derivations_src = root / "derivations"
        if derivations_src.is_dir():
            shutil.copytree(
                derivations_src,
                stage / "derivations",
                dirs_exist_ok=True,
                # Compiled bytecode is not portable evidence, and a stale .pyc beside
                # the source it shadows is a hazard the archive has no reason to carry.
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        else:
            warn(
                "no derivations/ in this checkout -- the archive's code half will be "
                "evidence-only (records/derivations/*.source.py), not importable. "
                "Run export from the checkout that produced this deployment."
            )
        yaml_src = root / "elbi.yaml"
        if yaml_src.is_file():
            shutil.copy2(yaml_src, stage / "elbi.yaml")

        counts: dict[str, int] = {}
        for row in _get(client, "/api/derivations"):
            name = row["name"]
            document = _get(client, f"/api/exports/derivations/{name}")
            _write_record(stage / "records" / "derivations" / f"{name}.json", document)
            source = document.get("source") or ""
            if source:
                (stage / "records" / "derivations" / f"{name}.source.py").write_text(
                    f"# Recorded source of {name!r} -- a function body only, not a\n"
                    "# module: it will not import on its own (see docs/exports.md).\n"
                    f"{source}\n",
                    encoding="utf-8",
                )
            counts["derivations"] = counts.get("derivations", 0) + 1
        for row in _get(client, "/api/dashboards"):
            document = _get(client, f"/api/exports/dashboards/{row['id']}")
            _write_record(
                stage / "records" / "dashboards" / f"{row['name']}.json", document
            )
            counts["dashboards"] = counts.get("dashboards", 0) + 1
        for row in _get(client, "/api/metrics"):
            document = _get(client, f"/api/exports/metrics/{row['name']}")
            _write_record(
                stage / "records" / "metrics" / f"{row['name']}.json", document
            )
            counts["metrics"] = counts.get("metrics", 0) + 1
        arrow(
            "records: " + (", ".join(f"{n} {k}" for k, n in counts.items()) or "none")
        )

        manifest = {
            "schema": f"{_SCHEMA_PREFIX}{_SUPPORTED_MAJOR}",
            "surfaces": sorted(s.name for s in surfaces()),
            "records": sorted(counts),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        _zip_dir(stage, out)
    ok(f"export: wrote {out}")


def export(
    out: Path = typer.Option(
        ..., "--out", "-o", help="Path to write the workspace archive to."
    ),
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Repo directory.", exists=True
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
) -> None:
    """Export the workspace: every definition, plus the evidence behind it, as one zip.

    Definitions come from ``pull`` (metrics, dashboards, feature views, monitors,
    schedules, workflows, checks, models, saved queries, notebooks, certificates), so
    the archive round-trips through ``import`` the way a checkout does. Evidence -- a
    derivation's verdict and result history, a dashboard's current values, a metric's
    version history -- is fetched alongside into ``records/``; it documents the work
    and is never replayed by ``import``.

    Derivation source is copied from *this checkout's* ``derivations/*.py`` verbatim,
    because the app only stores a decorated function's source text, which does not
    import as a module on its own. Run this from the checkout that produced the
    deployment, or the archive's code half is evidence-only.
    """
    root = directory.resolve()
    try:
        host = resolve_host(root, target, url)
        with client_for(host, token) as client:
            _export(client, root, out)
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except SyncError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        fail(f"could not reach the app: {exc}")
        raise typer.Exit(code=1) from exc


def _check_schema(manifest: dict[str, Any]) -> None:
    schema = str(manifest.get("schema") or "")
    if not schema.startswith(_SCHEMA_PREFIX):
        raise SyncError(f"not an elbi export archive (schema: {schema!r})")
    major = schema[len(_SCHEMA_PREFIX) :].split(".", 1)[0]
    if not major.isdigit() or int(major) != _SUPPORTED_MAJOR:
        raise SyncError(
            f"archive schema {schema!r} is not supported by this CLI "
            f"(this version reads {_SCHEMA_PREFIX}{_SUPPORTED_MAJOR})"
        )


def _extract(archive: Path, into: Path) -> Path:
    """Safely unzip ``archive`` under ``into``; return the extracted project root.

    Reads and checks ``manifest.json`` before writing anything else, and refuses any
    member whose resolved path would land outside the extraction root (zip-slip) --
    an archive is untrusted input the moment it did not come from this run.
    """
    root = (into / archive.stem).resolve()
    with zipfile.ZipFile(archive) as zf:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except KeyError:
            raise SyncError(
                f"{archive}: no manifest.json -- not an export archive"
            ) from None
        _check_schema(manifest)
        for member in zf.infolist():
            target_path = (root / member.filename).resolve()
            if not target_path.is_relative_to(root):
                raise SyncError(
                    f"{archive}: refusing to extract {member.filename!r} -- "
                    "it escapes the archive root"
                )
        root.mkdir(parents=True, exist_ok=True)
        zf.extractall(root)
    # An archive written before export learned to leave this behind still carries the
    # source deployment's sync state. Drop it here too, so `sync` below diffs against
    # the target it is actually talking to rather than the one the archive came from.
    shutil.rmtree(root / ".elbi", ignore_errors=True)
    return root


def _verify_certificates(client: httpx.Client, root: Path) -> None:
    """Report whether each archived certificate matches this instance now.

    Never inserted: ``POST /api/certificates/{name}`` is the app's own verify gate
    (400 tampered, 409 stale, else ok) -- this only reports what it said. A clean
    target has nothing certified yet, so "not certified here yet" is the expected
    result until its own derivations are certified.
    """
    cert_dir = root / "certificates"
    if not cert_dir.is_dir():
        return
    for path in sorted(cert_dir.glob("*.json")):
        name = path.stem
        body = json.loads(path.read_text(encoding="utf-8"))
        resp = client.post(f"/api/certificates/{name}", json=body)
        if resp.status_code == 200:
            arrow(f"verified   certificates/{name}.json")
        elif resp.status_code == 404:
            warn(f"{name}: not certified on this instance yet -- not verified")
        elif resp.status_code == 409:
            warn(f"{name}: does not match this instance's current certification")
        else:
            # Cross-instance this is always 400: the route checks the signature against
            # THIS instance's key before it asks whether the name is certified here, and
            # a certificate signed elsewhere never matches. Carry the server's own
            # wording so a key mismatch does not read like a tampered certificate.
            warn(f"{name}: {_detail(resp)}")


def import_(
    archive: Path = typer.Argument(
        ..., help="Path to an exported .zip archive.", exists=True, dir_okay=False
    ),
    into: Path = typer.Option(
        Path(),
        "--into",
        "-C",
        help="Directory to extract the archive under.",
        exists=True,
    ),
    url: str | None = typer.Option(None, "--url", help="App base URL."),
    token: str | None = typer.Option(None, "--token", help="API key, if required."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Which declared target to act on."
    ),
) -> None:
    """Import a workspace archive: extract, then sync every definition into the app.

    Definitions are applied through the same ``sync`` engine a real checkout uses.
    Certificates are excluded from that sync and verified separately instead --
    pushing them the ordinary way would 404 against a clean instance, which has
    nothing certified yet to compare them to.

    Records land on disk under ``records/`` for inspection; nothing in them is
    inserted. Derivation source (``derivations/*.py``) still needs delivering to the
    target the way any derivation does -- baked into its image or pulled from git
    (see docs/deployment/project-delivery.md) -- ``import`` does not restart the app.
    """
    try:
        root = _extract(archive, into.resolve())
        host = resolve_host(root, target, url)
        with client_for(host, token) as client:
            defs = [s for s in surfaces() if s.name != "certificates"]
            _report(sync_engine(client, root, surfaces_=defs), "import")
            _verify_certificates(client, root)
        ok(f"import: extracted to {root}; records were written, not inserted")
    except LoginRequired as exc:
        fail(str(exc) or "the app refused this credential")
        raise typer.Exit(code=1) from exc
    except SyncError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
