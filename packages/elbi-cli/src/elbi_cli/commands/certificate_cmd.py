"""``elbi certificate``: issue, verify, and inspect verification certificates.

The point of a certificate is that it checks offline: ``verify`` loads no project and
needs no running system, only the file and, for independent trust, a separately
distributed public key.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from elbi_core import load_issuer, load_or_create_key, verify_chain
from elbi_core.certificate import verify_certificate
from elbi_core.errors import CertificateError, ElbiError

from .._console import arrow, fail, ok, warn
from ..project import LoadedProject, default_issuer_identity, load_project

certificate_app = typer.Typer(
    name="certificate",
    help="Issue, verify, and inspect tamper-evident verification certificates.",
    no_args_is_help=True,
)

_DIRECTORY = typer.Option(
    Path(), "--directory", "-C", help="Project directory.", exists=True
)


def _load(directory: Path) -> LoadedProject:
    """Load the project at ``directory``, or exit non-zero with the reason."""
    try:
        return load_project(directory.resolve())
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc


@certificate_app.command("issue")
def issue(
    name: str = typer.Argument(..., help="The derivation to certify."),
    directory: Path = _DIRECTORY,
    version: str | None = typer.Option(
        None, "--version", help="Certify the run whose version starts with this prefix."
    ),
    issuer: str | None = typer.Option(
        None, "--issuer", help="The certifier identity (defaults to the OS user)."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the certificate here (default: stdout)."
    ),
) -> None:
    """Issue a signed certificate for a derivation's newest certified run."""
    project = _load(directory)
    runs = project.run_log.runs(name=name)
    if version is not None:
        runs = [run for run in runs if run.derivation_version.startswith(version)]
    if not runs:
        fail(f"no certified run found for {name!r}")
        raise typer.Exit(code=1)
    try:
        signer = load_issuer(
            project.certificate_key_dir,
            issuer=issuer or default_issuer_identity(project),
        )
        document = signer.issue(runs[0])
    except CertificateError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.write_text(rendered, encoding="utf-8")
        ok(f"wrote certificate for {name!r} to {output}")
    else:
        typer.echo(rendered, nl=False)


@certificate_app.command("verify")
def verify(
    certificate: Path = typer.Argument(
        ..., help="The certificate file to verify.", exists=True
    ),
    public_key: str | None = typer.Option(
        None,
        "--public-key",
        help="Expected public key: a base64 string or a path to a .pub file.",
    ),
) -> None:
    """Verify a certificate offline; exit non-zero if it does not check out."""
    try:
        document = json.loads(certificate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"could not read certificate: {exc}")
        raise typer.Exit(code=1) from exc
    expected = _resolve_public_key(public_key)
    if expected is None:
        warn("self-consistency only: no trust anchor checked; pass --public-key")
    try:
        cert = verify_certificate(document, public_key=expected)
    except CertificateError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    ok(f"certificate verified for {cert.derivation!r}")
    arrow(f"verdict: {cert.verdict}")
    if cert.estimate is not None:
        label = cert.estimate_label or "estimate"
        arrow(f"{label}: {cert.estimate}")
    arrow(f"issued by {cert.issuer} at {cert.issued_at}")


@certificate_app.command("chain")
def chain(
    directory: Path = _DIRECTORY,
    path: Path | None = typer.Option(
        None, "--path", help="Verify this audit log instead of the project's."
    ),
) -> None:
    """Verify the hash-chained audit log; exit non-zero if the chain is broken."""
    if path is None:
        path = _load(directory).audit_log_path
    report = verify_chain(path)
    arrow(f"records: {report.records} ({report.chained} chained)")
    if report.head is not None:
        arrow(f"head: {report.head}")
    if not report.ok:
        fail(f"chain broken: {report.error}")
        raise typer.Exit(code=1)
    ok("audit chain intact")


@certificate_app.command("public-key")
def public_key(
    directory: Path = _DIRECTORY,
) -> None:
    """Print the project's certificate public key, creating the keypair if absent."""
    project = _load(directory)
    try:
        key = load_or_create_key(project.certificate_key_dir)
    except CertificateError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(key.public_key)


def _resolve_public_key(value: str | None) -> str | None:
    """A public key from a base64 string or a ``.pub`` file, or ``None``."""
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return value
