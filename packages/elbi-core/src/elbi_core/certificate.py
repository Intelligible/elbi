"""Exportable, tamper-evident verification certificates.

A :class:`Certificate` is a canonical, signed statement of what the verification
oracle certified for one derivation: the verdict, the gates that ran and were
skipped, the certified estimate, the content-addressed identity of the code and
data, and who certified it, when. It is built from the verdict model already
recorded on a :class:`~elbi_core.tracking.run.CertifiedRun`; nothing here
recomputes a verdict.

Tamper-evidence is a detached Ed25519 signature over the certificate's canonical
JSON bytes. The signature travels in an envelope beside the payload:

    {
      "schema": "elbi.certificate/v1",
      "certificate": { ...the signed payload... },
      "signature": {"algorithm": "ed25519", "value": "<base64>"}
    }

Because the signature covers only ``canonical_json(payload)`` (sorted keys, fixed
separators), an exported file may be pretty-printed without breaking verification:
:func:`verify_certificate` re-canonicalizes the parsed payload before checking. The
signing public key rides inside the payload, so it too is signed; independent
verification against a separately distributed public key
(:meth:`SigningKey.public_key`) is what makes a re-signed forgery detectable.

Signing needs asymmetric crypto, which the standard library lacks, so it lives
behind the ``certificate`` extra (``cryptography``). Verifying the audit-log chain
needs no extra; see :mod:`elbi_core.audit`.
"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import CertificateError
from .tracking.run import CertifiedRun, Check
from .versioning import canonical_json

if TYPE_CHECKING:
    # Only for type-checking: cryptography is optional at runtime (the certificate
    # extra), and this module must import cleanly without it (see _crypto below).
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

CERTIFICATE_SCHEMA = "elbi.certificate/v1"
SIGNATURE_ALGORITHM = "ed25519"

#: Private-key filename under a project's ``.elbi`` dir; the base64 public
#: key is written beside it as ``<name>.pub``.
KEY_FILENAME = "certificate.key"
#: A path override for the private key, for app or CI deployments.
KEY_ENV = "ELBI_CERTIFICATE_KEY"

_SEED_LEN = 32

#: A skipped gate's ``(name, reason)`` pair.
SkipReason = tuple[str, str]


def _crypto() -> tuple[Any, Any]:
    """The Ed25519 primitives and the signature error, or a hinted failure.

    Imported lazily so importing this module never requires the optional
    ``cryptography`` dependency; only signing and verifying do.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - exercised via the seam in tests
        raise CertificateError(
            "certificates require the 'certificate' extra: "
            'pip install "elbi-core[certificate]"'
        ) from exc
    return ed25519, InvalidSignature


class SigningKey:
    """An Ed25519 private key, reconstructed from its 32-byte seed.

    The seed is the durable secret persisted by :func:`load_or_create_key`.
    Signing is deterministic (RFC 8032), so the same key over the same bytes
    yields the same signature.
    """

    def __init__(self, seed: bytes) -> None:
        if len(seed) != _SEED_LEN:
            raise CertificateError(
                f"a signing key seed must be {_SEED_LEN} bytes, got {len(seed)}"
            )
        ed25519, _ = _crypto()
        self._private: Ed25519PrivateKey = ed25519.Ed25519PrivateKey.from_private_bytes(
            seed
        )

    @property
    def public_key(self) -> str:
        """The base64-encoded raw Ed25519 public key, for distribution."""
        from cryptography.hazmat.primitives import serialization

        raw = self._private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return _b64encode(raw)

    def sign(self, data: bytes) -> bytes:
        """The detached Ed25519 signature over ``data``."""
        return self._private.sign(data)


def load_or_create_key(root: str | Path) -> SigningKey:
    """Load the project's signing key, creating it on first use.

    The private seed lives at ``<root>/certificate.key`` (mode ``0o600``), or at
    the path in ``ELBI_CERTIFICATE_KEY`` when set. A ``.pub`` sibling with
    the base64 public key is written for distribution. Creation follows the
    exclusive-create-then-read-the-winner pattern used for the cache key, so a
    concurrent first write cannot clobber another.
    """
    override = os.environ.get(KEY_ENV)
    key_path = Path(override) if override else Path(root) / KEY_FILENAME
    try:
        seed = key_path.read_bytes()
    except FileNotFoundError:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        seed = secrets.token_bytes(_SEED_LEN)
        try:  # exclusive create; lose the race -> read the winner's seed
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # pragma: no cover - concurrent first-write race
            seed = key_path.read_bytes()
        else:
            with os.fdopen(fd, "wb") as handle:
                handle.write(seed)
    key = SigningKey(seed)
    _write_public_key(key_path, key)
    return key


def _write_public_key(key_path: Path, key: SigningKey) -> None:
    """Write the base64 public key beside the private key (idempotent)."""
    pub_path = key_path.with_name(key_path.name + ".pub")
    text = key.public_key + "\n"
    if pub_path.exists() and pub_path.read_text(encoding="utf-8") == text:
        return
    pub_path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class Certificate:
    """The signed payload: what the oracle certified for one derivation.

    Every field is copied from a :class:`CertifiedRun` (the verdict model) plus the
    certifier's identity and the time and key of issue; nothing is recomputed.
    ``skipped`` carries ``(name, reason)`` pairs; a run records only names, so a
    certificate built from one carries an empty reason unless a caller holding the
    live report supplies it.
    """

    derivation: str
    question: str
    verdict: str
    claim: dict[str, str]
    estimate: float | None
    estimate_label: str | None
    adjusted_for: tuple[str, ...]
    checks: tuple[Check, ...]
    skipped: tuple[SkipReason, ...]
    data_hash: str | None
    derivation_version: str
    code_version: str | None
    input_versions: dict[str, str]
    issued_at: str
    issuer: str
    public_key: str

    def to_payload(self) -> dict[str, Any]:
        """The JSON-native payload dict signed and exported (tuples become lists)."""
        return {
            "derivation": self.derivation,
            "question": self.question,
            "verdict": self.verdict,
            "claim": dict(self.claim),
            "estimate": self.estimate,
            "estimate_label": self.estimate_label,
            "adjusted_for": list(self.adjusted_for),
            "checks": [list(c) for c in self.checks],
            "skipped": [list(s) for s in self.skipped],
            "data_hash": self.data_hash,
            "derivation_version": self.derivation_version,
            "code_version": self.code_version,
            "input_versions": dict(self.input_versions),
            "issued_at": self.issued_at,
            "issuer": self.issuer,
            "public_key": self.public_key,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> Certificate:
        """Rebuild a certificate from a payload dict, or raise on a malformed one."""
        try:
            return cls(
                derivation=data["derivation"],
                question=data["question"],
                verdict=data["verdict"],
                claim=dict(data["claim"]),
                estimate=data["estimate"],
                estimate_label=data["estimate_label"],
                adjusted_for=tuple(data["adjusted_for"]),
                checks=tuple(tuple(c) for c in data["checks"]),
                skipped=tuple((s[0], s[1]) for s in data["skipped"]),
                data_hash=data["data_hash"],
                derivation_version=data["derivation_version"],
                code_version=data["code_version"],
                input_versions=dict(data["input_versions"]),
                issued_at=data["issued_at"],
                issuer=data["issuer"],
                public_key=data["public_key"],
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise CertificateError(f"malformed certificate payload: {exc}") from exc


def certificate_from_run(
    run: CertifiedRun,
    *,
    issuer: str,
    public_key: str,
    issued_at: str | None = None,
    skipped: tuple[SkipReason, ...] = (),
) -> Certificate:
    """Build a certificate from a certified run, recomputing nothing.

    ``issued_at`` defaults to the run's own ``created_at`` so re-issuing the same
    run yields byte-identical bytes (and, with Ed25519, an identical signature).
    ``skipped`` overrides the run's skipped names when a caller holds the reasons.
    """
    resolved_skipped = skipped or tuple((name, "") for name in run.skipped)
    return Certificate(
        derivation=run.name,
        question=run.question,
        verdict=run.verdict,
        claim=dict(run.claim),
        estimate=run.estimate,
        estimate_label=run.estimate_label,
        adjusted_for=tuple(run.adjusted_for),
        checks=tuple(run.checks),
        skipped=resolved_skipped,
        data_hash=run.data_hash,
        derivation_version=run.derivation_version,
        code_version=run.code_version,
        input_versions=dict(run.input_versions),
        issued_at=issued_at or run.created_at,
        issuer=issuer,
        public_key=public_key,
    )


@dataclass(frozen=True)
class CertificateIssuer:
    """Signs certificates with one key under one certifier identity."""

    key: SigningKey
    issuer: str

    @property
    def public_key(self) -> str:
        """The base64 public key that verifies this issuer's signatures."""
        return self.key.public_key

    def sign(self, certificate: Certificate) -> dict[str, Any]:
        """Wrap a certificate in a signed envelope."""
        payload = certificate.to_payload()
        signature = self.key.sign(canonical_json(payload).encode("utf-8"))
        return {
            "schema": CERTIFICATE_SCHEMA,
            "certificate": payload,
            "signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "value": _b64encode(signature),
            },
        }

    def issue(
        self,
        run: CertifiedRun,
        *,
        issued_at: str | None = None,
        skipped: tuple[SkipReason, ...] = (),
    ) -> dict[str, Any]:
        """Build and sign a certificate for a certified run."""
        certificate = certificate_from_run(
            run,
            issuer=self.issuer,
            public_key=self.public_key,
            issued_at=issued_at,
            skipped=skipped,
        )
        return self.sign(certificate)


def load_issuer(root: str | Path, *, issuer: str) -> CertificateIssuer:
    """Load (or create) the project key and bind it to a certifier identity."""
    return CertificateIssuer(load_or_create_key(root), issuer)


def verify_certificate(document: Any, *, public_key: str | None = None) -> Certificate:
    """Verify a certificate envelope offline and return its payload.

    Raises :class:`CertificateError` on any inconsistency: a malformed envelope, an
    unexpected schema or algorithm, a bad signature, or, when ``public_key`` is
    given, an embedded key that does not match it (a key-substitution guard, so a
    re-signed forgery fails against a separately distributed key). With no
    ``public_key`` the check proves only self-consistency.
    """
    if not isinstance(document, dict):
        raise CertificateError("certificate must be a JSON object")
    if document.get("schema") != CERTIFICATE_SCHEMA:
        raise CertificateError(
            f"unexpected certificate schema: {document.get('schema')!r}"
        )
    payload = document.get("certificate")
    signature_block = document.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature_block, dict):
        raise CertificateError("certificate envelope is missing payload or signature")
    if signature_block.get("algorithm") != SIGNATURE_ALGORITHM:
        raise CertificateError(
            f"unsupported signature algorithm: {signature_block.get('algorithm')!r}"
        )
    embedded_key = payload.get("public_key")
    if not isinstance(embedded_key, str):
        raise CertificateError("certificate payload is missing its public key")
    if public_key is not None and public_key != embedded_key:
        raise CertificateError("certificate public key does not match the expected key")
    signature = _b64decode(signature_block.get("value"))
    _verify_signature(embedded_key, signature, canonical_json(payload).encode("utf-8"))
    return Certificate.from_payload(payload)


def _verify_signature(public_key: str, signature: bytes, data: bytes) -> None:
    """Ed25519-verify ``signature`` over ``data``, or raise CertificateError."""
    ed25519, invalid_signature = _crypto()
    try:
        loaded = ed25519.Ed25519PublicKey.from_public_bytes(_b64decode(public_key))
    except (ValueError, TypeError) as exc:
        raise CertificateError("certificate public key is not a valid key") from exc
    try:
        loaded.verify(signature, data)
    except invalid_signature as exc:
        raise CertificateError("certificate signature does not verify") from exc


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(value: Any) -> bytes:
    if not isinstance(value, str):
        raise CertificateError("expected a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CertificateError("invalid base64 in certificate") from exc
