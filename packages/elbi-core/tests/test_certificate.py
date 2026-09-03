"""Tests for exportable, tamper-evident verification certificates.

The security-critical behavior is that any change to a signed certificate fails
verification. The tamper matrix and the property-based byte-flip test are the spec
for that; the round-trip and field-mapping tests pin the shape.
"""

from __future__ import annotations

import copy
import json
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elbi_core.certificate import (
    CERTIFICATE_SCHEMA,
    Certificate,
    CertificateIssuer,
    SigningKey,
    load_issuer,
    load_or_create_key,
    verify_certificate,
)
from elbi_core.errors import CertificateError
from elbi_core.tracking.run import CertifiedRun
from elbi_core.versioning import canonical_json

_SEED = bytes(range(32))
_OTHER_SEED = bytes(range(32, 64))


def _run() -> CertifiedRun:
    return CertifiedRun(
        name="revenue_by_price",
        derivation_version="abc123def456ghi",
        verdict="sound",
        created_at="2026-07-20T00:00:00+00:00",
        data_hash="d" * 64,
        estimate=1.5,
        estimate_label="elasticity",
        adjusted_for=("region", "season"),
        claim={"x": "price", "y": "quantity"},
        checks=(("effect", "sound", "holds under controls"),),
        skipped=("did",),
        code_version="cv1",
        input_versions={"sales": "v1"},
        question="How does price affect quantity?",
    )


def _unicode_run() -> CertifiedRun:
    # Non-ASCII text in every text-bearing field: canonical_json's ensure_ascii
    # default escapes these to \uXXXX in the signed bytes, a path the plain-ASCII
    # _run() fixture never exercises.
    return CertifiedRun(
        name="prix_élasticité_日本語",
        derivation_version="abc123def456ghi",
        verdict="sound",
        created_at="2026-07-20T00:00:00+00:00",
        data_hash="d" * 64,
        estimate=1.5,
        estimate_label="élasticité",
        adjusted_for=("région", "saison"),
        claim={"x": "prix (€)", "y": "quantité"},
        checks=(("effet", "sound", "tient sous contrôle ✓"),),
        skipped=("did",),
        code_version="cv1",
        input_versions={"ventes": "v1"},
        question=(
            "Comment le prix affecte-t-il la quantité? 価格は数量にどう影響しますか"
        ),
    )


def _issuer(seed: bytes = _SEED) -> CertificateIssuer:
    return CertificateIssuer(SigningKey(seed), "sam@intelligible.ai")


def test_issue_verify_round_trip() -> None:
    document = _issuer().issue(_run())
    assert document["schema"] == CERTIFICATE_SCHEMA
    cert = verify_certificate(document)
    assert isinstance(cert, Certificate)
    assert cert.verdict == "sound"
    assert cert.derivation == "revenue_by_price"
    assert cert.issuer == "sam@intelligible.ai"


def test_issue_verify_round_trip_with_unicode_fields() -> None:
    run = _unicode_run()
    cert = verify_certificate(_issuer().issue(run))
    assert cert.derivation == run.name
    assert cert.question == run.question
    assert cert.claim == run.claim
    assert cert.estimate_label == run.estimate_label
    assert cert.checks == run.checks


def test_issue_is_deterministic() -> None:
    # Same run, key, and (run-derived) timestamp: byte-identical, Ed25519 being
    # deterministic. This is what lets a re-export round-trip through pull/sync.
    assert _issuer().issue(_run()) == _issuer().issue(_run())


def test_field_mapping_is_verbatim() -> None:
    cert = verify_certificate(_issuer().issue(_run()))
    run = _run()
    assert cert.checks == run.checks
    assert cert.adjusted_for == run.adjusted_for
    assert cert.input_versions == run.input_versions
    assert cert.data_hash == run.data_hash
    assert cert.derivation_version == run.derivation_version
    assert cert.code_version == run.code_version
    # A run records skipped names only, so each rides with an empty reason.
    assert cert.skipped == (("did", ""),)
    # issued_at defaults to the run's own timestamp.
    assert cert.issued_at == run.created_at


def test_skipped_reasons_override() -> None:
    document = _issuer().issue(_run(), skipped=(("did", "not a panel"),))
    cert = verify_certificate(document)
    assert cert.skipped == (("did", "not a panel"),)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verdict", "unsound"),
        ("estimate", 99.0),
        ("data_hash", "e" * 64),
        ("issued_at", "2020-01-01T00:00:00+00:00"),
        ("issuer", "attacker@evil.example"),
        ("derivation", "other"),
    ],
)
def test_tampering_a_payload_field_fails(field: str, value: Any) -> None:
    document = _issuer().issue(_run())
    document["certificate"][field] = value
    with pytest.raises(CertificateError):
        verify_certificate(document)


def test_tampering_a_check_detail_fails() -> None:
    document = _issuer().issue(_run())
    document["certificate"]["checks"][0][2] = "does not hold"
    with pytest.raises(CertificateError):
        verify_certificate(document)


def test_substituting_the_public_key_fails() -> None:
    document = _issuer().issue(_run())
    document["certificate"]["public_key"] = _issuer(_OTHER_SEED).public_key
    with pytest.raises(CertificateError):
        verify_certificate(document)


def test_adding_or_removing_a_field_fails() -> None:
    added = _issuer().issue(_run())
    added["certificate"]["surprise"] = 1
    with pytest.raises(CertificateError):
        verify_certificate(added)
    removed = _issuer().issue(_run())
    del removed["certificate"]["data_hash"]
    with pytest.raises(CertificateError):
        verify_certificate(removed)


def test_signature_block_tampering_fails() -> None:
    stripped = _issuer().issue(_run())
    del stripped["signature"]
    with pytest.raises(CertificateError):
        verify_certificate(stripped)

    wrong_algo = _issuer().issue(_run())
    wrong_algo["signature"]["algorithm"] = "rsa"
    with pytest.raises(CertificateError):
        verify_certificate(wrong_algo)

    corrupt = _issuer().issue(_run())
    corrupt["signature"]["value"] = "!!!not base64!!!"
    with pytest.raises(CertificateError):
        verify_certificate(corrupt)


def test_malformed_embedded_public_key_fails() -> None:
    # A public key that is valid base64 but not a valid Ed25519 key is rejected
    # cleanly (as a CertificateError, not a raw crypto error).
    document = _issuer().issue(_run())
    document["certificate"]["public_key"] = "AAAA"  # decodes to 3 bytes, not a key
    with pytest.raises(CertificateError):
        verify_certificate(document)


def test_missing_embedded_public_key_fails() -> None:
    # A payload whose public key is stripped is rejected before any crypto runs, as a
    # clean CertificateError rather than a raw type error.
    document = _issuer().issue(_run())
    del document["certificate"]["public_key"]
    with pytest.raises(CertificateError, match="missing its public key"):
        verify_certificate(document)


def test_non_string_signature_value_fails() -> None:
    # A signature value that is not a base64 string fails as a CertificateError, not a
    # TypeError from feeding a non-string into base64 decoding.
    document = _issuer().issue(_run())
    document["signature"]["value"] = 1234
    with pytest.raises(CertificateError, match="expected a base64 string"):
        verify_certificate(document)


def test_wrong_schema_fails() -> None:
    document = _issuer().issue(_run())
    document["schema"] = "something/else"
    with pytest.raises(CertificateError):
        verify_certificate(document)


def test_non_object_document_fails() -> None:
    with pytest.raises(CertificateError):
        verify_certificate(["not", "a", "dict"])


def test_explicit_public_key_right_and_wrong() -> None:
    issuer = _issuer()
    document = issuer.issue(_run())
    # Correct expected key: passes.
    verify_certificate(document, public_key=issuer.public_key)
    # A different expected key: the key-substitution guard fires.
    with pytest.raises(CertificateError):
        verify_certificate(document, public_key=_issuer(_OTHER_SEED).public_key)


def test_a_forgery_re_signed_with_another_key_fails_against_the_real_key() -> None:
    # Attacker edits the payload, re-signs with their own key, and swaps the embedded
    # public key so the document is self-consistent. An independent verifier holding the
    # real public key rejects it.
    real = _issuer()
    original = real.issue(_run())
    attacker = _issuer(_OTHER_SEED)
    # Edit the payload, embed the attacker's key, and re-sign with it.
    tampered = replace(
        Certificate.from_payload(original["certificate"]),
        verdict="unsound",
        public_key=attacker.public_key,
    )
    forged = attacker.sign(tampered)
    # Self-consistent on its own:
    assert verify_certificate(forged).verdict == "unsound"
    # But not against the real distributed key:
    with pytest.raises(CertificateError):
        verify_certificate(forged, public_key=real.public_key)


@pytest.mark.parametrize("run_factory", [_run, _unicode_run])
@settings(deadline=None, max_examples=75)
@given(offset=st.integers(min_value=0))
def test_any_byte_flip_fails_verification(run_factory: Any, offset: int) -> None:
    document = _issuer().issue(run_factory())
    serialized = json.dumps(document)
    if offset >= len(serialized):
        return
    flipped = list(serialized)
    original = flipped[offset]
    # Flip to a different character in a small alphabet, keeping it text.
    flipped[offset] = "0" if original != "0" else "1"
    mutated_text = "".join(flipped)
    if mutated_text == serialized:
        return
    try:
        mutated = json.loads(mutated_text)
    except json.JSONDecodeError:
        return  # a mutation that no longer parses is caught upstream
    if mutated == document:
        return  # semantically identical (e.g. whitespace); nothing to detect
    with pytest.raises(CertificateError):
        verify_certificate(mutated)


def test_canonical_bytes_ignore_key_order() -> None:
    document = _issuer().issue(_run())
    payload = document["certificate"]
    reordered = dict(reversed(list(payload.items())))
    assert canonical_json(reordered) == canonical_json(payload)
    document["certificate"] = reordered
    # Re-canonicalization means key order in the file never affects verification.
    verify_certificate(document)


def test_load_or_create_key_persists_with_safe_mode(tmp_path: Path) -> None:
    key = load_or_create_key(tmp_path)
    key_path = tmp_path / "certificate.key"
    pub_path = tmp_path / "certificate.key.pub"
    assert key_path.exists() and pub_path.exists()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert pub_path.read_text(encoding="utf-8").strip() == key.public_key
    # A second load returns the same key.
    assert load_or_create_key(tmp_path).public_key == key.public_key


def test_invalid_base64_signature_is_rejected() -> None:
    document = _issuer().issue(_run())
    # Invalid characters inserted mid-string, not appended: without validate=True,
    # b64decode silently strips them and decodes the rest, so the corruption would
    # surface only as a signature-verification failure, not a base64 error.
    valid = document["signature"]["value"]
    document["signature"]["value"] = valid[:10] + "!!!!" + valid[10:]
    with pytest.raises(CertificateError, match="base64"):
        verify_certificate(document)


def test_load_or_create_key_repairs_a_stale_public_key_file(tmp_path: Path) -> None:
    key = load_or_create_key(tmp_path)
    pub_path = tmp_path / "certificate.key.pub"
    pub_path.write_text("not the real public key\n", encoding="utf-8")

    load_or_create_key(tmp_path)  # same private key on disk; rewrites the stale .pub
    assert pub_path.read_text(encoding="utf-8").strip() == key.public_key


def test_certificate_and_issuer_are_immutable() -> None:
    cert = Certificate.from_payload(_issuer().issue(_run())["certificate"])
    with pytest.raises(FrozenInstanceError):
        cert.verdict = "unsound"  # type: ignore[misc]

    issuer = _issuer()
    with pytest.raises(FrozenInstanceError):
        issuer.issuer = "someone-else"  # type: ignore[misc]


def test_load_or_create_key_creates_missing_nested_directories(tmp_path: Path) -> None:
    root = tmp_path / "a" / "b" / "c"
    assert not root.exists()
    key = load_or_create_key(root)
    assert (root / "certificate.key").exists()
    assert (root / "certificate.key.pub").read_text(encoding="utf-8").strip() == (
        key.public_key
    )


def test_key_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "elsewhere" / "signing.key"
    monkeypatch.setenv("ELBI_CERTIFICATE_KEY", str(override))
    key = load_or_create_key(tmp_path)
    assert override.exists()
    assert not (tmp_path / "certificate.key").exists()
    assert isinstance(key, SigningKey)


def test_bad_seed_length_raises() -> None:
    # Both too short and too long: a seed must be exactly 32 bytes, not merely enough.
    with pytest.raises(CertificateError):
        SigningKey(b"too short")
    with pytest.raises(CertificateError):
        SigningKey(b"x" * 33)


def test_load_issuer_binds_identity(tmp_path: Path) -> None:
    issuer = load_issuer(tmp_path, issuer="ci@intelligible.ai")
    cert = verify_certificate(issuer.issue(_run()))
    assert cert.issuer == "ci@intelligible.ai"


def test_from_payload_rejects_malformed() -> None:
    with pytest.raises(CertificateError):
        Certificate.from_payload({"derivation": "x"})  # missing required keys


def test_deepcopy_document_still_verifies() -> None:
    # Guards against accidental shared-mutable-state bugs in the envelope.
    document = _issuer().issue(_run())
    verify_certificate(copy.deepcopy(document))
