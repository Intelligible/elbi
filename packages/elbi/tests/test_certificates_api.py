"""Tests for the certificate API: JSON + PDF export and the sync verify gate."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from elbi import create_app
from elbi.app import _safe_filename_part
from elbi.certificate_pdf import (
    _LABEL_WIDTH,
    _MARGIN,
    _PAGE_WIDTH,
    _field,
    _mono_line,
)
from elbi.db import Derivation, open_store
from elbi_cli import config_sync
from elbi_core import load_issuer, verify_certificate
from elbi_core.tracking import CertifiedRun

_ATTESTATION = {
    "schema": "elbi.verification/v1",
    "verdict": "sound",
    "claim": {"x": "x", "y": "y"},
    "estimate": 2.0,
    "estimate_label": "+2.00 in y per +1 unit of x",
    "adjusted_for": ["region"],
    "checks": [{"name": "effect", "verdict": "sound", "detail": "holds"}],
    "skipped": ["did"],
    "data_hash": "d" * 64,
}


def _app_with_certified(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    store.save_derivation(
        Derivation(
            name="eff",
            question="does x move y?",
            source="def eff(ctx): ...",
            verdict="sound",
            data_hash="d" * 64,
            claim_json=json.dumps({"x": "x", "y": "y"}),
            attestation_json=json.dumps(_ATTESTATION),
        )
    )
    issuer = load_issuer(tmp_path, issuer="acme-corp")
    app = create_app(load_datasets=dict, store=store, certificate_issuer=issuer)
    return app, issuer


def test_get_certificate_verifies_and_downloads(tmp_path: Path) -> None:
    app, _ = _app_with_certified(tmp_path)
    with TestClient(app) as http:
        resp = http.get("/api/certificates/eff")
    assert resp.status_code == 200
    assert 'filename="eff-certificate.json"' in resp.headers["content-disposition"]
    cert = verify_certificate(resp.json())
    assert cert.derivation == "eff"
    assert cert.verdict == "sound"
    assert cert.issuer == "acme-corp"
    assert cert.skipped == (("did", ""),)


def test_get_certificate_is_deterministic(tmp_path: Path) -> None:
    app, _ = _app_with_certified(tmp_path)
    with TestClient(app) as http:
        first = http.get("/api/certificates/eff").json()
        second = http.get("/api/certificates/eff").json()
    assert first == second


def test_get_certificate_pdf(tmp_path: Path) -> None:
    app, _ = _app_with_certified(tmp_path)
    with TestClient(app) as http:
        resp = http.get("/api/certificates/eff/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"
    assert 'filename="eff-certificate.pdf"' in resp.headers["content-disposition"]


def _pdf_text(data: bytes) -> bytes:
    """The concatenated bytes of every Flate-compressed content stream in a PDF.

    fpdf2's page content is plain ``(text) Tj`` operators inside one zlib stream;
    stdlib zlib is enough to recover it without a PDF-parsing dependency. A stream
    that fails to inflate (e.g. font metadata, not Flate-encoded) is skipped.
    """
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)
    out = b""
    failures = 0
    for raw in streams:
        try:
            # decompressobj rather than decompress: it stops at the end of the zlib
            # stream and puts anything after it in unused_data, so a delimiter the
            # regex happened to include cannot turn a good stream into an error.
            out += zlib.decompressobj().decompress(raw)
        except zlib.error:
            failures += 1
    # Extracting nothing has meant, on rare occasions in CI, an assertion reading
    # "assert b'(eff)' in b''" -- true, unhelpful, and giving nothing to work from. Say
    # what arrived instead, so an intermittent failure is diagnosable the first time it
    # is seen rather than after it has been seen three times.
    assert out or not data, (
        f"no PDF content stream could be read: {len(data)} bytes, "
        f"starts {data[:8]!r}, {len(streams)} stream(s) found, "
        f"{failures} failed to inflate"
    )
    return out


def test_get_certificate_pdf_renders_the_certificate_content(tmp_path: Path) -> None:
    app, _ = _app_with_certified(tmp_path)
    with TestClient(app) as http:
        pdf_bytes = http.get("/api/certificates/eff/pdf").content
    text = _pdf_text(pdf_bytes)
    # The exact "(eff) Tj" token, not a bare substring match: "eff" is also a
    # substring of "effect" (the fixture's check name), which would let a broken
    # renderer that dropped the derivation name pass this assertion by accident.
    assert b"(eff)" in text
    assert b"Verdict: sound" in text

    # A differently-named, differently-certified derivation renders different bytes:
    # the PDF reflects its input rather than a fixed template.
    store = open_store(f"sqlite:{tmp_path / 'other.db'}")
    store.save_derivation(
        Derivation(
            name="other",
            question="q2",
            source="def other(ctx): ...",
            verdict="sound",
            data_hash="e" * 64,
            claim_json=json.dumps({"x": "a", "y": "b"}),
            attestation_json=json.dumps({**_ATTESTATION, "verdict": "sound"}),
        )
    )
    issuer = load_issuer(tmp_path / "other", issuer="acme-corp")
    other_app = create_app(load_datasets=dict, store=store, certificate_issuer=issuer)
    with TestClient(other_app) as http:
        other_pdf = http.get("/api/certificates/other/pdf").content
    assert other_pdf != pdf_bytes


def test_every_row_returns_the_cursor_to_the_left_margin() -> None:
    # `multi_cell` leaves the cursor at the cell's right edge by default, so a row
    # that ends with wrapped text carries that offset into the next row and marches
    # the page contents off the right edge. Every row helper must end at the margin.
    pdf = FPDF(format="A4")
    pdf.set_margins(_MARGIN, _MARGIN)
    pdf.add_page()
    rows = (
        lambda: _field(pdf, "Question", "does x move y?"),
        lambda: _field(pdf, "Estimate", "2.0 (+2.00 in y per +1 unit of x)"),
        lambda: _field(pdf, "+2.00 in y per +1 unit of x", "2.0"),  # oversized label
        lambda: _mono_line(pdf, f"data hash:   {'d' * 64}"),
    )
    for draw in rows:
        draw()
        assert pdf.get_x() == pdf.l_margin
        assert pdf.get_x() + _PAGE_WIDTH <= pdf.w - pdf.r_margin + 0.01


def test_field_label_too_wide_for_its_column_takes_its_own_line() -> None:
    # `cell` neither wraps nor clips, so a label wider than the label column is
    # drawn straight through the value beside it and the two collide. Such a
    # label must take a full-width line of its own, with the value beneath.
    pdf = FPDF(format="A4")
    pdf.set_margins(_MARGIN, _MARGIN)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 10)
    oversized = "+1.13 in response per +1 unit of dose"
    assert pdf.get_string_width(f"{oversized}:") > _LABEL_WIDTH

    start = pdf.get_y()
    _field(pdf, oversized, "1.134901518698129")
    label_and_value = pdf.get_y() - start
    # Two stacked 6mm rows, not one shared row with the value overdrawing the label.
    assert label_and_value >= 12

    pdf.set_font("Helvetica", "B", 10)
    start = pdf.get_y()
    _field(pdf, "Estimate", "1.134901518698129")
    assert pdf.get_y() - start < 12  # a short label still shares one row


def test_estimate_prose_is_a_value_not_a_label(tmp_path: Path) -> None:
    # A real oracle estimate_label is a sentence, not a field name. Rendering it
    # in the fixed-width label column overdrew the estimate beside it, so the
    # prose belongs in the wrapping value and the column keeps a short label.
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    store.save_derivation(
        Derivation(
            name="eff",
            question="does x move y?",
            source="def eff(ctx): ...",
            verdict="sound",
            data_hash="d" * 64,
            claim_json=json.dumps({"x": "x", "y": "y"}),
            attestation_json=json.dumps(_ATTESTATION),
        )
    )
    issuer = load_issuer(tmp_path, issuer="acme-corp")
    app = create_app(load_datasets=dict, store=store, certificate_issuer=issuer)
    with TestClient(app) as http:
        text = _pdf_text(http.get("/api/certificates/eff/pdf").content)
    assert b"(Estimate:)" in text


def test_certificate_404s(tmp_path: Path) -> None:
    # Unknown name, a derivation with no attestation, and another owner's row all 404.
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    store.save_derivation(
        Derivation(name="bare", question="q", source="def bare(ctx): ...")
    )
    issuer = load_issuer(tmp_path, issuer="acme")
    app = create_app(load_datasets=dict, store=store, certificate_issuer=issuer)
    with TestClient(app) as http:
        assert http.get("/api/certificates/nope").status_code == 404
        assert http.get("/api/certificates/bare").status_code == 404

    app2, _ = _app_with_certified(tmp_path / "b")
    with TestClient(app2) as http:
        # Single-user mode has one fully trusted caller who holds manage on
        # everything (Store.effective_level), owned rows included: with no login
        # there is nobody else the owner could be.
        assert http.get("/api/certificates/eff").status_code == 200


def test_certificate_404s_for_a_claimless_certification(tmp_path: Path) -> None:
    # A derivation whose oracle attestation is empty (the fail-open shape: claimless
    # AutoCertifyOnVerify, or a legacy row persisted before this gate existed) must
    # never yield a certificate, even though attestation_json="{}" is a truthy string.
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    store.save_derivation(
        Derivation(
            name="eff",
            question="does x move y?",
            source="def eff(ctx): ...",
            verdict="sound",
            data_hash="d" * 64,
            claim_json=json.dumps({"x": "x", "y": "y"}),
            attestation_json=json.dumps(_ATTESTATION),
        )
    )
    store.save_derivation(
        Derivation(
            name="unverified",
            question="q",
            source="def unverified(ctx): ...",
            attestation_json=json.dumps({}),
        )
    )
    issuer = load_issuer(tmp_path, issuer="acme")
    app = create_app(load_datasets=dict, store=store, certificate_issuer=issuer)
    with TestClient(app) as http:
        assert http.get("/api/certificates/unverified").status_code == 404
        assert http.get("/api/certificates/unverified/pdf").status_code == 404
        # A validly signed certificate for a DIFFERENT, verified derivation still
        # 404s when posted against this name: verify_certificate accepts it (it is
        # genuinely well-formed and signed), but the gate inside
        # _certificate_document("unverified", ...) refuses it before the stale-check.
        valid_other_cert = http.get("/api/certificates/eff").json()
        resp = http.post("/api/certificates/unverified", json=valid_other_cert)
        assert resp.status_code == 404


def test_post_certificate_verify_gate(tmp_path: Path) -> None:
    app, issuer = _app_with_certified(tmp_path)
    with TestClient(app) as http:
        current = http.get("/api/certificates/eff").json()

        # The exact current certificate verifies.
        ok = http.post("/api/certificates/eff", json=current)
        assert ok.status_code == 200 and ok.json()["verified"] is True

        # A tampered payload fails the signature check (400).
        tampered = json.loads(json.dumps(current))
        tampered["certificate"]["verdict"] = "unsound"
        resp = http.post("/api/certificates/eff", json=tampered)
        assert resp.status_code == 400

        # A validly signed but stale certificate (different content) conflicts (409).
        stale_run = CertifiedRun(
            name="eff",
            derivation_version="",
            verdict="sound",
            created_at="2000-01-01T00:00:00+00:00",  # a different issued_at
            data_hash="d" * 64,
            claim={"x": "x", "y": "y"},
        )
        stale = issuer.issue(stale_run)
        assert http.post("/api/certificates/eff", json=stale).status_code == 409


def test_post_certificate_verify_gate_rejects_malformed_json(tmp_path: Path) -> None:
    app, _ = _app_with_certified(tmp_path)
    with TestClient(app) as http:
        resp = http.post(
            "/api/certificates/eff",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 400


def test_certificates_remote_ids_avoids_the_per_derivation_fetch(
    tmp_path: Path,
) -> None:
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    store.save_derivation(
        Derivation(
            name="eff",
            question="does x move y?",
            source="def eff(ctx): ...",
            verdict="sound",
            data_hash="d" * 64,
            claim_json=json.dumps({"x": "x", "y": "y"}),
            attestation_json=json.dumps(_ATTESTATION),
        )
    )
    store.save_derivation(
        Derivation(name="bare", question="q", source="def bare(ctx): ...")
    )
    issuer = load_issuer(tmp_path, issuer="acme")
    app = create_app(load_datasets=dict, store=store, certificate_issuer=issuer)
    surface = config_sync._certificates_surface()

    with TestClient(app) as http:
        cert_gets = 0
        original_get = http.get

        def counting_get(url, *args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal cert_gets
            if str(url).startswith("/api/certificates/"):
                cert_gets += 1
            return original_get(url, *args, **kwargs)

        http.get = counting_get  # type: ignore[method-assign]

        ids = surface.remote_ids(http)
        assert set(ids) == {"eff", "bare"}  # every derivation, certified or not
        assert cert_gets == 0  # no per-derivation certificate fetch

        # remote() still does the expensive per-derivation fetch, for contrast: one
        # GET per derivation, including "bare", which 404s and is skipped from the
        # result but still costs a request.
        remote = surface.remote(http)
        assert set(remote) == {"eff"}
        assert cert_gets == 2


def test_post_certificate_verify_gate_rejects_a_different_signing_key(
    tmp_path: Path,
) -> None:
    app, _ = _app_with_certified(tmp_path)
    # A second app over the SAME store, but with an independently keyed issuer: its
    # certificate for "eff" is genuinely well-formed and self-consistently signed,
    # just not under a key this deployment ever trusted.
    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    forger_issuer = load_issuer(tmp_path / "forger", issuer="acme-corp")
    forger_app = create_app(
        load_datasets=dict, store=store, certificate_issuer=forger_issuer
    )
    with TestClient(forger_app) as forger_http:
        forged = forger_http.get("/api/certificates/eff").json()

    with TestClient(app) as http:
        resp = http.post("/api/certificates/eff", json=forged)
    assert resp.status_code == 400


def test_post_certificate_verify_gate_400s_before_404ing(tmp_path: Path) -> None:
    # A garbage certificate posted against a name that does not even exist: the
    # signature check must fire first (400), not the derivation-existence check
    # inside _certificate_document (404) - the two failure reasons are different
    # and a caller should not be told "not found" when their input was malformed.
    app, _ = _app_with_certified(tmp_path)
    with TestClient(app) as http:
        resp = http.post(
            "/api/certificates/does-not-exist", json={"not": "a certificate"}
        )
    assert resp.status_code == 400


def test_certificate_filename_sanitizes_an_unsafe_derivation_name(
    tmp_path: Path,
) -> None:
    assert _safe_filename_part('eff"; evil') == "eff___evil"

    store = open_store(f"sqlite:{tmp_path / 'app.db'}")
    unsafe_name = 'eff"; evil'
    store.save_derivation(
        Derivation(
            name=unsafe_name,
            question="q",
            source="def eff(ctx): ...",
            verdict="sound",
            data_hash="d" * 64,
            claim_json=json.dumps({"x": "x", "y": "y"}),
            attestation_json=json.dumps(_ATTESTATION),
        )
    )
    issuer = load_issuer(tmp_path, issuer="acme")
    app = create_app(load_datasets=dict, store=store, certificate_issuer=issuer)
    with TestClient(app) as http:
        resp = http.get(f"/api/certificates/{unsafe_name}")
    assert resp.status_code == 200
    disposition = resp.headers["content-disposition"]
    # Exactly the two quotes that legitimately wrap the filename; the unsafe
    # character in the derivation name must not have produced a third.
    assert disposition.count('"') == 2


def test_certificate_round_trips_pull_sync(tmp_path: Path) -> None:
    app, _ = _app_with_certified(tmp_path)
    surfaces = [config_sync._certificates_surface()]
    repo = tmp_path / "repo"
    with TestClient(app) as http:
        written = config_sync.pull(http, repo, surfaces_=surfaces)
        assert ("certificates", "eff") in {(c.surface, c.name) for c in written}
        cert_file = repo / "certificates" / "eff.json"
        assert cert_file.exists()
        # The pulled file verifies offline: the casing boundary left it byte-for-byte.
        verify_certificate(json.loads(cert_file.read_text(encoding="utf-8")))

        # Pull then sync is a no-op; nothing is applied.
        residual = [
            c
            for c in config_sync.plan(http, repo, surfaces_=surfaces)
            if c.action != "unchanged"
        ]
        assert residual == []
        assert config_sync.sync(http, repo, surfaces_=surfaces) == []

        # Tampering the file fails the sync at the app's verify gate.
        doc = json.loads(cert_file.read_text(encoding="utf-8"))
        doc["certificate"]["verdict"] = "unsound"
        cert_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        with pytest.raises(config_sync.SyncError):
            config_sync.sync(http, repo, surfaces_=surfaces)
