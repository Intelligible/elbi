"""Render a verification certificate envelope as a one-page PDF.

A human-readable companion to the signed JSON: an auditor reads the PDF, then
checks the JSON with ``elbi certificate verify``. Uses fpdf2 core fonts
only, so no font files ship in the wheel. Core fonts are latin-1, so text is
sanitized to that range before it is drawn.
"""

from __future__ import annotations

import hashlib
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

_MARGIN = 15  # Page margin on every side, in mm.
_PAGE_WIDTH = 210 - 2 * _MARGIN  # A4 is 210mm wide; this is the content width.
_LABEL_WIDTH = 35  # The bold label column in a field row, in mm.


def render_certificate_pdf(envelope: dict[str, Any]) -> bytes:
    """Render a certificate envelope to a one-page A4 PDF as bytes."""
    payload = envelope.get("certificate", {})
    signature = envelope.get("signature", {})
    pdf = FPDF(format="A4")
    pdf.set_margins(_MARGIN, _MARGIN)
    pdf.set_auto_page_break(auto=True, margin=_MARGIN)
    pdf.add_page()

    for family, size, height, text in (
        ("Helvetica", 18, 10, "Certificate of Verification"),
        ("Courier", 13, 8, str(payload.get("derivation", ""))),
        ("Helvetica", 11, 7, f"Verdict: {payload.get('verdict', '')}"),
    ):
        pdf.set_font(family, "B", size)
        pdf.cell(0, height, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    _field(pdf, "Question", str(payload.get("question", "") or "(none)"))
    estimate = payload.get("estimate")
    if estimate is not None:
        described = payload.get("estimate_label")
        value = f"{estimate} ({described})" if described else str(estimate)
        _field(pdf, "Estimate", value)
    adjusted = payload.get("adjusted_for") or []
    if adjusted:
        _field(pdf, "Adjusted for", ", ".join(str(a) for a in adjusted))

    _section(pdf, "Gates")
    for name, verdict, detail in payload.get("checks", []):
        _mono_line(pdf, f"[{_gate_mark(str(verdict))}] {name}: {detail}")
    for name, reason in payload.get("skipped", []):
        _mono_line(pdf, f"[SKIP] {name}{f': {reason}' if reason else ''}")

    _section(pdf, "Integrity")
    _mono_line(pdf, f"data hash:   {payload.get('data_hash') or '(none)'}")
    _mono_line(pdf, f"code:        {payload.get('code_version') or '(none)'}")
    _mono_line(pdf, f"version:     {payload.get('derivation_version') or '(none)'}")
    _mono_line(pdf, f"issued at:   {payload.get('issued_at') or '(none)'}")
    _mono_line(pdf, f"issued by:   {payload.get('issuer') or '(none)'}")
    _mono_line(
        pdf,
        f"signature:   {_fingerprint(signature)} ({signature.get('algorithm', '')})",
    )
    _mono_line(pdf, f"schema:      {envelope.get('schema', '')}")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    footer = "Verify offline with: elbi certificate verify <file>"
    _wrapped(pdf, _PAGE_WIDTH, 5, _latin1(footer))

    return bytes(pdf.output())


def _fingerprint(signature: dict[str, Any]) -> str:
    """A short, stable fingerprint of the signature bytes for at-a-glance matching."""
    value = str(signature.get("value", ""))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _gate_mark(verdict: str) -> str:
    return {"sound": "PASS", "unsound": "FAIL"}.get(verdict, verdict.upper()[:4])


def _field(pdf: FPDF, label: str, value: str) -> None:
    """Draw a bold label beside its wrapping value.

    ``cell`` neither wraps nor clips, so a label wider than the column would be
    overdrawn by the value beside it. A label that does not fit takes a
    full-width line of its own instead, with the value beneath it.
    """
    text = _latin1(f"{label}:")
    pdf.set_font("Helvetica", "B", 10)
    if pdf.get_string_width(text) > _LABEL_WIDTH:
        _wrapped(pdf, _PAGE_WIDTH, 6, text)
        pdf.set_font("Helvetica", "", 10)
        _wrapped(pdf, _PAGE_WIDTH, 6, _latin1(value))
        return
    pdf.cell(_LABEL_WIDTH, 6, text, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 10)
    _wrapped(pdf, _PAGE_WIDTH - _LABEL_WIDTH, 6, _latin1(value))


def _section(pdf: FPDF, title: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _latin1(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _mono_line(pdf: FPDF, text: str) -> None:
    pdf.set_font("Courier", "", 9)
    _wrapped(pdf, _PAGE_WIDTH, 5, _latin1(text))


def _wrapped(pdf: FPDF, width: float, height: float, text: str) -> None:
    """Draw wrapping text that ends its row.

    ``multi_cell`` defaults to leaving the cursor at the cell's right edge, which
    is only right when another cell follows on the same row. Every row here ends
    with its wrapped text, so the cursor returns to the left margin instead; left
    at the right edge it would carry over and push the next row off the page.
    """
    pdf.multi_cell(width, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


#: Characters the oracle emits that latin-1 cannot encode, and their ASCII forms.
#: Mapped before the lossy replace so an arrow reads as "->" rather than "?".
_ASCII_FALLBACKS = {
    "\u2192": "->",  # rightwards arrow
    "\u2190": "<-",  # leftwards arrow
    "\u2264": "<=",  # less-than or equal to
    "\u2265": ">=",  # greater-than or equal to
    "\u2212": "-",  # minus sign
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
}


def _latin1(text: str) -> str:
    """Coerce text into the latin-1 range fpdf2 core fonts can encode."""
    for char, ascii_form in _ASCII_FALLBACKS.items():
        text = text.replace(char, ascii_form)
    return text.encode("latin-1", "replace").decode("latin-1")
