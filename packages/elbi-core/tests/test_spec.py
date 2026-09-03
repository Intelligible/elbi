"""Tests for spec loading, validation, and schema sync."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from elbi_core import (
    SPEC_VERSION,
    SpecValidationError,
    is_valid_manifest,
    load_schema,
    validate_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SCHEMA = REPO_ROOT / "spec" / "derivation.schema.json"


def test_spec_version_is_major_minor() -> None:
    parts = SPEC_VERSION.split(".")
    assert len(parts) == 2
    assert all(part.isdigit() for part in parts)


def test_bundled_schema_matches_canonical_spec() -> None:
    """The schema shipped in the package must not drift from spec/."""
    bundled = load_schema()
    canonical = json.loads(CANONICAL_SCHEMA.read_text(encoding="utf-8"))
    assert bundled == canonical


def test_validate_manifest_accepts_valid() -> None:
    manifest = {
        "specVersion": SPEC_VERSION,
        "kind": "Derivation",
        "name": "churn_risk",
        "serve": {"format": "text"},
    }
    validate_manifest(manifest)  # does not raise
    assert is_valid_manifest(manifest) is True


def test_validate_manifest_rejects_and_reports() -> None:
    manifest = {"specVersion": "1.0", "kind": "Derivation", "name": "Bad-Name"}
    with pytest.raises(SpecValidationError) as excinfo:
        validate_manifest(manifest)
    assert excinfo.value.messages
    assert is_valid_manifest(manifest) is False
