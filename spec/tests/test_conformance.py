"""Conformance suite for the Open Derivation Spec.

Validates every ``*.json`` fixture in this directory against the canonical
``derivation.schema.json`` and asserts the verdict matches the fixture's
``expected.valid``. This suite is implementation-independent: it exercises the
published schema directly, so a second-language SDK can run the same files.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SPEC_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = SPEC_DIR / "derivation.schema.json"
FIXTURE_DIR = SPEC_DIR / "tests"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _make_validator() -> jsonschema.protocols.Validator:
    """Build a validator whose dialect is chosen from the schema's ``$schema``.

    Dispatching via ``validator_for`` means a second-language implementation and
    this one agree on dialect selection from the document alone.
    """
    schema = _load_schema()
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _fixture_paths() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("*.json"))


def test_schema_is_itself_a_valid_json_schema() -> None:
    # Raises SchemaError if the schema is malformed.
    _make_validator()


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda p: p.stem)
def test_fixture_matches_expected_verdict(fixture_path: Path) -> None:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert {"description", "data", "valid"} <= fixture.keys(), (
        f"{fixture_path.name} is missing required conformance keys"
    )

    validator = _make_validator()
    errors = list(validator.iter_errors(fixture["data"]))
    is_valid = not errors
    expected = fixture["valid"]

    assert is_valid == expected, (
        f"{fixture_path.name}: expected valid={expected}, got valid={is_valid}. "
        f"Errors: {[e.message for e in errors]}"
    )


def test_examples_directory_agrees_with_schema() -> None:
    validator = _make_validator()
    for path in (SPEC_DIR / "examples" / "valid").glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert validator.is_valid(manifest), f"{path} should be valid"
    for path in (SPEC_DIR / "examples" / "invalid").glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert not validator.is_valid(manifest), f"{path} should be invalid"
