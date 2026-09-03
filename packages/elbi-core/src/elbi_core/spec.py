"""Loading and validating against the Open Derivation Spec.

The spec ships *as data* inside this package (``spec/derivation.schema.json``) and
is loaded at run time. The SDK never hand-mirrors the schema's constraints in
Python; validation always goes through the JSON Schema, so the spec cannot
silently drift from the implementation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

import jsonschema

from .errors import SpecValidationError

#: The Open Derivation Spec version this SDK implements (MAJOR.MINOR).
SPEC_VERSION = "1.1"


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the bundled derivation JSON Schema as a dict."""
    resource = files("elbi_core") / "spec" / "derivation.schema.json"
    data: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return data


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate a derivation manifest against the spec.

    Raises:
        SpecValidationError: if the manifest does not conform. The error carries
            a list of human-readable messages, one per violation.
    """
    errors = sorted(_validator().iter_errors(manifest), key=lambda e: list(e.path))
    if errors:
        raise SpecValidationError([_format_error(error) for error in errors])


def is_valid_manifest(manifest: dict[str, Any]) -> bool:
    """Return whether a manifest conforms to the spec, without raising."""
    return bool(_validator().is_valid(manifest))


def _format_error(error: jsonschema.ValidationError) -> str:
    location = "/".join(str(part) for part in error.path)
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"
