"""Manifest hygiene: every trap validates, ids are unique, references + reviews resolve.

This makes the JSON schema and the review policy part of the CI gate, mirroring the
spec conformance suite: a malformed, duplicate-id, dangling-reference, or
under-reviewed trap fails here rather than producing a confusing error at run time.
"""

from __future__ import annotations

import json

import jsonschema
import pytest
from benchmark.gates import GATES
from benchmark.generators import GENERATORS
from benchmark.loader import (
    load_schema,
    load_traps,
    manifest_paths,
)


def test_schema_is_valid_json_schema() -> None:
    schema = load_schema()
    jsonschema.validators.validator_for(schema).check_schema(schema)


def test_there_are_manifests() -> None:
    assert manifest_paths(), "no trap manifests found under traps/"


def test_manifest_glob_only_matches_manifest_files() -> None:
    # A trap folder may also hold data.jsonl; only manifest.json is a trap.
    assert all(p.name == "manifest.json" for p in manifest_paths())


def test_load_traps_succeeds_and_ids_are_unique() -> None:
    traps = load_traps()  # raises on invalid schema or duplicate id
    ids = [t.id for t in traps]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("path", manifest_paths(), ids=lambda p: p.parent.name)
def test_manifest_references_resolve(path) -> None:  # type: ignore[no-untyped-def]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["gate"] in GATES, f"unknown gate {manifest['gate']!r}"
    data = manifest["data"]
    if "file" in data:
        assert (path.parent / data["file"]).exists(), (
            f"missing data file {data['file']!r}"
        )
    if "generated_by" in manifest:
        assert manifest["generated_by"]["generator"] in GENERATORS, (
            f"unknown generator {manifest['generated_by']['generator']!r}"
        )


def test_reviewer_count_policy() -> None:
    # Adjudication (AC5): an *accepted* real trap needs >=2 independent sign-offs
    # against the rubric; a candidate (under review) or a synthetic trap needs >=1.
    # CODEOWNERS gates that the team reviewed; this enforces the count from reviewed_by.
    for trap in load_traps():
        need = 2 if (trap.is_real and trap.status == "accepted") else 1
        kind = "accepted real" if need == 2 else "candidate/synthetic"
        assert len(trap.reviewed_by) >= need, (
            f"{trap.id}: {kind} trap needs >={need} reviewer(s), "
            f"has {len(trap.reviewed_by)}"
        )
