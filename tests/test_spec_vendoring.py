"""The published spec and the copy the SDK validates against, held together.

`spec/` is the standard; `elbi_core/spec/` is the same schemas vendored into the
package, because an installed wheel cannot reach up to a sibling directory at runtime.
Two copies means they can diverge, and they have: `metric.schema.json` once gained a
`format` object in the vendored copy that the published one never got, so the spec
documented less than the implementation enforced. Nothing noticed, because the
conformance suite in `spec/tests` only exercises `derivation.schema.json`.

The check lives here rather than in `spec/tests` so that suite stays self-contained: it
is the part a second-language SDK runs, and it must not reach into `packages/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "spec"
VENDORED = ROOT / "packages/elbi-core/src/elbi_core/spec"

#: Schemas the package vendors without publishing. `osi.schema.json` and
#: `component.schema.json` are somebody else's standards (Open Semantic Interchange and
#: OpenReasoningComponents, respectively), consumed for import/validation, so neither
#: has a counterpart under `spec/`.
NOT_PUBLISHED = {"osi.schema.json", "component.schema.json"}

PUBLISHED_SCHEMAS = sorted(path.name for path in PUBLISHED.glob("*.schema.json"))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_there_are_schemas_to_check() -> None:
    """Guard the guard: a glob that silently matches nothing would pass everything."""
    assert PUBLISHED_SCHEMAS


@pytest.mark.parametrize("name", PUBLISHED_SCHEMAS)
def test_published_schema_is_vendored(name: str) -> None:
    assert (VENDORED / name).is_file(), (
        f"spec/{name} is published but not vendored, so the SDK validates against "
        f"nothing for it"
    )


@pytest.mark.parametrize("name", PUBLISHED_SCHEMAS)
def test_vendored_schema_matches_the_published_one(name: str) -> None:
    """Compared as parsed JSON: the contract is the structure, not the whitespace."""
    assert _load(VENDORED / name) == _load(PUBLISHED / name), (
        f"{name} has drifted. The implementation and the published standard disagree; "
        f"reconcile them and record the change in the spec's changelog."
    )


def test_every_vendored_schema_is_published_or_declared_foreign() -> None:
    """A new schema in the package is published, or named above as somebody else's."""
    vendored = {path.name for path in VENDORED.glob("*.schema.json")}
    unaccounted = vendored - set(PUBLISHED_SCHEMAS) - NOT_PUBLISHED
    assert not unaccounted, (
        f"vendored but neither published nor declared foreign: {sorted(unaccounted)}"
    )
