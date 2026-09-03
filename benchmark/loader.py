"""Load and validate trap manifests into runtime ``Trap`` objects.

Each trap is a folder `traps/<id>/` holding `manifest.json` (validated against
`trap.schema.json`, the same `jsonschema` machinery `spec/tests` uses) and, for
file-backed real traps, a `data.jsonl`. The loader derives the trap id from the folder
name, resolves the manifest's `data` block to a rows-builder and its `gate` + `claim`
to a verify call, and returns the `Trap` objects the harness iterates. A malformed
manifest or an unresolvable reference is an error caught by `test_schema.py`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema

from .gates import run_gate
from .generators import GENERATORS
from .trap import Rows, Trap

BENCHMARK_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BENCHMARK_DIR / "trap.schema.json"
TRAPS_DIR = BENCHMARK_DIR / "traps"

# The gate module the bite gate mutates. A single-purpose gate mutates its own module; a
# verify_all trap runs the whole catalog, so its detection module cannot be inferred and
# must be named via the manifest's `bite.module`.
_GATE_MODULE = {
    "verify_experiment": "experiment.py",
    "verify_leakage": "leakage.py",
    "verify_multiverse": "multiverse.py",
    "verify_rtm": "rtm.py",
    "verify_powerlaw": "powerlaw.py",
    "verify_spatial": "spatial.py",
    "verify_extrapolation": "extrapolation.py",
    "verify_proportional_hazards": "hazards.py",
}


def _bite_module(manifest: dict[str, Any]) -> str | None:
    """The verification module the bite gate mutates: explicit `bite`, else by gate."""
    bite = manifest.get("bite")
    if bite is not None:
        module: str = bite["module"]
        return module
    return _GATE_MODULE.get(manifest["gate"])


def load_schema() -> dict[str, Any]:
    """Return the parsed trap-manifest JSON schema."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator() -> jsonschema.protocols.Validator:
    schema = load_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def manifest_paths() -> list[Path]:
    """Every trap manifest, sorted for a stable order (one folder per trap)."""
    return sorted(TRAPS_DIR.glob("*/manifest.json"))


def _read_file_rows(path: Path) -> Rows:
    """Load rows from a committed dataset (JSONL / JSON / CSV) in a trap folder."""
    if not path.exists():
        raise FileNotFoundError(f"trap data file not found: {path}")
    if path.suffix == ".jsonl":
        rows: Rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append({k: str(v) for k, v in json.loads(line).items()})
        return rows
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return [{k: str(v) for k, v in row.items()} for row in loaded]


def _data_builder(spec: dict[str, Any], folder: Path) -> Callable[[], Rows]:
    """Turn a manifest ``data`` block into a zero-argument rows builder.

    The loader reads committed rows only (a ``data.jsonl`` file or inline ``rows``);
    synthetic rows are produced offline by ``generated_by`` and committed, not here.
    """
    if "file" in spec:
        path = folder / spec["file"]
        return lambda: _read_file_rows(path)
    rows: Rows = [{k: str(v) for k, v in row.items()} for row in spec["rows"]]
    return lambda: [dict(r) for r in rows]


def generate_rows(manifest: dict[str, Any]) -> Rows:
    """Produce a synthetic trap's rows from its ``generated_by`` block (authoring)."""
    spec = manifest["generated_by"]
    name = spec["generator"]
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}")
    kwargs = {"seed": spec["seed"]}
    if "n" in spec:
        kwargs["n"] = spec["n"]
    return GENERATORS[name](**kwargs)


def _verify_builder(
    gate: str, claim: dict[str, Any], gate_args: dict[str, Any]
) -> Callable[[Rows], Any]:
    """Bind a gate, its role->column claim, and any non-role args into a verify call."""
    call = {**claim, **gate_args}
    return lambda rows: run_gate(gate, rows, call)


def _to_trap(manifest: dict[str, Any], path: Path) -> Trap:
    folder = path.parent
    return Trap(
        id=folder.name,
        pitfall=manifest["pitfall"],
        description=manifest["description"],
        provenance=manifest["provenance"],
        added_by=manifest["added_by"],
        reviewed_by=tuple(manifest["reviewed_by"]),
        status=manifest["status"],
        gate=manifest["gate"],
        claim=dict(manifest["claim"]),
        expected_verdict=manifest["expected_verdict"],
        expected_pivotal=manifest.get("expected_pivotal"),
        rationale=manifest["rationale"],
        naive_verdict=manifest["naive_verdict"],
        naive_rationale=manifest["naive_rationale"],
        bite_module=_bite_module(manifest),
        data=_data_builder(manifest["data"], folder),
        verify=_verify_builder(
            manifest["gate"],
            dict(manifest["claim"]),
            dict(manifest.get("gate_args", {})),
        ),
        # A synthetic trap declares generated_by; a real dataset does not (and needs >=2
        # reviewers, enforced in test_schema).
        is_real="generated_by" not in manifest,
        fixture=str(path.relative_to(BENCHMARK_DIR)),
    )


def load_traps() -> list[Trap]:
    """Load, validate, and resolve every trap manifest; raise on any malformed one."""
    validator = _validator()
    traps: list[Trap] = []
    seen: dict[str, str] = {}
    for path in manifest_paths():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(manifest), key=str)
        if errors:
            raise ValueError(
                f"{path}: invalid trap manifest: "
                + "; ".join(e.message for e in errors)
            )
        trap_id = path.parent.name
        if trap_id in seen:
            raise ValueError(
                f"duplicate trap id {trap_id!r} in {path} and {seen[trap_id]}"
            )
        seen[trap_id] = str(path)
        traps.append(_to_trap(manifest, path))
    return traps
