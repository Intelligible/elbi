"""Check a table of rows against a contract and return a three-valued verdict.

:func:`verify_contract` enumerates one clause per declared constraint, runs each over
the engine-agnostic backend, and combines them as a conjunction. The backend memoizes
per-column work, so many clauses on one column share a single pass. Foreign-key clauses
resolve their referenced rows from ``references``; a reference the caller does
not supply yields ``inconclusive``, never a false pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import checks
from .backend import ComputeBackend, Row, backend_for, key_set
from .contract import DataContract, FieldSpec
from .report import (
    INCONCLUSIVE,
    ClauseResult,
    ContractReport,
    Violation,
    combine_verdict,
)

#: Located violations kept per clause and in the whole report, so a verdict stays small
#: and deterministic on dirty data while still showing concrete offending rows.
_CLAUSE_SAMPLE_CAP = 10
_REPORT_SAMPLE_CAP = 50


def verify_contract(
    data: Any,
    contract: DataContract,
    *,
    references: Mapping[str, Sequence[Row]] | None = None,
) -> ContractReport:
    """Check ``data`` against ``contract`` and return the combined verdict.

    ``data`` is anything a compute backend handles (``list[dict]`` rows or an engine
    handle over a larger-than-memory file), or a backend itself. ``references`` maps a
    foreign key's referenced resource name to its rows, for referential integrity;
    omit it and those clauses are ``inconclusive``.
    """
    backend = data if isinstance(data, ComputeBackend) else backend_for(data)
    resolved = references or {}
    clauses: list[ClauseResult] = []

    for spec in contract.fields:
        clauses.extend(_field_clauses(backend, spec))

    clauses.extend(_table_clauses(backend, contract, resolved))

    verdict = combine_verdict(tuple(clauses))
    violations: list[Violation] = []
    for clause in clauses:
        violations.extend(clause.sample)
    return ContractReport(
        verdict=verdict,
        clauses=tuple(clauses),
        row_count=backend.row_count(),
        violations=tuple(violations[:_REPORT_SAMPLE_CAP]),
    )


def _field_clauses(backend: Any, spec: FieldSpec) -> list[ClauseResult]:
    """The clauses a single field contributes."""
    if not backend.has_column(spec.name):
        return [
            ClauseResult(
                f"{spec.name}.present",
                INCONCLUSIVE,
                "column not present in data",
                0,
                0,
            )
        ]
    cap = _CLAUSE_SAMPLE_CAP
    mostly = spec.mostly
    out = [
        checks.check_type(
            backend, f"{spec.name}.type", spec.name, spec.type, mostly, cap
        )
    ]
    c = spec.constraints
    if c.required:
        out.append(
            checks.check_required(
                backend, f"{spec.name}.required", spec.name, mostly, cap
            )
        )
    if c.minimum is not None or c.maximum is not None:
        out.append(
            checks.check_range(
                backend,
                f"{spec.name}.range",
                spec.name,
                c.minimum,
                c.maximum,
                mostly,
                cap,
            )
        )
    if c.pattern is not None:
        out.append(
            checks.check_pattern(
                backend, f"{spec.name}.pattern", spec.name, c.pattern, mostly, cap
            )
        )
    if c.min_length is not None or c.max_length is not None:
        out.append(
            checks.check_length(
                backend,
                f"{spec.name}.length",
                spec.name,
                c.min_length,
                c.max_length,
                mostly,
                cap,
            )
        )
    if c.enum is not None:
        out.append(
            checks.check_enum(
                backend, f"{spec.name}.enum", spec.name, c.enum, mostly, cap
            )
        )
    if c.unique:
        out.append(
            checks.check_unique(backend, f"{spec.name}.unique", spec.name, mostly, cap)
        )
    return out


def _table_clauses(
    backend: Any, contract: DataContract, references: Mapping[str, Sequence[Row]]
) -> list[ClauseResult]:
    """The clauses the table block contributes."""
    table = contract.table
    cap = _CLAUSE_SAMPLE_CAP
    out: list[ClauseResult] = []

    if table.row_count_min is not None or table.row_count_max is not None:
        out.append(
            checks.check_row_count(
                backend, "table.row_count", table.row_count_min, table.row_count_max
            )
        )

    if table.primary_key:
        out.append(
            _key_clause(
                backend,
                "table.primary_key",
                table.primary_key,
                require_present=True,
                cap=cap,
            )
        )

    for columns in table.unique_keys:
        name = f"table.unique[{'+'.join(columns)}]"
        out.append(_key_clause(backend, name, columns, require_present=False, cap=cap))

    for fk in table.foreign_keys:
        name = f"table.foreign_key[{'+'.join(fk.fields)}]"
        out.append(_foreign_key_clause(backend, name, fk, references, cap))

    if table.columns_match:
        out.append(
            checks.check_columns_match(
                backend, "table.columns_match", contract.field_names()
            )
        )
    return out


def _key_clause(
    backend: Any,
    name: str,
    columns: Sequence[str],
    *,
    require_present: bool,
    cap: int,
) -> ClauseResult:
    """A key clause, or inconclusive if a key column is absent from the data."""
    absent = [c for c in columns if not backend.has_column(c)]
    if absent:
        return ClauseResult(name, INCONCLUSIVE, f"columns not present: {absent}", 0, 0)
    return checks.check_key(
        backend, name, columns, require_present=require_present, cap=cap
    )


def _foreign_key_clause(
    backend: Any,
    name: str,
    fk: Any,
    references: Mapping[str, Sequence[Row]],
    cap: int,
) -> ClauseResult:
    """A referential-integrity clause, resolving the reference from ``references``."""
    absent = [c for c in fk.fields if not backend.has_column(c)]
    if absent:
        return ClauseResult(name, INCONCLUSIVE, f"columns not present: {absent}", 0, 0)
    resource = fk.reference.resource
    ref_rows = references.get(resource)
    allowed = key_set(ref_rows, fk.reference.fields) if ref_rows is not None else None
    return checks.check_foreign_key(backend, name, fk.fields, allowed, resource, cap)
