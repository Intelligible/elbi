"""The contract check catalog: one function per constraint, over a backend.

Each check computes its violations through the :class:`~elbi.quality.backend`
primitives and returns a :class:`~elbi.quality.report.ClauseResult` with a
three-valued verdict and a capped sample of located violations. Value checks tolerate a
share of violations via the field's ``mostly`` (the fraction of present values that must
be valid); structural checks (keys, row count, referential integrity, schema shape) are
strict. A check that cannot be evaluated returns ``inconclusive`` rather than passing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .backend import Cell, CleaningBackend
from .report import INCONCLUSIVE, SOUND, UNSOUND, ClauseResult, Violation


def check_required(
    backend: CleaningBackend, name: str, column: str, mostly: float, cap: int
) -> ClauseResult:
    """Completeness: at least ``mostly`` of all rows have a present value."""
    total = backend.row_count()
    if total == 0:
        return ClauseResult(name, INCONCLUSIVE, "no rows to check", 0, 0)
    missing = backend.missing_indices(column)
    cells: list[Cell] = [(index, None) for index in missing]
    return _tolerance_outcome(
        name, "required", column, total, cells, mostly, cap, _note_missing
    )


def check_type(
    backend: CleaningBackend,
    name: str,
    column: str,
    type_name: str,
    mostly: float,
    cap: int,
) -> ClauseResult:
    """Type conformance: present values parse as the declared type."""
    cells = backend.type_violations(column, type_name)
    return _tolerance_outcome(
        name,
        "type",
        column,
        backend.present_count(column),
        cells,
        mostly,
        cap,
        lambda value: f"{value!r} is not a valid {type_name}",
    )


def check_range(
    backend: CleaningBackend,
    name: str,
    column: str,
    minimum: float | None,
    maximum: float | None,
    mostly: float,
    cap: int,
) -> ClauseResult:
    """Numeric range: present values fall within the inclusive bounds."""
    cells = backend.range_violations(column, minimum, maximum)
    return _tolerance_outcome(
        name,
        "range",
        column,
        backend.present_count(column),
        cells,
        mostly,
        cap,
        lambda value: f"{value!r} is outside [{_bound(minimum)}, {_bound(maximum)}]",
    )


def check_pattern(
    backend: CleaningBackend,
    name: str,
    column: str,
    pattern: str,
    mostly: float,
    cap: int,
) -> ClauseResult:
    """Pattern: present values match the regular expression."""
    cells = backend.pattern_violations(column, pattern)
    return _tolerance_outcome(
        name,
        "pattern",
        column,
        backend.present_count(column),
        cells,
        mostly,
        cap,
        lambda value: f"{value!r} does not match /{pattern}/",
    )


def check_length(
    backend: CleaningBackend,
    name: str,
    column: str,
    min_length: int | None,
    max_length: int | None,
    mostly: float,
    cap: int,
) -> ClauseResult:
    """String length within the inclusive bounds."""
    cells = backend.length_violations(column, min_length, max_length)
    return _tolerance_outcome(
        name,
        "length",
        column,
        backend.present_count(column),
        cells,
        mostly,
        cap,
        lambda value: (
            f"length {len(str(value))} is outside "
            f"[{_bound(min_length)}, {_bound(max_length)}]"
        ),
    )


def check_enum(
    backend: CleaningBackend,
    name: str,
    column: str,
    allowed: Sequence[Any],
    mostly: float,
    cap: int,
) -> ClauseResult:
    """Membership: present values are in the allowed set."""
    cells = backend.enum_violations(column, allowed)
    return _tolerance_outcome(
        name,
        "enum",
        column,
        backend.present_count(column),
        cells,
        mostly,
        cap,
        lambda value: f"{value!r} is not one of the allowed values",
    )


def check_unique(
    backend: CleaningBackend, name: str, column: str, mostly: float, cap: int
) -> ClauseResult:
    """Distinctness: present values do not repeat."""
    cells = backend.duplicate_indices([column])
    return _tolerance_outcome(
        name,
        "unique",
        column,
        backend.present_count(column),
        cells,
        mostly,
        cap,
        lambda value: f"duplicate value {value!r}",
    )


def check_row_count(
    backend: CleaningBackend, name: str, minimum: int | None, maximum: int | None
) -> ClauseResult:
    """The number of rows falls within the declared bounds."""
    total = backend.row_count()
    below = minimum is not None and total < minimum
    above = maximum is not None and total > maximum
    if below or above:
        return ClauseResult(
            name,
            UNSOUND,
            f"{total} rows, outside [{_bound(minimum)}, {_bound(maximum)}]",
            total,
            1,
            (Violation(name, None, None, total, f"row count {total} out of bounds"),),
        )
    return ClauseResult(name, SOUND, f"{total} rows, within bounds", total, 0)


def check_key(
    backend: CleaningBackend,
    name: str,
    columns: Sequence[str],
    *,
    require_present: bool,
    cap: int,
) -> ClauseResult:
    """Uniqueness of a composite key, optionally requiring every key part present.

    A primary key requires presence and uniqueness; an additional unique key requires
    only uniqueness of the rows whose key is fully present.
    """
    total = backend.row_count()
    violations: list[Violation] = []
    if require_present:
        for index in backend.key_missing_indices(columns):
            violations.append(
                Violation(name, None, index + 1, None, "key value missing")
            )
    for index, value in backend.duplicate_indices(columns):
        violations.append(
            Violation(name, None, index + 1, value, f"duplicate key {value!r}")
        )
    violated = len(violations)
    verdict = SOUND if violated == 0 else UNSOUND
    label = "+".join(columns)
    detail = (
        f"key ({label}) holds over {total} rows"
        if violated == 0
        else f"key ({label}) violated by {violated} rows"
    )
    return ClauseResult(name, verdict, detail, total, violated, tuple(violations[:cap]))


def check_foreign_key(
    backend: CleaningBackend,
    name: str,
    columns: Sequence[str],
    allowed_keys: set[tuple[str, ...]] | None,
    resource: str,
    cap: int,
) -> ClauseResult:
    """Referential integrity: local key values exist in the referenced resource.

    ``allowed_keys`` is ``None`` when the referenced resource is not available to the
    checker, which yields ``inconclusive`` rather than a false pass.
    """
    total = backend.row_count()
    if allowed_keys is None:
        return ClauseResult(
            name,
            INCONCLUSIVE,
            f"reference {resource!r} not available to check",
            total,
            0,
        )
    cells = backend.reference_violations(columns, allowed_keys)
    violations = tuple(
        Violation(name, None, index + 1, value, f"{value!r} not in {resource}")
        for index, value in cells[:cap]
    )
    violated = len(cells)
    verdict = SOUND if violated == 0 else UNSOUND
    label = "+".join(columns)
    detail = (
        f"all ({label}) values found in {resource}"
        if violated == 0
        else f"{violated} ({label}) values missing from {resource}"
    )
    return ClauseResult(name, verdict, detail, total, violated, violations)


def check_columns_match(
    backend: CleaningBackend, name: str, field_names: Sequence[str]
) -> ClauseResult:
    """The table's columns are exactly the declared fields, in order."""
    actual = backend.columns()
    expected = list(field_names)
    if actual == expected:
        return ClauseResult(name, SOUND, "columns match the declared fields", 0, 0)
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    notes = []
    if missing:
        notes.append(f"missing {missing}")
    if extra:
        notes.append(f"unexpected {extra}")
    if not notes:
        notes.append("columns are out of declared order")
    detail = "; ".join(notes)
    return ClauseResult(
        name, UNSOUND, detail, 0, 1, (Violation(name, None, None, actual, detail),)
    )


def _tolerance_outcome(
    name: str,
    kind: str,
    column: str,
    checked: int,
    violated_cells: list[Cell],
    mostly: float,
    cap: int,
    note: Any,
) -> ClauseResult:
    """Build a value-check result under ``mostly`` tolerance over present values.

    With no present values there is nothing to violate, so the clause passes vacuously.
    Otherwise the clause is sound when the valid share meets ``mostly``.
    """
    violated = len(violated_cells)
    if checked == 0:
        return ClauseResult(name, SOUND, "no present values to check", 0, 0)
    valid_rate = (checked - violated) / checked
    verdict = SOUND if valid_rate >= mostly else UNSOUND
    sample = tuple(
        Violation(name, column, index + 1, value, note(value))
        for index, value in violated_cells[:cap]
    )
    if violated == 0:
        detail = f"{kind}: {checked} values, all valid"
    else:
        tolerated = "" if mostly >= 1.0 else f"; tolerance {1.0 - mostly:.1%}"
        rate = violated / checked
        detail = f"{kind}: {violated}/{checked} invalid ({rate:.1%}){tolerated}"
    return ClauseResult(name, verdict, detail, checked, violated, sample)


def _bound(value: float | int | None) -> str:
    return "-inf" if value is None else str(value)


def _note_missing(_value: Any) -> str:
    return "value is missing"
