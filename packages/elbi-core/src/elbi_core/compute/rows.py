"""The dependency-free reference backend over ``list[dict]`` rows.

Implements every operation in the :class:`~elbi.compute.backend.ComputeBackend`
catalog in pure Python, with no third-party dependency. It is the reference semantics
the engine backends must match, and the backend used whenever data is already an
in-memory list of row dicts. Column values are extracted lazily and memoized, so many
operations on one column share a single pass over the data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from ._stats import is_missing, parses_as, to_float
from .backend import Cell, ColumnStats, Row

#: Type inference order: the narrowest type whose parser accepts every present value.
#: Integer precedes number so an all-whole column reads as integer, not float.
_TYPE_ORDER = ("boolean", "integer", "number", "date", "datetime")

#: Most-frequent values retained per column when profiling.
_TOP_VALUES = 20

#: Constants for a deterministic, position-uniform subsample: a multiplicative hash of
#: the one-based row position, with the seed folded in additively so it stays native
#: (no per-row callback) on the engine backends.
_HASH_MULT = 2654435761
_SEED_MULT = 40503
_MASK32 = 0xFFFFFFFF


def key_set(rows: Sequence[Row], columns: Sequence[str]) -> set[tuple[str, ...]]:
    """The set of present combined-key tuples over ``columns`` in ``rows``.

    Keys with any missing component are excluded (they cannot serve as a reference
    target). Values are normalized to strings so a key compares equal across engines
    and source formats (a CSV ``"5"`` matches a JSON ``5``).
    """
    keys: set[tuple[str, ...]] = set()
    for row in rows:
        values = [row.get(column) for column in columns]
        if any(is_missing(value) for value in values):
            continue
        keys.add(tuple(_norm(value) for value in values))
    return keys


class RowsFrame:
    """A :class:`ComputeBackend` over an in-memory ``list[dict]`` of rows."""

    def __init__(self, rows: Sequence[Row]) -> None:
        self._rows = rows
        self._columns: list[str] | None = None
        self._present: dict[str, list[Cell]] = {}

    # -- structure ------------------------------------------------------------
    def row_count(self) -> int:
        """The number of rows."""
        return len(self._rows)

    def columns(self) -> list[str]:
        """Column names, in first-seen order across all rows."""
        if self._columns is None:
            seen: dict[str, None] = {}
            for row in self._rows:
                for key in row:
                    seen.setdefault(key, None)
            self._columns = list(seen)
        return self._columns

    def has_column(self, column: str) -> bool:
        """Whether ``column`` appears in any row."""
        return column in self.columns()

    # -- profiling ------------------------------------------------------------
    def profile(self, columns: Sequence[str]) -> dict[str, ColumnStats]:
        """Per-column statistics, computed by scanning each column's present values."""
        return {column: self._profile_column(column) for column in columns}

    def _profile_column(self, column: str) -> ColumnStats:
        present_values = [value for _, value in self._present_cells(column)]
        counts = Counter(_norm(value) for value in present_values)
        distinct = len(counts)
        seen_once = sum(1 for count in counts.values() if count == 1)
        numbers = [n for n in (to_float(v) for v in present_values) if n is not None]
        inferred = _infer_type(present_values)
        numeric = inferred in ("integer", "number")
        return ColumnStats(
            name=column,
            count=len(self._rows),
            present=len(present_values),
            distinct=distinct,
            inferred_type=inferred,
            minimum=min(numbers) if numbers and numeric else None,
            maximum=max(numbers) if numbers and numeric else None,
            top_values=tuple(counts.most_common(_TOP_VALUES)),
            fraction_unique_once=seen_once / distinct if distinct else 0.0,
        )

    # -- located validity primitives ------------------------------------------
    def _present_cells(self, column: str) -> list[Cell]:
        cached = self._present.get(column)
        if cached is None:
            cached = [
                (index, row.get(column))
                for index, row in enumerate(self._rows)
                if not is_missing(row.get(column))
            ]
            self._present[column] = cached
        return cached

    def present_count(self, column: str) -> int:
        """The number of rows whose ``column`` value is not missing."""
        return len(self._present_cells(column))

    def missing_indices(self, column: str) -> list[int]:
        """Row indices where ``column`` is missing."""
        return [
            index for index, row in enumerate(self._rows) if is_missing(row.get(column))
        ]

    def type_violations(self, column: str, type_name: str) -> list[Cell]:
        """Present cells that do not parse as ``type_name``."""
        return [
            (index, _norm(value))
            for index, value in self._present_cells(column)
            if not parses_as(value, type_name)
        ]

    def range_violations(
        self, column: str, minimum: float | None, maximum: float | None
    ) -> list[Cell]:
        """Present cells outside the inclusive numeric range (or non-numeric)."""
        out: list[Cell] = []
        for index, value in self._present_cells(column):
            number = to_float(value)
            if (
                number is None
                or (minimum is not None and number < minimum)
                or (maximum is not None and number > maximum)
            ):
                out.append((index, _norm(value)))
        return out

    def pattern_violations(self, column: str, pattern: str) -> list[Cell]:
        """Present cells whose string form does not match ``pattern``."""
        import re

        compiled = re.compile(pattern)
        return [
            (index, _norm(value))
            for index, value in self._present_cells(column)
            if compiled.search(str(value)) is None
        ]

    def length_violations(
        self, column: str, min_length: int | None, max_length: int | None
    ) -> list[Cell]:
        """Present cells whose string length is outside the inclusive bounds."""
        out: list[Cell] = []
        for index, value in self._present_cells(column):
            length = len(str(value))
            if (min_length is not None and length < min_length) or (
                max_length is not None and length > max_length
            ):
                out.append((index, _norm(value)))
        return out

    def enum_violations(self, column: str, allowed: Sequence[Any]) -> list[Cell]:
        """Present cells whose value is not in the allowed set."""
        allowed_set = {_norm(value) for value in allowed}
        return [
            (index, _norm(value))
            for index, value in self._present_cells(column)
            if _norm(value) not in allowed_set
        ]

    def duplicate_indices(self, columns: Sequence[str]) -> list[Cell]:
        """Rows whose combined ``columns`` value (all present) repeats."""
        counts: dict[tuple[str, ...], int] = {}
        located: list[tuple[int, tuple[str, ...]]] = []
        for index, row in enumerate(self._rows):
            values = [row.get(column) for column in columns]
            if any(is_missing(value) for value in values):
                continue
            key = tuple(_norm(value) for value in values)
            counts[key] = counts.get(key, 0) + 1
            located.append((index, key))
        return [(index, list(key)) for index, key in located if counts[key] > 1]

    def key_missing_indices(self, columns: Sequence[str]) -> list[int]:
        """Rows where any of the key ``columns`` is missing."""
        return [
            index
            for index, row in enumerate(self._rows)
            if any(is_missing(row.get(column)) for column in columns)
        ]

    def reference_violations(
        self, columns: Sequence[str], allowed_keys: set[tuple[str, ...]]
    ) -> list[Cell]:
        """Rows whose combined key (all present) is not in ``allowed_keys``."""
        out: list[Cell] = []
        for index, row in enumerate(self._rows):
            values = [row.get(column) for column in columns]
            if any(is_missing(value) for value in values):
                continue
            key = tuple(_norm(value) for value in values)
            if key not in allowed_keys:
                out.append((index, list(key)))
        return out

    # -- materialization ------------------------------------------------------
    def to_rows(self, limit: int | None = None) -> list[Row]:
        """Return (up to ``limit``) rows as fresh dicts."""
        rows = self._rows if limit is None else self._rows[:limit]
        return [dict(row) for row in rows]

    def sample_rows(self, cap: int, *, seed: int = 0) -> list[Row]:
        """A deterministic, position-uniform subsample of at most ``cap`` rows.

        Returns every row when there are no more than ``cap``. Otherwise keeps a row
        when a multiplicative hash of its position falls in the retained fraction, so
        the sample is reproducible (no global RNG) and roughly uniform over positions
        rather than truncated to the first rows.
        """
        total = len(self._rows)
        if total <= cap:
            return [dict(row) for row in self._rows]
        offset = seed * _SEED_MULT
        out: list[Row] = []
        for index, row in enumerate(self._rows):
            digest = ((index + 1) * _HASH_MULT + offset) & _MASK32
            if digest % total < cap:
                out.append(dict(row))
        return out


def _infer_type(values: Sequence[Any]) -> str:
    """The narrowest declared type every present value conforms to, else ``string``.

    Inference is string-first: each value is judged by its ``str`` form, so a column is
    typed the same whether it arrives as native objects (a JSON integer) or as text (a
    CSV cell), and the engine backends -- which see a columnar string cast -- agree.
    """
    if not values:
        return "string"
    for type_name in _TYPE_ORDER:
        if all(parses_as(_norm(value), type_name) for value in values):
            return type_name
    return "string"


def _norm(value: Any) -> str:
    """Normalize a cell value to a string for cross-engine comparison."""
    return str(value)
