"""The Polars compute backend: push the operation catalog into a lazy frame.

Polars is the in-process engine: a multi-threaded columnar query engine with a lazy
optimizer and a streaming (larger-than-memory) collect. Every operation builds a lazy
expression that the engine plans and runs, materializing only the small result.

Parity with the ``list[dict]`` reference backend is the contract, so this backend keeps
the same *string-first* semantics: at construction every column is cast to ``Utf8`` (the
reference's ``str(value)`` view), numbers parse with a non-strict cast to ``Float64``
(like ``to_float``), and a cell is missing when null or empty. Type conformance
uses the reference's own :func:`parses_as` through an expression, so the richer parsing
rules (``"5.0"`` is a valid integer, the boolean spellings) match exactly.

An optional extra (``pip install "elbi[polars]"``); the core never imports it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from ._stats import parses_as
from .backend import Cell, ColumnStats, Row

_TYPE_ORDER = ("boolean", "integer", "number", "date", "datetime")
_TOP_VALUES = 20
_HASH_MULT = 2654435761
_SEED_MULT = 40503
_MASK32 = 0xFFFFFFFF


def _pl() -> Any:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            'the Polars compute backend needs polars: pip install "elbi[polars]"'
        ) from exc
    return pl


class PolarsFrame:
    """A :class:`ComputeBackend` over a Polars frame, string-first for parity."""

    def __init__(self, frame: Any) -> None:
        pl = _pl()
        lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        self._columns = list(lazy.collect_schema().names())
        # The canonical string view: every column as Utf8, matching the reference's
        # str(value). Numeric and type operations parse from this view.
        self._lf = lazy.select([pl.col(c).cast(pl.Utf8) for c in self._columns])

    # -- structure ------------------------------------------------------------
    def row_count(self) -> int:
        """The number of rows."""
        pl = _pl()
        return int(self._lf.select(pl.len()).collect().item())

    def columns(self) -> list[str]:
        """Column names, in order."""
        return list(self._columns)

    def has_column(self, column: str) -> bool:
        """Whether ``column`` is present."""
        return column in self._columns

    # -- profiling ------------------------------------------------------------
    def profile(self, columns: Sequence[str]) -> dict[str, ColumnStats]:
        """Per-column statistics via lazy aggregates."""
        total = self.row_count()
        return {column: self._profile_column(column, total) for column in columns}

    def _profile_column(self, column: str, total: int) -> ColumnStats:
        pl = _pl()
        present = _present(pl, column)
        col = pl.col(column)
        aggs = [
            present.sum().alias("present"),
            col.filter(present).n_unique().alias("distinct"),
            _num(pl, column).filter(present).min().alias("min"),
            _num(pl, column).filter(present).max().alias("max"),
        ]
        aggs += [
            (present & _conforms(pl, column, t)).sum().alias(f"n_{t}")
            for t in _TYPE_ORDER
        ]
        stats = self._lf.select(aggs).collect().to_dicts()[0]
        present_n = int(stats["present"] or 0)
        conforms = {t: int(stats[f"n_{t}"] or 0) for t in _TYPE_ORDER}
        inferred = _infer_type(present_n, conforms)
        numeric = inferred in ("integer", "number")
        # Top-k and values-seen-once from the present value counts.
        vc = (
            self._lf.select(col.filter(present).alias("v"))
            .collect()
            .get_column("v")
            .to_list()
        )
        counts = Counter(vc)
        seen_once = sum(1 for n in counts.values() if n == 1)
        minimum = float(stats["min"]) if numeric and stats["min"] is not None else None
        maximum = float(stats["max"]) if numeric and stats["max"] is not None else None
        return ColumnStats(
            name=column,
            count=total,
            present=present_n,
            distinct=len(counts),
            inferred_type=inferred,
            minimum=minimum,
            maximum=maximum,
            top_values=tuple(counts.most_common(_TOP_VALUES)),
            fraction_unique_once=seen_once / len(counts) if counts else 0.0,
        )

    # -- located validity primitives ------------------------------------------
    def present_count(self, column: str) -> int:
        """Rows whose ``column`` is present."""
        pl = _pl()
        return int(self._lf.select(_present(pl, column).sum()).collect().item() or 0)

    def missing_indices(self, column: str) -> list[int]:
        """Row positions where ``column`` is missing."""
        pl = _pl()
        return self._located_indices(~_present(pl, column))

    def type_violations(self, column: str, type_name: str) -> list[Cell]:
        """Present cells that do not parse as ``type_name`` (via the shared parser)."""
        pl = _pl()
        bad = _present(pl, column) & ~_conforms(pl, column, type_name)
        return self._located(bad, column)

    def range_violations(
        self, column: str, minimum: float | None, maximum: float | None
    ) -> list[Cell]:
        """Present cells outside the inclusive numeric range (or non-numeric)."""
        pl = _pl()
        num = _num(pl, column)
        bad = num.is_null()
        if minimum is not None:
            bad = bad | (num < minimum)
        if maximum is not None:
            bad = bad | (num > maximum)
        return self._located(_present(pl, column) & bad, column)

    def pattern_violations(self, column: str, pattern: str) -> list[Cell]:
        """Present cells whose string form does not match ``pattern``."""
        pl = _pl()
        bad = ~pl.col(column).str.contains(pattern)
        return self._located(_present(pl, column) & bad, column)

    def length_violations(
        self, column: str, min_length: int | None, max_length: int | None
    ) -> list[Cell]:
        """Present cells whose string length is outside the inclusive bounds."""
        pl = _pl()
        length = pl.col(column).str.len_chars()
        conditions = []
        if min_length is not None:
            conditions.append(length < min_length)
        if max_length is not None:
            conditions.append(length > max_length)
        if not conditions:
            return []
        bad = conditions[0]
        for condition in conditions[1:]:
            bad = bad | condition
        return self._located(_present(pl, column) & bad, column)

    def enum_violations(self, column: str, allowed: Sequence[Any]) -> list[Cell]:
        """Present cells whose value is not in the allowed set."""
        pl = _pl()
        listed = [str(value) for value in allowed]
        bad = ~pl.col(column).is_in(listed)
        return self._located(_present(pl, column) & bad, column)

    def duplicate_indices(self, columns: Sequence[str]) -> list[Cell]:
        """Rows whose combined key (all present) repeats."""
        pl = _pl()
        present = _all_present(pl, columns)
        counts = pl.struct(list(columns)).count().over(list(columns))
        frame = (
            self._lf.with_row_index("_row")
            .filter(present)
            .filter(counts > 1)
            .select(["_row", *columns])
            .collect()
        )
        return [
            (int(record["_row"]), [str(record[c]) for c in columns])
            for record in frame.to_dicts()
        ]

    def key_missing_indices(self, columns: Sequence[str]) -> list[int]:
        """Rows where any key column is missing."""
        pl = _pl()
        return self._located_indices(~_all_present(pl, columns))

    def reference_violations(
        self, columns: Sequence[str], allowed_keys: set[tuple[str, ...]]
    ) -> list[Cell]:
        """Rows whose combined key (all present) is not in ``allowed_keys``."""
        pl = _pl()
        present = _all_present(pl, columns)
        frame = (
            self._lf.with_row_index("_row")
            .filter(present)
            .select(["_row", *columns])
            .collect()
        )
        out: list[Cell] = []
        for record in frame.to_dicts():
            key = tuple(str(record[c]) for c in columns)
            if key not in allowed_keys:
                out.append((int(record["_row"]), list(key)))
        return out

    # -- materialization ------------------------------------------------------
    def to_rows(self, limit: int | None = None) -> list[Row]:
        """Materialize (up to ``limit``) rows as dicts."""
        lf = self._lf if limit is None else self._lf.head(limit)
        return [dict(record) for record in lf.collect().to_dicts()]

    def sample_rows(self, cap: int, *, seed: int = 0) -> list[Row]:
        """A deterministic, position-uniform subsample of at most ``cap`` rows.

        Matches the reference and DuckDB backends' hash-of-position rule.
        """
        pl = _pl()
        total = self.row_count()
        if total <= cap:
            return self.to_rows()
        offset = seed * _SEED_MULT
        position = pl.int_range(1, total + 1, dtype=pl.UInt64)
        digest = (position * _HASH_MULT + offset) % (_MASK32 + 1)
        frame = self._lf.with_columns(digest.alias("_h")).filter(
            (pl.col("_h") % total) < cap
        )
        return [
            {k: v for k, v in record.items() if k != "_h"}
            for record in frame.collect().to_dicts()
        ]

    # -- internals ------------------------------------------------------------
    def _located(self, mask: Any, column: str) -> list[Cell]:
        frame = (
            self._lf.with_row_index("_row")
            .filter(mask)
            .select(["_row", column])
            .collect()
        )
        return [(int(r["_row"]), r[column]) for r in frame.to_dicts()]

    def _located_indices(self, mask: Any) -> list[int]:
        frame = self._lf.with_row_index("_row").filter(mask).select("_row").collect()
        return [int(r["_row"]) for r in frame.to_dicts()]


def _present(pl: Any, column: str) -> Any:
    return pl.col(column).is_not_null() & (pl.col(column) != "")


def _all_present(pl: Any, columns: Sequence[str]) -> Any:
    mask = _present(pl, columns[0])
    for column in columns[1:]:
        mask = mask & _present(pl, column)
    return mask


def _num(pl: Any, column: str) -> Any:
    return pl.col(column).cast(pl.Float64, strict=False)


def _conforms(pl: Any, column: str, type_name: str) -> Any:
    return pl.col(column).map_elements(
        lambda v: v is None or parses_as(str(v), type_name),
        return_dtype=pl.Boolean,
    )


def _infer_type(present: int, conforms: dict[str, int]) -> str:
    """The narrowest type all present values conform to, else ``string``."""
    if present == 0:
        return "string"
    for type_name in _TYPE_ORDER:
        if conforms[type_name] == present:
            return type_name
    return "string"
