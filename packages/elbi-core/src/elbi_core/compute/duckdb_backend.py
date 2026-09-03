"""The DuckDB compute backend: push the operation catalog down to SQL.

DuckDB is the primary engine for scale: it queries an in-memory Arrow table zero-copy
and scans a larger-than-memory Parquet/CSV without loading it, spilling to disk under a
memory limit. Every operation compiles to one SQL statement that runs in the engine, and
only the small result crosses back into Python; the rows themselves never do (except a
bounded sample or an explicit materialization).

Parity with the ``list[dict]`` reference backend is the contract: this backend must
return the same profile and the same located violations. It gets there by matching the
reference's *string-first* semantics -- values compare as ``CAST(... AS VARCHAR)`` (like
the reference's ``str(value)``), numbers as ``TRY_CAST(... AS DOUBLE)`` (like
``to_float``), and a cell is missing when NULL or the empty string. Type conformance
has richer rules (``"5.0"`` is a valid integer, the truthy/falsey boolean spellings), so
rather than re-derive them in SQL this backend registers the reference's own
:func:`parses_as` as a scalar UDF and calls it in the query -- identical semantics, and
the scan still streams in the engine.

This backend is an optional extra (``pip install "elbi[duckdb]"``); the core
never imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._stats import parses_as
from .backend import Cell, ColumnStats, Row

#: Field types the profiler infers, narrowest first (matches the reference order).
_TYPE_ORDER = ("boolean", "integer", "number", "date", "datetime")
_TOP_VALUES = 20
_HASH_MULT = 2654435761
_SEED_MULT = 40503
_MASK32 = 0xFFFFFFFF


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            'the DuckDB compute backend needs duckdb: pip install "elbi[duckdb]"'
        ) from exc
    return duckdb


def _connect() -> Any:
    """A fresh DuckDB connection with the shared type-parsing UDF registered."""
    con = _require_duckdb().connect()
    con.create_function(
        "intel_parses_as", _parses_as_udf, ["VARCHAR", "VARCHAR"], "BOOLEAN"
    )
    return con


class DuckDBFrame:
    """A :class:`ComputeBackend` over a DuckDB relation, pushing every op to SQL."""

    def __init__(self, source: Any, *, view: str = "_src") -> None:
        duckdb = _require_duckdb()
        self._con = _connect()
        self._view = view
        if isinstance(source, duckdb.DuckDBPyRelation):
            self._con.register(view, source.arrow())
        else:
            self._con.register(view, source)
        self._columns = [
            d[0] for d in self._q(f"SELECT * FROM {view} LIMIT 0").description
        ]

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_arrow(cls, table: Any) -> DuckDBFrame:
        """Wrap an Arrow table (registered zero-copy)."""
        return cls(table)

    @classmethod
    def from_path(cls, path: Path) -> DuckDBFrame:
        """Scan a data file lazily, without loading it (out-of-core)."""
        reader = {
            ".parquet": "read_parquet",
            ".csv": "read_csv_auto",
            ".json": "read_json_auto",
            ".ndjson": "read_json_auto",
            ".jsonl": "read_json_auto",
        }.get(path.suffix.lower())
        if reader is None:
            raise ValueError(f"DuckDB cannot scan {path.suffix!r}: {path}")
        frame = cls.__new__(cls)
        frame._con = _connect()
        frame._view = "_src"
        literal = _escape_literal(str(path))
        frame._con.execute(f"CREATE VIEW _src AS SELECT * FROM {reader}('{literal}')")
        frame._columns = [
            d[0] for d in frame._q("SELECT * FROM _src LIMIT 0").description
        ]
        return frame

    # -- structure ------------------------------------------------------------
    def row_count(self) -> int:
        """The number of rows."""
        return int(self._q(f"SELECT count(*) FROM {self._view}").fetchone()[0])

    def columns(self) -> list[str]:
        """Column names, in schema order."""
        return list(self._columns)

    def has_column(self, column: str) -> bool:
        """Whether ``column`` is present."""
        return column in self._columns

    # -- profiling ------------------------------------------------------------
    def profile(self, columns: Sequence[str]) -> dict[str, ColumnStats]:
        """Per-column statistics, via pushed-down aggregates."""
        total = self.row_count()
        out: dict[str, ColumnStats] = {}
        for column in columns:
            out[column] = self._profile_column(column, total)
        return out

    def _profile_column(self, column: str, total: int) -> ColumnStats:
        text, present = _text(column), _present(column)
        # One aggregate per candidate type counts how many present values conform, so
        # the narrowest all-conforming type (the reference's inference) is read off.
        type_counts = ", ".join(
            f"count(*) FILTER (WHERE {present} AND intel_parses_as({text}, '{t}')) "
            f"AS n_{t}"
            for t in _TYPE_ORDER
        )
        row = self._q(
            f"SELECT count(*) FILTER (WHERE {present}) AS present, "
            f"count(DISTINCT {text}) FILTER (WHERE {present}) AS distinct, "
            f"{type_counts} FROM {self._view}"
        ).fetchone()
        present_n, distinct = int(row[0]), int(row[1])
        conforms = {t: int(v) for t, v in zip(_TYPE_ORDER, row[2:], strict=True)}
        inferred = _infer_type(present_n, conforms)
        minimum = maximum = None
        if inferred in ("integer", "number") and present_n:
            num = _num(column)
            mn, mx = self._q(
                f"SELECT min({num}), max({num}) FROM {self._view} WHERE {present}"
            ).fetchone()
            minimum = None if mn is None else float(mn)
            maximum = None if mx is None else float(mx)
        top = self._q(
            f"SELECT {text}, count(*) AS _c FROM {self._view} WHERE {present} "
            f"GROUP BY {text} ORDER BY _c DESC, {text} ASC LIMIT {_TOP_VALUES}"
        ).fetchall()
        seen_once = self._q(
            f"SELECT count(*) FROM (SELECT {text} FROM {self._view} WHERE {present} "
            f"GROUP BY {text} HAVING count(*) = 1)"
        ).fetchone()[0]
        return ColumnStats(
            name=column,
            count=total,
            present=present_n,
            distinct=distinct,
            inferred_type=inferred,
            minimum=minimum,
            maximum=maximum,
            top_values=tuple((v, int(c)) for v, c in top),
            fraction_unique_once=(int(seen_once) / distinct) if distinct else 0.0,
        )

    # -- located validity primitives ------------------------------------------
    def present_count(self, column: str) -> int:
        """Rows whose ``column`` is present."""
        return int(
            self._q(
                f"SELECT count(*) FROM {self._view} WHERE {_present(column)}"
            ).fetchone()[0]
        )

    def missing_indices(self, column: str) -> list[int]:
        """Row positions where ``column`` is missing."""
        return self._rows_where(_missing(column), [])

    def type_violations(self, column: str, type_name: str) -> list[Cell]:
        """Present cells that do not parse as ``type_name`` (via the shared UDF)."""
        text = _text(column)
        cond = f"{_present(column)} AND NOT intel_parses_as({text}, '{type_name}')"
        return self._located(cond, text)

    def range_violations(
        self, column: str, minimum: float | None, maximum: float | None
    ) -> list[Cell]:
        """Present cells outside the inclusive numeric range (or non-numeric)."""
        num = _num(column)
        clauses = [f"{num} IS NULL"]
        if minimum is not None:
            clauses.append(f"{num} < {minimum!r}")
        if maximum is not None:
            clauses.append(f"{num} > {maximum!r}")
        cond = f"{_present(column)} AND ({' OR '.join(clauses)})"
        return self._located(cond, _text(column))

    def pattern_violations(self, column: str, pattern: str) -> list[Cell]:
        """Present cells whose string form does not match ``pattern``."""
        text = _text(column)
        cond = (
            f"{_present(column)} AND NOT regexp_matches({text}, "
            f"'{_escape_literal(pattern)}')"
        )
        return self._located(cond, text)

    def length_violations(
        self, column: str, min_length: int | None, max_length: int | None
    ) -> list[Cell]:
        """Present cells whose string length is outside the inclusive bounds."""
        text = _text(column)
        clauses = []
        if min_length is not None:
            clauses.append(f"length({text}) < {min_length}")
        if max_length is not None:
            clauses.append(f"length({text}) > {max_length}")
        cond = f"{_present(column)} AND ({' OR '.join(clauses) or 'FALSE'})"
        return self._located(cond, text)

    def enum_violations(self, column: str, allowed: Sequence[Any]) -> list[Cell]:
        """Present cells whose value is not in the allowed set."""
        text = _text(column)
        listed = ", ".join(f"'{_escape_literal(str(v))}'" for v in allowed) or "NULL"
        cond = f"{_present(column)} AND {text} NOT IN ({listed})"
        return self._located(cond, text)

    def duplicate_indices(self, columns: Sequence[str]) -> list[Cell]:
        """Rows whose combined key (all present) repeats."""
        keys = [_text(c) for c in columns]
        aliased = ", ".join(f"{k} AS _k{i}" for i, k in enumerate(keys))
        names = ", ".join(f"_k{i}" for i in range(len(keys)))
        present = " AND ".join(_present(c) for c in columns)
        numbered = (
            f"SELECT row_number() OVER () - 1 AS _row, {aliased}, "
            f"count(*) OVER (PARTITION BY {', '.join(keys)}) AS _n "
            f"FROM {self._view} WHERE {present}"
        )
        rows = self._q(
            f"SELECT _row, {names} FROM ({numbered}) WHERE _n > 1"
        ).fetchall()
        return [(int(r[0]), [str(v) for v in r[1:]]) for r in rows]

    def key_missing_indices(self, columns: Sequence[str]) -> list[int]:
        """Rows where any key column is missing."""
        cond = " OR ".join(_missing(c) for c in columns)
        return self._rows_where(cond, [])

    def reference_violations(
        self, columns: Sequence[str], allowed_keys: set[tuple[str, ...]]
    ) -> list[Cell]:
        """Rows whose combined key (all present) is not in ``allowed_keys``."""
        keys = [_text(c) for c in columns]
        aliased = ", ".join(f"{k} AS _k{i}" for i, k in enumerate(keys))
        names = [f"_k{i}" for i in range(len(keys))]
        name_sql = ", ".join(names)
        present = " AND ".join(_present(c) for c in columns)
        numbered = (
            f"SELECT row_number() OVER () - 1 AS _row, {aliased} "
            f"FROM {self._view} WHERE {present}"
        )
        if not allowed_keys:
            rows = self._q(f"SELECT _row, {name_sql} FROM ({numbered})").fetchall()
            return [(int(r[0]), [str(v) for v in r[1:]]) for r in rows]
        values = ", ".join(
            "(" + ", ".join(f"'{_escape_literal(part)}'" for part in key) + ")"
            for key in allowed_keys
        )
        tuple_expr = "(" + name_sql + ")"
        rows = self._q(
            f"SELECT _row, {name_sql} FROM ({numbered}) "
            f"WHERE {tuple_expr} NOT IN (VALUES {values})"
        ).fetchall()
        return [(int(r[0]), [str(v) for v in r[1:]]) for r in rows]

    # -- materialization ------------------------------------------------------
    def to_rows(self, limit: int | None = None) -> list[Row]:
        """Materialize (up to ``limit``) rows as dicts."""
        clause = "" if limit is None else f" LIMIT {int(limit)}"
        return self._fetch_rows(f"SELECT * FROM {self._view}{clause}")

    def sample_rows(self, cap: int, *, seed: int = 0) -> list[Row]:
        """A deterministic, position-uniform subsample of at most ``cap`` rows.

        Matches the reference backend's hash-of-position rule so the same table yields
        the same sample on either engine: keep a row when a multiplicative hash of its
        one-based position falls in the retained fraction.
        """
        total = self.row_count()
        if total <= cap:
            return self.to_rows()
        offset = seed * _SEED_MULT
        numbered = (
            f"SELECT *, (((row_number() OVER ()) * {_HASH_MULT} + {offset}) "
            f"& {_MASK32}) AS _h FROM {self._view}"
        )
        return self._fetch_rows(
            f"SELECT * EXCLUDE (_h) FROM ({numbered}) WHERE _h % {total} < {cap}"
        )

    # -- internals ------------------------------------------------------------
    def _q(self, sql: str) -> Any:
        return self._con.execute(sql)

    def _fetch_rows(self, sql: str) -> list[Row]:
        """Run ``sql`` and materialize its result as ``list[dict]``."""
        table = self._q(sql).arrow().read_all()
        return [dict(record) for record in table.to_pylist()]

    def _located(self, condition: str, value_expr: str) -> list[Cell]:
        numbered = f"SELECT row_number() OVER () - 1 AS _row, * FROM {self._view}"
        rows = self._q(
            f"SELECT _row, {value_expr} AS _v FROM ({numbered}) WHERE {condition}"
        ).fetchall()
        return [(int(index), value) for index, value in rows]

    def _rows_where(self, condition: str, _unused: list[Any]) -> list[int]:
        numbered = f"SELECT row_number() OVER () - 1 AS _row, * FROM {self._view}"
        rows = self._q(f"SELECT _row FROM ({numbered}) WHERE {condition}").fetchall()
        return [int(r[0]) for r in rows]


def _parses_as_udf(value: Any, type_name: Any) -> bool:
    """The reference type-parser, callable from SQL (NULL text is never a violation)."""
    if value is None:
        return True
    return parses_as(str(value), str(type_name))


def _infer_type(present: int, conforms: dict[str, int]) -> str:
    """The narrowest type all present values conform to, else ``string``."""
    if present == 0:
        return "string"
    for type_name in _TYPE_ORDER:
        if conforms[type_name] == present:
            return type_name
    return "string"


def _qi(name: str) -> str:
    """Quote an identifier, neutralizing embedded quotes (the injection guard)."""
    return '"' + name.replace('"', '""') + '"'


def _escape_literal(value: str) -> str:
    """Escape a single-quoted string literal."""
    return value.replace("'", "''")


def _text(column: str) -> str:
    return f"CAST({_qi(column)} AS VARCHAR)"


def _num(column: str) -> str:
    return f"TRY_CAST(CAST({_qi(column)} AS VARCHAR) AS DOUBLE)"


def _missing(column: str) -> str:
    return f"({_qi(column)} IS NULL OR CAST({_qi(column)} AS VARCHAR) = '')"


def _present(column: str) -> str:
    return f"NOT {_missing(column)}"
