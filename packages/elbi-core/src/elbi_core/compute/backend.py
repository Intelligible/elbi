"""The engine-agnostic compute seam: one fixed operation set, many engines.

A :class:`ComputeBackend` computes a small, finite catalog of operations over a concrete
data representation: structural queries (columns, row count), a one-pass per-column
**profile**, the located **validity primitives** a data contract is checked in terms of,
and **materialization** (a bounded sample of rows, or an Arrow export). The trusted
first-party consumers -- the profiler, the contract checker, and the verification oracle
-- are written once against this protocol; each engine implements the operations for its
representation and is selected by :func:`backend_for` on the *type* of the data. So the
same profile or contract runs on ``list[dict]`` rows today and pushes down to DuckDB or
Polars over a larger-than-memory file tomorrow, without changing a consumer.

The operations *are* the catalog (no separate expression IR): each is a protocol
method, so a backend's supported ops are exactly the methods it implements. Engine
backends are optional: :func:`backend_for` detects an engine by inspecting
``sys.modules`` and never imports one to test for it, so importing this package pulls in
no dataframe engine.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: A single table row.
Row = dict[str, Any]
#: A located cell result: its zero-based row index and the cell value in string form (a
#: display/audit representation, normalized so every backend reports it identically).
Cell = tuple[int, Any]


@dataclass(frozen=True)
class ColumnStats:
    """A one-column profile summary, engine-neutral.

    The result of one backend's :meth:`ComputeBackend.profile` pass, mapped by the
    profiler into its public ``ColumnProfile``. ``inferred_type`` is the narrowest of
    the declared field types (string/integer/number/boolean/date/datetime) the column's
    present values all conform to.
    """

    name: str
    count: int
    present: int
    distinct: int
    inferred_type: str
    minimum: float | None
    maximum: float | None
    top_values: tuple[tuple[Any, int], ...]
    fraction_unique_once: float


@runtime_checkable
class ComputeBackend(Protocol):
    """The fixed catalog of operations a compute engine implements.

    Consumers compose these calls and never touch the data directly, so one profiler,
    contract checker, and oracle run across every backend unchanged.
    """

    # -- structure ------------------------------------------------------------
    def row_count(self) -> int:
        """The number of rows."""
        ...

    def columns(self) -> list[str]:
        """Column names, in order."""
        ...

    def has_column(self, column: str) -> bool:
        """Whether ``column`` is present in the data."""
        ...

    # -- profiling ------------------------------------------------------------
    def profile(self, columns: Sequence[str]) -> dict[str, ColumnStats]:
        """Per-column statistics for ``columns``, computed in a single pass."""
        ...

    # -- located validity primitives (the contract checker's vocabulary) ------
    def present_count(self, column: str) -> int:
        """The number of rows whose ``column`` value is not missing."""
        ...

    def missing_indices(self, column: str) -> list[int]:
        """Row indices where ``column`` is missing (absent or blank)."""
        ...

    def type_violations(self, column: str, type_name: str) -> list[Cell]:
        """Present cells of ``column`` that do not parse as ``type_name``."""
        ...

    def range_violations(
        self, column: str, minimum: float | None, maximum: float | None
    ) -> list[Cell]:
        """Present cells outside the inclusive numeric range (or non-numeric)."""
        ...

    def pattern_violations(self, column: str, pattern: str) -> list[Cell]:
        """Present cells whose string form does not match ``pattern``."""
        ...

    def length_violations(
        self, column: str, min_length: int | None, max_length: int | None
    ) -> list[Cell]:
        """Present cells whose string length is outside the inclusive bounds."""
        ...

    def enum_violations(self, column: str, allowed: Sequence[Any]) -> list[Cell]:
        """Present cells whose value is not in the allowed set."""
        ...

    def duplicate_indices(self, columns: Sequence[str]) -> list[Cell]:
        """Rows whose combined ``columns`` value (all present) repeats."""
        ...

    def key_missing_indices(self, columns: Sequence[str]) -> list[int]:
        """Rows where any of the key ``columns`` is missing."""
        ...

    def reference_violations(
        self, columns: Sequence[str], allowed_keys: set[tuple[str, ...]]
    ) -> list[Cell]:
        """Rows whose combined key (all present) is not in ``allowed_keys``."""
        ...

    # -- materialization / interchange ----------------------------------------
    def to_rows(self, limit: int | None = None) -> list[Row]:
        """Materialize (up to ``limit``) rows as ``list[dict]``.

        The crossing back into Python objects; a genuine copy, so callers cap it. The
        engine backends keep everything columnar until this call.
        """
        ...

    def sample_rows(self, cap: int, *, seed: int = 0) -> list[Row]:
        """A deterministic sample of at most ``cap`` rows.

        Returns every row when the table has no more than ``cap``; otherwise a
        reproducible subsample, so a statistical gate over a large table is a pure
        function of its inputs. The oracle consumes this: it cannot push a bootstrap or
        permutation test into an engine, so the seam reduces the data and hands it a
        bounded, representative frame.
        """
        ...


#: A backend detector and its factory. Detectors inspect ``sys.modules`` and duck-type
#: the data; they never import an engine to test for it.
_Detector = Callable[[Any], bool]
_Factory = Callable[[Any], ComputeBackend]
_REGISTERED: list[tuple[str, _Detector, _Factory]] = []


def register_backend(name: str, detector: _Detector, factory: _Factory) -> None:
    """Register a third-party backend: a name, a detector, and a factory.

    The extension seam. Built-in backends (``list[dict]``, DuckDB, Polars) are handled
    by :func:`backend_for` directly; this lets a deployment add another engine without
    touching the core.
    """
    _REGISTERED.append((name, detector, factory))


def backend_for(data: Any) -> ComputeBackend:
    """Select the compute backend for a data object, by inspecting its type.

    ``list``/``tuple`` of row dicts uses the dependency-free reference backend. A DuckDB
    relation, a Polars frame, a PyArrow table, or a filesystem path to a data file uses
    the corresponding engine backend when that engine is importable. Detection reads
    ``sys.modules`` and never imports an engine that the caller has not, so importing
    this package stays dependency-light.

    Raises:
        TypeError: if no backend handles the data's type.
    """
    if isinstance(data, list | tuple):
        from .rows import RowsFrame

        return RowsFrame(data)

    duckdb = sys.modules.get("duckdb")
    if duckdb is not None and isinstance(data, duckdb.DuckDBPyRelation):
        from .duckdb_backend import DuckDBFrame

        return DuckDBFrame(data)

    polars = sys.modules.get("polars")
    if polars is not None and isinstance(data, polars.DataFrame | polars.LazyFrame):
        from .polars_backend import PolarsFrame

        return PolarsFrame(data)

    pyarrow = sys.modules.get("pyarrow")
    if pyarrow is not None and isinstance(data, pyarrow.Table):
        return _arrow_backend(data)

    if isinstance(data, Path):
        return _path_backend(data)

    for _name, detector, factory in _REGISTERED:
        if detector(data):
            return factory(data)

    raise TypeError(
        f"no compute backend for {type(data).__name__!r}; expected a list of row "
        "dicts, a DuckDB relation, a Polars frame, a PyArrow table, a data-file path, "
        "or a registered backend type"
    )


def _arrow_backend(table: Any) -> ComputeBackend:
    """Back an Arrow table with the best available engine (DuckDB, else rows)."""
    if sys.modules.get("duckdb") is not None:
        from .duckdb_backend import DuckDBFrame

        return DuckDBFrame.from_arrow(table)
    from .rows import RowsFrame

    return RowsFrame(table.to_pylist())


def _path_backend(path: Path) -> ComputeBackend:
    """Back a data-file path with an engine that scans it without loading it fully."""
    from .duckdb_backend import DuckDBFrame

    try:
        return DuckDBFrame.from_path(path)
    except ImportError as exc:
        raise TypeError(
            f"reading {path.name} as a compute source needs the 'duckdb' engine: "
            'pip install "elbi[duckdb]"'
        ) from exc
